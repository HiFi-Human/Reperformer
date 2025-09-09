#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from torch.utils.data import DataLoader
import random
import math
from random import randint
from utils.loss_utils import l1_loss
from utils.loss_utils import ssim,fast_ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from reperformer_module.reperformer_model import Reperformer_Model,Reperformer_Dataset
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,OptimizationParams_Unet, LossParamsS2, LossParamsS1
import shutil
from utils.warp_utils import Warpper
from utils.graph_utils import node_graph
from rich.console import Console
import time

CONSOLE = Console(width=120)
def forward_warpping(rep,warped_pos):
    pos_map = rep.pos_map.clone()
    pos_map[rep.non_empty_idx, rep.non_empty_idy] =  warped_pos.to(rep.device)
    pos_map = pos_map.unsqueeze(0).permute(0, 3, 1, 2) 
    pos = rep.unprojection(pos_map).detach()
    gs_motion_map = rep.gs_motion_unet(pos_map)
    gs_geo_map = rep.gs_geo_unet(pos_map)
    gs_color_map = rep.gs_color_unet(pos_map)
    # unprojection
    motion_attributes = rep.unprojection(gs_motion_map)
    geo_attributes = rep.unprojection(gs_geo_map)
    color_attributes = rep.unprojection(gs_color_map)
    rep.gaussian._xyz = pos # + motion_attributes[:, :3]
    rep.gaussian._rotation = motion_attributes[:, :] 
    rep.gaussian._scaling = geo_attributes[:, :3]
    rep.gaussian._opacity = geo_attributes[:, 3].unsqueeze(1)
    features = color_attributes.reshape(pos.shape[0], rep.color_channel, 3)
    rep.gaussian._features_dc = features[:, 0:1, :]
    rep.gaussian._features_rest = features[:, 1:, :]

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/",args.source_path.split('/')[-1] + '_' + unique_str[0:2])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    return tb_writer
def move_cameras_to_cpu(scene):
    for scale, cams in scene.train_cameras.items():
        scene.train_cameras[scale] = [c.to("cpu") for c in cams]
    for scale, cams in scene.test_cameras.items():
        scene.test_cameras[scale] = [c.to("cpu") for c in cams]
    return scene
def get_camera(scene, split="train", scale=1.0, idx=0, device="cuda:0"):
    if split == "train":
        cam = scene.train_cameras[scale][idx]
    else:
        cam = scene.test_cameras[scale][idx]
    return cam.to(device)
def canonical_preprocessing(dataset, opt, pipe, lossp, testing_iterations, debug_from, frame_idx = 1, args = None):
    ply_path = args.ply_path
    parallel_load = args.parallel_load
    map_resolution = args.map_resolution
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    skin_gaussian = GaussianModel(dataset.sh_degree, resolution=map_resolution)
    scene = Scene(dataset, skin_gaussian, dynamic_training = True, load_frame_id = frame_idx, ply_path = ply_path, parallel_load=parallel_load, stage = 2, warpDQB = warpDQB)

    skin_gaussian.training_setup_t2(opt)
    print("number of gaussians:", len(skin_gaussian.get_xyz))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        skin_gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            skin_gaussian.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_canonical = render(viewpoint_cam, skin_gaussian, pipe, background)
        image_canonical , viewspace_point_tensor, visibility_filter, radii = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_canonical, gt_image)
        ssim_value = fast_ssim(image_canonical, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        loss_info = {}
        loss_info["rgb_loss"] = loss.item()
        loss.backward()
        iter_end.record()

        loss_info['gs_num'] = skin_gaussian.get_xyz.shape[0]

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()
            # first frame Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                skin_gaussian.max_radii2D[visibility_filter] = torch.max(skin_gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                skin_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    skin_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    skin_gaussian.reset_opacity()
            if skin_gaussian.get_xyz.shape[0] > map_resolution * map_resolution:
                ratio = max(0, math.floor((1 - (map_resolution*map_resolution - 1) / skin_gaussian.get_xyz.shape[0]) * 100))
            else:
                ratio = 0
            if iteration % opt.prune_least_opacity_iter == 0 and  ratio > 0:
                skin_gaussian.prune_least_opacity(ratio+1)
                CONSOLE.log(f"Pruning {skin_gaussian.get_xyz.shape[0]} points to {map_resolution * map_resolution} points ({ratio:.2f}%)")
            if iteration < opt.iterations:
                skin_gaussian.optimizer.step()
                skin_gaussian.optimizer.zero_grad(set_to_none = True)
                
    skin_gaussian.save_ply(os.path.join(dataset.model_path, "ckt", "point_cloud_%d.ply"% (frame_idx )))
    skin_gaussian.save_morton_map(os.path.join(dataset.model_path, "morton_map"))
def training_joint(dataset, opt, pipe, lossp, testing_iterations, debug_from, is_start_frame, frame_idx = 1, args = None):
    CONSOLE.log("Training joint:", frame_idx)
    parallel_load = args.parallel_load
    subseq_iters = args.subseq_iters
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    ply_path = args.ply_path

    joint_gaussian = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, joint_gaussian, dynamic_training = True, load_frame_id = frame_idx, ply_path = ply_path, parallel_load=parallel_load, stage = 1, warpDQB=warpDQB)

    joint_gaussian.training_setup_t1(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if is_start_frame==False:
        opt.iterations = subseq_iters if subseq_iters else (opt.iterations // 2)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    loss = 0
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        joint_gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            joint_gaussian.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_canonical = render(viewpoint_cam, joint_gaussian, pipe, background)
        image_canonical , viewspace_point_tensor, visibility_filter, radii = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image_canonical, gt_image)
        if is_start_frame:
            ssim_value = ssim(image_canonical, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            loss = Ll1

        loss_info = {}
        loss_info["rgb_loss"] = loss.item()
        reg_loss, reg_loss_info = joint_graph.compute_loss(joint_gaussian, lossp, is_start_frame)
        loss_info.update(reg_loss_info)

        loss = loss + reg_loss
        loss.backward()

        iter_end.record()
        if is_start_frame:
            loss_info['gs_num'] = joint_gaussian.get_xyz.shape[0]
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            if is_start_frame and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                joint_gaussian.max_radii2D[visibility_filter] = torch.max(joint_gaussian.max_radii2D[visibility_filter], radii[visibility_filter])
                joint_gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    joint_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.1, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    joint_gaussian.reset_opacity()
                
            if iteration == opt.densify_until_iter and is_start_frame:
                points_num = joint_gaussian._xyz.shape[0]
                prune_scale = int(points_num // opt.joint_gs_num)
                prune_scale = max(prune_scale, 1)
                prune_mask = torch.ones(points_num, dtype=torch.bool, device=joint_gaussian._xyz.device)
                prune_mask[::prune_scale] = False
                joint_gaussian.prune_points(prune_mask)

            if iteration < opt.iterations:
                if  is_start_frame==False:
                    joint_gaussian.lock_gradient( lock_opacity = True, lock_scaling= True, lock_features= True)       
                joint_gaussian.optimizer.step()
                joint_gaussian.optimizer.zero_grad(set_to_none = True)

    joint_gaussian.save_ply(os.path.join(dataset.model_path, "ckt", "point_cloud_%d.ply"% (frame_idx )))
    joint_gaussian.save_relative_motion(dataset.model_path,args.frame_st,frame_idx)
    if is_start_frame:
        joint_graph.graph_init(joint_gaussian.get_xyz, k = 8)
    
    joint_graph.regular_term_setup(joint_gaussian, velocity_option=True, warpDQB=warpDQB)
    

def training_unet(dataset, opt, pipe, lossp, testing_iterations, debug_from, frame_range, args = None):
    reperformer_unet = Reperformer_Model(args, dataset)
    reperformer_unet.training_setup_unet(opt)
    ply_path = args.ply_path
    parallel_load = args.parallel_load
    first_iter = 0
    first_iter = max(reperformer_unet.iter_step,first_iter)
    scene = Scene(
        dataset,
        None,
        dynamic_training=False,
        load_frame_id=args.frame_st,
        ply_path=ply_path,
        parallel_load=parallel_load,
        stage=2,
        warpDQB=warpDQB,
        no_image=True,
        device="cuda"
    )
    

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    ema_loss_for_log = 0.0

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    # training setup
    warpDQB.record_morton_map(reperformer_unet.verts,reperformer_unet.rotations)
    warpDQB.loadMotion(args.motion_folder, args.frame_st, start_frame=args.frame_st)
    warpDQB.skin2JointInterpolation(warpDQB.raw_xyz)
    unet_graph.graph_init(reperformer_unet.gaussian_canonical.get_xyz)
    unet_graph.regular_term_setup(reperformer_unet.gaussian_canonical, velocity_option=False)
    progress_bar = tqdm(range(first_iter, args.unet_iterations), desc="Unet Training progress")

    reperformer_dataset = Reperformer_Dataset(scene,args.frame_st,args.frame_ed,args.frame_step,warpDQB.raw_xyz, warpDQB.raw_rot,warpDQB.graph_weights_,warpDQB.indices_,args=args)
    
    train_dataloader = DataLoader(
        reperformer_dataset, batch_size=1, 
        shuffle=True, 
        num_workers = opt.worker_num,
        collate_fn= list
    )
    it = iter(train_dataloader)
    
    for iteration in range(first_iter, args.unet_iterations + 1):

        iter_start.record()
        ret_dict = next(it)[0]
        
        reperformer_unet.iter_step = iteration
        frame_idx = ret_dict["cur_frame_id"]
        camera_idx = ret_dict["cur_camera_id"]
        warped_pos = ret_dict["warped_pos"]
        warped_rot = ret_dict["warped_rot"]
        forward_warpping(reperformer_unet, warped_pos)
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if(iteration < reperformer_unet.warm_up_end):
            Pseudo_gs = reperformer_unet.gaussian_canonical
            losses = reperformer_unet.criterion(reperformer_unet.gaussian._features_dc, Pseudo_gs._features_dc) + reperformer_unet.criterion(reperformer_unet.gaussian._features_rest, Pseudo_gs._features_rest)+ reperformer_unet.criterion(reperformer_unet.gaussian._scaling, Pseudo_gs._scaling)+ reperformer_unet.criterion(reperformer_unet.gaussian._opacity, Pseudo_gs._opacity)+ reperformer_unet.criterion(reperformer_unet.gaussian._rotation, warped_rot.to(reperformer_unet.device))
            Ll1 = 0
            loss = losses
            loss_info = {}
            loss_info["pre_loss"] = loss.item()
            loss.backward()
            iter_end.record()
        else:
            viewpoint_cam = get_camera(scene, split="train", scale=1.0, idx=camera_idx, device=reperformer_unet.device)
            gt_image = ret_dict["ret_img"].cuda()
            render_canonical = render(viewpoint_cam, reperformer_unet.gaussian, pipe, background)
            image_canonical , _, _, _ = render_canonical["render"], render_canonical["viewspace_points"], render_canonical["visibility_filter"], render_canonical["radii"]
            Ll1 = l1_loss(image_canonical, gt_image)
            ssim_value = fast_ssim(image_canonical, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            loss_info = {}
            loss_info["rgb_loss"] = loss.item()
            reg_loss, reg_loss_info = unet_graph.compute_loss(reperformer_unet.gaussian, lossp, False)
            loss_info.update(reg_loss_info)
            loss = loss + reg_loss        
            loss.backward()
            iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", **{k: f"{v:.7f}" for k, v in loss_info.items()}})
                progress_bar.update(100)
            if iteration == args.unet_iterations:
                progress_bar.close()
            if iteration < args.unet_iterations:
                reperformer_unet.optimizer.step()
                reperformer_unet.optimizer.zero_grad(set_to_none = True)
                reperformer_unet.update_learning_rate()
        if (reperformer_unet.iter_step % reperformer_unet.save_freq == 0 and reperformer_unet.iter_step > 10):
            reperformer_unet.save_checkpoint()
            os.makedirs(os.path.join(reperformer_unet.dataset.model_path, "evaluate"), exist_ok=True)
            reperformer_unet.gaussian.save_ply(os.path.join(reperformer_unet.dataset.model_path, "evaluate","point_cloud_%d_%d.ply"% (frame_idx,iteration)))
            CONSOLE.log(f"Saved checkpoint at iteration {reperformer_unet.iter_step}")

def evaluate_unet(dataset, opt, frame_range, args = None):
    reperformer_unet = Reperformer_Model(args, dataset,evaluate=True)
    reperformer_unet.training_setup_unet(opt)
    first_iter = 0
    first_iter += 1
    # training setup
    unet_graph.graph_init(reperformer_unet.gaussian_canonical.get_xyz, k = 8)
    unet_graph.regular_term_setup(reperformer_unet.gaussian_canonical)
    
    warpDQB.record_morton_map(reperformer_unet.verts,reperformer_unet.rotations)
    warpDQB.loadMotion(args.motion_folder, args.frame_st, start_frame=args.frame_st)
    warpDQB.skin2JointInterpolation(warpDQB.raw_xyz)
    unet_graph.graph_init(reperformer_unet.gaussian_canonical.get_xyz)
    unet_graph.regular_term_setup(reperformer_unet.gaussian_canonical, velocity_option=False)
    for frame_idx in tqdm(frame_range):
        relative_motion = warpDQB.loadRelativeMotion(args.motion_folder, frame_idx, start_frame=args.frame_st)
        warpDQB.warping_reperformer(reperformer_unet.gaussian, warpDQB.raw_xyz, warpDQB.raw_rot,rel_trans=relative_motion)
        # record the original pos and rot
        forward_warpping(reperformer_unet, reperformer_unet.gaussian.get_xyz)
        os.makedirs(os.path.join(reperformer_unet.dataset.model_path, "save_ply"), exist_ok=True)
        reperformer_unet.gaussian.save_ply(os.path.join(reperformer_unet.dataset.model_path, "save_ply","point_cloud_%d.ply"% (frame_idx)))
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    unet_op = OptimizationParams_Unet()
    pp = PipelineParams(parser)
    lossp1 = LossParamsS1(parser)
    lossp2 = LossParamsS2()

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame_st", type=int, default=110)
    parser.add_argument("--frame_ed", type=int, default=200)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument('--motion_folder', type=str, default= None)
    parser.add_argument('--ply_path', type=str, default= None)
    parser.add_argument('--parallel_load', action='store_true', default=False)
    parser.add_argument("--subseq_iters", type=int, default=None)
    parser.add_argument('--map_resolution',type=int, default=512, help="The resolution of the Position Map")
    parser.add_argument('--unet_iterations',type=int, default=1000000, help='The number of iterations to train the UNet')
    parser.add_argument('--training_mode', type=int, default= 0, help="0: all,1: motion only, 2: preprocessing, 3: unet only, 4: evaluate only")
    args = parser.parse_args(sys.argv[1:])
    model_path = str(args.model_path)

    print("Optimizing " + args.model_path)
    
    os.makedirs(args.model_path, exist_ok = True)

    shutil.copy('arguments/__init__.py', args.model_path)
    shutil.copy('utils/graph_utils.py', args.model_path)
    shutil.copy('train.py', args.model_path)
    shutil.copy('scene/gaussian_model.py', args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.motion_folder is None:
        args.motion_folder = os.path.join(args.model_path, 'track')
    unet_op.end_iter = args.unet_iterations
    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    warpDQB = Warpper(args.frame_st, args.frame_ed, args.frame_step)

    joint_graph = node_graph()
    unet_graph = node_graph()
    frame_range = range(args.frame_st, args.frame_ed, args.frame_step)
    is_start_frame = True
    for frame_idx in frame_range:
        if args.training_mode == 0 or args.training_mode == 1:
            if args.motion_folder is None:
                args.model_path = os.path.join(model_path, 'track')
            else:
                args.model_path = args.motion_folder
            training_joint(lp.extract(args), op.extract(args), pp.extract(args), lossp1.extract(args), args.test_iterations, args.debug_from, is_start_frame, frame_idx, args=args)
        is_start_frame = False
    if args.training_mode == 0 or args.training_mode == 2:    
        args.model_path = model_path 
        canonical_preprocessing(lp.extract(args), op.extract(args), pp.extract(args), lossp2, args.test_iterations, args.debug_from, frame_range[0], args=args)
    if args.training_mode == 0 or args.training_mode == 3:   
        args.model_path = model_path 
        training_unet(lp.extract(args), unet_op, pp.extract(args), lossp2, args.test_iterations, args.debug_from, frame_range, args=args)
    if args.training_mode == 0 or args.training_mode == 4:
        args.model_path = model_path 
        evaluate_unet(lp.extract(args), unet_op, frame_range, args=args)
    print("\nTraining complete.")