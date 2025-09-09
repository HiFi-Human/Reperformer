import os
import torch
import sys
import cv2
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from utils.system_utils import mkdir_p
import torchvision.transforms as transforms
import torchvision
from torchvision.io import read_file, decode_image
from PIL import Image
from reperformer_module.sa_unet import unet
from utils.calc_utils import *
from utils.general_utils import read_ply_and_export_matrix
from torch import nn
from filelock import FileLock
import shutil

def save_normalized_image(tensor, save_path):
    tensor = tensor.squeeze(0)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(save_path)
def save_relative_motion(motionFolder, start_frame, frame_idx):
    src_path = os.path.join(motionFolder, 'ckt', f'point_cloud_{start_frame}.ply')
    dst_path = os.path.join(motionFolder, 'ckt', f'point_cloud_{frame_idx}.ply')
    src_gs = torch.from_numpy(read_ply_and_export_matrix(src_path)).to(torch.float32)
    dst_gs = torch.from_numpy(read_ply_and_export_matrix(dst_path)).to(torch.float32)
    gaussian_number = min(src_gs.shape[0], dst_gs.shape[0])
    src_gs = src_gs[:gaussian_number]
    dst_gs = dst_gs[:gaussian_number]

    rel_rotations = quaternion_multiply(norm_quaternion(dst_gs[:, -4:]), quaternion_inverse(src_gs[:, -4:]))
    rel_rotations = norm_quaternion(rel_rotations)
    rel_rots = build_rotation(rel_rotations)
    src_xyz = src_gs[:, :3].reshape(-1, 3, 1)
    dst_xyz = dst_gs[:, :3].reshape(-1, 3, 1)
    rel_xyz = dst_xyz - torch.einsum("ijk,ikn->ijn", rel_rots, src_xyz)

    rel_trans = torch.cat([rel_rots, rel_xyz], dim=2)
    rel_trans = rel_trans.reshape(-1, 3, 4)
    rel_trans = nn.Parameter(rel_trans, requires_grad=True)
    save_dir = os.path.join(motionFolder, 'relative_motion')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'rel_trans_{frame_idx}.bin')
    lock_path = save_path + ".lock"
    lock = FileLock(lock_path)
    with lock:
        torch.save(rel_trans.float().cpu(), save_path)
class Reperformer_Dataset(Dataset):
    def __init__(self,scene,start_frame,end_frame,frame_step,xyz,rot,graph_weight,indices,evaluate=False,args=None):
        # pass
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_step = frame_step 
        self.chosen_frame_id_list = [
            i for i in range(self.start_frame, self.end_frame, self.frame_step)
        ]
        self.train_cameras = scene.getTrainCameras()
        self.never_stop_size = int(1e7)
        self.source_path = scene.source_path
        self.resolution = scene.resolution_scales[0]
        self.evaluate = evaluate
        self.xyz = xyz.cpu().share_memory_()
        self.rot = rot.cpu().share_memory_()
        self.graph_weights_ = graph_weight.cpu().share_memory_()
        self.indices_ = indices.cpu().share_memory_()
        self.args = args
    def process_image(self, image):
        orig_w, orig_h = image.shape[:2]
        if self.resolution in [1, 2, 4, 8]:
            resolution = round(orig_h/( self.resolution)), round(orig_w /( self.resolution))
        else:  # should be a type that converts to float
            if self.resolution == -1:
                if orig_h > 1600:
                    global_down = orig_h / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / self.resolution
            scale = float(global_down)
            resolution = (int(orig_h / scale), int(orig_w / scale))
        resized_image = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
        resized_image_rgb = torch.from_numpy(resized_image).float() / 255.0
        original_image = resized_image_rgb.clamp(0.0, 1.0)
        return original_image
    def warpgaussian(self,rel_trans):
        # def warping_reperformer(self, gaussians, raw_xyz, raw_rot,rel_trans=None):
        RT = rel_trans
        R = RT[..., :3] 
        RT_gathered = RT[self.indices_] 
        rotations = RT_gathered[..., :3] 
        translations = RT_gathered[..., 3] 
        pos = self.xyz.detach()
        pos_expanded = pos.unsqueeze(1).expand(-1, self.indices_.size(1), -1)

        pos_transformed = torch.einsum('gkij,gkj->gki', rotations, pos_expanded) + translations
        pos_out = torch.sum(pos_transformed * self.graph_weights_, dim=1) 
        R_out = torch.einsum('gkij,gk->gij', rotations, self.graph_weights_.squeeze(-1))
        rots = norm_quaternion(self.rot.detach())
        R = batch_qvec2rotmat_torch(rots)
        result = torch.matmul(R_out, R)
        new_rots = batch_rotmat2qvec_torch(result)
        return pos_out,new_rots
    
    def generate_mask_and_image_info(self, camera_id, frame_id):
        # print("The image path is {}".format(os.path.join(self.source_path,"{}".format(frame_id),self.train_cameras[camera_id].image_name)))
        image_path = os.path.join(self.source_path,"{}".format(frame_id),self.train_cameras[camera_id].image_name)
        # print(image_path)
        original_image  = decode_image(read_file(image_path), mode=torchvision.io.ImageReadMode.RGB).float() / 255.0
        # color_image = np.array(Image.open(image_path))
        # original_image = self.process_image(color_image).permute([2,0,1])
        return {
            'color': original_image,
        }
    def loadRelativeMotion(self, motionFolder, frame_idx, sequential=False, start_frame=0):
        start_frame = start_frame
        relative_motion_path = os.path.join(motionFolder, 'relative_motion', f'rel_trans_{frame_idx}.bin')
        if (not os.path.exists(relative_motion_path)):
            save_relative_motion(motionFolder, start_frame, frame_idx)
        rel_trans = torch.load(relative_motion_path, map_location='cpu').float()
        return rel_trans
    def get_val_img_dict(self, current_frame_id, camera_id):
        rel_trans = self.loadRelativeMotion(self.args.motion_folder, current_frame_id, start_frame=self.args.frame_st)
        if self.evaluate:
            ret_img = None
        else:
            img_info = self.generate_mask_and_image_info(
                camera_id = camera_id,
                frame_id = current_frame_id
            )

            ret_img = img_info['color']
        pos,rot = self.warpgaussian(rel_trans)
        pos = pos.detach()
        rot = rot.detach()
        with torch.no_grad():
            ret_dict = {
                'cur_frame_id': current_frame_id,
                'cur_camera_id': camera_id,
                'ret_img': ret_img,
                "warped_pos":pos,
                "warped_rot":rot
            }
                
        return ret_dict

    def __getitem__(self, idx):
        
        idx = idx % (
            len(self.chosen_frame_id_list) * len(self.train_cameras)
        )
        
        current_frame_id = self.chosen_frame_id_list[(idx // len(self.train_cameras))]
        # current_frame_id = self.chosen_frame_id_list[-1]
        camera_id = idx % len(self.train_cameras)
        # camera_id = 33
        ret_dict = self.get_val_img_dict(
            current_frame_id, camera_id
        )
                        
        return ret_dict
    
    
    def __len__(self):
        return max(len(self.chosen_frame_id_list), self.never_stop_size)
class Reperformer_Model:
    def __init__(self, args, dataset, evaluate=False):
        self.device = "cuda"
        self.dataset = dataset
        self.summary_writer = SummaryWriter(
            dataset.model_path + '/logs'
        )

        ###################################################################################################################
        #                                                     test                                                        #
        ###################################################################################################################
        canonical_gaussian_path = os.path.join(dataset.model_path, "morton_map")

        # pos_map: [H, W, 3] float tensor
        self.pos_map = torch.from_numpy(
            np.load(os.path.join(canonical_gaussian_path, 'pos.npy'))
        ).float().to(self.device)

        # rotation_map: [H, W, 4] float tensor
        self.rotation_map = torch.from_numpy(
            np.load(os.path.join(canonical_gaussian_path, 'rotation.npy'))
        ).float().to(self.device)

        self.barycentric_tex_size = args.map_resolution

        # non_empty_idx, non_empty_idy: index arrays -> tensor on GPU
        non_empty_mask = self.pos_map[..., 0] != 0  # bool mask
        self.non_empty_idx, self.non_empty_idy = non_empty_mask.nonzero(as_tuple=True)

        self.gaussian_canonical = GaussianModel(dataset.sh_degree)
        self.gaussian_canonical.load_ply(os.path.join(canonical_gaussian_path, 'point_cloud.ply'), 0)

        self.non_empty_idx = torch.from_numpy(
            np.load(os.path.join(canonical_gaussian_path, 'u.npy'))
        ).long().to(self.device)

        self.non_empty_idy = torch.from_numpy(
            np.load(os.path.join(canonical_gaussian_path, 'v.npy'))
        ).long().to(self.device)

        # verts, rotations -> float tensors on GPU
        self.verts = self.pos_map[self.non_empty_idx, self.non_empty_idy, :]
        self.rotations = self.rotation_map[self.non_empty_idx, self.non_empty_idy, :]

        
        ###################################################################################################################
        #                                        camera related                                                           #
        ################################################################################################################### 
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        self.sh_degree = dataset.sh_degree
        self.gs_motion_unet = unet(3, 4, ).to(self.device)
        self.gs_geo_unet = unet(3, 4, ).to(self.device)
        self.color_channel = (self.sh_degree + 1) ** 2
        self.gs_color_unet = unet(3, self.color_channel * 3, ).to(self.device)
        ###################################################################################################################
        #                                        model related                                                            #
        ################################################################################################################### 
        self.iter_step = 0
        self.from_scratch = True
        self.warm_up_sign = True
        # gaussian
        self.gaussian = GaussianModel(self.sh_degree)
        self.gaussian.active_sh_degree = self.sh_degree
        self.criterion = torch.nn.L1Loss()
        
        self.get_latest_checkpoints()
        self.save_id = 0

    def training_setup_unet(self,training_args):
        # pass
        self.warm_up_end = training_args.warm_up_end
        self.learning_rate_alpha = training_args.learning_rate_alpha
        self.start_constant_lr = training_args.start_constant_lr
        self.end_iter = training_args.end_iter
        self.motion_learning_rate = training_args.motion_learning_rate 
        self.geo_learning_rate = training_args.geo_learning_rate
        self.color_learning_rate = training_args.color_learning_rate
        self.save_freq = training_args.save_interval
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.gs_motion_unet.parameters(), 'name': 'motion', 'lr': self.motion_learning_rate},
                {'params': self.gs_geo_unet.parameters(),  'name': 'geo',  'lr': self.geo_learning_rate},
                {'params': self.gs_color_unet.parameters(), 'name': 'color', 'lr': self.color_learning_rate},
            ], 
            eps=1e-15
        )
        
        self.gs_motion_unet.train()
        self.gs_geo_unet.train()
        self.gs_color_unet.train()


        self.update_learning_rate()
    # @torch.compile
    # def unprojection(self, feature_map):
    #     vertices = feature_map[0][:,self.non_empty_idx, self.non_empty_idy].permute(1, 0).contiguous()
    #     return vertices
    def unprojection(self, feature_map):
        vertices = feature_map[0][:, self.non_empty_idx, self.non_empty_idy]
        return vertices.permute(1, 0).contiguous()

    def update_learning_rate(self):


        if self.iter_step < self.warm_up_end:
            learning_factor = 1
            learning_factor = 1 - 0.9 * (self.iter_step / self.warm_up_end)
        else:
            alpha = self.learning_rate_alpha
            progress = (
                min(self.iter_step, self.start_constant_lr + 2000) - self.warm_up_end
            ) / (
                min(self.end_iter, self.start_constant_lr + 2000) - self.warm_up_end
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            learning_factor *= 0.05
        
        for g in self.optimizer.param_groups:
            if g['name'] == 'motion':
                g['lr'] = self.motion_learning_rate * learning_factor
            elif g['name'] == 'geo':
                g['lr'] = self.geo_learning_rate * learning_factor
            elif g['name'] == 'color':
                g['lr'] = self.color_learning_rate * learning_factor
            if self.iter_step > self.start_constant_lr:
                g['lr'] = alpha * self.motion_learning_rate
        return

    
    
        
    def save_checkpoint(self):
        print('saving checkpoints:', self.iter_step)
        cur_base_dir = os.path.join(self.dataset.model_path, 'checkpoints', str(self.iter_step))
        state_dict_file_name = os.path.join(cur_base_dir, 'state_dict.pth')
        mkdir_p(os.path.dirname(state_dict_file_name))

        cur_state_dict = {
            'iter_step': self.iter_step
        }
        if self.optimizer is not None:
            print('saving checkpoints optimizer')
            cur_state_dict['optimizer'] = self.optimizer.state_dict()
        if self.gs_motion_unet is not None:
            cur_state_dict['gs_motion_unet'] = self.gs_motion_unet.state_dict()
        if self.gs_geo_unet is not None:
            cur_state_dict['gs_geo_unet'] = self.gs_geo_unet.state_dict()
        if self.gs_color_unet is not None:
            cur_state_dict['gs_color_unet'] = self.gs_color_unet.state_dict()

        torch.save(cur_state_dict, state_dict_file_name)

        ckpt_root = os.path.join(self.dataset.model_path, 'checkpoints')
        all_ckpts = [d for d in os.listdir(ckpt_root) if d.isdigit()]
        all_ckpts = sorted([int(d) for d in all_ckpts], reverse=True)

        for old_iter in all_ckpts[2:] if len(all_ckpts) > 2 else []:
            old_path = os.path.join(ckpt_root, str(old_iter))
            print(f"Removing old checkpoint: {old_path}")
            shutil.rmtree(old_path, ignore_errors=True)


    def get_latest_checkpoints(self):
        print('+++++ run load checkpoints')
        # fin the newest
        if os.path.isdir(os.path.join(self.dataset.model_path, 'checkpoints')):
            model_list_raw = os.listdir(os.path.join(self.dataset.model_path, 'checkpoints'))
            model_id_list = []
            # find the last checkpoint, id is non-filled
            for each_model_name in model_list_raw:
                model_id_list.append(
                    int(each_model_name)
                )
            model_id_list.sort()
            print('avilable model id ', model_id_list)
            if len(model_id_list) >= 1:
                self.latest_model_name = str(model_id_list[-1])
                self.load_checkpoint()
            else:
                print('+++++ no checkpoints, start from scratch')
        else:
            print('+++++ not even checkpint folder, start from scratch')

        print('+++++ end load checkpoints')
        return
    
    def load_checkpoint(self):
        print('+++++ loading checkpoints from:', self.latest_model_name)
        
        fin_checkpoint_file_name = os.path.join(
            self.dataset.model_path, 'checkpoints', self.latest_model_name, 'state_dict.pth'
        )

        # Load the checkpoint file
        cur_state_dict = torch.load(fin_checkpoint_file_name, map_location=self.device)

        # Restore iteration step
        self.iter_step = cur_state_dict['iter_step']
        
        # Load optimizer state dict if available
        # if (self.optimizer is not None) and ('optimizer' in cur_state_dict.keys()):
        #     print('+++++ loading checkpoints optimizer')
        #     self.optimizer.load_state_dict(cur_state_dict['optimizer'])
        
        # Helper function to remove 'module.' prefix if necessary (for DataParallel cases)
        def remove_module_prefix(state_dict):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')  # remove 'module.' prefix
                new_state_dict[new_key] = v
            return new_state_dict

        # Function to safely load state_dict with error handling
        def safe_load_model(model, state_key):
            if model is not None and state_key in cur_state_dict.keys():
                print(f'+++++ loading checkpoints {state_key}')
                try:
                    state_dict = cur_state_dict[state_key]
                    # Handle 'module.' prefix
                    if any(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = remove_module_prefix(state_dict)
                    model.load_state_dict(state_dict, strict=False)  # Load with strict=False to ignore missing/unexpected keys
                except RuntimeError as e:
                    print(f"Error loading {state_key}: {e}")

        # Load individual UNet state dicts
        safe_load_model(self.gs_motion_unet, 'gs_motion_unet')
        safe_load_model(self.gs_geo_unet, 'gs_geo_unet')
        safe_load_model(self.gs_color_unet, 'gs_color_unet')

        print('+++++ end loading checkpoints')
