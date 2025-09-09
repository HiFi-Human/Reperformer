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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from rich.console import Console
CONSOLE = Console(width=120)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, frame_idx, view_chosen=[30,], no_image=False):
    render_path = os.path.join(model_path, name, "render/")
    gts_path = os.path.join(model_path, name, "gt/")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)   
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx in view_chosen:
            rendering = render(view, gaussians, pipeline, background)["render"]
            print(os.path.join(render_path, f"{frame_idx}_{idx}.png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{frame_idx}_{idx}.png"))
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{frame_idx}_{idx}.png"))
            print(os.path.join(gts_path, f"{frame_idx}_{idx}.png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, frame_st = 1, frame_ed = 2, parallel_load = False, args = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians = GaussianModel(dataset.sh_degree)
        for frame_idx in range(frame_st, frame_ed):
            CONSOLE.log(frame_idx)
            scene = Scene(dataset, gaussians, shuffle=False, dynamic_training = False, load_frame_id = frame_idx, parallel_load = parallel_load)
            scene.gaussians.load_ply(os.path.join(scene.model_path,
                                                            "save_ply",
                                                            "point_cloud_%d.ply") % frame_idx, scene.cameras_extent)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            render_set(dataset.model_path, "render_results", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, frame_idx, view_chosen=args.views)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frame_st", '-st', type=int, default=70)
    parser.add_argument("--frame_ed", '-e', type=int, default=76)
    parser.add_argument("--parallel_load", action="store_true", default=False)
    parser.add_argument("--views", nargs="+", type=int, default=[31,41,51] )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.frame_st, args.frame_ed, args.parallel_load, args)
    