import numpy as np
import torch
import sys 
from torch import nn
from sklearn.neighbors import NearestNeighbors

from rich.console import Console
from utils.calc_utils import *
from utils.general_utils import read_ply_and_export_matrix
import os
import copy
CONSOLE = Console(width=120)
def save_relative_motion(motionFolder, start_frame, frame_idx):
    src_path = os.path.join(motionFolder, 'ckt', f'point_cloud_{start_frame}.ply')
    dst_path = os.path.join(motionFolder, 'ckt', f'point_cloud_{frame_idx}.ply')
    src_gs = torch.from_numpy(read_ply_and_export_matrix(src_path)).cuda().to(torch.float32)
    dst_gs = torch.from_numpy(read_ply_and_export_matrix(dst_path)).cuda().to(torch.float32)
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
    if(not os.path.exists(os.path.join(motionFolder,'relative_motion'))):
        os.makedirs(os.path.join(motionFolder,'relative_motion'),exist_ok=True)
    save_path = os.path.join(motionFolder,'relative_motion', f'rel_trans_{frame_idx}.bin')
    torch.save(rel_trans.cpu(), save_path)
class Warpper():
    
    def __init__(self, stFrame= 1, edFrame = 2, step = 1):
        self.xyz_velocity = torch.empty(0).cuda()
        self.stFrame_ = stFrame
        self.edFrame_ = edFrame
        self.step_ = step

    def compute_next_frame(self, frame_idx):
        next_frame_idx = frame_idx + self.step_
        return next_frame_idx
    
    def compute_last_frame(self, frame_idx):
        next_frame_idx = frame_idx - self.step_
        return next_frame_idx
    
    def loadMotion(self, motionFolder, frame_idx, sequential = False, start_frame = 0):
        src = os.path.join(motionFolder, 'ckt', f'point_cloud_{start_frame}.ply')
        self.src_gs = torch.from_numpy( read_ply_and_export_matrix(src)).cuda().to(torch.float32)
        self.current_xyz = self.src_gs[:, :3]

        dst = os.path.join(motionFolder, 'ckt', f'point_cloud_{frame_idx}.ply')
        self.dst_gs = torch.from_numpy( read_ply_and_export_matrix(dst)).cuda().to(torch.float32)
        self.next_xyz = self.dst_gs[:, :3]
        self.current_number = self.src_gs.shape[0]
        self.next_number = self.dst_gs.shape[0]

        self.gaussian_number = min(self.src_gs.shape[0], self.dst_gs.shape[0])
        
        self.src_gs = self.src_gs[:self.gaussian_number]
        self.dst_gs = self.dst_gs[:self.gaussian_number]

        self.joint = self.src_gs[:, :3]
        rel_rotations = quaternion_multiply(norm_quaternion(self.dst_gs[:, -4:]), quaternion_inverse(self.src_gs[:, -4:]))
        rel_rotations = norm_quaternion(rel_rotations)
        rel_rots = build_rotation(rel_rotations)

        src_xyz = self.src_gs[:, :3].reshape(-1, 3, 1)
        dst_xyz = self.dst_gs[:, :3].reshape(-1, 3, 1)
        rel_xyz = dst_xyz - torch.einsum("ijk,ikn->ijn", rel_rots, src_xyz)

        rel_trans = torch.cat([rel_rots, rel_xyz], dim=2)
        self.rel_trans = rel_trans.reshape(-1, 3, 4)
        self.rel_trans = nn.Parameter(self.rel_trans, requires_grad=True)

    def loadRelativeMotion(self,motionFolder, frame_idx, sequential=False, start_frame=0):
        relative_motion_path = os.path.join(motionFolder, 'relative_motion', f'rel_trans_{frame_idx}.bin')
        if (not os.path.exists(relative_motion_path)):
            CONSOLE.log(f"Relative motion file for frame {frame_idx} not found, computing and saving it.")
            save_relative_motion(motionFolder, start_frame, frame_idx)
        rel_trans = torch.load(relative_motion_path, map_location='cpu') 
        rel_trans = rel_trans.view(-1, 3, 4)
        return rel_trans

    def skin2JointInterpolation(self, points, k=8, l = 0.02):
        points_np = points.detach().cpu().numpy() 
        joints_np = self.joint.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(joints_np)
        dists, indices = nbrs.kneighbors(points_np) 
        self.dist_ = torch.from_numpy(dists[:, ]).float().to(points.device) 
        self.indices_ = torch.from_numpy(indices[:, ]).long().to(points.device)
        self.graph_weights_ = torch.exp(-1 * self.dist_ ** 2 / l ** 2)
        self.graph_weights_ = self.graph_weights_ / self.graph_weights_.sum(dim=1, keepdim=True)
        self.graph_weights_ = self.graph_weights_.unsqueeze(-1).detach()
        return self.graph_weights_, self.indices_

    def record_gaussian(self, gaussians):
        self.canonical_gaussians = copy.deepcopy(gaussians)
        self.canonical_gaussians._xyz.requires_grad = False
        self.canonical_gaussians._features_dc.requires_grad = False
        self.canonical_gaussians._features_rest.requires_grad = False
        self.canonical_gaussians._scaling.requires_grad = False
        self.canonical_gaussians._rotation.requires_grad = False
        self.canonical_gaussians._opacity.requires_grad = False
        
        self.raw_xyz = gaussians.get_xyz.clone().detach().requires_grad_(False)
        self.raw_rot = gaussians.get_rotation.clone().detach().requires_grad_(False)
    def record_morton_map(self, pos, rot):
        self.raw_xyz = pos.clone().detach().requires_grad_(False).float().cuda()
        self.raw_rot = rot.clone().detach().requires_grad_(False).float().cuda()
    def save_motion(self, frame_idx, path):
        os.makedirs(path, exist_ok = True)
        rel_trans_np = self.rel_trans.detach().cpu().numpy()
        np.savez_compressed(os.path.join(path, f'joint_RT_{frame_idx}.npz'), rel_trans_np)

    def warping_reperformer(self, gaussians, raw_xyz, raw_rot,rel_trans=None):
        RT = rel_trans.cuda()
        R = RT[..., :3] 
        RT_gathered = RT[self.indices_] 
        rotations = RT_gathered[..., :3] 
        translations = RT_gathered[..., 3] 
        pos = raw_xyz.detach()
        pos_expanded = pos.unsqueeze(1).expand(-1, self.indices_.size(1), -1)

        pos_transformed = torch.einsum('gkij,gkj->gki', rotations, pos_expanded) + translations
        pos_out = torch.sum(pos_transformed * self.graph_weights_, dim=1) 
        R_out = torch.einsum('gkij,gk->gij', rotations, self.graph_weights_.squeeze(-1))
        rots = norm_quaternion(raw_rot.detach())
        R = batch_qvec2rotmat_torch(rots)
        result = torch.matmul(R_out, R)
        new_rots = batch_rotmat2qvec_torch(result)

        gaussians._xyz = pos_out
        gaussians._rotation = new_rots

    def warping(self, gaussians, raw_xyz, raw_rot):
        RT = self.rel_trans.cuda()
        R = RT[..., :3] 
        RT_gathered = RT[self.indices_] 
        rotations = RT_gathered[..., :3] 
        translations = RT_gathered[..., 3] 
        pos = raw_xyz.detach()
        pos_expanded = pos.unsqueeze(1).expand(-1, self.indices_.size(1), -1)

        pos_transformed = torch.einsum('gkij,gkj->gki', rotations, pos_expanded) + translations
        pos_out = torch.sum(pos_transformed * self.graph_weights_, dim=1) 
        R_out = torch.einsum('gkij,gk->gij', rotations, self.graph_weights_.squeeze(-1))
        rots = norm_quaternion(raw_rot.detach())
        R = batch_qvec2rotmat_torch(rots)
        result = torch.matmul(R_out, R)
        new_rots = batch_rotmat2qvec_torch(result)

        gaussians._xyz = pos_out
        gaussians._rotation = new_rots


