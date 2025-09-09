import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch.nn.functional as F
import os
from rich.console import Console
CONSOLE = Console(width=120)
from utils.calc_utils import *

class node_graph:
    regular_xyz = None
    regular_xyz_joint = None
    
    def regular_term_setup(self, GaussianModel, velocity_option=False, warpDQB=None):
        if self.regular_xyz is None:
            self.regular_xyz = GaussianModel._xyz.clone().detach()
        if velocity_option:
            warpDQB.xyz_velocity = GaussianModel._xyz.clone().detach() - self.regular_xyz
        self.regular_xyz = GaussianModel._xyz.clone().detach()
        self.regular_features_dc = GaussianModel._features_dc.clone().detach()
        self.regular_features_rest = GaussianModel._features_rest.clone().detach()
        self.regular_scaling = GaussianModel._scaling.clone().detach()
        self.regular_rotation = GaussianModel._rotation.clone().detach()
        self.regular_opacity = GaussianModel._opacity.clone().detach()

        self.pre_rotations_inv = quaternion_inverse(self.regular_rotation)
        self.prev_neighbor_points = self.regular_xyz[self.indices_]
        self.prev_diff = self.regular_xyz.unsqueeze(1) - self.prev_neighbor_points

    def graph_init(self, xyz, k=8, load_graph_path=None, filename=None, l=0.02):
        points_np = xyz.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points_np)
        self.dist_, self.indices_ = nbrs.kneighbors(points_np)
        self.dist_ = torch.tensor(self.dist_[:,1:], dtype=torch.float32, requires_grad=False).cuda()
        self.indices_ = torch.tensor(self.indices_[:,1:], device=self.dist_.device, dtype=torch.long)

        self.graph_weights_ = torch.exp(-1 * self.dist_ ** 2 / l ** 2)
        self.graph_weights_ = self.graph_weights_ / self.graph_weights_.sum(dim=1, keepdim=True)
        self.graph_weights_ = self.graph_weights_.unsqueeze(-1) 
        self.indices_ = self.indices_.to(dtype=torch.long).detach()

        if load_graph_path:
            self.load_graph_from_file(load_graph_path, points_np)
        if filename:
            self.save_graph_to_file(filename, points_np, self.indices_)
    
    def save_graph_to_file(self, filename, points, edges):
        with open(filename, 'w') as f:
            for i in range(points.shape[0]):
                f.write(f"v {points[i, 0]} {points[i, 1]} {points[i, 2]}\n")
            for i in range(edges.shape[0]):
                for j in range(edges.shape[1]):
                    f.write(f"l {i+1} {edges[i, j]+1}\n")
    
    def load_graph_from_file(self, load_graph_path, points_np):
            indices_path = os.path.join(load_graph_path, 'indices.npy')
            if os.path.exists(indices_path):
                CONSOLE.log(f"Loading indices from {indices_path}")
                self.indices_ = torch.tensor(np.load(indices_path), device=self.dist_.device, dtype=torch.long)
            else:
                CONSOLE.log(f"Saving indices to {indices_path}")
                np.save(indices_path, self.indices_.cpu().numpy())

    def compute_loss(self, gaussians_canonical, lossp,  is_start_frame):
        loss = 0
        loss_info = {}
        
        if lossp.scaling_term:
            scaling_loss = lossp.alpha_scaling * scaling_control_loss(gaussians_canonical, threshold_coefficient=lossp.scaling_threshold)
            loss_info["scaling"] = scaling_loss.item()
            loss += scaling_loss
            
        if  lossp.isotropic_term:
            isotropic_loss = lossp.alpha_isotropic * compute_isotropic_loss(gaussians_canonical)
            loss_info["isotropic"] = isotropic_loss.item()
            loss += isotropic_loss

        if is_start_frame == False and lossp.graph_term:
            rigid_loss = lossp.alpha_rigid * self.compute_rigid_loss(gaussians_canonical)
            loss_info["rigid"] = rigid_loss.item()
            loss += rigid_loss

        if is_start_frame == False and lossp.regular_term:
            regular_loss_pos, regular_loss_color = self.compute_regular_loss(gaussians_canonical)
            regular_loss_pos = lossp.alpha_regular_position * regular_loss_pos
            regular_loss_color = lossp.alpha_regular * regular_loss_color
            loss_info["reg_pos"] = regular_loss_pos.item()
            loss_info["reg_col"] = regular_loss_color.item()
            loss += regular_loss_pos + regular_loss_color

        return loss, loss_info
    
    # @torch.compile # for acceleration, need torch >= 2.0
    def compute_rigid_loss(self, GaussianModel):
        rotations = GaussianModel._rotation 
        rel_rotations = quaternion_multiply(norm_quaternion(rotations), self.pre_rotations_inv)
        rel_rotations = norm_quaternion(rel_rotations)
        rel_rots = build_rotation(rel_rotations)
        neighbor_points = GaussianModel._xyz[self.indices_]
        curr_diff = GaussianModel._xyz.unsqueeze(1) - neighbor_points
        offset = torch.einsum('bij,bnj->bni', rel_rots, self.prev_diff) - curr_diff
        loss = torch.sum(  (self.graph_weights_ * ( offset) ** 2).sum(2).sum(1)).mean()

        return loss
    
    def compute_regular_loss(self, GaussianModel):
        loss_features_dc = torch.norm(GaussianModel._features_dc - self.regular_features_dc, p = 2)
        loss_features_rest = torch.norm(GaussianModel._features_rest - self.regular_features_rest, p = 2)
        loss_scaling = torch.norm(GaussianModel._scaling - self.regular_scaling, p = 2)
        loss_opacity = torch.norm(GaussianModel._opacity - self.regular_opacity, p = 2)
        regular_loss_pos =  ( loss_opacity + loss_scaling) 
        regular_loss_color = (loss_features_dc + loss_features_rest ) 

        return regular_loss_pos, regular_loss_color
    
    
    def add_velocity_next(self, GaussianModel):
        GaussianModel._xyz = GaussianModel.get_xyz + self.xyz_velocity
        

def compute_isotropic_loss( GaussianModel, r=4):
    scaling_exp = torch.exp(GaussianModel.get_scaling_ori)
    epsilon = 1e-8 
    max_val, _ = torch.max(scaling_exp, dim=1)
    min_val, _ = torch.min(scaling_exp, dim=1)
    ratio = torch.max(max_val / (min_val + epsilon), torch.tensor([r]).cuda())
    ratio = torch.nan_to_num(ratio, nan=0.0)
    loss = torch.mean(ratio) - r

    return loss


def scaling_control_loss(GaussianModel, threshold_coefficient=2, lower_threshold_coefficient=0.2):
    avg_scaling = GaussianModel.get_scaling.mean().detach()
    upper_threshold = avg_scaling * threshold_coefficient
    lower_threshold = avg_scaling * lower_threshold_coefficient
    scaling = GaussianModel.get_scaling.max(axis = 1)[0]
    excess = scaling - upper_threshold
    deficit = lower_threshold - scaling
    positive_excess = F.relu(excess)
    positive_deficit = F.relu(deficit)
    loss = positive_excess.sum() + positive_deficit.sum()

    return loss

