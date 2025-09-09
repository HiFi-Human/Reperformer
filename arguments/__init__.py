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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
class OptimizationParams_Unet():
    def __init__(self):
        self.warm_up_end = 2000
        self.lambda_dssim = 0.2
        self.device = "cuda:0"
        self.random_seed = -1
        self.learning_rate = 5e-6
        self.motion_learning_rate = 1e-3
        self.geo_learning_rate = 1e-2
        self.color_learning_rate = 1e-2

        self.learning_rate_alpha = 0.05

        self.end_iter = 10000000
        self.worker_num = 4
        self.val_mesh_freq = 1000
        self.plot_histogram_freq = 2000

        self.anneal_end = 1000
        self.start_constant_lr = 200000
        self.color_weight = 0.9
        self.ssim_weight = 0.1

        # log parameters
        self.image_interval = 3000
        self.save_interval = 5000
        self.val_interval = 3000
        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init_t1 = 0.00016 / 5
        self.position_lr_final_t1 = 0.0000016 / 5 
        self.joint_gs_num = 50000

        self.position_lr_init_t2 = 0.00016
        self.position_lr_final_t2 = 0.0000016
        
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005 
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 150
        self.opacity_reset_interval = 4000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.prune_least_opacity_iter = 1000

        self.densify_grad_threshold = 0.00005

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)




class LossParamsS1(ParamGroup):
    def __init__(self, parser):
        self.regular_term = False
        self.alpha_regular = 0.0005
        self.alpha_regular_position = 0.000025
        
        self.graph_term = True
        self.alpha_graph = 30000
        self.alpha_rigid = 0.05
        self.isotropic_term = True
        self.alpha_isotropic = 0.002
        self.scaling_term = True
        self.alpha_scaling = 1
        self.scaling_threshold = 5

        super().__init__(parser, "Loss Parameters")




class LossParamsS2():
    def __init__(self):
        self.regular_term = False
        self.alpha_regular = 0.00001
        self.alpha_regular_position = 0.001

        self.scaling_term = False
        self.alpha_scaling = 1

        self.color_term = False
        self.alpha_color = 0.5

        self.graph_term = False
        self.alpha_graph = 300
        self.alpha_rigid = 0.0002
        self.alpha_rotation = 0.000000001

        self.isotropic_term = False
        self.alpha_isotropic = 0.0001

        