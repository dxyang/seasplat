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
        self.sh_degree = 0 #3 # dxy: default to SH 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.znear=0.01
        self.zfar=100.0
        self.subsample = 0
        self.skip_first_n_images = 0
        self.start_cam = -1
        self.end_cam = -1
        self.rescale_units = 1.0
        self.scene_bounds_xxyyzz = None
        self.do_scene_bb = False
        self.bb_xlo = 0.0
        self.bb_xhi = 0.0
        self.bb_ylo = 0.0
        self.bb_yhi = 0.0
        self.bb_zlo = 0.0
        self.bb_zhi = 0.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

        self.do_isotropic = False
        self.freeze_gs_from_iter = 9_000_000    # freeze the gaussian splat (do not update parameters)
        self.unfreeze_gs_from_iter = 9_000_000  # unfreeze the gaussian splat (do not update parameters)
        self.shuffle = True                     # shuffle cameras
        self.use_depth_weighted_l1 = False
        self.use_depth_weighted_l2 = False

        # background
        self.learn_background = True          # learn background color that is composited with splat render
        self.bg_lambda = 0.01                # lambda for background alpha loss
        self.add_bg_binf = False              # lambda for background alpha loss
        self.use_lab = False                  # use color difference in L*a*b* space for calculating close colors to background
        self.bg_from_bs = True               # once seathru is enabled, do not use the learned background
        self.bg_lr = 1e-2                     # learning rate for background values
        self.alpha_bg_opacities = False
        self.alpha_binf_uw = True
        self.alpha_binf_render = False
        self.alpha_bg_uw = False
        self.turn_off_bg_loss = False

        # depth
        self.use_gt_depth = False             # swap out rendered depth with pseudo ground truth depth
        self.use_depth_l1_loss = False        # use l1 depth reconstruction loss
        self.use_depth_smooth_loss = True    # depth smoothness loss based on xy gradients
        self.depth_smooth_lambda = 2.0       # lambda for depth smooth loss
        self.filter_depth = True             # depth = depth / alpha with nans filled in as depth.max() or depth = depth + (1 - alpha) * depth.max() or both
        self.normalize_depth = 1.0            # normalize depth by this value (if not -1) for use in downstream network input
        self.norm_depth_max = True
        self.depth_alpha_threshold = 0.5

        self.use_alpha_smooth_loss = False    # alpha smoothness loss based on xy gradients
        self.alpha_smooth_lambda = 1.0       # lambda for depth smooth loss

        self.use_opacity_prior = False
        self.opacity_prior_lambda = 0.0001

        # general losses
        self.add_recon_depth_l1 = True
        self.dwr_lambda = 1.0

        self.use_dcp_loss = True             # use dark channel prior loss
        self.dcp_loss_lambda = 1.0             # use dark channel prior loss

        self.use_rgb_sat_loss = True         # use RGB saturation loss
        self.sat_loss_lambda = 2.0         #

        self.use_gw_loss = True              # use gray world prior loss
        self.gw_loss_lambda = 0.1            # used to be 0.01 /shrug
        self.gw_reverse_J = False
        self.use_render_for_gw = False
        self.gw_detach_alpha_bg = False
        self.gw_from_iter = 10_000
        self.gw_filter_by_alpha = 0.0

        self.use_rgb_sv_loss = False          # use RGB spatial variation loss from DeepSeeColor

        self.use_binf_loss = False            # B_inf should approach the atmospheric light in the image
        self.binf_loss_lambda = 1.0
        self.use_dsc_at_loss = False          # DeepSeeColor attenuation loss term
        self.dsc_at_lambda = 1.0

        # seathru
        self.do_seathru = False
        self.bs_at_lr = 1e-2
        self.bs_scale = 5.0                   # model parameters for conv scaled between 0 and scale (i.e. depth dependent effect has 1/e^scale most change in color value)
        self.at_scale = 5.0                   # model parameters for conv scaled between 0 and scale (i.e. depth dependent effect has 1/e^scale most change in color value)
        self.do_sigmoid_bs = False
        self.do_sigmoid_at = False
        self.use_bs_residual = False          # use the residual terms in equation 10 from SeaThru ("depending on the scene, the residual can be left out...")
        self.use_at_v2 = False                # use attenuate net v2 (drops some terms)
        self.use_at_v3 = True                # use attenuate net v3 implx (simplest)
        self.disable_attenuation = False      # simplified model that only accounts for backscatter

        self.seathru_from_iter = 9_000_000
        self.update_bs_at_interval = 100      # every num gs updates, update the bs and at models
        self.update_bs_at_count = 50          # update bs and at models this many times

        self.scale_grad_threshold = 1.0
        self.do_z_score = False               # z threshold direct image to +/- 5 stdevs

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
