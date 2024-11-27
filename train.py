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
from datetime import datetime
import glob
import json
from pathlib import Path
import torch
from typing import List
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_weighted_l1_loss, depth_weighted_l2_loss
from gaussian_renderer import render, render_depth, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.general_utils import inverse_sigmoid
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# import kornia.morphology as morph
from deepseecolor.models import (
    AttenuateNet,
    AttenuateNetV2,
    AttenuateNetV3,
    BackscatterNet,
    BackscatterNetV2,
)
from deepseecolor.losses import (
    AlphaBackgroundLoss,
    AttenuateLoss,
    BackscatterLoss,
    DarkChannelPriorLoss,
    DarkChannelPriorLossV2,
    DarkChannelPriorLossV3,
    GrayWorldPriorLoss,
    RgbSpatialVariationLoss,
    RgbSaturationLoss,
    mixture_of_laplacians_loss
)
from deepseecolor.depth_losses import SmoothDepthLoss

from lpipsPyTorch import lpips
from metrics import readImages, read_image
from render_uw import estimate_atmospheric_light, render_set
from utils.sh_utils import SH2RGB

def training(model_params, opt_params, pipe_params, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if model_params.do_scene_bb:
        scene_bounds_xxyyzz = [
            model_params.bb_xlo,
            model_params.bb_xhi,
            model_params.bb_ylo,
            model_params.bb_yhi,
            model_params.bb_zlo,
            model_params.bb_zhi,
        ]
        print(f"boundaries for camera/pcd: {scene_bounds_xxyyzz}")
        model_params.scene_bounds_xxyyzz = scene_bounds_xxyyzz

    first_iter = 0
    tb_writer = prepare_output_and_logger(model_params)
    gaussians = GaussianModel(model_params.sh_degree, opt_params.do_isotropic)
    scene = Scene(model_params, gaussians, shuffle=opt_params.shuffle)

    # deep see color
    if opt_params.do_seathru:
        bs_model = BackscatterNetV2(use_residual=opt_params.use_bs_residual, scale=opt_params.bs_scale, do_sigmoid=opt_params.do_sigmoid_bs).cuda()
        if opt_params.use_at_v2:
            at_model = AttenuateNetV2(scale=opt_params.at_scale, do_sigmoid=opt_params.do_sigmoid_at).cuda()
        elif opt_params.use_at_v3:
            at_model = AttenuateNetV3(scale=opt_params.at_scale, do_sigmoid=opt_params.do_sigmoid_at, init_vals=not opt_params.do_sigmoid_at).cuda()
        else:
            at_model = AttenuateNet(scale=opt_params.at_scale, do_sigmoid=opt_params.do_sigmoid_at).cuda()
        bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=opt_params.bs_at_lr)
        at_optimizer = torch.optim.Adam(at_model.parameters(), lr=opt_params.bs_at_lr)

    depth_smooth_criterion = SmoothDepthLoss().cuda()
    gw_criterion = GrayWorldPriorLoss().cuda()
    rgb_sv_criterion = RgbSpatialVariationLoss().cuda()
    rgb_01_criterion = RgbSaturationLoss(saturation_val=1.0).cuda()
    rgb_sat_criterion = RgbSaturationLoss(saturation_val=0.7).cuda()
    alpha_bg_criterion = AlphaBackgroundLoss(use_kornia=opt_params.use_lab).cuda()

    dsc_at_criterion = AttenuateLoss().cuda()
    dcp_criterion = DarkChannelPriorLossV3().cuda()

    gaussians.training_setup(opt_params)
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt_params)

    '''
    this bg color gets passed to the renderer
    '''
    bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if opt_params.random_background:
        bg = torch.rand((3), device="cuda")
    else:
        bg = background

    '''
    this lets us learn background
    '''
    learned_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    if opt_params.learn_background:
        assert not model_params.white_background
        bg_init = torch.rand((3), device="cuda")
        bg_init[2] = 0.8 # make the b channel high to start
        bg_init[1] = 0.25 # make the g channel high to start
        bg_init[0] = 0.05 # make the r channel low to start
        learned_bg = torch.nn.Parameter(inverse_sigmoid(bg_init.requires_grad_(True)))
        bg_optimizer = torch.optim.Adam([learned_bg], lr=opt_params.bg_lr)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_params.iterations), desc="Training progress")
    first_iter += 1

    # allocate tensors outside the training loop
    depth_l1_loss = torch.Tensor([0.0]).squeeze().cuda()
    depth_smooth_loss = torch.Tensor([0.0]).squeeze().cuda()
    dcp_loss = torch.Tensor([0.0]).squeeze().cuda()
    gw_loss = torch.Tensor([0.0]).squeeze().cuda()
    rgb_01_loss = torch.Tensor([0.0]).squeeze().cuda()
    rgb_sv_loss = torch.Tensor([0.0]).squeeze().cuda()
    rgb_sat_loss = torch.Tensor([0.0]).squeeze().cuda()
    alpha_bg_loss = torch.Tensor([0.0]).squeeze().cuda()
    dsc_at_loss = torch.Tensor([0.0]).squeeze().cuda()
    binf_loss = torch.Tensor([0.0]).squeeze().cuda()
    alpha_smooth_loss = torch.Tensor([0.0]).squeeze().cuda()
    opacity_prior_loss = torch.Tensor([0.0]).squeeze().cuda()
    dl1 = torch.Tensor([0.0]).squeeze().cuda()

    '''
    training loop
    '''
    iteration = 1
    bs_update_counter = 0
    at_update_counter = 0
    bs_update_iter = 0
    at_update_iter = 0
    at_inited = False
    bs_inited = False

    update_gs_color_counter = 0
    adjust_gs_colors_for_cc = False

    done_binf_init_with_bg = False
    print_message_once = False
    while iteration < opt_params.iterations + 1:
        # # maybe this will help?
        # torch.cuda.empty_cache()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Freeze the gaussian model as necessary
        if iteration == opt_params.freeze_gs_from_iter:
            gaussians.freeze_parameters(xyz=True, colors=False, opacity=True, scaling=True, rotation=True)
        if iteration == opt_params.unfreeze_gs_from_iter:
            print(f"[{iteration}] unfreezing gs parameters")
            gaussians.freeze_parameters(xyz=False, colors=False, opacity=False, scaling=False, rotation=False)
        if iteration == opt_params.seathru_from_iter + 1:
            if not print_message_once:
                print(f"[{iteration}] freezing gs parameters except color at seathru + 1")
                print_message_once = True
            gaussians.freeze_parameters(xyz=True, colors=False, opacity=True, scaling=True, rotation=True)
        if iteration == opt_params.seathru_from_iter + 2:
            print(f"[{iteration}] unfreezing gs parameters at seathru + 2")
            gaussians.freeze_parameters(xyz=False, colors=False, opacity=False, scaling=False, rotation=False)


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe_params, bg)
        rendered_image, image_alpha, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        if opt_params.learn_background:
            if opt_params.bg_from_bs and opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
                # do not use learned background; rely on backscatter to hopefully fill this in
                image = rendered_image
                if not done_binf_init_with_bg:
                    print(f"[{iteration} updated bs_model B_inf with learnred_bg parameters {torch.sigmoid(learned_bg)}")
                    bs_model.B_inf = torch.nn.Parameter(torch.clone(learned_bg.data).reshape(3, 1, 1)).cuda()
                    bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=opt_params.bs_at_lr)
                    done_binf_init_with_bg = True
            else:
                bg_image = torch.sigmoid(learned_bg).reshape(3, 1, 1) * (1 - image_alpha)
                bgdetached_image = torch.sigmoid(learned_bg.detach()).reshape(3, 1, 1) * (1 - image_alpha)
                image = rendered_image + bg_image
        else:
            image = rendered_image

        render_depth_pkg = render_depth(viewpoint_cam, gaussians, pipe_params, bg)
        depth_image = render_depth_pkg["render"][0].unsqueeze(0)
        if opt_params.filter_depth:
            depth_image = depth_image / image_alpha
            if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
                valid_depth_vals = depth_image[torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))]
                if len(valid_depth_vals) == 0:
                    print(f"[training] everything is nan")
                    not_nan_max = 100.0
                else:
                    not_nan_max = torch.max(valid_depth_vals).item()
                depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
            depth_image = depth_image / opt_params.normalize_depth
            if opt_params.norm_depth_max:
                if depth_image.min() != depth_image.max():
                    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
                else:
                    depth_image = depth_image = depth_image / depth_image.max()

        if opt_params.use_gt_depth:
            assert viewpoint_cam.original_depth_image is not None
            depth_image = viewpoint_cam.original_depth_image.cuda()

        '''
        deep see color
        '''
        if opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
            image_batch = torch.unsqueeze(image, dim=0)
            depth_image_batch = torch.unsqueeze(depth_image, dim=0)

            # estimate attenuation
            if opt_params.disable_attenuation:
                direct = image_batch
            else:
                attenuation_map = at_model(depth_image_batch)
                attenuation_map_depth_detached = at_model(depth_image_batch.detach())
                direct = image_batch * attenuation_map

            # z score
            if opt_params.do_z_score:
                direct_mean = direct.mean(dim=[2, 3], keepdim=True)
                direct_std = direct.std(dim=[2, 3], keepdim=True)
                direct_z = (direct - direct_mean) / direct_std
                clamped_z = torch.clamp(direct_z, -3, 3)
                direct_filtered = torch.clamp(
                    (clamped_z * direct_std) + torch.maximum(direct_mean, torch.Tensor([1. / 255]).cuda()), 0, 1)
                direct = direct_filtered

            # estimate backscatter
            backscatter = bs_model(depth_image_batch)
            backscatter_depth_detached = bs_model(depth_image_batch.detach())

            # combined image
            underwater_image = torch.clamp(direct + backscatter, 0.0, 1.0)

        # gaussian splating loss
        gt_image = viewpoint_cam.original_image.cuda()
        if iteration > opt_params.seathru_from_iter and opt_params.do_seathru:
            if opt_params.use_depth_weighted_l1:
                Ll1 = depth_weighted_l1_loss(underwater_image, gt_image, depth_image.detach())
            elif opt_params.use_depth_weighted_l2:
                Ll1 = depth_weighted_l2_loss(underwater_image, gt_image, depth_image.detach())
            else:
                Ll1 = l1_loss(underwater_image, gt_image)
            l_ssim = ssim(underwater_image, gt_image)
        else:
            if opt_params.use_depth_weighted_l1:
                Ll1 = depth_weighted_l1_loss(image, gt_image, depth_image.detach())
            elif opt_params.use_depth_weighted_l2:
                Ll1 = depth_weighted_l2_loss(image, gt_image, depth_image.detach())
            else:
                Ll1 = l1_loss(image, gt_image)
            l_ssim = ssim(image, gt_image)
        loss = (1.0 - opt_params.lambda_dssim) * Ll1 + opt_params.lambda_dssim * (1.0 - l_ssim)

        # depth weighted l1
        if opt_params.add_recon_depth_l1:
            if iteration > opt_params.seathru_from_iter and opt_params.do_seathru:
                dl1 = depth_weighted_l1_loss(underwater_image, gt_image, depth_image.detach())
            else:
                dl1 = depth_weighted_l1_loss(image, gt_image, depth_image.detach())
            loss += opt_params.dwr_lambda * dl1

        # binary accumulation loss
        if opt_params.use_opacity_prior:
            opacity_prior_loss = mixture_of_laplacians_loss(scene.gaussians.get_opacity)
            loss += opt_params.opacity_prior_lambda * opacity_prior_loss

        # alpha loss
        if opt_params.learn_background:
            alpha_bg_loss = alpha_bg_criterion(rendered_image.detach(), torch.sigmoid(learned_bg.detach()), image_alpha)
            if opt_params.alpha_bg_opacities:
                rgb_colors = SH2RGB(gaussians.get_features).squeeze()
                # alpha_bg_loss = alpha_bg_criterion(rgb_colors.detach(), torch.sigmoid(learned_bg.detach()), gaussians.get_opacity.squeeze())
            if opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
                if opt_params.alpha_binf_uw or opt_params.alpha_binf_render or opt_params.alpha_bg_opacities:
                    if not opt_params.add_bg_binf:
                        alpha_bg_loss = torch.Tensor([0.0]).squeeze().cuda()
                    if opt_params.alpha_binf_uw:
                        alpha_bg_loss += alpha_bg_criterion(underwater_image.squeeze().detach(), torch.sigmoid(bs_model.B_inf.detach()), image_alpha)
                    if opt_params.alpha_bg_uw:
                        alpha_bg_loss += alpha_bg_criterion(underwater_image.squeeze().detach(), torch.sigmoid(bs_model.B_inf.detach()), image_alpha)
                    if opt_params.alpha_binf_render:
                        alpha_bg_loss += alpha_bg_criterion(rendered_image.detach(), torch.sigmoid(bs_model.B_inf.detach()), image_alpha)
                    if opt_params.alpha_bg_opacities:
                        alpha_bg_loss += alpha_bg_criterion(rgb_colors.detach(), torch.sigmoid(bs_model.B_inf.detach()).squeeze(), gaussians.get_opacity.squeeze())
                elif opt_params.bg_from_bs:
                    if opt_params.turn_off_bg_loss:
                        alpha_bg_loss = torch.Tensor([0.0]).squeeze().cuda()
                    else:
                        alpha_bg_loss = alpha_bg_criterion(underwater_image.squeeze().detach(), torch.sigmoid(learned_bg.detach()), image_alpha)
            loss += opt_params.bg_lambda * alpha_bg_loss

        # depth l1 loss
        if opt_params.use_depth_l1_loss:
            gt_depth_image = viewpoint_cam.original_depth_image.cuda()
            depth_l1_loss = l1_loss(depth_image, gt_depth_image)
            loss += 0.1 * depth_l1_loss

        # depth smooth loss
        if opt_params.use_depth_smooth_loss:
            gt_rgb_batch = torch.unsqueeze(gt_image, dim=0)
            depth_image_batch = torch.unsqueeze(depth_image, dim=0)
            depth_smooth_loss = depth_smooth_criterion(gt_rgb_batch, depth_image_batch)
            loss += opt_params.depth_smooth_lambda * depth_smooth_loss # used to be 0.01

        # alpha smooth loss
        if opt_params.use_alpha_smooth_loss:
            gt_rgb_batch = torch.unsqueeze(gt_image, dim=0)
            alpha_image_batch = torch.unsqueeze(image_alpha, dim=0)
            alpha_smooth_loss = depth_smooth_criterion(gt_rgb_batch, alpha_image_batch)
            loss += opt_params.alpha_smooth_lambda * alpha_smooth_loss

        # gray world loss
        if opt_params.use_gw_loss:
            if opt_params.use_render_for_gw:
                image_batch = torch.unsqueeze(rendered_image, dim=0)
            elif opt_params.gw_detach_alpha_bg:
                image_batch = torch.unsqueeze(rendered_image + bgdetached_image, dim=0)
            elif opt_params.gw_reverse_J:
                J = direct.detach() / attenuation_map
                image_batch = J
            else:
                image_batch = torch.unsqueeze(image, dim=0)

            if opt_params.gw_filter_by_alpha != 0.0:
                mask = image_alpha.detach() > opt_params.gw_filter_by_alpha
                image_batch = image_batch[:, :, mask.squeeze()]

            gw_loss = gw_criterion(image_batch)
            if iteration > opt_params.gw_from_iter:
                loss += opt_params.gw_loss_lambda * gw_loss

        # rgb spatial variation loss
        if iteration > opt_params.seathru_from_iter and opt_params.do_seathru:
            if opt_params.use_rgb_sv_loss:
                image_batch = torch.unsqueeze(image, dim=0)
                rgb_sv_loss = rgb_sv_criterion(image_batch.detach(), direct)
                loss += 0.01 * rgb_sv_loss

        # rgb satuation loss
        if iteration > opt_params.seathru_from_iter and opt_params.do_seathru:
            if opt_params.use_rgb_sat_loss:
                image_batch = torch.unsqueeze(image, dim=0)
                rgb_sat_loss = rgb_sat_criterion(image_batch)
                loss += opt_params.sat_loss_lambda * rgb_sat_loss

        # B_inf loss
        if iteration > opt_params.seathru_from_iter and opt_params.do_seathru:
            if opt_params.use_binf_loss:
                image_batch_detached = torch.unsqueeze(image, dim=0).detach()
                binf_loss = bs_model.forward_rgb(image_batch_detached)
                loss += opt_params.binf_loss_lambda * binf_loss

        # deep see color backscatter loss for bs network
        # dark channel prior loss
        if opt_params.use_dcp_loss:
            gt_rgb_batch = torch.unsqueeze(gt_image, dim=0)
            if opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
                # direct_grad_only_through_bsmodel_batch = underwater_image.detach() - backscatter_depth_detached
                direct_reversefromgt_through_bsmodel_batch = gt_rgb_batch.detach() - backscatter_depth_detached
                depth_image_batch = torch.unsqueeze(depth_image, dim=0)
                dcp_loss, _ = dcp_criterion(direct_reversefromgt_through_bsmodel_batch, depth_image_batch.detach())
                loss += opt_params.dcp_loss_lambda * dcp_loss
            else:
                # image_batch = torch.unsqueeze(image, dim=0)
                depth_image_batch = torch.unsqueeze(depth_image, dim=0)
                dcp_loss, _ = dcp_criterion(gt_rgb_batch, depth_image_batch.detach())

        # deep see color attenuation loss for at network
        if opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
            if opt_params.use_dsc_at_loss:
                gt_rgb_batch = torch.unsqueeze(gt_image, dim=0)
                direct_reversefromgt_detached = (gt_rgb_batch - backscatter).detach()
                if opt_params.disable_attenuation:
                    J_through_atmodel = torch.zeros_like(direct_reversefromgt_detached)
                else:
                    J_through_atmodel = direct_reversefromgt_detached / attenuation_map_depth_detached
                dsc_at_loss = dsc_at_criterion(direct_reversefromgt_detached, J_through_atmodel)
                loss += opt_params.dsc_at_lambda * dsc_at_loss

        gaussians.optimizer.zero_grad(set_to_none = True)
        if opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
            bs_optimizer.zero_grad()
            at_optimizer.zero_grad()
        if opt_params.learn_background:
            bg_optimizer.zero_grad()

        loss.backward()

        '''
        periodically update the backscatter and attenuation functions for a given splat
        '''
        if opt_params.do_seathru and iteration > opt_params.seathru_from_iter :
            do_at_bs_update = (not bs_inited and not at_inited) or (iteration % opt_params.update_bs_at_interval == 0)
            if do_at_bs_update:
                update_count = opt_params.update_bs_at_count if bs_inited else 1000
                if bs_update_counter == update_count:
                    bs_update_counter = 0
                    at_update_counter = 0
                    if not bs_inited and not at_inited:
                        print(f"[{iteration}] Backscatter and attenuation init'd")
                        bs_inited = True
                        at_inited = True
                        adjust_gs_colors_for_cc = True
                else:
                    bs_optimizer.step()
                    at_optimizer.step()

                    bs_update_counter += 1
                    at_update_counter += 1
                    bs_update_iter += 1
                    at_update_iter += 1
                    continue

            # adjust gaussian splat colors the first time the color correction models are used
            if adjust_gs_colors_for_cc:
                if update_gs_color_counter == 2000:
                    print(f"[{iteration}] Adjusted gs colors for color correction")
                    adjust_gs_colors_for_cc = False
                else:
                    gaussians.optimizer.step()
                    update_gs_color_counter += 1
                    continue

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt_params.iterations:
                progress_bar.close()

            # Log and save
            if opt_params.do_seathru and iteration > opt_params.seathru_from_iter:
                training_report(
                    tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe_params, background),
                    learned_bg, opt_params.learn_background,
                    rgb_01_loss, rgb_01_criterion,
                    opt_params.normalize_depth, opt_params.norm_depth_max, depth_l1_loss, opt_params.use_gt_depth,
                    opt_params.do_seathru, opt_params.seathru_from_iter, opt_params.bg_from_bs,
                    None, dsc_at_loss, bs_model, at_model, None, dsc_at_criterion,
                    opt_params.do_z_score, opt_params.filter_depth, opt_params.disable_attenuation,
                    dcp_loss=dcp_loss, dcp_criterion=dcp_criterion,
                    depth_smooth_loss=depth_smooth_loss, depth_smooth_criterion=depth_smooth_criterion, alpha_smooth_loss=alpha_smooth_loss,
                    gw_loss=gw_loss, gw_criterion=gw_criterion,
                    rgb_sv_loss=rgb_sv_loss, rgb_sv_criterion=rgb_sv_criterion,
                    rgb_sat_loss=rgb_sat_loss, rgb_sat_criterion=rgb_sat_criterion,
                    alpha_bg_loss=alpha_bg_loss, alpha_bg_criterion=alpha_bg_criterion,
                    depth_alpha_threshold=opt_params.depth_alpha_threshold,
                    use_render_for_gw=opt_params.use_render_for_gw,
                    binf_loss=binf_loss,
                    opacity_prior_loss=opacity_prior_loss,
                    recon_depth_loss=dl1,
                )
            else:
                training_report(
                    tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe_params, background),
                    learned_bg, opt_params.learn_background,
                    rgb_01_loss, rgb_01_criterion,
                    opt_params.normalize_depth, opt_params.norm_depth_max, depth_l1_loss, opt_params.use_gt_depth,
                    filter_depth=opt_params.filter_depth,
                    dcp_loss=dcp_loss, dcp_criterion=dcp_criterion,
                    depth_smooth_loss=depth_smooth_loss, depth_smooth_criterion=depth_smooth_criterion, alpha_smooth_loss=alpha_smooth_loss,
                    gw_loss=gw_loss, gw_criterion=gw_criterion,
                    rgb_sat_loss=rgb_sat_loss, rgb_sat_criterion=rgb_sat_criterion,
                    alpha_bg_loss=alpha_bg_loss, alpha_bg_criterion=alpha_bg_criterion,
                    depth_alpha_threshold=opt_params.depth_alpha_threshold,
                    use_render_for_gw=opt_params.use_render_for_gw,
                    opacity_prior_loss=opacity_prior_loss,
                    recon_depth_loss=dl1,
                )

            # Densification
            if iteration < opt_params.densify_until_iter:
                if iteration < opt_params.freeze_gs_from_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration % opt_params.densification_interval == 0:
                        size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                        if iteration > opt_params.densify_from_iter:
                            gaussians.densify_and_prune(opt_params.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt_params.opacity_reset_interval == 0 or (model_params.white_background and iteration == opt_params.densify_from_iter):
                        print(f"[{iteration}] opacity reset")
                        gaussians.reset_opacity()
                elif iteration >= opt_params.unfreeze_gs_from_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration % opt_params.densification_interval == 0:
                        size_threshold = 20 if iteration > opt_params.opacity_reset_interval else None
                        if iteration > opt_params.densify_from_iter:
                            gaussians.densify_and_prune(opt_params.scale_grad_threshold * opt_params.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt_params.opacity_reset_interval == 0 or (model_params.white_background and iteration == opt_params.densify_from_iter):
                        print(f"[{iteration}] opacity reset")
                        gaussians.reset_opacity()
                else:
                    # do nothing when gaussian splat is completely frozen
                    pass


            # Optimizer step
            if iteration < opt_params.iterations:
                gaussians.optimizer.step()
                if opt_params.learn_background:
                    bg_optimizer.step()

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint in {scene.model_path}")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if opt_params.do_seathru:
                    torch.save(bs_model.state_dict(), f"{scene.model_path}/backscatter_{iteration}.pth")
                    torch.save(at_model.state_dict(), f"{scene.model_path}/attenuate_{iteration}.pth")
                if opt_params.learn_background:
                    torch.save(learned_bg, f"{scene.model_path}/bg_{iteration}.pth")

        iteration += 1

    '''
    post training save images
    '''
    print(f"Rendering images for eval")
    if opt_params.do_seathru:
        at_model.eval()
        bs_model.eval()
    else:
        bs_model = None
        at_model = None

    skip_eval_train = False
    if 'metashape' in str(model_params.model_path):
        skip_eval_train = True

    train_image_dir = Path(model_params.model_path) / "train" / f"{'with_water' if opt_params.do_seathru else 'render'}"
    gt_dir = Path(model_params.source_path) / model_params.images
    png_images = glob.glob(f"{gt_dir}/*.png")
    use_jpeg = len(png_images) == 0

    if opt_params.bg_from_bs:
        learned_bg = None

    if not skip_eval_train:
        metrics_dirs = [train_image_dir]
        with torch.no_grad():
            render_set(
                Path(model_params.model_path), "train", iteration, scene.getTrainCameras(), gaussians, pipe_params, background,
                opt_params.do_seathru,
                False,
                False,
                learned_bg,
                bs_model,
                at_model,
                save_as_jpeg=use_jpeg,
            )
    else:
        metrics_dirs = []

    if model_params.eval:
        test_image_dir = Path(model_params.model_path) / "test" / f"{'with_water' if opt_params.do_seathru else 'render'}"
        with torch.no_grad():
            render_set(
                Path(model_params.model_path), "test", iteration, scene.getTestCameras(), gaussians, pipe_params, background,
                opt_params.do_seathru,
                False,
                False,
                learned_bg,
                bs_model,
                at_model,
                save_as_jpeg=use_jpeg,
            )
        metrics_dirs.append(test_image_dir)

    '''
    eval images
    '''
    print("Running metrics")
    results = {}
    chunk_size = 128
    for eval_idx, image_dir in enumerate(metrics_dirs):
        ssims = []
        psnrs = []
        lpipss = []

        fname_list = os.listdir(image_dir)
        for i in tqdm(range(0, len(fname_list), chunk_size)):
            upper =  min(i + chunk_size, len(fname_list))
            fnames = fname_list[i : upper]

            renders, gts, image_name = readImages(image_dir, gt_dir, fnames)

            for idx in range(len(renders)):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        print(f"-----------{'Train' if eval_idx == 0 else 'Test'}")
        print("  SSIM ^: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR ^: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS v: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        key = 'Train' if eval_idx == 0 else 'Test'
        results[key] = {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }

    results_file = Path(model_params.model_path) / "eval_metrics.json"
    with open(str(results_file), 'w') as f:
        json.dump(results, f)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    learned_bg, do_learn_bg,
                    rgb_01_loss, rgb_01_criterion,
                    depth_norm_value=1.0, norm_depth_max=False, depth_l1_loss=None, use_gt_depth=False,
                    do_seathru=False, seathru_from_iter=9999999999, bg_from_bs=False,
                    bs_loss=None, at_loss=None, bs_model=None, at_model=None, bs_criterion=None, at_criterion=None,
                    do_z_score=False, filter_depth=False, disable_attenuation=False,
                    dcp_loss=None, dcp_criterion=None,
                    depth_smooth_loss=None, depth_smooth_criterion=None, alpha_smooth_loss=None,
                    gw_loss=None, gw_criterion=None,
                    rgb_sv_loss=None, rgb_sv_criterion=None,
                    rgb_sat_loss=None, rgb_sat_criterion=None,
                    alpha_bg_loss=None, alpha_bg_criterion=None,
                    depth_alpha_threshold=0.0,
                    use_render_for_gw=False,
                    binf_loss=None,
                    opacity_prior_loss=None,
                    recon_depth_loss=None

):
    if tb_writer:
        # if bs_loss is not None:
        #     tb_writer.add_scalar('train_losses/bs_loss', bs_loss.item(), iteration)
        if at_loss is not None:
            tb_writer.add_scalar('train_losses/at_loss', at_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_losses/rgb_01_loss', rgb_01_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/depth_l1_loss', depth_l1_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/depth_smooth_loss', depth_smooth_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/alpha_smooth_loss', alpha_smooth_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/dcp_loss', dcp_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/gw_loss', gw_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/alpha_bg_loss', alpha_bg_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/recon_depth_loss', recon_depth_loss.item(), iteration)
        if binf_loss is not None:
            tb_writer.add_scalar('train_losses/binf_loss', binf_loss.item(), iteration)
        if rgb_sv_loss is not None:
            tb_writer.add_scalar('train_losses/rgb_sv_loss', rgb_sv_loss.item(), iteration)
        if rgb_sat_loss is not None:
            tb_writer.add_scalar('train_losses/rgb_sat_loss', rgb_sat_loss.item(), iteration)
        if opacity_prior_loss is not None:
            tb_writer.add_scalar('train_losses/opacity_prior_loss', opacity_prior_loss.item(), iteration)
        tb_writer.add_scalar('train_losses/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if do_learn_bg:
            tb_writer.add_scalar(f"background/r", torch.sigmoid(learned_bg)[0].item(), global_step=iteration)
            tb_writer.add_scalar(f"background/g", torch.sigmoid(learned_bg)[1].item(), global_step=iteration)
            tb_writer.add_scalar(f"background/b", torch.sigmoid(learned_bg)[2].item(), global_step=iteration)
            if iteration % 100 == 0:
                tb_writer.add_images(f"background/image", torch.sigmoid(learned_bg).reshape(1, 3, 1, 1), global_step=iteration)
        if bs_model is not None:
            for idx, val in enumerate(bs_model.B_inf):
                tb_writer.add_scalar(f'bs_model/B_inf_{idx}', torch.sigmoid(val).item(), iteration)
            if iteration % 100 == 0:
                tb_writer.add_images(f"bs_model/B_inf", torch.sigmoid(bs_model.B_inf).reshape(1, 3, 1, 1), global_step=iteration)
            for idx, val in enumerate(bs_model.backscatter_conv_params):
                if bs_model.do_sigmoid:
                    tb_writer.add_scalar(f'bs_model/backscatter_conv_{idx}', torch.sigmoid(val).item(), iteration)
                else:
                    tb_writer.add_scalar(f'bs_model/backscatter_conv_{idx}', val.item(), iteration)
            if bs_model.use_residual:
                for idx, val in enumerate(bs_model.J_prime):
                    tb_writer.add_scalar(f'bs_model/J_prime_{idx}', torch.sigmoid(val).item(), iteration)
                for idx, val in enumerate(bs_model.residual_conv_params):
                    tb_writer.add_scalar(f'bs_model/residual_conv_{idx}', torch.sigmoid(val).item(), iteration)
                if iteration % 100 == 0:
                    tb_writer.add_images(f"bs_model/J_prime", torch.sigmoid(bs_model.J_prime).reshape(1, 3, 1, 1), global_step=iteration)
        if at_model is not None:
            if type(at_model) is not AttenuateNetV3:
                for idx, val in enumerate(at_model.attenuation_coef):
                    if at_model.do_sigmoid:
                        tb_writer.add_scalar(f'at_model/attenuation_coeff_{idx}', torch.sigmoid(val).item(), iteration)
                    else:
                        tb_writer.add_scalar(f'at_model/attenuation_coeff_{idx}', val.item(), iteration)
            for idx, val in enumerate(at_model.attenuation_conv_params):
                if at_model.do_sigmoid:
                    tb_writer.add_scalar(f'at_model/attenuation_conv_{idx}', torch.sigmoid(val).item(), iteration)
                else:
                    tb_writer.add_scalar(f'at_model/attenuation_conv_{idx}', val.item(), iteration)


    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        assert not torch.is_grad_enabled()
        validation_configs = ({'name': 'eval_test', 'cameras' : scene.getTestCameras()},
                              {'name': 'eval_train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 51, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                dsc_at_test = 0.0
                rgb_01_test = 0.0
                depth_l1_test = 0.0
                depth_smooth_test = 0.0
                alpha_smooth_test = 0.0
                dcp_test = 0.0
                gw_test = 0.0
                rgb_sv_test = 0.0
                rgb_sat_test = 0.0
                alpha_bg_test = 0.0
                binf_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    render_depth_pkg = render_depth(viewpoint, scene.gaussians, *renderArgs)

                    rendered_image = render_pkg["render"]
                    image_alpha = render_pkg["alpha"]
                    depth_image = render_depth_pkg["render"][0].unsqueeze(0)

                    if filter_depth:
                        depth_image = depth_image / image_alpha
                        if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
                            valid_depth_vals = depth_image[torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))]
                            if len(valid_depth_vals) == 0:
                                print(f"[eval] everything is nan")
                                not_nan_max = 100.0
                            else:
                                not_nan_max = torch.max(valid_depth_vals).item()
                            depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
                        depth_image = depth_image / depth_norm_value
                        if norm_depth_max:
                            if depth_image.min() != depth_image.max():
                                depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
                            else:
                                depth_image = depth_image / depth_image.max()

                        depth_mask = image_alpha.detach() > depth_alpha_threshold
                        masked_depth_image = depth_image * depth_mask
                        normalized_masked_depth_image = torch.clamp(masked_depth_image / torch.max(masked_depth_image), 0.0, 1.0)

                    if do_learn_bg:
                        if bg_from_bs and do_seathru and iteration > seathru_from_iter:
                            bg_image = torch.zeros_like(rendered_image) * 1.0
                            image = rendered_image
                        else:
                            bg_image = torch.sigmoid(learned_bg).reshape(3, 1, 1) * (1 - image_alpha)
                            image = rendered_image + bg_image
                        clamped_bg_image = torch.clamp(bg_image, 0.0, 1.0)
                    else:
                        image = rendered_image

                    normalized_depth_image = torch.clamp(depth_image / torch.max(depth_image), 0.0, 1.0)
                    clamped_raw_render_image = torch.clamp(rendered_image, 0.0, 1.0)
                    clamped_rgb_image = torch.clamp(image, 0.0, 1.0)

                    # implement the forward process here
                    if bs_model is not None and at_model is not None:
                        if use_gt_depth:
                            assert viewpoint.original_depth_image is not None
                            depth_image = viewpoint.original_depth_image.cuda()
                        depth_batch = torch.unsqueeze(depth_image, dim=0)
                        rgb_batch = torch.unsqueeze(image, dim=0)
                        if disable_attenuation:
                            attenuation_map = torch.ones_like(rgb_batch)
                        else:
                            attenuation_map = at_model(depth_batch)
                        direct = rgb_batch * attenuation_map
                        attenuation_map_normalized = attenuation_map / torch.max(attenuation_map)

                        if do_z_score:
                            direct_mean = direct.mean(dim=[2, 3], keepdim=True)
                            direct_std = direct.std(dim=[2, 3], keepdim=True)
                            direct_z = (direct - direct_mean) / direct_std
                            clamped_z = torch.clamp(direct_z, -5, 5)
                            direct_filtered = torch.clamp(
                                (clamped_z * direct_std) + torch.maximum(direct_mean, torch.Tensor([1. / 255]).cuda()), 0, 1)
                            direct = direct_filtered

                        backscatter = bs_model(depth_batch)
                        normalized_bs = backscatter / torch.max(backscatter)
                        underwater_image = torch.clamp(direct + backscatter, 0.0, 1.0)

                    # get the groundtruth rgb image
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if viewpoint.original_depth_image is not None:
                        gt_depth_image = viewpoint.original_depth_image.to("cuda")
                        gt_normalized_depth_image = torch.clamp(gt_depth_image / torch.max(gt_depth_image), 0.0, 1.0)

                    # deepseecolor at loss
                    if at_criterion is not None:
                        reverse_direct = underwater_image - backscatter
                        if not disable_attenuation:
                            J = reverse_direct / attenuation_map
                        else:
                            J = torch.zeros_like(reverse_direct)
                        dsc_at_loss = at_criterion(reverse_direct, J)

                    # rgb [0, 1] loss
                    rgb_01_loss = rgb_01_criterion(image)

                    # alpha bg loss
                    alpha_bg_loss = 0.0
                    # if do_learn_bg:
                    #     alpha_bg_loss = alpha_bg_criterion(rendered_image.detach(), torch.sigmoid(learned_bg.detach()), image_alpha)
                    #     if do_seathru and iteration > seathru_from_iter:
                    #         alpha_bg_loss += alpha_bg_criterion(rendered_image.detach(), torch.sigmoid(bs_model.B_inf.detach().clone()), image_alpha)

                    '''
                    losses
                    '''
                    depth_batch = torch.unsqueeze(depth_image, dim=0)
                    rgb_batch = torch.unsqueeze(image, dim=0)
                    gt_rgb_batch = torch.unsqueeze(gt_image, dim=0)
                    alpha_batch = torch.unsqueeze(image_alpha, dim=0)

                    # dcp
                    if do_seathru and iteration > seathru_from_iter:
                        reverse_direct = underwater_image - backscatter
                        dcp_loss, dcp_image = dcp_criterion(reverse_direct, depth_batch)
                    else:
                        dcp_loss, dcp_image = dcp_criterion(rgb_batch, depth_batch)

                    # smooth depth criterion
                    depth_smooth_loss = depth_smooth_criterion(gt_rgb_batch, depth_batch)

                    # smooth alpha
                    alpha_smooth_loss = depth_smooth_criterion(gt_rgb_batch, alpha_batch)

                    # gw criterion
                    if use_render_for_gw:
                        gw_loss = gw_criterion(rendered_image.unsqueeze(0))
                    else:
                        gw_loss = gw_criterion(rgb_batch)

                    # rgb sv and sat criterion
                    if bs_model is not None and at_model is not None:
                        rgb_sv_loss = rgb_sv_criterion(rgb_batch, direct)
                        rgb_sat_loss = rgb_sat_criterion(direct + backscatter)

                    # B_inf loss
                    binf_loss = 0.0

                    if tb_writer and (idx < 5):
                        tag_header = config['name'] + "_view_{}".format(viewpoint.image_name)
                        tb_writer.add_images(f"{tag_header}/render", clamped_raw_render_image[None], global_step=iteration)
                        tb_writer.add_images(f"{tag_header}/depth_render", normalized_depth_image[None], global_step=iteration)
                        # tb_writer.add_images(f"{tag_header}/dcp_image", dcp_image, global_step=iteration)
                        tb_writer.add_images(f"{tag_header}/alpha_image", image_alpha[None], global_step=iteration)
                        # if filter_depth:
                        #     tb_writer.add_images(f"{tag_header}/depth_mask", depth_mask[None], global_step=iteration)
                        if do_learn_bg:
                            tb_writer.add_images(f"{tag_header}/bg_image", clamped_bg_image[None], global_step=iteration)
                            tb_writer.add_images(f"{tag_header}/render (with bg)", clamped_rgb_image[None], global_step=iteration)
                            tb_writer.add_histogram(f"{tag_header}/depth_histogram", depth_image, global_step=iteration)
                            tb_writer.add_histogram(f"{tag_header}/alpha_histogram", image_alpha, global_step=iteration)
                        if bs_model is not None and at_model is not None:
                            tb_writer.add_images(f"{tag_header}/backscatter_image", torch.clamp(backscatter, 0.0, 1.0), global_step=iteration)
                            tb_writer.add_images(f"{tag_header}/backscatter_image_normalized", torch.clamp(normalized_bs, 0.0, 1.0), global_step=iteration)
                            tb_writer.add_images(f"{tag_header}/attenuation_map", torch.clamp(attenuation_map, 0.0, 1.0), global_step=iteration)
                            tb_writer.add_images(f"{tag_header}/attenuation_map_normalized", torch.clamp(attenuation_map_normalized, 0.0, 1.0), global_step=iteration)
                            tb_writer.add_images(f"{tag_header}/attenuated_image", direct, global_step=iteration)
                            tb_writer.add_images(f"{tag_header}/underwater_image", underwater_image, global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{tag_header}/ground_truth", gt_image[None], global_step=iteration)
                            if viewpoint.original_depth_image is not None:
                                tb_writer.add_images(f"{tag_header}/gt_depth_image", gt_normalized_depth_image[None], global_step=iteration)
                                # tb_writer.add_histogram(f"{tag_header}/gt_depth_histogram", gt_depth_image, global_step=iteration)

                    if bs_model is not None and at_model is not None:
                        l1_test += l1_loss(underwater_image.squeeze(), gt_image).mean().double()
                        psnr_test += psnr(underwater_image.squeeze(), gt_image).mean().double()
                        rgb_sv_test += rgb_sv_loss
                        rgb_sat_test += rgb_sat_loss
                    else:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                    if viewpoint.original_depth_image is not None:
                        depth_l1_test += l1_loss(depth_image, gt_depth_image)
                    rgb_01_test += rgb_01_loss
                    dcp_test += dcp_loss
                    depth_smooth_test += depth_smooth_loss
                    alpha_smooth_test += alpha_smooth_loss
                    gw_test += gw_loss
                    binf_test += binf_loss
                    if do_learn_bg:
                        alpha_bg_test += alpha_bg_loss
                    if at_criterion is not None:
                        dsc_at_test += dsc_at_loss

                l1_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                rgb_01_test /= len(config['cameras'])
                dcp_test /= len(config['cameras'])
                depth_smooth_test /= len(config['cameras'])
                alpha_smooth_test /= len(config['cameras'])
                gw_test /= len(config['cameras'])
                binf_test /= len(config['cameras'])
                if viewpoint.original_depth_image is not None:
                    depth_l1_test /= len(config['cameras'])
                if bs_model is not None and at_model is not None:
                    rgb_sv_test /= len(config['cameras'])
                    rgb_sat_test /= len(config['cameras'])
                if do_learn_bg:
                    alpha_bg_test /= len(config['cameras'])
                if at_criterion is not None:
                    dsc_at_test /= len(config['cameras'])

                # write loss stats to tensorboard
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - rgb_01_loss', rgb_01_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - dcp_loss', dcp_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - depth_smooth_loss', depth_smooth_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - alpha_smooth_test', alpha_smooth_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - gw_loss', gw_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - binf_loss', binf_test, iteration)
                    if viewpoint.original_depth_image is not None:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - depth_l1_loss', depth_l1_test, iteration)
                    if bs_model is not None and at_model is not None:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - rgb_sv_loss', rgb_sv_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - rgb_sat_loss', rgb_sat_test, iteration)
                    if do_learn_bg:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - alpha_bg_loss', alpha_bg_test, iteration)
                    if at_criterion is not None:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - dsc_at_loss', dsc_at_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(0, 60_000, 1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 15_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1_000, 7_000, 15_000, 30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--exp", type=str, default = "test")
    parser.add_argument("--seed", type=int, default = -1)
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.insert(0, 1)
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    now = datetime.now()
    today = now.strftime("%m%d%Y")
    args.model_path = str(Path(args.source_path) / "experiments" / today / args.exp)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Print out all the parameters in the run
    for k,v in vars(args).items():
        print(f"{k}: {v}")

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from
    )

    # All done
    print("\nTraining complete.")
