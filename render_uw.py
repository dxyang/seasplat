# Script that renders underwater image given RGB+D and backscatter / attenuation parameters
import math
import os
from os import makedirs
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render, render_depth
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from deepseecolor.models import (
    AttenuateNetV3,
    BackscatterNetV2,
)

# WATER_BETA_D = [1.3, 1.2, 0.9]
WATER_BETA_D = [2.6, 2.4, 1.8]
WATER_BETA_B = [1.9, 1.7, 1.4]
WATER_B_INF = [0.07, 0.2, 0.39]
# FOG_BETA_B= 1.2
FOG_BETA_B= 2.4

def render_uw(rgb, d):
    J = rgb

    tmp = torch.from_numpy(np.array(WATER_BETA_D)).view((3, 1, 1, 1)).float().cuda()
    at_exp_coeff = -1.0 * torch.nn.functional.conv2d(d, tmp)
    attenuation = torch.exp(at_exp_coeff)

    tmp2 = torch.from_numpy(np.array(WATER_BETA_B)).view((3, 1, 1, 1)).float().cuda()
    bs_exp_coeff = -1.0 * torch.nn.functional.conv2d(d, tmp2)
    backscatter = 1 - torch.exp(bs_exp_coeff)

    tmp3 = torch.from_numpy(np.array(WATER_B_INF)).view(3, 1, 1).float().cuda()
    I = J * attenuation + tmp3 * backscatter
    return I

def render_fog(rgb, d):
    J = rgb
    transmittance = torch.exp(-1.0 * FOG_BETA_B * d)

    atmospheric = estimate_atmospheric_light(rgb).view(3, 1, 1)

    I = J * transmittance + atmospheric * (1 - transmittance).expand((3, *list(transmittance.shape[1:])))
    return I

def dark_channel_estimate(rgb):
    patch_size = 41
    padding =  patch_size // 2
    rgb_max = torch.nn.MaxPool3d(
        kernel_size=(3, patch_size, patch_size),
        stride=1,
        padding=(0, padding, padding)
    )
    if len(rgb.size()) == 3:
        dcp = torch.abs(rgb_max(-rgb.unsqueeze(0)))
    else:
        dcp = torch.abs(rgb_max(-rgb))
    return dcp.squeeze()

def estimate_atmospheric_light(rgb):
    dcp = dark_channel_estimate(rgb)

    flat_dcp = torch.flatten(dcp)
    flat_r = torch.flatten(rgb[0])
    flat_g = torch.flatten(rgb[1])
    flat_b = torch.flatten(rgb[2])

    k = math.ceil(0.001 * len(flat_dcp))
    vals, idxs = torch.topk(flat_dcp, k)

    median_val = torch.median(vals)
    median_idxs = torch.where(vals == median_val)
    color_idxs = idxs[median_idxs]

    atmospheric_light = torch.stack([torch.mean(flat_r[color_idxs]), torch.mean(flat_g[color_idxs]), torch.mean(flat_b[color_idxs])])

    return atmospheric_light

def render_set(model_path, name, iteration, views, gaussians, pipeline, render_background,
               do_seathru: bool = False,
               add_water: bool = False,
               add_fog: bool = False,
               learned_bg = None,
               bs_model = None,
               at_model = None,
               save_as_jpeg: bool = False):
    if do_seathru:
        assert bs_model is not None
        assert at_model is not None
        no_water_dir = model_path / name / "no_water"
        backscatter_dir = model_path / name / "backscatter"
        attenuation_dir = model_path / name / "attenuation"
        with_water_dir = model_path / name / "with_water"
        makedirs(no_water_dir, exist_ok=True)
        makedirs(backscatter_dir, exist_ok=True)
        makedirs(attenuation_dir, exist_ok=True)
        makedirs(with_water_dir, exist_ok=True)

    if learned_bg is not None:
        bg_dir = model_path / name / "bg"
        makedirs(bg_dir, exist_ok=True)

    if add_water:
        water_dir = model_path / name / "add_water"
        makedirs(water_dir, exist_ok=True)

    if add_fog:
        fog_dir = model_path / name / "add_fog"
        makedirs(fog_dir, exist_ok=True)

    render_dir = model_path / name / "render"
    depth_dir = model_path / name / "depth"
    makedirs(render_dir, exist_ok=True)
    makedirs(depth_dir, exist_ok=True)

    timings_3dgs = []
    timings_seathru = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        render_pkg = render(view, gaussians, pipeline, render_background)
        rendered_image, image_alpha = render_pkg["render"], render_pkg["alpha"]
        render_depth_pkg = render_depth(view, gaussians, pipeline, render_background)
        depth_image = render_depth_pkg["render"][0].unsqueeze(0)
        depth_image = depth_image / image_alpha
        end_3dgs = time.time()
        if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
            # print(f"nans/infs in depth image")
            valid_depth_vals = depth_image[torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))]
            if len(valid_depth_vals) == 0:
                print(f"[training] everything is nan {view.image_name}")
                not_nan_max = 100.0
            else:
                not_nan_max = torch.max(valid_depth_vals).item()
            depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
        if depth_image.min() != depth_image.max():
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        else:
            depth_image = depth_image = depth_image / depth_image.max()

        # normalized_depth_image = depth_image / depth_image.max()

        if learned_bg is not None:
            bg_image = torch.sigmoid(learned_bg).reshape(3, 1, 1) * (1 - image_alpha)
            image = rendered_image + bg_image
            torchvision.utils.save_image(bg_image, bg_dir / f"{view.image_name}.png")
        else:
            image = rendered_image

        if add_water:
            added_uw_image = render_uw(view.original_image, depth_image)
            torchvision.utils.save_image(added_uw_image, water_dir / f"{view.image_name}.JPG")

        if add_fog:
            added_fog_image = render_fog(view.original_image, depth_image)
            torchvision.utils.save_image(added_fog_image, fog_dir / f"{view.image_name}.JPG")

        if do_seathru:
            image_batch = torch.unsqueeze(image, dim=0)
            depth_image_batch = torch.unsqueeze(depth_image, dim=0)

            attenuation_map_batch = at_model(depth_image_batch)
            backscatter_batch = bs_model(depth_image_batch)
            direct_batch = image_batch * attenuation_map_batch
            underwater_image_batch = torch.clamp(direct_batch + backscatter_batch, 0.0, 1.0)
            end_seathru = time.time()
            timings_seathru.append(end_seathru - start_time)

            torchvision.utils.save_image(backscatter_batch.squeeze(), backscatter_dir / f"{view.image_name}.png")
            torchvision.utils.save_image(attenuation_map_batch.squeeze(), attenuation_dir / f"{view.image_name}.png")
            if save_as_jpeg:
                torchvision.utils.save_image(underwater_image_batch.squeeze(), with_water_dir / f"{view.image_name}.JPG")
            else:
                torchvision.utils.save_image(underwater_image_batch.squeeze(), with_water_dir / f"{view.image_name}.png")

        if save_as_jpeg:
            torchvision.utils.save_image(rendered_image, render_dir / f"{view.image_name}.JPG")
        else:
            torchvision.utils.save_image(rendered_image, render_dir / f"{view.image_name}.png")
        plt.imsave(depth_dir / f"{view.image_name}.png", depth_image.detach().cpu().numpy().squeeze())
        # torchvision.utils.save_image(normalized_depth_image, depth_dir / f"{view.image_name}.png")

        timings_3dgs.append(end_3dgs - start_time)

    if do_seathru:
        print(f"[Seathru] Average time for {len(views)} images: {np.mean(timings_seathru)}")
    print(f"[3DGS] Average time for {len(views)} images: {np.mean(timings_3dgs)}")

def render_sets(
        model_params : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
        do_seathru: bool,
        add_water: bool,
        add_fog: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # learned background
        learned_bg = None
        learned_bg_path =  f"{model_params.model_path}/bg_{scene.loaded_iter}.pth"
        # if os.path.exists(learned_bg_path):
        #     print(f"Loading background {learned_bg_path}")
        #     learned_bg = torch.load(learned_bg_path)

        # backscatter attenuation models
        attenuation_model_path =  f"{model_params.model_path}/attenuate_{scene.loaded_iter}.pth"
        backscatter_model_path =  f"{model_params.model_path}/backscatter_{scene.loaded_iter}.pth"
        bs_model = None
        at_model = None
        if do_seathru:
            print(f"Loading backscatter model {backscatter_model_path}")
            print(f"Loading attenuation model {attenuation_model_path}")
            assert os.path.exists(attenuation_model_path)
            assert os.path.exists(backscatter_model_path)
            at_model = AttenuateNetV3(scale=5.0)
            bs_model = BackscatterNetV2(use_residual=False, scale=5.0)
            at_model.load_state_dict(torch.load(attenuation_model_path))
            bs_model.load_state_dict(torch.load(backscatter_model_path))
            at_model.cuda()
            bs_model.cuda()
            at_model.eval()
            bs_model.eval()

        if not skip_train:
             render_set(Path(model_params.model_path), "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, do_seathru, add_water, add_fog, learned_bg, bs_model, at_model)

        if not skip_test:
             render_set(Path(model_params.model_path), "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, do_seathru, add_water, add_fog, learned_bg, bs_model, at_model)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser) # need to specify -s or -m and maybe --eval
    pipeline = PipelineParams(parser) # safe to ignore
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seathru", action="store_true")
    parser.add_argument("--add_water", action="store_true")
    parser.add_argument("--add_fog", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=0)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.seathru, args.add_water, args.add_fog)
