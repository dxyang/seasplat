# Script that renders underwater image given RGB+D and backscatter / attenuation parameters
import math
import os
from os import makedirs
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch
from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render, render_depth, homogenize_points
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from deepseecolor.models import (
    AttenuateNetV3,
    BackscatterNetV2,
)

from utils.graphics_utils import getProjectionMatrix

from scene.cameras import Camera, MiniCam

def render_set(model_path, name, iteration, views, gaussians, pipeline, render_background,
               do_seathru: bool = False,
               bs_model = None,
               at_model = None,
               render_path_override = None):
    if do_seathru:
        assert bs_model is not None
        assert at_model is not None
        backscatter_dir = model_path / name / "backscatter"
        attenuation_dir = model_path / name / "attenuation"
        backscatter_normed_dir = model_path / name / "backscatter_normed"
        attenuation_normed_dir = model_path / name / "attenuation_normed"
        with_water_dir = model_path / name / "with_water"
        makedirs(backscatter_dir, exist_ok=True)
        makedirs(attenuation_dir, exist_ok=True)
        makedirs(backscatter_normed_dir, exist_ok=True)
        makedirs(attenuation_normed_dir, exist_ok=True)
        makedirs(with_water_dir, exist_ok=True)

    render_dir = model_path / name / "render"
    depth_dir = model_path / name / "depth"
    depth_viridis_dir = model_path / name / "depth_viridis"
    alpha_dir = model_path / name / "alpha"
    makedirs(render_dir, exist_ok=True)
    makedirs(depth_dir, exist_ok=True)
    makedirs(depth_viridis_dir, exist_ok=True)
    makedirs(alpha_dir, exist_ok=True)

    just_viridis = False

    view = views[0]
    fov_x = view.FoVx
    fov_y = view.FoVy
    znear = view.znear
    zfar = view.zfar

    width = view.image_width
    height = view.image_height

    if render_path_override is None:
        return

    for frame_num, T_world_cam in enumerate(tqdm(render_path_override)):
        frame_num_str = f"{frame_num:04d}"
        T_world_cam = torch.Tensor(T_world_cam).cuda()
        T_cam_world = T_world_cam.inverse()

        world_view_transform = T_cam_world.T

        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y).transpose(0,1).cuda()

        full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)

        render_view = MiniCam(
            width=width,
            height=height,
            fovy=fov_y,
            fovx=fov_x,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform
        )

        render_pkg = render(render_view, gaussians, pipeline, render_background)
        rendered_image, image_alpha = render_pkg["render"], render_pkg["alpha"]

        render_depth_pkg = render_depth(render_view, gaussians, pipeline, render_background)
        depth_image = render_depth_pkg["render"][0].unsqueeze(0)
        depth_image = depth_image / image_alpha

        if torch.any(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image))):
            # print(f"nans/infs in depth image")
            valid_depth_vals = depth_image[torch.logical_not(torch.logical_or(torch.isnan(depth_image), torch.isinf(depth_image)))]
            if len(valid_depth_vals) == 0:
                print(f"[training] everything is nan")
                not_nan_max = 100.0
            else:
                not_nan_max = torch.max(valid_depth_vals).item()
            depth_image = torch.nan_to_num(depth_image, not_nan_max, not_nan_max)
        if depth_image.min() != depth_image.max():
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        else:
            depth_image = depth_image = depth_image / depth_image.max()

        # normalized_depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
        normalized_depth_image = depth_image / depth_image.max()

        if do_seathru:
            image_batch = torch.unsqueeze(rendered_image, dim=0)
            depth_image_batch = torch.unsqueeze(depth_image, dim=0)

            # estimate attenuation
            attenuation_map = at_model(depth_image_batch)
            direct = image_batch * attenuation_map
            attenuation_map_normalized = attenuation_map / torch.max(attenuation_map)

            # estimate backscatter
            backscatter = bs_model(depth_image_batch)
            backscatter_normalized = backscatter / torch.max(backscatter)

            # combined image
            underwater_image = torch.clamp(direct + backscatter, 0.0, 1.0)

            underwater_image = underwater_image.squeeze()
            if not just_viridis:
                torchvision.utils.save_image(underwater_image, with_water_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(attenuation_map, attenuation_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(attenuation_map_normalized, attenuation_normed_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(backscatter, backscatter_dir / f"render_{frame_num_str}.png")
                torchvision.utils.save_image(backscatter_normalized, backscatter_normed_dir / f"render_{frame_num_str}.png")
        if not just_viridis:
            torchvision.utils.save_image(rendered_image, render_dir / f"render_{frame_num_str}.png")
            torchvision.utils.save_image(normalized_depth_image, depth_dir / f"render_{frame_num_str}.png")
            torchvision.utils.save_image(image_alpha, alpha_dir / f"render_{frame_num_str}.png")
        plt.imsave(depth_viridis_dir / f"render_{frame_num_str}.png", normalized_depth_image.detach().cpu().numpy().squeeze())

def render_sets(
        model_params : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
        do_seathru: bool, render_path: str, out_folder_name: str,
):
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
            at_model = AttenuateNetV3()
            bs_model = BackscatterNetV2()
            at_model.load_state_dict(torch.load(attenuation_model_path))
            bs_model.load_state_dict(torch.load(backscatter_model_path))
            at_model.cuda()
            bs_model.cuda()
            at_model.eval()
            bs_model.eval()

        if render_path is not None:
            render_path_poses = np.load(render_path)
            if 'stn' in render_path:
                render_path_poses = render_path_poses @ np.diag([1, -1, -1, 1])

        render_set(Path(model_params.model_path), out_folder_name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, do_seathru, bs_model, at_model, render_path_poses)

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
    parser.add_argument("--render_path", default=None, type=str)
    parser.add_argument("--out_folder_name", default="icra", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=0)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.seathru, args.render_path, args.out_folder_name)
