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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

WARNED_ONCE = False

def read_image(renders_dir, gt_dir, fname):
    render = Image.open(renders_dir / fname)
    gt = Image.open(gt_dir / fname)

    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
    gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

    image_name = fname

    render_h, render_w = render.size()[2], render.size()[3]
    gt_h, gt_w = gt.size()[2], gt.size()[3]
    if gt_h != render_h or gt_w != render_w:
        global WARNED_ONCE
        if not WARNED_ONCE:
            print(f"resizing gt h,w: {gt_h},{gt_w} to {render_h},{render_w}")
            WARNED_ONCE = True
        resized_gt = tf.resize(gt, (render_h, render_w))
        return render, resized_gt, image_name
    else:
        return render, gt, image_name

def readImages(renders_dir, gt_dir, fname_list=None):
    renders = []
    gts = []
    image_names = []

    if fname_list is None:
        fnames = os.listdir(renders_dir)
    else:
        fnames = fname_list

    for fname in tqdm(fnames, desc="Reading image batch"):
        img_name = fname.split('.')[0]
        if os.path.exists(gt_dir / (img_name + ".png")):
            gt = Image.open(gt_dir / (img_name + ".png"))
        else:
            gt = Image.open(gt_dir / (img_name + ".JPG"))

        render = Image.open(renders_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

    render_h, render_w = renders[0].size()[2], renders[0].size()[3]
    gt_h, gt_w = gts[0].size()[2], gts[0].size()[3]
    if gt_h != render_h or gt_w != render_w:
        print(f"resizing gt h,w: {gt_h},{gt_w} to {render_h},{render_w}")
        resized_gts = [tf.resize(gt, (render_h, render_w)) for gt in gts]
        return renders, resized_gts, image_names
    else:
        return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in ["skip"]:
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                ssims = []
                psnrs = []
                lpipss = []

                # method_dir = test_dir / method
                gt_dir = test_dir/ "gt"
                renders_dir = test_dir / "render" # renders
                fname_list = os.listdir(renders_dir)
                chunk_size = 128
                for i in tqdm(range(0, len(fname_list), chunk_size)):
                    upper =  min(i + chunk_size, len(fname_list))
                    fnames = fname_list[i : upper]

                    renders, gts, image_names = readImages(renders_dir, gt_dir, fnames)

                    for idx in range(len(renders)):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM ^: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR ^: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS v: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

def evaluate_single(model_path, comparison_dir="with_water", skip_train=False):
    print("Scene:", model_path)
    results = {}

    dirs = []
    print_strs = []
    if 'seathru_NeRF' in model_path:
        test_dir = Path(model_path)
        dirs.append(test_dir)
        dset = str(Path(model_path).parent.parent.stem).split('_')[0]

        if dset in ["Curacao", "JapaneseGradens-RedSea", "Panama", "IUI3-RedSea"]:
            gt_dir = Path(model_path).parent.parent.parent.parent.parent / "data" / dset / "images_wb"
        elif dset in ["GardenSimFog", "GardenSimWater"]:
            if comparison_dir == "with_water":
                if dset == "GardenSimFog":
                    gt_dir = Path("/srv/warplab/dxy/mipnerf360_dataset/garden/experiments/08282024/vanilla/train/add_fog")
                else:
                    gt_dir = Path("/srv/warplab/dxy/mipnerf360_dataset/garden/experiments/08282024/vanilla/train/add_water")
            elif comparison_dir == "render":
                gt_dir = Path("/srv/warplab/dxy/mipnerf360_dataset/garden/images_4")
            else:
                assert False
        elif dset in ["saltpond"]:
            gt_dir = Path("/srv/warplab/dxy/dsc_datasets/2022-08-01-12-56-48-salt-pond-dataset/colmap_1x/images")
        else:
            assert False

        # gt_dir = Path("/home/user/localdata/SeathruNeRF_dataset/Curasao/images")
        print_strs.append('Test')
    elif 'mipnerf' in model_path:
        test_dir = Path(model_path) / "test" / comparison_dir
        train_dir = Path(model_path) / "train" / comparison_dir

        if comparison_dir == "with_water":
            if 'simfog' in model_path:
                gt_dir = Path(model_path).parent.parent / "08282024/vanilla/test/add_fog"
            elif 'simwater' in model_path:
                gt_dir = Path(model_path).parent.parent / "08282024/vanilla/test/add_water"
            else:
                assert False, "Unknown mipnerf model"
        elif comparison_dir == "render":
            gt_dir = Path(model_path).parent.parent.parent / "images_4"
        else:
            assert False

        if os.path.exists(test_dir):
            dirs.append(test_dir)
            print_strs.append("Test")
        if not skip_train:
            if os.path.exists(train_dir):
                dirs.append(train_dir)
                print_strs.append("Train")
    else:
        test_dir = Path(model_path) / "test" / comparison_dir
        train_dir = Path(model_path) / "train" / comparison_dir
        gt_dir = Path(model_path).parent.parent.parent / "images"

        if os.path.exists(test_dir):
            dirs.append(test_dir)
            print_strs.append("Test")
        if not skip_train:
            if os.path.exists(train_dir):
                dirs.append(train_dir)
                print_strs.append("Train")

    print(print_strs)
    for eval_idx, img_dir in enumerate(dirs):
        print(f"Comparing {img_dir} to {gt_dir}")
        ssims = []
        psnrs = []
        lpipss = []

        fname_list = os.listdir(img_dir)
        chunk_size = 64
        for i in tqdm(range(0, len(fname_list), chunk_size), desc="Overall image batches"):
            upper =  min(i + chunk_size, len(fname_list))
            fnames = fname_list[i : upper]

            renders, gts, image_names = readImages(img_dir, gt_dir, fnames)

            for idx in tqdm(range(len(renders)), desc="Calculating metrics"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        print(f"-----------{print_strs[eval_idx]}")
        print("  SSIM ^: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR ^: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS v: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        key = 'Test' if eval_idx == 0 else 'Train'
        results[key] = {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }

    if comparison_dir == "with_water":
        results_file = Path(model_path) / "eval_metrics_with_water.json"
    elif comparison_dir == "render":
        results_file = Path(model_path) / "eval_metrics_render.json"
    with open(str(results_file), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--comparison_dir', required=False, type=str, default="with_water")
    parser.add_argument('--skip_train', required=False, action='store_true')
    args = parser.parse_args()
    if len(args.model_paths) == 1:
        evaluate_single(args.model_paths[0], args.comparison_dir, args.skip_train)
    else:
        evaluate(args.model_paths)
