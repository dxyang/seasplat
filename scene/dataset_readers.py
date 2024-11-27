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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_image_path: str
    depth_image: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def in_bounds(xyz, scene_bounds_xxyyzz):
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = scene_bounds_xxyyzz
    x, y, z = xyz
    if x < x_lo or x > x_hi:
        return False
    if y < y_lo or y > y_hi:
        return False
    if z < z_lo or z > z_hi:
        return False
    return True


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, scale_xyz=1.0, scene_bounds_xxyyzz=None):
    cam_infos = []
    num_skipped = 0
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{} with {} skipped".format(idx+1, len(cam_extrinsics), num_skipped))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec) * scale_xyz

        translation_world = -R @ T
        if scene_bounds_xxyyzz is not None:
            if not in_bounds(translation_world, scene_bounds_xxyyzz):
                num_skipped += 1
                continue

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_image_path = Path(images_folder) / "depth" / f"{os.path.basename(extr.name)}.npy"
        if os.path.exists(depth_image_path):
            depth_image = Image.fromarray(np.load(depth_image_path, allow_pickle=True))
        else:
            depth_image = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_image_path=depth_image_path, depth_image=depth_image)
        cam_infos.append(cam_info)

    print(f"{len(cam_infos)} valid cameras out of {len(cam_extrinsics)}, {num_skipped} removed")
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, scale_pcd=1.0, scene_bounds_xxyyzz=None):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions *= scale_pcd
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    if scene_bounds_xxyyzz is not None:
        num_points_original = positions.shape[0]
        valid_x = np.logical_and(positions[:, 0] > scene_bounds_xxyyzz[0], positions[:, 0] < scene_bounds_xxyyzz[1])
        valid_y = np.logical_and(positions[:, 1] > scene_bounds_xxyyzz[2], positions[:, 1] < scene_bounds_xxyyzz[3])
        valid_z = np.logical_and(positions[:, 2] > scene_bounds_xxyyzz[4], positions[:, 2] < scene_bounds_xxyyzz[5])
        valid = np.logical_and(np.logical_and(valid_x, valid_y), valid_z)

        positions = positions[valid]
        colors = colors[valid]
        normals = normals[valid]
        num_points_new = positions.shape[0]
        print(f"num points from {num_points_original} to {num_points_new}")

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

SEATHRU_NERF_DATASET = {
    "Curasao": {
        "train": [
            'MTN_1289', 'MTN_1290', 'MTN_1291', 'MTN_1292', 'MTN_1293',
            'MTN_1294', 'MTN_1295', 'MTN_1297', 'MTN_1298', 'MTN_1299',
            'MTN_1300', 'MTN_1301', 'MTN_1302', 'MTN_1303', 'MTN_1305',
            'MTN_1306', 'MTN_1307', 'MTN_1308'
            ],
        "test": [
            'MTN_1288', 'MTN_1296', 'MTN_1304'
        ],
    }, "Panama": {
        "train": [
            'MTN_1532', 'MTN_1533', 'MTN_1534', 'MTN_1535', 'MTN_1536',
            'MTN_1537', 'MTN_1538', 'MTN_1540', 'MTN_1541', 'MTN_1542',
            'MTN_1543', 'MTN_1544', 'MTN_1545', 'MTN_1546', 'MTN_1548'
        ],
        "test": [
            'MTN_1529', 'MTN_1539', 'MTN_1547'
        ],
    },
    "JapaneseGradens-RedSea": {
        "train": [
            'MTN_1091', 'MTN_1092', 'MTN_1093', 'MTN_1094', 'MTN_1095',
            'MTN_1096', 'MTN_1097', 'MTN_1099', 'MTN_1100', 'MTN_1101',
            'MTN_1102', 'MTN_1103', 'MTN_1104', 'MTN_1105', 'MTN_1107',
            'MTN_1108', 'MTN_1109'
        ],
        "test": [
            'MTN_1090', 'MTN_1098', 'MTN_1106'
        ],
    },
    "IUI3-RedSea": {
        "train": [
            'MTN_5895', 'MTN_5896', 'MTN_5898', 'MTN_5899', 'MTN_5900',
            'MTN_5901', 'MTN_5902', 'MTN_5904', 'MTN_5905', 'MTN_5906',
            'MTN_5907', 'MTN_5908', 'MTN_5909', 'MTN_5910', 'MTN_5912',
            'MTN_5913', 'MTN_5914', 'MTN_5915', 'MTN_5916', 'MTN_5917',
            'MTN_5927', 'MTN_5929', 'MTN_5930', 'MTN_5931', 'MTN_5933'
        ],
        "test": [
            'MTN_5894', 'MTN_5903',  'MTN_5911', 'MTN_5928'
        ],
    },
}

def readColmapSceneInfo(path, images, eval, llffhold=8, subsample=0, skip_first=0, start_cam=-1, end_cam=-1, rescale_units=1, scene_bounds_xxyyzz=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), scale_xyz=rescale_units, scene_bounds_xxyyzz=scene_bounds_xxyyzz)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    dataset = None
    if Path(path).stem in SEATHRU_NERF_DATASET.keys():
        dataset = Path(path).stem

    if skip_first != 0:
        original_size = len(cam_infos)
        cam_infos = cam_infos[skip_first:]
        new_size = len(cam_infos)
        print(f"[Warning] Skipping first {skip_first} cameras! Dataset size from {original_size} to {new_size}")

    if start_cam != -1 and end_cam != -1:
        original_size = len(cam_infos)
        cam_infos = cam_infos[start_cam:end_cam]
        new_size = len(cam_infos)
        print(f"[Warning] Using cams {start_cam} to {end_cam}! Dataset size from {original_size} to {new_size}")

    if subsample != 0:
        original_size = len(cam_infos)
        cam_infos = cam_infos[::subsample]
        new_size = len(cam_infos)
        print(f"[Warning] Subsampling dataset and taking every {subsample}th frame! Dataset size from {original_size} to {new_size}")

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        # for _, c in enumerate(cam_infos):
        #     if c.image_name in SEATHRU_NERF_DATASET[dataset]['train']:
        #         train_cam_infos.append(c)
        #     else:
        #         test_cam_infos.append(c)
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    print(f"{len(train_cam_infos)} train images and {len(test_cam_infos)} train images")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path, scale_pcd=rescale_units, scene_bounds_xxyyzz=scene_bounds_xxyyzz)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}