import math
from typing import List, Literal, Optional, Tuple

import numpy as np
import splines.quaternion
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

import splines

from utils.colmap_utils import qvec2rotmat, rotmat2qvec

def tf_matrix(rotation, translation):
    mat1 = np.c_[rotation, translation]
    mat2 = np.array([0, 0, 0, 1]).reshape((1, 4))
    trans_matrix = np.vstack([mat1, mat2])
    return trans_matrix


def get_interpolated_poses_spline(
    poses: Float[Tensor, "num_poses 3 4"],
    loop: bool = True,
    num_frames: int = 300,
) -> Tuple[Float[Tensor, "num_poses 3 4"], Float[Tensor, "num_poses 3 3"]]:
    """Return interpolated poses for many camera poses.

    Args:
        poses: list of camera poses

    Returns:
        spline object
    """
    tension = 0.5
    num_cameras = poses.shape[0]
    max_camera_idx = num_cameras if loop else num_cameras - 1

    orientation_spline = splines.quaternion.KochanekBartels(
        [
            splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(rotmat2qvec(pose[:3, :3]), shift=-1))
            for pose in poses
        ],
        tcb=(tension, 0.0, 0.0),
        endconditions="closed" if loop else "natural",
    )
    position_spline = splines.KochanekBartels(
        [pose[:3, 3] for pose in poses],
        tcb=(tension, 0.0, 0.0),
        endconditions="closed" if loop else "natural",
    )

    T_world_cams = []

    for i in range(num_frames):
        normalized_t = i / num_frames * max_camera_idx

        quat = orientation_spline.evaluate(normalized_t)

        rot = qvec2rotmat(quat.wxyz)
        translation = position_spline.evaluate(normalized_t)
        T_world_cam = tf_matrix(rot, translation)

        T_world_cams.append(T_world_cam)

    return T_world_cams