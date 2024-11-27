from typing import List, Mapping, Optional, Text, Tuple, Union

import numpy as np

# Constants for generate_spiral_path():
NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]

def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def generate_spiral_path(poses: np.ndarray,
                         bounds: np.ndarray,
                         n_frames: int = 120,
                         n_rots: int = 2,
                         zrate: float = .5) -> np.ndarray:
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of conservative near and far bounds in disparity space.
  near_bound = bounds.min() * NEAR_STRETCH
  far_bound = bounds.max() * FAR_STRETCH

  # All cameras will point towards the world space point (0, 0, -focal).
  focal = 1 / (((1 - FOCUS_DISTANCE) / near_bound + FOCUS_DISTANCE / far_bound))

  # import pdb; pdb.set_trace()

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = average_pose(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses

# def generate_ellipse_path(poses: np.ndarray,
#                           n_frames: int = 120,
#                           const_speed: bool = True,
#                           z_variation: float = 0.,
#                           z_phase: float = 0.) -> np.ndarray:
#   """Generate an elliptical render path based on the given poses."""
#   # Calculate the focal point for the path (cameras point toward this).
#   center = focus_point_fn(poses)
#   # Path height sits at z=0 (in middle of zero-mean capture pattern).
#   offset = np.array([center[0], center[1], 0])

#   # Calculate scaling for ellipse axes based on input camera positions.
#   sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
#   # Use ellipse that is symmetric about the focal point in xy.
#   low = -sc + offset
#   high = sc + offset
#   # Optional height variation need not be symmetric
#   z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
#   z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

#   def get_positions(theta):
#     # Interpolate between bounds with trig functions to get ellipse in x-y.
#     # Optionally also interpolate in z to change camera height along path.
#     return np.stack([
#         low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
#         low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
#         z_variation * (z_low[2] + (z_high - z_low)[2] *
#                        (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
#     ], -1)

#   theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
#   positions = get_positions(theta)

#   if const_speed:
#     # Resample theta angles so that the velocity is closer to constant.
#     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
#     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
#     positions = get_positions(theta)

#   # Throw away duplicated last position.
#   positions = positions[:-1]

#   # Set path's up vector to axis closest to average of input pose up vectors.
#   avg_up = poses[:, :3, 1].mean(0)
#   avg_up = avg_up / np.linalg.norm(avg_up)
#   ind_up = np.argmax(np.abs(avg_up))
#   up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

#   return np.stack([viewmatrix(p - center, up, p) for p in positions])
