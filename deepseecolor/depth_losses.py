import math

import torch
import torch.nn as nn

class SmoothDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb, depth):
        """
        Args:
            rgb: [batch, 3, H, W]
            depth: [batch, 1, H, W]
        """
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        rgb_dx = torch.mean(rgb.diff(dim=-1), axis=-3, keepdim=True)
        rgb_dy = torch.mean(rgb.diff(dim=-2), axis=-3, keepdim=True)

        depth_dx *= torch.exp(-rgb_dx)
        depth_dy *= torch.exp(-rgb_dy)

        return torch.abs(depth_dx).mean() + torch.abs(depth_dy).mean()

        # grad_depth_x = torch.abs(depth[..., :, :, :-1] - depth[..., :, :, 1:])
        # grad_depth_y = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :])
        # grad_img_x = torch.mean(torch.abs(rgb[..., :, :, :-1] - rgb[..., :, :, 1:]), 1, keepdim=True)
        # grad_img_y = torch.mean(torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), 1, keepdim=True)

        # rgb = rgb.permute((0, 2, 3, 1))
        # depth = depth.permute((0, 2, 3, 1))
        # grad_depth_x = torch.abs(depth[..., :, :-1, :] - depth[..., :, 1:, :]) BH(W-1)1
        # grad_depth_y = torch.abs(depth[..., :-1, :, :] - depth[..., 1:, :, :]) B(H-1)W1
        # grad_img_x = torch.mean(torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True) # BH(W-1)1
        # grad_img_y = torch.mean(torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True) # B(H-1)W1
        # grad_depth_x *= torch.exp(-grad_img_x)
        # grad_depth_y *= torch.exp(-grad_img_y)


