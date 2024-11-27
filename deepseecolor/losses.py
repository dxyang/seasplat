import math

from kornia.color import rgb_to_lab
import torch
import torch.nn as nn

class AttenuateLoss(nn.Module):
    def __init__(self, override=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5
        self.override=override

    def forward(self, direct, J):
        if self.override is not None:
            return self.override

        # dxy: this doesn't seem to do much (just pushes to be between 0-1)
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        return intensity_loss + spatial_variation_loss + saturation_loss

class BackscatterLoss(nn.Module):
    def __init__(self, override=None, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio
        self.override=override

    def forward(self, direct, depth=None):
        if self.override is not None:
            return self.override

        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        if (neg > 0):
            print(f"negative values inducing loss: {neg}")
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss

class DarkChannelPriorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 41
        self.padding = self.patch_size // 2
        self.rgb_max = nn.MaxPool3d(
            kernel_size=(3, self.patch_size, self.patch_size),
            stride=1,
            padding=(0, self.padding, self.padding))
        self.l1 = nn.L1Loss()

    def forward(self, rgb, d=None):
        '''
        Following the procedure from:
        (1) Single image haze removal using dark channel prior (2009)

        * Run a 15x15 kernel across the whole image and take the minimum
        between all color channels

        Assumes rgb is in BCHW
        '''
        dcp = torch.abs(self.rgb_max(-rgb))
        loss = self.l1(dcp, torch.zeros_like(dcp))
        return loss, dcp

class DarkChannelPriorLossV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.num_depth_bins = 10
        self.pct = 0.01

    def rgb2gray(self, rgb):
        # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
        # let's just use opencv standard values
        bs, _, h, w = rgb.size()
        gray = torch.zeros((bs, 1, h, w)).to(rgb.device)
        gray[:] = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        return gray

    def forward(self, rgb, d):
        '''
        Following the procedure from:
        (1) Single image haze removal using dark channel prior (2009)
        (2) A Method for Removing Water From Underwater Images (2019)

        * Bin up depth
        * Estimate darkest 1% color intensity value in each bin
        * Push these pixels to be black

        Assumes rgb is in BCHW
        '''
        d_min, d_max = d.min(), d.max()
        d_range = d_max - d_min

        loss = 0.0

        gray = self.rgb2gray(rgb)
        dcp = torch.zeros_like(gray)

        for i in range(self.num_depth_bins):
            # yes i'm creating a copy of the gray image every time
            gray = self.rgb2gray(rgb)

            lo = d_min + i * d_range / self.num_depth_bins
            hi = d_min + (i + 1) * d_range / self.num_depth_bins

            d_mask = torch.logical_and(d >= lo, d < hi)
            num_vals = torch.sum(d_mask)

            if num_vals == 0:
                print(f"empty depth bin")
                continue

            gray_bin = gray[d_mask]
            sorted_gray_bin, _ = torch.sort(gray_bin)

            try:
                gray_threshold = sorted_gray_bin[math.ceil(num_vals * self.pct) - 1]
            except:
                import pdb; pdb.set_trace()

            lt_gray_thresh = gray <= gray_threshold

            dcp_mask = torch.logical_and(d_mask, lt_gray_thresh)
            dcp[dcp_mask] = gray[dcp_mask]

        loss = self.l1(dcp, torch.zeros_like(dcp))

        return loss, dcp

class DarkChannelPriorLossV3(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio

    def forward(self, direct, depth=None):
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        # if (neg > 0):
        #     print(f"negative values inducing loss: {neg}")
        bs_loss = self.cost_ratio * neg + pos
        return bs_loss, torch.zeros_like(direct)

class DeattenuateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self, direct, J):
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        return saturation_loss + intensity_loss + spatial_variation_loss

class GrayWorldPriorLoss(nn.Module):
    def __init__(self, target_intensity=0.5):
        super().__init__()
        self.target_intensity = target_intensity

    def forward(self, J):
        '''
        J: bchw or bc(flat)
        '''
        if len(J.size()) == 4:
            channel_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
        elif len(J.size()) == 3:
            channel_intensities = torch.mean(J, dim=[-1])
        else:
            assert False
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        if torch.any(torch.isnan(intensity_loss)):
            print("NaN intensity loss!")
        return intensity_loss

class RgbSpatialVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, J, direct):
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(spatial_variation_loss)):
            print("NaN spatial variation loss!")
        return spatial_variation_loss

class RgbSaturationLoss(nn.Module):
    '''
    useful for keeping tensor values corresponding to an image within a range
    e.g. to penalize for being outside of [0, 1] set saturation val to 1.0
    '''
    def __init__(self, saturation_val: float):
        super().__init__()
        self.relu = nn.ReLU()
        self.saturation_val = saturation_val

    def forward(self, rgb):
        saturation_loss = (self.relu(-rgb) + self.relu(rgb - self.saturation_val)).square().mean()
        if torch.any(torch.isnan(saturation_loss)):
            print("NaN saturation loss!")
        return saturation_loss

class AlphaBackgroundLoss(nn.Module):
    '''
    penalize alpha mask for values close (by L2 norm RGB) to the background color

    should probably detach rgb and background when they come into this
    '''
    def __init__(self, use_kornia: bool = False):
        super().__init__()
        self.use_kornia = use_kornia
        if use_kornia:
            self.range = math.sqrt(100 * 100 + 255 * 255 + 255 * 255)
            self.threshold = 50
        else:
            self.range = math.sqrt(3)
            self.threshold = 0.2 * math.sqrt(3)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, rgb, background, alpha):
        if self.use_kornia:
            lab_background = rgb_to_lab(background.reshape(3, 1, 1))
            lab_image = rgb_to_lab(rgb)
            diff = lab_image - lab_background
            dist = torch.linalg.vector_norm(diff, dim=0)
        else:
            if len(rgb.size()) == 2:
                diff = rgb - background
                dist = torch.linalg.vector_norm(diff, dim=1)
            else:
                diff = rgb - background.reshape(3, 1, 1)
                dist = torch.linalg.vector_norm(diff, dim=0)

        # other approach
        new_approach = False
        if new_approach:
            clamped_diff = torch.max(dist - self.threshold, torch.Tensor([0.0]).cuda())
            if self.use_kornia:
                mask = torch.exp(-clamped_diff / 10)
            else:
                mask = torch.exp(-clamped_diff / 0.05)
            masked_alpha = alpha * mask
            try:
                if torch.sum(mask) == 0:
                    loss = torch.Tensor([0.0]).squeeze().cuda()
                else:
                    loss = self.mse(masked_alpha, torch.zeros_like(masked_alpha))
            except:
                import pdb; pdb.set_trace()
        else:
            mask = dist < self.threshold
            if len(alpha.size()) == 1:
                masked_alpha = alpha[mask]
            else:
                masked_alpha = alpha[:, mask]
            try:
                if torch.sum(mask) == 0:
                    loss = torch.Tensor([0.0]).squeeze().cuda()
                else:
                    loss = self.l1(masked_alpha, torch.zeros_like(masked_alpha))
            except:
                import pdb; pdb.set_trace()
        return loss


def mixture_of_laplacians_loss(x):
    lp1 = torch.exp(-torch.abs(x)/0.1)
    lp2 = torch.exp(-torch.abs(1-x)/0.1)
    return -torch.mean(torch.log(lp1 + lp2))

