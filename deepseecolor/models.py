import torch
import torch.nn as nn

class BackscatterNet(nn.Module):
    def __init__(self, use_residual: bool = True):
        super().__init__()

        self.backscatter_conv = nn.Conv2d(1, 3, 1, bias=False)
        nn.init.uniform_(self.backscatter_conv.weight, 0, 5)

        self.use_residual = use_residual
        if use_residual:
            self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)
            nn.init.uniform_(self.residual_conv.weight, 0, 5)
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, depth, uw_image=None):
        beta_b_conv = self.relu(self.backscatter_conv(depth))
        Bc = self.B_inf * (1 - torch.exp(-beta_b_conv))
        if self.use_residual:
            beta_d_conv = self.relu(self.residual_conv(depth))
            Bc += self.J_prime * torch.exp(-beta_d_conv)
        backscatter = self.sigmoid(Bc)

        # if depth is zero'd out (i.e. bad estimate), do not use it for backscatter either
        backscatter_masked = backscatter * (depth > 0.).repeat(1, 3, 1, 1)

        # backwards compat with og code
        if uw_image is not None:
            direct = uw_image - backscatter_masked
            return direct, backscatter_masked
        else:
            return backscatter_masked

class BackscatterNetV2(nn.Module):
    '''
    backscatter = B_inf * (1 - exp(- a * z)) + J_prime * exp(- b * z)

    main difference with bsv1 is B_inf and J_prime go through a sigmoid
    which might make them more easily learnable (and keep them constrained to [0, 1])
    '''
    def __init__(self, use_residual: bool = False, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = False):
        super().__init__()

        self.scale = scale

        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))

        self.use_residual = use_residual
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))

        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()

        self.do_sigmoid = do_sigmoid

        print(f"Using backscatterv2 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        if self.do_sigmoid:
            beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.backscatter_conv_params)))
        else:
            # beta_b_conv = self.relu(torch.nn.functional.conv2d(depth, self.backscatter_conv_params))
            beta_b_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.backscatter_conv_params), 0.0)

        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        if self.use_residual:
            if self.do_sigmoid:
                beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.residual_conv_params)))
            else:
                # beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.residual_conv_params))
                beta_d_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.residual_conv_params), 0.0)
            Bc += torch.sigmoid(self.J_prime) * torch.exp(-beta_d_conv)
        backscatter = Bc

        # if depth is zero'd out (i.e. bad estimate), do not use it for backscatter either
        # backscatter_masked = backscatter * (depth > 0.).repeat(1, 3, 1, 1)

        # backwards compat with og code
        return backscatter

    def forward_rgb(self, rgb):
        from render_uw import estimate_atmospheric_light

        atmospheric_colors = []
        for rgb_image in rgb:
            atmospheric_colors.append(estimate_atmospheric_light(rgb_image.detach()))

        atmospheric_color = torch.mean(torch.stack(atmospheric_colors), dim=0)

        return self.l2(atmospheric_color.squeeze(), self.B_inf.squeeze())

class AttenuateNet(nn.Module):
    '''
    beta_d(z) = a  * exp(-b * z) + c * exp(-d * z)
    a, c: (0, inf)
    b, d: (0, inf)

    attenuation_map = exp(-beta_d * z)
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False):
        super().__init__()
        self.attenuation_conv_params = nn.Parameter(torch.rand(6, 1, 1, 1)) # b, d from SeaThru
        self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1)) # a, c from SeaThru

        self.relu = nn.ReLU()

        self.scale = scale
        self.do_sigmoid = do_sigmoid
        print(f"Using attenuatenetv1 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        # true_color: J
        # generates attenuation coefficients, a_c(z) (DSC eqn 12)
        if self.do_sigmoid:
            attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))))
            beta_d = torch.stack(tuple(
                torch.sum(attn_conv[:, i:i + 2, :, :] * torch.sigmoid(self.attenuation_coef[i:i + 2]), dim=1) for i in
                range(0, 6, 2)), dim=1)
        else:
            # attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.attenuation_conv_params)))
            attn_conv = torch.exp(-torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0))
            beta_d = torch.stack(tuple(
                torch.sum(attn_conv[:, i:i + 2, :, :] * torch.clamp(self.attenuation_coef[i:i + 2]), dim=1) for i in
                range(0, 6, 2)), dim=1)

        # generate attenuation map A_c(z) = GED(z * a_c(z)) (DSC eqn 13)
        # attenuation_map = torch.exp(-1.0 * torch.relu(beta_d) * depth)
        attenuation_map = torch.exp(-1.0 * torch.clamp(beta_d, 0.0) * depth)

        # if depth is zero'd out (i.e. bad estimate), do not use it for attenuation either
        # attenuation_map_masked = attenuation_map * ((depth == 0.) / attenuation_map + (depth > 0.))
        # nanmask = torch.isnan(attenuation_map_masked)
        # if torch.any(nanmask):
        #     print("Warning! NaN values in J")
        #     attenuation_map_masked[nanmask] = 0

        return attenuation_map

class AttenuateNetV2(nn.Module):
    '''
    beta_d(z) = a  * exp(-b * z) + c * exp(-d * z)
    a, c: (0, inf)
    b, d: (0, inf)

    attenuation_map = exp(-beta_d * z)

    this version drops c, d terms
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False):
        super().__init__()
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        self.attenuation_coef = nn.Parameter(torch.rand(3, 1, 1))
        self.relu = nn.ReLU()

        self.scale = scale
        self.do_sigmoid = do_sigmoid
        print(f"Using attenuatenetv2 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        # true_color: J

        # generates attenuation coefficients, a_c(z) (DSC eqn 12)
        if self.do_sigmoid:
            attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params))))
            beta_d = torch.concatenate(tuple(
                torch.sum(attn_conv[:, i:i+1, :, :] * torch.sigmoid(self.attenuation_coef[i]), dim=1, keepdim=True) for i in
                range(3)), dim=1)
        else:
            # attn_conv = torch.exp(-self.relu(torch.nn.functional.conv2d(depth, self.attenuation_conv_params)))
            attn_conv = torch.exp(-torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0))
            beta_d = torch.concatenate(tuple(
                torch.sum(attn_conv[:, i:i+1, :, :] * torch.clamp(self.attenuation_coef[i], 0.0), dim=1, keepdim=True) for i in
                range(3)), dim=1)

        # generate attenuation map A_c(z) = GED(z * a_c(z)) (DSC eqn 13)
        # attenuation_map = torch.exp(-1.0 * torch.relu(beta_d) * depth)
        attenuation_map = torch.exp(-1.0 * torch.clamp(beta_d, 0.0) * depth)

        # if depth is zero'd out (i.e. bad estimate), do not use it for attenuation either
        # attenuation_map_masked = attenuation_map * ((depth == 0.) / attenuation_map + (depth > 0.))
        # nanmask = torch.isnan(attenuation_map_masked)
        # if torch.any(nanmask):
        #     print("Warning! NaN values in J")
        #     attenuation_map_masked[nanmask] = 0

        return attenuation_map

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class AttenuateNetV3(nn.Module):
    '''
    attenuation_map = exp(-beta_d * z)

    this one
    * does not try to scale the parameters (so here, they lie between 0 and 1 i.e. sigmoid output)
    * does not have any max attenuation
    '''
    def __init__(self, scale: float = 1.0, do_sigmoid: bool = False, init_vals: bool = True):
        super().__init__()

        # self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.3, 1.2, 0.1]).reshape(3, 1, 1, 1)) #nn.Parameter(torch.rand(3, 1, 1, 1))
        # self.attenuation_conv_params = nn.Parameter(inverse_sigmoid(torch.Tensor([0.8, 0.8, 0.2])).reshape(3, 1, 1, 1))
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
        self.attenuation_coef = None
        self.scale = scale
        self.do_sigmoid = do_sigmoid

        self.relu = nn.ReLU()
        print(f"Using attenuatenetv3 with scale: {self.scale}, sigmoid: {self.do_sigmoid}")

    def forward(self, depth):
        if self.do_sigmoid:
            beta_d_conv = self.relu(torch.nn.functional.conv2d(depth, self.scale * torch.sigmoid(self.attenuation_conv_params)))
        else:
            beta_d_conv = torch.clamp(torch.nn.functional.conv2d(depth, self.attenuation_conv_params), 0.0)

        attenuation_map = torch.exp(-beta_d_conv)

        # if depth is zero'd out (i.e. bad estimate), do not use it for attenuation either
        # attenuation_map_masked = attenuation_map * ((depth == 0.) / attenuation_map + (depth > 0.))
        # nanmask = torch.isnan(attenuation_map_masked)
        # if torch.any(nanmask):
        #     print("Warning! NaN values in J")
        #     attenuation_map_masked[nanmask] = 0

        return attenuation_map

