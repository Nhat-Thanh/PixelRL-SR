import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelRL_model(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)

        # pi network
        self.conv_5_pi = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_pi = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.W_xr = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_hr = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_xz = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_hz = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_xh = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.W_hh = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        # Should not apply softmax here because of vanishing values caused by log(softmax)
        # Use softmax when calculate prob, use log_softmax in MyEntropy instead of log(softmax)
        self.conv_7_pi = nn.Conv2d(64, n_actions, kernel_size=3, padding=1)

        # v network
        self.conv_5_v = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv_6_v = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv_7_v = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # reward map convolution
        self.conv_R = nn.Conv2d(1, 1, kernel_size=33, padding=16, bias=False)

    def pi_and_v(self, X_in):
        X = X_in[:, 0:3, :, :]
        X = F.relu(self.conv_1(X))
        X = F.relu(self.conv_2(X))
        X = F.relu(self.conv_3(X))
        X = F.relu(self.conv_4(X))

        # pi network
        X_t = F.relu(self.conv_5_pi(X))
        X_t = F.relu(self.conv_6_pi(X_t))

        # ConvGRU
        H_t1 = X_in[:, -64:, :, :]
        R_t = torch.sigmoid(self.W_xr(X_t) + self.W_hr(H_t1))
        Z_t = torch.sigmoid(self.W_xz(X_t) + self.W_hz(H_t1))
        H_tilde_t = torch.tanh(self.W_xh(X_t) + self.W_hh(R_t * H_t1))
        H_t = Z_t * H_t1 + (1 - Z_t) * H_tilde_t

        pi = self.conv_7_pi(H_t)

        # v network
        X_v = F.relu(self.conv_5_v(X))
        X_v = F.relu(self.conv_6_v(X_v))
        v = self.conv_7_v(X_v)

        return pi, v, H_t

    def conv_smooth(self, v):
        return self.conv_R(v)

    def choose_best_actions(self, state):
        pi, v, inner_state = self.pi_and_v(state)
        actions_prob = torch.softmax(pi, dim=1)
        actions = torch.argmax(actions_prob, dim=1)
        return actions, v, inner_state


# ======================================================================================
# SRCNN network 
# ======================================================================================

class SRCNN_model(nn.Module):
    def __init__(self) -> None:
        super(SRCNN_model, self).__init__()
        self.patch_extraction = nn.Conv2d(3, 64, kernel_size=9)
        self.nonlinear_map = nn.Conv2d(64, 32, kernel_size=5)
        self.recon = nn.Conv2d(32, 3, kernel_size=5)

    def forward(self, X_in):
        X = F.relu(self.patch_extraction(X_in))
        X = F.relu(self.nonlinear_map(X))
        X = self.recon(X)
        X_out = torch.clip(X, 0.0, 1.0)
        return X_out


# ======================================================================================
# FSRCNN network 
# ======================================================================================

class FSRCNN_model(nn.Module):
    def __init__(self, scale: int) -> None:
        super(FSRCNN_model, self).__init__()

        if scale not in [2, 3, 4]:
            raise ValueError("must be 2, 3 or 4")

        d = 56
        s = 12

        self.feature_extract = nn.Conv2d(3, d, kernel_size=5, padding=2)
        self.activation_1 = nn.PReLU(num_parameters=d)

        self.shrink = nn.Conv2d(d, s, kernel_size=1)
        self.activation_2 = nn.PReLU(num_parameters=s)
        
        # m = 4
        self.map_1 = nn.Conv2d(s, s, kernel_size=3, padding=1)
        self.map_2 = nn.Conv2d(s, s, kernel_size=3, padding=1)
        self.map_3 = nn.Conv2d(s, s, kernel_size=3, padding=1)
        self.map_4 = nn.Conv2d(s, s, kernel_size=3, padding=1)
        self.activation_3 = nn.PReLU(num_parameters=s)

        self.expand = nn.Conv2d(s, d, kernel_size=1)
        self.activation_4 = nn.PReLU(num_parameters=d)

        self.deconv = nn.ConvTranspose2d(d, 3, kernel_size=9, stride=scale, 
                                         padding=4, output_padding=scale-1)

    def forward(self, X_in):
        X = self.feature_extract(X_in)
        X = self.activation_1(X)
        
        X = self.shrink(X)
        X = self.activation_2(X)

        X = self.map_1(X)
        X = self.map_2(X)
        X = self.map_3(X)
        X = self.map_4(X)
        X = self.activation_3(X)

        X = self.expand(X)
        X = self.activation_4(X)
        X = self.deconv(X)
        X_out = torch.clip(X, 0.0, 1.0)
        return X_out


# ======================================================================================
# ESPCN network 
# ======================================================================================

class ESPCN_model(nn.Module):
    def __init__(self, scale : int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(32, (3 * scale * scale), kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, X_in):
        X = self.tanh(self.conv_1(X_in))
        X = self.tanh(self.conv_2(X))
        X = self.conv_3(X)
        X = self.pixel_shuffle(X)
        X_out = torch.clip(X, 0.0, 1.0)
        return X_out


# ======================================================================================
# VDSR network 
# ======================================================================================

class VDSR_model(nn.Module):
    def __init__(self) -> None:
        super(VDSR_model, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_2_to_19 = nn.Sequential()
        for i in range(2, 20):
            self.conv_2_to_19.add_module(f"conv_{i}", nn.Conv2d(64, 64, 3, padding=1))
            self.conv_2_to_19.add_module(f"relu_{i}", nn.ReLU())
        self.conv_20 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, X_in):
        x_in = X_in.clone()
        x = self.conv_1(X_in)
        torch.relu_(x)
        x = self.conv_2_to_19(x)
        x = self.conv_20(x)
        x = torch.add(x, x_in)
        x = torch.clip(x, 0.0, 1.0)
        return x
