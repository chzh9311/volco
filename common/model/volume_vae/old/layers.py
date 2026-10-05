
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxPool3D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool3D, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


class FiLM(nn.Module):
    def __init__(self, input_dim, num_features):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.gamma_layer = nn.Linear(input_dim, num_features)
        self.beta_layer = nn.Linear(input_dim, num_features)

    def forward(self, x, cond):
        gamma = self.gamma_layer(cond.flatten())
        beta = self.beta_layer(cond.flatten())
        out = gamma * x + beta
        return out


class UpSample3D(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(UpSample3D, self).__init__()
        self.upscale = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        return self.upscale(x)


class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class Deconv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Deconv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)

class Bottleneck(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    Actually a bottleneck architecture
    """

    def __init__(self, in_dim, res_h_dim):
        super(Bottleneck, self).__init__()
        self.relu = nn.ReLU(True)
        self.res_block = nn.Sequential(
            nn.Conv3d(in_dim, res_h_dim, kernel_size=1,
                      stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(res_h_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(res_h_dim, in_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        x = self.relu(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [Bottleneck(in_dim, res_h_dim) for _ in range(n_res_layers)])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return x


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, final_nl=False):
        net = self.fc_0(x)
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        x_out = x_s + dx
        if final_nl:
            return F.leaky_relu(x_out, negative_slope=0.2)
        return x_out


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar, dim=1)
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=1)

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean



if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 64, 2, 2, 2))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(64, 64, 128)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(64, 64, 128, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
