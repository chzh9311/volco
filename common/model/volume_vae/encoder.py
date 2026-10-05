
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.layers import ResidualStack, Conv3D, MaxPool3D, ResnetBlockFC, MLPResStack
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


# class Encoder(nn.Module):
#     """
#     This is the q_theta (z|x) network. Given a data sample x q_theta 
#     maps to the latent space x -> z.

#     For a VQ VAE, q_theta outputs parameters of a categorical distribution.

#     Inputs:
#     - in_dim : the input dimension
#     - h_dim : the hidden layer dimension
#     - res_h_dim : the hidden dimension of the residual block
#     - n_res_layers : number of layers to stack

#     """

#     def __init__(self, in_dim, h_dims, n_res_layers, res_h_dim, condition=True):
#         super(Encoder, self).__init__()
#         self.condition = condition
#         self.conv0 = Conv3D(in_dim, h_dims[0], kernel_size=1, stride=1, padding=0)
#         ## Double the channels if conditional input is provided
#         self.conv1 = Conv3D(h_dims[0] * (1+condition), h_dims[1], kernel_size=3, stride=1, padding=1) 
#         self.pool1 = MaxPool3D(kernel_size=2)
#         self.conv2 = Conv3D(h_dims[1] * (1+condition), h_dims[2], kernel_size=3, stride=1, padding=1)
#         self.pool2 = MaxPool3D(kernel_size=2)
#         self.conv3 = Conv3D(h_dims[2] * (1+condition), h_dims[3], kernel_size=3, stride=1, padding=1)
#         self.res_stack = ResidualStack(h_dims[3], h_dims[3], res_h_dim, n_res_layers)

        # self.conv_stack = nn.Sequential(
        #     nn.Conv3d(in_dim, h_dim // 2, kernel_size=kernel,
        #               stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2),
        #     nn.Conv3d(h_dim // 2, h_dim, kernel_size=kernel,
        #               stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2),
        #     nn.Conv3d(h_dim, h_dim, kernel_size=kernel-1,
        #               stride=stride-1, padding=1),
        #     ResidualStack(
        #         h_dim, h_dim, res_h_dim, n_res_layers)

        # )

    # def forward(self, x, cond=None):
    #     ## Concatenate the features once for each feature scale
    #     x0 = self.conv0(x)
    #     if not self.condition:
    #         x1 = self.conv1(x0)
    #         x1 = self.pool1(x1)
    #         x2 = self.conv2(x1)
    #         x2 = self.pool2(x2)
    #         x3 = self.conv3(x2)
    #     else:
    #         assert cond is not None, "Conditional features must be provided!"
    #         x1 = self.conv1(torch.cat([x0, cond[0]], dim=1))
    #         x1 = self.pool1(x1)
    #         x2 = self.conv2(torch.cat([x1, cond[1]], dim=1))
    #         x2 = self.pool2(x2)
    #         x3 = self.conv3(torch.cat([x2, cond[2]], dim=1))
    #     x3 = self.res_stack(x3)
    #     return x3, [x0, x1, x2]


class GridEncoder3D(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, in_dim, h_dims, res_h_dim, n_res_layers, feat_dim, N, condition_dim=None):
        super(GridEncoder3D, self).__init__()
        self.num_layers = len(h_dims)
        self.conv_layers = nn.ModuleList()
        self.actvn = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        if condition_dim is None:
            condition_dim = [0] * (self.num_layers + 1)
        for i in range(self.num_layers):
            in_channels = in_dim if i == 0 else h_dims[i-1]
            # First conv: 1x1 to change channels
            conv1 = Conv3D(in_channels, h_dims[i], kernel_size=1, stride=1, padding=0)
            # Second conv: 3x3 to process features
            conv2 = Conv3D(h_dims[i] + condition_dim[i], h_dims[i], kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(nn.ModuleList([conv1, conv2]))
            self.actvn.append(nn.ModuleList([nn.ReLU(), nn.ReLU()]))
            self.bn.append(nn.ModuleList([nn.BatchNorm3d(h_dims[i]), nn.BatchNorm3d(h_dims[i])]))

            # Pool layer (not used after the last layer group)
            if i < self.num_layers - 1:
                # self.pool_layers.append(MaxPool3D(kernel_size=2))
                ## Use strided conv for downsampling to restore the maximum information
                self.downsample_layers.append(Conv3D(h_dims[i], h_dims[i], kernel_size=3, stride=2, padding=1))
        
        self.final_residual = ResidualStack(h_dims[-1], res_h_dim=res_h_dim, n_res_layers=n_res_layers)
        self.final_layer = nn.Sequential(
            # ResnetBlockFC(h_dims[-1]*(N//2**(self.num_layers-1))**3, feat_dim, 2 * feat_dim),
            nn.Linear(h_dims[-1]*(N//2**(self.num_layers-1))**3 + condition_dim[-1], 2 * feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * feat_dim, feat_dim)
        )

    def forward(self, x, cond=None):
        ## Enable conditional concatenation if cond is provided
        outputs = []

        for i in range(self.num_layers):
            x = self.conv_layers[i][0](x)
            x = self.actvn[i][0](self.bn[i][0](x))
            if cond is not None:
                x = self.conv_layers[i][1](torch.cat([x, cond[i]], dim=1))
            else:
                x = self.conv_layers[i][1](x)
            x = self.actvn[i][1](self.bn[i][1](x))
            outputs.append(x)

            # Apply pooling if not the last layer
            if i < self.num_layers - 1:
                x = self.downsample_layers[i](x)
        
        x = self.final_residual(x)
        x = x.view(x.size(0), -1)  # Flatten the spatial dimensions
        if cond is not None:
            x = torch.cat([x, cond[-1]], dim=1)
        x = self.final_layer(x)

        return x, outputs


class GridEncoder3Dv2(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, in_dim, h_dims, res_expansion, n_res_layers, feat_dim, N, condition_dim=None,
                 norm='bn', num_groups=8):
        super(GridEncoder3Dv2, self).__init__()
        self.num_layers = len(h_dims)
        self.conv_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        norm_kw = dict(norm=norm, num_groups=num_groups)

        if condition_dim is None:
            condition_dim = [0] * (self.num_layers + 1)
        for i in range(self.num_layers):
            in_channels = in_dim if i == 0 else h_dims[i-1]
            # First conv: 1x1 to change channels
            conv1 = Conv3D(in_channels, h_dims[i], kernel_size=1, stride=1, padding=0, **norm_kw)
            # Second conv: 3x3 to process features
            conv2 = Conv3D(h_dims[i] + condition_dim[i], h_dims[i], kernel_size=3, stride=1, padding=1, **norm_kw)
            self.conv_layers.append(nn.ModuleList([conv1, conv2]))

            # Pool layer (not used after the last layer group)
            if i < self.num_layers - 1:
                if i == 0 and N == 4:
                    # do not downsample:
                    self.downsample_layers.append(Conv3D(h_dims[i], h_dims[i], kernel_size=1, stride=1, padding=0, **norm_kw))
                else:
                    # self.pool_layers.append(MaxPool3D(kernel_size=2))
                    ## Use strided conv for downsampling to restore the maximum information
                    self.downsample_layers.append(Conv3D(h_dims[i], h_dims[i], kernel_size=2, stride=2, padding=0, **norm_kw))
        
        # Count the downsamples that actually halve the resolution. For N==4 the
        # first downsample (i==0) is a stride-1 no-op, so one fewer halving occurs.
        num_downsamples = self.num_layers - 1
        if N == 4:
            num_downsamples -= 1
        final_N = N // 2**num_downsamples
        flattened_dim = h_dims[-1]*final_N**3
        self.final_residual = MLPResStack(flattened_dim, expansion_factor=res_expansion, n_res_layers=n_res_layers)
        self.final_layer = nn.Sequential(
            # ResnetBlockFC(h_dims[-1]*(N//2**(self.num_layers-1))**3, feat_dim, 2 * feat_dim),
            nn.Linear(flattened_dim + condition_dim[-1], 4 * feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4 * feat_dim, feat_dim)
        )

    def forward(self, x, cond=None):
        ## Enable conditional concatenation if cond is provided
        outputs = []

        for i in range(self.num_layers):
            x = self.conv_layers[i][0](x)
            if cond is not None:
                x = self.conv_layers[i][1](torch.cat([x, cond[i]], dim=1))
            else:
                x = self.conv_layers[i][1](x)
            outputs.append(x)

            # Apply pooling if not the last layer
            if i < self.num_layers - 1:
                x = self.downsample_layers[i](x)
        
        x = x.view(x.size(0), -1)  # Flatten the spatial dimensions
        x = self.final_residual(x)
        if cond is not None:
            x = torch.cat([x, cond[-1]], dim=1)
        x = self.final_layer(x)

        return x, outputs



class GridConv3DEnc(nn.Module):
    def __init__(self, cfg):
        super(GridConv3DEnc, self).__init__()
        self.cfg = cfg
        self.conv1 = Conv3D(cfg.in_dim, cfg.h_dims[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv3D(cfg.h_dims[0], cfg.h_dims[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv3D(cfg.h_dims[1], cfg.h_dims[2], kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool3D(kernel_size=2)
        self.pool2 = MaxPool3D(kernel_size=2)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 17, 8, 8, 8))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(17, 128, 3, 128)
    encoder_out, encoder_feats = encoder(x)
    print('Encoder out shape:', encoder_out.shape, [feat.shape for feat in encoder_feats])
