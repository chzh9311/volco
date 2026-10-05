
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.layers import ResidualStack, Conv3D, MaxPool3D, ResnetBlockFC, MLPResStack
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class VolumeEncoder3D(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, in_dim, h_dims, res_expansion, n_res_layers, feat_dim, N, condition_dim=None,
                 norm='bn', num_groups=8):
        super(VolumeEncoder3D, self).__init__()
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



class VolumeConv3DEnc(nn.Module):
    def __init__(self, cfg):
        super(VolumeConv3DEnc, self).__init__()
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
