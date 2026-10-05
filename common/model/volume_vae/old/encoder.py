
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import ResidualStack, Conv3D, MaxPool3D


class GridEncoder3D(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, in_dim, h_dims, res_h_dim, n_res_layers, feat_dim, N, condition=False, has_final_layer=True):
        super(GridEncoder3D, self).__init__()
        self.num_layers = len(h_dims)
        self.conv_layers = nn.ModuleList()
        self.actvn = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        self.has_final_layer = has_final_layer
        for i in range(self.num_layers):
            in_channels = in_dim if i == 0 else h_dims[i-1]
            # First conv: 1x1 to change channels
            conv1 = Conv3D(in_channels, h_dims[i], kernel_size=1, stride=1, padding=0)
            # Second conv: 3x3 to process features
            conv2 = Conv3D(h_dims[i] * (1+condition), h_dims[i], kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(nn.ModuleList([conv1, conv2]))
            self.actvn.append(nn.ModuleList([nn.ReLU(), nn.ReLU()]))
            self.bn.append(nn.ModuleList([nn.BatchNorm3d(h_dims[i]), nn.BatchNorm3d(h_dims[i])]))

            # Pool layer (not used after the last layer group)
            if i < self.num_layers - 1:
                self.pool_layers.append(MaxPool3D(kernel_size=2))
        
        if self.has_final_layer:
            self.final_layer = nn.Sequential(
                ResidualStack(h_dims[-1], res_h_dim=res_h_dim, n_res_layers=n_res_layers),
                nn.Flatten(),
                # ResnetBlockFC(h_dims[-1]*(N//2**(self.num_layers-1))**3, feat_dim, 2 * feat_dim),
                nn.Linear(h_dims[-1]*(N//2**(self.num_layers-1))**3, 2 * feat_dim),
                nn.ReLU(),
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
                x = self.pool_layers[i](x)
        
        if self.has_final_layer:
            x = self.final_layer(x)

            return x, outputs
        else:
            return x, outputs


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 17, 8, 8, 8))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(17, 128, 3, 128)
    encoder_out, encoder_feats = encoder(x)
    print('Encoder out shape:', encoder_out.shape, [feat.shape for feat in encoder_feats])
