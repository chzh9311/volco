
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import ResidualStack, Deconv3D, UpSample3D, Conv3D

class GridDecoder3D(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, latent_dim, h_dims, res_h_dim, n_res_layers, out_dim, N, condition=False):
        super(GridDecoder3D, self).__init__()
        self.h_dim0 = h_dims[0]
        self.num_layers = len(h_dims)
        self.init_N = N // 2**(self.num_layers-1)
        self.init_layer = nn.Sequential(
            nn.Linear(latent_dim, 2*latent_dim),
            nn.ReLU(),
            nn.Linear(2*latent_dim, h_dims[0]*(self.init_N)**3),
            nn.ReLU()
        )
        self.residual_layer = ResidualStack(h_dims[0], res_h_dim=res_h_dim, n_res_layers=n_res_layers)
        self.deconv_layers = nn.ModuleList()
        self.actvn = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i in range(self.num_layers):
            out_channels = h_dims[-1] if i == self.num_layers - 1 else h_dims[i+1]
            # First deconv: 3x3 to process features
            deconv1 = Deconv3D(h_dims[i], h_dims[i], kernel_size=3, stride=1, padding=1)
            # First conv: 1x1 to change channels
            deconv2 = Conv3D(h_dims[i] * (1+condition), out_channels, kernel_size=1, stride=1, padding=0)

            self.deconv_layers.append(nn.ModuleList([deconv1, deconv2]))
            self.actvn.append(nn.ModuleList([nn.ReLU(), nn.ReLU()]))
            self.bn.append(nn.ModuleList([nn.BatchNorm3d(h_dims[i]), nn.BatchNorm3d(out_channels)]))

            # Pool layer (not used after the last layer group)
            if i < self.num_layers - 1:
                self.upsample_layers.append(UpSample3D(scale_factor=2))

        self.final_cse = nn.Conv3d(h_dims[-1], out_dim-1, kernel_size=1,
                            stride=1, padding=0)
        self.final_contact = nn.Sequential(
                nn.Conv3d(h_dims[-1], 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, x, cond=None):
        ## Enable conditional concatenation if cond is provided
        outputs = []
        x = self.init_layer(x)
        x = x.view((-1, self.h_dim0, self.init_N, self.init_N, self.init_N))
        x = self.residual_layer(x)

        for i in range(self.num_layers):
            x = self.deconv_layers[i][0](x)
            x = self.actvn[i][0](self.bn[i][0](x))
            if cond is not None:
                x = self.deconv_layers[i][1](torch.cat([x, cond[i]], dim=1))
            else:
                x = self.deconv_layers[i][1](x)
            x = self.actvn[i][1](self.bn[i][1](x))
            outputs.append(x)

            # Apply pooling if not the last layer
            if i < self.num_layers - 1:
                x = self.upsample_layers[i](x)
        
        contact = self.final_contact(x)
        cse = self.final_cse(x)

        return contact, cse, outputs


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 128, 2, 2, 2))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(128, 32, 17, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
