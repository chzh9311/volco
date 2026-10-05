
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.layers import ResidualStack, Deconv3D, UpSample3D, Conv3D, MLPResStack
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class VolumeDecoder3D(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, latent_dim, h_dims, res_expansion, n_res_layers, out_dim, N, condition_dim=None,
                 norm='bn', num_groups=8):
        super(VolumeDecoder3D, self).__init__()
        norm_kw = dict(norm=norm, num_groups=num_groups)
        self.h_dim0 = h_dims[0]
        self.num_layers = len(h_dims)
        # Mirror the encoder's downsampling: for N==4 one upsample is a stride-1
        # no-op, so one fewer doubling occurs. init_N must match the encoder's
        # final spatial resolution.
        num_upsamples = self.num_layers - 1
        if N == 4:
            num_upsamples -= 1
        self.init_N = N // 2**num_upsamples
        if condition_dim is None:
            condition_dim = [0] * (self.num_layers + 1)

        self.init_layer = nn.Sequential(
            nn.Linear(latent_dim + condition_dim[0], 4*latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4*latent_dim, h_dims[0]*(self.init_N)**3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.residual_layer = MLPResStack(h_dims[0]*(self.init_N)**3, expansion_factor=res_expansion, n_res_layers=n_res_layers)
        self.deconv_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i in range(self.num_layers):
            out_channels = h_dims[-1] if i == self.num_layers - 1 else h_dims[i+1]
            # First deconv: 3x3 to process features
            deconv1 = Deconv3D(h_dims[i], h_dims[i], kernel_size=3, stride=1, padding=1, **norm_kw)
            # First conv: 1x1 to change channels
            deconv2 = Conv3D(h_dims[i] + condition_dim[i+1], out_channels, kernel_size=1, stride=1, padding=0, **norm_kw)

            self.deconv_layers.append(nn.ModuleList([deconv1, deconv2]))

            # Pool layer (not used after the last layer group)
            if i < self.num_layers - 1:
                if N == 4 and i == self.num_layers - 2:
                    ## Do not up sample in the last layer if N==4,
                    self.upsample_layers.append(Deconv3D(out_channels, out_channels, kernel_size=1, stride=1, padding=0, **norm_kw))
                else:
                    self.upsample_layers.append(Deconv3D(out_channels, out_channels, kernel_size=2, stride=2, padding=0, **norm_kw))

        self.final_cse = nn.Conv3d(h_dims[-1], out_dim-1, kernel_size=1,
                            stride=1, padding=0)
        self.final_contact = nn.Sequential(
                nn.Conv3d(h_dims[-1], 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

    def forward(self, x, cond=None):
        ## Enable conditional concatenation if cond is provided
        outputs = []
        if cond is not None:
            x = torch.cat([x, cond[0]], dim=1)
        x = self.init_layer(x)
        x = self.residual_layer(x)
        x = x.view((-1, self.h_dim0, self.init_N, self.init_N, self.init_N))

        for i in range(self.num_layers):
            x = self.deconv_layers[i][0](x)
            if cond is not None:
                x = self.deconv_layers[i][1](torch.cat([x, cond[i+1]], dim=1))
            else:
                x = self.deconv_layers[i][1](x)
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
    decoder = VolumeDecoder3D(128, [32, 64, 128], 32, 3, 64, 8)
    decoder_out = decoder(x)
    print('Decoder out shape:', [o.shape for o in decoder_out])