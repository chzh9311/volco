import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import GridEncoder3D, GridEncoder3Dv2
from .decoder import GridDecoder3D, GridDecoder3Dv2
from common.msdf.utils.msdf import get_grid
from common.model.layers import DiagonalGaussianDistribution

class GRIDAEAbstract(nn.Module):
    """
    MSDF-based 3D contact VQVAE of one local 3D patch
    The input contact representation is expected to be kernel_size^3 x (1 + 16); 1 refers to contact likelihood, 16 refers to Hand CSE.
    Also, the object local patch is represented as kernel_size^3 SDF values.
    TODO 1: how to integrate scale? Fix the scale to a constant value like 0.01 (1cm);
    Thus the input is k x k x k x (1 + 1 + 16) = k x k x k x 18
    Will be processed by 3D convolutions & deconvolutions.
    TODO 2: how to enable conditional input, i.e., predict the contact given the object local geometry? 
    We can try add this as part of the input to the encoder & decoder.
    """
    def __init__(self, **kwargs):
        super(GRIDAEAbstract, self).__init__()
    
    def encode(self, x, obj_msdf):
        obj_feat, obj_cond = self.obj_encoder(obj_msdf)
        obj_cond = obj_cond + [obj_feat]
        z_e, _ = self.encoder(x, cond=obj_cond + [obj_feat])
        posterior = DiagonalGaussianDistribution(z_e)
        return posterior, obj_feat, obj_cond
    
    def encode_object(self, obj_msdf):
        obj_feat, obj_cond = self.obj_encoder(obj_msdf)
        return obj_feat, obj_cond
    
    def decode(self, z, obj_cond):
        c_hat, cse_hat, _ = self.decoder(z, cond=obj_cond[::-1])
        x_hat = torch.cat([c_hat, cse_hat], dim=1)
        return x_hat

    def forward(self, x, obj_msdf, sample_posterior=True):
        posterior, obj_feat, obj_cond = self.encode(x, obj_msdf)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        ## Adding skip connections for object decoder
        # dec_obj_cond = self.obj_decoder(obj_feat, cond=enc_obj_cond[::-1])

        x_hat = self.decode(z, obj_cond=obj_cond)
        return x_hat, posterior, obj_feat
    
    def inference(self, z_e, obj_msdf):
        _, obj_cond = self.obj_encoder(obj_msdf)
        c_hat, cse_hat, _ = self.decoder(z_e, cond=obj_cond[::-1])
        x_hat = torch.cat([c_hat, cse_hat], dim=1)
        return x_hat
    

class GRIDAEResidual(GRIDAEAbstract):
    def __init__(self, cfg):
        super(GRIDAEAbstract, self).__init__()
        self.cfg = cfg
        self.obj_encoder = GridEncoder3D(in_dim=cfg.obj_in_dim,
                                         h_dims=cfg.obj_h_dims,
                                         res_h_dim=cfg.obj_res_h_dim,
                                         n_res_layers=cfg.obj_n_res_layers,
                                         feat_dim=cfg.obj_feat_dim,
                                         N=cfg.kernel_size,
                                         condition_dim=None)
        self.encoder = GridEncoder3D(in_dim=cfg.in_dim,
                                     h_dims=cfg.h_dims,
                                     res_h_dim=cfg.res_h_dim,
                                     n_res_layers=cfg.n_res_layers,
                                     feat_dim=cfg.feat_dim*2,
                                     N=cfg.kernel_size,
                                     condition_dim=cfg.obj_h_dims + [cfg.obj_feat_dim])
        # pass continuous latent vector through discretization bottleneck
        # decode the discrete latent representation
        # self.obj_decoder = Decoder(h_dims[-1], h_dims[::-1], obj_in_dim, obj_n_res_layers, obj_res_h_dim, condition=True, final_layer=False)
        self.decoder = GridDecoder3D(latent_dim=cfg.feat_dim,
                                     h_dims=cfg.h_dims[::-1],
                                     res_h_dim=cfg.res_h_dim,
                                     n_res_layers=cfg.n_res_layers,
                                     out_dim=cfg.out_dim,
                                     N=cfg.kernel_size,
                                     condition_dim=[cfg.obj_feat_dim] + cfg.obj_h_dims[::-1])
        # if save_img_embedding_map:
        #     self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        # else:
        #     self.img_to_embedding_map = None
        self.grid_coords = get_grid(cfg.msdf.kernel_size) * cfg.msdf.scale


class GRIDAEResidualv2(GRIDAEAbstract):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Normalization for the volumetric blocks. Defaults to 'bn' so existing
        # checkpoints keep loading; set norm: gn in the ae config to drop the
        # BatchNorm running buffers (see make_norm3d).
        norm = cfg.get('norm', 'bn')
        num_groups = cfg.get('num_groups', 8)
        self.obj_encoder = GridEncoder3Dv2(in_dim=cfg.obj_in_dim,
                                         h_dims=cfg.obj_h_dims,
                                         res_expansion=cfg.obj_res_expansion,
                                         n_res_layers=cfg.obj_n_res_layers,
                                         feat_dim=cfg.obj_feat_dim,
                                         N=cfg.kernel_size,
                                         condition_dim=None,
                                         norm=norm, num_groups=num_groups)
        self.encoder = GridEncoder3Dv2(in_dim=cfg.in_dim,
                                     h_dims=cfg.h_dims,
                                     res_expansion=cfg.res_expansion,
                                     n_res_layers=cfg.n_res_layers,
                                     feat_dim=cfg.feat_dim*2,
                                     N=cfg.kernel_size,
                                     condition_dim=cfg.obj_h_dims + [cfg.obj_feat_dim],
                                     norm=norm, num_groups=num_groups)
        # pass continuous latent vector through discretization bottleneck
        # decode the discrete latent representation
        # self.obj_decoder = Decoder(h_dims[-1], h_dims[::-1], obj_in_dim, obj_n_res_layers, obj_res_h_dim, condition=True, final_layer=False)
        self.decoder = GridDecoder3Dv2(latent_dim=cfg.feat_dim,
                                     h_dims=cfg.h_dims[::-1],
                                     res_expansion=cfg.res_expansion,
                                     n_res_layers=cfg.n_res_layers,
                                     out_dim=cfg.out_dim,
                                     N=cfg.kernel_size,
                                     condition_dim=[cfg.obj_feat_dim] + cfg.obj_h_dims[::-1],
                                     norm=norm, num_groups=num_groups)