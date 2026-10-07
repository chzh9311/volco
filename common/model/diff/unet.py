"""
Adopted from SceneDiffuser: https://github.com/scenediffuser/Scene-Diffuser 
"""
from typing import Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .utils import timestep_embedding
from .utils import ResBlock, SpatialTransformer, DualSpatialTransformer
from common.model.pointnet_module import Pointnet_feat_net
# from .scene_model import create_scene_model

class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super(UNetModel, self).__init__()

        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.obj_feat_dim = cfg.obj_feat_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x

        ## create scene model from config
        # self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)
        self.obj_feat_net = Pointnet_feat_net(**cfg.obj_encoder)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.obj_feat_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    # context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb, cond)
            h = self.layers[i * 2 + 1](h)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition feature
        """
        b = data['obj_pc'].shape[0]
        pc = data['obj_pc'].to(torch.float32)
        local_obj_feat, glob_obj_feat = self.obj_feat_net(pc)
        # obj_feat = torch.cat([local_obj_feat, glob_obj_feat.unsqueeze(2).repeat(1, 1, local_obj_feat.shape[2])], dim=1)
        obj_feat = torch.cat([local_obj_feat, glob_obj_feat.unsqueeze(2).repeat(1, 1, local_obj_feat.shape[2])], dim=1)

        return obj_feat


class DualUNetModel(nn.Module):
    """
    Dual diffusion UNet model that separately processes local and global features, with cross attention to fuse them.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super(DualUNetModel, self).__init__()

        self.d_x = cfg.d_x
        self.d_y = cfg.d_y
        self.n_pts = cfg.n_pts
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.n_timesteps = cfg.n_timesteps
        self.context_dim = cfg.context_dim
        self.obj_feat_dim = cfg.obj_feat_dim
        self.use_position_embedding = cfg.use_position_embedding # for input sequence x
        ## create scene model from config
        # self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)
        self.obj_feat_net = Pointnet_feat_net(**cfg.obj_encoder)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.local_in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.global_in_layers = nn.Sequential(
            nn.Conv1d(self.d_y, self.d_model, 1)
        )
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        time_embed_dim = self.d_model * cfg.time_embed_mult

        for i in range(self.nblocks):
            self.local_layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.obj_feat_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.global_layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    0, # no object feature here
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            ## With cross Attention
            self.cross_layers.append(
                DualSpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )

        self.local_out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )

        self.global_out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_y, 1),
        )


    def forward(self, x_t: torch.Tensor,  ts: torch.Tensor, cond: Dict, is_train=True) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input latent, <B, C_global+C_local>
            ts: timestep, 1-D batch of timesteps
            cond: condition dict with 'obj_feat' and optionally 'obj_msdf'

        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        obj_feat = cond['obj_feat']
        obj_msdf = cond.get('obj_msdf', None)

        y_t, x_t = torch.split(x_t, [self.d_y, self.d_x * self.n_pts], dim=-1)
        x_t = x_t.view(y_t.shape[0], self.n_pts, self.d_x)
        x_in_shape = len(x_t.shape)
        if x_in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        y_in_shape = len(y_t.shape)
        if y_in_shape == 2:
            y_t = y_t.unsqueeze(1)
        assert len(y_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h_local = rearrange(x_t, 'b l c -> b c l')
        h_local = self.local_in_layers(h_local) # <B, d_model, L>
        h_global = rearrange(y_t, 'b l c -> b c l')
        h_global = self.global_in_layers(h_global) # <B, d_model, L>
        # print(h.shape, cond.shape) # <B, d_model, L>, <B, T , c_dim>

        for i in range(self.nblocks):
            h_local = self.local_layers[i](h_local, t_emb, obj_feat)
            h_global = self.global_layers[i](h_global, t_emb)
            h_local, h_global = self.cross_layers[i](x=h_local, y=h_global)
        h_local = self.local_out_layers(h_local)
        h_global = self.global_out_layers(h_global)
        h_local = rearrange(h_local, 'b c l -> b l c')
        h_global = rearrange(h_global, 'b c l -> b l c')

        ## reverse to original shape
        if x_in_shape == 2:
            h_local = h_local.squeeze(1)

        if y_in_shape == 2:
            h_global = h_global.squeeze(1)

        if self.global2local_fn is not None and is_train:
            with torch.no_grad():
                recon_local = self.global2local_fn(h_global, obj_msdf) # <B, N, d_x>
                # w = (ts / (self.n_timesteps - 1)).view(-1, 1, 1)
                ## Gradually fuze the global info back to local.
                ## t=0: only reconstructed; t=999: only diffused
                # h_local = (h_local + recon_local) / 2
                # h_local = (1-w) * recon_local + w * h_local
                # difference_loss = F.mse_loss(h_local, recon_local.detach())
        else:
            recon_local = torch.zeros_like(h_local, device=h_local.device).float()

        x_ret = torch.cat([h_global, h_local.reshape(h_local.shape[0], self.n_pts * self.d_x)], dim=-1)
        return x_ret, recon_local

    def condition(self, data: Dict) -> Dict:
        """ Obtain scene feature with scene model

        Args:
            data: dataloader-provided data

        Return:
            Condition dict with 'obj_feat' and optionally 'obj_msdf'
        """
        pc = data['obj_pc'].to(torch.float32)
        local_obj_feat, glob_obj_feat = self.obj_feat_net(pc)
        obj_feat = torch.cat([local_obj_feat, glob_obj_feat.unsqueeze(2).repeat(1, 1, local_obj_feat.shape[2])], dim=1)

        cond = {'obj_feat': obj_feat}
        if 'obj_msdf' in data:
            cond['obj_msdf'] = data['obj_msdf']
        return cond