import os.path as osp
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import lightning as L
import open3d as o3d
import trimesh
from copy import copy, deepcopy
import numpy as np
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
import wandb
from matplotlib import pyplot as plt

from common.manopth.manopth.manolayer import ManoLayer
from common.model.handobject import recover_hand_verts_from_contact
from common.model.pose_optimizer import optimize_pose_wrt_local_grids, optimize_pose_by_contact
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.geometry import GridDistanceToContact
from common.utils.physics import StableLoss
from common.utils.vis import visualize_recon_hand_w_object, visualize_grid_contact, o3dmesh, o3dmesh_from_trimesh
from common.msdf.utils.msdf import get_grid, calc_local_grid_all_pts_gpu
from common.evaluation.eval_fns import calculate_metrics, calc_diversity
from einops import rearrange
from common.model.sam import SAM


class LGCDiffTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, grid_ae, model, diffusion, cfg, **kwargs):
        super().__init__()
        self.automatic_optimization = cfg.train.optimizer != 'asam'
        self.grid_ae = grid_ae
        # if cfg.pose_optimizer.name == 'hand_ae':
        #     self.hand_ae = kwargs.get('hand_ae', None)
        self.model = model
        self.diffusion = diffusion
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.debug = cfg.get('debug', False)
        self.msdf_k = cfg.msdf.kernel_size
        self.lr = cfg.train.lr
        self.loss_weights = cfg.train.get('loss_weights', {})
        self.grid_dist_to_contact = GridDistanceToContact.from_config(cfg.msdf, method=cfg.msdf.contact_method)
        self.stable_loss = StableLoss(k=cfg.physics.k, mu=cfg.physics.mu, pene_th=cfg.physics.pene_th, eps=cfg.physics.eps)

        ## Load autoencoder pretrained weights and freeze before DDP wrapping
        self.pretrained_keys = []
        if cfg.run_phase == 'train':
            self._load_pretrained_weights(cfg.ae.get('pretrained_weight', None), target_prefix='grid_ae')
            self.grid_ae.eval()
            self.grid_ae.requires_grad_(False)

        # if cfg.pose_optimizer.name == 'hand_ae':
        #     self._load_pretrained_weights(cfg.hand_ae.get('pretrained_weight', None), target_prefix='hand_ae')
    
        mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=cfg.pose_optimizer.use_pca, ncomps=cfg.pose_optimizer.ncomps, flat_hand_mean=True)
        object.__setattr__(self, 'mano_layer', mano_layer.eval().requires_grad_(False))
        self.closed_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'))
        cse_ckpt = torch.load(cfg.data.hand_cse_path, weights_only=False)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.cse_dim = cse_ckpt['emb_dim']
        self.hand_cse = HandCSE(n_verts=778, emb_dim=self.cse_dim, cano_faces=handF.cpu().numpy()).to(self.device)
        self.hand_cse.load_state_dict(cse_ckpt['state_dict'])
        self.hand_cse.eval().requires_grad_(False)
        self.normalized_grid_coords = get_grid(self.cfg.msdf.kernel_size)
        self.grid_coords = self.normalized_grid_coords * self.cfg.msdf.scale  # (K^3, 3)
        self.msdf_scale = self.cfg.msdf.scale
        self.pool = Pool(processes=min(self.cfg.test.batch_size, 16))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                batch[key] = value.float()
        return batch

    def _load_pretrained_weights(self, checkpoint_path, target_prefix='grid_ae'):
        """
        Load pretrained weights from a Lightning checkpoint.
        Only loads weights for layers that exist in both the checkpoint and current model.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file
            target_prefix: Prefix to add to checkpoint keys when loading into current model
                          (e.g., 'grid_ae' will map 'model.encoder.xxx' to 'grid_ae.encoder.xxx')
        """
        import os

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Skipping weight initialization.")
            return

        print(f"Loading pretrained weights from {checkpoint_path} with prefix '{target_prefix}'")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state dict from Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter state dict and remap keys
        # Lightning saves with 'model.' prefix, we need to extract and remap weights
        remapped_state_dict = {}
        for key, value in state_dict.items():
            # Look for keys like 'model.encoder.xxx' or 'model.decoder.xxx'
            if key.startswith('model.'):
                # Remove 'model.' prefix to get the actual model key
                model_key = key[6:]  # Remove 'model.'
                # Add target prefix to match our model structure
                new_key = f'{target_prefix}.{model_key}'
                remapped_state_dict[new_key] = value

        # Get current model state dict
        current_state_dict = self.state_dict()

        # Filter to only load weights that exist in current model (handle extra layers)
        filtered_state_dict = {}
        for key, value in remapped_state_dict.items():
            if key in current_state_dict:
                # Check if shapes match
                if current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Warning: Shape mismatch for {key}. "
                          f"Checkpoint: {value.shape}, Current: {current_state_dict[key].shape}. Skipping.")
            else:
                print(f"Info: {key} in checkpoint but not in current model. Skipping.")

        # Load the filtered state dict
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

        # Track which keys were successfully loaded
        self.pretrained_keys.extend(filtered_state_dict.keys())

        # Count only missing keys with the target prefix
        missing_prefixed_keys = [k for k in missing_keys if k.startswith(f'{target_prefix}.')]

        print(f"Successfully loaded {len(filtered_state_dict)} layers from pretrained checkpoint")
        if missing_prefixed_keys:
            print(f"Missing {target_prefix} keys: {len(missing_prefixed_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    def _freeze_pretrained_weights(self):
        """
        Freeze all parameters that were loaded from the pretrained checkpoint.
        Also sets BatchNorm layers to eval mode to prevent running stats updates.
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if name in self.pretrained_keys:
                param.requires_grad = False
                frozen_count += 1

        # Set grid_ae BatchNorm layers to eval mode to prevent running stats drift
        bn_count = 0
        for module in self.grid_ae.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
                # Disable running stats updates during forward pass
                module.track_running_stats = False
                bn_count += 1

        print(f"Frozen {frozen_count} pretrained parameters in grid_ae")
        print(f"Set {bn_count} BatchNorm layers to eval mode (prevents running stats drift)")

    # def on_fit_start(self):
    #     """
    #     Called after checkpoint restoration but before training starts.
    #     Freezes pretrained weights here to ensure they remain frozen even when resuming.
    #     """
    #     if self.cfg.run_phase == 'train' and hasattr(self, 'pretrained_keys'):
    #         self._freeze_pretrained_weights()

    def training_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        if self.global_step % 50 == 0 and self.trainer.is_global_zero:
            logged = self.trainer.callback_metrics
            loss_str = ' | '.join(f'{k}: {v:.4f}' for k, v in logged.items() if 'train' in k)
            print(f'[step {self.global_step}] {loss_str}', flush=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')
        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        self.mano_layer.to(self.device)
        self.grid_coords = self.grid_coords.view(-1, 3).to(self.device)
        obj_names = batch['objName']
        # simp_obj_mesh = getattr(self.trainer.datamodule, f'train_set').simp_obj_mesh
        # obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
        #                  for name in obj_names]
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)
        handobject.load_from_batch(batch)
        lg_contact = handobject.ml_contact
        batch_size, n_grids = lg_contact.shape[:2]
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k)
        # n_ho_dist = handobject.n_ho_dist

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        ## First process all grids separately using GRIDAE
        flat_lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> b n (k1 k2 k3) c')
        lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> (b n) c k1 k2 k3')
        posterior, obj_feat, multi_scale_obj_cond = self.grid_ae.encode(lg_contact, obj_msdf)
        # obj_pc = torch.cat([obj_msdf_center, obj_feat.view(batch_size, n_grids, -1)], dim=-1)
        obj_pc = handobject.obj_msdf
        # z = torch.cat([n_ho_dist.unsqueeze(-1), posterior.sample().view(batch_size, n_grids, -1)], dim=-1) # n_dim + 1
        z = posterior.sample().view(batch_size, n_grids, -1) # n_dim

        input_data = {'x': z, 'obj_pc': obj_pc.permute(0, 2, 1)}

        # Check for NaN in input data
        for key, value in input_data.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in input_data['{key}'] at batch_idx {batch_idx}")

        # grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        losses, pred_x0 = self.diffusion.training_losses(self.model, input_data, grid_ae=self.grid_ae, ms_obj_cond=multi_scale_obj_cond,
                                              hand_cse=self.hand_cse, msdf_k=self.msdf_k, grid_scale=self.msdf_scale,
                                              gt_lg_contact=flat_lg_contact,
                                              adj_pt_indices=batch['adjPointIndices'],
                                              adj_pt_distances=batch['adjPointDistances'])

        batch_size, n_grids = pred_x0.shape[:2]
        pred_x0 = pred_x0.reshape(batch_size*n_grids, -1)
        recon_lg_contact = self.grid_ae.decode(pred_x0, multi_scale_obj_cond)
        # recon_lg_contact = recon_lg_contact.view(batch_size, n_grids, -1, self.msdf_k ** 3).permute(0, 1, 3, 2)
        recon_lg_contact = rearrange(recon_lg_contact, '(b n) c k1 k2 k3 -> b (n k1 k2 k3) c', b=batch_size, n=n_grids)

        sdf_vals = handobject.obj_msdf[:, :, :self.msdf_k**3]        # (B, N, K^3)
        centres = handobject.obj_msdf[:, :, self.msdf_k**3:]          # (B, N, 3)
        all_pts = centres.unsqueeze(2) + handobject.normalized_coords[None, None, :, :] * self.msdf_scale
        sdf_flat = sdf_vals.reshape(batch_size, n_grids * self.msdf_k**3)
        sdf_grad = handobject.msdf_grad.reshape(batch_size, n_grids * self.msdf_k**3, 3)
        n_adj_pt = handobject.n_adj_pt.view(batch_size, n_grids * self.msdf_k**3)
        sdf_flat = sdf_flat * self.msdf_scale * np.sqrt(3)
        ## Assume gravity direction is always (0, 0, -1), since there're some tolerance for penetration error.
        lgc = recon_lg_contact[..., 0]  # B x N*K^3
        lgc = torch.where(lgc < 0.05, torch.zeros_like(lgc), lgc)

        stable_loss = self.stable_loss(sdf_flat, all_pts.view(batch_size, -1, 3), lgc, sdf_grad, n_adj_pt,
                                obj_mass=handobject.obj_mass, gravity_direction=torch.FloatTensor([[0, 0, -1]]).to(self.device),
                                J=handobject.obj_inertia)
        losses['stable_loss'] = stable_loss.mean()

        total_loss = sum([losses[k] * self.loss_weights[k] for k in losses.keys()])
        losses['total_loss'] = total_loss

        if stage == 'train' and self.cfg.train.optimizer == 'asam':
            optimizer = self.optimizers()
            no_sync = self.trainer.model.no_sync if hasattr(self.trainer.model, 'no_sync') else nullcontext
            with no_sync():
                self.manual_backward(total_loss)
            optimizer.first_step(zero_grad=True)

            # Recompute forward pass with perturbed weights for the second step
            lg_contact_2 = rearrange(handobject.ml_contact, 'b n k1 k2 k3 c -> (b n) c k1 k2 k3')
            flat_lg_contact_2 = rearrange(handobject.ml_contact, 'b n k1 k2 k3 c -> b n (k1 k2 k3) c')
            posterior_2, obj_feat_2, multi_scale_obj_cond_2 = self.grid_ae.encode(lg_contact_2, obj_msdf.detach())
            obj_pc_2 = torch.cat([obj_msdf_center.detach(), obj_feat_2.view(batch_size, n_grids, -1)], dim=-1)
            z_2 = posterior_2.sample().view(batch_size, n_grids, -1)
            input_data_2 = {'x': z_2, 'obj_pc': obj_pc_2.permute(0, 2, 1)}

            losses2 = self.diffusion.training_losses(self.model, input_data_2, grid_ae=self.grid_ae, ms_obj_cond=multi_scale_obj_cond_2,
                                                  hand_cse=self.hand_cse, msdf_k=self.msdf_k, grid_scale=self.msdf_scale,
                                                  gt_lg_contact=flat_lg_contact_2,
                                                  adj_pt_indices=batch['adjPointIndices'],
                                                  adj_pt_distances=batch['adjPointDistances'])
            total_loss = sum([losses2[k] * self.loss_weights[k] for k in losses2.keys()])
            losses2['total_loss'] = total_loss
            self.manual_backward(total_loss)
            optimizer.second_step(zero_grad=True)

        # grid_loss = F.mse_loss(err[:, :, 0], torch.zeros_like(err[:, :, 0]), reduction='mean')  # only compute loss on n_ho_dist dimension
        # diff_loss = F.mse_loss(err[:, :, 1:], torch.zeros_like(err[:, :, 1:]), reduction='none')
        # obj_pt_mask = input_data['x'][:, :, 0:1] < 0
        # diff_loss = torch.sum(diff_loss * obj_pt_mask) / (torch.sum(obj_pt_mask) + 1e-6) / diff_loss.shape[2]

        # loss_dict = {f'{stage}/grid_contact_loss': grid_loss,
        #              f'{stage}/latent_diff_loss': diff_loss,
        #              f'{stage}/total_loss': grid_loss + diff_loss}
        loss_dict = {f'{stage}/{k}': v for k, v in losses.items()}
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        ## Also sample and reconstruct
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            condition = self.model.condition(input_data)
            samples = self.diffusion.p_sample_loop(self.model, input_data['x'].shape, condition, clip_denoised=False)
            # samples = self.diffusion.p_sample_loop(self.model, input_data['x'].shape, input_data['obj_pc'])
            vis_idx = 0
            simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
            rot = batch['aug_rot'][vis_idx].cpu().numpy() if 'aug_rot' in batch else np.eye(3)
            obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
                            for i, name in enumerate(batch['objName'])]
            handobject._load_templates(idx=vis_idx, obj_templates=obj_templates)
            # sample_latent = samples[:, 0] ## 1 + latent
            sample_latent = samples  ## latent
            # grid_contact, sample_latent = sample_x[:, :, 0], sample_x[:, :, 1:]
            # grid_contact = grid_contact < 0
            pred_hand_verts, pred_verts_mask, pred_grid_contact = self.reconstruct_from_latent(sample_latent, multi_scale_obj_cond, obj_msdf_center)

            gt_latent = input_data['x']

            # recon_lg_contact, z_e, obj_feat = self.grid_ae(
            #     lg_contact, obj_msdf=obj_msdf, sample_posterior=False)
            # recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x N x K x K x K x (1 + cse_dim)
            # recon_lg_contact = recon_lg_contact.view(batch_size, 128, self.msdf_k, self.msdf_k, self.msdf_k, -1)
            # # recon_lg_contact = recon_lg_contact.view(batch_size, n_grids, -1, self.msdf_k ** 3).permute(0, 1, 3, 2)
            # recon_lg_contact = recon_lg_contact * handobject.obj_pt_mask[:, :, None, None, None, None]
            # sample_contact = recon_lg_contact[..., 0].reshape(batch_size, -1)
            # sample_cse = recon_lg_contact[..., 1:].reshape(batch_size, -1, self.cse_dim)
            # grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3

            # pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            #     self.hand_cse, None,
            #     sample_contact.reshape(batch_size, -1), sample_cse.reshape(batch_size, -1, self.cse_dim),
            #     grid_coords=grid_coords.reshape(batch_size, -1, 3),
            # )

            gt_rec_hand_verts, gt_rec_verts_mask, gt_rec_grid_contact = self.reconstruct_from_latent(gt_latent, multi_scale_obj_cond, obj_msdf_center)

            contact_img = self._visualize_contact_comparison(
                obj_msdf_center, pred_grid_contact, gt_rec_grid_contact, handobject, vis_idx)
            img = self._visualize_hand_comparison(
                pred_hand_verts, pred_verts_mask, gt_rec_hand_verts, gt_rec_verts_mask,
                handobject, obj_msdf_center, rot, vis_idx)
            # plt.imsave(f'tmp/contact_vis_{batch_idx}.png', contact_img)
            # plt.imsave(f'tmp/rec_img_{batch_idx}.png', img)

            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_contact', contact_img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_hand', img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_contact': wandb.Image(contact_img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_hand': wandb.Image(img)}, step=self.global_step)

        return losses['total_loss']
                                  
        # recon_loss = F.mse_loss(recon_grid_contact, gt_grid_contact.permute(0, 4, 1, 2, 3))
        # loss_dict = {f'{stage}/embedding_loss': loss, f'{stage}/recon_loss': recon_loss, f'{stage}/perplexity': perplexity}
        # total_loss = sum(loss_dict.values())
        # total_loss = loss + self.recon_weight * recon_loss
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            # Log image - works for both WandbLogger and TensorBoardLogger
            vis_idx = 0
            pred_img, pred_geoms = self.visualize_recon_hand_w_object(hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=handobject.obj_models[vis_idx],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)
            gt_img, gt_geoms = self.visualize_recon_hand_w_object(hand_verts=handobject.hand_verts[vis_idx].detach().cpu().numpy(),
                                        # hand_verts_mask=handobject.hand_vert_mask[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=handobject.obj_models[vis_idx],
                                        mesh_color=[0.2, 0.4, 0.8],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)
            # vis_geoms = pred_geoms + [g.translate((0, 0.25, 0)) for g in gt_geoms]
            # o3d.visualization.draw_geometries(vis_geoms, window_name='GT Hand-Object')
            img = np.concatenate([gt_img, pred_img], axis=0)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/compare_hand', img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/compare_hand': wandb.Image(img)},
                                              step=self.global_step)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict['total_loss']
    
    def _visualize_contact_comparison(self, obj_msdf_center, pred_grid_contact, gt_rec_grid_contact, handobject, vis_idx):
        backend = self.cfg.get('vis_backend', 'open3d')
        pred_contact_img, _ = visualize_grid_contact(
            contact_pts=obj_msdf_center[vis_idx].detach().cpu().numpy(),
            pt_contact=pred_grid_contact[vis_idx].detach().cpu().numpy(),
            grid_scale=self.cfg.msdf.scale, obj_mesh=handobject.vis_obj_models[vis_idx], w=400, h=400,
            backend=backend)
        gt_rec_contact_img, _ = visualize_grid_contact(
            contact_pts=obj_msdf_center[vis_idx].detach().cpu().numpy(),
            pt_contact=gt_rec_grid_contact[vis_idx].detach().cpu().numpy(),
            grid_scale=self.cfg.msdf.scale, obj_mesh=handobject.vis_obj_models[vis_idx], w=400, h=400,
            backend=backend)
        gt_grid_contact = rearrange(handobject.ml_contact[vis_idx, ..., 0], 'n k1 k2 k3 -> n (k1 k2 k3)').max(dim=-1)[0]
        gt_contact_img, _ = visualize_grid_contact(
            contact_pts=obj_msdf_center[vis_idx].detach().cpu().numpy(),
            pt_contact=gt_grid_contact.detach().cpu().numpy(),
            grid_scale=self.cfg.msdf.scale, obj_mesh=handobject.vis_obj_models[vis_idx], w=400, h=400,
            backend=backend)
        return np.concatenate([gt_contact_img, gt_rec_contact_img, pred_contact_img], axis=0)

    def _visualize_hand_comparison(self, pred_hand_verts, pred_verts_mask, gt_rec_hand_verts, gt_rec_verts_mask,
                                   handobject, obj_msdf_center, rot, vis_idx):
        backend = self.cfg.get('vis_backend', 'open3d')
        common_kwargs = dict(
            hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
            obj_mesh=handobject.vis_obj_models[vis_idx],
            msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
            part_ids=handobject.hand_part_ids,
            grid_scale=self.cfg.msdf.scale, h=400, w=400, backend=backend)
        pred_img, _ = visualize_recon_hand_w_object(
            hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
            hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(), **common_kwargs)
        gt_rec_img, _ = visualize_recon_hand_w_object(
            hand_verts=gt_rec_hand_verts[vis_idx].detach().cpu().numpy(),
            hand_verts_mask=gt_rec_verts_mask[vis_idx].detach().cpu().numpy(), **common_kwargs)
        gt_img, _ = visualize_recon_hand_w_object(
            hand_verts=handobject.hand_verts[vis_idx].detach().cpu().numpy(),
            hand_verts_mask=handobject.hand_vert_mask[vis_idx].any(dim=0).detach().cpu().numpy(), **common_kwargs)
        return np.concatenate([gt_img, gt_rec_img, pred_img], axis=0)

    def reconstruct_from_latent(self, latent, multi_scale_obj_cond, obj_msdf_center):
        batch_size, n_grids = latent.shape[:2]
        latent = latent.reshape(batch_size*n_grids, -1)
        recon_lg_contact = self.grid_ae.decode(latent, multi_scale_obj_cond)
        recon_lg_contact = recon_lg_contact.view(batch_size, n_grids, -1, self.msdf_k ** 3).permute(0, 1, 3, 2)
        sample_contact = recon_lg_contact[..., 0] # * grid_contact_mask[:, :, None].float()
        sample_cse = recon_lg_contact[..., 1:]
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        # sample_contact[sample_contact < 0.03] = 0  ## maskout low contact prob

        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.hand_cse, None,
            sample_contact.reshape(batch_size, -1), sample_cse.reshape(batch_size, -1, self.cse_dim),
            grid_coords=grid_coords.reshape(batch_size, -1, 3),
            mask_th=self.msdf_k / 4,
            chunk_size=10
        )

        grid_contact = sample_contact.max(dim=-1)[0]

        return pred_hand_verts, pred_verts_mask, grid_contact
    
    def on_test_epoch_start(self):
        self.all_results = []
        self.cache_results = {}
        self.sample_joints = []

        ## Testing metrics
        self.runtime = 0
        if not self.debug:
            # Access logger.experiment to trigger WandbLogger's lazy wandb.init()
            # _ = self.logger.experiment
            # for metric in self.cfg.test.criteria:
            #     wandb.define_metric(metric, summary='mean')
            # Initialize W&B table for test images
            self.test_images_table = wandb.Table(columns=[
                "batch_idx", "obj_name", "surrounding_hands", "sampled_grasp",
                "sim_displacement", "penetration_depth", "contact_area", "intersection_volume"
            ])

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        Handles unfreezing pretrained weights at the specified epoch.
        """
        # self.pool = ThreadPoolExecutor(max_workers=min(self.cfg.train.batch_size, 16))
        self.pool = None

    def on_validation_epoch_start(self):
        # self.pool = ThreadPoolExecutor(max_workers=min(self.cfg.val.batch_size, 16))
        self.pool = None

    def on_train_epoch_end(self):
        if self.pool is not None:
            # self.pool.shutdown(wait=True)
            self.pool.close()
            self.pool.join()

    def on_validation_epoch_end(self):
        if self.pool is not None:
            # self.pool.shutdown(wait=True)
            self.pool.close()
            self.pool.join()

    def test_step(self, batch, batch_idx):
        self.grid_coords = self.grid_coords.to(self.device)
        self.mano_layer.to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=True)
        obj_hulls = getattr(self.trainer.datamodule, 'test_set').obj_hulls
        obj_name = batch['objName'][0]
        obj_hulls = obj_hulls[obj_name]
        obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').obj_info[obj_name]
        simp_obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh[obj_name]
        n_grids = batch['objMsdf'].shape[1]
        obj_mesh = trimesh.Trimesh(obj_mesh_dict['verts'], obj_mesh_dict['faces'])
        simp_obj_mesh = trimesh.Trimesh(simp_obj_mesh_dict['verts'], simp_obj_mesh_dict['faces'])


        if self.cfg.generator.model_type == 'gt':
            ## Test using gt contact grids.
            handobject.load_from_batch(batch)
            n_grids = handobject.obj_msdf.shape[1]
            obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:]
            recon_lg_contact = handobject.ml_contact
        else:
            ## Test the reconstrucion 
            n_samples = self.cfg.test.get('n_samples', 1)
            handobject.load_from_batch_obj_only(batch, n_samples, obj_template=obj_mesh, vis_obj_template=simp_obj_mesh, obj_hulls=obj_hulls)
            # lg_contact = handobject.ml_contact
            # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            # obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3

            obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k) # N x ...
            obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # N x 3
            obj_feat, multi_scale_obj_cond = self.grid_ae.encode_object(obj_msdf)
            # obj_pc = torch.cat([obj_msdf_center, obj_feat.unsqueeze(0)], dim=-1)
            obj_pc = handobject.obj_msdf

            ## 'x' only indicates the latent shape; latents are sampled inside the model
            input_data = {'x': torch.randn(n_samples, n_grids, self.cfg.ae.feat_dim, device=self.device), 'obj_pc': obj_pc.permute(0, 2, 1).to(self.device), 'obj_msdf': obj_msdf}

            def project_latent(latent):
                """Closure that captures obj context to project latent through hand mesh."""
                return self._project_latent(latent, n_grids, obj_msdf, obj_msdf_center, multi_scale_obj_cond, obj_mesh=handobject.vis_obj_models[0], part_ids=handobject.hand_part_ids)

            self.vis_geoms = []
            self._proj_step = 0
            samples = self.diffusion.sample(self.model, input_data, k=n_samples, proj_fn=None, progress=True)

            # Visualize hand geometries at every 100 steps along with object
            if self.vis_geoms:
                obj_geom = o3dmesh_from_trimesh(handobject.vis_obj_models[0], color=[0.7, 0.7, 0.7])
                all_geoms = []
                for i, hand_geom in enumerate(self.vis_geoms):
                    offset = np.array([i * 0.25, 0, 0])
                    h = deepcopy(hand_geom).translate(offset)
                    o = deepcopy(obj_geom).translate(offset)
                    all_geoms.extend([h, o])
                o3d.visualization.draw_geometries(all_geoms, window_name='Projection Progress (every 100 steps)')

            # sample_latent = samples ## B x latent
            latent = samples.reshape(n_samples*n_grids, -1)
            ## repeat the multi-scale obj cond here
            multi_scale_obj_cond = [cond.repeat(n_samples, *([1] * (cond.ndim - 1))) for cond in multi_scale_obj_cond]
            multi_scale_obj_cond.append(obj_feat.repeat(n_samples, 1))
            recon_lg_contact = self.grid_ae.decode(latent, multi_scale_obj_cond)
            # recon_lg_contact, mu, logvar = self.model(
            #     lg_contact.permute(0, 1, 5, 2, 3, 4), obj_msdf=obj_msdf, msdf_center=obj_msdf_center)
            # recon_lg_contact, z_e, obj_feat = self.grid_ae(
            #     lg_contact.view(n_samples*n_pts, self.msdf_k, self.msdf_k, self.msdf_k, -1).permute(0, 4, 1, 2, 3),
            #     obj_msdf=obj_msdf.unsqueeze(1), sample_posterior=False)
            recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x K x K x K x (1 + cse_dim)
            recon_lg_contact = recon_lg_contact.view(n_samples, n_grids, self.msdf_k, self.msdf_k, self.msdf_k, -1)
            # recon_lg_contact = lg_contact

            ## If masked with GT pt mask
            # recon_lg_contact = recon_lg_contact * handobject.obj_pt_mask[:, :, None, None, None, None]
            recon_lg_contact[..., 0][recon_lg_contact[..., 0] < 0.03] = 0  ## maskout low contact prob

            # handobject.load_from_batch_object_only(batch)
            # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            # recon_lg_contact = self.model.sample(obj_msdf, obj_msdf_center).permute(0, 1, 3, 4, 5, 2)

        pred_grid_contact = recon_lg_contact[..., 0].reshape(n_samples, -1)  # B x N x K^3
        obj_msdf_center = obj_msdf_center.repeat(n_samples, 1, 1)  # B x N x 3
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        pred_grid_cse = recon_lg_contact[..., 1:].reshape(n_samples, -1, self.cse_dim)
        pred_targetWverts = self.hand_cse.emb2Wvert(pred_grid_cse.view(n_samples, -1, self.cse_dim))

        if self.cfg.pose_optimizer.name == 'hand_ae':
            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                mask_th=self.msdf_k / 4,
            )
            recon_param, _ = self.hand_ae(pred_hand_verts.permute(0, 2, 1), mask=pred_verts_mask.unsqueeze(1), is_training=False)
            nrecon_trans, recon_pose, recon_betas = torch.split(recon_param, [3, 48, 10], dim=1)
            recon_trans = nrecon_trans * 0.2
            handV, handJ, _ = self.mano_layer(recon_pose, th_betas=recon_betas, th_trans=recon_trans)
        else:
            with torch.enable_grad():
                recon_hand_verts, recon_verts_mask = recover_hand_verts_from_contact(
                    self.hand_cse, None,
                    pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                    grid_coords=grid_coords.reshape(n_samples, -1, 3),
                    mask_th=self.msdf_k / 4,
                    chunk_size=10
                )
                obj_msdf_grid = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k) # (B*N) x 1 x k x k x k
                params, contact_mask = optimize_pose_wrt_local_grids(
                            self.mano_layer, grid_centers=obj_msdf_center, target_pts=grid_coords.view(n_samples, -1, 3),
                            dist2contact_fn=self.grid_dist_to_contact,
                            target_W_verts=pred_targetWverts, weights=pred_grid_contact, grid_sdfs=obj_msdf_grid.squeeze(1),
                            n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr,
                            recon_hand_verts=recon_hand_verts, recon_verts_mask=recon_verts_mask,
                            grid_scale=self.cfg.msdf.scale, w_repulsive=self.cfg.pose_optimizer.w_repulsive)
                mano_trans, global_pose, mano_pose, mano_shape = params

                # mano_trans, global_pose, mano_pose, mano_shape = optimize_pose_by_contact(
                #             self.mano_layer, grid_centers=obj_msdf_center, target_pts=grid_coords.view(n_samples, -1, 3),
                #             target_W_verts=pred_targetWverts, pred_contact=pred_grid_contact, dist2contact_fn=self.grid_dist_to_contact,
                #             n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr,
                #             grid_scale=self.cfg.msdf.scale, w_repulsive=self.cfg.pose_optimizer.w_repulsive)

            handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_name}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_name, 'hand_joints': handJ[i], 'obj_model': handobject.obj_models[0], 'obj_hulls': handobject.obj_hulls[0],
                       'idx': i} for i in range(handV.shape[0])]
            
        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')

        self.cache_results[obj_name] = {
            "hand_verts": handV,
            "hand_joints": handJ,
        }
        print('saved to cache_results')
        self.sample_joints.append(handJ)

        # Log raw per-sample metrics to wandb
        if not self.debug:
            # Option 1: Log each sample as individual rows (creates distributions in wandb)
            for i in range(len(next(iter(result.values())))):
                sample_metrics = {f"sample/{metric_name}": float(metric_values[i])
                                 for metric_name, metric_values in result.items()}
                wandb.log(sample_metrics, commit=False)

            # Option 2 (alternative): Use wandb Table for structured logging
            # table_data = [[obj_names[i]] + [float(result[m][i]) for m in result.keys()]
            #               for i in range(batch_size)]
            # table = wandb.Table(data=table_data, columns=["object"] + list(result.keys()))
            # wandb.log({f"test_metrics_batch_{batch_idx}": table})
        ## Visualization
        # for vis_idx in range(handV.shape[0]):
        # gt_geoms = handobject.get_vis_geoms(idx=vis_idx, obj_templates=obj_meshes)
        pred_ho = copy(handobject)
        pred_ho.hand_verts = torch.tensor(handV, dtype=torch.float32)
        pred_ho.hand_joints = torch.tensor(handJ, dtype=torch.float32)

        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.hand_cse, None,
            pred_grid_contact.reshape(n_samples, -1),
            pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
            grid_coords=grid_coords.reshape(n_samples, -1, 3),
            mask_th=self.msdf_k / 4,
        )

        # Visualize all samples
        backend = self.cfg.get('vis_backend', 'open3d')
        recon_imgs = []
        pred_imgs = []
        # for vis_idx in range(n_samples):
        if False:
            recon_img, pred_geoms = visualize_recon_hand_w_object(
                hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                obj_mesh=handobject.vis_obj_models[vis_idx],
                part_ids=handobject.hand_part_ids,
                msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                grid_scale=self.cfg.msdf.scale,
                h=400, w=400, backend=backend)
            recon_imgs.append(recon_img)

            if self.debug:
                ho_geoms = pred_ho.get_vis_geoms(idx=vis_idx)
                o3d.visualization.draw_geometries(pred_geoms + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g.translate((0, 0.25, 0)) for g in ho_geoms], window_name='Predicted Hand-Object')
            else:
                pred_img = pred_ho.vis_img(idx=vis_idx, h=400, w=400, backend=backend)
                pred_imgs.append(pred_img)

        # Concatenate all sample images horizontally
        recon_img_grid = np.concatenate(recon_imgs, axis=0)
        if not self.debug:
            pred_img_grid = np.concatenate(pred_imgs, axis=0)

        if hasattr(self.logger, 'experiment'):
            if hasattr(self.logger.experiment, 'add_image'):
                # TensorBoardLogger
                global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.test_dataloader()')) + batch_idx
                self.logger.experiment.add_image(f'test/Surrounding_hands', recon_img_grid, global_step, dataformats='HWC')
                # self.logger.experiment.add_image(f'test/Sampled_grasp', pred_img_grid, global_step, dataformats='HWC')
            elif hasattr(self.logger.experiment, 'log') and not self.debug:
                # WandbLogger - add row to table
                self.test_images_table.add_data(
                    batch_idx,
                    obj_name,
                    wandb.Image(recon_img_grid),
                    wandb.Image(pred_img_grid),
                    float(np.mean(result.get("Simulation Displacement", [0]))),
                    float(np.mean(result.get("Penetration Depth", [0]))),
                    float(np.mean(result.get("Contact Area", [0]))),
                    float(np.mean(result.get("Intersection Volume", [0]))),
                )
            # o3d.visualization.draw(pred_geoms)
            # o3d.visualization.draw(gt_geoms + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g for g in pred_geoms])

        return result
    
    def _project_latent(self, latent, n_grids, obj_msdf, obj_msdf_center, multi_scale_obj_cond, **kwargs):
        """
        Project latent through hand mesh fitting and re-encode.

        Args:
            latent: (n_samples, n_grids, feat_dim) diffusion latent
            n_grids: number of grids per sample
            obj_msdf: (N, 1, K, K, K) object MSDF
            obj_msdf_center: (1, N, 3) grid centers
            multi_scale_obj_cond: list of multi-scale object conditioning tensors
        Returns:
            proj_latent: (n_samples, n_grids, feat_dim) projected latent
        """
        K = self.cfg.msdf.kernel_size
        n_samples = latent.shape[0]

        # Decode latent to contact grid
        flat_latent = latent.reshape(n_samples * n_grids, -1)
        ms_obj_cond = [cond.repeat(n_samples, 1, 1, 1, 1) for cond in multi_scale_obj_cond]
        recon = self.grid_ae.decode(flat_latent, ms_obj_cond)  # (n_samples*n_grids, C, K, K, K)
        recon = recon.view(n_samples, n_grids, -1, K ** 3).permute(0, 1, 3, 2)  # (B, N, K^3, C)

        grid_contact = recon[..., 0].reshape(n_samples, -1)  # (B, N*K^3)
        grid_cse = recon[..., 1:].reshape(n_samples, -1, self.cse_dim)  # (B, N*K^3, cse_dim)
        grid_centers = obj_msdf_center.repeat(n_samples, 1, 1)  # (B, N, 3)
        grid_coords = grid_centers[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # (B, N, K^3, 3)
        grid_coords = grid_coords.reshape(n_samples, -1, 3)  # (B, N*K^3, 3)

        # Project through hand mesh
        proj_contact, proj_cse, proj_handV = self.contact_grid_projection(
            grid_contact, grid_cse, grid_coords, grid_centers, **kwargs
        )

        self._proj_step = getattr(self, '_proj_step', 0) + 1
        if self._proj_step % 100 == 0:
            hand_geom = o3dmesh(proj_handV[0].detach().cpu().numpy(), self.closed_mano_faces)
            self.vis_geoms.append(hand_geom)

        # Re-encode projected contact grid back to latent
        proj_lg = torch.cat([
            proj_contact.reshape(n_samples, n_grids, K ** 3, 1),
            proj_cse.reshape(n_samples, n_grids, K ** 3, self.cse_dim)
        ], dim=-1)  # (B, N, K^3, C)
        proj_lg = rearrange(proj_lg, 'b n (k1 k2 k3) c -> (b n) c k1 k2 k3', k1=K, k2=K, k3=K)
        posterior, _, _ = self.grid_ae.encode(proj_lg, obj_msdf.repeat(n_samples, 1, 1, 1, 1))
        proj_latent = posterior.sample().view(n_samples, n_grids, -1)
        return proj_latent

    def contact_grid_projection(self, grid_contact, grid_cse, grid_coords, grid_centers, **kwargs):
        B = grid_contact.shape[0]
        N_total = grid_contact.shape[1]  # N * K^3
        device = grid_contact.device
        cse_dim = grid_cse.shape[-1]

        # --- Filter to contact-active points to speed up emb2Wvert ---
        # emb2Wvert computes cdist (B, n_pts, 1538) which is the bottleneck.
        # By filtering from ~65k to only contact-active points, this becomes tractable.
        contact_th = 0
        active_mask = grid_contact > contact_th  # (B, N_total)
        n_active_per_sample = active_mask.sum(dim=1)  # (B,)
        max_active = n_active_per_sample.max().item()

        if max_active == 0:
            return torch.zeros_like(grid_contact), torch.zeros_like(grid_cse)

        # Gather active points into padded tensors for batched emb2Wvert
        active_cse = torch.zeros(B, max_active, cse_dim, device=device)
        active_contact = torch.zeros(B, max_active, device=device)
        active_coords = torch.zeros(B, max_active, 3, device=device)
        for b in range(B):
            n = n_active_per_sample[b].item()
            if n > 0:
                idx = active_mask[b].nonzero(as_tuple=True)[0]
                active_cse[b, :n] = grid_cse[b, idx]
                active_contact[b, :n] = grid_contact[b, idx]
                active_coords[b, :n] = grid_coords[b, idx]

        # emb2Wvert on filtered points only
        targetWverts = self.hand_cse.emb2Wvert(active_cse, None)  # (B, max_active, 778)
        weight = (targetWverts * active_contact.unsqueeze(-1)).transpose(-1, -2)  # (B, 778, max_active)
        recon_verts_mask = torch.sum(weight, dim=-1) > 2  # (B, 778)
        weight[recon_verts_mask] = weight[recon_verts_mask] / torch.sum(weight[recon_verts_mask], dim=-1, keepdim=True)
        recon_hand_verts = weight @ active_coords  # (B, 778, 3)

        recon_param, _ = self.hand_ae(recon_hand_verts.permute(0, 2, 1), mask=recon_verts_mask.unsqueeze(1), is_training=False)
        nrecon_trans, recon_pose, recon_betas = torch.split(recon_param, [3, 48, 10], dim=1)
        recon_trans = nrecon_trans * 0.2
        handV, handJ, _ = self.mano_layer(recon_pose, th_betas=recon_betas, th_trans=recon_trans)

        # _, contact_geoms = visualize_recon_hand_w_object(hand_verts=recon_hand_verts[0].detach().cpu().numpy(),
        #                             # hand_verts_mask=handobject.hand_vert_mask[vis_idx].detach().cpu().numpy(),
        #                             hand_verts_mask=recon_verts_mask[0].detach().cpu().numpy(),
        #                             hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
        #                             obj_mesh=kwargs.get('obj_mesh', None),
        #                             part_ids=kwargs.get('part_ids', None),
        #                             h=400, w=400)

        ## Project back to contact grid
        hand_faces = self.mano_layer.th_faces
        hand_cse = self.hand_cse.embedding_tensor  # (778, cse_dim)
        K = self.cfg.msdf.kernel_size
        normalized_coords = get_grid(kernel_size=K, device=device).reshape(-1, 3).float()

        proj_lg_contact = torch.zeros_like(grid_contact)  # (B, N*K^3)
        proj_lg_cse = torch.zeros_like(grid_cse)  # (B, N*K^3, cse_dim)

        for b in range(B):
            grid_distance, verts_mask_b, grid_mask, ho_dist, nn_face_idx, nn_point = calc_local_grid_all_pts_gpu(
                contact_points=grid_centers[b],  # (N, 3)
                normalized_coords=normalized_coords,
                hand_verts=handV[b],
                faces=hand_faces,
                kernel_size=K,
                grid_scale=self.cfg.msdf.scale,
            )

            if grid_mask.any():
                nn_face_idx_flat = nn_face_idx.reshape(-1)
                nn_point_flat = nn_point.reshape(-1, 3)

                nn_vert_idx = hand_faces[nn_face_idx_flat]
                face_verts = handV[b, nn_vert_idx]
                face_cse = hand_cse[nn_vert_idx]

                w = torch.linalg.inv(face_verts.transpose(1, 2)) @ nn_point_flat.unsqueeze(-1)
                w = torch.clamp(w, 0, 1)
                w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)

                grid_hand_cse = torch.sum(face_cse * w, dim=1)

                flat_mask = grid_mask.unsqueeze(1).expand(-1, K ** 3).reshape(-1)
                proj_lg_contact[b, flat_mask] = self.grid_dist_to_contact(grid_distance.reshape(-1))
                proj_lg_cse[b, flat_mask] = grid_hand_cse
        

        return proj_lg_contact, proj_lg_cse, handV

    def on_test_epoch_end(self):
        final_metrics = {}

        # Compute statistics for all metrics from test results
        for m in self.cfg.test.criteria:
            if "Entropy" not in m and "Cluster Size" not in m:
                all_metrics = np.concatenate([res[m] for res in self.all_results], axis=0)
                # Log comprehensive statistics for each metric
                final_metrics[f"{m}/mean"] = np.mean(all_metrics).item()
                final_metrics[f"{m}/std"] = np.std(all_metrics).item()
                final_metrics[f"{m}/min"] = np.min(all_metrics).item()
                final_metrics[f"{m}/max"] = np.max(all_metrics).item()
                # final_metrics[f"{m}/median"] = np.median(all_metrics).item()
                # final_metrics[f"{m}/p25"] = np.percentile(all_metrics, 25).item()
                # final_metrics[f"{m}/p75"] = np.percentile(all_metrics, 75).item()

        ## Calculate diversity
        sample_joints = np.concatenate(self.sample_joints, axis=0)
        # run_time = np.concatenate(run_time, axis=0)
        entropy, cluster_size, entropy_2, cluster_size_2 = calc_diversity(sample_joints)
        final_metrics.update({
            "Entropy": entropy.item(),
            "Cluster Size": cluster_size.item(),
            "Canonical Entropy": entropy_2.item(),
            "Canonical Cluster Size": cluster_size_2.item()
        })

        # Log final metrics and test images table to wandb
        if not self.debug:
            wandb.log(final_metrics, commit=False)
            wandb.log({"test/images": self.test_images_table})

        ## Save the results to study detailed criteria.
        for k, v in self.cache_results.items():
            with open(osp.join('tmp', 'grab', f'result_{k}.pkl'), 'wb') as f:
                pickle.dump(v, f)

        # return final_metrics
    
        ## GT visualization
        # vis_idx = 0
        # gt_geoms = visualize_local_grid_with_hand(
        #         batch['localGrid'][vis_idx].cpu().numpy(), hand_verts=batch['nHandVerts'][vis_idx].cpu().numpy(),
        #         hand_faces=self.mano_layer.th_faces.cpu().numpy(), hand_cse=self.hand_cse.embedding_tensor.detach().cpu().numpy(),
        #         kernel_size=self.cfg.msdf.kernel_size, grid_scale=self.cfg.msdf.scale
        #     )
        # o3d.visualization.draw_geometries(gt_geoms, window_name='GT Local Grid Visualization')

    def configure_optimizers(self):
        if self.cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.cfg.optimizer == 'asam':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(self.model.parameters(), base_optimizer=base_optimizer, lr=self.lr, momentum=0.9, adaptive=True)

        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.8
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # def lr_scheduler_step(self, scheduler, metric):
    #     if not self.cfg.optimizer == 'asam':
    #         super().lr_scheduler_step(scheduler, metric)
    #     else:
    #         # Forward the scheduler step to the base_optimizer,
    #         # since SAM wraps it and shares param_groups
    #         scheduler.step()