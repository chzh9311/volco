import os
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import open3d as o3d
import trimesh
from copy import copy, deepcopy
import numpy as np
import pickle
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
import wandb
from matplotlib import pyplot as plt

from common.manopth.manopth.manolayer import ManoLayer
from common.model.pose_optimizer import optimize_pose_by_contact, optimize_pose_wrt_local_grids
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.geometry import GridDistanceToContact
from common.utils.physics import StableLoss
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh, geom_to_img, visualize_recon_hand_w_object, visualize_grid_contact, geom_to_video_o3d
from common.msdf.utils.msdf import get_grid, calc_local_grid_all_pts_gpu
from common.evaluation.eval_fns import calculate_metrics, calc_diversity
from einops import rearrange


class VolCoDiffTrainer(L.LightningModule):
    """
    The Lightning trainer for latent diffusion over the VolumeVAE local-grid contact latents.

    dual=True:  DualUNetModel denoises the hand latent (HandVAE) together with the per-grid contact
                latents, concatenated as B x (hand_latent_dim + N*latent_dim).
    dual=False: UNetModel denoises the per-grid contact latents only, B x N x latent_dim; the hand is
                recovered from the decoded contact at test time.
    """
    def __init__(self, volume_vae, hand_ae, model, diffusion, cfg):
        super().__init__()
        self.dual = cfg.generator.model_name == 'dual_latent_diffusion'
        self.volume_vae = volume_vae
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
            self._load_pretrained_weights(cfg.ae.get('pretrained_weight', None), target_prefix='volume_vae')
            self.volume_vae.eval()
            self.volume_vae.requires_grad_(False)

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

        if self.dual:
            self.hand_ae = hand_ae
            if cfg.run_phase == 'train':
                self._load_pretrained_weights(cfg.hand_ae.get('pretrained_weight', None), target_prefix='hand_ae')

            def global2local(hand_latent, grid_centers):
                return self._global2local(hand_latent, grid_centers)

            object.__setattr__(self.model, 'global2local_fn', global2local)
        else:
            ## Only used at test time to fit the hand; keep it out of the state dict so that
            ## contact-only checkpoints hold just volume_vae and model.
            object.__setattr__(self, 'hand_ae', hand_ae)
        self.hand_ae.eval().requires_grad_(False)
        self.pool = None # Pool(processes=min(self.cfg.test.batch_size, 16))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                batch[key] = value.float()
        return batch

    def _load_pretrained_weights(self, checkpoint_path, target_prefix='volume_vae'):
        """
        Load pretrained weights from a Lightning checkpoint.
        Only loads weights for layers that exist in both the checkpoint and current model.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file
            target_prefix: Prefix to add to checkpoint keys when loading into current model
                          (e.g., 'volume_vae' will map 'model.encoder.xxx' to 'volume_vae.encoder.xxx')
        """
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

        # Set volume_vae BatchNorm layers to eval mode to prevent running stats drift
        bn_count = 0
        for module in self.volume_vae.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
                # Disable running stats updates during forward pass
                module.track_running_stats = False
                bn_count += 1

        print(f"Frozen {frozen_count} pretrained parameters in volume_vae")
        print(f"Set {bn_count} BatchNorm layers to eval mode (prevents running stats drift)")

    # def on_fit_start(self):
    #     """
    #     Called after checkpoint restoration but before training starts.
    #     Freezes pretrained weights here to ensure they remain frozen even when resuming.
    #     """
    #     if self.cfg.run_phase == 'train' and hasattr(self, 'pretrained_keys'):
    #         self._freeze_pretrained_weights()

    def _global2local(self, hand_latent, obj_msdf):
        _, handV, handJ = self.hand_ae.decode(hand_latent)
        hand_faces = self.mano_layer.th_faces

        batch_size, n_grids, _ = obj_msdf.shape
        k = self.cfg.msdf.kernel_size
        grid_centers = obj_msdf[:, :, k**3:]
        grid_msdf = rearrange(obj_msdf[:, :, :k**3], 'b n (k1 k2 k3) -> (b n) 1 k1 k2 k3', k1=k, k2=k, k3=k)
        proj_lg_contact = torch.zeros(batch_size, k**3 * grid_centers.shape[1]).to(self.device)  # (B, N*K^3)
        proj_lg_cse = torch.zeros(batch_size, k**3 * grid_centers.shape[1], self.cse_dim).to(self.device)  # (B, N*K^3, cse_dim)

        for b in range(batch_size):
            grid_distance, verts_mask_b, grid_mask, ho_dist, nn_face_idx, nn_point = calc_local_grid_all_pts_gpu(
                contact_points=grid_centers[b],  # (N, 3)
                normalized_coords=self.normalized_grid_coords.view(-1, 3).to(self.device),  # (K^3, 3)
                hand_verts=handV[b],
                faces=hand_faces,
                kernel_size=k,
                grid_scale=self.cfg.msdf.scale,
                apply_grid_mask=not self.cfg.ae.use_noncontact_grids
            )

            if grid_mask.any():
                nn_face_idx_flat = nn_face_idx.reshape(-1)
                nn_point_flat = nn_point.reshape(-1, 3)

                nn_vert_idx = hand_faces[nn_face_idx_flat]
                face_verts = handV[b, nn_vert_idx]
                face_cse = self.hand_cse.vert2emb(nn_vert_idx)

                A = face_verts.transpose(1, 2) + 1e-6 * torch.eye(3, device=face_verts.device).unsqueeze(0)
                w = torch.linalg.solve(A, nn_point_flat.unsqueeze(-1))
                w = torch.clamp(w, 0, 1)
                w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-8)

                grid_hand_cse = torch.sum(face_cse * w, dim=1)

                flat_mask = grid_mask.unsqueeze(1).expand(-1, k ** 3).reshape(-1)
                proj_lg_contact[b, flat_mask] = self.grid_dist_to_contact(grid_distance.reshape(-1))
                proj_lg_cse[b, flat_mask] = grid_hand_cse

        proj_lg = torch.cat([proj_lg_contact.unsqueeze(-1), proj_lg_cse], dim=-1) # (B, N*K^3, 1+cse_dim)
        proj_lg = rearrange(proj_lg, 'b (n k1 k2 k3) c -> (b n) c k1 k2 k3', k1=k, k2=k, k3=k)
        posterior, _, _ = self.volume_vae.encode(proj_lg, grid_msdf)
        local_latent = posterior.sample().view(batch_size, n_grids, -1) # (B, N, latent_dim)
        return local_latent

    def _split_latent(self, x, n_grids):
        """
        Split a diffusion latent into the hand latent and the per-grid contact latents.
        Returns (hand_latent, grid_latent): B x hand_latent_dim (None if not dual), B x N x latent_dim
        """
        if not self.dual:
            return None, x
        d_y = self.cfg.generator.unet.d_y
        return x[:, :d_y], x[:, d_y:].reshape(x.shape[0], n_grids, -1)

    def _cache_path(self, obj_name):
        prefix = 'VolCoDiff' if self.dual else 'result'
        return osp.join('tmp', 'grab', f'{prefix}_{obj_name}.pkl')

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
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, apply_grid_mask=not self.cfg.ae.use_noncontact_grids)
        handobject.load_from_batch(batch, pool=self.pool)

        lg_contact = handobject.ml_contact
        batch_size, n_grids = lg_contact.shape[:2]
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k)

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        ## First process all grids separately using VolumeVAE
        lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> (b n) c k1 k2 k3')
        posterior, obj_feat, multi_scale_obj_cond = self.volume_vae.encode(lg_contact, obj_msdf)
        gt_contact_latent = posterior.sample().view(batch_size, n_grids, -1) # n_dim
        obj_pc = handobject.obj_msdf

        if self.dual:
            gt_hand_latent = self.hand_ae.encode(handobject.hand_verts.permute(0, 2, 1)).sample()
            x = torch.cat([gt_hand_latent, gt_contact_latent.view(batch_size, -1)], dim=-1) # B x (hand_latent_dim + n_grids*n_dim)
        else:
            x = gt_contact_latent # B x n_grids x n_dim
        input_data = {'x': x, 'obj_pc': obj_pc.permute(0, 2, 1), 'obj_msdf': handobject.obj_msdf}

        # Check for NaN in input data
        for key, value in input_data.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in input_data['{key}'] at batch_idx {batch_idx}")

        if self.dual:
            losses, model_output, recon_LVClatent = self.diffusion.training_losses(self.model, input_data,
                                                    hand_ae=self.hand_ae, gt_handV=handobject.hand_verts, gt_handJ=handobject.hand_joints)
        else:
            losses, model_output = self.diffusion.training_losses(self.model, input_data)

        _, pred_grid_latent = self._split_latent(model_output, n_grids) # B x n_grids x n_dim
        if self.dual:
            losses['difference_loss'] = F.mse_loss(pred_grid_latent, recon_LVClatent.reshape(pred_grid_latent.shape), reduction='mean')

        ## For stable loss
        if self.loss_weights.get('stable_loss', 0) > 0:
            recon_grid_contact = self.volume_vae.decode(pred_grid_latent.reshape(batch_size * n_grids, -1), multi_scale_obj_cond)  # (B*N) x c x k x k x k
            lgc = rearrange(recon_grid_contact[:, 0], '(b n) k1 k2 k3 -> b (n k1 k2 k3)', b=batch_size, n=n_grids)
            lgc = torch.where(lgc < 0.05, torch.zeros_like(lgc), lgc)
            sdf_vals = handobject.obj_msdf[:, :, :self.msdf_k**3]        # (B, N, K^3)
            sdf_flat = sdf_vals.reshape(batch_size, n_grids * self.msdf_k**3)
            sdf_flat = sdf_flat * self.msdf_scale * np.sqrt(3)
            centres = handobject.obj_msdf[:, :, self.msdf_k**3:]          # (B, N, 3)
            sdf_grad = handobject.msdf_grad.reshape(batch_size, n_grids * self.msdf_k**3, 3)
            n_adj_pt = handobject.n_adj_pt.view(batch_size, n_grids * self.msdf_k**3)
            all_pts = centres.unsqueeze(2) + handobject.normalized_coords[None, None, :, :] * self.msdf_scale
            ## Assume gravity direction is always (0, 0, -1), since there're some tolerance for penetration error.
            stable_loss = self.stable_loss(sdf_flat, all_pts.view(batch_size, -1, 3), lgc, sdf_grad, n_adj_pt,
                                    obj_mass=handobject.obj_mass, gravity_direction=torch.FloatTensor([[0, 0, -1]]).to(self.device),
                                    J=handobject.obj_inertia)
            losses['stable_loss'] = stable_loss.mean()

        total_loss = sum([losses[k] * self.loss_weights[k] for k in losses.keys()])
        losses['total_loss'] = total_loss

        loss_dict = {f'{stage}/{k}': v for k, v in losses.items()}
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        ## Also sample and reconstruct
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            vis_idx = 0
            vis_data = {k: v[vis_idx:vis_idx+1] for k, v in input_data.items()}
            condition = self.model.condition(vis_data)
            samples = self.diffusion.p_sample_loop(self.model, vis_data['x'].shape, condition, clip_denoised=False, progress=True)
            simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
            rot = batch['aug_rot'][vis_idx].cpu().numpy() if 'aug_rot' in batch else np.eye(3)
            obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
                            for i, name in enumerate(batch['objName'])]
            handobject._load_templates(idx=vis_idx, obj_templates=obj_templates)
            hand_latent, grid_latent = self._split_latent(samples, n_grids)

            vis_ms_obj_cond = [c[vis_idx*n_grids:(vis_idx+1)*n_grids] for c in multi_scale_obj_cond]
            vis_obj_msdf_center = obj_msdf_center[vis_idx:vis_idx+1]

            pred_hand_verts, pred_verts_mask, pred_grid_contact = self.reconstruct_from_latent(grid_latent, vis_ms_obj_cond, vis_obj_msdf_center)
            gt_rec_hand_verts, gt_rec_verts_mask, gt_rec_grid_contact = self.reconstruct_from_latent(gt_contact_latent[vis_idx:vis_idx+1], vis_ms_obj_cond, vis_obj_msdf_center)

            contact_img = self._visualize_contact_comparison(
                vis_obj_msdf_center, pred_grid_contact, gt_rec_grid_contact, handobject, vis_idx)
            img = self._visualize_hand_comparison(
                pred_hand_verts, pred_verts_mask, gt_rec_hand_verts, gt_rec_verts_mask,
                handobject, vis_obj_msdf_center, rot, vis_idx)
            vis_imgs = {f'{stage}/GT_vs_GTRec_vs_sampled_contact': contact_img,
                        f'{stage}/GT_vs_GTRec_vs_sampled_hand': img}
            if self.dual:
                _, recon_handV, recon_handJ = self.hand_ae.decode(hand_latent)
                vis_imgs[f'{stage}/GT_vs_sampled_full_hand'] = self.visualize_full_hand_comparison(
                    recon_handV[vis_idx], handobject.hand_verts[vis_idx], handobject.vis_obj_models[vis_idx])

            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    for name, vis_img in vis_imgs.items():
                        self.logger.experiment.add_image(name, vis_img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    for name, vis_img in vis_imgs.items():
                        self.logger.experiment.log({name: wandb.Image(vis_img)}, step=self.global_step)
        return total_loss

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

    def visualize_full_hand_comparison(self, pred_handV, gt_handV, obj_template):
        hand_faces = self.mano_layer.th_faces.detach().cpu().numpy()
        pred_handV = pred_handV.detach().cpu().numpy()
        gt_handV = gt_handV.detach().cpu().numpy()
        pred_mesh = o3dmesh(pred_handV, hand_faces, color=[0.8, 0.7, 0.6])
        gt_mesh = o3dmesh(gt_handV, hand_faces, color=[0.2, 0.4, 0.8])
        obj_mesh = o3dmesh_from_trimesh(obj_template, color=[0.7, 0.7, 0.7])
        pred_img = geom_to_img([pred_mesh, obj_mesh], w=400, h=400, scale=0.5, half_range=0.12)
        gt_img = geom_to_img([gt_mesh, obj_mesh], w=400, h=400, scale=0.5, half_range=0.12)
        return np.concatenate([gt_img, pred_img], axis=0)

    def reconstruct_from_latent(self, latent, multi_scale_obj_cond, obj_msdf_center):
        batch_size, n_grids = latent.shape[:2]
        latent = latent.reshape(batch_size*n_grids, -1)
        recon_lg_contact = self.volume_vae.decode(latent, multi_scale_obj_cond)
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

    def on_train_epoch_start(self):
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
        # if batch_idx < 4:  # TODO: temporary skip for debugging
        #     return {}
        self.grid_coords = self.grid_coords.to(self.device)
        self.mano_layer.to(self.device)
        self.hand_ae.to(self.device)  # not a registered submodule when not dual
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=True, apply_grid_mask=not self.cfg.ae.use_noncontact_grids)
        obj_hulls = getattr(self.trainer.datamodule, 'test_set').obj_hulls
        obj_name = batch['objName'][0]
        obj_hulls = obj_hulls[obj_name]
        obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').obj_info[obj_name]
        simp_obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh[obj_name]
        n_grids = batch['objMsdf'].shape[1]
        obj_mesh = trimesh.Trimesh(obj_mesh_dict['verts'], obj_mesh_dict['faces'])
        simp_obj_mesh = trimesh.Trimesh(simp_obj_mesh_dict['verts'], simp_obj_mesh_dict['faces'])

        ## Test the reconstrucion
        n_samples = self.cfg.test.get('n_samples', 1)
        handobject.load_from_batch_obj_only(batch, n_samples, obj_template=obj_mesh, vis_obj_template=simp_obj_mesh, obj_hulls=obj_hulls)

        use_cache = self.cfg.test.get('use_cache', False)
        cache_path = self._cache_path(obj_name)
        if use_cache:
            # Reuse previously saved hand results instead of rerunning diffusion + pose optimization
            if not osp.exists(cache_path):
                raise FileNotFoundError(
                    f"use_cache=True but cached result not found: {cache_path}. "
                    f"Run a pass with use_cache=False first to populate the cache."
                )
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            handV, handJ = cached['hand_verts'], cached['hand_joints']
            # No diffusion/optimization performed on this step
            self.diffusion_times.append(0.0)
            self.vae_times.append(0.0)
            self.optimization_times.append(0.0)
            self._test_step_metrics(batch_idx, obj_name, handV, handJ, handobject)
            return

        obj_msdf_grid = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k) # (B*N) x 1 x k x k x k
        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x N x 3
        obj_feat, multi_scale_obj_cond = self.volume_vae.encode_object(obj_msdf_grid)
        obj_pc = handobject.obj_msdf

        if self.dual:
            noise = torch.randn(n_samples, n_grids * self.cfg.ae.feat_dim + self.hand_ae.latent_dim, device=self.device)
        else:
            noise = torch.randn(n_samples, n_grids, self.cfg.ae.feat_dim, device=self.device)
        ## 'x' only indicates the latent shape; latents are sampled inside the model
        input_data = {'x': noise, 'obj_pc': obj_pc.permute(0, 2, 1).to(self.device), 'obj_msdf': handobject.obj_msdf}

        def project_latent(latent):
            """Closure that captures obj context to project latent through hand mesh."""
            return self._project_latent(latent, n_grids, obj_msdf_grid, obj_msdf_center, multi_scale_obj_cond, obj_mesh=handobject.obj_models[0], part_ids=handobject.hand_part_ids)

        self.vis_geoms = []
        self._proj_step = 0
        if self.device.type == 'cuda':
            _t0, _t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            _t0.record()
            samples = self.diffusion.sample(self.model, input_data, k=n_samples, proj_fn=None, progress=True)
            _t1.record()
            torch.cuda.synchronize()
            self.diffusion_times.append(_t0.elapsed_time(_t1) / 1000.0)  # ms -> s
        else:
            _t0 = time.perf_counter()
            samples = self.diffusion.sample(self.model, input_data, k=n_samples, proj_fn=None, progress=True)
            self.diffusion_times.append(time.perf_counter() - _t0)

        hand_latent, grid_latent = self._split_latent(samples, n_grids)

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

        _vae_t0 = time.perf_counter()
        grid_latent = grid_latent.reshape(n_samples*n_grids, -1)
        ## repeat the multi-scale obj cond here
        multi_scale_obj_cond = [cond.repeat(n_samples, 1, 1, 1, 1) for cond in multi_scale_obj_cond]
        multi_scale_obj_cond.append(obj_feat.repeat(n_samples, 1))
        recon_lg_contact = self.volume_vae.decode(grid_latent, multi_scale_obj_cond)
        recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x K x K x K x (1 + cse_dim)
        recon_lg_contact = recon_lg_contact.view(n_samples, n_grids, self.msdf_k, self.msdf_k, self.msdf_k, -1)
        recon_lg_contact[..., 0][recon_lg_contact[..., 0] < self.cfg.pose_optimizer.contact_th] = 0  ## maskout low contact prob

        pred_grid_contact = recon_lg_contact[..., 0].reshape(n_samples, -1)  # B x N x K^3
        obj_msdf_center = obj_msdf_center.repeat(n_samples, 1, 1)  # B x N x 3
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        pred_grid_cse = recon_lg_contact[..., 1:].reshape(n_samples, -1, self.cse_dim)
        pred_targetWverts = self.hand_cse.emb2Wvert(pred_grid_cse.view(n_samples, -1, self.cse_dim))
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.vae_times.append(time.perf_counter() - _vae_t0)

        pose_optimizer = self.cfg.pose_optimizer.name
        if not self.dual and pose_optimizer != 'hand_ae':
            ## No sampled hand latent to initialize from; fit the hand to the contact only.
            pose_optimizer = 'lg_base'
        init_handV = init_handJ = contact_mask = None

        _opt_t0 = time.perf_counter()
        if pose_optimizer == 'hand_ae':
            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                mask_th= 2 if self.msdf_k == 8 else 2,
                chunk_size=10
            )
            recon_param, _ = self.hand_ae(pred_hand_verts.permute(0, 2, 1), mask=pred_verts_mask.unsqueeze(1), is_training=False)
            nrecon_trans, recon_pose, recon_betas = torch.split(recon_param, [3, 48, 10], dim=1)
            recon_trans = nrecon_trans * 0.2
            handV, handJ, _ = self.mano_layer(recon_pose, th_betas=recon_betas, th_trans=recon_trans)
        elif pose_optimizer == 'lg_base':
            recon_hand_verts, recon_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                mask_th= 2 if self.msdf_k == 8 else 2,
                chunk_size=10
            )
            if self.dual:
                recon_params, init_handV, init_handJ = self.hand_ae.decode(hand_latent)
            with torch.enable_grad():
                params, contact_mask = optimize_pose_wrt_local_grids(
                            self.mano_layer, grid_centers=obj_msdf_center, target_pts=grid_coords.view(n_samples, -1, 3),
                            target_W_verts=pred_targetWverts, weights=pred_grid_contact, grid_sdfs=obj_msdf_grid.squeeze(1),
                            dist2contact_fn=self.grid_dist_to_contact, recon_hand_verts=recon_hand_verts, recon_verts_mask=recon_verts_mask,
                            n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr,
                            grid_scale=self.cfg.msdf.scale, w_repulsive=self.cfg.pose_optimizer.w_repulsive)
                mano_trans, global_pose, mano_pose, mano_shape = params

            handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        elif pose_optimizer == 'hybrid':
            recon_hand_verts, recon_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                mask_th= 2 if self.msdf_k == 8 else 2,
                chunk_size=10
            )
            recon_params, init_handV, init_handJ = self.hand_ae.decode(hand_latent)
            # recon_params = self.hand_ae.decoder(hand_latent)
            with torch.enable_grad():
                params, contact_mask = optimize_pose_wrt_local_grids(
                            self.mano_layer, grid_centers=obj_msdf_center, target_pts=grid_coords.view(n_samples, -1, 3),
                            target_W_verts=pred_targetWverts, weights=pred_grid_contact, grid_sdfs=obj_msdf_grid.squeeze(1),
                            dist2contact_fn=self.grid_dist_to_contact, recon_hand_verts=recon_hand_verts, recon_verts_mask=recon_verts_mask,
                            n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr,
                            grid_scale=self.cfg.msdf.scale, w_repulsive=self.cfg.pose_optimizer.w_repulsive,
                            w_reg_loss=self.cfg.pose_optimizer.w_regularization, init_pose=recon_params)
                mano_trans, global_pose, mano_pose, mano_shape = params

            handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
            # handV, handJ = init_handV, init_handJ
        else:
            recon_params, handV, handJ = self.hand_ae.decode(hand_latent)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.optimization_times.append(time.perf_counter() - _opt_t0)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        result = self._test_step_metrics(batch_idx, obj_name, handV, handJ, handobject)

        ## Visualization
        pred_ho = copy(handobject)
        pred_ho.hand_verts = torch.tensor(handV, dtype=torch.float32)
        pred_ho.hand_joints = torch.tensor(handJ, dtype=torch.float32)

        init_ho = None
        if init_handV is not None:
            init_ho = copy(handobject)
            init_ho.hand_verts = torch.tensor(init_handV, dtype=torch.float32)
            init_ho.hand_joints = torch.tensor(init_handJ, dtype=torch.float32)

        if pose_optimizer != 'hybrid':
            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1),
                pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                mask_th= 2 if self.msdf_k == 8 else 2,
                chunk_size=10
            )
        else:
            pred_hand_verts, pred_verts_mask = recon_hand_verts, recon_verts_mask

        vis_result = self.cfg.test.get('vis_result', True)

        if vis_result:
            # One turntable video per sample: surrounding | predicted | contact_mask | initial
            # (contact_mask only with lg_base/hybrid; initial only with a sampled hand latent)
            vid_side = self.cfg.test.get('vis_video_side', 480)
            vid_dir = osp.join('tmp', 'process_vid', f'batch_{batch_idx:04d}_{obj_name}')
            for vis_idx in range(n_samples):
                _, recon_geoms = visualize_recon_hand_w_object(
                    hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                    hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                    hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                    obj_mesh=handobject.vis_obj_models[vis_idx],
                    part_ids=handobject.hand_part_ids,
                    msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                    grid_scale=self.cfg.msdf.scale,
                    render=False)

                # Same hand + object geometries as HandObject.vis_img
                pred_geoms = [g for g in pred_ho.get_vis_geoms(idx=vis_idx) if g['name'] in ['hand', 'object']]
                panels = [recon_geoms, pred_geoms]

                if contact_mask is not None:
                    _, contact_geoms = visualize_grid_contact(
                        contact_pts=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                        pt_contact=contact_mask[vis_idx].detach().cpu().numpy().astype(float),
                        grid_scale=self.cfg.msdf.scale,
                        obj_mesh=handobject.vis_obj_models[vis_idx],
                        w=vid_side, h=vid_side, render=False)
                    panels.append(contact_geoms)

                if init_ho is not None:
                    panels.append([g for g in init_ho.get_vis_geoms(idx=vis_idx) if g['name'] in ['hand', 'object']])

                geom_to_video_o3d(panels, osp.join(vid_dir, f'sample_{vis_idx:02d}.mp4'), w=vid_side, h=vid_side,
                                  n_frames=self.cfg.test.get('vis_video_frames', 90))

        if hasattr(self.logger, 'experiment') and not self.debug:
            metric_row = [
                float(np.mean(result.get("Simulation Displacement", [0]))),
                float(np.mean(result.get("Penetration Depth", [0]))),
                float(np.mean(result.get("Intersection Volume", [0]))),
                float(np.mean(result.get("Contact Area", [0]))),
            ]
            if hasattr(self.logger.experiment, 'log'):
                # WandbLogger
                self.test_images_table.add_data(batch_idx, obj_name, *metric_row)

        return result

    def _test_step_metrics(self, batch_idx, obj_name, handV, handJ, handobject):
        """Compute grasp quality metrics for a batch of predicted hands and record them.

        Shared by the normal (diffusion + optimization) path and the use_cache path.
        handV/handJ are numpy arrays of shape (n_samples, ...).
        Returns the per-sample metric dict.
        """
        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_name}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_name, 'hand_joints': handJ[i], 'obj_model': handobject.obj_models[0], 'obj_hulls': handobject.obj_hulls[0],
                       'idx': i} for i in range(handV.shape[0])]

        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')

        self.cache_results[obj_name] = {
            "hand_verts": handV,
            "hand_joints": handJ,
        }

        raw_depths = result.pop("Penetration Depth Raw", [])
        vert_ids = result.pop("Penetration Depth Vert IDs", [])
        if obj_name not in self.raw_depth_data:
            self.raw_depth_data[obj_name] = {'raw_depth': [], 'penetr_vert_ids': []}
        self.raw_depth_data[obj_name]['raw_depth'].append(raw_depths)
        self.raw_depth_data[obj_name]['penetr_vert_ids'].append(vert_ids)
        for metric_name, metric_vals in result.items():
            self.raw_depth_data[obj_name].setdefault(metric_name, [])
            self.raw_depth_data[obj_name][metric_name].extend(metric_vals.tolist())

        # Print average of all metrics
        avg_metrics = {k: v.mean() for k, v in result.items()}
        print(f"Average metrics: {avg_metrics}")

        self.all_results.append(result)
        self.sample_joints.append(handJ)

        # Log per-sample scalar metrics to wandb (skip multi-dim arrays like Part Intersection Volumes)
        if not self.debug:
            scalar_result = {k: v for k, v in result.items()
                             if isinstance(v, np.ndarray) and v.ndim == 1}
            if scalar_result:
                n_samples = len(next(iter(scalar_result.values())))
                for i in range(n_samples):
                    sample_metrics = {f"sample/{k}": float(v[i]) for k, v in scalar_result.items()}
                    wandb.log(sample_metrics, commit=False)
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
        recon = self.volume_vae.decode(flat_latent, ms_obj_cond)  # (n_samples*n_grids, C, K, K, K)
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
        posterior, _, _ = self.volume_vae.encode(proj_lg, obj_msdf.repeat(n_samples, 1, 1, 1, 1))
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

    def on_test_epoch_start(self):
        self.all_results = []
        self.sample_joints = []
        self.cache_results = {}
        self.raw_depth_data = {}  # {obj_name: [raw_depths_sample1, raw_depths_sample2, ...]}

        ## Testing metrics
        self.runtime = 0
        self.diffusion_times = []
        self.vae_times = []
        self.optimization_times = []
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log') and not self.debug:
            self.test_images_table = wandb.Table(columns=[
                'batch_idx', 'obj_name',
                'simu_disp', 'pene_depth', 'intersect_vol', 'contact_area',
            ])

    def on_test_epoch_end(self):
        # Compute quality metrics and diversity
        final_metrics = {}
        for m in self.cfg.test.criteria:
            if "Entropy" not in m and "Cluster Size" not in m:
                all_metrics = np.concatenate([res[m] for res in self.all_results], axis=0)
                final_metrics[f"{m}/mean"] = np.mean(all_metrics).item()
                final_metrics[f"{m}/std"]  = np.std(all_metrics).item()
                final_metrics[f"{m}/min"]  = np.min(all_metrics).item()
                final_metrics[f"{m}/max"]  = np.max(all_metrics).item()

        # Timing
        avg_unet = float(np.mean(self.diffusion_times))    if self.diffusion_times    else 0.0
        avg_vae  = float(np.mean(self.vae_times))          if self.vae_times          else 0.0
        avg_opt  = float(np.mean(self.optimization_times)) if self.optimization_times else 0.0
        final_metrics.update({
            'Inference Time/UNet (s)':         avg_unet,
            'Inference Time/VAE (s)':          avg_vae,
            'Inference Time/Optimization (s)': avg_opt,
            'Inference Time/Total (s)':        avg_unet + avg_vae + avg_opt,
        })
        print(f"[Timing] UNet: {avg_unet:.3f}s | VAE: {avg_vae:.3f}s | Optimization: {avg_opt:.3f}s | Total: {avg_unet + avg_vae + avg_opt:.3f}s")

        sample_joints = np.concatenate(self.sample_joints, axis=0)
        entropy, cluster_size, entropy_2, cluster_size_2 = calc_diversity(sample_joints)
        final_metrics.update({
            "Entropy": entropy.item(), "Cluster Size": cluster_size.item(),
            "Canonical Entropy": entropy_2.item(), "Canonical Cluster Size": cluster_size_2.item(),
        })
        if not self.debug:
            wandb.log(final_metrics, commit=False)
            wandb.log({'test/results': self.test_images_table})

        for k, v in self.cache_results.items():
            with open(self._cache_path(k), 'wb') as f:
                pickle.dump(v, f)

        os.makedirs(osp.join('tmp', 'pene_analysis'), exist_ok=True)
        dataset_name = 'grab'
        # Finalise: convert per-metric lists to numpy arrays
        list_keys = {'raw_depth', 'penetr_vert_ids'}  # variable-length per-sample arrays, kept as lists
        pene_data = {}
        for obj_name, obj_data in self.raw_depth_data.items():
            pene_data[obj_name] = {}
            for k, v in obj_data.items():
                pene_data[obj_name][k] = v if k in list_keys else np.array(v)
        pene_save_path = osp.join('tmp', 'pene_analysis', f'{dataset_name}.pkl')
        with open(pene_save_path, 'wb') as f:
            pickle.dump(pene_data, f)
        print(f"[pene_analysis] Saved raw penetration depths to {pene_save_path}")

    def configure_optimizers(self):
        if self.cfg.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
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
