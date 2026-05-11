import os
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import open3d as o3d
import trimesh
from copy import copy
import numpy as np
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
import wandb
from matplotlib import pyplot as plt
from copy import deepcopy
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from common.model.pose_optimizer import optimize_pose_by_contact, optimize_pose_wrt_local_grids

from .lgcdifftrainer import LGCDiffTrainer
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh, geom_to_img, visualize_recon_hand_w_object, visualize_grid_contact, geom_to_img_unified
from common.msdf.utils.msdf import get_grid, calc_local_grid_all_pts_gpu
from common.evaluation.eval_fns import calculate_metrics
from einops import rearrange


class GraspDiffTrainer(LGCDiffTrainer):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, grid_ae, hand_ae, model, diffusion, cfg):
        super(GraspDiffTrainer, self).__init__(grid_ae=grid_ae, model=model, diffusion=diffusion, cfg=cfg)
        self.hand_ae = hand_ae
        if cfg.run_phase == 'train':
            self._load_pretrained_weights(cfg.hand_ae.get('pretrained_weight', None), target_prefix='hand_ae')
        self.hand_ae.eval().requires_grad_(False)
        
        def global2local(hand_latent, grid_centers):
            return self._global2local(hand_latent, grid_centers)
        
        object.__setattr__(self.model, 'global2local_fn', global2local)
    
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
        posterior, _, _ = self.grid_ae.encode(proj_lg, grid_msdf)
        local_latent = posterior.sample().view(batch_size, n_grids, -1) # (B, N, latent_dim)
        return local_latent
    
    # def on_fit_start(self):
    #     # Set grid_ae BatchNorm layers to eval mode to prevent running stats drift
    #     super(GraspDiffTrainer, self).on_fit_start()
    #     bn_count = 0
    #     for module in self.hand_ae.modules():
    #         if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
    #             module.eval()
    #             # Disable running stats updates during forward pass
    #             module.track_running_stats = False
    #             bn_count += 1

    #     print(f"Set {bn_count} BatchNorm layers to eval mode (prevents running stats drift)")
    
    def train_val_step(self, batch, batch_idx, stage):
        self.mano_layer.to(self.device)
        self.grid_coords = self.grid_coords.view(-1, 3).to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, apply_grid_mask=not self.cfg.ae.use_noncontact_grids)
        handobject.load_from_batch(batch, pool=self.pool)

        # mask = handobject.hand_vert_mask.any(dim=1) # B, H
        # print(f"Average number of visible hand vertices: {torch.sum(mask, dim=1).float().mean().item()}")
        # return

        lg_contact = handobject.ml_contact
        batch_size, n_grids = lg_contact.shape[:2]
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k)
        # n_ho_dist = handobject.n_ho_dist

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        ## First process all grids separately using GRIDAE
        flat_lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> b n (k1 k2 k3) c')
        lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> (b n) c k1 k2 k3')
        posterior, obj_feat, multi_scale_obj_cond = self.grid_ae.encode(lg_contact, obj_msdf)
        # z = torch.cat([n_ho_dist.unsqueeze(-1), posterior.sample().view(batch_size, n_grids, -1)], dim=-1) # n_dim + 1
        gt_contact_latent = posterior.sample().view(batch_size, n_grids, -1) # n_dim
        # obj_pc = torch.cat([obj_msdf_center, obj_feat.view(batch_size, n_grids, -1)], dim=-1)
        obj_pc = handobject.obj_msdf
        gt_hand_latent = self.hand_ae.encode(handobject.hand_verts.permute(0, 2, 1)).sample()

        ## Debug the hand data
        # sbj_id = batch['sbjId'][0]
        # hmodel = self.trainer.datamodule.train_set.rh_models[sbj_id].to(self.device)
        # full_pose = torch.cat([handobject.hand_root_rot, handobject.hand_pose], dim=1)
        # handV, handJ, _ = hmodel(full_pose, th_trans=handobject.hand_trans)
        # hmodel.to('cpu')
        # print(torch.norm(handV[0] - handobject.hand_verts[0], dim=-1).max())

        ## Pad the shape params with 0
        # betas = torch.zeros(batch_size, 10, device=self.device)
        # mano_param_vec = handobject.get_99_dim_mano_params()  # B x 99
        # mano_param_vec = torch.cat([mano_param_vec, betas], dim=-1) # B x 109
        cat_latent = torch.cat([gt_hand_latent, gt_contact_latent.view(batch_size, -1)], dim=-1) # B x (n_grids*n_dim + hand_latent_dim)

        input_data = {'x': cat_latent, 'obj_pc': obj_pc.permute(0, 2, 1), 'obj_msdf': handobject.obj_msdf}

        # grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        losses, model_output, recon_LVClatent = self.diffusion.training_losses(self.model, input_data,
                                                hand_ae=self.hand_ae, gt_handV=handobject.hand_verts, gt_handJ=handobject.hand_joints)

        pred_grid_latent = model_output[:, self.cfg.generator.unet.d_y:]
        losses['difference_loss'] = F.mse_loss(pred_grid_latent, recon_LVClatent.reshape(batch_size, -1), reduction='mean')

        ## For stable loss
        if self.loss_weights.get('stable_loss', 0) > 0:
            pred_grid_latent = rearrange(pred_grid_latent, 'b (n l) -> (b n) l', n=n_grids)
            recon_grid_contact = self.grid_ae.decode(pred_grid_latent, multi_scale_obj_cond)  # (B*N) x c x k x k x k
            lgc = rearrange(recon_grid_contact[:, 0], '(b n) k1 k2 k3 -> b (n k1 k2 k3)', b=batch_size, n=n_grids)
            lgc = torch.where(lgc < 0.05, torch.zeros_like(lgc), lgc)
            sdf_vals = handobject.obj_msdf[:, :, :self.msdf_k**3]        # (B, N, K^3)
            sdf_flat = sdf_vals.reshape(batch_size, n_grids * self.msdf_k**3)
            sdf_flat = sdf_flat * self.msdf_scale * np.sqrt(3)
            centres = handobject.obj_msdf[:, :, self.msdf_k**3:]          # (B, N, 3)
            sdf_grad = handobject.msdf_grad.reshape(batch_size, n_grids * self.msdf_k**3, 3)
            n_adj_pt = handobject.n_adj_pt.view(batch_size, n_grids * self.msdf_k**3)
            all_pts = centres.unsqueeze(2) + handobject.normalized_coords[None, None, :, :] * self.msdf_scale
            stable_loss = self.stable_loss(sdf_flat, all_pts.view(batch_size, -1, 3), lgc, sdf_grad, n_adj_pt,
                                    obj_mass=handobject.obj_mass, gravity_direction=torch.FloatTensor([[0, 0, -1]]).to(self.device),
                                    J=handobject.obj_inertia)
            losses['stable_loss'] = stable_loss.mean()

        ## Debugging code:
        # if batch_idx % 10 == 0:
        #     print(recon_grid_contact[:, 0].max().item())

        total_loss = sum([losses[k] * self.loss_weights[k] for k in losses.keys()])
        losses['total_loss'] = total_loss
        # grid_loss = F.mse_loss(err[:, :, 0], torch.zeros_like(err[:, :, 0]), reduction='mean')  # only compute loss on n_ho_dist dimension
        # diff_loss = F.mse_loss(err[:, :, 1:], torch.zeros_like(err[:, :, 1:]), reduction='none')
        # obj_pt_mask = input_data['x'][:, :, 0:1] < 0
        # diff_loss = torch.sum(diff_loss * obj_pt_mask) / (torch.sum(obj_pt_mask) + 1e-6) / diff_loss.shape[2]

        loss_dict = {f'{stage}/{k}': v for k, v in losses.items()}
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:#  and batch_idx > 0:
            vis_idx = 0
            vis_data = {k: v[vis_idx:vis_idx+1] for k, v in input_data.items()}
            condition = self.model.condition(vis_data)
            samples = self.diffusion.p_sample_loop(self.model, vis_data['x'].shape, condition, clip_denoised=False, progress=True)
            simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
            rot = batch['aug_rot'][vis_idx].cpu().numpy() if 'aug_rot' in batch else np.eye(3)
            obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
                            for i, name in enumerate(batch['objName'])]
            handobject._load_templates(idx=vis_idx, obj_templates=obj_templates)
            hand_latent, grid_latent = samples[:, :self.cfg.generator.unet.d_y], samples[:, self.cfg.generator.unet.d_y:].view(1, n_grids, -1)

            vis_ms_obj_cond = [c[vis_idx*n_grids:(vis_idx+1)*n_grids] for c in multi_scale_obj_cond]
            vis_obj_msdf_center = obj_msdf_center[vis_idx:vis_idx+1]

            pred_hand_verts, pred_verts_mask, pred_grid_contact = self.reconstruct_from_latent(grid_latent, vis_ms_obj_cond, vis_obj_msdf_center)
            gt_rec_hand_verts, gt_rec_verts_mask, gt_rec_grid_contact = self.reconstruct_from_latent(gt_contact_latent[vis_idx:vis_idx+1], vis_ms_obj_cond, vis_obj_msdf_center)

            contact_img = self._visualize_contact_comparison(
                vis_obj_msdf_center, pred_grid_contact, gt_rec_grid_contact, handobject, vis_idx)
            img = self._visualize_hand_comparison(
                pred_hand_verts, pred_verts_mask, gt_rec_hand_verts, gt_rec_verts_mask,
                handobject, vis_obj_msdf_center, rot, vis_idx)
            _, recon_handV, recon_handJ = self.hand_ae.decode(hand_latent)
            full_hand_img = self.visualize_full_hand_comparison(recon_handV[vis_idx], handobject.hand_verts[vis_idx], handobject.vis_obj_models[vis_idx])

            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_contact', contact_img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_hand', img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_sampled_full_hand', full_hand_img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_contact': wandb.Image(contact_img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_hand': wandb.Image(img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_sampled_full_hand': wandb.Image(full_hand_img)}, step=self.global_step)
        return total_loss
    
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
    
    def test_step(self, batch, batch_idx):
        # if batch_idx < 4:  # TODO: temporary skip for debugging
        #     return {}
        self.grid_coords = self.grid_coords.to(self.device)
        self.mano_layer.to(self.device)
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
        # lg_contact = handobject.ml_contact
        # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
        # obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3

        obj_msdf_grid = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k) # (B*N) x 1 x k x k x k
        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x N x 3
        obj_feat, multi_scale_obj_cond = self.grid_ae.encode_object(obj_msdf_grid)
        # obj_pc = torch.cat([obj_msdf_center, obj_feat.unsqueeze(0)], dim=-1)
        obj_pc = handobject.obj_msdf

        cat_noise = torch.randn(n_samples, n_grids * self.cfg.ae.feat_dim + self.hand_ae.latent_dim, device=self.device)
        ## 'x' only indicates the latent shape; latents are sampled inside the model
        input_data = {'x': cat_noise, 'obj_pc': obj_pc.permute(0, 2, 1).to(self.device), 'obj_msdf': handobject.obj_msdf}

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

        hand_latent, grid_latent = samples[:, :self.cfg.generator.unet.d_y], samples[:, self.cfg.generator.unet.d_y:].view(n_samples, n_grids, -1)

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

        grid_latent = grid_latent.reshape(n_samples*n_grids, -1)
        ## repeat the multi-scale obj cond here
        multi_scale_obj_cond = [cond.repeat(n_samples, 1, 1, 1, 1) for cond in multi_scale_obj_cond]
        multi_scale_obj_cond.append(obj_feat.repeat(n_samples, 1))
        recon_lg_contact = self.grid_ae.decode(grid_latent, multi_scale_obj_cond)
        recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x K x K x K x (1 + cse_dim)
        recon_lg_contact = recon_lg_contact.view(n_samples, n_grids, self.msdf_k, self.msdf_k, self.msdf_k, -1)
        recon_lg_contact[..., 0][recon_lg_contact[..., 0] < self.cfg.pose_optimizer.contact_th] = 0  ## maskout low contact prob

        pred_grid_contact = recon_lg_contact[..., 0].reshape(n_samples, -1)  # B x N x K^3
        obj_msdf_center = obj_msdf_center.repeat(n_samples, 1, 1)  # B x N x 3
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        pred_grid_cse = recon_lg_contact[..., 1:].reshape(n_samples, -1, self.cse_dim)
        pred_targetWverts = self.hand_cse.emb2Wvert(pred_grid_cse.view(n_samples, -1, self.cse_dim))

        _opt_t0 = time.perf_counter()
        if self.cfg.pose_optimizer.name == 'hand_ae':
            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                chunk_size=10
            )
            recon_param, _ = self.hand_ae(pred_hand_verts.permute(0, 2, 1), mask=pred_verts_mask.unsqueeze(1), is_training=False)
            nrecon_trans, recon_pose, recon_betas = torch.split(recon_param, [3, 48, 10], dim=1)
            recon_trans = nrecon_trans * 0.2
            handV, handJ, _ = self.mano_layer(recon_pose, th_betas=recon_betas, th_trans=recon_trans)
        elif self.cfg.pose_optimizer.name == 'lg_base':
            recon_hand_verts, recon_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                chunk_size=10
            )
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
        elif self.cfg.pose_optimizer.name == 'hybrid':
            recon_hand_verts, recon_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1), pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
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

                ## Do NMS: nms_mask[b, i] = True iff contact[b, i] >= contact[b, j] for all neighbours j
                # adj_pt_idx = batch['adjPointIndices'][0]   # (M, 2) — shared across the batch
                # batch_size = pred_grid_contact.shape[0]
                # i_idx, j_idx = adj_pt_idx[:, 0], adj_pt_idx[:, 1]
                # # For each point i, find max contact value among its neighbours across all samples at once
                # # pred_grid_contact[:, j_idx]: (B, M) — neighbour contact values per sample
                # # scatter_reduce along dim=1 into (B, N)
                # max_neighbor = torch.full((batch_size, n_grids * self.msdf_k**3), -float('inf'), device=pred_grid_contact.device)
                # max_neighbor.scatter_reduce_(1, i_idx.unsqueeze(0).expand(batch_size, -1),
                #                              pred_grid_contact[:, j_idx], reduce='amax', include_self=True)
                # nms_mask = pred_grid_contact >= max_neighbor  # (B, N)

                # mano_trans, global_pose, mano_pose, mano_shape = optimize_pose_by_contact(
                #             self.mano_layer, grid_centers=obj_msdf_center, target_pts=grid_coords.view(n_samples, -1, 3),
                #             target_W_verts=pred_targetWverts, pred_contact=pred_grid_contact, dist2contact_fn=self.grid_dist_to_contact,
                #             n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr,
                #             grid_scale=self.cfg.msdf.scale, w_repulsive=self.cfg.pose_optimizer.w_repulsive,
                #             w_reg_loss=self.cfg.pose_optimizer.w_regularization, init_pose=recon_params, nms_mask=nms_mask)

            handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
            # handV, handJ = init_handV, init_handJ
        else:
            recon_params, handV, handJ = self.hand_ae.decode(hand_latent)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.optimization_times.append(time.perf_counter() - _opt_t0)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_name}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_name, 'hand_joints': handJ[i], 'obj_model': handobject.obj_models[0], 'obj_hulls': handobject.obj_hulls[0],
                       'idx': i} for i in range(handV.shape[0])]
            
        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')

        # Print average of all metrics
        avg_metrics = {k: v.mean() for k, v in result.items()}
        print(f"Average metrics: {avg_metrics}")

        self.all_results.append(result)
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
        
        init_ho = copy(handobject)
        init_ho.hand_verts = torch.tensor(init_handV, dtype=torch.float32)
        init_ho.hand_joints = torch.tensor(init_handJ, dtype=torch.float32)

        if self.cfg.pose_optimizer.name != 'hybrid':
            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.hand_cse, None,
                pred_grid_contact.reshape(n_samples, -1),
                pred_grid_cse.reshape(n_samples, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(n_samples, -1, 3),
                chunk_size=10  # Process in chunks of 10 to reduce memory peak
            )
        else:
            pred_hand_verts, pred_verts_mask = recon_hand_verts, recon_verts_mask

        vis_result = self.cfg.test.get('vis_result', True)

        if vis_result:
            backend = self.cfg.get('vis_backend', 'open3d')
            img_side = 800
            recon_imgs = []
            pred_imgs = []
            init_imgs = []
            contact_mask_imgs = []
            for vis_idx in range(n_samples):
                recon_img, pred_geoms = visualize_recon_hand_w_object(
                    hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                    hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                    hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                    obj_mesh=handobject.vis_obj_models[vis_idx],
                    part_ids=handobject.hand_part_ids,
                    msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                    grid_scale=self.cfg.msdf.scale,
                    h=img_side, w=img_side, backend=backend)
                recon_imgs.append(recon_img)

                contact_img, contact_geoms = visualize_grid_contact(
                    contact_pts=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                    pt_contact=contact_mask[vis_idx].detach().cpu().numpy().astype(float),
                    grid_scale=self.cfg.msdf.scale,
                    obj_mesh=handobject.vis_obj_models[vis_idx],
                    w=img_side, h=img_side, backend=backend)
                contact_mask_imgs.append(contact_img)

                if self.debug:
                    ho_geoms = pred_ho.get_vis_geoms(idx=vis_idx)
                    contact_geoms_offset = [g['geometry'].translate((0, 0.5, 0)) if isinstance(g, dict) else g.translate((0, 0.5, 0)) for g in contact_geoms]
                    # o3d.visualization.draw_geometries(pred_geoms + [g['geometry'].translate((0, 0.25, 0)) if isinstance(g, dict) else g.translate((0, 0.25, 0)) for g in ho_geoms] + contact_geoms_offset, window_name='Predicted Hand-Object')
                else:
                    pred_img = pred_ho.vis_img(idx=vis_idx, h=img_side, w=img_side, backend=backend)
                    pred_imgs.append(pred_img)
                    init_img = init_ho.vis_img(idx=vis_idx, h=img_side, w=img_side, backend=backend)
                    init_imgs.append(init_img)

            recon_img_grid = np.concatenate(recon_imgs, axis=0)
            contact_mask_img_grid = np.concatenate(contact_mask_imgs, axis=0)
            if not self.debug:
                pred_img_grid = np.concatenate(pred_imgs, axis=0)
                init_img_grid = np.concatenate(init_imgs, axis=0)

        if hasattr(self.logger, 'experiment') and not self.debug:
            metric_row = [
                float(np.mean(result.get("Simulation Displacement", [0]))),
                float(np.mean(result.get("Penetration Depth", [0]))),
                float(np.mean(result.get("Intersection Volume", [0]))),
                float(np.mean(result.get("Contact Area", [0]))),
            ]
            if hasattr(self.logger.experiment, 'add_image') and vis_result:
                # TensorBoardLogger
                global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.test_dataloader()')) + batch_idx
                self.logger.experiment.add_image(f'test/Surrounding_hands', recon_img_grid, global_step, dataformats='HWC')
                self.logger.experiment.add_image(f'test/Initial_hands', init_img_grid, global_step, dataformats='HWC')
            elif hasattr(self.logger.experiment, 'log'):
                # WandbLogger
                if vis_result:
                    from PIL import Image as PILImage
                    img_dir = os.path.join(self.logger.experiment.dir, 'test_images', f'batch_{batch_idx:04d}_{obj_name}')
                    os.makedirs(img_dir, exist_ok=True)
                    def _save(arr, name):
                        path = os.path.join(img_dir, name)
                        PILImage.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).save(path)
                        return wandb.Image(path)
                    img_row = [
                        _save(recon_img_grid, 'surrounding.png'),
                        _save(pred_img_grid, 'predicted.png'),
                        _save(contact_mask_img_grid, 'contact_mask.png'),
                        _save(init_img_grid, 'initial.png'),
                    ]
                else:
                    img_row = []
                self.test_images_table.add_data(batch_idx, obj_name, *img_row, *metric_row)
            # o3d.visualization.draw(pred_geoms)
            # o3d.visualization.draw(gt_geoms + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g for g in pred_geoms])

        return result

    def on_test_epoch_start(self):
        self.all_results = []
        self.sample_joints = []

        ## Testing metrics
        self.runtime = 0
        self.diffusion_times = []
        self.optimization_times = []
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log') and not self.debug:
            import wandb
            vis_result = self.cfg.test.get('vis_result', True)
            img_cols = ['Surrounding_hands', 'Predicted_hands', 'Contact_mask', 'Initial_hands'] if vis_result else []
            self.test_images_table = wandb.Table(columns=[
                'batch_idx', 'obj_name',
                *img_cols,
                'simu_disp', 'pene_depth', 'intersect_vol', 'contact_area',
            ])

    def on_test_epoch_end(self):
        # Compute quality metrics and diversity (same as parent, without logging the table)
        final_metrics = {}
        for m in self.cfg.test.criteria:
            if "Entropy" not in m and "Cluster Size" not in m:
                all_metrics = np.concatenate([res[m] for res in self.all_results], axis=0)
                final_metrics[f"{m}/mean"] = np.mean(all_metrics).item()
                final_metrics[f"{m}/std"]  = np.std(all_metrics).item()
                final_metrics[f"{m}/min"]  = np.min(all_metrics).item()
                final_metrics[f"{m}/max"]  = np.max(all_metrics).item()

        # Timing
        avg_diff = float(np.mean(self.diffusion_times)) if self.diffusion_times else 0.0
        avg_opt  = float(np.mean(self.optimization_times)) if self.optimization_times else 0.0
        final_metrics.update({
            'Inference Time/Diffusion (s)':    avg_diff,
            'Inference Time/Optimization (s)': avg_opt,
            'Inference Time/Total (s)':         avg_diff + avg_opt,
        })
        print(f"[Timing] Diffusion: {avg_diff:.3f}s | Optimization: {avg_opt:.3f}s | Total: {avg_diff + avg_opt:.3f}s")

        from common.evaluation.eval_fns import calc_diversity
        sample_joints = np.concatenate(self.sample_joints, axis=0)
        entropy, cluster_size, entropy_2, cluster_size_2 = calc_diversity(sample_joints)
        final_metrics.update({
            "Entropy": entropy.item(), "Cluster Size": cluster_size.item(),
            "Canonical Entropy": entropy_2.item(), "Canonical Cluster Size": cluster_size_2.item(),
        })
        if not self.debug:
            wandb.log(final_metrics, commit=False)
            wandb.log({'test/results': self.test_images_table})
