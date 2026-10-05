import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import open3d as o3d
import trimesh
from copy import copy
import numpy as np
from multiprocessing import Pool
import wandb

from common.manopth.manopth.manolayer import ManoLayer
from common.model.handobject import recover_hand_verts_from_contact
from common.model.pose_optimizer import optimize_pose_wrt_local_grids
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import o3dmesh_from_trimesh, visualize_local_grid_with_hand, geom_to_img, extract_masked_mesh_components
from common.model.losses import masked_rec_loss, kl_div_normal_muvar
from common.msdf.utils.msdf import get_grid
from common.evaluation.eval_fns import calculate_metrics, calc_diversity


class MLCTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.debug = cfg.get('debug', False)
        self.loss_weights = cfg.train.loss_weights
        self.msdf_k = cfg.msdf.kernel_size
        self.lr = cfg.train.lr

        # KL annealing configuration
        self.kl_anneal_enabled = cfg.train.loss_weights.get('kl_anneal', True)
        self.kl_warmup_epochs = cfg.train.loss_weights.get('kl_warmup_epochs', 5)
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=True, ncomps=cfg.pose_optimizer.ncomps, flat_hand_mean=True)
        self.closed_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'))
        cse_ckpt = torch.load(cfg.data.hand_cse_path)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.cse_dim = cse_ckpt['emb_dim']
        self.handcse = HandCSE(n_verts=778, emb_dim=self.cse_dim, cano_faces=handF.cpu().numpy()).to(self.device)
        self.handcse.load_state_dict(cse_ckpt['state_dict'])
        self.handcse.eval()
        self.grid_coords = get_grid(self.cfg.msdf.kernel_size) * self.cfg.msdf.scale  # (K^3, 3)

    def training_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')
        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        self.grid_coords = self.grid_coords.view(-1, 3).to(self.device)
        obj_names = batch['objName']
        simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
        obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
                         for name in obj_names]
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)
        handobject.load_from_batch(batch)
        lg_contact = handobject.ml_contact
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        recon_lg_contact, mu, logvar = self.model(
            lg_contact.permute(0, 1, 5, 2, 3, 4), obj_msdf=obj_msdf, msdf_center=obj_msdf_center)
        recon_lg_contact = recon_lg_contact.permute(0, 1, 3, 4, 5, 2)  # B x N x K x K x K x (1 + cse_dim)

        batch_size = handobject.batch_size
        contact_hat, cse_hat = recon_lg_contact[..., 0], recon_lg_contact[..., 1:]
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords[None, None, :, :]  # B x N x K^3 x 3
        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.handcse,
            contact_hat.reshape(batch_size, -1),
            cse_hat.reshape(batch_size, -1, self.cse_dim),
            grid_coords=grid_coords.view(batch_size, -1, 3),
            # mask_th = 0.02
        )

        loss_dict = self.loss_net(recon_lg_contact, lg_contact, mu, logvar,
                                  pred_hand_verts, handobject.hand_verts,
                                  handobject.hand_vert_mask)
                                #   pred_verts_mask)
                                  
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
    
    def on_test_epoch_start(self):
        self.pool = Pool(min(self.cfg.test.batch_size, 16))
        self.all_results = []
        self.sample_joints = []

        ## Testing metrics
        self.runtime = 0
        if not self.debug:
            for metric in self.cfg.test.criteria:
                wandb.define_metric(metric, summary='mean')

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        Handles unfreezing pretrained weights at the specified epoch.
        """
        # Check if model has unfreeze capability and if we've reached the unfreeze epoch
        if hasattr(self.model, 'unfreeze_pretrained_weights') and hasattr(self.cfg.generator, 'ae_freeze_until_epoch'):
            if self.current_epoch == self.cfg.generator.ae_freeze_until_epoch:
                print(f"\n{'='*60}")
                print(f"Epoch {self.current_epoch}: Unfreezing pretrained autoencoder weights")
                print(f"{'='*60}\n")
                self.model.unfreeze_pretrained_weights()

                # Recreate optimizer to include newly unfrozen parameters
                self._recreate_optimizer()
                print("Optimizer recreated with newly unfrozen parameters\n")

    def test_step(self, batch, batch_idx):
        self.grid_coords = self.grid_coords.to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=True)
        obj_hulls = getattr(self.trainer.datamodule, 'test_set').obj_hulls
        obj_names = batch['objName']
        obj_hulls = [obj_hulls[name] for name in obj_names]
        obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh
        obj_meshes = []
        batch_size, n_pts = batch['objMsdf'].shape[:2]
        for name in obj_names:
            obj_meshes.append(trimesh.Trimesh(obj_mesh_dict[name]['verts'], obj_mesh_dict[name]['faces']))

        if self.cfg.generator.model_type == 'gt':
            ## Test using gt contact grids.
            handobject.load_from_batch(batch)
            obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:]
            recon_lg_contact = handobject.ml_contact
        else:
            ## Test the reconstrucion 
            handobject.load_from_batch(batch)
            lg_contact = handobject.ml_contact
            obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
            # recon_lg_contact, mu, logvar = self.model(
            #     lg_contact.permute(0, 1, 5, 2, 3, 4), obj_msdf=obj_msdf, msdf_center=obj_msdf_center)
            recon_lg_contact, z_e, obj_feat = self.model.volume_vae(
                lg_contact.view(batch_size*n_pts, self.msdf_k, self.msdf_k, self.msdf_k, -1).permute(0, 4, 1, 2, 3),
                obj_msdf=obj_msdf.unsqueeze(1), sample_posterior=False)
            recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x N x K x K x K x (1 + cse_dim)
            recon_lg_contact = recon_lg_contact.view(batch_size, n_pts, self.msdf_k, self.msdf_k, self.msdf_k, -1)
            # recon_lg_contact = lg_contact

            ## If masked with GT pt mask
            # recon_lg_contact = recon_lg_contact * handobject.obj_pt_mask[:, :, None, None, None, None]

            recon_lg_contact = recon_lg_contact.view(batch_size, n_pts, self.msdf_k**3, -1)
            sample_contact = recon_lg_contact[..., 0] * handobject.obj_pt_mask[:, :, None].float()
            sample_cse = recon_lg_contact[..., 1:]
            grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3

            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.handcse, None,
                sample_contact.reshape(batch_size, -1), sample_cse.reshape(batch_size, -1, self.cse_dim),
                grid_coords=grid_coords.reshape(batch_size, -1, 3),
                # mask_th = 0.02
            )
            # handobject.load_from_batch_object_only(batch)
            # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            # recon_lg_contact = self.model.sample(obj_msdf, obj_msdf_center).permute(0, 1, 3, 4, 5, 2)

        batch_size, n_pts = recon_lg_contact.shape[:2]
        pred_grid_contact = recon_lg_contact[..., 0].reshape(batch_size, -1)  # B x N x K^3
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        pred_grid_cse = recon_lg_contact[..., 1:].reshape(batch_size, -1, self.cse_dim)
        pred_targetWverts = self.handcse.emb2Wvert(pred_grid_cse.view(batch_size, -1, self.cse_dim))

        with torch.enable_grad():
            mano_trans, global_pose, mano_pose, mano_shape = optimize_pose_wrt_local_grids(
                        self.mano_layer, target_pts=grid_coords.view(batch_size, -1, 3),
                        target_W_verts=pred_targetWverts, weights=pred_grid_contact,
                        n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr)

        handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_names[i]}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_names[i], 'hand_joints': handJ[i], 'obj_model': obj_meshes[i], 'obj_hulls': obj_hulls[i],
                       'idx': i} for i in range(handV.shape[0])]
            
        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')
        ## MPJPE:
        if self.cfg.test_gt:
            mpjpe = np.linalg.norm(handobject.hand_joints.cpu().numpy() - handJ, axis=-1).mean(axis=1) * 1000  # B,
            mpvpe = np.linalg.norm(handobject.hand_verts.cpu().numpy() - handV, axis=-1).mean(axis=1) * 1000  # B,
            result.update({'MPJPE': mpjpe, 'MPVPE': mpvpe})

        print(result)
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
        vis_idx = 2
        gt_geoms = handobject.get_vis_geoms(idx=vis_idx, obj_templates=obj_meshes)
        pred_ho = copy(handobject)
        pred_ho.hand_verts = torch.tensor(handV, dtype=torch.float32)
        pred_ho.hand_joints = torch.tensor(handJ, dtype=torch.float32)
        pred_geoms = pred_ho.get_vis_geoms(idx=vis_idx, obj_templates=obj_meshes)
        o3d.visualization.draw(gt_geoms + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g for g in pred_geoms])

        return result

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

        # Log final metrics to wandb
        if not self.debug:
            wandb.log(final_metrics, commit=False)

        # return final_metrics
    
        ## GT visualization
        # vis_idx = 0
        # gt_geoms = visualize_local_grid_with_hand(
        #         batch['localGrid'][vis_idx].cpu().numpy(), hand_verts=batch['nHandVerts'][vis_idx].cpu().numpy(),
        #         hand_faces=self.mano_layer.th_faces.cpu().numpy(), hand_cse=self.handcse.embedding_tensor.detach().cpu().numpy(),
        #         kernel_size=self.cfg.msdf.kernel_size, grid_scale=self.cfg.msdf.scale
        #     )
        # o3d.visualization.draw_geometries(gt_geoms, window_name='GT Local Grid Visualization')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Cosine annealing scheduler with warmup for stable VAE training
        # Warmup for first 5% of training, then cosine decay
        max_epochs = self.cfg.trainer.max_epochs
        warmup_epochs = max(1, int(0.05 * max_epochs))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=self.lr * 0.01  # Decay to 1% of initial LR
        )

        # Linear warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of lr
            end_factor=1.0,    # Reach full lr
            total_iters=warmup_epochs
        )

        # Sequential scheduler: warmup -> cosine annealing
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _recreate_optimizer(self):
        """
        Recreate the optimizer and scheduler to include newly unfrozen parameters.
        This is necessary when parameters are unfrozen after optimizer initialization,
        as PyTorch optimizers fix their parameter list at creation time.
        """
        # Get current optimizer state
        old_optimizer = self.trainer.optimizers[0]

        # Count trainable parameters before and for logging
        trainable_params_before = sum(p.numel() for p in old_optimizer.param_groups[0]['params'] if p.requires_grad)
        trainable_params_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Create new optimizer with all currently trainable parameters
        new_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )

        # Recreate scheduler for the remaining epochs
        max_epochs = self.cfg.trainer.max_epochs
        remaining_epochs = max_epochs - self.current_epoch
        warmup_epochs = max(1, int(0.05 * remaining_epochs))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            new_optimizer,
            T_max=remaining_epochs - warmup_epochs,
            eta_min=self.lr * 0.01
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            new_optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        new_scheduler = torch.optim.lr_scheduler.SequentialLR(
            new_optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]
        )

        # Replace the optimizer and scheduler in Lightning's trainer
        self.trainer.optimizers = [new_optimizer]
        self.trainer.lr_scheduler_configs[0].scheduler = new_scheduler

        print(f"Optimizer updated: {trainable_params_before:,} -> {trainable_params_after:,} trainable parameters")
        print(f"Scheduler recreated for remaining {remaining_epochs} epochs")
    
    def get_kl_weight(self):
        """
        Compute dynamic KL weight with linear warmup annealing.
        Gradually increases from 0 to target weight over kl_warmup_epochs.
        """
        if not self.kl_anneal_enabled or self.kl_warmup_epochs == 0:
            return self.loss_weights.w_kl

        # Linear warmup: w_kl/n -> w_kl over kl_warmup_epochs
        warmup_progress = min(1.0, (self.current_epoch + 1) / self.kl_warmup_epochs)
        return warmup_progress * self.loss_weights.w_kl

    def loss_net(self, pred_lgc, gt_lgc, mu, logvar, pred_hand_verts, gt_hand_verts, contact_verts_mask):
        """
        Compute the loss for training the VolumeVAE
        1. Reconstruction loss between x and x_hat
        2. KL-divergence loss on z_e with dynamic weighting (annealing)
        """
        pred_contact, pred_cse = pred_lgc[..., 0], pred_lgc[..., 1:]
        gt_contact, gt_cse = gt_lgc[..., 0], gt_lgc[..., 1:]
        contact_loss = F.mse_loss(pred_contact, gt_contact)
        cse_loss = F.mse_loss(pred_cse, gt_cse).item()
        kl_loss = kl_div_normal_muvar(mu, logvar).item()
        hand_rec_loss = masked_rec_loss(pred_hand_verts, gt_hand_verts, contact_verts_mask).item()

        # Dynamic KL weight with annealing
        kl_weight = self.get_kl_weight()

        loss_dict = {
            'contact_loss': contact_loss,
            'cse_loss': cse_loss,
            'kl_loss': kl_loss,
            'hand_rec_loss': hand_rec_loss,
            'kl_weight': kl_weight,  # Log the current KL weight
            'total_loss': self.loss_weights.w_contact * contact_loss + self.loss_weights.w_cse * cse_loss\
                + kl_weight * kl_loss + self.loss_weights.w_rec * hand_rec_loss
        }

        return loss_dict
    
    @staticmethod
    def visualize_recon_hand_w_object(hand_verts, hand_verts_mask, hand_faces, obj_mesh, msdf_center, grid_scale, mesh_color=[0.8, 0.7, 0.6], h=500, w=500):
        masked_hand_geometries = extract_masked_mesh_components(
            hand_verts=hand_verts,
            hand_faces=hand_faces,
            vertex_mask=hand_verts_mask,
            create_geometries=True,
            mesh_color=mesh_color
        )
        obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

        # Create bounding boxes for each MSDF center
        bbox_geometries = []
        for center in msdf_center:
            # Define 8 corners of the bounding box
            bbox_points = np.array([
                center + grid_scale * np.array([-1, -1, -1]),
                center + grid_scale * np.array([1, -1, -1]),
                center + grid_scale * np.array([1, 1, -1]),
                center + grid_scale * np.array([-1, 1, -1]),
                center + grid_scale * np.array([-1, -1, 1]),
                center + grid_scale * np.array([1, -1, 1]),
                center + grid_scale * np.array([1, 1, 1]),
                center + grid_scale * np.array([-1, 1, 1]),
            ])

            # Define edges of the bounding box
            bbox_lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
            ]

            # Create LineSet for bounding box
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bbox_points)
            line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])  # Green
            bbox_geometries.append(line_set)

        vis_geoms = masked_hand_geometries + [obj_o3d_mesh] + bbox_geometries
        img = geom_to_img(vis_geoms, w=w, h=h, scale=0.5)
        return img, vis_geoms
    
    @staticmethod
    def vis_contact(obj_mesh, hand_mesh, obj_pts, obj_pt_mask):
        hand_mesh = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.7, 0.6])
        obj_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

        # Convert obj_pts to numpy if it's a tensor
        if isinstance(obj_pts, torch.Tensor):
            obj_pts_np = obj_pts.cpu().numpy()
        else:
            obj_pts_np = obj_pts

        # Convert obj_pt_mask to numpy if it's a tensor
        if isinstance(obj_pt_mask, torch.Tensor):
            obj_pt_mask_np = obj_pt_mask.cpu().numpy()
        else:
            obj_pt_mask_np = obj_pt_mask

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pts_np)

        # Color points: red for masked (True), blue for rest (False)
        colors = np.zeros((len(obj_pts_np), 3))
        colors[obj_pt_mask_np] = [1.0, 0.0, 0.0]  # Red for selected points
        colors[~obj_pt_mask_np] = [0.0, 0.0, 1.0]  # Blue for rest
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        o3d.visualization.draw_geometries([hand_mesh, obj_mesh, pcd])