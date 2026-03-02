import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import trimesh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from common.manopth.manopth.manolayer import ManoLayer
from common.model.losses import kl_div_normal, masked_rec_loss
from common.model.handobject import recover_hand_verts_from_contact
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh, visualize_local_grid_with_hand, geom_to_img, extract_masked_mesh_components, clip_mesh_to_aabb
from common.model.losses import masked_rec_loss
from common.msdf.utils.msdf import get_grid


class LGTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.debug = cfg.get('debug', False)
        self.loss_weights = cfg.train.loss_weights
        self.lr = cfg.train.lr
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=False, ncomps=45, flat_hand_mean=True).requires_grad_(False)
        self.hand_part_ids = torch.argmax(self.mano_layer.th_weights, dim=-1).detach().cpu().numpy()
        cse_ckpt = torch.load(cfg.data.hand_cse_path)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.handcse = HandCSE(n_verts=778, emb_dim=cse_ckpt['emb_dim'], cano_faces=handF.cpu().numpy()).to(self.device)
        self.handcse.load_state_dict(cse_ckpt['state_dict'])
        self.handcse.eval()
        for param in self.handcse.parameters():
            param.requires_grad = False
        self.grid_coords = get_grid(self.cfg.msdf.kernel_size) * self.cfg.msdf.scale  # (K^3, 3)

    def training_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')

        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        self.grid_coords = self.grid_coords.to(self.device)
        grid_sdf = batch['gridSDF'].squeeze(-1)
        gt_grid_contact = torch.cat([batch['gridContact'], batch['gridHandCSE']], dim=-1)
        hand_in_mask = batch['handVertMask'].any(dim=1)

        posterior, _, obj_cond = self.model.encode(gt_grid_contact.permute(0, 4, 1, 2, 3), grid_sdf.unsqueeze(1))
        recon_cgrid = self.model.decode(posterior.sample(), obj_cond=obj_cond).permute(0, 2, 3, 4, 1)
        contact, contact_hat = gt_grid_contact[..., 0], recon_cgrid[..., 0]
        cse, cse_hat = gt_grid_contact[..., 1:], recon_cgrid[..., 1:]
        # self.check_latents(posterior)
        batch_size = grid_sdf.shape[0]
        loss_dict = self.loss_net(x=gt_grid_contact, x_hat=recon_cgrid, posterior=posterior, gt_face_idx=batch['face_idx'],
                                  gt_w=batch['cse_weights'], proc=stage)

        if stage == 'val':
            pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                self.handcse, batch['face_idx'],
                contact_hat.reshape(batch_size, -1),
                cse_hat.reshape(batch_size, -1, cse.shape[-1]),
                grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
            )
            rec_loss = masked_rec_loss(pred_hand_verts, batch['nHandVerts'], batch['handVertMask']) * 1000
            loss_dict[f'{stage}/rec_loss'] = rec_loss.detach()
            loss_dict[f'{stage}/total_loss'] = rec_loss.detach()
            # Random latent sampling stats (mirrors test_step criteria)
            random_z = torch.randn_like(posterior.mode())
            random_recon_cgrid = self.model.decode(random_z, obj_cond=obj_cond)
            contact_value = random_recon_cgrid[:, 0].reshape(batch_size, -1).max(dim=-1).values
            loss_dict[f'{stage}/avg_contact_values'] = contact_value.mean().detach()
            loss_dict[f'{stage}/contact_ratio'] = (contact_value > 0.03).float().mean().detach()
            loss_dict[f'{stage}/in_ratio'] = (contact_value > 0.5).float().mean().detach()
        
        # recon_loss = F.mse_loss(recon_grid_contact, gt_grid_contact.permute(0, 4, 1, 2, 3))
        # loss_dict = {f'{stage}/embedding_loss': loss, f'{stage}/recon_loss': recon_loss, f'{stage}/perplexity': perplexity}
        # total_loss = sum(loss_dict.values())
        # total_loss = loss + self.recon_weight * recon_loss

        # Log losses - for validation, compute epoch average; for training, log per step
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=False, sync_dist=True)
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0 and batch_idx > 0:
            pred_grid_contact = recon_cgrid
            if stage == 'train':
                pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                    self.handcse, batch['face_idx'],
                    contact_hat.reshape(batch_size, -1),
                    cse_hat.reshape(batch_size, -1, cse.shape[-1]),
                    grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
                )
            gt_geoms = self.visualize_grid_and_hand(
                grid_coords=self.grid_coords.view(-1, 3),
                # grid_contact=gt_grid_contact[..., 0].reshape(batch_size, -1),
                grid_contact=(grid_sdf.view(batch_size, -1) + 0.5) / 2,
                pred_hand_verts=batch['nHandVerts'],
                hand_faces=self.mano_layer.th_faces,
                pred_mask=batch['handVertMask'],
                part_ids=self.hand_part_ids,
                batch_idx=0
            )
            pred_geoms = self.visualize_grid_and_hand(
                grid_coords=self.grid_coords.view(-1, 3),
                # grid_contact=pred_grid_contact[..., 0].reshape(batch_size, -1),
                grid_contact=(grid_sdf.view(batch_size, -1) + 0.5) / 2,
                pred_hand_verts=pred_hand_verts,
                hand_faces=self.mano_layer.th_faces,
                pred_mask=batch['handVertMask'],
                gt_mask=batch['handVertMask'],
                gt_hand_verts=batch['nHandVerts'],
                part_ids=self.hand_part_ids,
                batch_idx=0
            )
            if self.debug:
                all_geoms = gt_geoms + [g.translate((self.cfg.msdf.scale * 3, 0, 0)) for g in pred_geoms]
                o3d.visualization.draw_geometries(all_geoms, window_name=f'{stage} Local Grid Visualization')
            else:
                gt_img = geom_to_img(gt_geoms, w=400, h=400)
                pred_img = geom_to_img(pred_geoms, w=400, h=400)
                img = np.concatenate([gt_img, pred_img], axis=0)
                # Log image - works for both WandbLogger and TensorBoardLogger
                if hasattr(self.logger, 'experiment'):
                    if hasattr(self.logger.experiment, 'add_image'):
                        # TensorBoardLogger
                        global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                        self.logger.experiment.add_image(f'{stage}/local_grid', img, global_step, dataformats='HWC')
                    elif hasattr(self.logger.experiment, 'log'):
                        # WandbLogger
                        import wandb
                        self.logger.experiment.log({f'{stage}/local_grid': wandb.Image(img)},
                                                commit=False)
        return loss_dict[f'{stage}/total_loss']
    
    def test_step(self, batch, batch_idx):

        self.grid_coords = self.grid_coords.to(self.device)
        grid_sdf = batch['gridSDF'].squeeze(-1)
        gt_grid_contact = torch.cat([batch['gridContact'], batch['gridHandCSE']], dim=-1)
        batch_size = grid_sdf.shape[0]
        ## Explore latent space properties:
        # print("Random latent contact stats - avg contact value: {:.4f}, ratio of samples with contact > 0.03: {:.4f}".format(avg_contact, contact_ratio))

        posterior, obj_feat, obj_cond = self.model.encode(gt_grid_contact.permute(0, 4, 1, 2, 3), grid_sdf.unsqueeze(1))
        # recon_grid_contact, z_e, obj_feat = self.model(gt_grid_contact.permute(0, 4, 1, 2, 3), grid_sdf.unsqueeze(1))
        recon_cgrid = self.model.decode(posterior.sample(), obj_cond=obj_cond)
        recon_cgrid = recon_cgrid.permute(0, 2, 3, 4, 1)

        gt_rec_hand_verts, gt_rec_verts_mask = recover_hand_verts_from_contact(
            self.handcse, batch['face_idx'],
            gt_grid_contact[..., 0].view(batch_size, -1), gt_grid_contact[..., 1:].view(batch_size, -1, gt_grid_contact.shape[-1]-1),
            grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
        )
        # gt_geoms = self.visualize_grid_and_hand(
        #     grid_coords=self.grid_coords.view(-1, 3),
        #     grid_contact=gt_grid_contact[..., 0].view(batch_size, -1),
        #     pred_hand_verts=gt_rec_hand_verts,
        #     hand_faces=self.mano_layer.th_faces,
        #     pred_mask=batch['handVertMask'],
        #     gt_hand_verts=batch['nHandVerts'],
        #     gt_mask=batch['handVertMask'],
        #     batch_idx=0
        # )

        gt_rec_error = masked_rec_loss(gt_rec_hand_verts, batch['nHandVerts'], gt_rec_verts_mask) * 1000
        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.handcse, None,
            recon_cgrid[..., 0].reshape(batch_size, -1),
            recon_cgrid[..., 1:].reshape(batch_size, -1, gt_grid_contact.shape[-1] - 1),
            grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
        )
        # pred_geoms = self.visualize_grid_and_hand(
        #     grid_coords=self.grid_coords.view(-1, 3),
        #     grid_contact=recon_cgrid[..., 0].view(batch_size, -1),
        #     pred_hand_verts=pred_hand_verts,
        #     hand_faces=self.mano_layer.th_faces,
        #     pred_mask=batch['handVertMask'],
        #     # pred_mask=pred_verts_mask,
        #     gt_hand_verts=batch['nHandVerts'],
        #     gt_mask=batch['handVertMask'],
        #     batch_idx=0
        # )
        # all_geoms = gt_geoms + [g.translate((self.cfg.msdf.scale * 3, 0, 0)) for g in pred_geoms]
        # o3d.visualization.draw_geometries(all_geoms, window_name='GT and Pred Local Grid Visualization')

        pred_rec_error = masked_rec_loss(pred_hand_verts, batch['nHandVerts'], gt_rec_verts_mask) * 1000
        loss_dict = {'test/gt_rec_error': gt_rec_error.item(),
                     'test/pred_rec_error': pred_rec_error.item()}

        if batch_idx % self.cfg.test.vis_every_n_batches == 0:
            # Build local_grid (K,K,K,C) for sample 0 to extract the bbox lineset
            # vis_idx = 0
            for vis_idx in tqdm(range(batch_size)):
                contact_point_np = batch['objSamplePt'][vis_idx].detach().cpu().numpy()

                # Hand mesh
                hand_verts_np = batch['nHandVerts'][vis_idx].detach().cpu().numpy() + contact_point_np[np.newaxis, :]
                hand_faces_np = self.mano_layer.th_faces.cpu().numpy()
                hand_mesh = o3dmesh(vert=hand_verts_np, face=hand_faces_np, color=[0xF2/255, 0x71/255, 0x41/255])

                # Object mesh
                obj_rot = batch['objRot'][vis_idx].cpu().numpy()
                if obj_rot.shape == (3,):
                    from pytorch3d.transforms import axis_angle_to_matrix
                    objR = axis_angle_to_matrix(torch.from_numpy(obj_rot)).numpy()
                    objR = objR.T
                else:
                    objR = obj_rot
                obj_trans = batch['objTrans'][vis_idx].cpu().numpy()
                obj_name = batch['obj_name'][vis_idx]
                simp_obj_mesh = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh
                obj_mesh_data = simp_obj_mesh[obj_name]
                obj_verts = (objR @ obj_mesh_data['verts'].T).T + obj_trans[np.newaxis, :]
                obj_mesh = o3dmesh(vert=obj_verts, face=obj_mesh_data['faces'], color=[0.7, 0.7, 0.7])

                # Bounding box lineset
                gs = self.cfg.msdf.scale
                bbox_corners = np.array([contact_point_np + gs * np.array([sx, sy, sz])
                                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                bbox_lines = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]
                bbox_ls = o3d.geometry.LineSet()
                bbox_ls.points = o3d.utility.Vector3dVector(bbox_corners)
                bbox_ls.lines = o3d.utility.Vector2iVector(bbox_lines)
                bbox_ls.paint_uniform_color([0.0, 1.0, 0.0])

                lg_geoms = [
                    {'geometry': hand_mesh, 'alpha': 0.3},
                    {'geometry': obj_mesh, 'alpha': 0.3},
                    bbox_ls,
                ]

                gt_geoms = self.visualize_grid_and_hand(
                    grid_coords=self.grid_coords.view(-1, 3),
                    # grid_contact=gt_grid_contact[..., 0].reshape(batch_size, -1),
                    grid_contact=(grid_sdf.view(batch_size, -1) + 0.5) / 2,
                    pred_hand_verts=batch['nHandVerts'],
                    hand_faces=self.mano_layer.th_faces,
                    pred_mask=batch['handVertMask'],
                    part_ids=self.hand_part_ids,
                    batch_idx=vis_idx
                )
                cropped_obj = clip_mesh_to_aabb(
                    obj_mesh,
                    min_bound=contact_point_np - gs,
                    max_bound=contact_point_np + gs,
                    color=[0.7, 0.7, 0.7],
                )
                cropped_obj.translate(-contact_point_np)

                pred_geoms = self.visualize_grid_and_hand(
                    grid_coords=self.grid_coords.view(-1, 3),
                    # grid_contact=recon_cgrid[..., 0].reshape(batch_size, -1),
                    grid_contact=(grid_sdf.view(batch_size, -1) + 0.5) / 2,
                    pred_hand_verts=pred_hand_verts,
                    hand_faces=self.mano_layer.th_faces,
                    pred_mask=batch['handVertMask'],
                    gt_mask=batch['handVertMask'],
                    gt_hand_verts=batch['nHandVerts'],
                    part_ids=self.hand_part_ids,
                    batch_idx=vis_idx
                )
                # --- Random sampling: Option D + accumulated contact ---
                n_rand = 128
                obj_cond_single = [c[vis_idx:vis_idx+1].expand(n_rand, *c.shape[1:]) for c in obj_cond]  # (1, ...) -> (N, ...)
                rand_zs = torch.randn(n_rand, *posterior.mode().shape[1:], device=self.device)
                rand_grids = self.model.decode(
                    rand_zs, obj_cond=obj_cond_single
                ).permute(0, 2, 3, 4, 1)  # (N, K, K, K, C)

                # Accumulated contact: sum contact values across samples, normalize to [0,1]
                acc_contact = rand_grids[..., 0].sum(dim=0).reshape(-1)  # (K^3,)
                acc_contact_np = acc_contact.detach().cpu().numpy()
                acc_contact_np = (acc_contact_np - acc_contact_np.min()) / (acc_contact_np.max() - acc_contact_np.min() + 1e-8)
                grid_coords_np = self.grid_coords.view(-1, 3).detach().cpu().numpy()
                acc_pcd = o3d.geometry.PointCloud()
                acc_pcd.points = o3d.utility.Vector3dVector(grid_coords_np)
                cmap_inferno = plt.get_cmap('plasma')
                acc_pcd.colors = o3d.utility.Vector3dVector(cmap_inferno(acc_contact_np)[:, :3])
                acc_point_sizes = acc_contact_np * 180 + 20  # scale: [2, 42]
                # acc_geoms = [acc_pcd]
                # acc_img = geom_to_img(acc_geoms, w=400, h=400)

                # Option D: overlay N transparent reconstructed hand meshes
                hand_faces_np = self.mano_layer.th_faces.cpu().numpy()
                sdf_vals_np = ((grid_sdf[vis_idx].reshape(-1) + 0.5) / 2).clamp(0, 1).detach().cpu().numpy()
                sdf_pcd = o3d.geometry.PointCloud()
                sdf_pcd.points = o3d.utility.Vector3dVector(grid_coords_np)
                sdf_pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap('coolwarm')(sdf_vals_np)[:, :3])
                # sample_geoms = [sdf_pcd]
                sample_geoms = [{'geometry': acc_pcd, 'point_sizes': acc_point_sizes}]
                img_side = 800
                # with torch.no_grad():
                #     for si in range(n_rand):
                #         s_grid = rand_grids[si]  # (K, K, K, C)
                #         s_verts, s_mask = recover_hand_verts_from_contact(
                #             self.handcse, None,
                #             s_grid[..., 0].reshape(1, -1),
                #             s_grid[..., 1:].reshape(1, -1, s_grid.shape[-1] - 1),
                #             grid_coords=self.grid_coords.view(1, -1, 3),
                #         )
                #         s_verts_np = s_verts[0].detach().cpu().numpy()
                #         s_mask_np = s_mask[0].detach().cpu().numpy()
                #         s_mesh_geoms = extract_masked_mesh_components(
                #             s_verts_np, hand_faces_np, s_mask_np, part_ids=self.hand_part_ids,
                #             create_geometries=True, uniform_color=[0xF2/255, 0x71/255, 0x41/255],
                #         )
                #         sample_geoms.extend([{'geometry': g, 'alpha': 0.1} for g in s_mesh_geoms
                #                              if isinstance(g, o3d.geometry.TriangleMesh)])
                sample_img = geom_to_img(sample_geoms + [cropped_obj], w=img_side, h=img_side, half_range=0.01)
        
                os.makedirs('tmp', exist_ok=True)
                gt_img = geom_to_img(gt_geoms, w=img_side, h=img_side)
                pred_img = geom_to_img(pred_geoms, w=img_side, h=img_side)
                img = np.concatenate([gt_img, pred_img], axis=0)
                lg_img = geom_to_img(lg_geoms, w=img_side, h=img_side, scale=0.6)
                plt.imsave(f'tmp/local_grid_{batch_idx*batch_size + vis_idx:04d}.png', img)
                plt.imsave(f'tmp/hand_object_{batch_idx*batch_size + vis_idx:04d}.png', lg_img)
                plt.imsave(f'tmp/sample_overlay_{batch_idx*batch_size + vis_idx:04d}.png', sample_img)
                # plt.imsave(f'tmp/acc_contact_{batch_idx:04d}.png', acc_img)
                # if self.debug:
                #     all_geoms = gt_geoms + [g.translate((self.cfg.msdf.scale * 3, 0, 0)) for g in pred_geoms]
                #     o3d.visualization.draw_geometries(all_geoms, window_name='test Local Grid Visualization')
                #     o3d.visualization.draw_geometries(lg_geoms, window_name='test Hand/Object/BBox Visualization')
                # else:
            
            ## Test random sampling from latent space
            # random_z = torch.randn_like(posterior.mode())
            # random_recon_cgrid = self.model.decode(random_z, obj_cond=obj_cond)
            # random_c = random_recon_cgrid[:, 0].reshape(batch_size, -1)
            # contact_value = random_c.max(dim=-1).values
            # loss_dict.update({
            #     'test/avg_contact_values': contact_value.mean().item(),
            #     'test/contact_ratio': (contact_value > 0.05).float().mean().item(),
            #     'test/in_ratio': (contact_value > 0.5).float().mean().item()
            # })

            ## Test zero-contact grid reconstruction
            # gt_grid_contact[..., 0] = 0.0
            # gt_grid_contact[:] = 0
            # recon_cgrid, posterior, obj_feat = self.model(gt_grid_contact.permute(0, 4, 1, 2, 3), grid_sdf.unsqueeze(1))
            # zero_center = posterior.mode()
            random_z = torch.randn_like(posterior.mode())
            random_recon_cgrid = self.model.decode(random_z, obj_cond=obj_cond)
            random_c = random_recon_cgrid[:, 0].reshape(batch_size, -1)
            contact_value = random_c.max(dim=-1).values
            loss_dict.update({
                'test/avg_contact_values': contact_value.mean().item(),
                'test/contact_ratio': (contact_value > 0.05).float().mean().item(),
                'test/in_ratio': (contact_value > 0.5).float().mean().item()
            })
            # recon_cgrid = recon_cgrid.permute(0, 2, 3, 4, 1)
            # zero_contact_rec_error = F.l1_loss(recon_cgrid[..., 0], gt_grid_contact[..., 0])
            # loss_dict['test/zero_contact_rec_error'] = zero_contact_rec_error
            # loss_dict['test/zero_contact_rec_max'] = torch.max(torch.abs(recon_cgrid[..., 0]))
            print(loss_dict)

            self.log_dict(loss_dict, prog_bar=True, on_step=False, on_epoch=True)

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
        return optimizer
    
    def loss_net(self, x, x_hat, posterior, gt_face_idx, gt_w, proc='train'):
        """
        Compute the loss for training the GRIDAE
        1. Reconstruction loss between x and x_hat
        2. Regularization loss on z_e (KL-divergence)
        """
        contact, contact_hat = x[..., 0], x_hat[..., 0] ## (B, K, K, K)
        cse, cse_hat = x[..., 1:], x_hat[..., 1:]
        contact_diff = contact - contact_hat
        cse_diff = (cse - cse_hat) * contact[..., None] # weighing by contact likelihood
        cse_value_loss = F.l1_loss(cse_diff, torch.zeros_like(cse_diff))
        cse_rec_loss = self.handcse.cse_rec_loss(cse_hat.reshape(x.shape[0], -1, cse.shape[-1]), gt_face_idx, gt_w)
        cse_rec_loss = (cse_rec_loss * contact.reshape(x.shape[0], -1)).mean()
        contact_loss = F.mse_loss(contact_diff, torch.zeros_like(contact_diff))
        # kl_loss = kl_div_normal(z_e)
        kl_loss = posterior.kl().mean()
        # rec_loss = masked_rec_loss(pred_hand_verts, gt_hand_verts, gt_verts_mask)
        
        total_loss = (self.loss_weights.w_contact * contact_loss + self.loss_weights.w_cse_rec * cse_rec_loss
                      + self.loss_weights.w_cse_value * cse_value_loss + self.loss_weights.w_kl * kl_loss)
        loss_dict = {
            f'{proc}/contact_loss': contact_loss.detach(),
            f'{proc}/cse_rec_loss': cse_rec_loss.detach(),
            f'{proc}/cse_value_loss': cse_value_loss.detach(),
            f'{proc}/kl_loss': kl_loss.detach(),
            f'{proc}/total_loss': total_loss,
        }
        return loss_dict
    
    
    @staticmethod
    def visualize_grid_and_hand(grid_coords, grid_contact, pred_hand_verts, hand_faces, pred_mask,
                                part_ids, batch_idx=0, gt_hand_verts=None, gt_mask=None):
        """
        Visualize grid points colored by contact likelihood and masked hand mesh.

        Args:
            grid_coords: (K^3, 3) grid coordinates in world space (same for all batches)
            grid_contact: (B, K^3) contact values between 0 and 1
            pred_hand_verts: (B, H, 3) predicted hand vertices
            hand_faces: (F, 3) hand face indices
            pred_mask: (B, H) boolean mask for predicted hand vertices
            batch_idx: which sample in the batch to visualize (default: 0)
            gt_hand_verts: (B, H, 3) optional ground truth hand vertices
            gt_mask: (B, H) optional boolean mask for GT hand vertices

        Returns:
            list: List of open3d geometries (point cloud for grid, mesh/points for hand)
        """
        # Convert tensors to numpy if needed
        if isinstance(grid_coords, torch.Tensor):
            grid_coords_np = grid_coords.detach().cpu().numpy()
        else:
            grid_coords_np = grid_coords

        if isinstance(grid_contact, torch.Tensor):
            grid_contact_np = grid_contact[batch_idx].detach().cpu().numpy()
        else:
            grid_contact_np = grid_contact[batch_idx]

        if isinstance(pred_hand_verts, torch.Tensor):
            pred_hand_verts_np = pred_hand_verts[batch_idx].detach().cpu().numpy()
        else:
            pred_hand_verts_np = pred_hand_verts[batch_idx]

        if isinstance(hand_faces, torch.Tensor):
            hand_faces_np = hand_faces.detach().cpu().numpy()
        else:
            hand_faces_np = hand_faces

        if isinstance(pred_mask, torch.Tensor):
            pred_mask_np = pred_mask[batch_idx].detach().cpu().numpy()
        else:
            pred_mask_np = pred_mask[batch_idx]

        geometries = []

        # 1. Create grid point cloud with inferno colormap based on contact values
        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(grid_coords_np)

        # Apply inferno colormap to contact values
        # cmap = plt.get_cmap('inferno')
        cmap = plt.get_cmap('coolwarm')
        grid_colors = np.array([cmap(val)[:3] for val in grid_contact_np])
        grid_pcd.colors = o3d.utility.Vector3dVector(grid_colors)
        geometries.append(grid_pcd)

        # 2. Create predicted masked hand mesh and isolated vertices
        pred_geometries = extract_masked_mesh_components(
            pred_hand_verts_np, hand_faces_np, pred_mask_np, part_ids=part_ids,
            create_geometries=True, uniform_color=[0xF2/255, 0x71/255, 0x41/255],
        )
        geometries.extend(pred_geometries)

        # 4. Visualize GT hand if provided
        if gt_hand_verts is not None and gt_mask is not None:
            # Convert GT tensors to numpy
            if isinstance(gt_hand_verts, torch.Tensor):
                gt_hand_verts_np = gt_hand_verts[batch_idx].detach().cpu().numpy()
            else:
                gt_hand_verts_np = gt_hand_verts[batch_idx]

            if isinstance(gt_mask, torch.Tensor):
                gt_mask_np = gt_mask[batch_idx].detach().cpu().numpy()
            else:
                gt_mask_np = gt_mask[batch_idx]

            # Create GT masked hand mesh and isolated vertices
            gt_geometries = extract_masked_mesh_components(
                gt_hand_verts_np, hand_faces_np, gt_mask_np, part_ids=part_ids,
                create_geometries=True, uniform_color=[0x2A/255, 0x5E/255, 0x8C/255],
            )
            geometries.extend(gt_geometries)

        return geometries

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


    def check_latents(self, posterior):
        """
        Encodes an image and checks the VAE latents for outliers or NaNs.
        """
        with torch.no_grad():
            # We sample from it (just like during training)
            latents = posterior.sample()
            
            # CRITICAL: Apply the Scaling Factor
            # SD 1.5/SDXL uses 0.18215. Without this, variance is too high.
            # scaling_factor = 0.18215
            scaling_factor = 1
            scaled_latents = latents * scaling_factor

        # 4. Statistical Analysis
        l_min = scaled_latents.min().item()
        l_max = scaled_latents.max().item()
        l_mean = scaled_latents.mean().item()
        l_std = scaled_latents.std().item()
        
        print(f"Latent Statistics (Scaled):")
        print(f"  Min:  {l_min:.4f}")
        print(f"  Max:  {l_max:.4f}")
        print(f"  Mean: {l_mean:.4f} (Should be close to 0)")
        print(f"  Std:  {l_std:.4f}  (Should be close to 1)")

        # 5. Health Checks
        if torch.isnan(scaled_latents).any():
            print("❌ FAILURE: Latents contain NaNs!")
        elif torch.isinf(scaled_latents).any():
            print("❌ FAILURE: Latents contain Infinity!")
        elif abs(l_max) > 15 or abs(l_min) > 15:
            print("⚠️ WARNING: Extreme outliers detected. Max value > 15.")
            print("   Action: These images might destabilize training. Consider removing or clamping.")
        else:
            print("✅ SUCCESS: Latents look healthy and stable.")