import os.path as osp
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

from common.manopth.manopth.manolayer import ManoLayer
from common.model.pose_optimizer import optimize_pose_by_contact, optimize_pose_wrt_local_grids, optimize_pose_contactopt

from common.model.hand_cse.hand_cse import HandCSE
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.utils.geometry import GridDistanceToContact
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh, geom_to_img, visualize_recon_hand_w_object, visualize_grid_contact
from common.msdf.utils.msdf import get_grid, calc_local_grid_all_pts_gpu, nn_dist_to_mesh_gpu
from common.evaluation.eval_fns import calculate_metrics, calc_diversity
from einops import rearrange


class PointContactDiffTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, model, diffusion, cfg):
        super().__init__()
        mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                               use_pca=cfg.pose_optimizer.use_pca, ncomps=cfg.pose_optimizer.ncomps, flat_hand_mean=False)
        object.__setattr__(self, 'mano_layer', mano_layer.eval().requires_grad_(False))
        self.automatic_optimization = cfg.train.optimizer != 'asam'
        self.closed_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'))
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

        cse_ckpt = torch.load(cfg.data.hand_cse_path, weights_only=False)
        self.cse_dim = cse_ckpt['emb_dim']
        handF = self.mano_layer.th_faces
        self.hand_cse = HandCSE(n_verts=778, emb_dim=self.cse_dim, cano_faces=handF.cpu().numpy()).to(self.device)
        self.hand_cse.load_state_dict(cse_ckpt['state_dict'])
        self.hand_cse.eval().requires_grad_(False)
        self.pool = None # Pool(processes=16)

        def global2local(hand_latent, grid_centers):
            return self._global2local(hand_latent, grid_centers)
        
        # object.__setattr__(self.model, 'global2local_fn', global2local)
    
    def _global2local(self, hand_latent, contact_points):
        _, handV, handJ = self.hand_ae.decode(hand_latent)
        hand_faces = self.mano_layer.th_faces

        batch_size, n_pts = contact_points.shape[:2]
        k = self.cfg.msdf.kernel_size
        proj_contact = torch.zeros(n_pts).to(self.device)  # (B, N*K^3)
        proj_cse = torch.zeros(batch_size, n_pts, self.cse_dim).to(self.device)  # (B, N*K^3, cse_dim)

        for b in range(batch_size):
            nn_dist, nn_face_idx, nn_point = nn_dist_to_mesh_gpu(
                contact_points[b], handV[b], hand_faces
            )

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

            proj_contact[b] = self.grid_dist_to_contact(nn_dist.reshape(-1))
            proj_cse[b] = grid_hand_cse

        proj_lg = torch.cat([proj_contact.unsqueeze(-1), proj_cse], dim=-1) # (B, N*K^3, 1+cse_dim)
        # proj_lg = rearrange(proj_lg, 'b n c -> (b n) c')
        return proj_lg
    
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
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, apply_grid_mask=True)
        handobject.load_from_batch(batch, pool=self.pool)
        batch_size, n_pts = handobject.hand_verts.shape[:2]

        # mask = handobject.hand_vert_mask.any(dim=1) # B, H
        # print(f"Average number of visible hand vertices: {torch.sum(mask, dim=1).float().mean().item()}")
        # return

        obj_pc = torch.cat([handobject.obj_verts, handobject.obj_normals], dim=-1)  # B x N x 6
        # gt_hand_latent = self.hand_ae.encode(handobject.hand_verts.permute(0, 2, 1)).sample()
        gt_contact = torch.cat([handobject.contact_map, handobject.point_cse], dim=-1)  # B x N x (1+cse_dim)

        ## Debug the hand data

        input_data = {'x': gt_contact, 'obj_pc': obj_pc.permute(0, 2, 1)}

        # grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        _, pred_x0 = self.diffusion.training_losses(self.model, input_data)
        # losses['difference_loss'] = F.mse_loss(pred_contact, recon_contact.reshape(batch_size, -1), reduction='mean')

        ## Debugging code:
        # if batch_idx % 10 == 0:
        #     print(recon_grid_contact[:, 0].max().item())

        pred_c, pred_cse = pred_x0.split([1, self.cse_dim], dim=-1)
        contact_loss = F.mse_loss(pred_c, gt_contact[:, :, :1], reduction='mean')
        cse_loss = F.mse_loss(gt_contact[:, :, :1] * (pred_cse - gt_contact[:, :, 1:]),
                              torch.zeros_like(pred_cse).to(self.device).float(), reduction='sum') / (torch.sum(gt_contact[:, :, :1]) + 1e-8)
        losses = {}
        losses['contact_loss'] = contact_loss
        losses['cse_loss'] = cse_loss
        losses['total_loss'] = contact_loss + cse_loss * 0.05

        loss_dict = {f'{stage}/{k}': v for k, v in losses.items()}
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        if False:
        # if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            vis_idx = 0
            vis_data = {k: v[vis_idx:vis_idx+1] for k, v in input_data.items()}
            condition = self.model.condition(vis_data)
            sample_contact = self.diffusion.p_sample_loop(self.model, vis_data['x'].shape, condition, clip_denoised=False, progress=True)
            obj_mesh_dict = getattr(self.trainer.datamodule, f'{stage}_set').obj_info
            # obj_mesh_dict = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh

            obj_templates = [trimesh.Trimesh(obj_mesh_dict[name]['verts'], obj_mesh_dict[name]['faces'])
                            for i, name in enumerate(batch['objName'])]
            handobject._load_templates(idx=vis_idx, obj_templates=obj_templates)
            contact, cse = sample_contact[:, :, 0], sample_contact[:, :, 1:]
            # gt_cmap_img, gt_pmap_img = handobject.vis_maps(idx=vis_idx, w=400, h=400)
            handobject.contact_map = contact
            Wverts = self.hand_cse.emb2Wvert(cse)
            vert_idx = Wverts.argmax(dim=-1)
            handobject.pmap[vis_idx] = torch.as_tensor(handobject.hand_part_ids).to(self.device)[vert_idx[vis_idx]]

            # hand_img = handobject.vis_img(idx=vis_idx, h=400, w=400, draw_maps=False)
            pred_cmap_img, pred_pmap_img = handobject.vis_maps(idx=vis_idx, w=400, h=400)

            # contact_img = np.concatenate([gt_cmap_img, pred_cmap_img], axis=0)
            # part_img = np.concatenate([gt_pmap_img, pred_pmap_img], axis=0)

            import os
            from PIL import Image
            os.makedirs('tmp', exist_ok=True)
            def _to_uint8(img):
                img = np.asarray(img)
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                return img
            Image.fromarray(_to_uint8(pred_cmap_img)).save(f'tmp/{stage}_contact_e{self.current_epoch}_b{batch_idx}.png')
            Image.fromarray(_to_uint8(pred_pmap_img)).save(f'tmp/{stage}_part_e{self.current_epoch}_b{batch_idx}.png')
            # Image.fromarray(_to_uint8(hand_img)).save(f'tmp/{stage}_hand_e{self.current_epoch}_b{batch_idx}.png')

            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/GT_vs_sampled_contact', pred_cmap_img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_sampled_part', pred_pmap_img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/GT_vs_sampled_contact': wandb.Image(pred_cmap_img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_sampled_part': wandb.Image(pred_pmap_img)}, step=self.global_step)
        return losses['total_loss']
    
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
    
    def on_test_epoch_start(self):
        self.all_results = []
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
                "batch_idx", "obj_name", "sampled_grasp",
                "sim_displacement", "penetration_depth", "intersection_volume"
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
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, apply_grid_mask=True)

        obj_hulls = getattr(self.trainer.datamodule, 'test_set').obj_hulls
        obj_name = batch['objName'][0]
        obj_hulls = obj_hulls[obj_name]
        obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').obj_info[obj_name]
        simp_obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh[obj_name]
        obj_mesh = trimesh.Trimesh(obj_mesh_dict['verts'], obj_mesh_dict['faces'])
        simp_obj_mesh = trimesh.Trimesh(simp_obj_mesh_dict['verts'], simp_obj_mesh_dict['faces'])

        # mask = handobject.hand_vert_mask.any(dim=1) # B, H
        # print(f"Average number of visible hand vertices: {torch.sum(mask, dim=1).float().mean().item()}")
        # return


        ## Test the reconstrucion 
        n_samples = self.cfg.test.get('n_samples', 1)
        handobject.load_from_batch_obj_only(batch, n_samples, obj_template=obj_mesh, vis_obj_template=simp_obj_mesh, obj_hulls=obj_hulls)
        _, n_pts = handobject.obj_verts.shape[:2]
        obj_pc = torch.cat([handobject.obj_verts, handobject.obj_normals], dim=-1)  # B x N x 6
        # lg_contact = handobject.ml_contact
        # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
        # obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3

        # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k) # N x ...
        # obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # N x 3
        # obj_feat, multi_scale_obj_cond = self.grid_ae.encode_object(obj_msdf)
        # obj_pc = torch.cat([obj_msdf_center, obj_feat.unsqueeze(0)], dim=-1)

        ## 'x' only indicates the latent shape; latents are sampled inside the model
        input_data = {'x': torch.randn(n_samples, n_pts, 5, device=self.device),
                      'obj_pc': obj_pc.permute(0, 2, 1).to(self.device)}

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
        self.mano_layer.to(self.device)

        with torch.enable_grad():
            contact, cse = samples.split([1, self.cse_dim], dim=-1)
            thin_objects = ['wineglass', 'mug', 'fryingpan']
            global_pose, mano_pose, mano_shape, mano_trans, init_pose = optimize_pose_contactopt(
                                self.mano_layer, handobject.obj_verts.repeat(n_samples, 1, 1).float(),
                                handobject.obj_normals.repeat(n_samples, 1, 1).float(),
                                contact.squeeze(-1).float(), cse.float(), n_iter=1000, save_history=False,
                                partition_type=self.cfg.correspondence_type, w_pen_cost=40, w_cont_asym=10,
                                hand_cse=self.hand_cse if self.cfg.correspondence_type=='cse' else None,
                                is_thin=torch.LongTensor([obj_name in thin_objects for _ in range(n_samples)]).to(self.device))

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
        ## MPJPE:
        if self.cfg.test_gt:
            mpjpe = np.linalg.norm(handobject.hand_joints.cpu().numpy() - handJ, axis=-1).mean(axis=1) * 1000  # B,
            mpvpe = np.linalg.norm(handobject.hand_verts.cpu().numpy() - handV, axis=-1).mean(axis=1) * 1000  # B,
            result.update({'MPJPE': mpjpe, 'MPVPE': mpvpe})

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

        # Visualize all samples
        vis_idx = 0
        # for vis_idx in range(handV.shape[0]):
        pred_imgs = []
        obj_mesh_dict = getattr(self.trainer.datamodule, f'test_set').obj_info
        # obj_mesh_dict = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh

        obj_templates = [trimesh.Trimesh(obj_mesh_dict[name]['verts'], obj_mesh_dict[name]['faces'])
                        for i, name in enumerate(batch['objName'])]
        handobject._load_templates(idx=vis_idx, obj_templates=obj_templates)
        # gt_cmap_img, gt_pmap_img = handobject.vis_maps(idx=vis_idx, w=400, h=400)
        handobject.contact_map = contact
        Wverts = self.hand_cse.emb2Wvert(cse)
        vert_idx = Wverts.argmax(dim=-1)
        handobject.pmap = torch.zeros(n_samples, n_pts).to(self.device)
        handobject.pmap = torch.as_tensor(handobject.hand_part_ids).to(self.device)[vert_idx.squeeze(0)]

        # hand_img = handobject.vis_img(idx=vis_idx, h=400, w=400, draw_maps=False)
        pcd = o3d.geometry.PointCloud()
        pts_np = handobject.obj_verts[0].cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        contact_np = handobject.contact_map.squeeze(-1)[vis_idx].detach().cpu().numpy()
        contact_np = np.clip(contact_np, 0, 1)
        cmap = plt.colormaps['inferno']
        pcd.colors = o3d.utility.Vector3dVector(cmap(contact_np)[:, :3])

        if self.debug:
            ho_geoms = pred_ho.get_vis_geoms(idx=vis_idx)
            o3d.visualization.draw_geometries([pcd] + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g.translate((0, 0.25, 0)) for g in ho_geoms],
                                            window_name='Predicted Hand-Object')
        else:
            pred_img = pred_ho.vis_img(idx=vis_idx, h=400, w=400)
            pred_cmap_img, pred_pmap_img = handobject.vis_maps(idx=vis_idx, w=400, h=400)
            # pcd_img = geom_to_img([pcd], w=400, h=400)
            pred_imgs.append(np.concatenate([pred_img, pred_cmap_img, pred_pmap_img], axis=0))

        if not self.debug:
            pred_img_grid = np.concatenate(pred_imgs, axis=0)

        # Concatenate all sample images horizontally

        if hasattr(self.logger, 'experiment'):
            if hasattr(self.logger.experiment, 'add_image'):
                # TensorBoardLogger
                global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.test_dataloader()')) + batch_idx
                self.logger.experiment.add_image(f'test/Sampled_grasp', pred_img_grid, global_step, dataformats='HWC')
            elif hasattr(self.logger.experiment, 'log') and not self.debug:
                # WandbLogger - add row to table
                self.test_images_table.add_data(
                    batch_idx,
                    obj_name,
                    wandb.Image(pred_img_grid),
                    float(np.mean(result.get("Simulation Displacement", [0]))),
                    float(np.mean(result.get("Penetration Depth", [0]))),
                    float(np.mean(result.get("Intersection Volume", [0])))
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
            from common.model.sam import SAM
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