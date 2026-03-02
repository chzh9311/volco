import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from copy import copy, deepcopy
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_quaternion
import open3d as o3d
import open3d.visualization as vis
from matplotlib import pyplot as plt

from common.manopth.manopth.manolayer import ManoLayer
from common.utils.physics import StableLoss
from common.utils.vis import (parse_hex_color, o3dmesh_from_trimesh, o3d_arrow, geom_to_img, extract_masked_mesh_components,
                               visualize_grid_contact, visualize_local_grid_with_hand, visualize_recon_hand_w_object,
                               geom_to_img_o3d)
from common.utils.misc import linear_normalize
from common.utils.geometry import (
        flip_x_axis,
        transform_mesh,
        sdf_to_contact,
        calculate_contact_capsule
    )

from common.msdf.utils.msdf import msdf2mlcontact, get_grid, get_grid_points, calc_local_grid_all_pts_gpu, nn_dist_to_mesh_gpu
from common.utils.geometry import GridDistanceToContact

class HandObject:
    def __init__(self, cfg, device, mano_layer=None, normalize=True, apply_grid_mask=False):
        self.device = device
        self.cfg = cfg
        if mano_layer is None:
            self.mano_layer = ManoLayer(mano_root=cfg.mano_root, side='right', use_pca=False, flat_hand_mean=True)
        else:
            self.mano_layer = mano_layer

        self.closed_hand_faces = np.load("data/misc/closed_mano_r_faces.npy")
        self.hand_cse = torch.load(cfg.get('hand_cse_path', 'data/misc/hand_cse.ckpt'), weights_only=True)['state_dict']['embedding_tensor'].detach().to(self.device)
        self.hand_part_ids = torch.argmax(self.mano_layer.th_weights, dim=-1).detach().cpu().numpy()
        self.apply_grid_mask = apply_grid_mask ## Only apply this mask when training the old version of GRIDAE for backward compatibility.
        self.hand_verts = None
        self.hand_pose = None
        self.hand_root_rot = None
        self.hand_trans = None
        # self.obj_verts = None
        self.contact_map = None
        self.part_map = None
        self.obj_normals = None
        self.hand_joints = None
        self.hand_sides = None
        self.hand_models = []
        self.obj_models = []
        self.vis_obj_models = []
        self.obj_hulls = []
        self.obj_names = []
        self.obj_rot = None
        self.obj_trans = None
        self.normalize = normalize
        self.obj_com = None
        self.obj_inertia = None
        self.contact_unit = cfg.contact_unit
        self.correspondence_type = cfg.get('correspondence_type', 'cse')
        self.normalized_coords = get_grid(kernel_size=self.cfg.msdf.kernel_size, device=self.device).reshape(-1, 3).float()

    def __copy__(self):
        new_ho = HandObject(cfg=self.cfg, device=self.device, normalize=self.normalize)
        new_ho.obj_names = self.obj_names
        new_ho.mano_layer = self.mano_layer
        new_ho.hand_part_ids = copy(self.hand_part_ids)
        new_ho.hand_sides = copy(self.hand_sides)
        # new_ho.obj_verts = copy(self.obj_verts)
        new_ho.hand_verts = copy(self.hand_verts)
        new_ho.hand_root_rot = copy(self.hand_root_rot)
        new_ho.hand_pose = copy(self.hand_pose)
        new_ho.hand_trans = copy(self.hand_trans)
        new_ho.contact_map = copy(self.contact_map)
        new_ho.part_map = copy(self.part_map)
        # new_ho.obj_normals = copy(self.obj_normals)
        new_ho.hand_joints = copy(self.hand_joints)
        new_ho.hand_models = deepcopy(self.hand_models)
        new_ho.obj_models = deepcopy(self.obj_models)
        new_ho.vis_obj_models = deepcopy(self.vis_obj_models)
        new_ho.obj_hulls = deepcopy(self.obj_hulls)
        new_ho.obj_rot = copy(self.obj_rot)
        new_ho.obj_trans = copy(self.obj_trans)
        new_ho.obj_com = copy(self.obj_com)
        new_ho.obj_inertia = copy(self.obj_inertia)
        new_ho.batch_size = self.batch_size
        return new_ho

    def load_from_batch(self, batch, obj_templates=None, vis_obj_template=None, obj_hulls=None, pool=None):
        """
        Load the sampled vertices of objects from batched data. Used for training & testing.
        Force labels are also loaded.
        Used in training and validating
        """
        # self.obj_verts = batch['objSamplePts'].clone()
        # self.obj_normals = batch['objSampleNormals'].clone()
        self.obj_rot = batch['objRot'].clone().to(self.device).float()
        self.obj_trans = batch['objTrans'].clone().to(self.device).float()
        self.obj_com = batch['objCoM'].clone().to(self.device).float()
        self.obj_inertia = batch['objInertia'].clone().to(self.device).float()
        self.obj_mass = batch['objMass'].clone().to(self.device).float()
        self.obj_names = batch['objName']
        self.hand_pose = batch['handPose'].clone().to(self.device).float()
        self.hand_root_rot = batch['handRot'].clone().to(self.device).float()
        self.hand_trans = batch['handTrans'].clone().to(self.device).float()
        self.hand_verts = batch['handVerts'].clone().to(self.device).float()
        self.hand_normals = batch['handNormals'].clone().to(self.device).float()
        self.hand_joints = batch['handJoints'].clone().to(self.device).float()
        self.cano_joints = batch['canoJoints'].clone().to(self.device).float()
        self.batch_size = self.obj_rot.shape[0]
        # handV, handJ = batch['handVerts'].clone().cpu().numpy(), batch['handJoints'][:, :16].clone().cpu().numpy()
        # self.hand_models = [trimesh.Trimesh(handV[i], self.closed_hand_faces.copy()) for i in range(handV.shape[0])]

        # hand_frames = batch['handPartT'].clone()

        # Do transformations
        objT = torch.eye(4).to(self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        objR = axis_angle_to_matrix(self.obj_rot)
        objT[:, :3, :3] = objR
        objT[:, :3, 3] = self.obj_trans

        # Load object templates
        if obj_templates is not None:
            self.obj_models = []
            for b in range(self.batch_size):
                obj_mesh = copy(obj_templates[b])
                self.obj_models.append(obj_mesh)

        # Load simplified visualization templates
        if vis_obj_template is not None:
            self.vis_obj_models = []
            for b in range(self.batch_size):
                vis_mesh = copy(vis_obj_template[b])
                self.vis_obj_models.append(vis_mesh)

        # Load object hulls
        if obj_hulls is not None:
            self.obj_hulls = []
            for b in range(self.batch_size):
                ohs = []
                for h in obj_hulls[b]:
                    h0 = copy(h)
                    ohs.append(h0)
                self.obj_hulls.append(ohs)

        self.grav_dire = torch.tensor([[0, 0, -1]], dtype=torch.float32, device=self.device).repeat(self.batch_size, 1)
        if self.normalize:
            # Apply inverse transformation objT^-1 to hand related attributes
            # Compute inverse transformation
            if 'aug_rot' in batch:
                self.augR = batch['aug_rot'].clone().to(self.device).float()
            else:
                self.augR = torch.eye(3).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)
            
            self.obj_inertia = self.augR @ self.obj_inertia @ self.augR.transpose(-1, -2)
            objT_inv = torch.eye(4).to(self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
            objT_inv[:, :3, :3] = self.augR @ objT[:, :3, :3].transpose(-1, -2)  # R^T
            objT_inv[:, :3, 3:4] = - objT_inv[:, :3, :3] @ objT[:, :3, 3:4] - self.augR @ self.obj_com.unsqueeze(-1)  # -R^T * t
            self.grav_dire = (objT_inv[:, :3, :3] @ self.grav_dire.unsqueeze(-1)).squeeze(-1)

            # Transform hand vertices
            homo_hand_verts = F.pad(self.hand_verts, (0, 1), 'constant', 1)
            self.hand_verts = (objT_inv.unsqueeze(1) @ homo_hand_verts.unsqueeze(-1))[:, :, :3, 0]

            # Transform hand joints
            homo_hand_joints = F.pad(self.hand_joints, (0, 1), 'constant', 1)
            self.hand_joints = (objT_inv.unsqueeze(1) @ homo_hand_joints.unsqueeze(-1))[:, :, :3, 0]

            root_j = self.cano_joints[:, 0]
            ## Transform hand rotation & translation
            self.hand_root_rot = matrix_to_axis_angle(objT_inv[:, :3, :3] @ axis_angle_to_matrix(self.hand_root_rot))
            self.hand_trans = (objT_inv[:, :3, :3] @ (self.hand_trans + root_j).unsqueeze(-1)).squeeze(-1) + objT[:, :3, 3] - root_j
            # self.hand_trans = (objT_inv[:, :3, :3] @ self.hand_trans[:, :, None]).squeeze(-1) + objT_inv[:, :3, 3]

            # Compute and transform hand normals
            # self.hand_normals = torch.stack([
            #     torch.as_tensor(self.hand_models[i].vertex_normals, dtype=torch.float32, device=self.device)
            #     for i in range(len(self.hand_models))
            # ], dim=0)
            self.hand_normals = (objT_inv[:, :3, :3].unsqueeze(1) @ self.hand_normals.unsqueeze(-1)).squeeze(-1)

            # Apply inverse transformation objT^-1 to object vertices and normals
            # (they are already transformed when loaded from batch)
            # homo_obj_verts = F.pad(self.obj_verts, (0, 1), 'constant', 1)
            # self.obj_verts = (objT_inv.unsqueeze(1) @ homo_obj_verts.unsqueeze(-1))[:, :, :3, 0]

            # Transform object normals with inverse transformation
            # self.obj_normals = (objT_inv[:, :3, :3].unsqueeze(1) @ self.obj_normals.unsqueeze(-1)).squeeze(-1)

            # Transform object meshes and convex hulls (canonical -> CoM-centered + augR)
            for i in range(len(self.obj_models)):
                T = np.eye(4)
                T[:3, :3] = self.augR[i].detach().cpu().numpy()
                T[:3, 3:4] = -T[:3, :3] @ self.obj_com[i].view(3, 1).detach().cpu().numpy()
                self.obj_models[i] = transform_mesh(self.obj_models[i], T)
            for i in range(len(self.vis_obj_models)):
                T = np.eye(4)
                T[:3, :3] = self.augR[i].detach().cpu().numpy()
                T[:3, 3:4] = -T[:3, :3] @ self.obj_com[i].view(3, 1).detach().cpu().numpy()
                self.vis_obj_models[i] = transform_mesh(self.vis_obj_models[i], T)
            for i in range(len(self.obj_hulls)):
                T = np.eye(4)
                T[:3, :3] = self.augR[i].detach().cpu().numpy()
                T[:3, 3:4] = -T[:3, :3] @ self.obj_com[i].view(3, 1).detach().cpu().numpy()
                for j in range(len(self.obj_hulls[i])):
                    self.obj_hulls[i][j] = transform_mesh(self.obj_hulls[i][j], T)
        else:
            # When normalize=False, object vertices and normals are already transformed
            # Only transform object models and hulls which are loaded from canonical space

            # Transform object models
            if obj_templates is not None:
                for i in range(len(self.obj_models)):
                    objT_np = objT[i].detach().cpu().numpy()
                    self.obj_models[i] = transform_mesh(self.obj_models[i], objT_np)

            # Transform simplified visualization models
            if vis_obj_template is not None:
                for i in range(len(self.vis_obj_models)):
                    objT_np = objT[i].detach().cpu().numpy()
                    self.vis_obj_models[i] = transform_mesh(self.vis_obj_models[i], objT_np)

            # Transform object hulls
            if obj_hulls is not None:
                for i in range(len(self.obj_hulls)):
                    objT_np = objT[i].detach().cpu().numpy()
                    for j in range(len(self.obj_hulls[i])):
                        self.obj_hulls[i][j] = transform_mesh(self.obj_hulls[i][j], objT_np)
            # self.obj_com = self.obj_verts.mean(dim=1, keepdim=True)
            self.obj_verts = self.obj_verts - self.obj_com
            self.hand_verts = self.hand_verts - self.obj_com
            self.hand_joints = self.hand_joints - self.obj_com
        
        ## Compute Local grid contact representation
        if self.contact_unit == 'grid':
            self.obj_msdf = batch['objMsdf'].clone().to(self.device).float()
            self.obj_msdf_grad = batch['objMsdfGrad'].clone().to(self.device).float()
            self.adj_point_indices = [b.to(self.device) for b in batch['adjPointIndices']]
            self.n_adj_pt = batch['nAdjPoints'].clone().to(self.device)
            self.msdf_grad = self.obj_msdf_grad

            K = self.cfg.msdf.kernel_size
            scale = self.cfg.msdf.scale
            contact_method = self.cfg.msdf.get('contact_method', 2)
            grid_dist_to_contact = GridDistanceToContact(scale, K, method=contact_method)
            mano_faces = self.mano_layer.th_faces.to(self.device)
            cse_dim = self.hand_cse.shape[-1]

            # Pre-allocate output tensors
            local_grid_contact = torch.zeros(
                self.batch_size, self.obj_msdf.shape[1], K, K, K, 1,
                device=self.device, dtype=torch.float32)
            local_grid_cse = torch.zeros(
                self.batch_size, self.obj_msdf.shape[1], K, K, K, cse_dim,
                device=self.device, dtype=torch.float32)
            all_grid_mask = torch.zeros(
                self.batch_size, self.obj_msdf.shape[1],
                device=self.device, dtype=torch.bool)
            all_verts_mask = torch.zeros(
                self.batch_size, self.obj_msdf.shape[1], self.hand_verts.shape[1],
                device=self.device, dtype=torch.bool)
            all_ho_dist = torch.zeros(
                self.batch_size, self.obj_msdf.shape[1], self.hand_verts.shape[1],
                device=self.device, dtype=torch.float32)

            for b in range(self.batch_size):
                grid_distance, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point = calc_local_grid_all_pts_gpu(
                    contact_points=self.obj_msdf[b, :, -3:],
                    normalized_coords=self.normalized_coords,
                    hand_verts=self.hand_verts[b],
                    faces=mano_faces,
                    kernel_size=K,
                    grid_scale=scale,
                    apply_grid_mask=self.apply_grid_mask
                )

                all_grid_mask[b] = grid_mask
                all_verts_mask[b] = verts_mask
                all_ho_dist[b] = ho_dist

                M = grid_mask.sum().item()
                if M == 0:
                    continue

                # Convert distance to contact
                contact_vals = grid_dist_to_contact(grid_distance)  # (M, K, K, K, 1)
                local_grid_contact[b, grid_mask] = contact_vals

                # CSE via barycentric interpolation on nearest triangle
                nn_face_flat = nn_face_idx.reshape(-1)  # (M * K^3,)
                nn_point_flat = nn_point.reshape(-1, 3)  # (M * K^3, 3)

                nn_vert_idx = mano_faces[nn_face_flat]  # (M*K^3, 3)
                face_verts = self.hand_verts[b][nn_vert_idx]  # (M*K^3, 3, 3)
                face_cse = self.hand_cse[nn_vert_idx]  # (M*K^3, 3, cse_dim)

                # Barycentric weights via matrix inversion
                face_verts_T = face_verts.transpose(-1, -2)  # (M*K^3, 3, 3)
                A = face_verts_T + 1e-6 * torch.eye(3, device=face_verts_T.device).unsqueeze(0)
                w = torch.linalg.solve(A, nn_point_flat.unsqueeze(-1))  # (M*K^3, 3, 1)
                w = torch.clamp(w, 0, 1)
                w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # (M*K^3, 3, 1)

                grid_hand_cse = (face_cse * w).sum(dim=1)  # (M*K^3, cse_dim)
                grid_hand_cse = grid_hand_cse.reshape(M, K, K, K, cse_dim)
                local_grid_cse[b, grid_mask] = grid_hand_cse

            self.obj_pt_mask = all_grid_mask
            self.hand_vert_mask = all_verts_mask  # (B, N, V) bool: which hand verts are within grid_scale of each center
            # n_ho_dist: 0 -> -1; 1 -> 0; infty -> 1
            min_ho_dist = all_ho_dist.min(dim=-1)[0] / scale
            self.n_ho_dist = 1 - 2 / (min_ho_dist + 1)
            self.ml_contact = torch.cat([local_grid_contact, local_grid_cse], dim=-1)


        elif self.contact_unit == 'point':
        ## Calculate contacts
            self.obj_verts = []
            self.obj_normals = []
            for obj_mesh in self.obj_models:
                points, fidx = trimesh.sample.sample_surface(obj_mesh, self.cfg.msdf.kernel_size ** 3 * self.cfg.msdf.num_grids)
                self.obj_verts.append(points)
                self.obj_normals.append(obj_mesh.face_normals[fidx])
            self.obj_verts = torch.as_tensor(np.stack(self.obj_verts, axis=0), dtype=torch.float32, device=self.device)  # (B, N*K^3, 3)
            self.obj_normals = torch.as_tensor(np.stack(self.obj_normals, axis=0), dtype=torch.float32, device=self.device)  # (B, N*K^3, 3)

            obj_cmap, _, nn_idx = calculate_contact_capsule(self.hand_verts, self.hand_normals,
                                                            self.obj_verts, self.obj_normals)
            self.contact_map = obj_cmap.to(self.device).squeeze(-1)
            if self.correspondence_type == 'part':
                self.pmap = torch.as_tensor(self.hand_part_ids).to(self.device)[nn_idx].squeeze(-1)
                self.part_map = F.one_hot(self.pmap, 16).float().to(self.device)
                self.obj2hand_nn_idx = nn_idx.to(self.device)

            elif self.correspondence_type == 'cse':
                grid_points_all = self.obj_verts.reshape(-1, 3)  # (M * K^3, 3)

                # GPU nearest distance using Kaolin
                cse_dim = self.hand_cse.shape[-1]
                self.point_cse = torch.zeros(
                    self.batch_size, self.obj_verts.shape[1], cse_dim,
                    device=self.device, dtype=torch.float32)
                mano_faces = self.mano_layer.th_faces.to(self.device)
                for b in range(self.batch_size):
                    grid_points_all = self.obj_verts[b].reshape(-1, 3)  # (M * K^3, 3)
                    nn_dist, nn_face_idx, nn_point = nn_dist_to_mesh_gpu(
                        grid_points_all, self.hand_verts[b], mano_faces
                    )

                    nn_face_flat = nn_face_idx.reshape(-1)  # (M * K^3,)
                    nn_point_flat = nn_point.reshape(-1, 3)  # (M * K^3, 3)

                    nn_vert_idx = mano_faces[nn_face_flat]  # (M*K^3, 3)
                    face_verts = self.hand_verts[b][nn_vert_idx]  # (M*K^3, 3, 3)
                    face_cse = self.hand_cse[nn_vert_idx]  # (M*K^3, 3, cse_dim)

                    # Barycentric weights via matrix inversion
                    face_verts_T = face_verts.transpose(-1, -2)  # (M*K^3, 3, 3)
                    A = face_verts_T + 1e-6 * torch.eye(3, device=face_verts_T.device).unsqueeze(0)
                    w = torch.linalg.solve(A, nn_point_flat.unsqueeze(-1))  # (M*K^3, 3, 1)
                    w = torch.clamp(w, 0, 1)
                    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # (M*K^3, 3, 1)

                    grid_hand_cse = (face_cse * w).sum(dim=1)  # (M*K^3, cse_dim)
                    grid_hand_cse = grid_hand_cse.reshape(-1, cse_dim)
                    self.point_cse[b] = grid_hand_cse


    @property
    def nhandV(self):
        # Return the canonical hand vertices (after inverse transformation)
        root_j = self.cano_joints[:, 0]
        R = axis_angle_to_matrix(self.hand_root_rot)
        nhandV = (self.hand_verts - root_j.unsqueeze(1)) @ R + root_j.unsqueeze(1) - self.hand_trans.unsqueeze(1)
        return nhandV
    

    def calculate_stable_loss(self, pred_grid_contact):
        B, N, D = self.obj_msdf.shape
        K = self.cfg.msdf.kernel_size
        scale = self.cfg.msdf.scale
        K3 = K ** 3

        # Split MSDF into SDF values and centres
        sdf_vals = self.obj_msdf[:, :, :K3]        # (B, N, K^3)
        centres = self.obj_msdf[:, :, K3:]          # (B, N, 3)

        # Expand centres into per-grid-point coordinates: (B, N, 1, 3) + (1, 1, K^3, 3) -> (B, N, K^3, 3)
        all_pts = centres.unsqueeze(2) + self.normalized_coords[None, None, :, :] * scale
        all_pts = all_pts.reshape(B, N * K3, 3)

        # Flatten SDF, contact, grad, n_adj to (B, N*K^3, ...)
        sdf_flat = sdf_vals.reshape(B, N * K3)
        sdf_grad = self.msdf_grad.reshape(B, N * K3, 3)
        n_adj = self.n_adj_pt.view(B, N * K3)
        ## Denormalize SDF values
        sdf_flat = sdf_flat * scale * np.sqrt(3)

        loss = self.stable_loss(sdf_flat, all_pts, pred_grid_contact, sdf_grad, n_adj,
                                obj_mass=self.obj_mass, gravity_direction=self.grav_dire, J=self.obj_inertia)
        return loss

    ## ---- Visualization wrappers ---- ##

    def vis_grid_contact(self, obj_templates=None, idx=0, w=500, h=500, bbox_alpha=0.2):
        """
        Visualize grid-level contact as coloured bounding boxes on the object.

        :param obj_templates: list of trimesh objects (one per batch element); if None, uses stored vis_obj_models
        :param idx: batch index
        :return: (img, vis_geoms)
        """
        obj_mesh, _ = self._load_templates(idx, obj_templates)
        K = self.cfg.msdf.kernel_size
        scale = self.cfg.msdf.scale
        contact_pts = self.obj_msdf[idx, :, -3:].detach().cpu().numpy()  # (N, 3)
        # Per-centre contact: max contact value inside each grid cell
        pt_contact = self.ml_contact[idx, :, :, :, :, 0].reshape(-1, K**3).max(dim=-1)[0].detach().cpu().numpy()
        return visualize_grid_contact(contact_pts, pt_contact, scale, obj_mesh, w, h, bbox_alpha)

    def vis_all_grid_points(self, obj_templates=None, idx=0, w=500, h=500, hue='sdf'):
        """
        Visualize all expanded grid points together with the object mesh.

        :param obj_templates: list of trimesh objects (one per batch element); if None, uses stored vis_obj_models
        :param idx: batch index
        :param w: image width
        :param h: image height
        :param hue: 'sdf' — blue/white/red by signed distance;
                    'overlap' — inferno colormap by n_adj_pt (grid overlap count)
        :return: (img, vis_geoms)
        """
        obj_mesh, _ = self._load_templates(idx, obj_templates)

        K = self.cfg.msdf.kernel_size
        K3 = K ** 3
        scale = self.cfg.msdf.scale

        # Compute all_pts and sdf_flat exactly as in calculate_stable_loss
        sdf_vals = self.obj_msdf[idx, :, :K3]       # (N, K^3)
        centres = self.obj_msdf[idx, :, K3:]         # (N, 3)
        all_pts = centres.unsqueeze(1) + self.normalized_coords[None, :, :] * scale  # (N, K^3, 3)
        all_pts = all_pts.reshape(-1, 3).detach().cpu().numpy()
        sdf_flat = sdf_vals.reshape(-1).detach().cpu().numpy()

        if hue == 'sdf':
            # Color: blue (inside, sdf<0) -> white (sdf=0) -> red (outside, sdf>0)
            sdf_min = sdf_flat.min()
            sdf_max = sdf_flat.max()
            colors = np.zeros((len(sdf_flat), 3))
            neg_mask = sdf_flat < 0
            pos_mask = ~neg_mask
            if sdf_min < 0:
                intensity = np.clip(np.abs(sdf_flat[neg_mask]) / abs(sdf_min), 0, 1)
                colors[neg_mask, 0] = 1 - intensity
                colors[neg_mask, 1] = 1 - intensity
                colors[neg_mask, 2] = 1
            if sdf_max > 0:
                intensity = np.clip(sdf_flat[pos_mask] / sdf_max, 0, 1)
                colors[pos_mask, 0] = 1
                colors[pos_mask, 1] = 1 - intensity
                colors[pos_mask, 2] = 1 - intensity
        elif hue == 'overlap':
            # Color by n_adj_pt: number of overlapping grids at each point
            n_adj = self.n_adj_pt[idx].reshape(-1).detach().cpu().numpy()  # (N*K^3,)
            heat_cmap = plt.colormaps['inferno']
            n_adj_max = n_adj.max()
            normalized = n_adj / n_adj_max if n_adj_max > 0 else n_adj
            colors = heat_cmap(normalized)[:, :3]
        else:
            raise ValueError(f"Unknown hue mode: {hue!r}. Use 'sdf' or 'overlap'.")

        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Build object mesh
        obj_o3d = o3dmesh_from_trimesh(obj_mesh, (0.5, 0.5, 0.5))

        default_mat = vis.rendering.MaterialRecord()
        default_mat.shader = 'defaultLit'
        obj_mat = vis.rendering.MaterialRecord()
        obj_mat.shader = "defaultLitTransparency"
        obj_mat.base_color = [0.5, 0.5, 0.5, 0.5]

        vis_geoms = [
            {'name': 'sdf_points', 'geometry': pcd, 'material': default_mat},
            {'name': 'object', 'geometry': obj_o3d, 'material': obj_mat},
        ]

        # img = geom_to_img(vis_geoms, w, h)
        return None, vis_geoms

    def vis_local_grid_with_hand(self, obj_templates=None, idx=0, pt_idx=0):
        """
        Visualize a single local grid (SDF + contact + CSE + SDF gradient arrows)
        together with the hand mesh.
        The object is normalized to align the center of mass

        :param obj_templates: list of trimesh objects (one per batch element); if None, uses stored vis_obj_models
        :param idx: batch index
        :param pt_idx: MSDF centre index within the sample
        :return: list of Open3D geometries
        """
        obj_mesh, _ = self._load_templates(idx, obj_templates)
        K = self.cfg.msdf.kernel_size
        K3 = K ** 3
        scale = self.cfg.msdf.scale

        # SDF values for this centre: (K, K, K, 1)
        sdf = self.obj_msdf[idx, pt_idx, :K3].detach().cpu().numpy().reshape(K, K, K, 1)
        # Contact + CSE: (K, K, K, 1+cse_dim)
        contact_cse = self.ml_contact[idx, pt_idx].detach().cpu().numpy()  # (K, K, K, 1+cse_dim)
        # Concatenate SDF as first channel: (K, K, K, 1 + 1 + cse_dim)
        local_grid = np.concatenate([sdf, contact_cse], axis=-1)

        contact_point = self.obj_msdf[idx, pt_idx, -3:].detach().cpu().numpy()
        hand_verts = self.hand_verts[idx].detach().cpu().numpy()
        hand_cse = self.hand_cse.detach().cpu().numpy()

        geoms = visualize_local_grid_with_hand(
            local_grid, hand_verts, self.closed_hand_faces, hand_cse,
            K, scale, contact_point=contact_point, obj_mesh=obj_mesh)

        # Add SDF gradient arrows at each grid point
        grad = self.msdf_grad[idx, pt_idx].detach().cpu().numpy()  # (K^3, 3)
        sdf_flat = sdf.flatten()  # (K^3,)

        # Compute grid point positions
        indices = np.array([(i, j, k) for i in range(K)
                            for j in range(K)
                            for k in range(K)])
        coords = (2 * indices - (K - 1)) / (K - 1)
        grid_points = contact_point[None, :] + coords * scale

        for i in range(K3):
            direction = grad[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            # Color: blue (inside) -> red (outside), matching SDF coloring
            if sdf_flat[i] < 0:
                color = [0.2, 0.2, 1.0]
            else:
                color = [1.0, 0.2, 0.2]
            arrow = o3d_arrow(grid_points[i], direction, color=color, scale=scale * 0.5)
            geoms.append(arrow)

        return geoms

    def vis_grid_detail(self, obj_templates=None, idx=0, pt_idx=0):
        """
        Two side-by-side panels for one MSDF grid cell:
          Left  — SDF-coloured grid points + object mesh fragment inside the bbox
          Right — contact-coloured grid points + hand mesh fragment inside the bbox

        :param obj_templates: list of trimesh objects (one per batch element); if None, uses stored vis_obj_models
        :param idx: batch index
        :param pt_idx: MSDF centre index within the sample
        :return: list of Open3D geometries for draw_geometries
        """
        obj_mesh, _ = self._load_templates(idx, obj_templates)
        K  = self.cfg.msdf.kernel_size
        K3 = K ** 3
        scale = self.cfg.msdf.scale

        center      = self.obj_msdf[idx, pt_idx, -3:].detach().cpu().numpy()          # (3,)
        sdf_flat    = self.obj_msdf[idx, pt_idx, :K3].detach().cpu().numpy()           # (K^3,)
        contact_flat = self.ml_contact[idx, pt_idx, :, :, :, 0].reshape(-1).detach().cpu().numpy()  # (K^3,)

        # Grid point world coordinates
        indices = np.array([(i, j, k) for i in range(K)
                            for j in range(K) for k in range(K)])
        coords      = (2 * indices - (K - 1)) / (K - 1)          # (K^3, 3), range [-1,1]
        grid_pts    = center[None, :] + coords * scale             # (K^3, 3)

        # ── helpers ──────────────────────────────────────────────────────────

        def _bbox_lineset(origin, color):
            pts = np.array([origin + scale * np.array([sx, sy, sz])
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines  = o3d.utility.Vector2iVector(edges)
            ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
            return ls

        def _clip_mesh_to_box(tmesh, box_center, box_half):
            """Return o3d mesh containing only faces whose centroid lies inside the bbox."""
            verts = np.asarray(tmesh.vertices)
            faces = np.asarray(tmesh.faces)
            face_centers = verts[faces].mean(axis=1)               # (F, 3)
            inside = np.all(np.abs(face_centers - box_center) <= box_half, axis=1)
            if not inside.any():
                return None
            kept_faces  = faces[inside]
            used_verts, inv = np.unique(kept_faces, return_inverse=True)
            new_verts   = verts[used_verts]
            new_faces   = inv.reshape(-1, 3)
            m = o3d.geometry.TriangleMesh()
            m.vertices  = o3d.utility.Vector3dVector(new_verts)
            m.triangles = o3d.utility.Vector3iVector(new_faces)
            m.compute_vertex_normals()
            return m

        # ── SDF coloring (blue inside / red outside) ─────────────────────────
        sdf_min, sdf_max = sdf_flat.min(), sdf_flat.max()
        sdf_colors = np.zeros((K3, 3))
        neg = sdf_flat < 0
        if sdf_min < 0:
            t = np.clip(np.abs(sdf_flat[neg]) / abs(sdf_min), 0, 1)
            sdf_colors[neg]  = np.stack([1-t, 1-t, np.ones_like(t)], axis=1)
        pos = ~neg
        if sdf_max > 0:
            t = np.clip(sdf_flat[pos] / sdf_max, 0, 1)
            sdf_colors[pos]  = np.stack([np.ones_like(t), 1-t, 1-t], axis=1)

        # ── contact coloring (inferno) ────────────────────────────────────────
        cmap = plt.colormaps['inferno']
        contact_colors = cmap(np.clip(contact_flat, 0, 1))[:, :3]

        # ── panel offset: 2.5 × bbox width along X ───────────────────────────
        offset = np.array([scale * 2.5, 0.0, 0.0])

        left_geoms = []
        right_geoms = []

        # ── LEFT PANEL: SDF grid + clipped object mesh ────────────────────────
        pcd_sdf = o3d.geometry.PointCloud()
        pcd_sdf.points = o3d.utility.Vector3dVector(grid_pts)
        pcd_sdf.colors = o3d.utility.Vector3dVector(sdf_colors)
        left_geoms.append(pcd_sdf)
        left_geoms.append(_bbox_lineset(center, [0.2, 0.8, 0.2]))

        obj_clip = _clip_mesh_to_box(obj_mesh, center, np.full(3, scale))
        if obj_clip is not None:
            obj_clip.paint_uniform_color(parse_hex_color("#2A5E8C"))
            obj_clip.compute_vertex_normals()
            left_geoms.append(obj_clip)

        # ── RIGHT PANEL: contact grid + clipped hand mesh ─────────────────────
        pcd_contact = o3d.geometry.PointCloud()
        pcd_contact.points = o3d.utility.Vector3dVector(grid_pts)
        pcd_contact.colors = o3d.utility.Vector3dVector(contact_colors)
        right_geoms.append(pcd_contact)
        right_geoms.append(_bbox_lineset(center, [0.2, 0.8, 0.2]))

        hand_verts = self.hand_verts[idx].detach().cpu().numpy()
        hand_tmesh = trimesh.Trimesh(hand_verts, self.closed_hand_faces.copy(), process=False)
        hand_clip  = _clip_mesh_to_box(hand_tmesh, center, np.full(3, scale))
        if hand_clip is not None:
            hand_clip.paint_uniform_color(parse_hex_color("#F27141"))
            hand_clip.compute_vertex_normals()
            right_geoms.append(hand_clip)

        return left_geoms, right_geoms

    def vis_all_grids_with_hand(self, obj_templates=None, idx=0, grid_idx=0):
        """
        Visualize all MSDF grid bounding boxes together with the hand mesh and object.
        Each grid is drawn as a semi-transparent axis-aligned cube (bbox = 2*scale per side).
        No per-grid features (contact/CSE) are encoded in the color.

        :param obj_templates: list of trimesh objects (one per batch element); if None, uses stored vis_obj_models
        :param idx: batch index
        :param grid_idx: index of specific grid to highlight (default 0)
        :return: list of geometry dicts for o3d.visualization.draw (supports transparency)
        """
        obj_mesh, _ = self._load_templates(idx, obj_templates)
        scale = self.cfg.msdf.scale
        K = self.cfg.msdf.kernel_size

        grid_centers = self.obj_msdf[idx, :, K**3:].detach().cpu().numpy()  # (N, 3)
        N = grid_centers.shape[0]

        box_color = parse_hex_color("#102540")
        highlight_color = parse_hex_color("#B26FE0")

        # 12 edges of a unit box, reused for every wireframe
        _corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        _edges = [
            [0,1],[1,2],[2,3],[3,0],  # bottom face
            [4,5],[5,6],[6,7],[7,4],  # top face
            [0,4],[1,5],[2,6],[3,7],  # verticals
        ]

        geom_list = []

        for i in range(N):
            center = grid_centers[i]
            corners = _corners * (2 * scale) + (center - scale)
            if i == grid_idx:
                # Solid filled box for the highlighted grid
                box = o3d.geometry.TriangleMesh.create_box(
                    width=2 * scale, height=2 * scale, depth=2 * scale)
                box.translate(center - scale * np.ones(3))
                box.paint_uniform_color(highlight_color)
                box.compute_vertex_normals()
                geom_list.append(box)
            else:
                # Wireframe LineSet for non-highlighted grids
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners)
                ls.lines = o3d.utility.Vector2iVector(_edges)
                ls.colors = o3d.utility.Vector3dVector([box_color] * len(_edges))
                geom_list.append(ls)

        hand_verts = self.hand_verts[idx].detach().cpu().numpy()
        hand_mesh = trimesh.Trimesh(hand_verts, self.closed_hand_faces.copy())
        hand_o3d = o3dmesh_from_trimesh(hand_mesh, color="#F27141")
        hand_o3d.compute_vertex_normals()
        geom_list.append(hand_o3d)

        obj_o3d = o3dmesh_from_trimesh(obj_mesh, color="#2A5E8C")
        obj_o3d.compute_vertex_normals()
        geom_list.append(obj_o3d)

        return geom_list

    def vis_recon_hand_w_object(self, handcse, obj_templates=None, idx=0, mask_th=0.02, h=500, w=500):
        """
        Reconstruct hand vertices from ml_contact via recover_hand_verts_from_contact,
        then visualize the masked reconstructed hand with the object mesh.

        :param handcse: HandCSE module instance
        :param obj_templates: list of trimesh objects (one per batch element); if None, uses stored vis_obj_models
        :param idx: batch index
        :param mask_th: threshold for vertex mask in recover_hand_verts_from_contact
        :return: (img, vis_geoms)
        """
        obj_mesh, _ = self._load_templates(idx, obj_templates)

        K = self.cfg.msdf.kernel_size
        K3 = K ** 3
        scale = self.cfg.msdf.scale
        B = self.obj_msdf.shape[0]
        N = self.obj_msdf.shape[1]

        # Flatten contact and CSE from ml_contact (B, N, K, K, K, 1+cse_dim)
        grid_contact = self.ml_contact[:, :, :, :, :, 0].reshape(B, N * K3)
        grid_cse = self.ml_contact[:, :, :, :, :, 1:].reshape(B, N * K3, -1)

        # Compute grid point coordinates
        centres = self.obj_msdf[:, :, K3:]  # (B, N, 3)
        grid_coords = centres.unsqueeze(2) + self.normalized_coords[None, None, :, :] * scale
        grid_coords = grid_coords.reshape(B, N * K3, 3)

        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            handcse, None, grid_contact, grid_cse, grid_coords, mask_th=mask_th)

        hand_verts = pred_hand_verts[idx].detach().cpu().numpy()
        verts_mask = pred_verts_mask[idx].detach().cpu().numpy()
        return visualize_recon_hand_w_object(
            hand_verts, verts_mask, self.closed_hand_faces,
            obj_mesh, self.hand_part_ids, h=h, w=w)

    def load_from_batch_obj_only(self, batch, n_samples, obj_template=None, vis_obj_template=None, obj_hulls=None):
        """
        Load only object-related data from batched data. Used for testing.
        No hand-related variables are loaded - those will be generated later.
        The batch size is 1: loading only one object each time.
        """
        self.batch_size = n_samples
        self.obj_names = batch['objName']
        self.obj_com = batch['objCoM'][0]
        # self.obj_verts = batch['objSamplePts'].clone().to(self.device).float()
        # self.obj_normals = batch['objSampleNormals'].clone().to(self.device).float()

        # Load object templates
        self.augR = torch.eye(3).unsqueeze(0).repeat(self.batch_size, 1, 1).to(self.device)
        if obj_template is not None:
            self.obj_models = []
            for i in range(self.batch_size):
                obj_mesh = copy(obj_template)
                obj_mesh.apply_translation(-self.obj_com.detach().cpu().numpy())
                self.obj_models.append(obj_mesh)

        # Load simplified visualization templates
        if vis_obj_template is not None:
            self.vis_obj_models = []
            for i in range(self.batch_size):
                vis_mesh = copy(vis_obj_template)
                vis_mesh.apply_translation(-self.obj_com.detach().cpu().numpy())
                self.vis_obj_models.append(vis_mesh)

        # Load object hulls
        if obj_hulls is not None:
            self.obj_hulls = []
            for i in range(self.batch_size):
                ohs = []
                for h in obj_hulls:
                    h0 = copy(h)
                    h0.apply_translation(-self.obj_com.detach().cpu().numpy())
                    ohs.append(h0)
                self.obj_hulls.append(ohs)

        # Center object at origin
        # self.obj_com = self.obj_verts.mean(dim=1, keepdim=True)
        # self.obj_verts = self.obj_verts - self.obj_com
        self.obj_msdf = batch['objMsdf'].clone().to(self.device).float()

    def get_99_dim_mano_params(self):
        """
        The mano parameters are: 
        3D translation (3) + 6D root rotation (6) + 6D hand pose (90) = 99D.
        """
        translation = self.hand_trans
        rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_root_rot))
        pose6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_pose.view(-1, 15, 3))).view(-1, 90)
        return torch.cat([translation, rot6d, pose6d], dim=-1)


    def _load_templates(self, idx, obj_templates=None, obj_hull=None):
        # Use pre-stored vis_obj_models if available, otherwise fall back to obj_templates arg
        if self.vis_obj_models:
            obj_mesh = self.vis_obj_models[idx]
        elif self.obj_models:
            obj_mesh = self.obj_models[idx]
        elif obj_templates is not None:
            self.hand_models = []
            self.vis_obj_models = []
            for i in range(len(obj_templates)):
                if not self.normalize:
                    objR = axis_angle_to_matrix(self.obj_rot[i]).detach().cpu().numpy()
                    objt = self.obj_trans[i].detach().cpu().numpy()
                    T = np.eye(4)
                    T[:3, :3] = objR
                    T[:3, 3] = objt
                    mesh = copy(obj_templates[i])
                    mesh.apply_transform(T)
                else:
                    mesh = copy(obj_templates[i])
                T = np.eye(4)
                T[:3, :3] = self.augR[i].detach().cpu().numpy()
                T[:3, 3:4] = -T[:3, :3] @ self.obj_com[i].view(3, 1).detach().cpu().numpy()
                mesh.apply_transform(T)
                self.vis_obj_models.append(mesh)
            obj_mesh = self.vis_obj_models[idx]
        else:
            obj_mesh = None

        if obj_hull is not None:
            ohs = []
            for h in obj_hull:
                h0 = copy(h)
                # if self.hand_sides[idx] == 'left':
                #     flip_x_axis(h0)
                # h0.apply_transform(T)
                ohs.append(h0)
        else:
            ohs = None

        return obj_mesh, ohs

    def get_vis_geoms(self, idx=0, draw_maps=False, draw_hand=True, draw_obj=True, **kwargs):
        if 'obj_templates' in kwargs:
            obj_mesh, _ = self._load_templates(idx=idx, obj_templates=kwargs['obj_templates'])
        elif self.vis_obj_models:
            obj_mesh = self.vis_obj_models[idx]
        elif self.obj_models:
            obj_mesh = self.obj_models[idx]
        else:
            obj_mesh = None

        if self.hand_verts is not None:
            hand_mesh = trimesh.Trimesh(self.hand_verts[idx].detach().cpu().numpy(), self.closed_hand_faces.copy())
        else:
            hand_mesh = None

        # Downsample object mesh for visualization if it has too many faces
        if obj_mesh is not None and len(obj_mesh.faces) > 6000:
            obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=6000)

        default_mat = vis.rendering.MaterialRecord()
        default_mat.shader = 'defaultLit'
        hand_mat = vis.rendering.MaterialRecord()
        hand_mat.shader = "defaultLitTransparency"
        hand_mat.base_color = [0.8, 0.7, 0.5, 0.8]
        obj_mat = vis.rendering.MaterialRecord()
        obj_mat.shader = "defaultLitTransparency"
        obj_mat.base_color = [0.5, 0.5, 0.5, 0.9]
        obj_mat1 = vis.rendering.MaterialRecord()
        obj_mat1.shader = "defaultLitTransparency"
        obj_mat1.base_color = [0.5, 0.9, 0.4, 0.7]
        heat_cmap = plt.colormaps['inferno']
        part_cmap = plt.colormaps['hsv']

        vis_geoms = []

        hand = o3dmesh_from_trimesh(hand_mesh, (0.8, 0.7, 0.5))
        hand_vcolor = part_cmap(self.hand_part_ids / 16)[:, :3] * 0.5 + 0.25
        hand.vertex_colors = o3d.utility.Vector3dVector(hand_vcolor)
        obj0 = o3dmesh_from_trimesh(obj_mesh, (0.5, 0.5, 0.5))
        if draw_hand:
            vis_geoms.append({'name': 'hand', 'geometry': hand, 'material': hand_mat})

        if draw_obj:
            vis_geoms.append({'name': 'object', 'geometry': obj0, 'material': obj_mat})

        if draw_maps:
            offset = np.array([0.25, 0, 0])
            comesh = copy(obj_mesh)
            comesh.apply_translation(offset)
            ## contact upscale
            dists = np.linalg.norm(self.obj_verts[idx].view(1, -1, 3).detach().cpu().numpy()
                                   - obj_mesh.vertices.reshape(-1, 1, 3), axis=-1)
            nn_idx = np.argmin(dists, axis=1)
            up_contact = self.contact_map[idx, nn_idx]
            up_contact = linear_normalize(up_contact, 0, 1).detach().cpu().numpy()
            vertex_colors = heat_cmap(up_contact)[:, :3]
            vertex_colors[up_contact <= 0.1] = 0.1
            comesh = o3dmesh_from_trimesh(comesh, (0.5, 0.5, 0.5))
            comesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            pomesh = copy(obj_mesh)
            pomesh.apply_translation(-offset)
            pomesh = o3dmesh_from_trimesh(pomesh)
            part_confs, part_ids = torch.max(self.part_map, dim=-1)
            up_part_ids = part_ids[idx, nn_idx].detach().cpu().numpy() / 16
            vertex_colors = part_cmap(up_part_ids)[:, :3]
            vertex_colors *= up_contact.reshape(-1, 1) # Regularize by the contact maps
            pomesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            vis_geoms.extend([{'name': 'obj_contacts', 'geometry': comesh, 'material': default_mat},
                              {'name': 'obj_parts', 'geometry': pomesh, 'material': default_mat}])

        return vis_geoms

    def vis_frame(self, idx=0, **kwargs):
        vis_geoms = self.get_vis_geoms(idx, **kwargs)
        o3d.visualization.draw(vis_geoms, show_skybox=False, lookat=[0, 1, 0], eye=[0, -1, 0], up=[0, 0, 1])
    
    def get_masked_recon_hand_meshes(self, recon_hand_verts, vis_idx=0):
        """
        get o3d meshes of reconstructed hand vertices. Only masked vertices are shown.
        :param recon_hand_verts: (B, 778, 3) reconstructed hand vertices
        :param vis_idx: int, index of the hand to visualize
        """
        hand_verts = recon_hand_verts[vis_idx].detach().cpu().numpy()
        hand_mask = self.hand_vert_mask[vis_idx].detach().cpu().numpy()
        geometries = extract_masked_mesh_components(hand_verts, self.closed_hand_faces, hand_mask,
                                                    self.hand_part_ids, create_geometries=True)
        return geometries
    

    def vis_recon_hand_with_object(self, recon_hand_verts, vis_idx=0, w=800, h=600):
        hand_geoms = self.get_masked_recon_hand_meshes(recon_hand_verts, vis_idx=vis_idx)
        vis_mesh = self.vis_obj_models[vis_idx] if self.vis_obj_models else self.obj_models[vis_idx]
        obj_geom = o3dmesh_from_trimesh(vis_mesh, (0.5, 0.5, 0.5))
        img = geom_to_img(hand_geoms + [obj_geom], w=w, h=h, scale=0.6)
        return img


    # def get_ho_features(self, rot_dim=3):
    #     """
    #     Obtain data directly used for training.
    #     The random rotations only affects the output of this function, not modifying the physics representation.
    #     """

    #     if self.hand_verts is not None:
    #         if rot_dim == 3:
    #             hand_pose = self.hand_pose
    #             root_rot = self.hand_root_rot
    #         elif rot_dim == 4:
    #             hand_pose = axis_angle_to_quaternion(self.hand_pose.view(-1, 15, 3)).view(-1, 60)
    #             root_rot = axis_angle_to_quaternion(self.hand_root_rot)
    #         elif rot_dim == 6:
    #             hand_pose = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_pose.view(-1, 15, 3))).view(-1, 90)
    #             root_rot = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_root_rot))
    #         elif rot_dim == 9:
    #             hand_pose = axis_angle_to_matrix(self.hand_pose.view(-1, 15, 3)).view(-1, 135)
    #             root_rot = axis_angle_to_matrix(self.hand_root_rot).view(-1, 9)
    #         else:
    #             raise NotImplementedError

    #         hand_params = {'handVerts': self.hand_verts, 'handPose': hand_pose,
    #                        'handTrans': self.hand_trans, 'handRot': root_rot}
    #     else:
    #         hand_params = None

    #     return self.obj_verts, self.obj_normals, hand_params

    def vis_img(self, idx:int, h:int=600, w:int=800, draw_maps=False, **kwargs) -> np.ndarray:
        """
        Visualize the hand-object as an image.
        pts: N x 3
        returns an image array of h x w x 3
        """
        vis_geoms = self.get_vis_geoms(idx, draw_maps=draw_maps, **kwargs)

        scale = 0.9
        # Separate geometries into three groups
        hand_obj_geoms = [g for g in vis_geoms if g['name'] in ['hand', 'object']]
        img_hand_obj = geom_to_img_o3d(hand_obj_geoms, w, h, scale=scale, concat_axis=kwargs.get('concat_axis', 1))

        if draw_maps:
            contact_geoms = [g for g in vis_geoms if g['name'] == 'obj_contacts']
            part_geoms = [g for g in vis_geoms if g['name'] == 'obj_parts']

            # Render each group separately
            img_contacts = geom_to_img(contact_geoms, w, h, scale=scale)
            img_parts = geom_to_img(part_geoms, w, h, scale=scale)
            ret_img = np.concatenate([img_hand_obj, img_contacts, img_parts], axis=0)
        else:
            ret_img = img_hand_obj

        # Concatenate vertically (along y axis)
        return ret_img
    

    ## Grid contact visualizations


def recover_hand_verts_from_contact(handcse, gt_face_idx, grid_contact, grid_cse, grid_coords, mask_th=2, chunk_size=0):
    """
    :param grid_contact: (B, N) contact values
    :param grid_cse: (B, N, D) contact signature embeddings
    :param grid_coords: (B, N, 3)
    :param chunk_size: If > 0, process in chunks to reduce memory peak. If 0, process all at once.
    """
    batch_size = grid_contact.shape[0]

    # If chunk_size is 0 or batch_size <= chunk_size, process all at once
    if chunk_size <= 0 or batch_size <= chunk_size:
        targetWverts = handcse.emb2Wvert(grid_cse, gt_face_idx)
        # verts_mask = torch.sum(targetWverts, dim=1) > 0.01  # (B, 778)
        weight = (targetWverts * grid_contact.unsqueeze(-1)).transpose(-1, -2)  # (B, 778, K^3)
        all_weight = torch.sum(weight, dim=-1)  # (B, 778)
        # Adaptive threshold: keep top rel_mask_th proportion of vertices
        # k = max(1, int(all_weight.shape[1] * (1 - rel_mask_th)))
        # threshold = all_weight.kthvalue(k, dim=1).values  # (B,)
        # verts_mask = all_weight > threshold.unsqueeze(1)  # (B, 778)
        verts_mask = all_weight > mask_th
        weight[verts_mask] = weight[verts_mask] / (torch.sum(weight[verts_mask], dim=-1, keepdim=True) + 1e-8)
        pred_verts = weight @ grid_coords  # (B, 778, 3)
        return pred_verts, verts_mask

    # Process in chunks to reduce memory peak
    pred_verts_list = []
    verts_mask_list = []
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)

        chunk_grid_contact = grid_contact[i:end_idx]
        chunk_grid_cse = grid_cse[i:end_idx]
        chunk_grid_coords = grid_coords[i:end_idx]

        targetWverts = handcse.emb2Wvert(chunk_grid_cse, gt_face_idx)
        weight = (targetWverts * chunk_grid_contact.unsqueeze(-1)).transpose(-1, -2)  # (chunk, 778, K^3)
        chunk_all_weight = torch.sum(weight, dim=-1)  # (chunk, 778)
        # chunk_k = max(1, int(chunk_all_weight.shape[1] * (1 - rel_mask_th)))
        # chunk_threshold = chunk_all_weight.kthvalue(chunk_k, dim=1).values  # (chunk,)
        # chunk_verts_mask = chunk_all_weight > chunk_threshold.unsqueeze(1)  # (chunk, 778)
        chunk_verts_mask = chunk_all_weight > mask_th
        weight[chunk_verts_mask] = weight[chunk_verts_mask] / (torch.sum(weight[chunk_verts_mask], dim=-1, keepdim=True) + 1e-8)
        chunk_pred_verts = weight @ chunk_grid_coords  # (chunk, 778, 3)

        pred_verts_list.append(chunk_pred_verts)
        verts_mask_list.append(chunk_verts_mask)

    pred_verts = torch.cat(pred_verts_list, dim=0)
    verts_mask = torch.cat(verts_mask_list, dim=0)
    return pred_verts, verts_mask