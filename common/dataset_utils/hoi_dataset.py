import numpy as np
from copy import deepcopy
import trimesh
import torch
import os.path as osp
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from common.utils.geometry import transform_obj, GridDistanceToContact, rodrigues_rot, grid_reorder_id_and_rot
from common.utils.vis import o3dmesh_from_trimesh, o3dmesh
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import h5py

def get_kine_parent(idx):
    if idx in [1, 4, 7, 10, 13]:
        return 0
    else:
        return idx - 1

kinetree = {'Index': [0, 1, 2, 3, 17], 'Middle': [0, 4, 5, 6, 18], 'Little': [0, 10, 11, 12, 20],
          'Ring': [0, 7, 8, 9, 19], 'Thumb': [0, 13, 14, 15, 16]}
    

class BaseHOIDataset(Dataset):
    def __init__(self, cfg, split, load_msdf=False, load_grid_contact=False, object_only=False):
        super(BaseHOIDataset, self).__init__()
        self.data_dir = cfg.dataset_path
        self.preprocessed_dir = cfg.get('preprocessed_dir', 'data/preprocessed')
        self.downsample_rate = cfg.downsample_rate[split]
        self.split = split
        self.object_only = object_only
        # self.n_obj_samples = cfg.object_sample
        self.mano_root = cfg.mano_root
        self.augment = cfg.augment and (split == 'train')
        self.hand_sides = None
        self.lh_data = None
        if load_msdf and 'msdf' in cfg:
            self.msdf_path = osp.join(self.preprocessed_dir, cfg.dataset_name,
                                    f'msdf_{cfg.msdf.num_grids}_{cfg.msdf.kernel_size}_{int(cfg.msdf.scale*1000):02d}mm')
            self.msdf_scale = cfg.msdf.scale
            self.msdf_kernel_size = cfg.msdf.kernel_size
            self.grid_dist_to_contact = GridDistanceToContact.from_config(cfg.msdf, method=cfg.msdf.contact_method)
        else:
            self.msdf_path = None
            self.grid_dist_to_contact = None
        
        self._load_data()

        if self.downsample_rate > 1:
            print(f"Downsampling data with rate {self.downsample_rate} for split {self.split}")
            self.rh_data = self.sample_frames(self.rh_data)
            self.object_data = self.sample_frames(self.object_data)
            self.frame_names = self.sample_frames(self.frame_names)
            if self.hand_sides is not None:
                self.hand_sides = self.sample_frames(self.hand_sides)
                self.lh_data = self.sample_frames(self.lh_data)

        if load_grid_contact:
            self.grid_contact_ds = osp.join(self.preprocessed_dir, cfg.dataset_name, split,
                            f'hand_grid_contact_{cfg.msdf.num_grids}_{cfg.msdf.kernel_size}_{int(cfg.msdf.scale*1000):02d}mm_every{self.downsample_rate}.h5')
        else:
            self.grid_contact_ds = None

        self.hand_cse = torch.load(cfg.get('hand_cse_path', 'data/misc/hand_cse.ckpt'), weights_only=False)['state_dict']['embedding_tensor'].detach().cpu().numpy()
        ## Calculate partitions

    def _load_data(self):
        self.obj_info = None
        self.rh_data = None
        self.object_data = None
        self.frame_names = None


    def sample_frames(self, data):
        if type(data) is dict:
            for k, v in data.items():
                data[k] = v[::self.downsample_rate]
        else:
            data = data[::self.downsample_rate]

        return data

    def __len__(self):
        # return 900
        if self.object_only:
            return len(self.test_objects)
        else:
            return len(self.frame_names)

    def __getitem__(self, idx):
        sample = {}
        if self.augment:
            reorder_id, Rot = grid_reorder_id_and_rot(self.msdf_kernel_size, np.random.randint(0, 24))
            sample['aug_rot'] = Rot
        if not self.object_only:
            fname_path = self.frame_names[idx].split('/')
            sbj_id = fname_path[2]
            obj_name = fname_path[3].split('_')[0]
            obj_rot = self.object_data['global_orient'][idx]
            obj_trans = self.object_data['transl'][idx]
            obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3

            ## Transform the vertices:
            objR = axis_angle_to_matrix(obj_rot).detach().cpu().numpy()
            if self.dataset_name == 'grab':
                objR = objR.T
            objt = obj_trans.detach().cpu().numpy()

            # if self.augment:
            #     objR = Rot @ objR
            #     objt = (Rot @ objt[:, np.newaxis])[:, 0]
            obj_sample_pts = obj_sample_pts @ objR.T + objt[np.newaxis, :]
            obj_sample_normals = obj_sample_normals @ objR.T

            if self.hand_sides is not None:
                hand_side = self.hand_sides[idx]
                if hand_side == 'left':
                    # change the hand
                    hmodels = self.lh_models
                    hdata = self.lh_data
                else:
                    hmodels = self.rh_models
                    hdata = self.rh_data
            else:
                hand_side = 'right'
                hmodels = self.rh_models
                hdata = self.rh_data

            sample.update({
                'frameName': '/'.join(fname_path[2:]),
                'objName': obj_name,
                'sbjId': sbj_id,
                'objSamplePts': obj_sample_pts,
                'objSampleNormals': obj_sample_normals,
                'objTrans': objt,
                'objRot': Rotation.from_matrix(objR).as_rotvec(),
                'handSide': hand_side,
            })


            # hand_out = hmodels[sbj_id](
            #     global_orient=hdata['global_orient'][idx:idx+1],
            #     hand_pose=hdata['fullpose'][idx:idx+1],
            #     transl=hdata['transl'][idx:idx+1])
            # handV, handJ = hand_out.vertices[0].detach().cpu().numpy(), hand_out.joints[0].detach().cpu().numpy()
            hand_rot = hdata['global_orient'][idx]
            hand_pose = hdata['fullpose'][idx].flatten()
            hand_trans = hdata['transl'][idx]
            handV, handJ, part_T = hmodels[sbj_id](
                th_pose_coeffs=torch.cat([hand_rot, hand_pose], dim=0).unsqueeze(0),
                th_trans=hand_trans.unsqueeze(0))
            handV, handJ, part_T = handV[0].detach().cpu().numpy(), handJ[0].detach().cpu().numpy(), part_T[0].detach().cpu().numpy()
            ## canonical joints for calculating transformations (use cache if available)
            if hasattr(self, 'cano_joints_cache') and sbj_id in self.cano_joints_cache:
                canoJ = self.cano_joints_cache[sbj_id]
            else:
                _, canoJt, _ = hmodels[sbj_id](th_pose_coeffs=torch.zeros(1, 48))
                canoJ = canoJt[0].detach().cpu().numpy()
            # model = ManoLayer(mano_root=self.mano_root, flat_hand_mean=True)
            hand_part_ids = torch.argmax(hmodels[sbj_id].th_weights, dim=-1).detach().cpu().numpy()

            handN = trimesh.Trimesh(handV, hmodels[sbj_id].th_faces).vertex_normals

                # handV = handV @ Rot.T
                # handJ = handJ @ Rot.T
                # # part_T[:, :3, :3] = Rot @ part_T[:, :3, :3]
                # handN = handN @ Rot.T
                # hand_rot = Rotation.from_matrix(Rot @ Rotation.from_rotvec(hand_rot.detach().cpu().numpy()).as_matrix())
                # hand_trans = (Rot @ hand_trans.detach().cpu().numpy()[:, np.newaxis]) + (Rot - np.eye(3)) @ canoJ[0, :, np.newaxis]
                # hand_trans = hand_trans[:, 0]
                # new_handV, _, _ = hmodels[sbj_id](
                #     th_pose_coeffs=torch.cat([torch.as_tensor(hand_rot.as_rotvec()).float(), hand_pose], dim=0).unsqueeze(0),
                #     th_trans=torch.as_tensor(hand_trans).float().unsqueeze(0))
                # hand_rot = hand_rot.as_rotvec()

            sample.update({
                'handRot': hand_rot,
                'handPose': hand_pose,
                'handTrans': hand_trans,
                'handVerts': handV,
                'handJoints': handJ,
                'handPartT': part_T,
                'handPartIds': hand_part_ids,
                'handNormals': np.stack(handN, axis=0),
                'canoJoints': canoJ,
            })
            
            # if self.grid_contact_ds is not None:
            #     with h5py.File(self.grid_contact_ds, 'r') as ds:
            #         ho_dist = ds['ho_dist'][idx]  # (n_grids, 778)
            #         contact_mask = ho_dist < self.msdf_scale
            #         grid_mask = np.any(contact_mask, axis=-1)
            #         local_grid_dist = ds['local_grid'][idx]

            #         nn_point = ds['nn_point'][idx][grid_mask].reshape(-1, self.msdf_kernel_size**3, 3)
            #         nn_face_idx = ds['nn_face_idx'][idx][grid_mask].reshape(-1, self.msdf_kernel_size**3)  # (M * K^3,)

            #         if self.augment:
            #             local_grid_dist = local_grid_dist.reshape(-1, self.msdf_kernel_size**3)[:, reorder_id].reshape(
            #                 -1, self.msdf_kernel_size, self.msdf_kernel_size, self.msdf_kernel_size, 1)
            #             nn_point = nn_point @ (Rot.T)[np.newaxis, :, :]
            #             nn_point = nn_point[:, reorder_id, :]
            #             nn_face_idx = nn_face_idx[:, reorder_id]

            #         sample['localGridContact'] = np.zeros_like(local_grid_dist)
            #         sample['localGridContact'][grid_mask] = self.grid_dist_to_contact(local_grid_dist[grid_mask])

            #     ## Apply inverse transform to hand vertices
            #     handV = (handV - objt[np.newaxis, :]) @ objR

            #     nn_vert_idx = hmodels[sbj_id].th_faces[nn_face_idx]  # (M, K^3, 3)
            #     face_verts = handV[nn_vert_idx]  # (M, K^3, 3, 3)
            #     face_cse_t = self.hand_cse[nn_vert_idx]  # (M, K^3, 3, cse_dim)

            #     # Calculate barycentric weights using matrix inversion (pure numpy operations)
            #     face_verts_transposed = np.swapaxes(face_verts, -1, -2)  # (M, K^3, 3, 3)
            #     w = np.linalg.inv(face_verts_transposed) @ nn_point[..., np.newaxis]  # (M, K^3, 3, 1)

            #     # Make sure weights are all positive and normalized
            #     w = np.clip(w, 0, 1)
            #     w = w / (np.sum(w, axis=2, keepdims=True)+1e-8)  # (M, K^3, 3, 1)

            #     grid_hand_cse = np.sum(face_cse_t * w, axis=2).reshape(-1, self.msdf_kernel_size, self.msdf_kernel_size, self.msdf_kernel_size, self.hand_cse.shape[1])  # (kernel_size, kernel_size, kernel_size, cse_dim)
            #     sample['localGridCSE'] = np.zeros((local_grid_dist.shape[0], self.msdf_kernel_size, self.msdf_kernel_size, self.msdf_kernel_size, self.hand_cse.shape[1]), dtype=np.float32)
            #     sample['localGridCSE'][grid_mask] = grid_hand_cse
            #     sample['nHoDist'] = 1 - 2 / (np.min(ho_dist / self.msdf_scale, axis=-1) + 1)
            #     sample['objPtMask'] = grid_mask
            #     sample['handVertMask'] = contact_mask

        else:
            obj_name = self.test_objects[idx]
            obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3
            ## For testing, return object with random rotations. The rng is set to make sure
            ## each idx generates fixed rotation.
            # obj_rot = R.identity()

            sample.update({
                'objName': obj_name,
                'objSamplePts': obj_sample_pts,
                'objSampleNormals': obj_sample_normals,
            })

        obj_com = self.obj_info[obj_name]['CoM']
        obj_inertia = self.obj_info[obj_name]['inertia']
        obj_mass = self.obj_info[obj_name]['mass']
        sample.update({
                'objCoM': obj_com,
                'objInertia': obj_inertia,
                'objMass': obj_mass
            })

        if self.msdf_path is not None:
            obj_msdf = self.obj_info[obj_name]['msdf'].copy() ## K^3 + 3
            obj_msdf[:, :-3] = obj_msdf[:, :-3] / self.msdf_scale / np.sqrt(3) # Normalize SDF
            obj_msdf[:, -3:] = obj_msdf[:, -3:] - obj_com
            sample['objMsdf'] = obj_msdf
            sample['objMsdfGrad'] = self.obj_info[obj_name]['msdf_grad'].copy()
            sample['adjPointIndices'] = self.obj_info[obj_name].get('adj_indices', None)
            sample['adjPointDistances'] = self.obj_info[obj_name].get('adj_distances', None)
            sample['nAdjPoints'] = self.obj_info[obj_name]['n_adj_points']
            if self.augment:
                obj_msdf_pts = obj_msdf[:, -3:] @ Rot.T
                obj_msdf[:, -3:] = obj_msdf_pts
                obj_msdf[:, :-3] = obj_msdf[:, :-3][:, reorder_id]
                sample['objMsdf'] = obj_msdf
                sample['objMsdfGrad'] = sample['objMsdfGrad'][:, reorder_id] @ Rot.T
                ## reorder adjacents
                adj_indices = sample['adjPointIndices']
                grid_id = adj_indices // (self.msdf_kernel_size ** 3)
                local_point_id = adj_indices % (self.msdf_kernel_size ** 3)
                new_local_point_id = reorder_id[local_point_id]
                new_adj_indices = grid_id * (self.msdf_kernel_size ** 3) + new_local_point_id
                plain_grid_ids = np.arange(obj_msdf.shape[0])
                glob_reorder = (plain_grid_ids[:, None] * self.msdf_kernel_size**3 + reorder_id[None, :]).flatten()
                sample['adjPointIndices'] = new_adj_indices
                sample['nAdjPoints'] = sample['nAdjPoints'][glob_reorder]

        return sample

    def get_obj_mesh(self, obj_name, obj_rot, obj_trans):
        obj_verts = self.obj_info[obj_name]['verts']
        obj_faces = self.obj_info[obj_name]['faces']
        obj_verts = transform_obj(obj_verts, obj_rot.cpu().numpy(), obj_trans.cpu().numpy())
        return obj_verts, obj_faces

    def get_obj_hulls(self, obj_name, obj_rot, obj_trans):
        obj_hull = deepcopy(self.obj_hulls[obj_name])
        for i, h in enumerate(obj_hull):
            obj_hull[i].vertices = transform_obj(h.vertices, obj_rot.cpu().numpy(), obj_trans.cpu().numpy())

        return obj_hull


def canonical_hand_parts(mano_layer, num_samples, batch_size, betas=None, device='cpu'):
    cano_verts, cano_joints, cano_part_T = mano_layer(th_pose_coeffs=torch.zeros(batch_size, 48).to(device),
                                                      th_betas=betas, th_trans=torch.zeros(batch_size, 3).to(device))
    cano_verts, cano_joints = cano_verts.detach().cpu().numpy(), cano_joints.detach().cpu().numpy()
    hand_part_ids = torch.argmax(mano_layer.th_weights, dim=-1).detach().cpu().numpy()
    faces = mano_layer.th_faces.detach().cpu().numpy()
    faces_in_part = []
    for part_idx in range(16):
        part_vert_ids = np.where(hand_part_ids == part_idx)[0]
        n_faces = np.sum(np.sum(part_vert_ids.reshape(1, -1) == faces.reshape(-1, 1),
                                      axis=1).reshape(-1, 3), axis=-1)
        faces_in_part.append(np.where(n_faces > 0))

    all_part_sample_pts = []
    all_part_sample_normals = []
    for b in range(batch_size):
        cano_mesh = trimesh.Trimesh(cano_verts[b], faces)
        part_meshes = []
        part_sample_pts = []
        part_sample_normals = []
        for part_idx in range(16):
            part_mesh = cano_mesh.submesh(faces_in_part[part_idx], append=True)
            part_meshes.append(part_mesh)
            if num_samples > 0:
                # part_samples, part_sampled_ids = trimesh.sample.sample_surface(part_mesh, count=num_samples)
                mesh = o3dmesh_from_trimesh(part_mesh)
                pc = mesh.sample_points_poisson_disk(num_samples, init_factor=8)
                part_samples = np.asarray(pc.points)
                part_sample_pts.append(part_samples - cano_joints[b, part_idx].reshape(1, 3))
                part_sample_normals.append(np.asarray(pc.normals))
                # part_sample_normals.append(part_mesh.face_normals[part_sampled_ids])
            else:
                ## Do not sample
                part_samples = part_mesh.vertices
                part_sample_pts.append(part_samples - cano_joints[b, part_idx].reshape(1, 3))
                part_sample_normals = part_mesh.vertex_normals

        if num_samples > 0:
            part_sample_pts = np.stack(part_sample_pts, axis=0)
            part_sample_normals = np.stack(part_sample_normals, axis=0)
        else:
            max_len = max([part_sample_pts[i].shape[0] for i in range(16)])
            part_sample_pts = np.stack([np.pad(p, pad_width=((0, max_len-p.shape[0]), (0, 0))) for p in part_sample_pts], axis=0)
            part_sample_normals = np.stack([np.pad(p, pad_width=((0, max_len-p.shape[0]), (0, 0))) for p in part_sample_normals], axis=0)

        all_part_sample_pts.append(part_sample_pts)
        all_part_sample_normals.append(part_sample_normals)

    all_part_sample_pts = np.stack(all_part_sample_pts, axis=0)
    all_part_sample_normals = np.stack(all_part_sample_normals, axis=0)

    return all_part_sample_pts, all_part_sample_normals, cano_joints
