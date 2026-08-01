import pickle
import sys
from tqdm import tqdm

import trimesh
from trimesh import sample

for p in ['.', '..']:
    sys.path.append(p)
import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
# from mesh_to_sdf import mesh_to_sdf
import open3d as o3d
from multiprocessing.pool import Pool
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from scipy.spatial.transform import Rotation

from common.manopth.manopth.manolayer import ManoLayer
from common.utils.geometry import make_watertight
from .hoi_dataset import BaseHOIDataset
from .local_grid_dataset import LocalGridDataset

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]


test_objects = ['wineglass', 'fryingpan', 'mug', 'toothpaste', 'camera', 'binoculars']
val_objects = ['hand', 'elephant', 'apple']
train_objects_select = ['airplane', 'alarmclock', 'apple', 'cubemedium', 'mouse', 'watch']

contact_ids={'Body': 1,
             'L_Thigh': 2,
             'R_Thigh': 3,
             'Spine': 4,
             'L_Calf': 5,
             'R_Calf': 6,
             'Spine1': 7,
             'L_Foot': 8,
             'R_Foot': 9,
             'Spine2': 10,
             'L_Toes': 11,
             'R_Toes': 12,
             'Neck': 13,
             'L_Shoulder': 14,
             'R_Shoulder': 15,
             'Head': 16,
             'L_UpperArm': 17,
             'R_UpperArm': 18,
             'L_ForeArm': 19,
             'R_ForeArm': 20,
             'L_Hand': 21,
             'R_Hand': 22,
             'Jaw': 23,
             'L_Eye': 24,
             'R_Eye': 25,
             'L_Index1': 26,
             'L_Index2': 27,
             'L_Index3': 28,
             'L_Middle1': 29,
             'L_Middle2': 30,
             'L_Middle3': 31,
             'L_Pinky1': 32,
             'L_Pinky2': 33,
             'L_Pinky3': 34,
             'L_Ring1': 35,
             'L_Ring2': 36,
             'L_Ring3': 37,
             'L_Thumb1': 38,
             'L_Thumb2': 39,
             'L_Thumb3': 40,
             'R_Index1': 41,
             'R_Index2': 42,
             'R_Index3': 43,
             'R_Middle1': 44,
             'R_Middle2': 45,
             'R_Middle3': 46,
             'R_Pinky1': 47,
             'R_Pinky2': 48,
             'R_Pinky3': 49,
             'R_Ring1': 50,
             'R_Ring2': 51,
             'R_Ring3': 52,
             'R_Thumb1': 53,
             'R_Thumb2': 54,
             'R_Thumb3': 55}

key2part_id = ['Hand', 'Index1', 'Index2', 'Index3', 'Middle1', 'Middle2', 'Middle3', 'Pinky1', 'Pinky2', 'Pinky3',
               'Ring1', 'Ring2', 'Ring3', 'Thumb1', 'Thumb2', 'Thumb3']

def regularize_part_id(contacts: torch.Tensor, hand_side: str):
    c2p = []
    for pstr in key2part_id:
        if hand_side == 'left':
            c2p.append(contact_ids['L_' + pstr])
        else:
            c2p.append(contact_ids['R_' + pstr])
    out_contacts = torch.zeros_like(contacts, dtype=torch.int64)
    for idx, cid in enumerate(c2p):
        out_contacts[contacts==cid] = idx + 1

    out_contacts = F.one_hot(out_contacts, num_classes=16 + 1)
    out_contacts = out_contacts[..., 1:] # no contact are labelled as all 0 vectors.

    return out_contacts


class GRABDataset(BaseHOIDataset):
    def __init__(self, cfg, split: str, load_msdf: bool = False, load_grid_contact: bool = False, object_only: bool = False):
        # self.part_samples = cfg.hand_part_samples
        self.obj_dir = cfg.dataset_path
        self.dataset_name = 'grab'
        super().__init__(cfg, split, load_msdf=load_msdf, load_grid_contact=load_grid_contact, object_only=object_only)
    
    @staticmethod
    def load_mesh_info(data_dir, msdf_path=None):
        """
        Load mesh information from the dataset.
        
        :param data_dir: the directory where the dataset is stored
        :param split: train/val/test split
        """
        obj_info = np.load(osp.join(data_dir, 'obj_info.npy'), allow_pickle=True).item()
        for k, v in obj_info.items():
            mesh = trimesh.Trimesh(v['verts'], v['faces'], process=False)

            if k == 'camera':
                # Shipped with 10 one-triangle holes, which makes the exact
                # boolean intersection metric fall back to voxelization.
                # The fix only appends the missing faces, so vertex order --
                # and therefore verts_sample_id below -- is preserved.
                mesh = make_watertight(mesh)
                v['verts'], v['faces'] = mesh.vertices, mesh.faces

            # sample_pts, fid = sample.sample_surface(mesh, self.n_obj_samples)
            v['samples'] = v.pop('verts_sample')
            v['sample_normals'] = mesh.vertex_normals[v['verts_sample_id']]
            v['CoM'] = mesh.center_mass

            ## Assume unit density 1kg/cm^3
            v['mass'] = mesh.mass
            v['inertia'] = mesh.moment_inertia

        if msdf_path is not None:
            adj_point_path = msdf_path.replace('msdf', 'msdf_adj_points')
            for k, v in obj_info.items():
                if os.path.exists(osp.join(msdf_path, f'{k}.npz')):
                    msdf_data = np.load(osp.join(msdf_path, f'{k}.npz'))
                    v['msdf'] = msdf_data['msdf']
                    if 'msdf_grad' in msdf_data:
                        v['msdf_grad'] = msdf_data['msdf_grad']
                if os.path.exists(osp.join(adj_point_path, f'{k}.npz')):
                    adj_data = np.load(osp.join(adj_point_path, f'{k}.npz'))
                    v['adj_indices'] = adj_data['indices']
                    v['adj_distances'] = adj_data['distances']
                    v['n_adj_points'] = adj_data['n_adj_points']
        return obj_info

    def _load_data(self):
        self.obj_info = self.load_mesh_info(self.data_dir, msdf_path=self.msdf_path)

        self.ds = np.load(osp.join(self.data_dir, self.split, f'grabnet_{self.split}.npz'), allow_pickle=True)
        frame_names = np.load(osp.join(self.data_dir, self.split, 'frame_names.npz'))['frame_names']
        rh_root_rotmat = torch.as_tensor(self.ds['global_orient_rhand_rotmat'], dtype=torch.float32)
        rh_root_rot = matrix_to_axis_angle(rh_root_rotmat.squeeze(1))
        rh_pose_rotmat = torch.as_tensor(self.ds['fpose_rhand_rotmat'], dtype=torch.float32)
        rh_pose = matrix_to_axis_angle(rh_pose_rotmat.squeeze(1)).view(-1, 45)
        self.rh_data = {
            "global_orient": rh_root_rot,
            "fullpose": rh_pose,
            "transl": torch.as_tensor(self.ds['trans_rhand'], dtype=torch.float32)
        }
        obj_rotmat = torch.as_tensor(self.ds['root_orient_obj_rotmat'], dtype=torch.float32)
        obj_orient = matrix_to_axis_angle(obj_rotmat.squeeze(1))
        self.object_data = {
            "global_orient": obj_orient,
            "transl": torch.as_tensor(self.ds['trans_obj'], dtype=torch.float32)
        }
        self.frame_names = np.array(frame_names, dtype=np.dtypes.StringDType())
        if self.split == 'test':
            self.test_objects = test_objects
        elif self.split == 'val':
            self.test_objects = val_objects
        else:
            self.test_objects = train_objects_select
        # self.test_objects = train_objects_select ## to see if overfitting happens

        with open(osp.join('data', 'preprocessed', 'grab', 'simplified_obj_mesh.pkl'), 'rb') as of:
            self.simp_obj_mesh = pickle.load(of)

        ## Object hulls for simulation.
        self.obj_hulls = {}
        # self.obj_mass = {}
        self.obj_model = {}
        for obj_name in self.obj_info.keys():
            hulls = []
            hull_path = osp.join(self.data_dir, 'obj_hulls', obj_name)
            for i in range(len(os.listdir(hull_path))):
                hulls.append(trimesh.load(osp.join(hull_path, f'hull_{i}.stl')))

            self.obj_hulls[obj_name] = hulls
            obj_model = trimesh.Trimesh(self.obj_info[obj_name]['verts'], self.obj_info[obj_name]['faces'])
            self.obj_model[obj_name] = obj_model

            ## downsample the object sampled points.
            # self.obj_mass[obj_name] = obj_model.volume * 1000 # sum([h.volume for h in hulls]) * 1000 # 1000 kg / m^3

        sbj_info = np.load(osp.join(self.data_dir, 'sbj_info.npy'), allow_pickle=True).item()
        self.rh_models = {}
        self.lh_models = {}
        self.cano_joints_cache = {}  # pre-computed canonical joints per subject
        for sbj_id in sbj_info.keys():
            self.rh_models[sbj_id] = ManoLayer(mano_root=self.mano_root,
                                               side='right',
                                               th_v_template=sbj_info[sbj_id]['rh_vtemp'],
                                               flat_hand_mean=True,
                                               use_pca=False,
                                               ncomps=45)
            with torch.no_grad():
                _, canoJ, _ = self.rh_models[sbj_id](th_pose_coeffs=torch.zeros(1, 48))
            self.cano_joints_cache[sbj_id] = canoJ[0].detach().cpu().numpy()

class GRABLocalGridDataset(LocalGridDataset):
    """
    A dataset class for loading local grid data for hand-object interactions.
    """
    def __init__(self, cfg, split):
        self.dataset_path = cfg.dataset_path
        super().__init__(cfg, split)

    def _load_data(self):
        self.obj_info = GRABDataset.load_mesh_info(self.dataset_path, msdf_path=None)
        with open(osp.join('data', 'preprocessed', 'grab', 'simplified_obj_mesh.pkl'), 'rb') as of:
            self.simp_obj_mesh = pickle.load(of)
        
        self.ds = np.load(osp.join(self.dataset_path, self.split, f'grabnet_{self.split}.npz'), allow_pickle=True)
        rh_root_rotmat = torch.as_tensor(self.ds['global_orient_rhand_rotmat'], dtype=torch.float32)
        rh_root_rot = matrix_to_axis_angle(rh_root_rotmat.squeeze(1))
        rh_pose_rotmat = torch.as_tensor(self.ds['fpose_rhand_rotmat'], dtype=torch.float32)
        rh_pose = matrix_to_axis_angle(rh_pose_rotmat.squeeze(1)).view(-1, 45)
        self.rh_data = {
            "global_orient": rh_root_rot,
            "fullpose": rh_pose,
            "transl": torch.as_tensor(self.ds['trans_rhand'], dtype=torch.float32)
        }
        obj_rotmat = torch.as_tensor(self.ds['root_orient_obj_rotmat'], dtype=torch.float32)
        obj_orient = matrix_to_axis_angle(obj_rotmat.squeeze(1))
        self.object_data = {
            "global_orient": obj_orient,
            "transl": torch.as_tensor(self.ds['trans_obj'], dtype=torch.float32)
        }

        sbj_info = np.load(osp.join(self.dataset_path, 'sbj_info.npy'), allow_pickle=True).item()
        self.rh_models = {}
        self.cano_joints_cache = {}  # pre-computed canonical joints per subject
        frame_names = np.load(osp.join(self.dataset_path, self.split, 'frame_names.npz'))['frame_names']
        self.frame_names = np.array(frame_names, dtype=np.dtypes.StringDType())
        for sbj_id in sbj_info.keys():
            self.rh_models[sbj_id] = ManoLayer(mano_root=self.mano_root,
                                               side='right',
                                               th_v_template=sbj_info[sbj_id]['rh_vtemp'],
                                               flat_hand_mean=True,
                                               use_pca=False,
                                               ncomps=45)
            with torch.no_grad():
                _, canoJ, _ = self.rh_models[sbj_id](th_pose_coeffs=torch.zeros(1, 48))
            self.cano_joints_cache[sbj_id] = canoJ[0].detach().cpu().numpy()