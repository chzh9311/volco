## Largely from https://github.com/shreyashampali/ho3d.git
## Implement LightningDataModule API for HO3Dv3
import sys
for p in ['.', '..']:
    sys.path.append(p)
import os
import os.path as osp
import numpy as np
import pickle
import trimesh
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from lightning import LightningDataModule
from copy import copy
import math
import cv2
from .hoi_dataset import BaseHOIDataset
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
# from manopth.manolayer import ManoLayer

jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]
obj_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
            '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']

class HO3DDataset(BaseHOIDataset):
    def __init__(self, cfg, split: str, load_msdf: bool = False, load_grid_contact: bool = False, object_only: bool = True):
        self.data_dir = cfg.dataset_path
        self.obj_dir = cfg.obj_model_path
        self.n_samples = cfg.object_sample
        self.split = split
        self.num_samples = cfg.test_samples
        self.dataset_name = 'ho3d'
        super().__init__(cfg, split, load_msdf=load_msdf, load_grid_contact=load_grid_contact, object_only=object_only)

    @staticmethod
    def load_mesh_info(data_dir, n_samples=1024, msdf_path=None):
        """
        Load mesh information from the dataset.
        
        :param data_dir: the directory where the dataset is stored
        :param split: train/val/test split
        """
        obj_info = {}
        for k in obj_names:
            mesh = trimesh.load(osp.join(data_dir, 'models', k, 'textured_simple.obj'), process=False)
            obj_info[k] = {
                'verts': np.copy(mesh.vertices),
                'faces': np.copy(mesh.faces)
            }
            # mesh = trimesh.Trimesh(v['verts'], v['faces'], process=False)
            sample_pts, fid = trimesh.sample.sample_surface(mesh, count=n_samples)
            obj_info[k]['samples'] = sample_pts
            obj_info[k]['sample_normals'] = mesh.face_normals[fid]
            obj_info[k]['CoM'] = mesh.center_mass

            ## Assume unit density 1kg/cm^3
            obj_info[k]['mass'] = mesh.mass
            obj_info[k]['inertia'] = mesh.moment_inertia

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
        if self.split != 'test':
            self.annots = self._load_annot(self.data_dir, self.obj_dir, self.split)

        self.obj_info = self.load_mesh_info(self.obj_dir, n_samples=self.n_samples, msdf_path=self.msdf_path)
        self.simp_obj_mesh = self.obj_info
        self.test_objects = obj_names

        self.obj_hulls = {}
        for obj_name in obj_names:
            hulls = []
            hull_path = osp.join(self.obj_dir, 'obj_hulls', obj_name)
            for i in range(len(os.listdir(hull_path))):
                hulls.append(trimesh.load(osp.join(hull_path, f'hull_{i}.stl')))
            self.obj_hulls[obj_name] = hulls

        self.rh_data = None
        self.object_data = None
        self.frame_names = None
    

    def _load_annot(self, data_dir, obj_dir, split):
        annots = []
        seq_cams = os.listdir(os.path.join(data_dir, split))
        calib_dir = os.path.join(data_dir, 'calibration')

        for seq_cam in seq_cams:
            for img in os.listdir(os.path.join(data_dir, split, seq_cam, 'rgb')):
                ## Load only seqs with camera extrinsics
                seqs = os.listdir(os.path.join(data_dir, 'calibration'))
                if seq_cam[:-1] in seqs:
                    annot = {'seq': seq_cam[:-1], 'cam_id': seq_cam[-1], 'id': img.split('.')[0],
                             'imgPath': os.path.join(data_dir, split, seq_cam, 'rgb', img)}
                    T = np.loadtxt(os.path.join(data_dir, 'calibration', seq_cam[:-1], 'calibration', f'trans_{seq_cam[-1]}.txt'))
                    annot['camEx'] = T
                    meta_file = os.path.join(data_dir, split, seq_cam, 'meta', annot['id'] + '.pkl')
                    assert os.path.exists(meta_file)
                    with open(meta_file, 'rb') as f:
                        meta = pickle.load(f)
                        try:
                            for k in ['handPose', 'handBeta', 'handTrans', 'objRot', 'objTrans', ]:
                                annot[k] = meta[k].flatten()
                            for k in ['objCorners3DRest', 'camMat', 'objName']:
                                annot[k] = meta[k]
                            annots.append(annot)
                        except AttributeError as e:
                            # print(f'Error loading annotation file {meta_file} due to {e}, ignored.')
                            continue
                # pose = meta['handPose']
                # glob_rot, theta = pose[:3], pose[3:]
                # annot['handRot'] = glob_rot
                # annot['handTheta'] = theta

        return annots


def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                           thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):

            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn)

    return imgIn

def project_pts(pts: np.ndarray, cam_mat: np.ndarray):
    """
    Project global 3D coordinate to 2D plane.
    Used only for HO3D
    """
    pts = copy(pts)
    pts[:, 1:] *= -1
    homo_jt2d = pts @ cam_mat.T
    pts2d = np.stack((homo_jt2d[:, 0] / homo_jt2d[:, 2], homo_jt2d[:, 1] / homo_jt2d[:, 2]), axis=1)
    return pts2d


def global2local(x: np.ndarray, T: np.ndarray):
    """
    param x: the points to transform (N x 3)
    param T: the 4 x 4 transformation matrix of local coordinate system from global.
    """
    xh_global = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    T_inv = np.zeros_like(T)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = - T[:3, 3]
    T_inv[3, 3] = 1
    xh_local = (T_inv @ xh_global.T).T
    x_local = xh_local[:, :3] / xh_local[:, 3:4]
    return x_local

def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': [], 'fn': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])
            if len(spl[0]) > 2 and spl[2] and 'fn' in d:
                d['fn'].append([np.array([int(l[2])-1 for l in spl[:3]])])

        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])

    for k, v in d.items():
        if k in ['v','vn','f','vt','ft', 'fn']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
