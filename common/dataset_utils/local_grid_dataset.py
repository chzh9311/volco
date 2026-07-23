import os
import os.path as osp
import pickle
import importlib
from collections import defaultdict
import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R
from pysdf import SDF
from lightning import LightningDataModule
from tqdm import tqdm
import h5py

from common.utils.geometry import GridDistanceToContact
from common.msdf.utils.msdf import calc_local_grid_1pt


class LocalGridDataset(Dataset):
    """
    A dataset class for loading local grid data for hand-object interactions.
    """
    def __init__(self, cfg, split):
        super().__init__()
        self.grid_scale = cfg.msdf.scale
        self.kernel_size = cfg.msdf.kernel_size
        self.grid_dist_to_contact = GridDistanceToContact.from_config(cfg.msdf, method=cfg.msdf.contact_method)
        self.dataset_name = cfg.dataset_name
        self.dataset_path = cfg.dataset_path
        self.mano_root = cfg.get('mano_root', 'data/mano')
        self.preprocessed_dir = cfg.get('preprocessed_dir', 'data/preprocessed')
        self.downsample_rate = cfg.downsample_rate[split]
        self.split = split

        # Load local grid data from H5 file
        local_grid_file = osp.join(cfg.preprocessed_dir, self.dataset_name,
                                   f'local_grid_{self.kernel_size}_{self.grid_scale*1000:.1f}mm',
                                   f'{split}.h5')
        if not osp.exists(local_grid_file):
            raise FileNotFoundError(f"Local grid file not found: {local_grid_file}. Please run LocalGridDataModule.prepare_data() first.")

        self.local_grid_file = local_grid_file
        # Load small arrays into RAM; build per-grid-item scalar lookups to avoid
        # expensive random chunk reads into large H5 datasets at __getitem__ time.
        with h5py.File(local_grid_file, 'r') as f:
            self.n_grids = f['grid_distance'].shape[0]
            # grid_mask: (n_frames, n_pts) bool — 66 MB, load fully
            self._grid_mask = f['grid_mask'][:]  # (n_frames, n_pts)
            # grid_sample_idx: (n_grids, 2) int — 64 MB, load fully
            grid_sample_idx = f['grid_sample_idx'][:]  # (n_grids, 2): [point_idx, sample_idx]
            # verts_mask: (n_frames, n_pts, 778) bool — 51 GB on disk, do NOT load fully.
            # Precompute the (778,) bool row per grid item needed at __getitem__ time.
            # Read verts_mask in chunks aligned to H5 chunk boundaries to minimise I/O.
            verts_mask_ds = f['verts_mask']
            n_frames = verts_mask_ds.shape[0]
            n_verts = verts_mask_ds.shape[2]
            chunk_rows = verts_mask_ds.chunks[0] if verts_mask_ds.chunks else 1  # H5 chunk size along frame axis
            self._hand_vert_mask = np.empty((self.n_grids, n_verts), dtype=bool)
            # Build a mapping: sample_idx -> list of (gi, point_idx) pairs
            sample_to_grids = defaultdict(list)
            for gi, (point_idx, sample_idx) in enumerate(grid_sample_idx):
                sample_to_grids[sample_idx].append((gi, point_idx))
            # Iterate frames in sorted order, reading chunk_rows at a time
            print(f"Precomputing handVertMask: reading {n_frames} frames "
                  f"in chunks of {chunk_rows} from verts_mask...")
            for chunk_start in tqdm(range(0, n_frames, chunk_rows), desc='Loading verts_mask'):
                chunk_end = min(chunk_start + chunk_rows, n_frames)
                chunk = verts_mask_ds[chunk_start:chunk_end]  # (chunk_rows, n_pts, 778)
                for frame_idx in range(chunk_start, chunk_end):
                    if frame_idx not in sample_to_grids:
                        continue
                    local_idx = frame_idx - chunk_start
                    frame_mask = self._grid_mask[frame_idx]            # (n_pts,) bool
                    masked_rows = chunk[local_idx][frame_mask]          # (n_masked, 778)
                    for gi, point_idx in sample_to_grids[frame_idx]:
                        self._hand_vert_mask[gi] = masked_rows[point_idx]
            self._grid_sample_idx = grid_sample_idx
        # Global validity mask: grid items where at least one hand vertex lies inside the grid.
        # Used to restrict training to meaningful contact examples.
        if cfg.use_noncontact_grids:
            self._valid_indices = np.arange(self.n_grids)
        else:
            self._valid_indices = np.where(self._hand_vert_mask.any(axis=-1))[0]
        print(f"Valid grid items (hand inside grid): {len(self._valid_indices)} / {self.n_grids} "
              f"({100 * len(self._valid_indices) / self.n_grids:.1f}%)")
        # Per-worker H5 handle, initialized lazily in __getitem__ to avoid
        # multiprocessing fork issues and repeated open/close overhead.
        self._h5_handle = None

        hand_cse_sd = torch.load(cfg.get('hand_cse_path', 'data/misc/hand_cse.ckpt'), weights_only=True)['state_dict']
        self.hand_cse = hand_cse_sd['embedding_tensor'].detach().cpu().numpy()
        self.mano_faces = hand_cse_sd['cano_faces'].detach().cpu().numpy()
        self.obj_info = {}
        self.close_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'), allow_pickle=True)
        ## Downsample data
        # self.idx2sample = self.idx2sample[::cfg.downsample_rate[split], :]
        coords = torch.tensor([(i, j, k) for i in range(self.kernel_size)
                                    for j in range(self.kernel_size)
                                    for k in range(self.kernel_size)], dtype=torch.float32) 
        normalized_coords = 2 * coords - (self.kernel_size - 1)
        self.normalized_coords = (normalized_coords / (self.kernel_size - 1))

        self._load_data()
        if self.downsample_rate > 1:
            self.rh_data = self.sample_frames(self.rh_data)
            self.object_data = self.sample_frames(self.object_data)
            self.frame_names = self.sample_frames(self.frame_names)

    def sample_frames(self, data):
        if type(data) is dict:
            for k, v in data.items():
                data[k] = v[::self.downsample_rate]
        else:
            data = data[::self.downsample_rate]

        return data

    def _load_data(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        # Remap through valid indices so only grid items with hand inside the grid are exposed.
        raw_idx = self._valid_indices[idx]
        # Lazily open the H5 file once per worker process, keeping it open for all subsequent reads.
        if self._h5_handle is None:
            self._h5_handle = h5py.File(self.local_grid_file, 'r')
        grid_data_raw = self._h5_handle

        # sample_idx, point_idx = self.idx2sample[idx].tolist()
        point_idx, sample_idx = self._grid_sample_idx[raw_idx]
        # frame_name = grid_data_raw['frame_names'][sample_idx].decode('utf-8')

        hand_vert_mask = self._hand_vert_mask[raw_idx]   # (778,) bool, preloaded
        grid_mask = self._grid_mask[sample_idx]           # (n_pts,) bool, preloaded
        # Convert unmasked point_idx to masked array index
        # masked_point_idx = np.sum(grid_mask[:point_idx]).item()
        # masked_point_idx = np.searchsorted(np.where(grid_mask)[0], point_idx)
        grid_sdf = grid_data_raw['grid_sdf'][raw_idx]
        grid_distance = grid_data_raw['grid_distance'][raw_idx]
        ## TODO: do shape transformation for nn_face_idx and nn_point when saving data.
        nn_face_idx = grid_data_raw['nn_face_idx'][raw_idx].reshape(self.kernel_size**3)
        nn_point = grid_data_raw['nn_point'][raw_idx].reshape(self.kernel_size**3, 3)

        fname_path = self.frame_names[sample_idx].split('/')
        sbj_id = fname_path[2]
        obj_name = fname_path[3].split('_')[0]
        obj_sample_pt = self.obj_info[obj_name]['samples'][grid_mask][point_idx] # 3
        # obj_sample_normal = self.obj_info[obj_name]['sample_normals'][point_idx] # 3
        obj_rot = self.object_data['global_orient'][sample_idx]
        obj_trans = self.object_data['transl'][sample_idx]
        
        objR = axis_angle_to_matrix(obj_rot).detach().cpu().numpy()
        if self.dataset_name == 'grab':
            objR = objR.T
        objt = obj_trans.detach().cpu().numpy()
        obj_sample_pt = (objR @ obj_sample_pt.reshape(3, 1)).reshape(3) + objt
        # obj_sample_normal = (objR @ obj_sample_normal.reshape(3, 1)).reshape(3)

        hdata = self.rh_data
        with torch.no_grad():
            handV, _, _ = self.rh_models[sbj_id](
                th_pose_coeffs=torch.cat([hdata['global_orient'][sample_idx], hdata['fullpose'][sample_idx].view(45)], dim=0)[None, :],
                th_trans=hdata['transl'][sample_idx][None, :])
        nhandV = handV[0].detach().cpu().numpy() - obj_sample_pt[np.newaxis, :]
        # nhandV = np.random.rand(778, 3)

        nn_point = nn_point - obj_sample_pt[np.newaxis, :]

        nn_vert_idx = self.mano_faces[nn_face_idx] # N,  3
        face_verts = nhandV[nn_vert_idx]  # (N, 3, 3)
        face_cse = self.hand_cse[nn_vert_idx]  # (M * K^3, 3, cse_dim)
        w = np.linalg.solve(face_verts.transpose((0, 2, 1)), nn_point[:, :, np.newaxis])  # (M * K^3, 3, 1)
        # Make sure weights are all positive and normalized
        w = np.clip(w, 0, 1)
        w = w / (np.sum(w, axis=1, keepdims=True) + 1e-8)  # (M * K^3, 3, 1)

        grid_hand_cse = np.sum(face_cse * w, axis=1).reshape(self.kernel_size, self.kernel_size, self.kernel_size, -1)  # (N, cse_dim)

        # hand_mesh = trimesh.Trimesh(vertices=nhandV, faces=self.close_mano_faces, process=False)
        # grid_data, verts_mask = calc_local_grid_1pt(self.normalized_coords.numpy(), obj_mesh, hand_mesh,
        #                                         self.kernel_size, self.grid_scale, self.hand_cse)
        ## contact data
        grid_sdf = torch.from_numpy(grid_sdf).float()
        grid_distance = torch.from_numpy(grid_distance).float()
        grid_contact = self.grid_dist_to_contact(grid_distance)
        grid_sdf = grid_sdf / self.grid_scale / np.sqrt(3)  # Normalize SDF

        if not hand_vert_mask.any():
            ## Mask out all grid data if no hand vertices are inside the grid, to allow latent space to cover zero cases.
            grid_contact = torch.zeros_like(grid_contact)
            if np.random.rand() < 0.5:
                ## Randomly mask out the cse as well, to expand the zero-contact region in latent space and improve generalization.
                grid_hand_cse = np.zeros_like(grid_hand_cse)

        sample = {
                  'gridSDF': grid_sdf,
                  'gridContact': grid_contact,
                  'gridHandCSE': grid_hand_cse,
                  'objSamplePt': obj_sample_pt,
                #   'objSampleNormal': obj_sample_normal,
                  'objRot': obj_rot,
                  'objTrans': obj_trans,
                  'nHandVerts': nhandV,
                  'handVertMask': hand_vert_mask,
                  'obj_name': obj_name,
                  'face_idx': nn_face_idx,
                  'cse_weights': w.squeeze(-1),
                }

        return sample

