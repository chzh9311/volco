"""
GT validator: compute penetration/contact metrics on ground-truth grasps from the test split.
Saves per-sample results to tmp/pene_analysis/<dataset_name>_gt.pkl as:
  {obj_name: [{'penetration_depth', 'contact_area', 'intersection_volume',
                'n_raw_pts', 'sum_raw_depth'}, ...]}
"""
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import os.path as osp
import pickle
import numpy as np
import trimesh
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from multiprocessing.pool import Pool
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix

from common.dataset_utils.datamodules import HOIDatasetModule
from common.evaluation.eval_fns import calculate_metrics

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("power", lambda x, y: x ** y, replace=True)


def _to_canonical(hand_verts, hand_joints, obj_rot_aa, obj_trans, obj_com, dataset_name):
    """Apply inverse object transform to bring hand into canonical (object) frame.
    Matches the logic in HandObject.load_from_batch (normalize=True, no aug_rot).
    hand_verts/hand_joints: (B, V/J, 3) float tensors
    obj_rot_aa: (B, 3) axis-angle
    obj_trans: (B, 3)
    obj_com: (B, 3)
    """
    batch_size = hand_verts.shape[0]
    objT = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    objR = axis_angle_to_matrix(obj_rot_aa)  # (B, 3, 3)
    if dataset_name == 'grab':
        objR = objR.transpose(-1, -2)  # grab stores transpose
    objT[:, :3, :3] = objR
    objT[:, :3, 3]  = obj_trans

    objT_inv = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    objT_inv[:, :3, :3] = objR.transpose(-1, -2)  # R^T
    objT_inv[:, :3, 3:4] = (
        - objT_inv[:, :3, :3] @ objT[:, :3, 3:4]  # -R^T t
        - obj_com.unsqueeze(-1)                     # subtract CoM (same as load_from_batch)
    )

    def transform_pts(pts):
        homo = F.pad(pts, (0, 1), 'constant', 1)   # (B, N, 4)
        return (objT_inv.unsqueeze(1) @ homo.unsqueeze(-1))[:, :, :3, 0]

    return transform_pts(hand_verts), transform_pts(hand_joints)


@hydra.main(version_base=None, config_path="../config", config_name="mlcdiff")
def main(cfg):
    if cfg.get('vis', False):  # python scripts/gt_validator.py +vis=true
        visualize_gt_contact(cfg)
        return

    closed_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'))
    dataset_name = cfg.data.dataset_name

    # Use the full (non-object_only) test split so we get per-frame GT hand data
    data_module = HOIDatasetModule(cfg)
    dataset_cls = data_module.dataset_class
    test_set   = dataset_cls(cfg.data, 'test',
                             load_msdf=False, load_grid_contact=False,
                             object_only=False)

    obj_hulls_all = test_set.obj_hulls   # dict: obj_name -> list[trimesh]
    obj_info_all  = test_set.obj_info    # dict: obj_name -> {verts, faces, ...}

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=HOIDatasetModule.collate_fn,
    )

    metrics_to_compute = [
        "Penetration Depth",
        "Contact Area",
        "Intersection Volume",
    ]

    n_processes = 16
    pool = Pool(n_processes) if n_processes > 1 else None

    # {obj_name: [{'penetration_depth', 'contact_area', 'intersection_volume',
    #               'n_raw_pts', 'sum_raw_depth'}, ...]}
    all_results = {}

    for batch_idx, batch in enumerate(tqdm(test_loader, desc='GT validation', total=len(test_loader))):
        obj_names = batch['objName']
        hand_verts_w  = batch['handVerts'].float()   # (B, V, 3) world frame
        hand_joints_w = batch['handJoints'].float()  # (B, J, 3) world frame
        obj_rot_aa    = batch['objRot'].float()       # (B, 3)
        obj_trans     = batch['objTrans'].float()     # (B, 3)
        obj_com       = batch['objCoM'].float()       # (B, 3)

        hand_verts, hand_joints = _to_canonical(
            hand_verts_w, hand_joints_w, obj_rot_aa, obj_trans, obj_com, dataset_name)

        hand_verts_np  = hand_verts.numpy()   # (B, V, 3) canonical
        hand_joints_np = hand_joints.numpy()  # (B, J, 3) canonical
        batch_size = hand_verts_np.shape[0]

        for i in tqdm(range(batch_size), desc=f'  batch {batch_idx}', leave=False):
            obj_name  = obj_names[i]
            obj_mesh_d = obj_info_all[obj_name]
            obj_mesh   = trimesh.Trimesh(obj_mesh_d['verts'], obj_mesh_d['faces'])
            obj_hulls  = obj_hulls_all[obj_name]

            param = [{
                'dataset_name':          dataset_name,
                'frame_name':            f"{obj_name}_{batch_idx}_{i}",
                'hand_model':            trimesh.Trimesh(hand_verts_np[i], closed_mano_faces),
                'obj_name':              obj_name,
                'hand_joints':           hand_joints_np[i],
                'obj_model':             obj_mesh,
                'obj_hulls':             obj_hulls,
                'idx':                   i,
            }]

            result = calculate_metrics(param, metrics=metrics_to_compute, pool=pool, reduction='none')

            raw_list   = result.pop("Penetration Depth Raw", [[]])
            _          = result.pop("Penetration Depth Vert IDs", [[]])
            raw        = np.asarray(raw_list[0]) if len(raw_list) > 0 else np.array([])
            pene_depth = float(result.get("Penetration Depth",   [0.0])[0])
            contact_a  = float(result.get("Contact Area",        [0.0])[0])
            int_vol    = float(result.get("Intersection Volume",  [0.0])[0])

            if obj_name not in all_results:
                all_results[obj_name] = []
            all_results[obj_name].append({
                'penetration_depth': pene_depth,
                'contact_area':      contact_a,
                'intersection_volume': int_vol,
                'n_raw_pts':         int(len(raw)),
                'sum_raw_depth':     float(raw.sum()) if len(raw) > 0 else 0.0,
            })

        n_total = sum(len(v) for v in all_results.values())
        if batch_idx % 50 == 0:
            print(f"[batch {batch_idx}] processed {n_total} samples so far", flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    os.makedirs(osp.join('tmp', 'pene_analysis'), exist_ok=True)
    save_path = osp.join('tmp', 'pene_analysis', f'{dataset_name}_gt.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved GT penetration analysis ({sum(len(v) for v in all_results.values())} samples) to {save_path}")


## Load GT data and visualize contact
def visualize_gt_contact(cfg, n_vis=8, side=800, out_dir=osp.join('tmp', 'gt_contact_vis'), seed=0):
    """
    Load GT grasps from the test split, build a HandObject with point-based contact,
    and save one image per sample with 3 columns: object | object + hand | object contact.
    Each column stacks the 4 camera views of geom_to_img_o3d vertically.
    Contact on the object mesh: each vertex takes the contact value of its nearest
    sampled point (HandObject.obj_verts), colored with the YlOrRd colormap.
    """
    from scipy.spatial import cKDTree
    from matplotlib import pyplot as plt
    from common.model.handobject import HandObject
    from common.utils.vis import o3dmesh_from_trimesh, parse_hex_color, geom_to_img_o3d
    import open3d as o3d

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Force point-based contact regardless of the top-level config
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    data_cfg.contact_unit = 'point'
    data_cfg.correspondence_type = 'cse'

    dataset_cls = HOIDatasetModule(cfg).dataset_class
    test_set = dataset_cls(data_cfg, 'test', load_msdf=False, load_grid_contact=False, object_only=False)
    obj_meshes_by_name = {n: trimesh.Trimesh(v['verts'], v['faces'], process=False)
                          for n, v in test_set.obj_info.items()}

    loader = DataLoader(test_set, batch_size=n_vis, shuffle=True, num_workers=4,
                        collate_fn=HOIDatasetModule.collate_fn,
                        generator=torch.Generator().manual_seed(seed))
    batch = next(iter(loader))
    obj_names = batch['objName']

    ho = HandObject(data_cfg, device=device, normalize=True)
    ho.load_from_batch(batch, obj_templates=[obj_meshes_by_name[n] for n in obj_names])

    heat_cmap = plt.colormaps['YlOrRd']
    os.makedirs(out_dir, exist_ok=True)
    for i in range(ho.batch_size):
        obj_mesh = ho.obj_models[i]  # CoM-centered, same frame as ho.obj_verts / ho.hand_verts
        hand_mesh = trimesh.Trimesh(ho.hand_verts[i].detach().cpu().numpy(), ho.closed_hand_faces.copy())

        # Per-vertex contact from the nearest sampled point
        sample_pts = ho.obj_verts[i].detach().cpu().numpy()             # (N, 3)
        sample_contact = ho.contact_map[i, :, 0].detach().cpu().numpy()  # (N,)
        _, nn_idx = cKDTree(sample_pts).query(np.asarray(obj_mesh.vertices))
        vert_contact = np.clip(sample_contact[nn_idx], 0, 1)

        obj_plain = o3dmesh_from_trimesh(obj_mesh, parse_hex_color("#6293A8"))
        obj_contact = o3dmesh_from_trimesh(obj_mesh)
        obj_contact.vertex_colors = o3d.utility.Vector3dVector(heat_cmap(vert_contact)[:, :3])
        hand = o3dmesh_from_trimesh(hand_mesh, parse_hex_color("#F29A8D"))

        panels = [[obj_plain], [obj_plain, hand], [obj_contact]]
        imgs = [geom_to_img_o3d(p, w=side, h=side, scale=0.9, concat_axis=0) for p in panels]
        row = np.clip(np.concatenate(imgs, axis=1), 0, 1)

        save_path = osp.join(out_dir, f'{i:03d}_{obj_names[i]}.png')
        plt.imsave(save_path, row)
        print(f"[{i}] {obj_names[i]}: contact max={sample_contact.max():.3f}, "
              f"#pts>0.5={(sample_contact > 0.5).sum()} -> {save_path}")


if __name__ == '__main__':
    main()
