import time
import torch
import torch.multiprocessing
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')
import cv2
import os
import os.path as osp

from torch.utils.data import DataLoader
import trimesh
import hydra
import numpy as np
import pandas as pd
import open3d as o3d
from omegaconf import OmegaConf
from einops import rearrange

from copy import copy
from common.manopth.manopth.manolayer import ManoLayer
from common.dataset_utils.grab_dataset import GRABDataset
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.dataset_utils.datamodules import HOIDatasetModule
from common.model.pose_optimizer import optimize_pose_wrt_local_grids, optimize_pose_contactopt
from common.msdf.utils.msdf import get_grid
from common.utils.vis import visualize_recon_hand_w_object
from common.utils.glb import (OBJ_RGB, GREY_RGB, save_glb, solid_mesh, vcolor_mesh, surface_values, contact_rgb,
                              part_rgb, mask_by_contact, cse_to_part, grasp_geoms, recon_hand_geoms, volco_grid_geoms)
from common.evaluation.eval_fns import calculate_fscore, EvalUtil
from common.utils.geometry import calculate_contact_capsule
from matplotlib import pyplot as plt

from prev_sota.contactgen import HandObject as ContactGenHandObject
from prev_sota.contactgen import optimize_pose_contactgen
from prev_sota.hand_model import ArtiHand


@hydra.main(config_path="../config", config_name="gt_test")
def test_contact(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override pose_optimizer to lg_base
    dataset = GRABDataset(cfg.data, split="test", load_msdf=True, object_only=False)
    mano_layer = ManoLayer(
        mano_root=cfg.data.mano_root, use_pca=cfg.pose_optimizer.use_pca,
        side='right', flat_hand_mean=cfg.pose_optimizer.flat_hand_mean,
        ncomps=cfg.pose_optimizer.ncomps
    ).requires_grad_(False).to(device)
    dm = HOIDatasetModule(cfg)
    dm.prepare_data()

    closed_mano_faces = np.load('data/misc/closed_mano_r_faces.npy')
    msdf_k = cfg.msdf.kernel_size
    msdf_scale = cfg.msdf.scale
    grid_coords = get_grid(msdf_k) * msdf_scale  # (K^3, 3)
    cse_ckpt = torch.load(cfg.data.hand_cse_path, weights_only=False)
    cse_dim = cse_ckpt['emb_dim']
    hand_cse = HandCSE(n_verts=778, emb_dim=cse_dim, cano_faces=mano_layer.th_faces.cpu().numpy()).to(device)
    hand_cse.load_state_dict(cse_ckpt['state_dict'])
    hand_cse.eval().requires_grad_(False)

    obj_info = dataset.obj_info
    simp_obj_mesh = dataset.simp_obj_mesh
    obj_hulls_dict = dataset.obj_hulls
    obj_meshes_by_name = {n: trimesh.Trimesh(obj_info[n]['verts'], obj_info[n]['faces']) for n in dataset.obj_info.keys()}
    simp_obj_meshes_by_name = {n: trimesh.Trimesh(simp_obj_mesh[n]['verts'], simp_obj_mesh[n]['faces']) for n in dataset.obj_info.keys()}

    test_loader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=8, collate_fn=HOIDatasetModule.collate_fn)
    print(f"Test dataset size: {len(dataset)}, batch size: {cfg.test.batch_size}")

    eval_util = EvalUtil(num_kp=778)  # one "keypoint" per mesh vertex for AUC
    all_mpvpe, all_f5, all_f15 = [], [], []
    saved_hand_verts, saved_hand_joints = [], []
    t_start = time.time()
    if cfg.contact.name.lower() == 'contactgen':
        config_file = "prev_sota/config.yaml"
        config = OmegaConf.load(config_file)
        hand_part_ids = torch.argmax(mano_layer.th_weights, dim=-1).detach().cpu().numpy()
        cg_ho = ContactGenHandObject(
            device=device, face_path="data/misc/closed_mano_r_faces.npy", hand_part_label=hand_part_ids,
        )
        hand_model = ArtiHand(config['model_params'], pose_size=config['pose_size'])
        checkpoint = torch.load("prev_sota/hand_model.pt")
        hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
        hand_model.eval()
        hand_model.to(device)

    all_handVs, all_handJs = [], []
    all_opt_peak_mb = []   # per-batch peak GPU memory during pose optimization
    for idx, batch in tqdm(enumerate(test_loader)):
        if idx % 20 != 0:
            continue
        elapsed = time.time() - t_start
        avg_batch_time = elapsed / (idx + 1) if idx > 0 else 0.0
        remaining = avg_batch_time * (len(test_loader) - idx - 1)
        print(f"[{idx+1}/{len(test_loader)}] elapsed: {elapsed:.0f}s  avg/batch: {avg_batch_time:.1f}s  ETA: {remaining:.0f}s")
        obj_names = batch['objName']
        obj_meshes = [obj_meshes_by_name[n] for n in obj_names]
        simp_meshes = [simp_obj_meshes_by_name[n] for n in obj_names]
        obj_hulls = [obj_hulls_dict[n] for n in obj_names]
        cur_batch_size = len(obj_names)

        if cfg.contact.name.lower() == 'contactgen':
            # Hand and object in the same (CoM-centred object) frame as the other methods
            ho_gt = HandObject(cfg.data, device=device, mano_layer=mano_layer, normalize=True)
            ho_gt.load_from_batch(batch, obj_templates=obj_meshes)
            world2obj = _rigid_fit(batch['handVerts'].float().to(device), ho_gt.hand_verts)  # (B, 4, 4)
            hand_frames = world2obj[:, None] @ batch['handPartT'].float().to(device)          # (B, 16, 4, 4)
            obj_verts, obj_normals = [], []
            for obj_mesh in ho_gt.obj_models:
                points, fidx = trimesh.sample.sample_surface(obj_mesh, 4096)
                obj_verts.append(points)
                obj_normals.append(obj_mesh.face_normals[fidx])
            obj_verts = torch.as_tensor(np.stack(obj_verts, axis=0), dtype=torch.float32, device=device)  # (B, N*K^3, 3)
            obj_normals = torch.as_tensor(np.stack(obj_normals, axis=0), dtype=torch.float32, device=device)  # (B, N*K^3, 3)
            sample = cg_ho.forward(ho_gt.hand_verts, hand_frames, obj_verts, obj_normals)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            global_pose, mano_pose, mano_shape, mano_trans = optimize_pose_contactgen(
                model=hand_model,
                mano_layer=mano_layer,
                obj_verts=sample['verts_object'],
                obj_cmap=sample['contacts_object'].squeeze(-1),
                obj_partition=sample['partition_object'],
                obj_uv=sample['uv_object']
            )
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                all_opt_peak_mb.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
            gt_handV = ho_gt.hand_verts.detach().cpu().numpy()  # (B, V, 3)

        else:
            ho_gt = HandObject(
                cfg.data, device=device, mano_layer=mano_layer,
                normalize=True, apply_grid_mask=True,
            )
            ho_gt.load_from_batch(
                batch,
                obj_templates=obj_meshes,
                vis_obj_template=simp_meshes,
                obj_hulls=obj_hulls,
                n_resample_obj_pts=4096,  # point contact only: same density as the ContactGen branch
            )

            # vis_geoms = ho_gt.get_vis_geoms(draw_maps=True)
            # o3d.visualization.draw(vis_geoms, show_skybox=False)

            # --- Extract GT contact and CSE from ml_contact ---
            # ml_contact: (B, N, K, K, K, 1+cse_dim)
            if cfg.contact_unit == 'grid':
                lg_contact = ho_gt.ml_contact  # (B, N, K, K, K, 1+cse_dim)

                gt_grid_contact = lg_contact[..., 0]  # (B, N, K, K, K)
                gt_grid_cse = lg_contact[..., 1:]     # (B, N, K, K, K, cse_dim)

                obj_msdf = ho_gt.obj_msdf  # (B, N, K^3 + 3)
                obj_msdf_center = obj_msdf[:, :, msdf_k**3:]  # (B, N, 3)

                grid_coords_dev = grid_coords.view(-1, 3).to(device)
                grid_pts = (obj_msdf_center[:, :, None, :] +
                            grid_coords_dev[None, None, :, :])  # (B, N, K^3, 3)

                # Flatten contact and CSE for the whole batch
                pred_grid_contact = gt_grid_contact.reshape(cur_batch_size, -1).clone()  # (B, N*K^3)
                pred_grid_cse = gt_grid_cse.reshape(cur_batch_size, -1, cse_dim)         # (B, N*K^3, cse_dim)
                grid_coords_flat = grid_pts.reshape(cur_batch_size, -1, 3)               # (B, N*K^3, 3)

                # Threshold low contact values
                # pred_grid_contact[pred_grid_contact < cfg.pose_optimizer.contact_th] = 0

                # Recover approximate hand verts from contact for the whole batch
                # pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
                #     hand_cse, None,
                #     pred_grid_contact,
                #     pred_grid_cse,
                #     grid_coords=grid_coords_flat,
                #     chunk_size=10
                # )
                pred_targetWverts = hand_cse.emb2Wvert(pred_grid_cse)  # (B, N*K^3, 778)

                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                with torch.enable_grad():
                    mano_params, _ = optimize_pose_wrt_local_grids(
                        mano_layer,
                        grid_centers=obj_msdf_center,      # (B, N, 3)
                        target_pts=grid_coords_flat,        # (B, N*K^3, 3)
                        target_W_verts=pred_targetWverts,   # (B, N*K^3, 778)
                        weights=pred_grid_contact,          # (B, N*K^3)
                        n_iter=cfg.pose_optimizer.n_opt_iter,
                        lr=cfg.contact.opt_lr,
                        grid_scale=msdf_scale,
                        w_repulsive=0,
                    )
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    all_opt_peak_mb.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
                mano_trans, global_pose, mano_pose, mano_shape = mano_params

            elif cfg.contact_unit == 'point':
                # 'part' expects (B, N) part ids; None (ContactOpt) starts from the mean pose
                pmap = ho_gt.point_cse if cfg.correspondence_type == 'cse' else \
                       ho_gt.pmap if cfg.correspondence_type == 'part' else None
                if cfg.correspondence_type == 'cse':
                    # ManiDext: capsule GT contact, as for ContactOpt. HandObject's point-contact ablation map
                    # (cfg.data.point_contact) mismatches the optimizer's contact model and breaks the fit.
                    ho_gt.contact_map = calculate_contact_capsule(ho_gt.hand_verts, ho_gt.hand_normals,
                                                                  ho_gt.obj_verts, ho_gt.obj_normals)[0]  # (B, N, 1)
                thin_objects = ['wineglass', 'mug', 'fryingpan']
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                global_pose, mano_pose, mano_shape, mano_trans, init_pose = optimize_pose_contactopt(
                                    mano_layer, ho_gt.obj_verts, ho_gt.obj_normals,
                                    ho_gt.contact_map.squeeze(-1), pmap, n_iter=1000, save_history=False,
                                    partition_type=cfg.correspondence_type, w_pen_cost=40,
                                    hand_cse=hand_cse if cfg.correspondence_type=='cse' else None,
                                    is_thin=torch.LongTensor([obj_name in thin_objects for obj_name in obj_names]).to(device))
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    all_opt_peak_mb.append(torch.cuda.max_memory_allocated(device) / 1024 ** 2)

            gt_handV = ho_gt.hand_verts.detach().cpu().numpy()  # (B, V, 3)

        handV, handJ, _ = mano_layer(
            torch.cat([global_pose, mano_pose], dim=1),
            th_betas=mano_shape, th_trans=mano_trans
        )
        all_handV = handV.detach().cpu().numpy()
        all_handJ = handJ.detach().cpu().numpy()
        all_handVs.append(all_handV)
        all_handJs.append(all_handJ)

        # MPVPE in mm
        mpvpe = np.linalg.norm(all_handV - gt_handV, axis=-1).mean(axis=-1) * 1000  # (B,)
        all_mpvpe.extend(mpvpe.tolist())

        # F-score @ 5mm and 15mm (surface points — use mesh vertices directly, unordered)
        for b in range(cur_batch_size):
            f5, _, _ = calculate_fscore(gt_handV[b], all_handV[b], th=0.005)
            f15, _, _ = calculate_fscore(gt_handV[b], all_handV[b], th=0.015)
            all_f5.append(f5)
            all_f15.append(f15)
            # AUC: feed per-vertex distances (all visible)
            vis = np.ones(778, dtype=bool)
            eval_util.feed(gt_handV[b], vis, all_handV[b], skip_check=False)

        saved_hand_verts.append(all_handV)
        saved_hand_joints.append(all_handJ)

        avg_mpvpe = float(np.mean(mpvpe))
        avg_f5 = float(np.mean(all_f5[-cur_batch_size:]))
        avg_f15 = float(np.mean(all_f15[-cur_batch_size:]))
        mem_str = f"  peak_mem={all_opt_peak_mb[-1]:.1f}MB" if all_opt_peak_mb else ""
        print(f"[{idx}] {obj_names}: MPVPE={avg_mpvpe:.2f}mm  F@5={avg_f5:.4f}  F@15={avg_f15:.4f}{mem_str}")

    # Aggregate
    _, _, auc, _, _ = eval_util.get_measures(val_min=0.0, val_max=0.05, steps=100)
    all_handVs = np.concatenate(all_handVs, axis=0)  # (N, V, 3)
    all_handJs = np.concatenate(all_handJs, axis=0)  # (N, J, 3)

    overall = {
        'MPVPE (mm)':   float(np.mean(all_mpvpe)),
        'F-score@5mm':  float(np.mean(all_f5)),
        'F-score@15mm': float(np.mean(all_f15)),
        'AUC (0-50mm)': float(auc),
    }
    if all_opt_peak_mb:
        overall['Opt peak GPU mem (MB)'] = float(np.max(all_opt_peak_mb))
        overall['Opt mean GPU mem (MB)'] = float(np.mean(all_opt_peak_mb))

    print("\n=== Overall Metrics ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    if all_opt_peak_mb:
        print(f"\n=== Pose optimization peak GPU memory (batch_size={cfg.test.batch_size}) ===")
        print(f"  max over batches : {np.max(all_opt_peak_mb):.1f} MB")
        print(f"  mean over batches: {np.mean(all_opt_peak_mb):.1f} MB")
        print(f"  min over batches : {np.min(all_opt_peak_mb):.1f} MB")

    npz_path = f'tmp/benchmark_contact_rep_hands_{cfg.contact.name}.npz'
    np.savez_compressed(
        npz_path,
        hand_verts=all_handVs,   # (N_total, 778, 3)
        hand_joints=all_handJs, # (N_total, 21, 3)
        mpvpe=np.array(all_mpvpe), # (N_total,)
    )
    print(f"Hand verts/joints saved to {npz_path}")

    df = pd.DataFrame([overall], index=['mean'])
    csv_path = f'tmp/benchmark_contact_rep_results_{cfg.contact.name}.csv'
    df.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")


## ---- GLB export for web viewing ---- ##
def _orientation_rgb(uv):
    """Unit direction -> HSV hue from azimuth; towards white (+z) / black (-z) with elevation."""
    hue = (np.arctan2(uv[:, 1], uv[:, 0]) / (2 * np.pi)) % 1.0
    rgb = plt.colormaps['hsv'](hue)[:, :3]
    z = np.clip(uv[:, 2:3], -1, 1)
    sat = 1 - np.abs(z)
    return np.where(z >= 0, rgb * sat + (1 - sat), rgb * sat)


def _rigid_fit(src, dst):
    """Batched Kabsch: (B, N, 3) -> (B, 4, 4) with dst ~= R @ src + t."""
    src_c, dst_c = src.mean(dim=1, keepdim=True), dst.mean(dim=1, keepdim=True)
    H = (src - src_c).transpose(1, 2) @ (dst - dst_c)
    U, _, Vt = torch.linalg.svd(H)
    D = torch.diag_embed(torch.ones_like(src[:, 0]))
    D[:, 2, 2] = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    T = torch.eye(4, device=src.device).repeat(src.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = dst_c[:, 0] - (R @ src_c.transpose(1, 2))[..., 0]
    return T


def export_comparison_glb(out_dir, b, obj_mesh, closed_faces, mano_faces, hand_part_ids, result_hands, gt_hand,
                          ho_part, ho_cse, cg_maps, hand_emb, grid_centers, grid_pts, grid_contact, grid_cse,
                          recon_verts, recon_mask, grid_scale, pt_radius=0.001, bbox_alpha=0.25,
                          grid_contact_th=0.5):
    """
    Save the per-method maps and fitted grasps of batch element b as .glb files in out_dir.
    All geometries share the CoM-centred object frame, so files can be overlaid in a viewer.
    :param result_hands: {method: (778, 3) fitted hand verts} for contactopt/manidext/contactgen/ours
    :param gt_hand: (778, 3) ground-truth hand verts
    :param ho_part / ho_cse: point-contact HandObjects with part-level / CSE correspondence
    :param cg_maps: ContactGen maps {'contacts_object', 'partition_object', 'uv_object'} for batch element b
    :param grid_centers: (N, 3); grid_pts: (N*K^3, 3); grid_contact: (N, K^3); grid_cse: (N*K^3, D)
    :param grid_contact_th: Volco grids with max contact <= this are not exported (boxes and points)
    """
    obj_v, obj_f = np.asarray(obj_mesh.vertices), np.asarray(obj_mesh.faces)
    pts = ho_cse.obj_verts[b].detach().cpu().numpy()  # sampled surface points (same for all point methods)

    def save(name, geoms):
        save_glb(osp.join(out_dir, f'{name}.glb'), geoms)

    def surface(name, vert_rgb):
        save(name, {name: vcolor_mesh(obj_v, obj_f, vert_rgb)})

    save('object', {'object': solid_mesh(obj_v, obj_f, OBJ_RGB)})
    save('gt_grasp', grasp_geoms(obj_mesh, gt_hand, closed_faces))

    ## ContactOpt: capsule contact + part-level correspondence
    contact = surface_values(obj_mesh, pts, ho_part.contact_map[b].reshape(-1).detach().cpu().numpy())
    parts = surface_values(obj_mesh, pts, ho_part.pmap[b].reshape(-1).cpu().numpy())
    surface('contactopt_contact', contact_rgb(contact))
    surface('contactopt_part_corr', mask_by_contact(part_rgb(parts), contact))
    save('contactopt_grasp', grasp_geoms(obj_mesh, result_hands['contactopt'], closed_faces))

    ## ManipDexterity-style point CSE: contact + CSE correspondence
    contact = surface_values(obj_mesh, pts, ho_cse.contact_map[b].reshape(-1).detach().cpu().numpy())
    cse_parts = cse_to_part(ho_cse.point_cse[b], hand_emb, hand_part_ids)
    parts = surface_values(obj_mesh, pts, cse_parts)
    surface('manidext_contact', contact_rgb(contact))
    surface('manidext_cse_corr', mask_by_contact(part_rgb(parts), contact))
    save('manidext_grasp', grasp_geoms(obj_mesh, result_hands['manidext'], closed_faces))

    ## ContactGen: contact + part-level correspondence + orientation (uv) map
    contact = surface_values(obj_mesh, pts, cg_maps['contacts_object'].reshape(-1).detach().cpu().numpy())
    parts = surface_values(obj_mesh, pts, cg_maps['partition_object'].reshape(-1).cpu().numpy())
    uv = surface_values(obj_mesh, pts, cg_maps['uv_object'].reshape(-1, 3).detach().cpu().numpy())
    surface('contactgen_contact', contact_rgb(contact))
    surface('contactgen_part_corr', mask_by_contact(part_rgb(parts), contact))
    surface('contactgen_orientation', mask_by_contact(_orientation_rgb(uv), contact))
    save('contactgen_grasp', grasp_geoms(obj_mesh, result_hands['contactgen'], closed_faces))

    ## Volco (ours): only grids whose maximum contact exceeds grid_contact_th
    volco = volco_grid_geoms(grid_centers, grid_pts, grid_contact, grid_cse, hand_emb, hand_part_ids, grid_scale,
                             contact_th=grid_contact_th, pt_radius=pt_radius, bbox_alpha=bbox_alpha)
    save('volco_grid_contact', {'object': solid_mesh(obj_v, obj_f, GREY_RGB), 'grids': volco['grids']})
    save('volco_pts_contact', {'grid_points': volco['contact_pts']})
    save('volco_pts_cse', {'grid_points': volco['cse_pts']})
    save('volco_local_hand', recon_hand_geoms(obj_mesh, recon_verts, recon_mask, mano_faces, hand_part_ids, pt_radius))
    save('volco_grasp', grasp_geoms(obj_mesh, result_hands['ours'], closed_faces))


@hydra.main(config_path="../config", config_name="gt_test")
def visualize_comparison(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override pose_optimizer to lg_base
    dataset = GRABDataset(cfg.data, split="test", load_msdf=True, object_only=False)
    mano_layer = ManoLayer(
        mano_root=cfg.data.mano_root, use_pca=cfg.pose_optimizer.use_pca,
        side='right', flat_hand_mean=cfg.pose_optimizer.flat_hand_mean, ncomps=cfg.pose_optimizer.ncomps
    ).requires_grad_(False).to(device)
    dm = HOIDatasetModule(cfg)
    dm.prepare_data()

    closed_mano_faces = np.load('data/misc/closed_mano_r_faces.npy')
    msdf_k = cfg.msdf.kernel_size
    msdf_scale = cfg.msdf.scale
    grid_coords = get_grid(msdf_k) * msdf_scale  # (K^3, 3)
    cse_ckpt = torch.load(cfg.data.hand_cse_path, weights_only=False)
    cse_dim = cse_ckpt['emb_dim']
    hand_cse = HandCSE(n_verts=778, emb_dim=cse_dim, cano_faces=mano_layer.th_faces.cpu().numpy()).to(device)
    hand_cse.load_state_dict(cse_ckpt['state_dict'])
    hand_cse.eval().requires_grad_(False)

    obj_info = dataset.obj_info
    simp_obj_mesh = dataset.simp_obj_mesh
    obj_hulls_dict = dataset.obj_hulls
    obj_meshes_by_name = {n: trimesh.Trimesh(obj_info[n]['verts'], obj_info[n]['faces']) for n in dataset.obj_info.keys()}
    simp_obj_meshes_by_name = {n: trimesh.Trimesh(simp_obj_mesh[n]['verts'], simp_obj_mesh[n]['faces']) for n in dataset.obj_info.keys()}

    train_loader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=8, collate_fn=HOIDatasetModule.collate_fn)
    print(f"Test dataset size: {len(dataset)}, batch size: {cfg.test.batch_size}")

    # Fitted hands saved by test_contact as tmp/benchmark_contact_rep_hands_{cfg.contact.name}.npz.
    # Rows follow the batches test_contact processed (same batch size, every 20th batch).
    result_names = {'contactopt': 'ContactOptPart', 'contactgen': 'ContactGen', 'manidext': 'ManiDext', 'ours': 'VolCo'}
    result_dict = {k: np.load(f'tmp/benchmark_contact_rep_hands_{v}.npz') for k, v in result_names.items()}
    n_results = {k: len(r['mpvpe']) for k, r in result_dict.items()}
    assert len(set(n_results.values())) == 1, f"Result files cover different numbers of samples: {n_results}"
    n_results = n_results['ours']
    ours_result = result_dict['ours']

    # Point-based HandObjects for the previous methods' maps (same batch, different contact unit)
    def point_cfg(corr):
        data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
        data_cfg.contact_unit = 'point'
        data_cfg.correspondence_type = corr
        return data_cfg
    part_cfg, cse_cfg = point_cfg('part'), point_cfg('cse')
    hand_part_ids = torch.argmax(mano_layer.th_weights, dim=-1).detach().cpu().numpy()
    ho_cg = ContactGenHandObject(device=device, face_path="data/misc/closed_mano_r_faces.npy",
                                 hand_part_label=hand_part_ids)
    mano_faces = mano_layer.th_faces.detach().cpu().numpy()

    t_start = time.time()
    res_offset = 0  # row in the result files of the first sample of the current batch
    for idx, batch in enumerate(train_loader):
        if idx % 20 != 0:  # must match the batch selection in test_contact
            continue
        if res_offset >= n_results:
            break
        elapsed = time.time() - t_start
        avg_batch_time = elapsed / (idx + 1) if idx > 0 else 0.0
        remaining = avg_batch_time * (len(train_loader) - idx - 1)
        print(f"[{idx+1}/{len(train_loader)}] elapsed: {elapsed:.0f}s  avg/batch: {avg_batch_time:.1f}s  ETA: {remaining:.0f}s")
        obj_names = batch['objName']
        obj_meshes = [obj_meshes_by_name[n] for n in obj_names]
        simp_meshes = [simp_obj_meshes_by_name[n] for n in obj_names]
        obj_hulls = [obj_hulls_dict[n] for n in obj_names]
        cur_batch_size = len(obj_names)
        res_start, res_offset = res_offset, res_offset + cur_batch_size
        ho_gt = HandObject(
            cfg.data, device=device, mano_layer=mano_layer,
            normalize=True, apply_grid_mask=True,
        )
        ho_gt.load_from_batch(
            batch,
            obj_templates=obj_meshes,
            vis_obj_template=simp_meshes,
            obj_hulls=obj_hulls
        )
        ho_pred = copy(ho_gt)

        lg_contact = ho_gt.ml_contact
        batch_size, n_grids = lg_contact.shape[:2]
        # n_ho_dist = handobject.n_ho_dist
        msdf_center = ho_gt.obj_msdf[:, :, msdf_k**3:]  # (B, N, 3)
        all_pt_coords = (msdf_center[:, :, None, :] + grid_coords.view(1, 1, -1, 3).to(device))  # (B, N, K^3, 3)

        ## First process all grids separately using GRIDAE
        # lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> (b n) c k1 k2 k3')
        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            hand_cse, None,
            lg_contact[..., 0].reshape(batch_size, -1),
            lg_contact[..., 1:].reshape(batch_size, -1, cse_dim),
            grid_coords=all_pt_coords.view(batch_size, -1, 3),
            chunk_size=10  # Process in chunks of 10 to reduce memory peak
        )

        os.makedirs('tmp/gt_recon', exist_ok=True)

        ## Maps of the previous methods, all in the CoM-centred object frame of ho_gt
        ho_part = HandObject(part_cfg, device=device, mano_layer=mano_layer, normalize=True)
        ho_part.load_from_batch(batch)
        ho_cse = HandObject(cse_cfg, device=device, mano_layer=mano_layer, normalize=True)
        ho_cse.load_from_batch(batch)
        # ManiDext is fitted to capsule contact in test_contact, so show that map
        ho_cse.contact_map = calculate_contact_capsule(ho_cse.hand_verts, ho_cse.hand_normals,
                                                       ho_cse.obj_verts, ho_cse.obj_normals)[0]
        # ContactGen maps need the hand part frames: map them from world to the object frame
        world2obj = _rigid_fit(batch['handVerts'].float().to(device), ho_gt.hand_verts)  # (B, 4, 4)
        hand_frames = world2obj[:, None] @ batch['handPartT'].float().to(device)          # (B, 16, 4, 4)
        cg_maps = [ho_cg.forward(ho_gt.hand_verts[b:b+1], hand_frames[b:b+1],
                                 ho_gt.obj_verts[b:b+1], ho_gt.obj_normals[b:b+1])
                   for b in range(cur_batch_size)]

        save_cnt = 0
        for vis_idx in range(cur_batch_size):
            res_idx = res_start + vis_idx                          # row in the result files
            sample_idx = idx * cfg.test.batch_size + vis_idx       # index in the test set (used in file names)

            # Check if ours beats all other methods by >2mm
            ours_mpvpe = ours_result['mpvpe'][res_idx]
            better_than_all = all(
                ours_mpvpe < result_dict[name]['mpvpe'][res_idx] - 2.0
                for name in result_dict if name != 'ours'
            )
            if not better_than_all:
                continue

            # Reconstruct image: GT
            side = 600
            img_gt = ho_gt.vis_img(vis_idx, h=side, w=side, concat_axis=0)  # (H, W, 3)

            # Reconstruct image: ours (recon_hand_w_object)
            img_recon, _ = visualize_recon_hand_w_object(
                pred_hand_verts[vis_idx].detach().cpu().numpy(),
                pred_verts_mask[vis_idx].detach().cpu().numpy(),
                mano_layer.th_faces.detach().cpu().numpy(), ho_gt.vis_obj_models[vis_idx],
                msdf_center=msdf_center[vis_idx].detach().cpu().numpy(),
                part_ids=ho_gt.hand_part_ids,
                grid_scale=cfg.msdf.scale, h=side, w=side,
                concat_axis=0)

            # Collect per-method images (set hand verts/joints then call vis_img)
            method_imgs = []
            method_names = []
            for name, result in result_dict.items():
                pred_handV = result['hand_verts'][res_start:res_start + cur_batch_size]
                pred_handJ = result['hand_joints'][res_start:res_start + cur_batch_size]
                ho_pred.hand_verts = torch.as_tensor(pred_handV, device=device)
                ho_pred.hand_joints = torch.as_tensor(pred_handJ, device=device)
                method_imgs.append(ho_pred.vis_img(vis_idx, h=side, w=side, concat_axis=0))
                method_names.append(name)

            # Concatenate horizontally: GT | recon | method1 | method2 | ...
            row = np.concatenate(method_imgs + [img_gt, img_recon], axis=1)
            row_uint8 = (np.clip(row, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(f'tmp/gt_recon/{"-".join(method_names)}-{sample_idx:05d}.png',
                        cv2.cvtColor(row_uint8, cv2.COLOR_RGB2BGR))
            print(f"  Saved sample {sample_idx} (ours={ours_mpvpe:.2f}mm)")

            # 3D geometries for web viewing
            result_hands = {name: result['hand_verts'][res_idx] for name, result in result_dict.items()}
            glb_dir = osp.join('tmp', 'contact_3dobj', f'{sample_idx:05d}_{obj_names[vis_idx]}')
            export_comparison_glb(
                glb_dir, vis_idx, ho_gt.obj_models[vis_idx], closed_mano_faces, mano_faces, hand_part_ids,
                result_hands, ho_gt.hand_verts[vis_idx].detach().cpu().numpy(), ho_part, ho_cse, {k: v[0] for k, v in cg_maps[vis_idx].items()}, ho_gt.hand_cse,
                grid_centers=msdf_center[vis_idx].detach().cpu().numpy(),
                grid_pts=all_pt_coords[vis_idx].reshape(-1, 3).detach().cpu().numpy(),
                grid_contact=lg_contact[vis_idx, ..., 0].reshape(n_grids, -1).detach().cpu().numpy(),
                grid_cse=lg_contact[vis_idx, ..., 1:].reshape(-1, cse_dim),
                recon_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                recon_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                grid_scale=msdf_scale)
            print(f"  Saved glb files to {glb_dir}")
            save_cnt += 1
            if save_cnt > 3:
                break


if __name__ == "__main__":
    test_contact()
    # visualize_comparison()
