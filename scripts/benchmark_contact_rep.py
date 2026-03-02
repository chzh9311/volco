import time
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import cv2
import os

from torch.utils.data import DataLoader
import trimesh
import hydra
import numpy as np
import pandas as pd
from multiprocessing.pool import Pool
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
from common.evaluation.eval_fns import calculate_fscore, EvalUtil

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

    batch_size = 16
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=HOIDatasetModule.collate_fn)
    print(f"Test dataset size: {len(dataset)}, batch size: {batch_size}")

    pool = Pool(processes=8)
    eval_util = EvalUtil(num_kp=778)  # one "keypoint" per mesh vertex for AUC
    all_mpvpe, all_f5, all_f15 = [], [], []
    t_start = time.time()
    if cfg.correspondence_type == 'contactgen':
        config_file = "prev_sota/config.yaml"
        config = OmegaConf.load(config_file)
        hand_part_ids = torch.argmax(mano_layer.th_weights, dim=-1).detach().cpu().numpy()
        ho_gt = ContactGenHandObject(
            device=device, face_path="data/misc/closed_mano_r_faces.npy", hand_part_label=hand_part_ids,
        )
        hand_model = ArtiHand(config['model_params'], pose_size=config['pose_size'])
        checkpoint = torch.load("prev_sota/hand_model.pt")
        hand_model.load_state_dict(checkpoint['state_dict'], strict=True)
        hand_model.eval()
        hand_model.to(device)

    all_handVs, all_handJs = [], []
    for idx, batch in enumerate(train_loader):
        if idx > 16: 
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

        if cfg.correspondence_type == 'contactgen':
            obj_verts, obj_normals = [], []
            for obj_mesh in obj_meshes:
                points, fidx = trimesh.sample.sample_surface(obj_mesh, 4096)
                obj_verts.append(points)
                obj_normals.append(obj_mesh.face_normals[fidx])
            obj_verts = torch.as_tensor(np.stack(obj_verts, axis=0), dtype=torch.float32, device=device)  # (B, N*K^3, 3)
            obj_normals = torch.as_tensor(np.stack(obj_normals, axis=0), dtype=torch.float32, device=device)  # (B, N*K^3, 3)
            sample = ho_gt.forward(batch['handVerts'].to(device), batch['handPartT'].to(device), obj_verts, obj_normals)

            global_pose, mano_pose, mano_shape, mano_trans = optimize_pose_contactgen(
                model=hand_model,
                mano_layer=mano_layer,
                obj_verts=sample['verts_object'],
                obj_cmap=sample['contacts_object'].squeeze(-1),
                obj_partition=sample['partition_object'],
                obj_uv=sample['uv_object']
            )
            gt_handV = batch['handVerts'].numpy()  # (B, V, 3)

        else:
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

                with torch.enable_grad():
                    mano_params, _ = optimize_pose_wrt_local_grids(
                        mano_layer,
                        grid_centers=obj_msdf_center,      # (B, N, 3)
                        target_pts=grid_coords_flat,        # (B, N*K^3, 3)
                        target_W_verts=pred_targetWverts,   # (B, N*K^3, 778)
                        weights=pred_grid_contact,          # (B, N*K^3)
                        n_iter=cfg.pose_optimizer.n_opt_iter,
                        lr=cfg.pose_optimizer.opt_lr,
                        grid_scale=msdf_scale,
                        w_repulsive=0,
                    )
                mano_trans, global_pose, mano_pose, mano_shape = mano_params
            
            elif cfg.contact_unit == 'point':
                pmap = ho_gt.point_cse if cfg.correspondence_type=='cse' else ho_gt.part_map
                thin_objects = ['wineglass', 'mug', 'fryingpan']
                global_pose, mano_pose, mano_shape, mano_trans, init_pose = optimize_pose_contactopt(
                                    mano_layer, ho_gt.obj_verts, ho_gt.obj_normals,
                                    ho_gt.contact_map, pmap, n_iter=1000, save_history=False,
                                    partition_type=cfg.correspondence_type, w_pen_cost=40, 
                                    hand_cse=hand_cse if cfg.correspondence_type=='cse' else None,
                                    is_thin=torch.LongTensor([obj_name in thin_objects for obj_name in obj_names]).to(device))

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

        avg_mpvpe = float(np.mean(mpvpe))
        avg_f5 = float(np.mean(all_f5[-cur_batch_size:]))
        avg_f15 = float(np.mean(all_f15[-cur_batch_size:]))
        print(f"[{idx}] {obj_names}: MPVPE={avg_mpvpe:.2f}mm  F@5={avg_f5:.4f}  F@15={avg_f15:.4f}")

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

    print("\n=== Overall Metrics ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    npz_path = f'tmp/benchmark_contact_rep_hands_{cfg.contact_unit}_{cfg.correspondence_type}.npz'
    np.savez_compressed(
        npz_path,
        hand_verts=all_handVs,   # (N_total, 778, 3)
        hand_joints=all_handJs, # (N_total, 21, 3)
        mpvpe=np.array(all_mpvpe), # (N_total,)
    )
    print(f"Hand verts/joints saved to {npz_path}")

    df = pd.DataFrame([overall], index=['mean'])
    csv_path = f'tmp/benchmark_contact_rep_results_{cfg.contact_unit}_{cfg.correspondence_type}.csv'
    df.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")

    pool.close()
    pool.join()


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

    batch_size = 16
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=HOIDatasetModule.collate_fn)
    print(f"Test dataset size: {len(dataset)}, batch size: {batch_size}")

    contactopt_result = np.load('tmp/benchmark_contact_rep_hands_point_none.npz')
    contactgen_result = np.load('tmp/benchmark_contact_rep_hands_point_contactgen.npz')
    manidext_result = np.load('tmp/benchmark_contact_rep_hands_point_cse.npz')
    ours_result = np.load('tmp/benchmark_contact_rep_hands_grid_cse.npz')
    result_dict = {
        'contactopt': contactopt_result,
        'contactgen': contactgen_result,
        'manidext': manidext_result,
        'ours': ours_result,
    }

    t_start = time.time()
    for idx, batch in enumerate(train_loader):
        if idx > 16: 
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

        save_cnt = 0
        for vis_idx in range(cur_batch_size):
            sample_idx = idx * cur_batch_size + vis_idx

            # Check if ours beats all other methods by >2mm
            ours_mpvpe = ours_result['mpvpe'][sample_idx]
            better_than_all = all(
                ours_mpvpe < result_dict[name]['mpvpe'][sample_idx] - 2.0
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
                pred_handV = result['hand_verts'][idx*cur_batch_size:(idx+1)*cur_batch_size]
                pred_handJ = result['hand_joints'][idx*cur_batch_size:(idx+1)*cur_batch_size]
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
            save_cnt += 1
            if save_cnt > 3:
                break


if __name__ == "__main__":
    # test_contact()
    visualize_comparison()
