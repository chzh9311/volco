import os
import os.path as osp
import random
from einops.array_api import rearrange
import torch
from torch.utils.data import DataLoader
import trimesh
import hydra
import open3d as o3d
from matplotlib import pyplot as plt
from common.manopth.manopth.manolayer import ManoLayer
from common.dataset_utils.grab_dataset import GRABDataset
from common.dataset_utils.datamodules import HOIDatasetModule, LocalGridDataModule
from common.utils.vis import visualize_local_grid, visualize_local_grid_with_hand, o3dmesh, parse_hex_color, visualize_recon_hand_w_object
from common.msdf.utils.msdf import get_grid
from common.dataset_utils.hoi4d_dataset import HOI4DHandDataModule
from common.utils.geometry import GridDistanceToContact
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
# from common.model.vae.grid_vae import MLCVAE
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf


def visualize_grid_sdf(batch, cfg, dm):
    """
    Visualize grid SDFs from batch['objMsdf'] along with the object mesh.

    The objMsdf tensor has shape (B, N, K^3 + 3) where:
    - First K^3 dimensions are SDF values of the local grid
    - Last 3 dimensions are the grid centers

    Args:
        batch: dict containing 'objMsdf' tensor
        cfg: config with msdf.kernel_size and msdf.scale
        dm: datamodule with access to object mesh info
    """
    obj_msdf = batch['objMsdf']  # (B, N, K^3 + 3)
    kernel_size = cfg.msdf.kernel_size
    scale = cfg.msdf.scale

    B, N, _ = obj_msdf.shape
    k3 = kernel_size ** 3

    # Get normalized grid coordinates from get_grid function
    normalized_coords = get_grid(kernel_size).reshape(-1, 3).numpy()  # (K^3, 3)

    # Use colormap for SDF values
    cmap = plt.colormaps['coolwarm']

    for b in range(min(B, 1)):  # Visualize first sample in batch
        sdf_values = obj_msdf[b, :, :k3].cpu().numpy()  # (N, K^3)
        grid_centers = obj_msdf[b, :, k3:].cpu().numpy()  # (N, 3)

        # Collect all points and their SDF values
        all_points = []
        all_sdf = []

        for i in range(N):
            # Scale and translate grid points to world coordinates
            grid_points = grid_centers[i] + normalized_coords * scale  # (K^3, 3)
            all_points.append(grid_points)
            all_sdf.append(sdf_values[i])

        all_points = np.concatenate(all_points, axis=0)  # (N * K^3, 3)
        all_sdf = np.concatenate(all_sdf, axis=0)  # (N * K^3,)

        # Normalize SDF values for coloring (clip to reasonable range)
        sdf_min, sdf_max = np.percentile(all_sdf, [5, 95])
        sdf_normalized = np.clip((all_sdf - sdf_min) / (sdf_max - sdf_min + 1e-8), 0, 1)

        # Apply colormap
        colors = cmap(sdf_normalized)[:, :3]  # (N * K^3, 3)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Also create point cloud for grid centers
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(grid_centers)
        centers_pcd.paint_uniform_color([0, 1, 0])  # Green for centers

        # Create object mesh
        obj_name = batch['objName'][b]
        obj_verts = dm.val_set.obj_info[obj_name]['verts'].copy()
        obj_faces = dm.val_set.obj_info[obj_name]['faces']

        # Apply aug_rot if present
        if 'aug_rot' in batch:
            aug_rot = batch['aug_rot'][b].cpu().numpy()  # (3, 3)
            obj_verts = obj_verts @ aug_rot.T

        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
        obj_mesh.compute_vertex_normals()
        obj_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color

        print(f"\nVisualizing sample {b} (object: {obj_name}):")
        print(f"  Total grid points: {all_points.shape[0]}")
        print(f"  Number of grids: {N}")
        print(f"  SDF range: [{all_sdf.min():.4f}, {all_sdf.max():.4f}]")
        print(f"  Color range (clipped): [{sdf_min:.4f}, {sdf_max:.4f}]")
        print("  Blue = negative SDF (inside), Red = positive SDF (outside)")
        print("  Green points = grid centers, Gray mesh = object")

        o3d.visualization.draw_geometries([pcd, centers_pcd, obj_mesh],
                                          window_name=f"Grid SDF Visualization - Sample {b} ({obj_name})")

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)

@hydra.main(config_path="../config", config_name="mlcdiff")
def vis_msdf_data_sample(cfg):
    dm = HOIDatasetModule(cfg)
    mano_layer = ManoLayer(mano_root=cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)

    print("Preparing data...")
    dm.prepare_data()

    print("Setting up train dataset...")
    dm.setup('fit')

    train_loader = DataLoader(dm.val_set, batch_size=1, shuffle=False,
                              num_workers=4, collate_fn=dm.collate_fn)
    print(f"Train dataset size: {len(dm.train_set)}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Batch size: {dm.train_batch_size}")

    obj_info = dm.train_set.obj_info
    return 

    print("\nVisualizing contact grids with hand...")
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches")):
        # if 'train' not in batch['objName'][0] and batch_idx % 20 != 0:
        #     continue  # Only visualize cube samples for now
        if batch_idx % 100 != 0:
            continue  # Only visualize every 20th batch to reduce load
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        handobject = HandObject(cfg.data, device=device, mano_layer=mano_layer)
        obj_templates = [trimesh.Trimesh(obj_info[name]['verts'], obj_info[name]['faces'], process=False)
                         for name in batch['objName']]
        handobject.load_from_batch(batch, obj_templates=obj_templates, vis_obj_template=obj_templates)
        index_verts = handobject.hand_verts[:, handobject.hand_part_ids == 3]
        index_centre = index_verts.mean(dim=1)
        grid_dist = torch.norm(handobject.obj_msdf[:, :, -3:] - index_centre[:, None, :], dim=-1)  # (B, N) distance from each grid center to index fingertip center
        grid_indices = torch.argmin(grid_dist, dim=-1)  # (B,) indices of closest grid to index fingertip

        # cse_ckpt = torch.load(cfg.data.hand_cse_path, weights_only=False)
        # hand_cse = HandCSE(n_verts=778, emb_dim=4, cano_faces=mano_layer.th_faces.cpu().numpy()).to(device)
        # hand_cse.load_state_dict(cse_ckpt['state_dict'])
        # hand_cse.eval().requires_grad_(False)

        # batch_size = handobject.batch_size
        # n_grids = handobject.obj_msdf.shape[1]

        # lg_contact = handobject.ml_contact
        # batch_size, n_grids = lg_contact.shape[:2]
        # msdf_k = cfg.msdf.kernel_size
        # # n_ho_dist = handobject.n_ho_dist

        # obj_msdf_center = handobject.obj_msdf[:, :, msdf_k**3:] # B x 3

        # normalized_grid_coords = get_grid(cfg.msdf.kernel_size).to(device)
        # grid_coords = normalized_grid_coords * cfg.msdf.scale  # (K^3, 3)
        # grid_coords = obj_msdf_center[:, :, None, :] + grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        # pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
        #     hand_cse, None,
        #     lg_contact[..., 0].reshape(batch_size, -1),
        #     lg_contact[..., 1:].reshape(batch_size, -1, 4),
        #     grid_coords=grid_coords.reshape(batch_size, -1, 3),
        #     chunk_size=10  # Process in chunks of 10 to reduce memory peak
        # )
        # img_side = 400

        for b in range(handobject.hand_verts.shape[0]):
            obj_name = batch['objName'][b]
            grid_idx = grid_indices[b].item()
            # recon_img, pred_geoms = visualize_recon_hand_w_object(
            #     hand_verts=pred_hand_verts[b].detach().cpu().numpy(),
            #     hand_verts_mask=pred_verts_mask[b].detach().cpu().numpy(),
            #     hand_faces=mano_layer.th_faces.detach().cpu().numpy(),
            #     obj_mesh=handobject.vis_obj_models[b],
            #     part_ids=handobject.hand_part_ids,
            #     msdf_center=obj_msdf_center[b].detach().cpu().numpy(),
            #     grid_scale=cfg.msdf.scale,
            #     h=img_side, w=img_side)

            # hand_geom = o3dmesh(handobject.hand_verts[b].cpu().numpy(), mano_layer.th_faces.detach().cpu().numpy(), color="#F2A98D") # pink hand
            # obj_geom = o3dmesh(obj_templates[b].vertices, obj_templates[b].faces, color="#6293A8") # blue object
            geoms = handobject.vis_all_grids_with_hand(obj_templates=obj_templates, idx=0, grid_idx=grid_idx)
            # bbox_geoms = [g for g in geoms if isinstance(g, o3d.geometry.LineSet)]
            # bbox_geoms = random.sample(bbox_geoms, len(bbox_geoms) // 2)
            # o3d.visualization.draw_geometries([hand_geom], window_name=f"All Grid BBoxes")
            # geoms = handobject.get_vis_geoms(idx=b)
            o3d.visualization.draw_geometries(geoms, window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | All Grids with Hand")
            # o3d.visualization.draw_geometries([g['geometry'] for g in geoms], window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | All Grids with Hand")
            # o3d.visualization.draw_geometries([obj_geom])
            # o3d.visualization.draw_geometries(pred_geoms)
            break


            # for grid_idx in range(10):
            #     left_geoms, right_geoms = handobject.vis_grid_detail(obj_templates, idx=b, pt_idx=grid_idx)
            #     # o3d.visualization.draw_geometries(
            #     #     left_geoms,
            #     #     window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | Grid {grid_idx} — SDF")
            #     o3d.visualization.draw_geometries(
            #         right_geoms[1:],
            #         window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | Grid {grid_idx} — Contact")

            #     o3d.visualization.draw_geometries(
            #         [right_geoms[0]],
            #         window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | Grid {grid_idx} — Contact")



@hydra.main(config_path="../config", config_name="gridae")
def vis_local_grid_interact(cfg):
    dm = LocalGridDataModule(cfg)
    mano_layer = ManoLayer(mano_root = cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    hand_faces = mano_layer.th_faces.numpy()
    # dm.prepare_data()
    phase = 'validate'
    dm.setup(phase)
    # train_loader = dm.train_dataloader()
    if phase == 'validate' or phase == 'train':
        loader = dm.val_dataloader()
        # train_dataset = dm.train_set
        # print("training dataset size:", len(train_dataset))
        dataset = dm.val_set
        print("validation dataset size:", len(dataset))
    else:
        loader = dm.test_dataloader()
        dataset = dm.test_set
        print(f"test dataset size: {len(dataset)}")
    # test_loader = dm.test_dataloader()
    # return
    index_tip_vid = 317  # MANO index fingertip vertex (same as ManoLayer tips)
    n_vis, max_vis = 0, 10
    export_video = True  # True: render animate_local_grid_section to mp4; False: interactive Open3D window
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Visualizing {phase} data"):
        if batch_idx % 100 != 0 and batch_idx < 200:
            continue
        grid_data = torch.cat([batch['gridSDF'], batch['gridContact'], batch['gridHandCSE']], dim=-1)  # (B, K, K, K, C)
        batch_size = grid_data.shape[0]
        # nHandVerts are relative to the grid center; the grid spans [-scale, scale] per axis
        index_tip = batch['nHandVerts'][:, index_tip_vid]  # (B, 3)
        tip_in_grid = (index_tip.abs() <= cfg.msdf.scale).all(dim=-1)  # (B,)
        for b in torch.nonzero(tip_in_grid).flatten().tolist():
            if b > 2:
                continue  # Only visualize first 3 samples in batch
            if n_vis >= max_vis:
                return
            n_vis += 1
            # Extract data for this sample
            local_grid = grid_data[b].cpu().numpy()  # (K, K, K, C)
            contact_point = batch['objSamplePt'][b].cpu().numpy()  # (3,)
            nhand_verts = batch['nHandVerts'][b].cpu().numpy()  # (778, 3)
            obj_rot = batch['objRot'][b].cpu().numpy()  # (3, 3) or axis-angle (3,)
            obj_trans = batch['objTrans'][b].cpu().numpy()  # (3,)
            obj_name = batch['obj_name'][b]

            # Reconstruct object mesh
            obj_mesh_data = dataset.simp_obj_mesh[obj_name]
            # Handle rotation - check if it's axis-angle or matrix
            if obj_rot.shape == (3,):
                from pytorch3d.transforms import axis_angle_to_matrix
                objR = axis_angle_to_matrix(torch.from_numpy(obj_rot)).numpy()
                if dataset.dataset_name == 'grab':
                    objR = objR.T
            else:
                objR = obj_rot

            obj_verts = (objR @ obj_mesh_data['verts'].T).T + obj_trans
            obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_mesh_data['faces'], process=False)
            hand_verts = nhand_verts + contact_point[np.newaxis, :]

            # Visualize
            print(f"\nVisualizing sample {b} from batch {batch_idx} (object: {obj_name})")
            if export_video:
                # Full-res object mesh + closed hand for clean close-ups and watertight sections
                full_obj = dataset.obj_info.get(obj_name, {})
                if 'verts' in full_obj:
                    obj_mesh = trimesh.Trimesh((objR @ full_obj['verts'].T).T + obj_trans, full_obj['faces'], process=False)
                hand_mesh = trimesh.Trimesh(hand_verts, dataset.close_mano_faces, process=False)
                out_path = osp.join('tmp', 'local_grid_anim', f'{batch_idx:04d}_{b:03d}_{obj_name}.mp4')
                animate_local_grid_section(hand_mesh, obj_mesh, contact_point, cfg.msdf.scale,
                                           hand_verts[index_tip_vid], out_path)
            else:
                geoms = visualize_local_grid_with_hand(local_grid, hand_verts, hand_faces, dataset.hand_cse, cfg.msdf.kernel_size,
                                               cfg.msdf.scale, contact_point=contact_point, obj_mesh=obj_mesh)
                o3d.visualization.draw_geometries(geoms)
            # break
        # Only visualize first batch
        # break


## ---- Local grid animation: full scene -> crop to grid -> zoom to sectional view ---- ##

def _split_mesh_by_halfspaces(verts, faces, normals, offsets):
    """
    Split a triangle mesh by the convex region {p : n_i . p <= d_i for all i}.
    Straddling triangles are cut exactly (sequential Sutherland-Hodgman), so the
    inside and outside parts share their boundary.
    Returns (inside_verts, inside_faces), (outside_verts, outside_faces).
    """
    normals = np.asarray(normals, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces)
    vert_out = verts @ normals.T > offsets[None]  # (V, P): vertex violates plane
    tri_out = vert_out[faces]                        # (F, 3, P)
    fully_in = ~tri_out.any(axis=(1, 2))
    fully_out = tri_out.all(axis=1).any(axis=-1)     # all 3 verts beyond one plane
    straddle = ~(fully_in | fully_out)

    def _clip(poly, n, d, keep_inside):
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            da, db = a @ n - d, b @ n - d
            a_in = da <= 0 if keep_inside else da >= 0
            b_in = db <= 0 if keep_inside else db >= 0
            if a_in:
                out.append(a)
            if a_in != b_in:
                out.append(a + da / (da - db) * (b - a))
        return out

    pieces_in, pieces_out = [], []
    for tri in faces[straddle]:
        poly = [verts[i] for i in tri]
        for n, d in zip(normals, offsets):
            outside_piece = _clip(poly, n, d, keep_inside=False)
            if len(outside_piece) >= 3:
                pieces_out.append(outside_piece)
            poly = _clip(poly, n, d, keep_inside=True)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            pieces_in.append(poly)

    def _assemble(tri_faces, polys):
        # Whole triangles keep the original vertex buffer; clipped polygons are fan-triangulated
        v_list, f_list = [verts], [tri_faces]
        base = len(verts)
        for poly in polys:
            v_list.append(np.asarray(poly))
            f_list.append(np.array([[base, base + k, base + k + 1] for k in range(1, len(poly) - 1)]))
            base += len(poly)
        f = np.concatenate([x.reshape(-1, 3) for x in f_list], axis=0).astype(np.int64)
        m = trimesh.Trimesh(np.concatenate(v_list, axis=0), f, process=False)
        m.remove_unreferenced_vertices()
        return m.vertices, m.faces

    return _assemble(faces[fully_in], pieces_in), _assemble(faces[fully_out], pieces_out)


def _section_cap(mesh, origin, normal, u, v, half_extent):
    """
    Filled cross-section of a (closed) mesh on the plane (origin, normal), cropped to the
    square [-half_extent, half_extent]^2 spanned by (u, v). Returns (verts, faces) or None.
    """
    from shapely.geometry import box as shapely_box
    to_2d = np.eye(4)
    to_2d[:3, :3] = np.stack([u, v, normal], axis=0)
    to_2d[:3, 3] = -to_2d[:3, :3] @ origin
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return None
    try:
        planar, _ = section.to_planar(to_2D=to_2d, check=False)
        polygons = planar.polygons_full
    except Exception:
        return None
    crop = shapely_box(-half_extent, -half_extent, half_extent, half_extent)
    v_list, f_list, base = [], [], 0
    for poly in polygons:
        clipped = poly.intersection(crop)
        for part in getattr(clipped, 'geoms', [clipped]):
            if part.is_empty or part.geom_type != 'Polygon' or part.area < 1e-12:
                continue
            v2, f = trimesh.creation.triangulate_polygon(part, engine='triangle')
            v_list.append(origin[None] + v2[:, :1] * u[None] + v2[:, 1:2] * v[None])
            f_list.append(f + base)
            base += len(v2)
    if not v_list:
        return None
    verts, faces = np.concatenate(v_list), np.concatenate(f_list)
    # Orient caps towards the camera side of the plane
    tri = verts[faces]
    if np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(axis=0) @ normal < 0:
        faces = faces[:, ::-1]
    return verts, faces


_OFFSCREEN_RENDERER = None


def animate_local_grid_section(hand_mesh, obj_mesh, center, scale, tip, out_path,
                               w=960, h=960, fps=30, up=(0.0, 0.0, 1.0), fov=30.0,
                               hand_color="#F2A98D", obj_color="#6293A8", bbox_color="#102540",
                               cap_shade=0.85, tip_radius=0.008,
                               durations=(1.0, 1.0, 0.5, 1.5, 0.5, 2.0, 1.5, 1.5)):
    """
    Render a video of one local grid:
      1. hold: full hand + object only
      2. grid bbox (wireframe only) gradually appears
      3. hold: full hand + object + grid bbox
      4. fade: parts outside the bbox vanish, parts inside remain
      5. hold
      6. camera zooms onto the bbox and levels to a horizontal view
      7. section plane sweeps from the bbox front face to the index fingertip (filled caps)
      8. hold on the sectional view of the fingertip-object contact

    :param hand_mesh: trimesh, closed MANO hand in the same frame as obj_mesh
    :param obj_mesh: trimesh object mesh
    :param center: (3,) grid center; the grid spans center +- scale per axis
    :param tip: (3,) index fingertip vertex; the section plane passes through the centroid of
                hand vertices within tip_radius of it (i.e. through the finger, not its surface)
    :param cap_shade: brightness factor of the flat fill drawn on cut surfaces
    :param durations: seconds for stages 1-8
    """
    import imageio
    from open3d.visualization import rendering

    center = np.asarray(center, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    bmin, bmax = center - scale, center + scale
    eye3 = np.eye(3)
    box_normals = np.concatenate([eye3, -eye3], axis=0)
    box_offsets = np.concatenate([bmax, -bmin], axis=0)

    # Horizontal view direction: a bbox axis perpendicular to `up`, chosen so the local contact
    # normal lies in the image plane (the fingertip-object gap is seen from the side).
    closest, _, _ = trimesh.proximity.closest_point(obj_mesh, tip[None])
    contact_n = tip - closest[0]
    contact_n = contact_n / (np.linalg.norm(contact_n) + 1e-12)
    horiz_axes = [a for a in eye3 if abs(a @ up) < 0.5]
    view_axis = min(horiz_axes, key=lambda a: abs(a @ contact_n))
    # Camera on the side away from the hand body, so the hand is not in front of the fingertip
    hand_side = (hand_mesh.vertices.mean(axis=0) - tip) @ view_axis
    view_dir = -view_axis if hand_side > 0 else view_axis   # unit vector from center towards camera
    sec_u = np.cross(up, view_dir)
    sec_v = up.copy()

    # Camera keyframes (spherical interpolation around the grid center)
    def _dir(azim, elev):
        right = np.cross(up, view_dir)
        horiz = np.cos(azim) * view_dir + np.sin(azim) * right
        return np.cos(elev) * horiz + np.sin(elev) * up

    tan_half = np.tan(np.radians(fov) / 2)

    def _fit_dist(pts, azim, elev, margin):
        # Smallest camera distance (looking at `center`) whose square frustum contains all pts
        d = _dir(azim, elev)
        right = np.cross(d, up)
        right = right / np.linalg.norm(right)
        cam_up = np.cross(right, d)
        rel = pts - center
        extent = np.maximum(np.abs(rel @ right), np.abs(rel @ cam_up)) * margin
        return (rel @ d + extent / tan_half).max()

    all_verts = np.concatenate([hand_mesh.vertices, obj_mesh.vertices], axis=0)
    box_corners = center + scale * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    cam_far = dict(azim=np.radians(40), elev=np.radians(30))
    cam_far['dist'] = _fit_dist(all_verts, cam_far['azim'], cam_far['elev'], margin=1.1)
    cam_near = dict(azim=0.0, elev=0.0)
    cam_near['dist'] = _fit_dist(box_corners, cam_near['azim'], cam_near['elev'], margin=1.3)

    # Split meshes once for the fade stage
    (hin_v, hin_f), (hout_v, hout_f) = _split_mesh_by_halfspaces(hand_mesh.vertices, hand_mesh.faces, box_normals, box_offsets)
    (oin_v, oin_f), (oout_v, oout_f) = _split_mesh_by_halfspaces(obj_mesh.vertices, obj_mesh.faces, box_normals, box_offsets)

    def _o3d_mesh(v, f):
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        m.compute_vertex_normals()
        return m

    def _material(color, alpha=1.0):
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultLit' if alpha >= 0.999 else 'defaultLitTransparency'
        mat.base_color = [*parse_hex_color(color), alpha]
        mat.base_roughness = 0.7
        return mat

    def cap_material(color):
        mat = _material(color)
        mat.base_color = [*(np.asarray(parse_hex_color(color)) * cap_shade), 1.0]
        return mat

    def _cap_mesh(v, f):
        # Constant normal facing the camera -> uniform flat fill
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
        m.vertex_normals = o3d.utility.Vector3dVector(np.repeat(view_dir[None], len(v), axis=0))
        return m

    hand_in, hand_out = _o3d_mesh(hin_v, hin_f), _o3d_mesh(hout_v, hout_f)
    obj_in, obj_out = _o3d_mesh(oin_v, oin_f), _o3d_mesh(oout_v, oout_f)

    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    bbox_ls = o3d.geometry.LineSet()
    bbox_ls.points = o3d.utility.Vector3dVector(bmin + corners * 2 * scale)
    bbox_ls.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                                                [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])
    def _line_material(t):
        # Line alpha is not blended by Filament; fade in by blending from white instead
        # (a white line renders identical to the background)
        mat = rendering.MaterialRecord()
        mat.shader = 'unlitLine'
        mat.base_color = [*((1 - t) * np.ones(3) + t * np.asarray(parse_hex_color(bbox_color))), 1.0]
        mat.line_width = 3.0
        return mat

    # Filament segfaults when a second OffscreenRenderer is created in one process, so reuse it
    global _OFFSCREEN_RENDERER
    if _OFFSCREEN_RENDERER is None:
        _OFFSCREEN_RENDERER = rendering.OffscreenRenderer(w, h)
    renderer = _OFFSCREEN_RENDERER
    scene = renderer.scene
    scene.clear_geometry()
    scene.set_background([1.0, 1.0, 1.0, 1.0])

    def _set(name, geom, mat):
        if scene.has_geometry(name):
            scene.remove_geometry(name)
        if geom is not None:
            scene.add_geometry(name, geom, mat)

    def _set_camera(cam):
        eye = center + cam['dist'] * _dir(cam['azim'], cam['elev'])
        # Explicit clip planes: the automatic near plane clips everything in the close-up
        renderer.setup_camera(fov, center.tolist(), eye.tolist(), up.tolist(),
                              0.01 * cam['dist'], 10.0 * cam_far['dist'])

    def _lerp_cam(t):
        return {k: (1 - t) * cam_far[k] + t * cam_near[k] for k in cam_far}

    def _smooth(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8, macro_block_size=16)

    def _emit(n_frames, update):
        for i in range(n_frames):
            update(i / max(n_frames - 1, 1))
            writer.append_data(np.asarray(renderer.render_to_image()))

    n = [max(int(round(d * fps)), 1) for d in durations]
    _set('hand_in', hand_in, _material(hand_color))
    _set('obj_in', obj_in, _material(obj_color))
    _set('hand_out', hand_out, _material(hand_color))
    _set('obj_out', obj_out, _material(obj_color))
    _set_camera(cam_far)

    # 1. Hand + object only
    _emit(n[0], lambda t: None)

    # 2. Bbox gradually appears, then hold on the full scene with bbox
    _emit(n[1], lambda t: _set('bbox', bbox_ls, _line_material(_smooth(t))))
    _emit(n[2], lambda t: None)

    # 4. Outside parts fade away
    def _stage_fade(t):
        alpha = 1.0 - _smooth(t)
        _set('hand_out', hand_out if alpha > 1e-3 else None, _material(hand_color, alpha))
        _set('obj_out', obj_out if alpha > 1e-3 else None, _material(obj_color, alpha))
    _emit(n[3], _stage_fade)
    _emit(n[4], lambda t: None)

    # 6. Zoom in + level to horizontal view
    _emit(n[5], lambda t: _set_camera(_lerp_cam(_smooth(t))))

    # 7. Section plane sweeps from the bbox front face to the fingertip. The tip vertex lies on the
    # finger surface, so cut through the finger interior (centroid of vertices near the tip) instead.
    tip_verts = hand_mesh.vertices[np.linalg.norm(hand_mesh.vertices - tip, axis=1) < tip_radius]
    tip_core = tip_verts.mean(axis=0) if len(tip_verts) else tip
    s_front, s_tip = scale, np.clip((tip_core - center) @ view_dir, -scale, scale)
    def _stage_section(t):
        s = (1 - _smooth(t)) * s_front + _smooth(t) * s_tip
        origin = center + s * view_dir
        normals = np.concatenate([box_normals, view_dir[None]], axis=0)
        offsets = np.concatenate([box_offsets, [origin @ view_dir]], axis=0)
        for name, mesh, color in [('hand', hand_mesh, hand_color), ('obj', obj_mesh, obj_color)]:
            (v, f), _ = _split_mesh_by_halfspaces(mesh.vertices, mesh.faces, normals, offsets)
            _set(f'{name}_in', _o3d_mesh(v, f) if len(f) else None, _material(color))
            cap = _section_cap(mesh, origin, view_dir, sec_u, sec_v, scale)
            # Flat, slightly darker fill marks the cut surface
            _set(f'{name}_cap', _cap_mesh(*cap) if cap is not None else None, cap_material(color))
    _emit(n[6], _stage_section)

    # 8. Hold on the sectional view
    _emit(n[7], lambda t: None)

    writer.close()
    print(f"Saved animation to {out_path}")


def test_obj():
    obj = 'toothpaste'
    # msdf = np.load(f'data/preprocessed/grab_msdf/{obj}.npz')['msdf']
    obj_info = GRABDataset.load_mesh_info('data/grab', msdf_path='data/preprocessed/grab_msdf')[obj]
    obj_mesh = trimesh.Trimesh(obj_info['verts'], obj_info['faces'], process=False)
    msdf = obj_info['msdf']
    kernel_size = 7

    # Visualize the first point's local grid
    visualize_local_grid(msdf, kernel_size, point_idx=0, obj_mesh=obj_mesh)


@hydra.main(config_path="../config", config_name="mlcontact_gen")
def test_pointvae(cfg):
    dummy_contact = torch.randn(4, 256, 5, 8, 8, 8)
    dummy_obj_msdf = torch.randn(4, 256, 1, 8, 8, 8)
    dummy_msdf_center = torch.randn(4, 256, 3)
    model = MLCVAE(cfg)
    recon, mu, logvar = model(dummy_contact, dummy_obj_msdf, dummy_msdf_center)


def compare_ckpt():
    ckpt1 = torch.load('logs/wandb_logs/LG3DContact/GRIDAE-128v1/checkpoints/last.ckpt')['state_dict']['handcse.embedding_tensor']
    ckpt2 = torch.load('common/model/hand_cse/hand_cse_4.ckpt')['state_dict']
    ckpt2 = ckpt2['embedding_tensor']
    print(ckpt1)
    print(ckpt2)


def test_manolayer():
    mano_layer = ManoLayer(mano_root = 'data/misc/mano_v1_2/models', use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    thetas = torch.randn(1, 48)
    betas = torch.randn(1, 10)
    trans = torch.randn(1, 3)
    faces = mano_layer.th_faces.detach().cpu().numpy()
    _, cano_joints, _ = mano_layer(torch.zeros_like(thetas), th_betas=betas)
    verts, joints, _ = mano_layer(thetas, th_betas=betas, th_trans=trans)
    root_j = cano_joints[:, 0]
    targetR = axis_angle_to_matrix(torch.randn(1, 3))
    targett = torch.randn(1, 3)
    verts = verts @ targetR.transpose(-1, -2) + targett.unsqueeze(1)
    root_rot = axis_angle_to_matrix(thetas[:, :3])
    new_root_rot = matrix_to_axis_angle(targetR @ root_rot)
    thetas[:, :3] = new_root_rot
    new_trans = (targetR @ trans.unsqueeze(-1) + targetR @ root_j.unsqueeze(-1)).squeeze(-1) + targett - root_j
    new_verts, new_joints, _ = mano_layer(thetas, th_betas=betas, th_trans=new_trans)

    ori_mesh = o3dmesh(verts[0].cpu().numpy(), faces, color=[0, 0, 1]) # blue
    new_mesh = o3dmesh(new_verts[0].cpu().numpy(), faces, color=[1, 0, 0]) # red
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([ori_mesh, new_mesh, coord_frame])


# Usage Example
def test_hoi4d_datamodule():
    from omegaconf import OmegaConf

    # Example config
    cfg = OmegaConf.create({
        'data': {
            'dataset_path': 'data/HOI4D',
            'preprocessed_dir': 'data/preprocessed',
            'release_file': 'release.txt',
            'num_workers': 8,
        },
        'train': {'batch_size': 32},
        'val': {'batch_size': 32},
        'test': {'batch_size': 32},
    })

    # Create datamodule
    dm = HOI4DHandDataModule(cfg)

    # Run preprocessing
    dm.prepare_data()

    # Setup for training
    dm.setup('fit')

    print(f"Train set size: {len(dm.train_set)}")
    print(f"Val set size: {len(dm.val_set)}")

    # Load a sample
    sample = dm.train_set[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Theta shape: {sample['theta'].shape}")
    print(f"Beta shape: {sample['beta'].shape}")
    print(f"Trans shape: {sample['trans'].shape}")
    print(f"Side: {'right' if sample['side'] == 1 else 'left'}")


@hydra.main(config_path="../config", config_name="handvae")
def fit_mano_beta(cfg):
    mano_layer = ManoLayer(mano_root = cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    train_set = GRABDataset(cfg.data, 'train', load_msdf=cfg.load_msdf, load_grid_contact=cfg.load_grid_contact)


def clip_mesh_to_aabb(verts: np.ndarray, faces: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray):
    """
    Clip a triangle mesh to an axis-aligned bounding box using Sutherland-Hodgman.
    Triangles straddling the boundary are split; only the inside portions are kept.
    Returns (clipped_verts, clipped_faces) with remapped indices.
    """
    def clip_poly_by_plane(poly, axis, sign, bound):
        # sign=+1: keep where pts[:,axis] <= bound  (max plane)
        # sign=-1: keep where pts[:,axis] >= bound  (min plane)
        # i.e. inside condition: sign * pts[:,axis] <= sign * bound
        if len(poly) == 0:
            return poly
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            a_in = sign * a[axis] <= sign * bound
            b_in = sign * b[axis] <= sign * bound
            if a_in:
                out.append(a)
            if a_in != b_in:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
        return out

    new_verts = []
    new_faces = []

    for tri_idx in faces:
        poly = [verts[i].copy() for i in tri_idx]

        # Clip against each of the 6 AABB planes
        for axis in range(3):
            poly = clip_poly_by_plane(poly, axis, +1, aabb_max[axis])  # upper bound
            if not poly:
                break
            poly = clip_poly_by_plane(poly, axis, -1, aabb_min[axis])  # lower bound
            if not poly:
                break

        if len(poly) < 3:
            continue

        # Triangulate the clipped polygon (fan triangulation)
        base_idx = len(new_verts)
        new_verts.extend(poly)
        for k in range(1, len(poly) - 1):
            new_faces.append([base_idx, base_idx + k, base_idx + k + 1])

    if not new_verts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    return np.array(new_verts, dtype=np.float32), np.array(new_faces, dtype=np.int64)


def pts_to_contact(pts: np.array, mesh: trimesh.Trimesh, dist_to_contact_fn: GridDistanceToContact):
    """
    pts: (N, 3) in world coordinates
    obj_mesh: trimesh of the object
    dist_to_contact_fn: function that takes points and returns distance to contact
    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, pts)  # (N,) unsigned distances
    dist_t = torch.as_tensor(dist, dtype=torch.float32)
    # contact = dist_to_contact_fn(dist_t).detach().cpu().numpy()                     # (N,) contact values in [0, 1]
    contact = dist_t.detach().cpu().numpy()                     # (N,) contact values in [0, 1]
    return contact


def vis_part_contact():
    """
    For teaser visualization test
    """
    mano_layer = ManoLayer(mano_root = "data/misc/mano_v1_2/models", use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    hand_part_ids = torch.argmax(mano_layer.th_weights, dim=-1).detach().cpu().numpy()
    canoV, canoJ, _ = mano_layer(torch.zeros(1, 48))
    canoV = canoV[0].detach().cpu().numpy()
    dist_to_contact2d = GridDistanceToContact(kernel_size=8, scale=0.01, method=2)
    dist_to_contact3d = GridDistanceToContact(kernel_size=4, scale=0.01, method=2)

    # Build submesh for the fingertip part
    tip_mask = (hand_part_ids == 3) | (hand_part_ids == 2)                             # (778,) bool
    tip_verts = canoV[tip_mask] # N x 3
    tip_vert_indices = np.where(tip_mask)[0]                  # original vertex indices
    faces = mano_layer.th_faces.cpu().numpy()                 # (F, 3)
    tip_face_mask = tip_mask[faces].all(axis=-1)              # keep faces where all 3 verts are in part
    tip_faces = faces[tip_face_mask]                          # (F', 3) in original index space
    # Remap to local index space
    remap = np.full(778, -1, dtype=np.int64)
    remap[tip_vert_indices] = np.arange(len(tip_vert_indices))
    tip_faces_local = remap[tip_faces]                        # (F', 3) in [0, N)

    # Rotate -30 degrees around Y axis
    angle = np.radians(-30)
    Rx = np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1, 0]])
    Ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                   [0,             1, 0            ],
                   [-np.sin(angle),0, np.cos(angle)]])
    tip_verts = (Ry @ Rx @ tip_verts.T).T

    # Translate so the lowest-z vertex aligns with the origin
    tip_verts -= tip_verts[np.argmin(tip_verts[:, 2])]
    tip_verts[:, 2] -= 0.003

    tip_mesh = trimesh.Trimesh(vertices=tip_verts, faces=tip_faces_local)

    ## points to calculate contact
    reso_2d = 8
    reso_3d = 4
    xy = np.stack(np.meshgrid(np.linspace(-0.01, 0.01, reso_2d), np.linspace(-0.01, 0.01, reso_2d)), axis=-1).reshape(-1, 2)  # (N, 2)
    pts2d = np.concatenate([xy, np.zeros((reso_2d**2, 1))], axis=-1)  # (N, 3)
    pts3d = np.stack(np.meshgrid(np.linspace(-0.01, 0.01, reso_3d), np.linspace(-0.01, 0.01, reso_3d), np.linspace(-0.01, 0.01, reso_3d)), axis=-1).reshape(-1, 3)  # (M, 3)

    hm_cmap = plt.colormaps['inferno']
    gt_contact2d = pts_to_contact(pts2d, tip_mesh, dist_to_contact2d)
    gt_contact3d = pts_to_contact(pts3d, tip_mesh, dist_to_contact3d)

    # tx, tz = np.meshgrid(np.linspace(-0.01, 0.01, 16), np.linspace(-0.01, 0.01, 16))
    # trans = np.stack([tx.flatten(), np.zeros_like(tx.flatten()), tz.flatten()], axis=-1)  # (256, 3)
    # ts = np.linspace(0, 0.01, 16)
    # rots = np.linspace(0, np.pi, 16)

    # cache_path = 'tmp/vis_part_contact_error_grids.npz'
    # if os.path.exists(cache_path):
    #     print(f"Loading cached error grids from {cache_path}")
    #     cache = np.load(cache_path)
    #     error2d_grid = cache['error2d_grid']
    #     error3d_grid = cache['error3d_grid']
    # else:
    #     all_axes = np.concatenate((np.eye(3), -np.eye(3)))
    #     error2d_grid = np.zeros((16, 16))
    #     error3d_grid = np.zeros((16, 16))
    #     for i, j in tqdm([(i, j) for i in range(16) for j in range(16)], total=16*16):
    #         for axx in all_axes:
    #             R = axis_angle_to_matrix(torch.from_numpy(axx * rots[i])).numpy()
    #             rotated_verts = (R @ tip_verts.T).T
    #             for axx in all_axes:
    #                 t = axx * ts[j]
    #                 transformed_verts = rotated_verts + t
    #                 transformed_mesh = trimesh.Trimesh(vertices=transformed_verts, faces=tip_faces_local)
    #                 contact2d = pts_to_contact(pts2d, transformed_mesh, dist_to_contact2d)
    #                 error2d = np.mean(np.abs(contact2d - gt_contact2d))
    #                 error2d_grid[i, j] += error2d
    #                 contact3d = pts_to_contact(pts3d, transformed_mesh, dist_to_contact3d)
    #                 error3d = np.mean(np.abs(contact3d - gt_contact3d))
    #                 error3d_grid[i, j] += error3d
    #         error2d_grid[i, j] /= len(all_axes) ** 2
    #         error3d_grid[i, j] /= len(all_axes) ** 2
    #     os.makedirs('tmp', exist_ok=True)
    #     np.savez(cache_path, error2d_grid=error2d_grid, error3d_grid=error3d_grid)
    #     print(f"Saved error grids to {cache_path}")

    # use_3d_plot = True
    # if use_3d_plot:
    #     rot_grid, t_grid = np.meshgrid(rots, ts * 1000)  # t_grid in mm

    #     from matplotlib.colors import LinearSegmentedColormap
    #     cmap_2d = LinearSegmentedColormap.from_list('blue_solid', ['#2A5E8C', '#2A5E8C'])
    #     cmap_3d = LinearSegmentedColormap.from_list('red_solid', ['#D9564A', '#D9564A'])

    #     fig = plt.figure(figsize=(9, 6))
    #     ax = fig.add_subplot(1, 1, 1, projection='3d')
    #     ax.plot_surface(rot_grid, t_grid, error2d_grid * 1000, cmap=cmap_2d, edgecolor='#2A5E8C', linewidth=0.3, alpha=0.7)
    #     ax.plot_surface(rot_grid, t_grid, error3d_grid * 1000, cmap=cmap_3d, edgecolor='#D9564A', linewidth=0.3, alpha=0.7)
    #     ax.view_init(elev=20, azim=140, roll=0)
    #     ax.xaxis.pane.fill = False
    #     ax.yaxis.pane.fill = False
    #     ax.zaxis.pane.fill = False
    #     ax.xaxis.pane.set_edgecolor('none')
    #     ax.yaxis.pane.set_edgecolor('none')
    #     ax.zaxis.pane.set_edgecolor('none')
    #     ax.grid(False)
    #     ax.set_xlabel('Rotation error (rad)')
    #     ax.set_ylabel('Translation error (mm)')
    #     ax.set_zlabel('Avg. distance error (mm)')
    #     ax.set_title('Contact map error: 2D vs 3D')
    #     from matplotlib.patches import Patch
    #     legend_handles = [Patch(facecolor='#2A5E8C', label='2D'), Patch(facecolor='#D9564A', label='3D')]
    #     ax.legend(handles=legend_handles)
    # else:
    #     rot_ticks = [f'{r:.1f}' for r in rots[::3]]
    #     t_ticks = [f'{t*1000:.1f}' for t in ts[::3]]

    #     vmin = min(error2d_grid.min(), error3d_grid.min()) * 1000
    #     vmax = max(error2d_grid.max(), error3d_grid.max()) * 1000

    #     fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    #     im0 = axes[0].imshow(error2d_grid * 1000, origin='lower', aspect='auto',
    #                          cmap='Blues', vmin=vmin, vmax=vmax,
    #                          extent=[rots[0], rots[-1], ts[0]*1000, ts[-1]*1000])
    #     axes[0].set_xlabel('Rotation error (rad)')
    #     axes[0].set_ylabel('Translation error (mm)')
    #     axes[0].set_title('2D contact map')
    #     fig.colorbar(im0, ax=axes[0], label='Avg. distance error (mm)')

    #     im1 = axes[1].imshow(error3d_grid * 1000, origin='lower', aspect='auto',
    #                          cmap='Reds', vmin=vmin, vmax=vmax,
    #                          extent=[rots[0], rots[-1], ts[0]*1000, ts[-1]*1000])
    #     axes[1].set_xlabel('Rotation error (rad)')
    #     axes[1].set_title('3D contact map')
    #     fig.colorbar(im1, ax=axes[1], label='Avg. distance error (mm)')

    # plt.tight_layout()
    # plt.show()

    o3d_pts2d = o3d.geometry.PointCloud()
    o3d_pts2d.points = o3d.utility.Vector3dVector(pts2d)
    o3d_pts2d.colors = o3d.utility.Vector3dVector(hm_cmap(dist_to_contact3d(gt_contact2d))[:,:3])  # color by contact value

    o3d_pts3d = o3d.geometry.PointCloud()
    o3d_pts3d.points = o3d.utility.Vector3dVector(pts3d)
    o3d_pts3d.colors = o3d.utility.Vector3dVector(hm_cmap(dist_to_contact3d(gt_contact3d))[:,:3])  # color by contact value

    tip_o3dmesh = o3dmesh(tip_verts, tip_faces_local, color="#F29A8D")
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    obj_mesh = o3d.geometry.TriangleMesh.create_box(0.02, 0.02, 0.01)
    obj_mesh.paint_uniform_color(parse_hex_color("#6293A8"))
    obj_mesh.compute_vertex_normals()
    obj_mesh.translate([-0.01, -0.01, -0.01])

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([-0.01, -0.01, -0.01]),
        max_bound=np.array([ 0.01,  0.01,  0.01]),
    )

    bbox.color = parse_hex_color("#102540")
    # clipped_verts, clipped_faces = clip_mesh_to_aabb(
    #     tip_verts, tip_faces_local,
    #     np.array([-0.01, -0.01, -0.01]),
    #     np.array([ 0.01,  0.01,  0.01]),
    # )
    # tip_o3dmesh_cropped = o3dmesh(clipped_verts, clipped_faces, color="#F27141")

    import math
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Fingertip Part Contact Visualization", width=640, height=480)
    for g in [tip_o3dmesh, obj_mesh, o3d_pts3d, bbox]:
        vis.add_geometry(g)
    # Camera: on -Y side, looking toward +Y with 30° downward pitch.
    d = 0.04  # scene is in [-0.01, 0.01] range
    eye = np.array([0.0, -d, d * math.tan(math.radians(30))])
    gaze = -eye / np.linalg.norm(eye)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(gaze, world_up); right /= np.linalg.norm(right)
    up = np.cross(right, gaze)
    R = np.stack([right, -up, gaze], axis=0)
    t = -R @ eye
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3,  3] = t
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    cam.extrinsic = extrinsic
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    vis.run()
    vis.destroy_window()


@hydra.main(config_path="../config", config_name="mlcdiff", version_base=None)
def test_ho3d_dataloader(cfg):
    """
    Test HO3DDataset loading via HOIDatasetModule on the test split.
    Runs prepare_data and setup('test'), then visualizes each loaded object mesh in 3D.
    """
    from common.dataset_utils.datamodules import HOIDatasetModule

    dm = HOIDatasetModule(cfg)

    print("Running prepare_data (precomputes MSDF if not cached)...")
    dm.prepare_data()

    print("Setting up test dataset...")
    dm.setup('test')

    test_set = dm.test_set
    print(f"Test dataset size: {len(test_set)} objects")
    print(f"Test objects: {test_set.test_objects}")

    loader = dm.test_dataloader()

    for batch_idx, batch in enumerate(loader):
        obj_name = batch['objName'][0]
        obj_com = batch['objCoM'][0].numpy()
        obj_mass = batch['objMass'][0].item()

        print(f"\n--- Object {batch_idx}: {obj_name} ---")
        print(f"  CoM: {obj_com}")
        print(f"  Mass: {obj_mass:.4f}")
        if 'objMsdf' in batch:
            msdf = batch['objMsdf'][0]
            print(f"  MSDF shape: {tuple(msdf.shape)}")
            print(f"  MSDF value range: [{msdf[:, :-3].min():.4f}, {msdf[:, :-3].max():.4f}]")

        # Build Open3D mesh from obj_info
        verts = test_set.obj_info[obj_name]['verts']
        faces = test_set.obj_info[obj_name]['faces']

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.6, 0.8, 1.0])

        # Mark CoM as a small sphere
        com_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        com_sphere.translate(obj_com)
        com_sphere.paint_uniform_color([1.0, 0.3, 0.3])
        com_sphere.compute_vertex_normals()

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        o3d.visualization.draw_geometries(
            [mesh, com_sphere, coord_frame],
            window_name=f"{obj_name}  (mass={obj_mass:.3f})"
        )


def _plot_error_curve(ax, sigmas, errors, gt_errors, ylabel, title):
    """
    Plot mean ± std of errors over sigmas, with a flat GT baseline.

    :param ax: matplotlib Axes
    :param sigmas: (11,) sigma values for x-axis
    :param errors: (N_samples, 11) predicted errors per sigma
    :param gt_errors: (N_samples,) ground-truth errors (sigma-independent)
    :param ylabel: y-axis label string
    :param title: plot title string
    """
    mean = errors.mean(axis=0)          # (11,)
    std  = errors.std(axis=0)           # (11,)
    gt_mean = gt_errors.mean()
    gt_std  = gt_errors.std()

    ax.errorbar(sigmas, mean, yerr=std, marker='o', capsize=4, label='Predicted')
    # GT is sigma-independent: draw as a horizontal band
    ax.axhline(gt_mean, color='C1', linestyle='--', label='GT')
    ax.axhspan(gt_mean - gt_std, gt_mean + gt_std, alpha=0.15, color='C1')

    ax.set_xlabel('Sigma')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def draw_error_plot():
    data = np.load('tmp/errorlists/sigma_test_results.npz')
    rec_errors    = data['rec_errors']     # (N_samples, 11)
    pene          = data['pene']           # (N_samples, 11)
    gt_rec_errors = data['gt_rec_errors']  # (N_samples,)
    gt_pene       = data['gt_pene']        # (N_samples,)
    sigmas        = data['sigmas']         # (11,)

    # Double-column A4 column width ≈ 88 mm = 3.46 in; 3:2 landscape → height 2.31 in.
    # Font sizes tuned so rendered text is ~11 pt at column width in the paper.
    FIG_W, FIG_H = 3.46, 3.46 * 2 / 3
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _plot_error_curve(ax, sigmas, rec_errors, gt_rec_errors,
                      ylabel='Reconstruction Error (mm)', title='Rec Error vs Sigma')
    fig.tight_layout()
    fig.savefig('tmp/errorlists/rec_error.svg', format='svg')
    # plt.show()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _plot_error_curve(ax, sigmas, pene * 1000, gt_pene * 1000,
                      ylabel='Penetration (mm)', title='Penetration vs Sigma')
    fig.tight_layout()
    fig.savefig('tmp/errorlists/penetration.svg', format='svg')
    # plt.show()



def crop_predicted_images(src, dst):
    """
    Walk src for all predicted.png files, crop the upper half, and save to dst.
    Each output is named after its immediate parent subfolder, e.g. batch_0000_wineglass.png.
    """
    from PIL import Image
    import glob

    os.makedirs(dst, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(src, "**", "predicted.png"), recursive=True))
    print(f"Found {len(paths)} predicted.png files.")
    for path in paths:
        subfolder = os.path.basename(os.path.dirname(path))
        img = Image.open(path)
        w, h = img.size
        cropped = img.crop((0, 0, w, h * 6 // 20))
        out_path = os.path.join(dst, f"{subfolder}.png")
        cropped.save(out_path)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    # test_ho3d_dataloader()
    # vis_msdf_data_sample()
    # test_obj()
    vis_local_grid_interact()
    # test_pointvae()
    # compare_ckpt()
    # test_hoi4d_datamodule()
    # test_manolayer()
    # vis_part_contact()
    # draw_error_plot()
    crop_predicted_images(
        # src="logs/wandb_logs/wandb/run-20260304_193313-u16fpzdr/files/test_images",
        # dst="tmp/grab",
        src="logs/wandb_logs/wandb/run-20260304_154407-m6y2sopw/files/test_images",
        dst="tmp/ho3d",
    )
