import os
import torch
from torch.utils.data import DataLoader
import trimesh
import hydra
import open3d as o3d
from matplotlib import pyplot as plt
from common.manopth.manopth.manolayer import ManoLayer
from common.dataset_utils.grab_dataset import GRABDataset
from common.dataset_utils.datamodules import HOIDatasetModule, LocalGridDataModule
from common.utils.vis import visualize_local_grid, visualize_local_grid_with_hand, o3dmesh, parse_hex_color
from common.msdf.utils.msdf import get_grid
from common.dataset_utils.hoi4d_dataset import HOI4DHandDataModule
from common.utils.geometry import GridDistanceToContact
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from common.model.handobject import HandObject
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

    train_loader = DataLoader(dm.train_set, batch_size=1, shuffle=False,
                              num_workers=4, collate_fn=dm.collate_fn)
    print(f"Train dataset size: {len(dm.train_set)}")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Batch size: {dm.train_batch_size}")

    obj_info = dm.train_set.obj_info

    print("\nVisualizing contact grids with hand...")
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches")):
        # if 'train' not in batch['objName'][0] and batch_idx % 20 != 0:
        #     continue  # Only visualize cube samples for now
        if batch_idx % 100 != 0:
            continue  # Only visualize every 20th batch to reduce load
        handobject = HandObject(cfg.data, device='cpu', mano_layer=mano_layer)
        obj_templates = [trimesh.Trimesh(obj_info[name]['verts'], obj_info[name]['faces'], process=False)
                         for name in batch['objName']]
        handobject.load_from_batch(batch, obj_templates=obj_templates, vis_obj_template=obj_templates)
        index_verts = handobject.hand_verts[:, handobject.hand_part_ids == 3]
        index_centre = index_verts.mean(dim=1)
        grid_dist = torch.norm(handobject.obj_msdf[:, :, -3:] - index_centre[:, None, :], dim=-1)  # (B, N) distance from each grid center to index fingertip center
        grid_indices = torch.argmin(grid_dist, dim=-1)  # (B,) indices of closest grid to index fingertip

        batch_size = handobject.batch_size
        n_grids = handobject.obj_msdf.shape[1]
        for b in range(batch_size):
            obj_name = batch['objName'][b]
            print(f"  Batch {batch_idx}, sample {b} ({obj_name}), {n_grids} grids")
            grid_idx = grid_indices[b].item()

            geoms = handobject.vis_all_grids_with_hand(obj_templates=obj_templates, idx=0, grid_idx=grid_idx)
            o3d.visualization.draw_geometries(geoms)

            # left_geoms, right_geoms = handobject.vis_grid_detail(obj_templates, idx=b, pt_idx=grid_idx)
            # o3d.visualization.draw_geometries(
            #     left_geoms,
            #     window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | Grid {grid_idx} — SDF")
            # o3d.visualization.draw_geometries(
            #     right_geoms,
            #     window_name=f"Batch {batch_idx} | Sample {b} ({obj_name}) | Grid {grid_idx} — Contact")


@hydra.main(config_path="../config", config_name="gridae")
def vis_local_grid_interact(cfg):
    dm = LocalGridDataModule(cfg)
    mano_layer = ManoLayer(mano_root = cfg.data.mano_root, use_pca=False, side='right', flat_hand_mean=True, ncomps=45)
    hand_faces = mano_layer.th_faces.numpy()
    dm.prepare_data()
    phase = 'test'
    dm.setup(phase)
    # train_loader = dm.train_dataloader()
    if phase == 'validate' or phase == 'train':
        loader = dm.val_dataloader()
        dataset = dm.val_set
    else:
        loader = dm.test_dataloader()
        dataset = dm.test_set
    # test_loader = dm.test_dataloader()
    print(f"{phase} dataset size: {len(dataset)}")
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Visualizing {phase} data"):
        grid_data = torch.cat([batch['gridSDF'], batch['gridContact'], batch['gridHandCSE']], dim=-1)  # (B, K, K, K, C)
        batch_size = grid_data.shape[0]
        for b in range(10):
            # break
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
            geoms = visualize_local_grid_with_hand(local_grid, hand_verts, hand_faces, dataset.hand_cse, cfg.msdf.kernel_size,
                                           cfg.msdf.scale, contact_point=contact_point, obj_mesh=obj_mesh)
            o3d.visualization.draw_geometries(geoms)
            # break
        # Only visualize first batch
        # break


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
    ts = np.linspace(0, 0.01, 16)
    rots = np.linspace(0, np.pi, 16)

    cache_path = 'tmp/vis_part_contact_error_grids.npz'
    if os.path.exists(cache_path):
        print(f"Loading cached error grids from {cache_path}")
        cache = np.load(cache_path)
        error2d_grid = cache['error2d_grid']
        error3d_grid = cache['error3d_grid']
    else:
        all_axes = np.concatenate((np.eye(3), -np.eye(3)))
        error2d_grid = np.zeros((16, 16))
        error3d_grid = np.zeros((16, 16))
        for i, j in tqdm([(i, j) for i in range(16) for j in range(16)], total=16*16):
            for axx in all_axes:
                R = axis_angle_to_matrix(torch.from_numpy(axx * rots[i])).numpy()
                rotated_verts = (R @ tip_verts.T).T
                for axx in all_axes:
                    t = axx * ts[j]
                    transformed_verts = rotated_verts + t
                    transformed_mesh = trimesh.Trimesh(vertices=transformed_verts, faces=tip_faces_local)
                    contact2d = pts_to_contact(pts2d, transformed_mesh, dist_to_contact2d)
                    error2d = np.mean(np.abs(contact2d - gt_contact2d))
                    error2d_grid[i, j] += error2d
                    contact3d = pts_to_contact(pts3d, transformed_mesh, dist_to_contact3d)
                    error3d = np.mean(np.abs(contact3d - gt_contact3d))
                    error3d_grid[i, j] += error3d
            error2d_grid[i, j] /= len(all_axes) ** 2
            error3d_grid[i, j] /= len(all_axes) ** 2
        os.makedirs('tmp', exist_ok=True)
        np.savez(cache_path, error2d_grid=error2d_grid, error3d_grid=error3d_grid)
        print(f"Saved error grids to {cache_path}")

    use_3d_plot = True
    if use_3d_plot:
        rot_grid, t_grid = np.meshgrid(rots, ts * 1000)  # t_grid in mm

        from matplotlib.colors import LinearSegmentedColormap
        cmap_2d = LinearSegmentedColormap.from_list('blue_solid', ['#2A5E8C', '#2A5E8C'])
        cmap_3d = LinearSegmentedColormap.from_list('red_solid', ['#D9564A', '#D9564A'])

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(rot_grid, t_grid, error2d_grid * 1000, cmap=cmap_2d, edgecolor='#2A5E8C', linewidth=0.3, alpha=0.7)
        ax.plot_surface(rot_grid, t_grid, error3d_grid * 1000, cmap=cmap_3d, edgecolor='#D9564A', linewidth=0.3, alpha=0.7)
        ax.view_init(elev=20, azim=140, roll=0)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)
        ax.set_xlabel('Rotation error (rad)')
        ax.set_ylabel('Translation error (mm)')
        ax.set_zlabel('Avg. distance error (mm)')
        ax.set_title('Contact map error: 2D vs 3D')
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor='#2A5E8C', label='2D'), Patch(facecolor='#D9564A', label='3D')]
        ax.legend(handles=legend_handles)
    else:
        rot_ticks = [f'{r:.1f}' for r in rots[::3]]
        t_ticks = [f'{t*1000:.1f}' for t in ts[::3]]

        vmin = min(error2d_grid.min(), error3d_grid.min()) * 1000
        vmax = max(error2d_grid.max(), error3d_grid.max()) * 1000

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        im0 = axes[0].imshow(error2d_grid * 1000, origin='lower', aspect='auto',
                             cmap='Blues', vmin=vmin, vmax=vmax,
                             extent=[rots[0], rots[-1], ts[0]*1000, ts[-1]*1000])
        axes[0].set_xlabel('Rotation error (rad)')
        axes[0].set_ylabel('Translation error (mm)')
        axes[0].set_title('2D contact map')
        fig.colorbar(im0, ax=axes[0], label='Avg. distance error (mm)')

        im1 = axes[1].imshow(error3d_grid * 1000, origin='lower', aspect='auto',
                             cmap='Reds', vmin=vmin, vmax=vmax,
                             extent=[rots[0], rots[-1], ts[0]*1000, ts[-1]*1000])
        axes[1].set_xlabel('Rotation error (rad)')
        axes[1].set_title('3D contact map')
        fig.colorbar(im1, ax=axes[1], label='Avg. distance error (mm)')

    # plt.tight_layout()
    # plt.show()

    o3d_pts2d = o3d.geometry.PointCloud()
    o3d_pts2d.points = o3d.utility.Vector3dVector(pts2d)
    o3d_pts2d.colors = o3d.utility.Vector3dVector(hm_cmap(dist_to_contact3d(gt_contact2d))[:,:3])  # color by contact value

    o3d_pts3d = o3d.geometry.PointCloud()
    o3d_pts3d.points = o3d.utility.Vector3dVector(pts3d)
    o3d_pts3d.colors = o3d.utility.Vector3dVector(hm_cmap(dist_to_contact3d(gt_contact3d))[:,:3])  # color by contact value

    tip_o3dmesh = o3dmesh(tip_verts, tip_faces_local, color="#F27141")
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    obj_mesh = o3d.geometry.TriangleMesh.create_box(0.02, 0.02, 0.01)
    obj_mesh.paint_uniform_color(parse_hex_color("#2A5E8C"))
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

    o3d.visualization.draw_geometries([tip_o3dmesh, obj_mesh, o3d_pts2d, bbox],
                                      window_name="Fingertip Part Contact Visualization")


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


if __name__ == "__main__":
    test_ho3d_dataloader()
    # vis_msdf_data_sample()
    # test_obj()
    # vis_local_grid_interact()
    # test_pointvae()
    # compare_ckpt()
    # test_hoi4d_datamodule()
    # test_manolayer()
    # vis_part_contact()
