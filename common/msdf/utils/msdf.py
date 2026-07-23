from itertools import product

import numpy as np
from tqdm import tqdm
import torch
import trimesh
from pysdf import SDF
from skimage.measure import marching_cubes
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces

def get_grid(kernel_size, device='cpu') -> torch.Tensor:
    """
    Get the cubic interpolant for the given kernel size in range [-1, 1].
    """
    interpolant = torch.zeros(kernel_size, kernel_size, kernel_size, 3, device=device)
    for i, j, k in product(range(kernel_size), repeat=3):
        coord = torch.tensor([i, j, k], device=device)
        val = 2 * coord - (kernel_size - 1)
        val = val / (kernel_size - 1)
        interpolant[i, j, k, :] = val
    return interpolant


def calculate_weights(X, centers, scales) -> torch.Tensor:
    """
    Calculate the weight for points X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    :return
    weights: torch.Tensor, (N, M), the weights for each point and each grid.
    """
    distances = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)
    max_norm = torch.norm(distances, p=torch.inf, dim=2)
    weights = torch.nn.functional.relu(1 - max_norm)  # N, M
    norm_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
    return norm_weights


def find_closest_vertices_indices(x, kernel_size):
    # this code was taken from the hashgrid implementation
    # https://github.com/Ending2015a/hash-grid-encoding/blob/master/encoding.py#L109
    dim = x.shape[-1]
    n_neigs = 2 ** dim
    neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
    dims = np.arange(dim, dtype=np.int64).reshape((1, -1))
    bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool).to(x.device)  # (neig, dim)

    bdims = len(x.shape[:-1])
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2 * (kernel_size - 1)
    xi = x.long()
    xi = torch.clamp(xi, 0, max=kernel_size - 2)
    xf = x - xi.float().detach()

    xi = xi.unsqueeze(dim=-2)  # (b..., 1, dim)
    xf = xf.unsqueeze(dim=-2)  # (b..., 1, dim)
    # to match the input batch shape
    bin_mask = bin_mask.reshape((1,) * bdims + bin_mask.shape)  # (1..., neig, dim)
    # get neighbors' indices and weights on each dim
    inds = torch.where(bin_mask, xi, xi + 1)  # (b..., neig, dim)
    ws = torch.where(bin_mask, 1 - xf, xf)  # (b...., neig, dim)
    # aggregate nehgibors' interp weights
    w = ws.prod(dim=-1, keepdim=True)  # (b..., neig, 1)
    return w, inds  # (b..., feat)


def get_values_at_indices(Vi: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Get the values from Vi at the given indices, optimized.
    :param Vi: matrix with values shape (M, k, k, k)
    :param indices: indices for Vi matrix shape (N, M, 8, 3)
    :return: values from Vi at indices shape (N, M, 8)
    """

    N, M, _, _ = indices.shape  # Extract dimensions
    m_range = torch.arange(M)[None, :, None].to(Vi.device)  # Shape: (1, M, 1) for broadcasting over M
    I, J, K = indices.to(Vi.device).unbind(-1)  # Each will have shape (N, M, 8)
    results = Vi[m_range, I, J, K]
    return results


def msdf_at_point_batched(X, centers, scales, Vi, kernel_size, batch_size: int = 4096 * 16) -> torch.Tensor:
    result = []
    for i in tqdm(range(0, X.size(0), batch_size)):
        range_end = min(i + batch_size, X.size(0))
        result.append(msdf_at_point(X[i:range_end], centers, scales, Vi, kernel_size))
    return torch.cat(result, dim=0)


def msdf_at_point(X, centers, scales, Vi, kernel_size) -> torch.Tensor:
    """
    Calculate the MSDF value at point X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    Vi: torch.Tensor, (M, k, k, k), the SDF values grid.
    kernel_size: int, the kernel size used for the grid.
    :return
    msdf: torch.Tensor, (N), the SDF value at the points.
    """

    centered = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)  # N, M, 3
    out_of_cube = torch.any(torch.abs(centered) > 1, dim=-1)
    out_of_cube = torch.all(out_of_cube, dim=-1)
    inside_cube = ~out_of_cube
    result = torch.zeros(X.size(0), device=X.device).float()
    if not torch.any(inside_cube):
        return result
    centered = centered[inside_cube]

    centered_flat = centered.view(-1, 3).float()  # N * M, 3
    weights, indices = find_closest_vertices_indices(centered_flat, kernel_size)
    indices = indices.view(centered.size(0), centers.size(0), 8, 3)
    weights = weights.view(centered.size(0), centers.size(0), 8)

    values_at_corners = get_values_at_indices(Vi, indices)
    weights = weights.to(values_at_corners.device)
    values = (values_at_corners * weights).sum(dim=-1)
    weights = calculate_weights(X[inside_cube], centers, scales)
    weighted_values = (values * weights).sum(dim=1).float()
    result[inside_cube] = weighted_values
    return result


def calculate_msdf_value(scale: float, points: torch.Tensor, mesh: trimesh.Trimesh, kernel_size) -> torch.Tensor:
    all_points = get_grid_points(points, scale, kernel_size)
    n_points, n_grids, _ = all_points.shape
    f = SDF(mesh.vertices, mesh.faces)
    all_points = all_points.detach().cpu().numpy()
    sdf_values = f(all_points.reshape(-1, 3)).reshape(n_points, n_grids)
    return -torch.tensor(sdf_values, device=points.device)


def sample_volume(bounds, resolution, bound_delata=0.1, device='cpu'):
    """
    Sample a 3D volume defined by 'bounds' at a given 'resolution'.
    bounds: (min, maxh)
    resolution: int, number of samples per dimension
    bound_delta: float, the delta to add to the bounds to avoid sampling on the boundary.
    device: torch device to use for the tensor
    Returns a tensor of shape (N, 3) of sampled points.
    """
    delta = (bounds[1] - bounds[0]) * bound_delata
    grid_x = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    grid_y = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    grid_z = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    meshgrid = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    points = torch.stack(meshgrid, dim=-1).reshape(-1, 3).to(device)
    return points


def get_grid_points(centers, scales, kernel_size):
    """
    Get the grid points for the given centers and scales.
    :param centers: Nx3 tensor of centers
    :param scales: N tensor of scales
    :param kernel_size: kernel size used
    :return: NxK*K*Kx3 tensor of grid points
    """
    grid = get_grid(kernel_size=kernel_size, device=centers.device)
    grid_flat = grid.reshape(-1, 3).to(centers.device)
    if isinstance(scales, torch.Tensor):
        scales = scales.view(-1, 1, 1)
    all_points = centers.reshape(centers.shape[0], 1, 3) + grid_flat.view(1, -1, 3) * scales
    return all_points


def reconstruct_mesh(Vi, scales, centers, kernel_size, resolution=16, threshold=0.0) -> trimesh.Trimesh:
    grid = sample_volume((-1, 1), resolution, device=centers.device)
    Vi = Vi.view(Vi.size(0), kernel_size, kernel_size, kernel_size)

    grid_sdf = msdf_at_point_batched(grid, centers, scales, Vi, kernel_size)
    grid_sdf = grid_sdf.detach().cpu().numpy().reshape(resolution, resolution, resolution)

    verts, faces, _, _ = marching_cubes(-grid_sdf, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def reconstruct_from_sdf(sdf, resolution):
    grid = sample_volume((-1, 1), resolution)
    grid_sdf = sdf(grid)
    grid_sdf = grid_sdf.reshape(resolution, resolution, resolution)

    verts, faces, _, _ = marching_cubes(grid_sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def calculate_contact_mask(point_cloud, hand_verts, grid_scale, device):
    """
    Calculate contact mask for a given point cloud.

    Args:
        point_cloud: torch tensor of shape (B, N, 3)
        hand_verts: torch tensor of shape (B, H, 3)
        grid_scale: float, scale of half the grid
        device: torch device to use
    Returns:
        mask: torch tensor of shape (B, N) indicating contact points
    """
    # Convert inputs to torch tensors on device if needed
    if not isinstance(point_cloud, torch.Tensor):
        point_cloud = torch.from_numpy(point_cloud).float().to(device)
    else:
        point_cloud = point_cloud.to(device)

    if not isinstance(hand_verts, torch.Tensor):
        hand_verts = torch.from_numpy(hand_verts).float().to(device)
    else:
        hand_verts = hand_verts.to(device)

    ## Determine grid contact using Chebyshev distance
    # point_cloud: (B, N, 3), hand_verts: (B, H, 3)
    cbdist = torch.max(torch.abs(point_cloud[:, :, None, :] - hand_verts[:, None, :, :]), dim=-1)[0]  # (B, N, H)
    obj_pt_mask = torch.min(cbdist, dim=-1)[0] < grid_scale  # (B, N)
    hand_vert_mask = torch.min(cbdist, dim=1)[0] < grid_scale  # (B, H)

    return obj_pt_mask, hand_vert_mask


def calc_local_grid_batch(contact_points, normalized_coords, kernel_size, grid_scale, hand_verts, hand_cse, device):
    M = contact_points.shape[0]

    if M == 0:
        # No contact points for this sample
        local_grids = torch.empty(0, kernel_size, kernel_size, kernel_size,
                                                1 + hand_cse.shape[-1], device=device)

    else:
        # Scale and translate to world coordinates for each contact point
        # contact_points: (M, 3), normalized_coords: (kernel_size^3, 3)
        # Result: (M, kernel_size^3, 3)
        grid_points_flat = (contact_points[:, None, :] + normalized_coords[None, :, :] * grid_scale).reshape(-1, 3)

        # Calculate SDF values if obj_mesh is provided
        # if obj_mesh is not None:
        #     objSDF = SDF(obj_mesh.vertices, obj_mesh.faces)
        #     grid_sdfs_np = objSDF(grid_points_flat.cpu().numpy())
        #     grid_sdfs = torch.from_numpy(grid_sdfs_np).float().to(device).reshape(M, kernel_size, kernel_size, kernel_size, 1)

        # Calculate distance to nearest hand vertex for this sample
        dist_mat = torch.norm(hand_verts[None, :, :] - grid_points_flat[:, None, :], dim=-1)  # (M*kernel_size^3, H)
        nn_dist, nn_idx = torch.min(dist_mat, dim=1)  # (M * kernel_size^3)

        grid_distance = nn_dist.reshape(M, kernel_size, kernel_size, kernel_size, 1)
        grid_hand_cse = hand_cse[nn_idx].reshape(M, kernel_size, kernel_size, kernel_size, -1)

        # Concatenate features based on whether obj_mesh is provided
        # if obj_mesh is not None:
        #     local_grids = torch.cat([grid_sdfs, grid_distance, grid_hand_cse], dim=-1)  # (M, kernel_size, kernel_size, kernel_size, 1 + 1 + cse_dim)
        # else:
        local_grids = torch.cat([grid_distance, grid_hand_cse], dim=-1)  # (M, kernel_size, kernel_size, kernel_size, 1 + cse_dim)

    return local_grids


def calc_local_grid_1pt(normalized_coords, obj_mesh, hand_mesh, kernel_size, grid_scale, hand_cse):
    """
    Calculate local grid for a single contact point using numpy on CPU.

    Args:
        normalized_coords: numpy array of shape (kernel_size^3, 3), normalized grid coordinates
        obj_mesh: trimesh.Trimesh, object mesh
        hand_mesh: trimesh.Trimesh, the normalized hand mesh
        kernel_size: int, size of the cubic grid
        grid_scale: float, scale of the grid
        hand_cse: numpy array of shape (778, cse_dim), hand contact surface embeddings

    Returns:
        local_grid: numpy array of shape (kernel_size, kernel_size, kernel_size, 2 + cse_dim)
        verts_mask: numpy array of shape (n_verts), boolean mask indicating valid vertices
    """
    # Scale and translate to world coordinates
    # contact_point: (3,), normalized_coords: (kernel_size^3, 3)
    # Result: (kernel_size^3, 3)
    grid_points_flat = normalized_coords * grid_scale

    ## Determine mask using Chebyshev distance
    hand_verts = hand_mesh.vertices  # (778, 3)
    verts_mask = np.max(np.abs(hand_verts[:, :]), axis=-1) < grid_scale

    # Calculate SDF values
    objSDF = SDF(obj_mesh.vertices, obj_mesh.faces)
    grid_sdfs_np = objSDF(grid_points_flat)
    grid_sdfs = grid_sdfs_np.reshape(kernel_size, kernel_size, kernel_size, 1)

    # Calculate distance to nearest hand vertex
    # grid_points_flat: (kernel_size^3, 3), hand_verts: (778, 3)

    ## Return only the points
    # dist_mat = np.linalg.norm(grid_points_flat[:, None, :] - hand_verts[None, :, :], axis=-1)  # (kernel_size^3, 778)
    # nn_idx = np.argmin(dist_mat, axis=1)  # (kernel_size^3,)
    # nn_dist = np.min(dist_mat, axis=1)  # (kernel_size^3,)
    # grid_hand_cse = hand_cse[nn_idx].reshape(kernel_size, kernel_size, kernel_size, -1)
    ## Use mesh to calculate the nearest distance

    # Calculate nearest point on hand mesh surface
    nn_dist, nn_face_idx, nn_point = nn_dist_to_mesh(grid_points_flat, hand_mesh)

    # Map face indices to vertex indices for CSE lookup
    # For each face, find which of its 3 vertices is closest to the query point
    # closest_face_verts = hand_mesh.vertices[hand_mesh.faces[nn_face_idx]]  # (N, 3, 3)
    # vert_dists = np.linalg.norm(closest_face_verts - grid_points_flat[:, None, :], axis=2)  # (N, 3)
    # local_vert_idx = np.argmin(vert_dists, axis=1)  # (N,) values in [0, 1, 2]
    # nn_idx = hand_mesh.faces[nn_face_idx][np.arange(len(nn_face_idx)), local_vert_idx]
    face_vert_idx = hand_mesh.faces[nn_face_idx]  # (N, 3)
    face_verts = hand_mesh.vertices[face_vert_idx]  # (N, 3, 3)
    face_cse = hand_cse[face_vert_idx] # (N, 3, cse_dim)
    ## Calculate the CSE
    ## Use Barycentric weights
    # d = np.linalg.norm(nn_point[:, np.newaxis, :] - face_verts, axis=-1)  # (N, 3)
    # w = 1 / (d + 1e-8)
    # w = w / np.sum(w, axis=1, keepdims=True)  #

    ## Use linear invert
    # w = np.linalg.inv(face_verts.transpose((0, 2, 1))) @ nn_point[:, :, np.newaxis]  # (N, 3, 1)
    # ## make sure weights are all positive
    # w = np.clip(w, 0, 1)
    # w = w / np.sum(w, axis=1, keepdims=True)  # (N, 3, 1)

    # grid_hand_cse = np.sum(face_cse * w, axis=1).reshape(kernel_size, kernel_size, kernel_size, -1)  # (N, cse_dim)

    grid_distance = nn_dist.reshape(kernel_size, kernel_size, kernel_size, 1)

    ## Calculate the distance to the nearest surface

    local_grid = np.concatenate([grid_sdfs, grid_distance, grid_hand_cse], axis=-1)  # (kernel_size, kernel_size, kernel_size, 2 + cse_dim)

    return local_grid, verts_mask


def calc_local_grid_all_pts(contact_points, normalized_coords, obj_mesh, hand_mesh, kernel_size, grid_scale, hand_cse=None, device='cpu'):
    """
    Calculate local grids for all contact points using numpy on CPU.

    :param contact_points: N x 3, contact points in world coordinates
    :param normalized_coords: K^3 x 3, normalized grid coordinates
    :param obj_mesh: trimesh.Trimesh, object mesh
    :param hand_mesh: trimesh.Trimesh, the normalized hand mesh
    :param kernel_size: K, size of the cubic grid
    :param grid_scale: float, scale of the grid
    :param hand_verts: H x 3, hand vertices (should be hand_mesh.vertices)
    :param hand_cse: H x cse_dim, hand contact surface embeddings

    Returns:
        local_grids: numpy array of shape (N, kernel_size, kernel_size, kernel_size, 2 + cse_dim)
        verts_mask: numpy array of shape (N, n_verts), boolean mask indicating valid vertices for each contact point
    """
    N = contact_points.shape[0]

    # Handle empty case
    hand_verts = hand_mesh.vertices
    if N == 0:
        # cse_dim = hand_cse.shape[-1]
        return np.empty((0, kernel_size, kernel_size, kernel_size, 2)), np.empty((0, hand_verts.shape[0]), dtype=bool)

    # Scale and translate to world coordinates for all contact points
    # contact_points: (N, 3), normalized_coords: (K^3, 3)
    # Result: (N, K^3, 3)

    # Determine mask using Chebyshev distance for all contact points
    # For each contact point, check which hand vertices are within the grid
    # hand_verts centered at each contact point
    ho_dist = np.max(np.abs(hand_verts[None, :, :] - contact_points[:, None, :]), axis=-1) 
    verts_mask = ho_dist < grid_scale  # N x H
    grid_mask = np.any(verts_mask, axis=1) # N
    M = np.sum(grid_mask)

    # ho_dist = np.min(ho_dist, axis=1)  # N

    grid_points_flat = contact_points[:, None, :] + normalized_coords[None, :, :] * grid_scale  # N x K^3 x 3
    grid_points_all = grid_points_flat[grid_mask].reshape(-1, 3)  # (M * K^3, 3)

    # Calculate distance to nearest hand mesh surface for all grid points at once
    nn_dist, nn_face_idx, nn_point = nn_dist_to_mesh(grid_points_all, hand_mesh)

    # Map face indices to get CSE using barycentric interpolation
    # face_vert_idx = hand_mesh.faces[nn_face_idx]  # (M * K^3, 3)
    # face_verts = hand_mesh.vertices[face_vert_idx]  # (M * K^3, 3, 3)
    # face_cse = hand_cse[face_vert_idx]  # (M * K^3, 3, cse_dim)

    # Calculate barycentric weights using linear invert
    # w = np.linalg.inv(face_verts.transpose((0, 2, 1))) @ nn_point[:, :, np.newaxis]  # (M * K^3, 3, 1)
    # # Make sure weights are all positive and normalized
    # w = np.clip(w, 0, 1)
    # # w = w / np.sum(w, axis=1, keepdims=True)  # (M * K^3, 3, 1)

    # # Interpolate CSE values
    # grid_hand_cse = np.sum(face_cse * w, axis=1).reshape(M, kernel_size, kernel_size, kernel_size, -1)  # (M, K, K, K, cse_dim)

    # Reshape distances
    grid_distance = nn_dist.reshape(M, kernel_size, kernel_size, kernel_size, 1)  # (M, K, K, K, 1)
    nn_face_idx = nn_face_idx.reshape(M, kernel_size, kernel_size, kernel_size)  # (M, K, K, K)
    nn_point = nn_point.reshape(M, kernel_size, kernel_size, kernel_size, 3)  # (M, K, K, K, 3)

    # Concatenate all features
    # Calculate SDF values for all grid points at once (same obj_mesh for all points)
    if obj_mesh is not None:
        objSDF = SDF(obj_mesh.vertices, obj_mesh.faces)

        grid_sdfs_np = objSDF(grid_points_all)  # (M * K^3,)
        grid_sdfs = grid_sdfs_np.reshape(M, kernel_size, kernel_size, kernel_size, 1)  # (M, K, K, K, 1)
        local_grids = np.concatenate([grid_sdfs, grid_distance], axis=-1)  # (M, K, K, K, 2)
    else:
        local_grids = grid_distance  # (M, K, K, K, 1)

    return local_grids, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point


def nn_dist_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the nearest distance from points to mesh surface.

    For each query point, finds the closest point on the mesh surface and returns
    the distance, which face contains that point, and the coordinates of the
    closest point.

    Args:
        points: numpy array of shape (N, 3), points to calculate distances for
        mesh: trimesh.Trimesh, the mesh to calculate distances to

    Returns:
        distances: numpy array of shape (N,), unsigned distances from each point
                   to nearest point on mesh surface (always >= 0)
        face_indices: numpy array of shape (N,), triangle indices indicating which
                      face contains the nearest point (values in range [0, num_faces))
        closest_points: numpy array of shape (N, 3), coordinates of the nearest
                        points on the mesh surface
    """
    # Handle edge case: empty points array
    if len(points) == 0:
        return np.array([]), np.array([], dtype=np.int64), np.array([]).reshape(0, 3)

    # Use trimesh's optimized closest_point function
    # Returns: (closest_points, distances, triangle_ids)
    closest_points, distances, face_indices = trimesh.proximity.closest_point(mesh, points)

    # Ensure distances are unsigned (they should already be, but make it explicit)
    distances = np.abs(distances)

    return distances, face_indices, closest_points


def _project_points_to_triangles(points, tri_verts):
    """
    Project query points onto their closest triangles to get the nearest surface point.
    Implements the full 7-region Voronoi classification from Ericson's
    "Real-Time Collision Detection" Section 5.1.5 (ClosestPtPointTriangle).

    Args:
        points: (N, 3) query points
        tri_verts: (N, 3, 3) the 3 vertices of each point's closest triangle

    Returns:
        closest_points: (N, 3) nearest points on triangle surfaces
    """
    a = tri_verts[:, 0]  # (N, 3)
    b = tri_verts[:, 1]  # (N, 3)
    c = tri_verts[:, 2]  # (N, 3)
    p = points

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = (ab * ap).sum(dim=-1)  # (N,)
    d2 = (ac * ap).sum(dim=-1)

    bp = p - b
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)

    cp = p - c
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)

    # Start with all zeros, fill in region by region
    # Use barycentric coords: result = a*u + b*v + c*w, where u+v+w=1
    v_coord = torch.zeros_like(d1)
    w_coord = torch.zeros_like(d1)
    assigned = torch.zeros_like(d1, dtype=torch.bool)

    # Region 1: vertex A (d1 <= 0 and d2 <= 0)
    r1 = (d1 <= 0) & (d2 <= 0)
    # v=0, w=0 (already zero)
    assigned = assigned | r1

    # Region 2: vertex B (d3 >= 0 and d4 <= d3)
    r2 = ~assigned & (d3 >= 0) & (d4 <= d3)
    v_coord[r2] = 1.0
    assigned = assigned | r2

    # Region 3: vertex C (d6 >= 0 and d5 <= d6)
    r3 = ~assigned & (d6 >= 0) & (d5 <= d6)
    w_coord[r3] = 1.0
    assigned = assigned | r3

    # Region 4: edge AB (vc <= 0, d1 >= 0, d3 <= 0)
    vc = d1 * d4 - d3 * d2
    r4 = ~assigned & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / (d1 - d3).clamp(min=1e-12)
    v_coord[r4] = v_ab[r4]
    assigned = assigned | r4

    # Region 5: edge AC (vb <= 0, d2 >= 0, d6 <= 0)
    vb = d5 * d2 - d1 * d6
    r5 = ~assigned & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / (d2 - d6).clamp(min=1e-12)
    w_coord[r5] = w_ac[r5]
    assigned = assigned | r5

    # Region 6: edge BC (va <= 0, (d4-d3) >= 0, (d5-d6) >= 0)
    va = d3 * d6 - d5 * d4
    r6 = ~assigned & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=1e-12)
    v_coord[r6] = (1.0 - w_bc)[r6]
    w_coord[r6] = w_bc[r6]
    assigned = assigned | r6

    # Region 7: inside triangle
    r7 = ~assigned
    denom = (va + vb + vc).clamp(min=1e-12)
    v_coord[r7] = (vb / denom)[r7]
    w_coord[r7] = (vc / denom)[r7]

    u_coord = 1.0 - v_coord - w_coord
    closest_points = u_coord.unsqueeze(-1) * a + v_coord.unsqueeze(-1) * b + w_coord.unsqueeze(-1) * c
    return closest_points


def nn_dist_to_mesh_gpu(points, hand_verts, faces):
    """
    GPU-accelerated nearest distance from points to mesh surface using Kaolin.

    Args:
        points: torch.Tensor (N, 3), query points on GPU
        hand_verts: torch.Tensor (V, 3), mesh vertices on GPU
        faces: torch.LongTensor (F, 3), face indices on GPU

    Returns:
        distances: torch.Tensor (N,), unsigned distances
        face_indices: torch.LongTensor (N,), nearest face indices
        closest_points: torch.Tensor (N, 3), nearest points on mesh surface
    """
    # Kaolin's CUDA kernel dispatches on the points' dtype and requires the mesh
    # to match, so cast both to float32 (points may arrive as float64 from the
    # dataset). kaolin 0.18.0 handles float32 natively.
    points = points.float()
    hand_verts = hand_verts.float()

    # Kaolin expects batched input: (B, N, 3) and (B, F, 3, 3)
    face_verts = index_vertices_by_faces(
        hand_verts.unsqueeze(0), faces
    )  # (1, F, 3, 3)

    sq_dist, face_idx, _ = point_to_mesh_distance(
        points.unsqueeze(0), face_verts
    )  # (1, N), (1, N), (1, N)

    sq_dist = sq_dist.squeeze(0)      # (N,)
    face_idx = face_idx.squeeze(0)    # (N,)

    # Gather the 3 vertices of each closest triangle
    nearest_tri_verts = face_verts[0, face_idx]  # (N, 3, 3)

    # Project points onto their nearest triangles
    closest_points = _project_points_to_triangles(points, nearest_tri_verts)

    distances = torch.sqrt(sq_dist.clamp(min=0))

    return distances, face_idx, closest_points


def calc_local_grid_all_pts_gpu(contact_points, normalized_coords, hand_verts, faces, kernel_size, grid_scale,
                                apply_grid_mask=True, sample_rate=1, far_point_rate=0):
    """
    GPU-accelerated version of calc_local_grid_all_pts using Kaolin.

    All inputs and outputs are torch tensors on GPU.

    Args:
        contact_points: (N, 3) contact points in world coordinates
        normalized_coords: (K^3, 3) normalized grid coordinates
        hand_verts: (V, 3) hand mesh vertices
        faces: (F, 3) face indices (LongTensor)
        kernel_size: int, size of the cubic grid
        grid_scale: float, scale of the grid
        apply_grid_mask: bool, whether to skip grid calculations for contact points with no nearby hand verts
        sample_rate: float, fraction of grid points to sample for distance calculation (to save memory)
        far_point_rate: float, fraction of points to add to the mask as "far" points even if they have no nearby hand verts;
                        The fraction is relative to the number of "near" points that have nearby hand verts. This is to provide negative samples for training.

    Returns:
        grid_distance: (M, K, K, K, 1) distances for active grids
        verts_mask: (N, V) bool, which hand verts are within grid_scale of each contact point
        grid_mask: (N,) bool, which contact points have any nearby hand verts
        ho_dist: (N, V) Chebyshev distances
        nn_face_idx: (M, K, K, K) face indices for nearest triangles
        nn_point: (M, K, K, K, 3) nearest points on mesh surface
    """
    N = contact_points.shape[0]
    device = contact_points.device

    # Chebyshev distance masking
    ho_dist = torch.max(
        torch.abs(hand_verts[None, :, :] - contact_points[:, None, :]), dim=-1
    )[0]  # (N, V)
    verts_mask = ho_dist < grid_scale  # (N, V)

    if apply_grid_mask:
        # Initial grid mask based on nearby hand vertices
        grid_mask_initial = verts_mask.any(dim=1)  # (N,)
        n_near_points = grid_mask_initial.sum().item()

        if n_near_points == 0:
            K = kernel_size
            V = hand_verts.shape[0]
            return (
                torch.empty(0, K, K, K, 1, device=device),
                verts_mask,
                torch.zeros(N, dtype=torch.bool, device=device),
                ho_dist,
                torch.empty(0, K, K, K, dtype=torch.long, device=device),
                torch.empty(0, K, K, K, 3, device=device),
            )

        # Step 1: Calculate number of masked points after sampling
        n_sampled_near = int(n_near_points * sample_rate)

        # Step 2: Calculate number of far points to add
        n_far_points = int(n_sampled_near * far_point_rate)

        # Step 3: Sample points from the initial mask
        if sample_rate < 1.0 and n_sampled_near > 0:
            near_indices = torch.where(grid_mask_initial)[0]
            sampled_indices = near_indices[torch.randperm(n_near_points, device=device)[:n_sampled_near]]
            grid_mask = torch.zeros(N, dtype=torch.bool, device=device)
            grid_mask[sampled_indices] = True
        else:
            grid_mask = grid_mask_initial.clone()

        # Step 4: Add far points to the mask
        if far_point_rate > 0 and n_far_points > 0:
            far_candidates = torch.where(~grid_mask_initial)[0]
            n_available_far = far_candidates.shape[0]
            if n_available_far > 0:
                n_far_points = min(n_far_points, n_available_far)
                far_indices = far_candidates[torch.randperm(n_available_far, device=device)[:n_far_points]]
                grid_mask[far_indices] = True

        M = grid_mask.sum().item()

        # Build grid points for active contact points
        grid_points_flat = (
            contact_points[grid_mask, None, :] + normalized_coords[None, :, :] * grid_scale
        )  # (M, K^3, 3)
    else:
        grid_mask = torch.ones(N, dtype=torch.bool, device=device)  # (N,)
        M = N

        # Build grid points for all contact points
        grid_points_flat = (
            contact_points[:, None, :] + normalized_coords[None, :, :] * grid_scale
        )  # (M, K^3, 3)

    grid_points_all = grid_points_flat.reshape(-1, 3)  # (M * K^3, 3)

    # GPU nearest distance using Kaolin
    nn_dist, nn_face_idx, nn_point = nn_dist_to_mesh_gpu(
        grid_points_all, hand_verts, faces
    )

    K = kernel_size
    grid_distance = nn_dist.reshape(M, K, K, K, 1)
    nn_face_idx = nn_face_idx.reshape(M, K, K, K)
    nn_point = nn_point.reshape(M, K, K, K, 3)

    return grid_distance, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point


def msdf2mlcontact(obj_msdf, hand_verts, hand_cse, kernel_size, grid_scale, hand_faces, pool=None):
    """
    calculate MLcontact from MSDF and hand vertices.

    obj_msdf: torch.Tensor, (B, N, R^3 + 3), the MSDF volume.
    hand_verts: torch.Tensor, (B, H, 3), the hand vertices.
    hand_cse: torch.Tensor, (H, C), the hand contact surface embeddings.
    """
    centers = obj_msdf[:, :, kernel_size**3:]  # B, N, 3
    # obj_pt_mask, hand_vert_mask = calculate_contact_mask(centers, hand_verts, grid_scale=grid_scale, device=obj_msdf.device)
    normalized_coords = get_grid(kernel_size=kernel_size, device=obj_msdf.device).reshape(-1, 3).float()
    # all_pts = centers[:, :, None, :] + normalized_coords[None, None, :, :] * grid_scale  # B, N, K^3, 3
    local_grid_dist = torch.zeros(
        obj_msdf.shape[0], obj_msdf.shape[1],
        kernel_size, kernel_size, kernel_size,
        device=obj_msdf.device
    )  # B, N, K, K, K, 1 + cse_dim
    local_grid_cse = torch.zeros(
        obj_msdf.shape[0], obj_msdf.shape[1],
        kernel_size, kernel_size, kernel_size, hand_cse.shape[-1], dtype=torch.float32,
        device=obj_msdf.device
    )  # B, N, K, K, K, 1 + cse_dim

    # Prepare shared data (threads share memory, no copying needed)
    hand_faces_np = hand_faces.cpu().numpy()
    normalized_coords_np = normalized_coords.cpu().numpy()
    hand_cse_np = hand_cse.cpu().numpy()
    centers_np = centers.cpu().numpy()
    hand_verts_np = hand_verts.cpu().numpy()

    def process_batch(b):
        """Worker function using shared memory via closure."""
        hand_mesh = trimesh.Trimesh(vertices=hand_verts_np[b], faces=hand_faces_np)
        local_grid, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point = calc_local_grid_all_pts(
            contact_points=centers_np[b],
            normalized_coords=normalized_coords_np,
            obj_mesh=None,
            hand_mesh=hand_mesh,
            kernel_size=kernel_size,
            grid_scale=grid_scale,
            hand_cse=hand_cse_np,
        )
        return b, local_grid, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point

    # Execute in parallel or sequentially
    if pool is not None:
        results = list(pool.map(process_batch, range(obj_msdf.shape[0])))
    else:
        results = [process_batch(b) for b in range(obj_msdf.shape[0])]

    obj_pt_mask, hand_vert_mask, ho_dists = [], [], []
    for b, local_grid, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point in results:
        # Convert numpy arrays to torch tensors
        nn_face_idx_t = torch.from_numpy(nn_face_idx.flatten()).to(device=hand_faces.device, dtype=hand_faces.dtype)
        nn_point_t = torch.from_numpy(nn_point.reshape(-1, 3)).to(device=hand_verts.device, dtype=hand_verts.dtype)

        nn_vert_idx = hand_faces[nn_face_idx_t]  # (N, 3)
        face_verts = hand_verts[b, nn_vert_idx]  # (N, 3, 3)
        face_cse_t = hand_cse[nn_vert_idx]  # (M * K^3, 3, cse_dim)

        # Calculate barycentric weights using matrix inversion (pure tensor operations)
        face_verts_transposed = face_verts.transpose(1, 2)  # (N, 3, 3)
        w = torch.linalg.inv(face_verts_transposed) @ nn_point_t.unsqueeze(-1)  # (M * K^3, 3, 1)

        # Make sure weights are all positive and normalized
        w = torch.clamp(w, 0, 1)
        w = w / torch.sum(w, dim=1, keepdim=True)  # (M * K^3, 3, 1)

        grid_hand_cse = torch.sum(face_cse_t * w, dim=1).reshape(kernel_size, kernel_size, kernel_size, -1)  # (kernel_size, kernel_size, kernel_size, cse_dim)

        # Convert local_grid from numpy to torch
        local_grid_t = torch.from_numpy(local_grid).to(device=obj_msdf.device, dtype=local_grid_dist.dtype).squeeze(-1)
        local_grid_dist[b][grid_mask] = local_grid_t
        local_grid_cse[b][grid_mask] = grid_hand_cse.reshape(-1, kernel_size, kernel_size, kernel_size, hand_cse.shape[-1])
        obj_pt_mask.append(grid_mask)
        ho_dists.append(ho_dist)
        hand_vert_mask.append(verts_mask)

    obj_pt_mask = torch.from_numpy(np.array(obj_pt_mask)).to(device=obj_msdf.device)
    hand_vert_mask = torch.from_numpy(np.array(hand_vert_mask)).to(device=obj_msdf.device)
    ho_dists = torch.from_numpy(np.array(ho_dists)).to(device=obj_msdf.device).float()
    
    return local_grid_dist, local_grid_cse, obj_pt_mask, hand_vert_mask, ho_dists, normalized_coords

