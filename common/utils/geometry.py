import os
from itertools import product
import numpy as np
import torch
import pytorch3d.ops
import trimesh
from pytorch3d.structures import Meshes
from mesh_to_sdf import surface_point_cloud
from skimage.measure import marching_cubes
from kornia.geometry.linalg import inverse_transformation
import torch.nn.functional as F
import open3d as o3d
from pysdf import SDF


def rodrigues_rot(axis, angle):
    """
    axis: ... x 3 or (3,)
    angle: ... x 1 or float/scalar
    """
    axis = np.asarray(axis)
    angle = np.asarray(angle)
    if angle.ndim == 0:
        angle = angle.reshape(1)
    else:
        angle = angle.reshape(angle.shape + (1,))
    axx = np.zeros(axis.shape+(3,)) # ... x 3 x 3
    axx[..., [2, 0, 1], [1, 2, 0]] = axis
    axx[..., [1, 2, 0], [2, 0, 1]] = -axis
    I = np.zeros_like(axx)
    I[..., [0, 1, 2], [0, 1, 2]] = 1
    R = I + np.sin(angle) * axx + (1 - np.cos(angle)) * axx @ axx
    return R

def rodrigues_layer(axis: torch.Tensor, angle: torch.Tensor):
    angle = angle.unsqueeze(-1)
    axx = torch.zeros(axis.shape+(3,), device=axis.device) # ... x 3 x 3
    axx[..., [2, 0, 1], [1, 2, 0]] = axis
    axx[..., [1, 2, 0], [2, 0, 1]] = -axis
    I = torch.zeros_like(axx, device=axis.device)
    I[..., [0, 1, 2], [0, 1, 2]] = 1
    R = I + torch.sin(angle) * axx + (1 - torch.cos(angle)) * axx @ axx
    return R


def flip_x_axis(mesh):
    mesh.vertices[:, 0] *= -1
    mesh.faces[:, [0, 1]] = mesh.faces[:, [1, 0]]


def quaternion2matrix(q:np.ndarray) -> np.ndarray:
    """
    q: N x 4 unit quaternion
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    angle = np.arccos(q[:, 0]) * 2
    axis = q[:, 1:] / (np.sin(angle / 2).reshape(-1, 1) + 1.0e-6)
    R = rodrigues_rot(axis, angle)
    return R


## The following functions are from contactopt
def calculate_contact_capsule(hand_verts, hand_normals, object_verts, object_normals,
                              caps_top=0.0005, caps_bot=-0.0015, caps_rad=0.001, caps_on_hand=False, contact_norm_method=0):
    """
    Calculates contact maps on object and hand.
    :param hand_verts: (batch, V, 3)
    :param hand_normals: (batch, V, 3)
    :param object_verts: (batch, O, 3)
    :param object_normals: (batch, O, 3)
    :param caps_top: ctop, distance to top capsule center
    :param caps_bot: cbot, distance to bottom capsule center
    :param caps_rad: crad, radius of the contact capsule
    :param caps_on_hand: are contact capsules placed on hand or object vertices
    :param contact_norm_method: select a distance-to-contact function
    :return: object contact (batch, O, 1), hand contact (batch, V, 1)
    """
    if caps_on_hand:
        sdf_obj, dot_obj, nn_idx = capsule_sdf(hand_verts, hand_normals, object_verts, object_normals, caps_rad, caps_top, caps_bot, False)
        sdf_hand, dot_hand, _ = capsule_sdf(hand_verts, hand_normals, object_verts, object_normals, caps_rad, caps_top, caps_bot, True)
    else:
        sdf_obj, dot_obj, nn_idx = capsule_sdf(object_verts, object_normals, hand_verts, hand_normals, caps_rad, caps_top, caps_bot, True)
        sdf_hand, dot_hand, _ = capsule_sdf(object_verts, object_normals, hand_verts, hand_normals, caps_rad, caps_top, caps_bot, False)

    obj_contact = sdf_to_contact(sdf_obj, dot_obj, method=contact_norm_method)# * (dot_obj/2+0.5) # TODO dotting contact normal
    hand_contact = sdf_to_contact(sdf_hand, dot_hand, method=contact_norm_method)# * (dot_hand/2+0.5)

    # print('oshape, nshape', obj_contact.shape, (dot_obj/2+0.5).shape)##

    return obj_contact.unsqueeze(2), hand_contact.unsqueeze(2), nn_idx


def capsule_sdf(mesh_verts, mesh_normals, query_points, query_normals, caps_rad, caps_top, caps_bot, foreach_on_mesh):
    """
    Find the SDF of query points to mesh verts
    Capsule SDF formulation from https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    :param mesh_verts: (batch, V, 3)
    :param mesh_normals: (batch, V, 3)
    :param query_points: (batch, Q, 3)
    :param caps_rad: scalar, radius of capsules
    :param caps_top: scalar, distance from mesh to top of capsule
    :param caps_bot: scalar, distance from mesh to bottom of capsule
    :param foreach_on_mesh: boolean, foreach point on mesh find closest query (V), or foreach query find closest mesh (Q)
    :return: normalized sdf + 1 (batch, V or Q)
    """
    # TODO implement normal check?
    if foreach_on_mesh:     # Foreach mesh vert, find closest query point
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(mesh_verts, query_points, K=1, return_nn=True)   # TODO should attract capsule middle?

        capsule_tops = mesh_verts + mesh_normals * caps_top
        capsule_bots = mesh_verts + mesh_normals * caps_bot
        delta_top = nearest_pos[:, :, 0, :] - capsule_tops
        normal_dot = torch.sum(mesh_normals * batched_index_select(query_normals, 1, nearest_idx.squeeze(2)), dim=2)

    else:   # Foreach query vert, find closest mesh point
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(query_points, mesh_verts, K=1, return_nn=True)   # TODO should attract capsule middle?
        closest_mesh_verts = batched_index_select(mesh_verts, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)
        closest_mesh_normals = batched_index_select(mesh_normals, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)

        capsule_tops = closest_mesh_verts + closest_mesh_normals * caps_top  # Coordinates of the top focii of the capsules (batch, V, 3)
        capsule_bots = closest_mesh_verts + closest_mesh_normals * caps_bot
        delta_top = query_points - capsule_tops
        normal_dot = torch.sum(query_normals * closest_mesh_normals, dim=2)

    bot_to_top = capsule_bots - capsule_tops  # Vector from capsule bottom to top
    along_axis = torch.sum(delta_top * bot_to_top, dim=2)   # Dot product
    top_to_bot_square = torch.sum(bot_to_top * bot_to_top, dim=2)
    h = torch.clamp(along_axis / top_to_bot_square, 0, 1)   # Could avoid NaNs with offset in division here
    dist_to_axis = torch.norm(delta_top - bot_to_top * h.unsqueeze(2), dim=2)   # Distance to capsule centerline

    return dist_to_axis / caps_rad, normal_dot, nearest_idx  # (Normalized SDF)+1 0 on endpoint, 1 on edge of capsule


def calculate_penetration_cost(hand_verts, hand_normals, object_verts, object_normals, is_thin, contact_norm_method, allowable_pen=0.002):
    """
    Calculates an increasing cost for hands heavily intersecting with objects.
    Foreach hand vertex, find the nearest object point, dot with object normal.
    Include "allowable-pen" buffer margin to account for hand deformation.
    """

    allowable_pen = (torch.zeros_like(is_thin) + allowable_pen) * (1 - is_thin)
    allowable_pen = allowable_pen.unsqueeze(1)

    if contact_norm_method == 5:
        hand_verts_offset = hand_verts + hand_normals * -0.004
    else:
        hand_verts_offset = hand_verts

    knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(hand_verts_offset, object_verts, K=1, return_nn=True)   # Foreach hand vert, find closest obj vert

    closest_obj_verts = batched_index_select(object_verts, 1, nearest_idx.squeeze(2))  # Shape (batch, V, 3)
    closest_obj_normals = batched_index_select(object_normals, 1, nearest_idx.squeeze(2))  # Shape (batch, V, 3)

    # print('nearest shape', nearest_pos.shape, closest_obj_verts.shape)
    delta_pos = hand_verts - closest_obj_verts
    dist_along_normal = torch.sum(delta_pos * closest_obj_normals, dim=2)   # Dot product. Negative means backward along normal

    # print('d along normal', dist_along_normal.shape)

    pen_score = torch.nn.functional.relu(-dist_along_normal - allowable_pen)
    # print('pen score', pen_score)

    return pen_score


def sdf_to_contact(sdf, dot_normal=None, method=0):
    """
    Transform normalized SDF into some contact value
    :param sdf: NORMALIZED SDF, 1 is surface of object (torch.Tensor or np.ndarray)
    :param dot_normal: dot product with normal (torch.Tensor or np.ndarray), required for method 4
    :param method: select method
    :return: contact (batch, S, 1)
    """
    is_numpy = isinstance(sdf, np.ndarray)

    if is_numpy:
        if method == 0:
            c = 1 / (sdf + 0.0001)   # Exponential dropoff
        elif method == 1:
            c = -sdf + 2    # Linear dropoff
        elif method == 2:
            c = 1 / (sdf + 0.0001)   # Exponential dropoff
            c = np.power(c, 2)
        elif method == 3:
            c = 1 / (1 + np.exp(-(-sdf + 2.5)))  # Sigmoid
        elif method == 4:
            c = (-dot_normal/2+0.5) / (sdf + 0.0001)   # Exponential dropoff with sharp normal
        elif method == 5:
            c = 1 / (sdf + 0.0001)   # Proxy for other stuff
        return np.clip(c, 0.0, 1.0)
    else:
        if method == 0:
            c = 1 / (sdf + 0.0001)   # Exponential dropoff
        elif method == 1:
            c = -sdf + 2    # Linear dropoff
        elif method == 2:
            c = 1 / (sdf + 0.0001)   # Exponential dropoff
            c = torch.pow(c, 2)
        elif method == 3:
            c = torch.sigmoid(-sdf + 2.5)
        elif method == 4:
            c = (-dot_normal/2+0.5) / (sdf + 0.0001)   # Exponential dropoff with sharp normal
        elif method == 5:
            c = 1 / (sdf + 0.0001)   # Proxy for other stuff
        return torch.clamp(c, 0.0, 1.0)


class GridDistanceToContact:
    """
    Unified interface for transforming grid distances to contact values.

    This class handles the normalization of distances based on grid scale and kernel size,
    then transforms them to contact values using sdf_to_contact.

    :param scale: The grid scale (physical size of the grid)
    :param kernel_size: The kernel size (number of grid points along each dimension)
    :param method: The contact transformation method (default: 2)
    """

    def __init__(self, scale: float, kernel_size: int, method: int = 2):
        self.scale = scale
        self.kernel_size = kernel_size
        self.method = method
        # Normalization factor: half distance of 1 grid cell
        self.norm_factor = scale / (kernel_size - 1)

    def __call__(self, dist, dot_normal=None):
        """
        Transform grid distances to contact values.

        :param dist: Raw distances (torch.Tensor or np.ndarray)
        :param dot_normal: Optional dot product with normal for method 4
        :return: Contact values in [0, 1]
        """
        normalized_dist = dist / self.norm_factor
        return sdf_to_contact(normalized_dist, dot_normal, method=self.method)

    @classmethod
    def from_config(cls, cfg, method: int = 2):
        """
        Create a GridDistanceToContact instance from a config object.

        :param cfg: Config object with 'scale' and 'kernel_size' attributes
        :param method: The contact transformation method (default: 2)
        :return: GridDistanceToContact instance
        """
        return cls(scale=cfg.scale, kernel_size=cfg.kernel_size, method=method)


def batched_index_select(t, dim, inds):
    """
    Helper function to extract batch-varying indicies along array
    :param t: array to select from
    :param dim: dimension to select along
    :param inds: batch-vary indicies
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near


def transform_obj(verts, rot_aa, trans):
    """
    verts: N x 3
    """
    angle = np.linalg.norm(rot_aa) + 1.0e-8
    axis = rot_aa / angle
    R = rodrigues_rot(axis, angle)
    verts = verts @ R
    return verts + trans.reshape(1, 3)


def transform_mesh(mesh:trimesh.Trimesh, T:np.array):
    """
    Seems trimesh apply_transformation does not work well.
    """
    verts = mesh.vertices
    homo_verts = np.concatenate((verts, np.ones((verts.shape[0], 1))), axis=1)
    verts = (T.reshape(1, 4, 4) @ homo_verts.reshape(-1, 4, 1))[:, :3, 0]
    mesh.vertices = verts

    return mesh


def point_set_register(pc1: np.ndarray, pc2: np.ndarray):
    """
    Register point cloud 1 to point cloud 2. Returns a 4 x 4 transformation matrix. s.t. T @ pc1 -> pc2
    :param pc1: point cloud 1, n1 x 3
    :param pc2: point cloud 2, n2 x 3
    :return: transformation matrix 4 x 4
    """

    mean1 = np.mean(pc1, axis=0, keepdims=True)
    pts1 = pc1 - mean1
    mean2 = np.mean(pc2, axis=0, keepdims=True)
    pts2 = pc2 - mean2
    W = np.sum(pts2.reshape(-1, 3, 1) @ pts1.reshape(-1, 1, 3), axis=0)
    U, S, Vh = np.linalg.svd(W)

    I = np.eye(3)
    # R = U @ Vh
    # I[:, 2, 2] = np.det(R)
    R = U @ I @ Vh
    t = (mean2.reshape(3, 1) - R @ mean1.reshape(3, 1)).flatten()
    return R, t

def nearest_neighbour(pts1: torch.Tensor, pts2: torch.Tensor):
    """
    Calculate the nearest neighbour distances in batch for two point sets.
    :param pts1: <*Bs x N1 x 3>
    :param pts2: <*Bs x N2 x 3>
    :return: nearest distance of <*Bs x N1>, nearest neighbour indices of <*Bs x N1>
    """
    diff_vecs = pts2.unsqueeze(-3) - pts1.unsqueeze(-2) # *Bs x N1 x N2 x 3
    dists = torch.norm(diff_vecs, dim=-1) # Bs x N1 x N2
    nn_dist, nn_idx = torch.min(dists, dim=-1)

    return nn_dist, nn_idx


def sample_bbox3d(verts, sdf_level, th):
    BBox3D = np.array([[np.min(verts[:, 0])-sdf_level-th, np.max(verts[:, 0])+sdf_level+th],
                       [np.min(verts[:, 1])-sdf_level-th, np.max(verts[:, 1])+sdf_level+th],
                       [np.min(verts[:, 2])-sdf_level-th, np.max(verts[:, 2])+sdf_level+th]])
    return BBox3D


def sample_sdf_level_set_parallel(params):
    print(f'Start processing {params["obj_name"]}')
    # try:
    res = sample_sdf_level_set(params['mesh'], params['num_samples'], params['sdf_levels'], params['th'])
    print(f'Done processing {params["obj_name"]}')
    # except Exception as exc:
    #     print(f'Failed to process {params["obj_name"]} due to {exc}')
    #     exit(-1)
    return res


def sample_sdf_parallel(params):
    print(f'Start processing {params["obj_name"]}')
    mesh = params['mesh']
    gap = params['sdf_sample_gap']
    BBox3D = sample_bbox3d(mesh.vertices, 0.2, 0) ## Bbox is +- 20cm
    n_bins = np.ceil(BBox3D / gap)
    x, y, z = np.meshgrid(np.arange(n_bins[0, 0], n_bins[0, 1]+1),
                          np.arange(n_bins[1, 0], n_bins[1, 1]+1),
                          np.arange(n_bins[2, 0], n_bins[2, 1]+1),
                          indexing='ij')
    xyz = np.stack([x, y, z], axis=-1) * gap
    # try:
    sdfs = mesh_to_sdf(mesh, xyz.reshape(-1, 3), sign_method='depth')
    ret_sdfs = np.concatenate([xyz, sdfs.reshape(*xyz.shape[:3], 1)], axis=-1)
    print(f'Done processing {params["obj_name"]}')
    # except Exception as exc:
    #     print(f'Failed to process {params["obj_name"]} due to {exc}.')
    #     exit(-1)

    return ret_sdfs


def sample_sdf_level_set(mesh: trimesh.Trimesh, num_samples:int, sdf_levels:np.ndarray, th:float=0.001):
    """
    :param mesh: The mesh to perform the sample
    :param num_samples: The number of samples
    :param sdf_level: the target sdf. level0 < 0, level1 = 0, level2+ > 0
    :param th: threshold of sdf
    :return: numpy of <num_samples x 3>
    """

    BBox3D = sample_bbox3d(mesh.vertices, sdf_levels[-1], th)
    samples = (BBox3D[:, 1] - BBox3D[:, 0]).reshape(1, 3) * np.random.rand(int(len(sdf_levels) * num_samples),
                                                                           3) + BBox3D[:, 0].reshape(1, 3)
    sdfs = mesh_to_sdf(mesh, samples, sign_method='depth')
    if np.min(sdfs) > sdf_levels[0]:
        sdf_levels[0] = np.min(sdfs)

    ret_sdf_levels = sdf_levels.copy()
    pts = [[] for _ in range(sdf_levels.shape[0])]
    nms = [[] for _ in range(sdf_levels.shape[0])]
    cnt = np.zeros_like(sdf_levels)

    while (cnt < num_samples).any():
        BBox3D = sample_bbox3d(mesh.vertices, sdf_levels[-1], th)
        samples = (BBox3D[:, 1] - BBox3D[:, 0]).reshape(1, 3) * np.random.rand(int(40 * num_samples),
                                                                               3) + BBox3D[:, 0].reshape(1, 3)
        sdfs, normals = mesh_to_sdf(mesh, samples, sign_method='depth', return_gradients=True)
        for i in range(sdf_levels.shape[0]):
            if i == 0:
                ub = min(sdf_levels[i] + th, 0)
            else:
                ub = sdf_levels[i] + th
            mask = np.logical_and(sdfs > sdf_levels[i] - th, sdfs < ub)
            pts[i].append(samples[mask])
            nms[i].append(normals[mask])
            cnt[i] += np.sum(mask)

        for c in cnt[len(sdf_levels)-1::-1]:
            if c >= num_samples:
                sdf_levels = sdf_levels[:-1]
            else:
                break

    if (cnt >= num_samples).all():
        pts = [np.concatenate(arrs, axis=0)[:num_samples] for arrs in pts]
        pts = np.stack(pts, axis=0)
        nms = [np.concatenate(arrs, axis=0)[:num_samples] for arrs in nms]
        nms = np.stack(nms, axis=0)
    else:
        raise Exception(f'Not enough points sampled after 30 iterations. Current sampled points are {cnt}.')

    return pts, nms, ret_sdf_levels


def sphere_contact(pc1: torch.Tensor, pc2: torch.Tensor, fading:float=0.005):
    """
    :param pc1: Point cloud 1 <Bs x N1 x 3>
    :param pc2: Point cloud 2 <Bs x N2 x 3>
    :param fading: The fading factor of the contact
    :return: The continuous contact of spherical contact on pc1 regarding pc2
    """
    dists = torch.norm(pc1.unsqueeze(-2) - pc2.unsqueeze(-3), dim=-1)
    nn_dists, nn_idx = torch.min(dists, dim=-1)
    # contact = 1 / (nn_dists / fading + 1)
    contact = torch.zeros_like(nn_dists, device=nn_dists.device)
    min_idx = torch.argsort(nn_dists, dim=-1)[:, :8]
    for i in range(contact.shape[0]):
        contact[i, min_idx[i]] = 1
    # contact = nn_dists < fading
    return contact, nn_idx


def sample_from_sdf_grid(sd: float, sdf_grid:np.ndarray, num_samples: int, lbcorner:np.ndarray, spacing=0.005):
    """
    :param sd: the signed distance
    :param sdf_grid: the grid <N1 x N2 x N3> of signed distance.
           (x,y,z) is the coordinate and d is the signed distance.
    :param num_samples: The number of samples
    :return:
    Use marching cubes in validation stage and TODO: change to faster algorithms
    """
    verts, faces, normals, values = marching_cubes(sdf_grid, level=sd)
    verts = verts * spacing + lbcorner.reshape(1, 3)
    level_set = trimesh.Trimesh(verts, faces)
    samples, ids = trimesh.sample.sample_surface(level_set, num_samples)
    sample_normals = level_set.face_normals[ids]
    return samples, sample_normals


## reimplemetation of the mesh_to_sdf in the package.
## Opening the API of return_gradients.
def mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='depth', bounding_radius=None,
                scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, return_gradients=False):
    if not isinstance(query_points, np.ndarray):
        raise TypeError('query_points must be a numpy array.')
    if len(query_points.shape) != 2 or query_points.shape[1] != 3:
        raise ValueError('query_points must be of shape N ✕ 3.')

    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    point_cloud = get_surface_point_cloud(mesh, surface_point_method, bounding_radius, scan_count, scan_resolution,
                                          sample_point_count, calculate_normals=return_gradients)

    if sign_method == 'normal':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False, return_gradients=return_gradients)
    elif sign_method == 'depth':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=True, sample_count=sample_point_count, return_gradients=return_gradients)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))


def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=None, scan_count=100,
                            scan_resolution=400, sample_point_count=10000000, calculate_normals=True):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1

    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans(mesh, bounding_radius=bounding_radius, scan_count=scan_count,
                                                     scan_resolution=scan_resolution,
                                                     calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count,
                                                    calculate_normals=calculate_normals)
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))


def parallel_mesh_to_sdf(params):
    if params['lib'] == 'mesh_to_sdf':
        return mesh_to_sdf(params['mesh'], params['query_points'], return_gradients=params['return_gradients'])
    elif params['lib'] == 'pySDF':
        mesh_sdf = SDF(params['mesh'].vertices, params['mesh'].faces)
        anc_sdf = - mesh_sdf(params['query_points'])
        anc_nn = mesh_sdf.nn(params['query_points'])
        anc_sdf_normal = np.sign(anc_sdf).reshape(32, 1) * (params['query_points'] - params['mesh'].vertices[anc_nn])
        anc_sdf_normal = normalize_vec(anc_sdf_normal)
        return anc_sdf, anc_sdf_normal
    elif params['lib'] == 'grid':
        anc_sdf, anc_sdf_normal = get_sdf_values([params['sdf_samples']], params['query_points'][np.newaxis, ...])
        return anc_sdf, anc_sdf_normal


def contact2rot(obj_verts, part_verts, contact, corr_map, method='hard'):
    """
    obj_verts: Bs x N1 x 3
    part_verts: Bs x N2 x 3
    contact: Bs x N1
    corr_map: Bs x N1 x N2
    Get the orientation of the local part
    """
    if method == 'hard':
        corr_map = torch.exp(corr_map * 50)
        corr_map = corr_map / torch.sum(corr_map, dim=-1, keepdim=True)
    #     contact = torch.exp(contact * 50)
    #     contact = contact / torch.sum(contact, dim=-1, keepdim=True)
    H = part_verts.transpose(-1, -2) @ corr_map.transpose(-1, -2) @ (obj_verts * contact.unsqueeze(-1))
    U, S, Vh = torch.linalg.svd(H)
    R = (U @ Vh).transpose(-1, -2)
    return R

def get_v2v_rot(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """
    Get the rotation matrix from n1 to n2;
    """
    ax = np.cross(n1, n2)
    if np.linalg.norm(ax) < 1e-8:
        if n1[0] < 1e-8:
            ax[1] = n1[2]
            ax[2] = -n1[1]
        else:
            ax[0] = n1[1]
            ax[1] = -n1[0]
    ax = normalize_vec(ax)
    ang = np.arccos(np.dot(n1, n2) /
                    (np.linalg.norm(n1, axis=-1, keepdims=True) * np.linalg.norm(n2, axis=-1, keepdims=True)))
    R = rodrigues_rot(ax, ang)
    return R

def v2v_rot_layer(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
    ax = torch.cross(n1, n2)
    ax = normalize_tensor(ax)
    ang = torch.arccos(torch.sum(n1 * n2, dim=-1) /
                (torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1)))
    R = rodrigues_layer(ax, ang.unsqueeze(-1))

    return R


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """
    vec: ... x N
    """
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)

def normalize_tensor(vec: torch.Tensor) -> torch.Tensor:
    return vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)


def trilinear_interpolation(values: torch.Tensor, coord: torch.Tensor):
    """
    :param values: (Bs x 2 x 2 x 2) cubic values. coordinate in the order of x, y, z
    :param coord: torch.Tensor of (Bs x 3). Normalized to (0, 1)^3
    :return: interpolated value
    """
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    v00 = values[:, 0, 0, 0] * (1 - x) + values[:, 1, 0, 0] * x
    v01 = values[:, 0, 0, 1] * (1 - x) + values[:, 1, 0, 1] * x
    v10 = values[:, 0, 1, 0] * (1 - x) + values[:, 1, 1, 0] * x
    v11 = values[:, 0, 1, 1] * (1 - x) + values[:, 1, 1, 1] * x
    v0 = v00 * (1 - y) + v10 * coord[:, 1]
    v1 = v01 * (1 - y) + v11 * coord[:, 1]
    v = v0 * (1 - z) + v1 * coord[:, 2]
    ## Normal by taking the gradient.
    nx =  (1-y) * (1-z) * (values[:, 1, 0, 0] - values[:, 0, 0, 0])\
         +  y * (1 - z) * (values[:, 1, 1, 0] - values[:, 0, 1, 0])\
         +  (1 - y) * z * (values[:, 1, 0, 1] - values[:, 0, 0, 1])\
         +    y  *  z   * (values[:, 1, 1, 1] - values[:, 0, 1, 1])

    ny =  (1-z) * (1-x) * (values[:, 0, 1, 0] - values[:, 0, 0, 0])\
         +  z * (1 - x) * (values[:, 0, 1, 1] - values[:, 0, 0, 1])\
         +  (1 - z) * x * (values[:, 1, 1, 0] - values[:, 1, 0, 0])\
         +    z  *  x   * (values[:, 1, 1, 1] - values[:, 1, 0, 1])

    nz =  (1-x) * (1-y) * (values[:, 0, 0, 1] - values[:, 0, 0, 0])\
         +  x * (1 - y) * (values[:, 1, 0, 1] - values[:, 1, 0, 0])\
         +  (1 - x) * y * (values[:, 0, 1, 1] - values[:, 0, 1, 0])\
         +    x  *  y   * (values[:, 1, 1, 1] - values[:, 1, 1, 0])

    normal = torch.stack([nx, ny, nz], dim=-1)
    normal = normal / torch.norm(normal, dim=-1, keepdim=True)

    return v, normal


def cp_match(pts1: torch.Tensor, pts2: torch.Tensor, weight: torch.Tensor=None):
    """
    Do closest point matching for point clouds pts1 and pts2.
    The points are in correspondence according to the order.
    """
    if weight is None:
        weight = torch.ones(*pts1.shape[:-1], 1, device=pts1.device)
    mean1 = torch.sum(pts1 * weight, dim=1, keepdim=True) / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
    pts1 = pts1 - mean1
    mean2 = torch.sum(pts2 * weight, dim=1, keepdim=True) / (torch.sum(weight, dim=1, keepdim=True)+1e-8)
    pts2 = pts2 - mean2
    W = torch.sum((pts2.unsqueeze(-1) @ pts1.unsqueeze(-2)) * weight.unsqueeze(-1), dim=1)
    U, S, Vh = torch.linalg.svd(W)
    R = U @ Vh
    I = torch.eye(3, device=pts1.device, dtype=pts1.dtype).view(1, 3, 3).repeat(pts1.shape[0], 1, 1)
    # Cast to float32 for determinant computation if needed (fp16 not supported)
    R_compute = R.float() if R.dtype == torch.float16 else R
    I[:, 2, 2] = torch.det(R_compute).to(I.dtype)
    R = U @ I @ Vh
    t = (mean2.transpose(-1, -2) - R @ mean1.transpose(-1, -2)).view(-1, 3)
    return R, t


def cp_match_Ronly(pts1: torch.Tensor, pts2: torch.Tensor, weight: torch.Tensor=None):
    """
    Do closest point matching for point clouds pts1 and pts2.
    The points are in correspondence according to the order.
    Only return rotation.
    """
    if weight is None:
        weight = torch.ones(*pts1.shape[:-1], 1, device=pts1.device)
    W = torch.sum((pts2.unsqueeze(-1) @ pts1.unsqueeze(-2)) * weight.unsqueeze(-1), dim=1)
    U, S, Vh = torch.linalg.svd(W)
    R = U @ Vh
    I = torch.eye(3, device=pts1.device, dtype=pts1.dtype).view(1, 3, 3).repeat(pts1.shape[0], 1, 1)
    # Cast to float32 for determinant computation if needed (fp16 not supported)
    R_compute = R.float() if R.dtype == torch.float16 else R
    I[:, 2, 2] = torch.det(R_compute).to(I.dtype)
    R = U @ I @ Vh
    return R


def geo_distance_3dmap(sdf_samples: torch.Tensor, target_idx: torch.Tensor, th=0, gap=0.005):
    """
    :param sdf_samples: n1 x n2 x n3
    :param target_idx: (i, j, k) the index of reference point.
    :param th: the threshold of signed distance to be redeemed inside points. default to 0
    :return: A 3D map of the geodesic distance of the target point.
    """
    n1, n2, n3 = sdf_samples.shape
    adj_offset = torch.as_tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]).int()
    def spread_frontier(frontier, interior):
        ft = (frontier.unsqueeze(0) + adj_offset.unsqueeze(1)).view(-1, 3)
        ft = torch.unique(ft, dim=0)
        ## Remove outside points
        hard_mask = (ft[:, 0] >= 0) & (ft[:, 0] < n1) & (ft[:, 1] >= 0) & (ft[:, 1] < n2) & (ft[:, 2] >= 0) & (ft[:, 2] < n3)
        ft = ft[hard_mask]
        dist = torch.as_tensor([sdf_samples[tuple(idx)] for idx in ft])
        outside_mask = dist > th
        new_frontier = ft[outside_mask]
        if len(interior):
            int_mask = torch.as_tensor([(idx.unsqueeze(0) == interior).all(dim=-1).any() for idx in new_frontier])
            new_frontier = new_frontier[~int_mask]
            new_interior = torch.cat([interior, frontier])
        else:
            new_interior = frontier
        return new_frontier, new_interior

    distance = torch.zeros_like(sdf_samples)
    frontier_set = target_idx
    interior_set = torch.tensor([])
    dist = 0
    n_pts = torch.sum(sdf_samples > th).item()
    while frontier_set.shape[0] > 0:
        for idx in frontier_set:
            distance[tuple(idx)] = dist
        dist += gap
        frontier_set, interior_set = spread_frontier(frontier_set, interior_set)
        print(f"{interior_set.shape[0]}/{n_pts}, frontier distance {dist: .3f}.")
        if dist > 0.2:
            break

    return distance

def get_sdf_values(sdf_samples, coords, step_size=0.005):
    """
    :param sdf_samples: list of n1 x n2 x n3, contains some nan values
    :param coords: B x n_samples x 3
    :param step_size: float
    :return:
    """

    pt_sdfs, pt_normals = torch.zeros(coords.shape[:2], device=coords.device), torch.zeros_like(coords, device=coords.device)
    for b in range(len(sdf_samples)):
        sdf_coord, sdf_values = sdf_samples[b][..., :3], sdf_samples[b][..., 3]
        bottom_idx = ((coords[b] - sdf_coord[0, 0, 0].unsqueeze(0)) / step_size).int()

        ## For points falling outside the object:
        inside_mask = (bottom_idx >= 0).all(dim=-1) & (bottom_idx < torch.as_tensor(sdf_values.shape, device=coords.device) - 2).all(dim=-1)

        if torch.sum(inside_mask):
            sdf_values = torch.stack([sdf_values[bi[0]: bi[0] + 2,
                                      bi[1]: bi[1] + 2, bi[2]: bi[2] + 2,
                                      ] for bi in bottom_idx[inside_mask]], dim=0)
            normalized_coord = coords[b] - bottom_idx * step_size
            b_pt_sdfs, b_pt_normals = trilinear_interpolation(sdf_values, normalized_coord[inside_mask])
            pt_sdfs[b, inside_mask] = b_pt_sdfs
            pt_normals[b, inside_mask] = b_pt_normals

        if torch.sum(~inside_mask):
            # dist = torch.norm(coords[b, ~inside_mask].view(-1, 1, 1, 1, 3) - sdf_coord[b, :n1s[b], :n2s[b], :n3s[b]].unsqueeze(0), dim=-1).view(torch.sum(~inside_mask), -1)
            # nn_dist, nn_idx = torch.min(dist, dim=-1)
            pt_sdfs[b, ~inside_mask] = torch.nan
            pt_normals[b, ~inside_mask] = torch.nan

    return pt_sdfs, pt_normals

def get_perpend_vecs(vs: np.ndarray) -> tuple:
    """
    :param vs: (N x 3) a set of 3D vectors;
    :return: tuple of (N x 3) -> the 2 perpendicular vector of the current vector
    if n = [a, b, c], then one perpendicular vector will be: [0, -c, b];
    By cross product, we can get the second vector as: [b^2+c^2, -ab, -ac]
    """
    n1 = np.zeros_like(vs)
    n1[..., 1] = -vs[..., 2]
    n1[..., 2] = vs[..., 1]
    n2 = np.cross(vs, n1, axis=-1)
    return n1, n2


def normalize_vec_tensor(vec: torch.Tensor) -> torch.Tensor:
    return vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)

def compute_uv(hand_frames, obj_verts, obj_parts):
    B, N, P = obj_verts.shape[0], obj_verts.shape[1], hand_frames.shape[1]
    inv_hand_frames = inverse_transformation(hand_frames.reshape(-1, 4, 4))
    part_label = F.one_hot(obj_parts.reshape(-1), num_classes=P).reshape(B, N, P).transpose(1, 2)
    obj_verts = obj_verts.unsqueeze(dim=1).expand(B, P, N, 3).reshape(-1, N, 3)
    local_verts = torch.bmm(obj_verts, inv_hand_frames[:, :3, :3].transpose(1, 2)) + inv_hand_frames[:, None, :3, 3]
    local_verts = local_verts.reshape(B, P, N, 3)
    local_verts = (part_label[:, :, :, None] * local_verts).sum(dim=1)
    uv_pred = local_verts / (torch.norm(local_verts, dim=2, keepdim=True) + 1e-10)
    return uv_pred


def compute_uv_loss(pred_uv, target_uv, weight=None):
    loss = 1 - torch.cosine_similarity(pred_uv, target_uv, dim=-1)
    if weight is not None:
        loss = loss * weight    
    return loss.sum(dim=1).mean(dim=0)


def compute_part_bps(hand_verts, part_labels, half_size=0.5, resolution=64):
    """
    Compute the part BPS for hand vertices.
    hand_verts: B x V x 3
    part_labels: B x V
    half_size: float
    resolution: int
    return: B x P x R^3
    """
    B, V = hand_verts.shape[0], hand_verts.shape[1]
    P = part_labels.max().item() + 1
    grid_pts = torch.stack(torch.meshgrid(
        torch.linspace(-half_size, half_size, resolution),
        torch.linspace(-half_size, half_size, resolution),
        torch.linspace(-half_size, half_size, resolution),
        indexing='ij'
    ), dim=-1).to(hand_verts.device).reshape(-1, 3)  # R^3 x 3

    part_bps = []
    for p in range(P):
        part_mask = (part_labels == p).unsqueeze(-1).expand(B, V, 3)
        part_verts = hand_verts[part_mask].reshape(B, -1, 3)
        if part_verts.shape[1] == 0:
            part_bps.append(torch.zeros(B, resolution**3, device=hand_verts.device))
            continue
        dists = torch.cdist(grid_pts.unsqueeze(0).expand(B, -1, -1), part_verts)  # B x R^3 x Vp
        min_dists, _ = torch.min(dists, dim=-1)  # B x R^3
        part_bps.append(min_dists)
    part_bps = torch.stack(part_bps, dim=1)  # B x P x R^3
    bin_width = 2 * half_size / resolution
    part_bps = 1 / (part_bps / bin_width + 1e-8)
    part_bps = torch.clamp(part_bps, max=1.0)
    return part_bps


def compute_point_bps(hand_verts, half_size=0.5, resolution=64):
    """
    Compute the point BPS for hand vertices.
    hand_verts: B x V x 3
    half_size: float
    resolution: int
    return: tuple of (B x R^3, B x R^3). The first is the BPS distance (uniform encoding),
            the second is the index of the nearest vertex.
    """
    B, V = hand_verts.shape[0], hand_verts.shape[1]
    grid_pts = torch.stack(torch.meshgrid(
        torch.linspace(-half_size, half_size, resolution),
        torch.linspace(-half_size, half_size, resolution),
        torch.linspace(-half_size, half_size, resolution),
        indexing='ij'
    ), dim=-1).to(hand_verts.device).reshape(-1, 3)  # R^3 x 3

    dists = torch.cdist(grid_pts.unsqueeze(0).expand(B, -1, -1), hand_verts)  # B x R^3 x V
    min_dists, nearest_indices = torch.min(dists, dim=-1)  # B x R^3
    # min_dists = min_dists / half_size
    bin_width = 2 * half_size / resolution
    point_bps = 1 / (min_dists / bin_width + 1e-8)
    point_bps = torch.clamp(point_bps, max=1.0)
    return point_bps, nearest_indices, grid_pts


def grid_reorder_id_and_rot(kernel_size, id):
    """
    Reorder the 3D grid points according to the rotation along axis by angle.
    :param kernel_size: int, the resolution of the grid
    :param id: 0 - 23, the rotation id covering all 24 axis-aligned cube symmetries.
        0 -  3: +X face up (identity base),          spin 0/90/180/270° around X
        4 -  7: +Y face up (90° around Y),           spin 0/90/180/270° around X
        8 - 11: +Z face up (90° around Z),           spin 0/90/180/270° around X
       12 - 15: -X face up (180° around Y),          spin 0/90/180/270° around X
       16 - 19: -Y face up (270° around Y = -90°),   spin 0/90/180/270° around X
       20 - 23: -Z face up (270° around Z = -90°),   spin 0/90/180/270° around X
    """
    idxs = np.arange(kernel_size ** 3).reshape(kernel_size, kernel_size, kernel_size)
    if id < 4:
        R1 = np.eye(3)
    elif id < 8:
        idxs = reorder_3d(idxs, 1, 1)  # 90° around Y
        R1 = rodrigues_rot(np.array([0, 1, 0]), np.pi / 2)
    elif id < 12:
        idxs = reorder_3d(idxs, 2, 1)  # 90° around Z
        R1 = rodrigues_rot(np.array([0, 0, 1]), np.pi / 2)
    elif id < 16:
        idxs = reorder_3d(idxs, 1, 2)  # 180° around Y
        R1 = rodrigues_rot(np.array([0, 1, 0]), np.pi)
    elif id < 20:
        idxs = reorder_3d(idxs, 1, 3)  # 270° around Y
        R1 = rodrigues_rot(np.array([0, 1, 0]), 3 * np.pi / 2)
    else:  # id < 24
        idxs = reorder_3d(idxs, 2, 3)  # 270° around Z
        R1 = rodrigues_rot(np.array([0, 0, 1]), 3 * np.pi / 2)

    idxs = reorder_3d(idxs, 0, id % 4)
    R = rodrigues_rot(np.array([1, 0, 0]), (id % 4) * np.pi / 2) @ R1

    return idxs.reshape(-1), R


def reorder_3d(input_array, axis_id, rot_id):
    """
    Reorder the 3D array according to the rotation around a specified axis.
    :param input_array: K x K x K array (numpy or torch tensor)
    :param axis_id: 0, 1, or 2 - the axis to rotate around (0: x, 1: y, 2: z)
    :param rot_id: 0 - 3, the rotation angle (0: 0, 1: pi/2, 2: pi, 3: 3pi/2)
    :return: output_array: K x K x K array (same type as input)
    """
    if rot_id == 0:
        return input_array

    # Check if input is numpy array or torch tensor
    is_numpy = isinstance(input_array, np.ndarray)

    # Define flip and transpose operations based on array type
    if is_numpy:
        def flip(arr, axis):
            return np.flip(arr, axis=axis)
        def transpose(arr, ax1, ax2):
            axes = list(range(arr.ndim))
            axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
            return np.transpose(arr, axes)
    else:
        def flip(arr, axis):
            return torch.flip(arr, dims=[axis])
        def transpose(arr, ax1, ax2):
            return arr.transpose(ax1, ax2)

    # Determine the two axes perpendicular to the rotation axis
    # Rotation around axis_id affects the other two axes
    # axis_id=0 (x): rotate in y-z plane (axes 1, 2)
    # axis_id=1 (y): rotate in x-z plane (axes 0, 2)
    # axis_id=2 (z): rotate in x-y plane (axes 0, 1)
    if axis_id == 0:
        # Rotate around x-axis: affects y (dim 1) and z (dim 2)
        # Rotation matrix for angle theta around x: [[1,0,0], [0,cos,-sin], [0,sin,cos]]
        if rot_id == 3:  # 90 degrees: y -> z, z -> -y
            return flip(transpose(input_array, 1, 2), 2)
        elif rot_id == 2:  # 180 degrees: y -> -y, z -> -z
            return flip(flip(input_array, 1), 2)
        elif rot_id == 1:  # 270 degrees: y -> -z, z -> y
            return flip(transpose(input_array, 1, 2), 1)
    elif axis_id == 1:
        # Rotate around y-axis: affects x (dim 0) and z (dim 2)
        # Rotation matrix for angle theta around y: [[cos,0,sin], [0,1,0], [-sin,0,cos]]
        if rot_id == 3:  # 90 degrees: x -> -z, z -> x
            return flip(transpose(input_array, 0, 2), 0)
        elif rot_id == 2:  # 180 degrees: x -> -x, z -> -z
            return flip(flip(input_array, 0), 2)
        elif rot_id == 1:  # 270 degrees: x -> z, z -> -x
            return flip(transpose(input_array, 0, 2), 2)
    elif axis_id == 2:
        # Rotate around z-axis: affects x (dim 0) and y (dim 1)
        # Rotation matrix for angle theta around z: [[cos,-sin,0], [sin,cos,0], [0,0,1]]
        if rot_id == 3:  # 90 degrees: x -> y, y -> -x
            return flip(transpose(input_array, 0, 1), 1)
        elif rot_id == 2:  # 180 degrees: x -> -x, y -> -y
            return flip(flip(input_array, 0), 1)
        elif rot_id == 1:  # 270 degrees: x -> -y, y -> x
            return flip(transpose(input_array, 0, 1), 0)

    raise ValueError('Unknown axis_id {:d} or rot_id {:d}'.format(axis_id, rot_id))


def get_perpend_vecs_tensor(vs: torch.Tensor, device='cpu') -> tuple:
    """
    :param vs: (batch_size x N x 3) a set of 3D vectors;
    :return: tuple of (batch_size x N x 3) -> the 2 perpendicular vector of the current vector
    if n = [a, b, c], then one perpendicular vector will be: [0, -c, b];
    By cross product, we can get the second vector as: [b^2+c^2, -ab, -ac]
    """
    n1 = torch.zeros_like(vs, device=device)
    n1[..., 1] = -vs[..., 2]
    n1[..., 2] = vs[..., 1]
    n2 = torch.cross(vs, n1, dim=-1)
    return n1, n2