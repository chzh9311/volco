import coacd
import os.path as osp
import pickle
import json
import subprocess
import trimesh
import numpy as np
import scipy
import scipy.cluster
from scipy.stats import entropy
import torch
import open3d as o3d
from pysdf import SDF
from sklearn.cluster import KMeans
from tqdm import tqdm
import open3d as o3d

import pybullet
import pybullet_utils.bullet_client as bc
from common.utils.vis import o3dmesh_from_trimesh

from .bullet_simulation import run_simulation
from common.utils.converter import transform_to_canonical, convert_joints
from common.utils.geometry import make_watertight
from sklearn.neighbors import NearestNeighbors

value_metrics = ["Contact Ratio", "Success Rate", "Pierce-Free Rate", "Cluster Size", "Entropy", "Canonical Entropy", "Canonical Cluster Size"]

# MANO joint indices for each hand part (argmax blend-weight partition)
PART_JOINTS = {
    "palm":   [0],
    "thumb":  [13, 14, 15],
    "index":  [1, 2, 3],
    "middle": [4, 5, 6],
    "little": [7, 8, 9],
    "ring":   [10, 11, 12],
}


def diversity_legacy(params_list, cls_num=20):
    # k-means (original scipy implementation)
    params_list = scipy.cluster.vq.whiten(params_list)
    codes, dist = scipy.cluster.vq.kmeans(params_list, cls_num)  # codes: [20, 72], dist: scalar
    vecs, dist = scipy.cluster.vq.vq(params_list, codes)  # assign codes, vecs/dist: [1200]
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences  count: [20]
    ee = entropy(counts)
    return ee, np.mean(dist)


def diversity(params_list, cls_num=20):
    # k-means using sklearn (more robust)
    # Whiten the data (normalize by standard deviation)
    params_std = params_list.std(axis=0)
    params_std[params_std == 0] = 1.0  # Avoid division by zero
    params_whitened = params_list / params_std

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=cls_num, max_iter=300, n_init=10, random_state=0)
    kmeans.fit(params_whitened)

    # Get cluster assignments
    labels = kmeans.labels_

    # Calculate distances to cluster centers
    distances = np.linalg.norm(params_whitened[:, np.newaxis] - kmeans.cluster_centers_, axis=2)
    min_distances = distances.min(axis=1)

    # Count occurrences in each cluster
    counts = np.bincount(labels, minlength=cls_num)

    # Calculate entropy
    ee = entropy(counts)

    return ee, np.mean(min_distances)


def downsample_mesh(mesh, target_faces=10000):
    """
    Downsample mesh to target number of faces to reduce memory usage.

    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target number of faces after downsampling

    Returns:
        Downsampled trimesh.Trimesh object
    """
    if len(mesh.faces) <= target_faces:
        return mesh

    # Calculate reduction ratio for fast_simplification
    # target_reduction is the fraction of faces to REMOVE (not keep)
    # target_reduction = 1.0 - (target_faces / len(mesh.faces))

    # Use quadric decimation to preserve mesh quality while reducing complexity
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def parallel_calculate_metrics(params:dict):
    metrics = params['metrics']
    result = {}

    # Downsample mesh at the beginning to avoid memory overflow in all metrics
    # MAX_FACES = 10000  # Adjust based on memory constraints
    # if len(params['obj_model'].faces) > MAX_FACES:
    #     params['obj_model'] = downsample_mesh(params['obj_model'], target_faces=MAX_FACES)

    if "Simulation Displacement" in metrics:
        ## decomposition
        if "obj_hulls" not in params:
            mesh = coacd.Mesh(params['obj_model'].vertices, params['obj_model'].faces)
            params['obj_hulls'] = coacd.run_coacd(mesh)
        pb_disp = pybullet_parallel_interface(params) * 100 # to cm
        result["Simulation Displacement"] = pb_disp
        if "Stable Rate @ 2cm" in metrics:
            result["Stable Rate @ 2cm"] = pb_disp < 2
    if "Intersection Volume" in metrics:
        int_vol = intersect_vox(params['obj_model'], params['hand_model'], pitch=0.001) * 1000000 # turn to cm3
        # int_vol = intersect_vol_boolean(params['obj_model'], params['hand_model'], engine='manifold', fallback_pitch=0.005) * 1000000 # turn to cm3
        result["Intersection Volume"] = int_vol
    if "Boolean Intersection Volume" in metrics:
        int_vol = intersect_vol_boolean(params['obj_model'], params['hand_model'], engine='manifold', fallback_pitch=0.001) * 1000000 # turn to cm3
        result["Boolean Intersection Volume"] = int_vol
    if "Penetration Depth" in metrics:
        pen_depth, result_distance, penetr_vert_ids = pene_depth(obj_mesh=params['obj_model'], hand_verts=params['hand_model'].vertices) # to cm
        result["Penetration Depth"] = pen_depth * 100
        result["Penetration Depth Raw"] = result_distance * 100
        result["Penetration Depth Vert IDs"] = penetr_vert_ids

    if "Contact Area" in metrics:
        hand_mesh = trimesh.Trimesh(vertices=params['hand_model'].vertices, faces=params['hand_model'].faces)
        contact_area = calculate_contact_area(hand_mesh, params['obj_model'], threshold=0.005) * 10000 # to cm2
        result["Contact Area"] = contact_area

    if "Pierce-Free Rate" in metrics:
        try:
            pierce, n_holes = determine_pierce(params['obj_model'], params['hand_model'])
            result["Pierce-Free Rate"] = pierce
        except Exception as e:
            print(f"Pierce check failed for {params['frame_name']} with error: {e}")
            result["Pierce-Free Rate"] = False
        # print(pierce)

    if "Contact Ratio" in metrics:
        penetration_tol = 0.005
        hand_verts = params['hand_model'].vertices
        obj_sdf = SDF(params['obj_model'].vertices, params['obj_model'].faces)
        hv_sds = obj_sdf(hand_verts)

        contact = hv_sds > - penetration_tol
        sample_contact = contact.sum() > 0
        result["Contact Ratio"] = sample_contact
    return result


def pybullet_parallel_interface(params:dict):
    client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    hand_verts, hand_faces, obj_verts, obj_faces, fid, obj_hulls = (
        params['hand_model'].vertices, params['hand_model'].faces, params['obj_model'].vertices,
        params['obj_model'].faces, params['idx'], params['obj_hulls'])

    disp = run_simulation(hand_verts, hand_faces, obj_verts, obj_faces, indicator=fid, client=client, obj_hulls=obj_hulls, save_video=False)
    return disp


def determine_pierce(obj_mesh: trimesh.Trimesh, hand_mesh: trimesh.Trimesh):
    """
    The criteria to determine whether the hand penetrates through the object.
    The hand mesh should be watertight.
    Args:
        obj_mesh:
        hand_mesh:

    Returns:
    """
    for i in range(3):
        if obj_mesh.is_watertight:
            break
        else:
            omesh = o3dmesh_from_trimesh(obj_mesh)
            n_faces = len(omesh.triangles)
            mesh_smp = omesh.simplify_quadric_decimation(target_number_of_triangles=int(n_faces / 2))
            obj_mesh = trimesh.Trimesh(vertices=np.asarray(mesh_smp.vertices), faces=np.asarray(mesh_smp.triangles))

    if not hand_mesh.is_watertight:
        return np.nan, np.nan
    subtracted = trimesh.boolean.difference([hand_mesh, obj_mesh])

    # Check if the subtraction was successful
    if subtracted is None:
        raise ValueError("Boolean difference operation failed or resulted in no mesh.")

    # Check the connected components of the resulting mesh
    components = subtracted.split(only_watertight=False)

    # vis_geoms = [o3dmesh_from_trimesh(hand_mesh), o3dmesh_from_trimesh(obj_mesh)]
    # for comp in components:
    #     vis_geoms.append(o3dmesh_from_trimesh(comp).translate((0.3, 0, 0)))
    # Determine if the resulting mesh is continuous
    # o3d.visualization.draw_geometries(vis_geoms)
    no_pierce = len(components) == 1

    return no_pierce, len(components) - 1


def _obj_voxel_points(obj_mesh, hand_mesh, pitch):
    """Grid points that lie inside the object, over the hand-object overlap box.

    Localizes the intersection of the hand and object bounding boxes, builds a
    regular grid at `pitch` resolution inside it, and keeps points whose signed
    distance to the object is > -pitch (pysdf: positive inside, negative outside).
    Returns an (N, 3) array of interior grid points (possibly empty).
    """
    obj_lo, obj_hi = obj_mesh.bounds
    hand_lo, hand_hi = hand_mesh.bounds
    lo = np.maximum(obj_lo, hand_lo)
    hi = np.minimum(obj_hi, hand_hi)
    if np.any(hi <= lo):
        return np.empty((0, 3), dtype=np.float64)  # bounding boxes do not overlap

    # Grid the overlap box at `pitch` resolution (cell centers).
    axes = [np.arange(lo[d] + pitch / 2.0, hi[d], pitch) for d in range(3)]
    if any(a.size == 0 for a in axes):
        return np.empty((0, 3), dtype=np.float64)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    obj_sdf = SDF(obj_mesh.vertices, obj_mesh.faces)
    inside_obj = obj_sdf(grid) > -pitch  # pysdf: positive inside, negative outside
    return grid[inside_obj]


def intersect_vox(obj_mesh, hand_mesh, pitch=0.001):
    """
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    """
    obj_points = _obj_voxel_points(obj_mesh, hand_mesh, pitch)
    if len(obj_points) == 0:
        return 0.0
    # pysdf sign test for hand containment (positive == inside); ~1000x faster
    # than trimesh's ray-based `contains` without a compiled ray backend.
    hand_sdf = SDF(hand_mesh.vertices, hand_mesh.faces)
    inside = hand_sdf(obj_points) > 0
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def intersect_vol_boolean(obj_mesh, hand_mesh, engine='manifold', fallback_pitch=0.005):
    """
    Exact intersection volume between hand and object via mesh boolean.

    Unlike intersect_vox, which voxelizes only the object surface shell, this
    computes the true solid intersection. Numbers are therefore NOT comparable
    to the voxel metric reported by GrabNet/GraspTTA-lineage work.

    Requires both meshes to be watertight; falls back to solid voxelization
    when they are not, or when the boolean engine fails.

    :param engine: trimesh boolean backend (manifold3d is bundled with trimesh>=4)
    :param fallback_pitch: voxel size used by the fallback path
    :return: intersection volume, in the cube of the mesh units
    """
    if obj_mesh.is_watertight and hand_mesh.is_watertight:
        try:
            inter = trimesh.boolean.intersection([obj_mesh, hand_mesh], engine=engine)
            if inter is None or inter.is_empty:
                return 0.0
            # volume is signed and flips with winding order
            return abs(float(inter.volume))
        except Exception as e:
            print(f"Boolean intersection failed ({e}), falling back to solid voxelization.")
    else:
        print("Non-watertight input, falling back to solid voxelization.")

    obj_vox = obj_mesh.voxelized(pitch=fallback_pitch).fill()
    inside = hand_mesh.contains(obj_vox.points)
    return float(inside.sum() * np.power(fallback_pitch, 3))


def calc_diversity(hand_joints):
    cluster = []
    cluster2 = []
    kps = hand_joints.copy()
    for count, kps_i in enumerate(kps):
        cluster.append(kps_i.flatten())

    """cluster2"""
    hand_kps = torch.as_tensor(kps.copy()).float()
    is_right_vec = torch.ones(hand_kps.shape[0], device=hand_kps.device)

    hand_kps = convert_joints(hand_kps, source="mano", target="biomech")

    hand_kps_after, _ = transform_to_canonical(hand_kps, is_right_vec)
    hand_kps_after = convert_joints(hand_kps_after, source="biomech", target="mano")

    for count, kps_flat in enumerate(hand_kps_after):
        cluster2.append(kps_flat.detach().reshape(-1).cpu().numpy())

    cluster_array = np.array(cluster)
    entropy, cluster_size = diversity(cluster_array, cls_num=20)

    cluster_array_2 = np.array(cluster2)
    entropy_2, cluster_size_2 = diversity(cluster_array_2, cls_num=20)

    return entropy, cluster_size, entropy_2, cluster_size_2

def pene_depth(obj_mesh, hand_verts):
    trimesh.repair.fix_normals(obj_mesh)

    # obj_triangles = obj_mesh.vertices[obj_mesh.faces]
    # exterior = batch_mesh_contains_points(torch.from_numpy(hand_verts[None, :, :]).float(),
    #                                                    torch.from_numpy(obj_triangles)[None, :, :, :].float())
    # penetr_mask = ~exterior.squeeze(dim=0)
    penetr_mask = obj_mesh.contains(hand_verts)

    if penetr_mask.sum() == 0:
        max_depth = 0
        result_distance = np.array([])
        penetr_vert_ids = np.array([], dtype=np.int64)
    else:
        penetr_vert_ids = np.where(penetr_mask)[0]
        (result_close, result_distance, _) = trimesh.proximity.closest_point(obj_mesh, hand_verts[penetr_vert_ids])
        max_depth = result_distance.max()

    return max_depth, result_distance, penetr_vert_ids

def calculate_metrics(param_list, pool=None, metrics=[], reduction='mean'):
    for p in param_list:
        p["metrics"] = metrics

    if pool is not None:
        result_list = pool.map(parallel_calculate_metrics, param_list)
    else:
        result_list = []
        for p in tqdm(param_list):
             result_list.append(parallel_calculate_metrics(p))
        # result_list = [parallel_calculate_metrics(p) for p in param_list]
    result = {}

    if "Success Rate" in metrics:
        res = {'hand_verts': torch.stack([torch.as_tensor(p['hand_model'].vertices).float() for p in param_list], dim=0),
            'hand_joints': torch.stack([torch.as_tensor(p['hand_joints']).float() for p in param_list], dim=0),
            'obj_verts': [torch.as_tensor(p['obj_model'].vertices).float() for p in param_list],
            'obj_faces': [torch.as_tensor(p['obj_model'].faces).float() for p in param_list]
            }

        batch_idx = param_list[0]['frame_name']
        with open(osp.join('tmp', 'exp_output', f'sample_result_{batch_idx}.pkl'), 'wb') as f:
            pickle.dump(res, f)
        subprocess.run(
            ['/home/zxc417/anaconda3/envs/ugg/bin/python', '/home/zxc417/Projects/reproductions/ugg/test_success_rate.py',
             '-i', f'tmp/exp_output/sample_result_{batch_idx}.pkl', '-o', f'tmp/exp_output/success_rate_{batch_idx}.json'])
        with open(f'tmp/exp_output/success_rate_{batch_idx}.json', 'r') as f:
            success_rate = json.load(f)
        result["Success Rate"] = np.array([success_rate['6d']])

    for k in result_list[0].keys():
        result[k] = [rit[k] for rit in result_list]

    # Per-sample ratio of Intersection Volume to Contact Area, then averaged by
    # the reduction below (so it is a mean of per-sample ratios, not a ratio of
    # means). Only defined when both base metrics were computed.
    if "Intersection Volume" in result and "Contact Area" in result:
        result["IV/CA"] = [
            iv / ca if ca > 0 else 0.0
            for iv, ca in zip(result["Intersection Volume"], result["Contact Area"])
        ]

    for m in result.keys():
        if m not in ("Penetration Depth Raw", "Penetration Depth Vert IDs"):  # Keep variable-length per-vertex arrays as lists
            if reduction == 'mean':
                result[m] = np.mean(np.asarray(result[m])).item()
            elif reduction == 'sum':
                result[m] = np.sum(np.asarray(result[m])).item()
            elif reduction == "none":
                result[m] = np.asarray(result[m])
            else:
                raise ValueError(f"Unknown reduction {reduction}")
    return result


def calculate_fscore(gt, pr, th=0.01):
    """
    :param gt: (N, 3) ground truth vertices
    :param pr: (N, 3) predicted vertices
    :param th: distance threshold in meters
    """
    dist = np.sqrt(np.sum((gt[:, None, :] - pr[None, :, :]) ** 2, axis=-1))  # (N, N)
    d1 = dist.min(axis=1)  # closest pr point for each gt point
    d2 = dist.min(axis=0)  # closest gt point for each pr point
    precision = float(np.mean(d1 < th))
    recall = float(np.mean(d2 < th))
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0
    return fscore, precision, recall


def calculate_contact_area(hand_mesh, obj_mesh, threshold=0.005):
    # Closest point on hand for each object vertex
    closest_pts, dists, closest_tri_ids = trimesh.proximity.closest_point(hand_mesh, obj_mesh.vertices)
    close_mask = dists < threshold

    # Interpolate hand normals at closest points using barycentric coordinates
    hand_normals = hand_mesh.face_normals[closest_tri_ids]
    obj_normals = obj_mesh.vertex_normals

    # Contact requires normals to be opposing (dot product < 0)
    dots = np.einsum('ij,ij->i', obj_normals, hand_normals)
    contact_verts = set(np.where(close_mask & (dots < 0))[0])

    area = 0.0
    for face in obj_mesh.faces:
        if contact_verts.intersection(face):
            area += trimesh.triangles.area([obj_mesh.vertices[face]])[0]
    return area


## From FreiHand

class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds
