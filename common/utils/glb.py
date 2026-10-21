"""
GLB export of hand-object geometries for web viewing (see scripts/glb_viewer.py).
Colours follow the project: hand #F29A8D, object #6293A8, contact with the 'cool' colormap,
hand parts / correspondence with hsv(part_id / 16).
"""
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from matplotlib import pyplot as plt

from common.utils.vis import extract_masked_mesh_components, parse_hex_color

HAND_RGB = np.array(parse_hex_color("#F29A8D"))  # same as HandObject.get_vis_geoms
OBJ_RGB = np.array(parse_hex_color("#6293A8"))
GREY_RGB = np.array([0.7, 0.7, 0.7])  # object under contact / reconstruction overlays
NEUTRAL_RGB = np.array([0.85, 0.85, 0.85])  # correspondence colour where there is no contact


def pbr(rgba):
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.8,
        alphaMode='BLEND' if rgba[3] < 1 else 'OPAQUE', doubleSided=True)


def solid_mesh(verts, faces, rgb, alpha=1.0):
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=pbr([*rgb, alpha]))
    return mesh


def vcolor_mesh(verts, faces, vert_rgb, alpha=1.0):
    """Per-vertex colours (COLOR_0) on top of a white non-metallic material."""
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=pbr([1.0, 1.0, 1.0, alpha]))
    rgba = np.concatenate([np.clip(vert_rgb, 0, 1), np.ones((len(vert_rgb), 1))], axis=1)
    mesh.visual.vertex_attributes['color'] = (rgba * 255).astype(np.uint8)
    return mesh


def instanced(template, centers, rgb, alpha=1.0):
    """Copy a small template mesh to every center, coloured per instance."""
    nv = len(template.vertices)
    verts = (centers[:, None] + template.vertices[None]).reshape(-1, 3)
    faces = (template.faces[None] + nv * np.arange(len(centers))[:, None, None]).reshape(-1, 3)
    return vcolor_mesh(verts, faces, np.repeat(rgb, nv, axis=0), alpha=alpha)


def point_glyphs(pts, rgb, radius):
    """Point cloud as tiny octahedra (glTF POINTS render at 1px in most web viewers)."""
    octa = trimesh.Trimesh(
        vertices=radius * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float),
        faces=[[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]],
        process=False)
    return instanced(octa, pts, rgb)


def boxes(centers, half, rgb, alpha):
    box = trimesh.creation.box(extents=[2 * half] * 3)
    box.unmerge_vertices()  # flat-shaded faces
    return instanced(box, centers, rgb, alpha=alpha)


def save_glb(path, named_geoms):
    os.makedirs(osp.dirname(path) or '.', exist_ok=True)
    scene = trimesh.Scene()
    for name, geom in named_geoms.items():
        scene.add_geometry(geom, geom_name=name, node_name=name)
    scene.export(path, include_normals=True)


def surface_values(mesh, sample_pts, sample_vals):
    """Assign each mesh vertex the value of its nearest sampled point."""
    from scipy.spatial import cKDTree
    _, nn_idx = cKDTree(sample_pts).query(np.asarray(mesh.vertices))
    return sample_vals[nn_idx]


def contact_rgb(contact):
    return plt.colormaps['cool'](np.clip(contact, 0, 1))[:, :3]


def part_rgb(part_ids):
    return plt.colormaps['hsv'](part_ids / 16)[:, :3]


def mask_by_contact(rgb, contact):
    c = np.clip(contact, 0, 1)[:, None]
    return c * rgb + (1 - c) * NEUTRAL_RGB


def cse_to_part(cse, hand_emb, hand_part_ids, chunk=16384):
    """Decode CSE embeddings to the part of the nearest hand vertex in embedding space."""
    vert_idx = torch.cat([torch.cdist(c, hand_emb).argmin(dim=-1) for c in cse.split(chunk)])
    return hand_part_ids[vert_idx.cpu().numpy()]


def grasp_geoms(obj_mesh, hand_verts, closed_faces):
    return {'object': solid_mesh(obj_mesh.vertices, obj_mesh.faces, OBJ_RGB),
            'hand': solid_mesh(hand_verts, closed_faces, HAND_RGB)}


def recon_hand_geoms(obj_mesh, recon_verts, recon_mask, mano_faces, hand_part_ids, pt_radius=0.001):
    """Local hand reconstruction: masked hand mesh + isolated verts, part-coloured, over a grey object."""
    masked_faces, _, isolated = extract_masked_mesh_components(
        recon_verts, mano_faces, recon_mask, hand_part_ids, create_geometries=False)
    geoms = {'object': solid_mesh(obj_mesh.vertices, obj_mesh.faces, GREY_RGB)}
    if len(masked_faces) > 0:
        geoms['hand'] = vcolor_mesh(recon_verts, masked_faces, part_rgb(hand_part_ids))
    if len(isolated) > 0:
        geoms['hand_points'] = point_glyphs(recon_verts[isolated], part_rgb(hand_part_ids[isolated]), pt_radius)
    return geoms


def volco_grid_geoms(grid_centers, grid_pts, grid_contact, grid_cse, hand_emb, hand_part_ids, grid_scale,
                     contact_th=0.5, pt_radius=0.001, bbox_alpha=0.25):
    """
    Volco local-grid maps, keeping only grids whose maximum contact exceeds contact_th.
    :param grid_centers: (N, 3); grid_pts: (N*K^3, 3) numpy; grid_contact: (N, K^3) numpy; grid_cse: (N*K^3, D) tensor
    :return: {'grids': boxes coloured by max contact, 'contact_pts': points coloured by contact,
              'cse_pts': points coloured by the CSE-decoded hand part (faded by contact)}
    """
    max_contact = grid_contact.max(axis=-1)
    keep = max_contact > contact_th
    K3 = grid_contact.shape[1]
    grid_centers, max_contact, grid_contact = grid_centers[keep], max_contact[keep], grid_contact[keep]
    grid_pts = grid_pts.reshape(-1, K3, 3)[keep].reshape(-1, 3)
    grid_cse = grid_cse.reshape(-1, K3, grid_cse.shape[-1])[torch.as_tensor(keep, device=grid_cse.device)]
    pt_contact = grid_contact.reshape(-1)
    pt_parts = cse_to_part(grid_cse.reshape(-1, grid_cse.shape[-1]), hand_emb, hand_part_ids)
    return {'grids': boxes(grid_centers, grid_scale, contact_rgb(max_contact), alpha=bbox_alpha),
            'contact_pts': point_glyphs(grid_pts, contact_rgb(pt_contact), pt_radius),
            'cse_pts': point_glyphs(grid_pts, mask_by_contact(part_rgb(pt_parts), pt_contact), pt_radius)}
