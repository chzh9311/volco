import numpy as np
import torch
import trimesh
import open3d as o3d
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.manifold import Isomap
from common.utils.geometry import point_set_register, get_v2v_rot
import os

def parse_hex_color(hex_color: str):
    if len(hex_color) == 6:
        hex_color = hex_color.lstrip("#")
    color = [int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
    return color

def clip_mesh_to_aabb(mesh, min_bound, max_bound, color=None):
    """
    Clip an Open3D TriangleMesh to an axis-aligned bounding box using
    Sutherland-Hodgman clipping. Triangles straddling a face of the box are
    split at the boundary, producing new vertices and triangles.

    Args:
        mesh: o3d.geometry.TriangleMesh
        min_bound: array-like (3,), lower corner of the box
        max_bound: array-like (3,), upper corner of the box
        color: optional RGB list/array to paint the result

    Returns:
        o3d.geometry.TriangleMesh with clipped geometry
    """
    min_b = np.asarray(min_bound, dtype=np.float64)
    max_b = np.asarray(max_bound, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles)

    # Each clipping plane: (axis, sign, limit)
    # sign=+1 → keep where v[axis] >= limit  (min plane)
    # sign=-1 → keep where v[axis] <= limit  (max plane)
    planes = [(ax, +1, min_b[ax]) for ax in range(3)] + \
             [(ax, -1, max_b[ax]) for ax in range(3)]

    def _intersect(p, q, ax, limit):
        """Linearly interpolate the crossing point of edge p→q at v[ax]=limit."""
        t = (limit - p[ax]) / (q[ax] - p[ax])
        return p + t * (q - p)

    def _clip_polygon(poly, ax, sign, limit):
        """Clip a convex polygon (list of 3-vectors) against one half-space."""
        if len(poly) == 0:
            return []
        out = []
        for i in range(len(poly)):
            cur, nxt = poly[i], poly[(i + 1) % len(poly)]
            cur_in = sign * (cur[ax] - limit) >= 0
            nxt_in = sign * (nxt[ax] - limit) >= 0
            if cur_in:
                out.append(cur)
            if cur_in != nxt_in:
                out.append(_intersect(cur, nxt, ax, limit))
        return out

    out_verts = []
    out_faces = []

    for tri in faces:
        polygon = [verts[tri[0]], verts[tri[1]], verts[tri[2]]]
        for (ax, sign, limit) in planes:
            polygon = _clip_polygon(polygon, ax, sign, limit)
            if not polygon:
                break
        if len(polygon) < 3:
            continue
        # Fan-triangulate the (possibly split) polygon
        base_idx = len(out_verts)
        out_verts.extend(polygon)
        for k in range(1, len(polygon) - 1):
            out_faces.append([base_idx, base_idx + k, base_idx + k + 1])

    if not out_verts:
        return o3d.geometry.TriangleMesh()

    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.array(out_verts))
    result.triangles = o3d.utility.Vector3iVector(np.array(out_faces))
    result.merge_close_vertices(1e-9)
    result.compute_vertex_normals()
    if color is not None:
        result.paint_uniform_color(color)
    return result


def o3dmesh(vert, face, color=None):
    """
    Input: vert_list, face_list: list of numpy arrays of shape [N, 3]
           color_list: list of str
           writer: Tensorboard Summary Writer
    """
    vert = o3d.utility.Vector3dVector(vert)
    face = o3d.utility.Vector3iVector(face)
    mesh = o3d.geometry.TriangleMesh(vertices=vert, triangles=face)
    cmap = plt.colormaps['hsv']
    if color is None:
        # color = np.clip(np.random.randn(3), 0, 1)
        color = cmap(np.random.rand())[:3]
    if isinstance(color, str):
        color = parse_hex_color(color)
    
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh
    # o3d.visualization.draw_geometries(meshes, mesh_show_wireframe=True)


def o3dmesh_from_trimesh(mesh, color=None):
    return o3dmesh(mesh.vertices, mesh.faces, color)


def register_and_vis_hand_parts(hpc:np.ndarray, opc:np.ndarray) -> list:
    """
    register all object pointclouds to the root & rotate all hand parts.
    To test whether the algorithm works as expected.
    :param hpc: hand point cloud <n_parts x n1 x 3>
    :param opc: object point cloud <n_parts x n2 x 3>
    :return: a list of <o3d.geometry.PointCloud>
    """
    vis_geoms = []
    part_cmap = plt.colormaps['hsv']
    for i in range(16):
        R, t = point_set_register(opc[i], opc[0])
        vis_hand_pc = hpc[i] @ R.T + t.reshape(1, 3)
        vis_hand_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vis_hand_pc))
        vis_hand_pc.paint_uniform_color(part_cmap(i / 16)[:3])
        vis_geoms.append(vis_hand_pc)

    obj_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(opc[0]))
    vis_geoms.append(obj_pc)

    return vis_geoms


def visualize_latent(obj_shape, sample_pts, pos_feat, sdf):
    """
    Visualize the predicted latent position features by isomapping reduction. Only one sample
    :param obj_shape: trimesh
    :param sample_pts: <N1 x 3>
    :param pos_feat: <N1 x D>
    :param sdf: <N1>
    """
    isomap = Isomap(n_neighbors=10, n_components=2)
    obj_mesh = o3dmesh_from_trimesh(obj_shape)
    pos_feat_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_pts))
    sdf_pc = deepcopy(pos_feat_pc)
    pos_isomap = isomap.fit_transform(pos_feat)
    up, bottom, right, left = pos_isomap[:, 0].max(), pos_isomap[:, 0].min(), pos_isomap[:, 1].max(), pos_isomap[:, 1].min()
    ## Normalize
    pos_isomap = (pos_isomap - np.array([[bottom, left]])) / np.array([[up - bottom], [right - left]])
    hsv_cmap = plt.get_cmap('hsv')
    pos_color = hsv_cmap(pos_isomap[:, 1])[:, :3]
    satr = 1 - np.abs(2 * pos_isomap[:, 0] - 1)
    pos_color[pos_isomap[:, 0] < 0.5] = pos_color[pos_isomap[:, 0] < 0.5] * satr + 1 * (1 - satr)
    pos_color[pos_isomap[:, 0] > 0.5] = pos_color[pos_isomap[:, 0] > 0.5] * satr
    hm_cmap = plt.get_cmap('inferno')

    pos_feat_pc.colors = o3d.utility.Vector3dVector(pos_color)
    sdf_pc.colors = o3d.utility.Vector3dVector(hm_cmap(sdf)[:, :3])
    sdf_pc.translate((0.4, 0, 0))
    o3d.visualization.draw_geometries([obj_mesh, pos_feat_pc, sdf_pc])


def o3d_arrow(anchor: np.ndarray, direction: np.ndarray,
              color: (np.ndarray, None) = None, scale: float = 1.0, log=False) -> o3d.geometry.TriangleMesh:
    arr_len = np.linalg.norm(direction, axis=-1)
    if log:
        arr_len = 4.12 * np.log(arr_len + 1) # 4.12 * log (9.8 + 1) = 9.8
    a = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=0.002,
        cone_height=0.002,
        cylinder_radius=0.001,
        cylinder_height=arr_len * scale,
    )
    # Transform to the right direction
    original_dire = np.array([0, 0, 1])
    R = get_v2v_rot(original_dire, direction)
    a.rotate(R, center=np.array([0, 0, 0]))
    a.translate(anchor)
    if color is not None:
        a.paint_uniform_color(color)
    a.compute_vertex_normals()

    return a

def o3d_point(pos, radius, color=None):
    pt = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    pt.translate(pos)
    if color is not None:
        pt.paint_uniform_color(color)
    pt.compute_vertex_normals()
    return pt


def extract_masked_mesh_components(hand_verts, hand_faces, vertex_mask, part_ids,
                                   create_geometries=True, uniform_color=None):
    """
    Extract masked faces and isolated vertices from a hand mesh based on a vertex mask.
    Optionally create Open3D geometries for visualization, colored by part_ids.

    Args:
        hand_verts: numpy array of shape (N, 3), hand vertex positions
        hand_faces: numpy array of shape (F, 3), hand face indices
        vertex_mask: numpy array of shape (N,), boolean mask indicating which vertices are active
        part_ids: numpy array of shape (N,), part IDs for each vertex (used for coloring with hsv colormap)
        create_geometries: bool, if True, create Open3D geometries for the mesh and isolated points
        uniform_color: optional RGB tuple/array of shape (3,) in [0, 1]. If set, all vertices are
            painted this color instead of the part-conditioned HSV colormap.

    Returns:
        If create_geometries is False:
            masked_faces: numpy array of shape (F', 3), faces where all vertices are masked
            vertices_in_faces: numpy array of vertex indices that are part of masked faces
            isolated_vert_indices: numpy array of vertex indices that are masked but not in any face

        If create_geometries is True:
            geometries: list of Open3D geometries (mesh and/or point cloud), colored by part_ids
    """
    # Find faces where all vertices are masked
    face_mask = vertex_mask[hand_faces].all(axis=1)  # (F,) boolean array
    masked_faces = hand_faces[face_mask]

    # Find vertices that are part of masked faces
    vertices_in_faces = np.unique(masked_faces.flatten()) if len(masked_faces) > 0 else np.array([])

    # Find isolated vertices (masked but not in any face)
    masked_vert_indices = np.where(vertex_mask)[0]
    isolated_vert_indices = np.setdiff1d(masked_vert_indices, vertices_in_faces)

    if not create_geometries:
        return masked_faces, vertices_in_faces, isolated_vert_indices

    # Create Open3D geometries
    geometries = []

    # Color vertices: uniform color if specified, otherwise part-conditioned HSV
    if uniform_color is not None:
        vertex_colors = np.tile(np.array(uniform_color, dtype=np.float64), (len(hand_verts), 1))
    else:
        part_cmap = plt.colormaps['hsv']
        vertex_colors = part_cmap(part_ids / 16)[:, :3]

    # Create mesh with only masked faces
    if len(masked_faces) > 0:
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
        hand_mesh.triangles = o3d.utility.Vector3iVector(masked_faces)
        hand_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        hand_mesh.compute_vertex_normals()
        geometries.append(hand_mesh)

    # Create point cloud for isolated vertices
    if len(isolated_vert_indices) > 0:
        isolated_pcd = o3d.geometry.PointCloud()
        isolated_pcd.points = o3d.utility.Vector3dVector(hand_verts[isolated_vert_indices])
        isolated_pcd.colors = o3d.utility.Vector3dVector(vertex_colors[isolated_vert_indices])
        geometries.append(isolated_pcd)

    return geometries

def geom_to_img_o3d(vis_geoms, w, h, scale=1, concat_axis=1):
    """
    Render a list of Open3D geometries to an image by opening an O3D window,
    setting the camera to each of 4 evenly-spaced azimuth views, capturing a
    screenshot, then closing. The 4 images are concatenated along concat_axis.

    vis_geoms: list of o3d geometry objects or dicts with key 'geometry'
    Returns: float32 numpy array in [0, 1], shape (H, W*4, 3) or (H*4, W, 3)
    """
    # Unwrap dicts the same way geom_to_img does
    raw_geoms = []
    for g in vis_geoms:
        raw_geoms.append(g['geometry'] if isinstance(g, dict) else g)

    # Camera azimuths matching geom_to_img: angle = pi*i/2 + pi/6
    elev = 30.0
    azimuths = [np.degrees(np.pi * i / 2 + np.pi / 6) for i in range(4)]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=w, height=h)

    for g in raw_geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.light_on = True

    # First poll lets O3D initialise the scene and run its internal reset_view_point
    vis.poll_events()
    vis.update_renderer()

    result_imgs = []
    for azim in azimuths:
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        front = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.sin(elev_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
        ])

        ctr = vis.get_view_control()
        ctr.set_front(front.tolist())
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_zoom(scale)

        # Two render cycles: first applies the camera, second stabilises it
        vis.poll_events()
        vis.update_renderer()
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(do_render=True)
        result_imgs.append(np.asarray(img).copy())  # copy before next camera change

    vis.destroy_window()

    return np.concatenate(result_imgs, axis=concat_axis)


def geom_to_img(vis_geoms, w, h, scale=0.07, half_range=None, concat_axis=1):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Collect all meshes and point clouds
    all_vertices = []
    mesh_data = []
    pointcloud_data = []

    for geom in vis_geoms:
        if type(geom) is dict:
            o3d_geom = geom['geometry']
            geom_alpha = geom.get('alpha', 0.95)  # Get custom alpha or use default
        else:
            o3d_geom = geom
            geom_alpha = 0.95  # Default alpha

        if isinstance(o3d_geom, o3d.geometry.TriangleMesh):
            vertices = np.asarray(o3d_geom.vertices)
            faces = np.asarray(o3d_geom.triangles)

            if len(vertices) == 0 or len(faces) == 0:
                continue

            # Get vertex colors if available
            if o3d_geom.has_vertex_colors():
                colors = np.asarray(o3d_geom.vertex_colors)
            else:
                colors = np.ones((len(vertices), 3)) * 0.5  # Default gray

            all_vertices.append(vertices)
            mesh_data.append({'vertices': vertices, 'faces': faces, 'colors': colors, 'alpha': geom_alpha})

        elif isinstance(o3d_geom, o3d.geometry.PointCloud):
            points = np.asarray(o3d_geom.points)

            if len(points) == 0:
                continue

            # Get point colors if available
            if o3d_geom.has_colors():
                colors = np.asarray(o3d_geom.colors)
            else:
                colors = np.ones((len(points), 3)) * 0.5  # Default gray

            per_point_sizes = geom.get('point_sizes') if type(geom) is dict else None
            all_vertices.append(points)
            pointcloud_data.append({'points': points, 'colors': colors, 'point_sizes': per_point_sizes})

        elif isinstance(o3d_geom, o3d.geometry.LineSet):
            pts = np.asarray(o3d_geom.points)
            lines = np.asarray(o3d_geom.lines)
            if len(pts) == 0 or len(lines) == 0:
                continue
            line_colors = np.asarray(o3d_geom.colors) if o3d_geom.has_colors() else np.zeros((len(lines), 3))
            all_vertices.append(pts)
            pointcloud_data.append({'_lineset': True, 'points': pts, 'lines': lines, 'colors': line_colors})

    if len(all_vertices) == 0:
        print("Warning: No valid geometries to render")
        return np.ones((h, w * 4, 3))

    # Compute scene bounds
    all_vertices = np.vstack(all_vertices)
    center = np.array([0.0, 0.0, 0.0])  # Center on origin instead of mean
    extent = np.ptp(all_vertices, axis=0)
    # Tight framing for 80% coverage - use smaller multiplier
    if half_range is None:
        max_extent = np.max(extent)
        half_range = max_extent * scale  # Even tighter for better zoom

    result_imgs = []
    for i in range(4):
        # Create figure
        dpi = 100
        fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Set camera position for each view
        angle = np.pi * i / 2 + np.pi / 6  # 30-degree offset for better view
        elev = 30  # Slight elevation for better 3D view
        azim = np.degrees(angle)
        ax.view_init(elev=elev, azim=azim)

        # Render meshes grouped by alpha value for proper transparency
        # Group meshes by alpha value
        alpha_groups = {}
        for mesh in mesh_data:
            alpha = mesh['alpha']
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(mesh)

        # Render each alpha group separately (opaque first, then transparent)
        for alpha in sorted(alpha_groups.keys(), reverse=True):
            all_faces = []
            all_face_colors = []

            for mesh in alpha_groups[alpha]:
                verts = mesh['vertices']
                faces = mesh['faces']
                colors = mesh['colors']

                # Create face colors (average vertex colors for each face)
                face_colors = colors[faces].mean(axis=1)

                # Add each face
                for face_idx, face in enumerate(faces):
                    face_verts = [verts[face[0]], verts[face[1]], verts[face[2]]]
                    all_faces.append(face_verts)
                    all_face_colors.append(face_colors[face_idx])

            # Create polygon collection for this alpha group
            if len(all_faces) > 0:
                collection = Poly3DCollection(all_faces, facecolors=all_face_colors,
                                             edgecolors='none', alpha=alpha, linewidths=0,
                                             zsort='average')  # Enable depth sorting
                ax.add_collection3d(collection)

        # Render all point clouds and linesets
        for pc in pointcloud_data:
            points = pc['points']
            colors = pc['colors']
            if pc.get('_lineset'):
                lines = pc['lines']
                for li, (i, j) in enumerate(lines):
                    seg = np.array([points[i], points[j]])
                    c = colors[li] if li < len(colors) else [0, 1, 0]
                    ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=c, linewidth=1.0)
            else:
                s = pc['point_sizes'] if pc['point_sizes'] is not None else point_size
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=colors, s=s, alpha=0.8, depthshade=True)

        # Set tight limits around the object
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)

        # Force equal aspect ratio and tight layout
        ax.set_box_aspect([1, 1, 1])
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        # Additional zoom control - set camera distance

        # Render to array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        result_imgs.append(img_array[:, :, :3] / 255.0)

        plt.close(fig)

    ret_img = np.concatenate(result_imgs, axis=concat_axis)
    return ret_img


def vis_nn_bps(hand_mesh, bps, nn_idx):
    """
    visualize the nearest point in the bps to the corresonding hand vertex.
    For all points in BPS that are in correspondence with the same hand vertex,
    selected only the nearest one. All seleted points are visualized as a point cloud, together with the hand mesh.

    :param hand_mesh: trimesh.Trimesh of 778 vertices
    :param bps: N^3 x 3 array of bps grid points
    :param nn_idx: N^3 array of nearest hand vertex indices for each bps point
    """
    # Convert hand mesh to open3d format
    o3d_hand_mesh = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.8, 0.8])

    # For each hand vertex, find the nearest BPS point
    num_hand_verts = hand_mesh.vertices.shape[0]
    selected_bps_points = []
    corresponding_hand_verts = []

    for v_idx in range(num_hand_verts):
        # Find all BPS points that correspond to this hand vertex
        mask = (nn_idx == v_idx)
        if np.sum(mask) == 0:
            continue

        # Get the BPS points corresponding to this vertex
        corresponding_bps = bps[mask]  # M x 3

        # Calculate distances from these BPS points to the hand vertex
        hand_vert = hand_mesh.vertices[v_idx]  # 3
        distances = np.linalg.norm(corresponding_bps - hand_vert, axis=1)

        # Select the nearest BPS point
        nearest_idx = np.argmin(distances)
        selected_bps_points.append(corresponding_bps[nearest_idx])
        corresponding_hand_verts.append(hand_vert)

    # Convert selected BPS points to open3d point cloud
    geometries = [o3d_hand_mesh]

    if len(selected_bps_points) > 0:
        selected_bps_points = np.array(selected_bps_points)
        corresponding_hand_verts = np.array(corresponding_hand_verts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(selected_bps_points)
        # Color the points red for visibility
        pcd.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(pcd)

        # Create arrows from BPS points to hand vertices
        for bps_pt, hand_vert in zip(selected_bps_points, corresponding_hand_verts):
            # Create line segment from BPS point to hand vertex
            points = np.array([bps_pt, hand_vert])
            lines = np.array([[0, 1]])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0.0, 1.0, 0.0])  # Green arrows
            geometries.append(line_set)

    # Visualize
    o3d.visualization.draw_geometries(geometries,
                                     window_name="Hand Mesh with Nearest BPS Points and Arrows",
                                     width=1024,
                                     height=768)


def visualize_local_grid_with_hand(local_grid, hand_verts, hand_faces, hand_cse, kernel_size, grid_scale,
                                            contact_point=None, obj_mesh=None):
    """
    Visualize local grid with object mesh, hand mesh, and grid points.

    Args:
        local_grid: numpy array of shape (kernel_size, kernel_size, kernel_size, C)
                   where C >= 18 (SDF, contact, CSE[16])
        contact_point: numpy array of shape (3,), the center of the grid
        obj_mesh: trimesh mesh of the object, or None to skip object mesh visualization
        hand_verts: numpy array of shape (778, 3), hand vertices
        hand_faces: numpy array of shape (N, 3), hand face indices
        hand_cse: numpy array of shape (778, 16), hand contact surface embeddings
        kernel_size: int, size of the cubic grid
        grid_scale: float, scale of the grid

    Returns:
        list: List of open3d geometries to be visualized
    """
    # Generate normalized grid coordinates
    indices = np.array([(i, j, k) for i in range(kernel_size)
                        for j in range(kernel_size)
                        for k in range(kernel_size)])

    # Convert to coordinates in [-1, 1] range
    coords = 2 * indices - (kernel_size - 1)
    coords = coords / (kernel_size - 1)

    # Scale and translate to world coordinates
    if contact_point is None:
        contact_point = np.array([0.0, 0.0, 0.0])
    grid_points = contact_point[None, :] + coords * grid_scale

    # Extract SDF values (first channel)
    sdf_values = local_grid[:, :, :, 0].flatten()

    # Create point cloud for grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_points)

    # Color points based on SDF values
    sdf_min, sdf_max = sdf_values.min(), sdf_values.max()
    colors = np.zeros((len(sdf_values), 3))
    for i, sdf_val in enumerate(sdf_values):
        if sdf_val < 0:  # Inside object - blue shades
            intensity = min(abs(sdf_val) / abs(sdf_min) if sdf_min < 0 else 1, 1)
            colors[i] = [1-intensity, 1-intensity, 1]  # Blue
        else:  # Outside object - red shades
            intensity = min(sdf_val / sdf_max if sdf_max > 0 else 1, 1)
            colors[i] = [1, 1-intensity, 1-intensity]  # Red

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create contact point marker (larger, yellow)
    contact_pcd = o3d.geometry.PointCloud()
    contact_pcd.points = o3d.utility.Vector3dVector([contact_point])
    contact_pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow

    # Create bounding box for the grid
    bbox_points = np.array([
        contact_point + grid_scale * np.array([-1, -1, -1]),
        contact_point + grid_scale * np.array([1, -1, -1]),
        contact_point + grid_scale * np.array([1, 1, -1]),
        contact_point + grid_scale * np.array([-1, 1, -1]),
        contact_point + grid_scale * np.array([-1, -1, 1]),
        contact_point + grid_scale * np.array([1, -1, 1]),
        contact_point + grid_scale * np.array([1, 1, 1]),
        contact_point + grid_scale * np.array([-1, 1, 1]),
    ])

    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_points)
    line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])  # Green bbox

    # Convert object mesh to open3d (if provided)
    obj_o3d_mesh = None
    if obj_mesh is not None:
        obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh)

    # Create hand mesh
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.paint_uniform_color([0.8, 0.6, 0.4])  # Skin color
    hand_mesh.compute_vertex_normals()

    # ==================== DUPLICATE GEOMETRIES WITH OFFSET ====================
    # Calculate offset along x-axis (make it large enough to separate the scenes)
    if obj_mesh is not None:
        bbox = obj_mesh.bounds
        offset_distance = (bbox[1][0] - bbox[0][0]) * 2  # 2x the object width
    else:
        # Use hand bounding box to calculate offset if no object mesh
        hand_bbox_min = hand_verts.min(axis=0)
        hand_bbox_max = hand_verts.max(axis=0)
        offset_distance = (hand_bbox_max[0] - hand_bbox_min[0]) * 2
    offset = np.array([offset_distance, 0, 0])

    # Duplicate and offset grid points
    grid_points_offset = grid_points + offset

    # Extract contact likelihood values (channel 1)
    contact_values = local_grid[:, :, :, 1].flatten()

    # Create point cloud for offset grid with contact likelihood coloring
    pcd_offset = o3d.geometry.PointCloud()
    pcd_offset.points = o3d.utility.Vector3dVector(grid_points_offset)

    # Color using inferno colormap (blue=0 to red=1)
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('inferno')
    contact_colors = np.array([cmap(val)[:3] for val in contact_values])  # Get RGB, ignore alpha
    pcd_offset.colors = o3d.utility.Vector3dVector(contact_colors)

    # Duplicate contact point marker
    contact_pcd_offset = o3d.geometry.PointCloud()
    contact_pcd_offset.points = o3d.utility.Vector3dVector([contact_point + offset])
    contact_pcd_offset.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow

    # Duplicate bounding box
    bbox_points_offset = bbox_points + offset
    line_set_offset = o3d.geometry.LineSet()
    line_set_offset.points = o3d.utility.Vector3dVector(bbox_points_offset)
    line_set_offset.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set_offset.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])

    # Duplicate object mesh (if provided)
    obj_o3d_mesh_offset = None
    if obj_mesh is not None:
        obj_o3d_mesh_offset = o3dmesh_from_trimesh(obj_mesh)
        obj_o3d_mesh_offset.translate(offset)

    # Duplicate hand mesh
    hand_mesh_offset = o3d.geometry.TriangleMesh()
    hand_verts_offset = hand_verts + offset
    hand_mesh_offset.vertices = o3d.utility.Vector3dVector(hand_verts_offset)
    hand_mesh_offset.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh_offset.paint_uniform_color([0.8, 0.6, 0.4])
    hand_mesh_offset.compute_vertex_normals()

    # ==================== CREATE CORRESPONDENCE LINES ====================
    # Extract predicted CSE from local_grid (indices 2:18, which is 16 dimensions)
    predicted_cse = local_grid[:, :, :, 2:].reshape(-1, hand_cse.shape[1])  # (kernel_size^3, 16)

    # Compute distance matrix between predicted CSE and hand CSE
    # predicted_cse: (K^3, 16), hand_cse: (778, 16)
    dist_matrix = np.linalg.norm(predicted_cse[:, None, :] - hand_cse[None, :, :], axis=2)  # (K^3, 778)
    nearest_hand_idx = np.argmin(dist_matrix, axis=1)  # (K^3,) - index of nearest hand vertex for each grid point

    # Filter to only high contact likelihood points (> 0.5)
    high_contact_mask = contact_values > 0.5
    high_contact_grid_points = grid_points_offset[high_contact_mask]
    high_contact_nearest_idx = nearest_hand_idx[high_contact_mask]

    # Create lines from high-contact grid points to their nearest hand vertices
    line_points = []
    line_indices = []
    for i, grid_point in enumerate(high_contact_grid_points):
        hand_vert_idx = high_contact_nearest_idx[i]
        hand_point = hand_verts_offset[hand_vert_idx]

        # Add both points
        line_points.append(grid_point)
        line_points.append(hand_point)

        # Add line index
        line_indices.append([len(line_points) - 2, len(line_points) - 1])

    # Create line set for correspondences
    correspondence_lines = o3d.geometry.LineSet()
    if len(line_points) > 0:
        correspondence_lines.points = o3d.utility.Vector3dVector(line_points)
        correspondence_lines.lines = o3d.utility.Vector2iVector(line_indices)
        correspondence_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in line_indices])  # Green lines

    # Print information
    print(f"\nVisualizing Local Grid")
    print(f"Contact point: {contact_point}")
    print(f"Grid scale: {grid_scale}")
    print(f"SDF values range: [{sdf_min:.4f}, {sdf_max:.4f}]")
    print(f"Contact likelihood range: [{contact_values.min():.4f}, {contact_values.max():.4f}]")
    print(f"Grid size: {kernel_size}x{kernel_size}x{kernel_size} = {kernel_size**3} points")
    print(f"Hand vertices: {len(hand_verts)}")
    print(f"Hand faces: {len(hand_faces)}")
    print(f"High contact points (>0.5): {high_contact_mask.sum()}")
    print(f"Correspondence lines: {len(line_indices)}")
    print(f"\nLeft scene - Original (SDF-colored):")
    print(f"  Blue: Negative SDF (inside object)")
    print(f"  Red: Positive SDF (outside object)")
    print(f"  Yellow: Contact point (grid center)")
    print(f"  Green: Grid bounding box")
    print(f"  Skin color: Hand mesh")
    print(f"  Gray: Object mesh")
    print(f"\nRight scene - Contact & Correspondence:")
    print(f"  Inferno colormap: Blue (no contact) → Red (high contact)")
    print(f"  Green lines: Correspondence to nearest hand vertex (CSE-based)")

    # Collect all geometries
    geometries = [
        # Original scene
        pcd, contact_pcd, line_set, hand_mesh,
    ]

    # Add object mesh if provided
    if obj_o3d_mesh is not None:
        geometries.insert(0, obj_o3d_mesh)

    # Add offset scene with contact likelihood
    geometries.extend([
        pcd_offset, contact_pcd_offset, line_set_offset, hand_mesh_offset
    ])

    # Add offset object mesh if provided
    if obj_o3d_mesh_offset is not None:
        geometries.insert(5 if obj_o3d_mesh is not None else 4, obj_o3d_mesh_offset)

    # Add correspondence lines if any exist
    if len(line_points) > 0:
        geometries.append(correspondence_lines)

    return geometries


def visualize_local_grid(msdf, kernel_size, point_idx, obj_mesh):
    """
    Visualize one local grid of MSDF with the object.

    Args:
        msdf: numpy array of shape (n_points, kernel_size**3 + 3 + 1)
        kernel_size: size of the cubic grid
        point_idx: which point's grid to visualize
        obj_mesh: trimesh mesh of the object
    """
    # Extract data for the selected point
    point_data = msdf[point_idx]
    sdf_values = point_data[:kernel_size**3]
    center = point_data[kernel_size**3:kernel_size**3+3]
    scale = point_data[-1]

    # Generate grid points (same as get_grid_points function)
    # Grid in range [-1, 1] for each dimension
    indices = np.array([(i, j, k) for i in range(kernel_size)
                        for j in range(kernel_size)
                        for k in range(kernel_size)])

    # Convert to coordinates in [-1, 1] range
    coords = 2 * indices - (kernel_size - 1)
    coords = coords / (kernel_size - 1)

    # Scale and translate to world coordinates
    grid_points = center + coords * scale

    # Create point cloud for grid points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_points)

    # Color points based on SDF values
    sdf_min, sdf_max = sdf_values.min(), sdf_values.max()

    # Use a colormap: blue (negative/inside) -> white (zero) -> red (positive/outside)
    # Map to RGB where blue=negative, red=positive
    colors = np.zeros((len(sdf_values), 3))
    for i, sdf_val in enumerate(sdf_values):
        if sdf_val < 0:  # Inside object - blue shades
            intensity = min(abs(sdf_val) / abs(sdf_min) if sdf_min < 0 else 1, 1)
            colors[i] = [1-intensity, 1-intensity, 1]  # Blue
        else:  # Outside object - red shades
            intensity = min(sdf_val / sdf_max if sdf_max > 0 else 1, 1)
            colors[i] = [1, 1-intensity, 1-intensity]  # Red

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a bounding box to show the grid extent
    bbox_points = np.array([
        center + scale * np.array([-1, -1, -1]),
        center + scale * np.array([1, -1, -1]),
        center + scale * np.array([1, 1, -1]),
        center + scale * np.array([-1, 1, -1]),
        center + scale * np.array([-1, -1, 1]),
        center + scale * np.array([1, -1, 1]),
        center + scale * np.array([1, 1, 1]),
        center + scale * np.array([-1, 1, 1]),
    ])

    bbox_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_points)
    line_set.lines = o3d.utility.Vector2iVector(bbox_lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in bbox_lines])  # Green bbox

    # Create center point marker for current grid (larger, yellow)
    center_pcd = o3d.geometry.PointCloud()
    center_pcd.points = o3d.utility.Vector3dVector([center])
    center_pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # Yellow

    # Create point cloud for ALL sampled points (all grid centers)
    all_centers = msdf[:, kernel_size**3:kernel_size**3+3]
    all_centers_pcd = o3d.geometry.PointCloud()
    all_centers_pcd.points = o3d.utility.Vector3dVector(all_centers)
    # Color all centers in cyan, except the current one
    all_centers_colors = np.tile([0, 1, 1], (len(all_centers), 1))  # Cyan
    all_centers_pcd.colors = o3d.utility.Vector3dVector(all_centers_colors)

    # Convert object mesh to open3d
    obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh)

    # Print information
    print(f"\nVisualizing MSDF Grid for Point {point_idx}")
    print(f"Center: {center}")
    print(f"Scale: {scale}")
    print(f"SDF values range: [{sdf_min:.4f}, {sdf_max:.4f}]")
    print(f"Grid size: {kernel_size}x{kernel_size}x{kernel_size} = {kernel_size**3} points")
    print(f"Total number of sampled points: {len(all_centers)}")
    print(f"\nColor scheme:")
    print(f"  Blue: Negative SDF (inside object)")
    print(f"  Red: Positive SDF (outside object)")
    print(f"  Yellow: Current grid center (point {point_idx})")
    print(f"  Cyan: All other sampled points (grid centers)")
    print(f"  Green: Grid bounding box")

    # Visualize
    o3d.visualization.draw_geometries(
        [obj_o3d_mesh, pcd, center_pcd, all_centers_pcd, line_set],
        window_name=f"MSDF Grid Visualization - Point {point_idx}"
    )


def visualize_grid_contact(contact_pts, pt_contact, grid_scale, obj_mesh, w, h, bbox_alpha=0.2):
    bboxes = create_bbox_geomtries(contact_pts, grid_scale, pt_contact, alpha=bbox_alpha)
    obj_geom = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])
    # Wrap bboxes with alpha values
    bbox_geoms = [{'geometry': bbox, 'alpha': bbox_alpha} for bbox in bboxes]
    vis_geoms = [obj_geom] + bbox_geoms
    img = geom_to_img(vis_geoms, w=w, h=h, scale=0.5)
    return img, vis_geoms


def create_bbox_geomtries(msdf_center, grid_scale, contact=None, alpha=0.3):
    # Create bounding boxes for each MSDF center using semi-transparent cubes
    bbox_geometries = []
    inferno_cmap = plt.colormaps['inferno']
    for i, center in enumerate(msdf_center):
        # Create a box mesh centered at origin with size 2*grid_scale
        box = o3d.geometry.TriangleMesh.create_box(
            width=2*grid_scale,
            height=2*grid_scale,
            depth=2*grid_scale
        )

        # Translate box to be centered at the contact point
        # create_box creates box from (0,0,0) to (width, height, depth), so we need to center it
        box.translate(center - grid_scale * np.array([1, 1, 1]))

        # Set color based on contact value using inferno colormap (0->blue, 1->red)
        if contact is None:
            box_color = [0, 0, 1]
        else:
            # Clamp value to [0, 1] and apply inferno colormap
            value = np.clip(contact[i], 0, 1)
            box_color = inferno_cmap(value)[:3]
        box.paint_uniform_color(box_color)

        bbox_geometries.append(box)

    return bbox_geometries


def visualize_recon_hand_w_object(hand_verts, hand_verts_mask, hand_faces, obj_mesh, part_ids, msdf_center=None, grid_scale=None, h=500, w=500,
                                  concat_axis=1):
    masked_hand_geometries = extract_masked_mesh_components(
        hand_verts=hand_verts,
        hand_faces=hand_faces,
        vertex_mask=hand_verts_mask,
        part_ids=part_ids,
        create_geometries=True,
    )
    obj_o3d_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

    # bbox_geometries = create_bbox_geomtries(msdf_center, grid_scale)

    vis_geoms = masked_hand_geometries + [obj_o3d_mesh] # + bbox_geometries
    img = geom_to_img_o3d(vis_geoms, w=w, h=h, scale=0.9, concat_axis=concat_axis)
    return img, vis_geoms


def vis_contact(obj_mesh, hand_mesh, obj_pts, obj_pt_mask):
    hand_mesh = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.7, 0.6])
    obj_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

    # Convert obj_pts to numpy if it's a tensor
    if isinstance(obj_pts, torch.Tensor):
        obj_pts_np = obj_pts.cpu().numpy()
    else:
        obj_pts_np = obj_pts

    # Convert obj_pt_mask to numpy if it's a tensor
    if isinstance(obj_pt_mask, torch.Tensor):
        obj_pt_mask_np = obj_pt_mask.cpu().numpy()
    else:
        obj_pt_mask_np = obj_pt_mask

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_pts_np)

    # Color points: red for masked (True), blue for rest (False)
    colors = np.zeros((len(obj_pts_np), 3))
    colors[obj_pt_mask_np] = [1.0, 0.0, 0.0]  # Red for selected points
    colors[~obj_pt_mask_np] = [0.0, 0.0, 1.0]  # Blue for rest
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([hand_mesh, obj_mesh, pcd])