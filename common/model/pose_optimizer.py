import numpy as np
import torch
import torch.nn.functional as F
from copy import copy

from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_axis_angle
from manotorch.anchorlayer import AnchorLayer
from common.utils.geometry import (
        cp_match,
        cp_match_Ronly,
        calculate_contact_capsule,
        calculate_penetration_cost,
        sdf_to_contact
    )


def optimize_pose_contactopt(mano_layer, obj_verts, obj_normals, obj_contact_target, obj_partition, n_iter=1200, lr=0.002, w_cont_obj=1,
                  save_history=False, w_cont_asym=2, caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False,
                  contact_norm_method=0, w_pen_cost=600, pen_it=0, hand_cse=None, partition_type='part', is_thin=None):
    """Runs differentiable optimization to align the hand with the target contact map.
    Minimizes the loss between ground truth contact and contact calculated with DiffContact"""
    batch_size = obj_contact_target.shape[0]
    device = obj_contact_target.device

    global_pose = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=obj_verts.dtype, device=obj_verts.device)
    
    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=obj_verts.dtype, device=obj_verts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=obj_verts.dtype, device=obj_verts.device)
    # opt_vector = torch.zeros((batch_size, ncomps + 16), device=device)   # 3 hand rot, 10 hand shape, 3 hand trans
    # opt_vector.requires_grad = True

    ## Initialization:
    hand_verts, hand_joints, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1),
                                            th_betas=mano_shape, th_trans=mano_trans)
    root = hand_joints[:, 0:1]
    if partition_type == 'part':
        anchor_layer = AnchorLayer("/home/zxc417/Downloads/Lib/manotorch/assets/anchor").to(device)
        part_anchor_idx = [19, 7, 8, 9, 13, 14, 15, 27, 28, 29, 20, 21, 22, 0, 1, 2]
        init_anchors = anchor_layer(hand_verts)[:, part_anchor_idx, :]  # (B, 16, 3)
        part_weight = torch.stack([torch.sum(obj_contact_target * (obj_partition == i).float(), dim=1) for i in range(16)], dim=1) # (B, 16)
        part_weight[:, 0] = 0
        part_centers = [torch.sum(obj_verts * obj_contact_target.unsqueeze(-1) * (obj_partition == i).float().unsqueeze(-1), dim=1)
                        / (torch.sum(obj_contact_target * (obj_partition == i).float(), dim=1, keepdim=True) + 1e-8) for i in range(16)]
        part_centers = torch.stack(part_centers, dim=1)  # (B, 16, 3)
        Rs, ts = cp_match(init_anchors - root, part_centers - root, part_weight.unsqueeze(-1))
    elif partition_type == 'cse':
        Wverts = hand_cse.emb2Wvert(obj_partition) # (B, N, 778)
        target_hand_pts0 = Wverts @ hand_verts # (B, N, 3)
        Rs, ts = cp_match(target_hand_pts0 - root, obj_verts - root, obj_contact_target.unsqueeze(-1))
    else:
        ## No initialization, start from mean pose
        Rs = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        ts = torch.zeros((batch_size, 3), device=device)

    global_pose = matrix_to_axis_angle(Rs).detach()
    mano_trans = ts.detach()

    glob_pose0 = copy(global_pose).detach()
    mano_trans0 = copy(mano_trans).detach()
    init_pose = [glob_pose0, mano_trans0]

    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True 

    optimizer = torch.optim.Adam([global_pose, mano_pose, mano_shape, mano_trans], lr=lr, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss(reduction='none')  # Benchmarked, L1 performs best vs MSE/SmoothL1
    opt_state = []

    for it in range(n_iter):
        optimizer.zero_grad()

        hand_verts, hand_joints, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1),
                                                th_betas=mano_shape, th_trans=mano_trans)

        if contact_norm_method != 0 and not caps_on_hand:
            with torch.no_grad():   # We need to calculate hand normals if using more complicated methods
                mano_mesh = Meshes(verts=hand_verts, faces=mano_layer.th_faces.repeat(batch_size, 1, 1))
                hand_normals = mano_mesh.verts_normals_padded()
        else:
            hand_normals = torch.zeros(hand_verts.shape, device=device)

        if partition_type == 'part':
            contact_obj = torch.zeros_like(obj_contact_target)
            for i in range(16):
                hand_mask = mano_layer.part_ids == i
                obj_mask = (obj_partition == i)
                # Calculate contact from this hand part to all object vertices for all batches
                c_full, _, _ = calculate_contact_capsule(
                    hand_verts[:, hand_mask],
                    hand_normals[:, hand_mask],
                    obj_verts,
                    obj_normals,
                    caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad,
                    caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method
                )
                # Assign contact values only to vertices belonging to this part
                contact_obj[obj_mask] = c_full.squeeze(-1)[obj_mask]
        
        elif partition_type == 'cse':
            # CSE correspondence: each obj point maps to a weighted combination of hand vertices.
            # Distance is point-to-point between obj vertex and its CSE-corresponding hand point.
            target_hand_pts = Wverts @ hand_verts  # (B, N, 3)
            dist = torch.norm(obj_verts - target_hand_pts, dim=-1) / caps_rad  # (B, N)
            contact_obj = sdf_to_contact(dist, None, method=contact_norm_method)
        else:
            c_full, _, _ = calculate_contact_capsule(
                hand_verts,
                hand_normals,
                obj_verts,
                obj_normals,
                caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad,
                caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method
            )
            contact_obj = c_full.squeeze(-1)


        contact_obj_sub = obj_contact_target - contact_obj
        contact_obj_weighted = contact_obj_sub + torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=1)

        # contact_hand_sub = hand_contact_target - contact_hand
        # contact_hand_weighted = contact_hand_sub + torch.nn.functional.relu(contact_hand_sub) * w_cont_asym  # Loss for 'missing' contact higher
        # loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        loss = loss_contact_obj * w_cont_obj

        if is_thin is None:
            is_thin = torch.zeros(batch_size, device=device)  # For hands, we never use thin mode
        if w_pen_cost > 0 and it >= pen_it:
            pen_cost = calculate_penetration_cost(hand_verts, hand_normals, obj_verts, obj_normals, is_thin, contact_norm_method)
            loss += pen_cost.mean(dim=1) * w_pen_cost

        out_dict = {'loss': loss.detach().cpu()}
        if save_history:
            out_dict['hand_verts'] = hand_verts.detach().cpu()#.numpy()
            out_dict['hand_joints'] = hand_joints.detach().cpu()#.numpy()
            out_dict['contact_obj'] = contact_obj.detach().cpu()#.numpy()
            # out_dict['contact_hand'] = contact_hand.detach().cpu()#.numpy()
        opt_state.append(out_dict)
        if it % 100 == 99:
            print(f"Iter {it} | Loss: {loss.mean().item():.6f} | Contact Obj Loss: {loss_contact_obj.mean().item():.6f} | Penetration Cost: {pen_cost.mean().item() if w_pen_cost > 0 and it >= pen_it else 0:.6f}")
        loss.mean().backward()
        optimizer.step()

    # opt_vector = opt_vector.detach()
    # global_pose, mano_pose, mano_shape, mano_trans = opt_vector[:, :3], opt_vector[:, 3:ncomps+3], opt_vector[:, ncomps+3:ncomps+13], opt_vector[:, ncomps+13:ncomps+16]
    return global_pose, mano_pose, mano_shape, mano_trans, init_pose


def optimize_pose_by_contact(mano_layer, grid_centers, target_pts, pred_contact, target_W_verts, dist2contact_fn,
                            n_iter=1000, lr=0.01, grid_scale=0.01, w_repulsive=1.0,
                            w_reg_loss=0.001, init_pose=None, nms_mask=None):
    """
    A simpler version of pose optimization that only uses the repulsive loss from local grids.
    This is used in the ablation study to verify the effect of the contact loss.
    """
    batch_size, num_grids = grid_centers.shape[:2]
    global_pose = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    batch_size, num_grids = grid_centers.shape[:2]
    global_pose = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    
    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=target_pts.dtype, device=target_pts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=target_pts.dtype, device=target_pts.device)

    # contact_grid_mask = (pred_contact.view(batch_size, num_grids, -1) > 0).any(dim=-1) # B x num_grids
    # contact_point_mask = contact_grid_mask.view(batch_size, num_grids, 1).repeat(1, 1, 512).view(batch_size, -1) # B x num_grids x 3 -> B x (num_grids * 3)
    # n_non_contact_grids = torch.sum(~contact_grid_mask).item()

    zero_contact = torch.zeros_like(pred_contact).to(pred_contact.device).float()
    w = pred_contact * 5 + 1

    if nms_mask is not None:
        target_W_verts[~nms_mask] = 0  # Zero out non-maximal contacts
        w[~nms_mask] = 0  # Zero out non-maximal weights

    if init_pose is not None:
        # init_pose format: [trans(3), full_pose(48), betas(10)] = 61 dims
        # full_pose = [global_pose(3), finger_pose(45)]
        mano_trans = init_pose[:, :3].clone()
        full_pose = init_pose[:, 3:51].clone()
        global_pose = full_pose[:, :3]
        mano_pose = full_pose[:, 3:]
        mano_shape = init_pose[:, 51:].clone()
        init_handV, init_handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        init_handV = init_handV.detach()
        init_handJ = init_handJ.detach()

    else:
        ## Initialization
        handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        root = handJ[:, 0:1]
        Rs, ts = cp_match(target_W_verts @ handV - root, target_pts - root, pred_contact.unsqueeze(-1))
        global_pose = matrix_to_axis_angle(Rs)
        mano_trans = ts
    
    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [mano_trans, global_pose, mano_pose, mano_shape]
    optimizer = torch.optim.AdamW(hand_opt_params, lr=lr)

    # Determine if regularization loss should be used
    use_reg_loss = init_pose is not None

    for it in range(n_iter):
        handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        optimizer.zero_grad()

        losses = {}
        ## Contact loss: truly aligning the contact values
        corr_pts = target_W_verts @ handV # (B x N x 3)
        distance = torch.norm(corr_pts - target_pts, dim=-1)  # (B x N)
        contact = dist2contact_fn(distance)  # (B x N)
        contact_diff = (contact - pred_contact) * w

        ### option1: use all contact
        losses['contact'] = F.l1_loss(contact_diff, zero_contact)

        if use_reg_loss:
            pred_param = torch.cat(hand_opt_params, dim=1)
            pose_diff = pred_param - init_pose
            losses['reg'] = torch.mean(pose_diff**2)
            # recon_diff = (torch.norm(handV - init_handV, dim=-1).mean() + torch.norm(handJ - init_handJ, dim=-1).mean()) / 2
            # losses['reg'] = recon_diff 

        # Total loss
        loss = losses['contact']
        if use_reg_loss:
            loss += w_reg_loss * losses['reg']

        loss.backward()
        optimizer.step()

        # Logging every 100 iterations
        if it % 100 == 99:
            loss_strs = [f"{k}: {v.item():.6f}" for k, v in losses.items()]
            print(f"Iter {it} | Loss: {loss.item():.6f} | {' | '.join(loss_strs)}")
    
    return [p.detach() for p in hand_opt_params]

def optimize_pose_wrt_local_grids(mano_layer, grid_centers, target_pts, target_W_verts, weights,
                                  grid_sdfs=None, dist2contact_fn=None, recon_hand_verts=None, recon_verts_mask=None,
                                  n_iter=1200, lr=0.01, grid_scale=0.01, w_repulsive=1.0,
                                  w_reg_loss=0.001, init_pose=None):
    """
    The loss used in pose optimization:
     * L2 loss between weighted target points and hand vertices
     * Repulsive loss from non-contact local grids (current version)
    :param mano_layer: Description
    :param target_pts: Description
    :param n_iter: Description
    :param lr: Description
    """
    batch_size, num_grids = grid_centers.shape[:2]
    global_pose = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    mano_trans = torch.zeros((batch_size, 3), dtype=target_pts.dtype, device=target_pts.device)
    
    mano_pose = torch.zeros((batch_size, mano_layer.ncomps), dtype=target_pts.dtype, device=target_pts.device)
    mano_shape = torch.zeros((batch_size, 10), dtype=target_pts.dtype, device=target_pts.device)

    contact_grid_mask = (weights.view(batch_size, num_grids, -1) > 0).any(dim=-1) # B x num_grids
    if w_repulsive > 0:
        ## Supress the non contact with a reconstructed vertices ie
        recon_grid_dist = torch.cdist(grid_centers.view(batch_size, num_grids, 3), recon_hand_verts)  # B x num_grids x num_recon_verts
        recon_grid_dist = recon_grid_dist.masked_fill(~recon_verts_mask.view(batch_size, 1, -1), grid_scale + 1)  # Mask out non-reconstructed verts
        recon_contact_mask = (recon_grid_dist < grid_scale).any(dim=-1)  # B x num_grids
        contact_grid_mask = contact_grid_mask | recon_contact_mask  # Consider grids close to reconstructed verts as contact

        n_non_contact_grids = torch.sum(~contact_grid_mask).item()

    if init_pose is not None:
        # init_pose format: [trans(3), full_pose(48), betas(10)] = 61 dims
        # full_pose = [global_pose(3), finger_pose(45)]
        # noise_scale = 0.5
        # init_pose = init_pose + torch.cat([torch.randn_like(init_pose[:, :3]) * 0.1, torch.randn_like(init_pose[:, 3:])], dim=1) * noise_scale
        mano_trans = init_pose[:, :3].clone()
        full_pose = init_pose[:, 3:51].clone()
        global_pose = full_pose[:, :3]
        mano_pose = full_pose[:, 3:]
        mano_shape = init_pose[:, 51:].clone()

    else:
        ## Initialization
        handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        root = handJ[:, 0:1]
        Rs, ts = cp_match(target_W_verts @ handV - root, target_pts - root, weights.unsqueeze(-1))
        global_pose = matrix_to_axis_angle(Rs)
        mano_trans = ts

    mano_pose.requires_grad = True
    mano_shape.requires_grad = True
    global_pose.requires_grad = True
    mano_trans.requires_grad = True
    hand_opt_params = [mano_trans, global_pose, mano_pose, mano_shape]
    optimizer = torch.optim.AdamW(hand_opt_params, lr=lr)

    # Determine if regularization loss should be used
    use_reg_loss = init_pose is not None

    for it in range(n_iter):
        handV, handJ, _ = mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)
        optimizer.zero_grad()

        # Dictionary to organize losses
        losses = {}

        # Contact loss: weighted MSE between predicted and target points
        losses['contact'] = torch.mean(weights.unsqueeze(-1) * F.mse_loss(target_W_verts @ handV, target_pts, reduction='none')) * 10000

        ## version 2: based on vertex-wise distance loss:
        # weightedW = weights.unsqueeze(-1) * target_W_verts
        # vertex_contact = torch.sum(weightedW, dim=1)
        # recon_V = weightedW.transpose(-1, -2) @ target_pts
        # losses['contact'] = F.mse_loss(recon_V, vertex_contact.unsqueeze(-1) * handV, reduction='mean') * 10

        # Repulsive loss: penalize hand vertices near non-contact grids
        # grid_dist = torch.cdist(grid_centers, handV)  # B x num_grids x num_hand_verts
        if w_repulsive > 0:
            repul_loss = torch.tensor(0.0, device=target_pts.device)
            if n_non_contact_grids > 0:
                # non_contact_grid_dist = grid_dist[~contact_grid_mask].min(-1)[0]
                # correct_mask = non_contact_grid_dist > grid_scale
                # increate the effect distance quickly beyond the grid scale
                # non_contact_grid_dist[correct_mask] = (non_contact_grid_dist[correct_mask] - grid_scale + 0.5)**2 + grid_scale - 0.25
                # n_non_contact_grid_dist = non_contact_grid_dist / grid_scale * 2 + 1
                # n_non_contact_c = sdf_to_contact(n_non_contact_grid_dist, None, method=2)
                # losses['repulsive'] = torch.sum(n_non_contact_c) / n_non_contact_grids

                ## Option2: use SDF values to indicate penetrations directly
                for b in range(batch_size):
                    contact_grid_mask_b = contact_grid_mask[b]
                    non_contact_grid_points = target_pts[b].view(num_grids, -1, 3)[~contact_grid_mask_b].view(-1, 3)  # num_non_contact_grids x 3
                    non_contact_grid_sdfs = grid_sdfs.view(num_grids, -1)[~contact_grid_mask_b].clone().view(-1)  # num_non_contact_grids
                    dist_mat = torch.cdist(non_contact_grid_points, handV[b])  # num_non_contact_grids
                    nn_pt_idx = dist_mat.argmin(dim=0) # num_hand_verts
                    outside_hand_mask = non_contact_grid_sdfs[nn_pt_idx] > 0
                    dist, nn_hand_idx = dist_mat.min(dim=-1)  # num_non_contact_grids
                    pred_contact = dist2contact_fn(dist)  # num_non_contact_grids
                    pred_contact[outside_hand_mask[nn_hand_idx]] = 0  # If the closest point is outside the hand, we don't consider it as penetration even if it's close
                    non_contact_grid_sdfs[non_contact_grid_sdfs > 0] = 0  # Only consider negative SDF values (penetrations)
                    repul_loss += - torch.sum(pred_contact * non_contact_grid_sdfs) / (n_non_contact_grids + 1e-8)

            losses['repulsive'] = repul_loss
            loss = losses['contact'] + w_repulsive * losses['repulsive']
        else:
            loss = losses['contact']

        # Regularization loss: only when init_pose is provided
        if use_reg_loss:
            pred_param = torch.cat(hand_opt_params, dim=1)
            pose_diff = pred_param - init_pose
            losses['reg'] = torch.mean(pose_diff**2)

        # Total loss
        if use_reg_loss:
            loss += w_reg_loss * losses['reg']

        loss.backward()
        optimizer.step()

        # Logging every 100 iterations
        if it % 100 == 99:
            loss_strs = [f"{k}: {v.item():.6f}" for k, v in losses.items()]
            print(f"Iter {it} | Loss: {loss.item():.6f} | {' | '.join(loss_strs)}")
    
    return [p.detach() for p in hand_opt_params], contact_grid_mask
