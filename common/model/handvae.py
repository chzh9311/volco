import torch
import numpy as np
import torch.nn as nn
from common.manopth.manopth.manolayer import ManoLayer
from common.model.handobject import HandObject
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from common.model.layers import DiagonalGaussianDistribution
from common.utils.vis import o3dmesh, geom_to_img
from lightning import LightningModule


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.fc_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.fc2(self.af(self.fc1(self.af(feature)))) + self.fc_res(feature)


class MLPPointEncoder(nn.Module):
    def __init__(self, in_channels=4, num_points=778, hidden_dim=256, feat_dim=1024):
        super(MLPPointEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, 1)
        self.fc2 = nn.Linear(16 * num_points, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feat_dim)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim)

        self.af = nn.ReLU(inplace=True)

    def forward(self, points):
        x = self.af(self.bn1(self.conv1(points)))
        x = x.view(x.size(0), -1)
        x = self.af(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x


class HandVAE(nn.Module):
    def __init__(self, cfg):
        super(HandVAE, self).__init__()
        object.__setattr__(self, 'mano_layer', ManoLayer(mano_root=cfg.mano_root, use_pca=False, flat_hand_mean=True).eval().requires_grad_(False))
        self.hidden_dim = cfg.hidden_dim
        self.latent_dim = cfg.latent_dim
        # Define encoder layers (outputs 2x latent_dim for mean and logvar)
        # self.hand_encoder = PointNetEncoder(
        #     channel=cfg.input_channel, hidden_dim=cfg.hidden_dim)
        self.hand_encoder = MLPPointEncoder(in_channels=cfg.input_channel, hidden_dim=cfg.hidden_dim, feat_dim=cfg.feat_dim)
        
        self.encoder = nn.Sequential(
            Linear_ResBlock(input_size=cfg.feat_dim, output_size=cfg.feat_dim),
            Linear_ResBlock(input_size=cfg.feat_dim, output_size=cfg.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            Linear_ResBlock(input_size=cfg.latent_dim, output_size=2 * cfg.latent_dim),
            Linear_ResBlock(input_size=2 * cfg.latent_dim, output_size=61)
        )

    def encode(self, x):
        feat = self.hand_encoder(x)
        h = self.encoder(feat)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        self.mano_layer.to(z.device)
        recon_param = self.decoder(z)
        recon_param[:, :3] = self.denormalize_trans(recon_param[:, :3])
        trans = recon_param[:, :3]
        pose = recon_param[:, 3:51]
        betas = recon_param[:, 51:]
        handV, handJ, _ = self.mano_layer(pose, th_betas=betas, th_trans=trans)
        return recon_param, handV, handJ

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.sample()
        recon_param, handV, handJ = self.decode(z)
        return recon_param, handV, handJ, posterior
    
    def normalize_trans(self, hand_trans):
        return hand_trans / 0.2

    def denormalize_trans(self, hand_trans):
        return hand_trans * 0.2


class HandVAETrainer(LightningModule):
    def __init__(self, model, cfg):
        super(HandVAETrainer, self).__init__()
        self.model = model
        self.criterion = nn.L1Loss()
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, stage='val')
    
    def train_val_step(self, batch, batch_idx, stage):

        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.model.mano_layer, normalize=True)
        handobject.load_from_batch(batch)
        handV_gt = handobject.hand_verts
        handJ_gt = handobject.hand_joints
        self.model.mano_layer.to(self.device)

        recon_param, handV_pred, handJ_pred, posterior = self.model(handV_gt.permute(0, 2, 1))

        recon_loss = (self.criterion(handV_pred, handV_gt) + self.criterion(handJ_pred, handJ_gt)) / 2
        kld_loss = posterior.kl().mean()
        loss = self.cfg.train.loss_weights.w_recon * recon_loss + self.cfg.train.loss_weights.w_kl * kld_loss

        loss_dict = {
            f'{stage}/total_loss': loss,
            f'{stage}/recon_loss': recon_loss,
            f'{stage}/kld_loss': kld_loss
        }

        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            pred_mesh = o3dmesh(handV_pred[0].detach().cpu().numpy(), self.model.mano_layer.th_faces.cpu().numpy(), color=[1, 0, 0])
            gt_mesh = o3dmesh(handV_gt[0].detach().cpu().numpy(), self.model.mano_layer.th_faces.cpu().numpy(), color=[0, 0, 1])
            vis_img = geom_to_img([pred_mesh, gt_mesh], w=200, h=200, half_range=0.1)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger expects (C, H, W), geom_to_img returns (H, W, C)
                    vis_img_chw = np.transpose(vis_img, (2, 0, 1))
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/hand_recon', vis_img_chw, global_step)
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger expects (H, W, C)
                    import wandb
                    self.logger.experiment.log({f'{stage}/hand_recon': wandb.Image(vis_img)}, step=self.global_step)

        return loss
    
    def on_test_batch_start(self, batch, batch_idx):
        self.recon_err_list = []

    def test_step(self, batch, batch_idx):
        thetas = batch['theta']
        betas = batch['beta']
        ## Augment the betas by adding noise.
        betas = betas + torch.randn_like(betas)
        pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(thetas[:, 3:].view(-1, 15, 3))).view(-1, 15*6)
        nhandV_gt, nhandJ_gt, _ = self.model.mano_layer(
            torch.cat([torch.zeros(thetas.shape[0], 3, device=self.device), thetas[:, 3:]], dim=-1),
            th_betas=betas)
            
        # handobject = HandObject(self.cfg.data, self.device, mano_layer=self.model.mano_layer, normalize=False)
        # handobject.load_from_batch(batch)
        # handV_gt = handobject.hand_verts
        # handJ_gt = handobject.hand_joints
        # root_j = handobject.cano_joints[:, 0]
        # hand_root_R = axis_angle_to_matrix(handobject.hand_root_rot)
        # hand_trans = handobject.hand_trans
        # nhandV = (handV_gt - root_j.unsqueeze(1) - hand_trans.unsqueeze(1)) @ hand_root_R + root_j.unsqueeze(1)
        recon_param, handV_pred, handJ_pred, posterior = self.model(nhandV_gt.permute(0, 2, 1))
        recon_error = torch.norm(handV_pred - nhandV_gt, dim=-1).mean(dim=-1)  # B
        self.recon_err_list.append(recon_error.detach().cpu().numpy())
    
    def on_test_epoch_end(self):
        avg_recon_err = np.concatenate(self.recon_err_list, axis=0).mean() * 1000
        print(f"Average hand vertex reconstruction error: {avg_recon_err:.6f} mm")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        return optimizer