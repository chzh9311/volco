import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import lightning as L
import hydra
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import datetime
from omegaconf import OmegaConf
from common.dataset_utils.datamodules import HOIDatasetModule
# from common.model.mlctrainer import MLCTrainer
import common.model.diff.mdm.gaussian_diffusion as mdm_gd
from common.model.diff.unet import UNetModel, DualUNetModel
from common.model.gridae import gridae as gridae_module
from common.model.gridae.old.gridae import GRIDAE as GRIDAEOld
from common.model.vae.handvae import HandVAE
# from common.model.hand_ipt_vae.hand_imputation import HandImputationVAE
from common.model.lgcdifftrainer import LGCDiffTrainer
from common.model.graspdifftrainer import GraspDiffTrainer
from common.utils.misc import set_seed, load_pl_ckpt
from lightning.pytorch.callbacks import ModelCheckpoint

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("power", lambda x, y: x ** y, replace=True)
@hydra.main(version_base=None, config_path="../config", config_name="mlcdiff")
def main(cfg):
    # Set seed for reproducibility FIRST
    # Set matmul precision for better performance on Tensor Cores
    torch.set_float32_matmul_precision('medium')

    # print("Configuration:")
    # print(OmegaConf.to_yaml(cfg))
    
    # Initialize the model, data module, and trainer
    # generator_module = importlib.import_module(f"common.model.{cfg.generator.model_type}.{cfg.generator.model_name}")
    # model = getattr(generator_module, cfg.generator.model_name.upper())(cfg)
    # MDM GaussianDiffusion
    mdm_cfg = cfg.generator.mdm
    # DDPM (the base version of diffusion)
    # diffusion1 = DDPM(cfg.generator.ddpm)
    if cfg.ae.name == 'GRIDAEOld':
        gridae = GRIDAEOld(cfg.ae, obj_1d_feat=True)
    else:
        gridae = getattr(gridae_module, cfg.ae.name)(cfg.ae)
    hand_ae = HandVAE(cfg.hand_ae)
    # sd = torch.load(cfg.hand_ae.pretrained_weight, map_location='cpu', weights_only=True)['state_dict']
    # load_pl_ckpt(hand_ae, sd, prefix='model.')
    
    t = datetime.datetime.now()
    exp_name = cfg.generator.model_name + '-' + cfg.run_phase + '-' + t.strftime('%Y%m%d-%H%M%S')
    if cfg.debug:
        logger = TensorBoardLogger(save_dir='logs/tb_logs', name=exp_name)
    else:
        import wandb
        logger = WandbLogger(name=exp_name,
                             project='LG3DContact',
                             log_model=True, save_dir='logs/wandb_logs')
    ckpt_dir = logger.log_dir

    # Only disable inference_mode for testing (needed for pose optimization gradients)
    inference_mode = False if cfg.run_phase == 'test' else True
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath=ckpt_dir,
        filename=cfg.checkpoint.filename + '-{epoch:02d}-{val/total_loss:.4f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode='min',
        save_last=True,
    )

    # Use DDPStrategy with static_graph=True to handle frozen parameters without memory overhead
    # This is optimal when model has frozen weights (e.g., pretrained autoencoder)
    devices = cfg.trainer.get('devices', 1)
    if type(devices) is not int:
        devices = len(devices)
    if devices > 1 and cfg.trainer.get('accelerator') == 'gpu':
        strategy = DDPStrategy(static_graph=True)
        trainer = L.Trainer(**cfg.trainer, strategy=strategy, inference_mode=inference_mode, logger=logger, callbacks=[checkpoint_callback])
    else:
        trainer = L.Trainer(**cfg.trainer, inference_mode=inference_mode, logger=logger, callbacks=[checkpoint_callback])

    data_module = HOIDatasetModule(cfg)

    # Start training
    if cfg.generator.model_name == 'dual_latent_diffusion':
        trainer_module = GraspDiffTrainer
        model_class = DualUNetModel
        diffusion_class = mdm_gd.DualGaussianDiffusion
    else:
        trainer_module = LGCDiffTrainer
        model_class = UNetModel
        diffusion_class = mdm_gd.GaussianDiffusion
    
    model = model_class(cfg.generator.unet)
    diffusion = diffusion_class(
        timesteps=mdm_cfg.timesteps,
        schedule_cfg=mdm_cfg.schedule_cfg,
        model_mean_type=mdm_gd.ModelMeanType[mdm_cfg.model_mean_type.upper()],
        model_var_type=mdm_gd.ModelVarType[mdm_cfg.model_var_type.upper()],
        rand_t_type=mdm_cfg.rand_t_type,
        rescale_timesteps=mdm_cfg.rescale_timesteps,
        msdf_cfg=cfg.msdf)

    if cfg.run_phase == 'train':
        pl_model = trainer_module(grid_ae=gridae, model=model, diffusion=diffusion, hand_ae=hand_ae, cfg=cfg)
        trainer.fit(pl_model, datamodule=data_module, ckpt_path=cfg.train.get('resume_ckpt', None))
    elif cfg.run_phase == 'val':
        # pl_model = LGCDiffTrainer(gridae, model, cfg)
        pl_model = trainer_module.load_from_checkpoint(cfg.val.get('ckpt_path', None), grid_ae=gridae, model=model, diffusion=diffusion, hand_ae=hand_ae, cfg=cfg)
        trainer.validate(pl_model, datamodule=data_module)
    else:
        if hasattr(cfg, 'seed'):
            set_seed(cfg.seed)
        else:
            set_seed(42)  # default seed

        if cfg.ckpt_path == "Random":
            print("Using random weights for model, grid_ae, and hand_ae.")
        else:
            sd = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=False)['state_dict']
            print('total keys in ckpt:', len(sd.keys()))
            load_pl_ckpt(gridae, sd, prefix='grid_ae.')
            load_pl_ckpt(model, sd, prefix='model.')
            load_pl_ckpt(hand_ae, sd, prefix='hand_ae.')
            unused_keys = [k for k in sd.keys() if not (k.startswith('grid_ae.') or k.startswith('model.') or k.startswith('hand_ae.'))]
            print(f'Unused keys in ckpt: {unused_keys}')

        pl_model = trainer_module(grid_ae=gridae, model=model, diffusion=diffusion, hand_ae=hand_ae, cfg=cfg)
        # pl_model = LGCDiffTrainer.load_from_checkpoint(cfg.ckpt_path, grid_ae=gridae, model=model, cfg=cfg)
        trainer.test(pl_model, datamodule=data_module)


if __name__ == '__main__':
    main()