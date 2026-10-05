import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import hydra
from omegaconf import OmegaConf
import torch
import datetime
import importlib
from common.dataset_utils.datamodules import LocalGridDataModule
from common.model.lgtrainer import LGTrainer
from common.model.gridae import gridae
from common.model.gridae.old.gridae import GRIDAE as GRIDAEOld
from lightning.pytorch.callbacks import ModelCheckpoint
# from common.model.gridvqvae.gridvqvae import GRIDVQVAE

# Register OmegaConf resolvers for arithmetic operations
OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x / y, replace=True)

@hydra.main(config_path="../config", config_name="gridae")
def main(cfg):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # wandb_key = ''
    # wandb.login(key=wandb_key)
    t = datetime.datetime.now()
    if cfg.debug:
        logger = TensorBoardLogger(save_dir='logs/tb_logs')
    else:
        import wandb
        logger = WandbLogger(name=cfg.ae.name+'-'+t.strftime('%Y%m%d-%H%M%S'),
                                    project='LG3DContact',
                                    log_model=True, save_dir='logs/wandb_logs')

    # Initialize the model, data module, and trainer
    if cfg.ae.name == 'GRIDAEOld':
        model = GRIDAEOld(cfg.ae, obj_1d_feat=False)
    else:
        model_class = getattr(gridae, cfg.ae.name)
        model = model_class(cfg.ae)
    ckpt_dir = logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath=ckpt_dir,
        filename=cfg.checkpoint.filename + '-{epoch:02d}-{val/total_loss:.4f}',
        save_top_k=cfg.checkpoint.save_top_k,
        mode='min',
        save_last=True,
    )
    # input = torch.randn(3, cfg.ae.in_dim, 8, 8, 8)
    # embedding_loss, recon, perplexity = model(input, verbose=True)
    print("="*50)
    if cfg.train.get('pretrained_ckpt', None) is not None:
        pretrained_ckpt = torch.load(cfg.train.pretrained_ckpt, weights_only=True)['state_dict']
        new_state_dict = model.state_dict()
        all_cnt, match_cnt = 0, 0
        for k, v in pretrained_ckpt.items():
            if k.startswith('model.'):
                k = k[len('model.'):]
                if k in new_state_dict and v.size() == new_state_dict[k].size():
                    new_state_dict[k] = v
                    match_cnt += 1
                all_cnt += 1
        print(f"Loaded {match_cnt}/{all_cnt} parameters from pretrained checkpoint.")
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print("Training from scratch.")

    data_module = LocalGridDataModule(cfg)
    if cfg.run_phase == 'train':
        pl_trainer = LGTrainer(model, cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=[checkpoint_callback])
        # trainer.fit(pl_trainer, datamodule=data_module)
        trainer.fit(pl_trainer, datamodule=data_module, ckpt_path=cfg.train.get('resume_ckpt', None))
    elif cfg.run_phase == 'val':
        pl_trainer = LGTrainer.load_from_checkpoint(cfg.ckpt_path, model=model, cfg=cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger)
        trainer.validate(pl_trainer, datamodule=data_module)
    else:
        cfg.trainer.enable_progress_bar = True
        pl_trainer = LGTrainer.load_from_checkpoint(cfg.ckpt_path, model=model, cfg=cfg)
        trainer = L.Trainer(**cfg.trainer, logger=logger)
        trainer.test(pl_trainer, datamodule=data_module)
    
    # Start training
    # trainer.fit(pl_trainer, datamodule=data_module)


if __name__ == '__main__':
    main()