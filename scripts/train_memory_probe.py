"""
Measure peak GPU memory of a REAL training step (GraspDiffTrainer.train_val_step
+ backward + optimizer step) across a sweep of batch sizes, on a single GPU.

This runs the actual training path -- not the inference path -- so the numbers
reflect what `scripts/train_diffusion.py` will use. Answers questions like
"does batch_size=64 fit on one 24GB card?" empirically instead of by estimate.

Usage:
    conda activate hoi_common

    # default config (mlcdiff), default sweep
    python scripts/train_memory_probe.py

    # a different config file (any config/*.yaml usable by train_diffusion.py)
    python scripts/train_memory_probe.py --config-name mlcdiff_128_16

    # custom sweep / soak length / target budget
    # (note the leading '+': probe.* is not part of the base config)
    python scripts/train_memory_probe.py --config-name mlcdiff_128_16 \
        '+probe.batch_sizes=[16,32,64]' +probe.soak_steps=10 +probe.budget_mb=24564

    # any normal hydra override still works, e.g. swapping the msdf variant
    python scripts/train_memory_probe.py msdf=dense

Notes:
  * Reports BOTH max_memory_allocated (tensor bytes; reproducible, method-
    intrinsic) and max_memory_reserved (caching-allocator bytes; what actually
    OOMs you and what nvidia-smi roughly shows). Fit decisions use `reserved`.
  * Visualization inside train_val_step is disabled during probing so the
    measurement covers the training graph only.
  * OOM at a given batch size is caught and reported, and the sweep stops there.
"""
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
import numpy as np
from omegaconf import OmegaConf, open_dict

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x / y, replace=True)
OmegaConf.register_new_resolver("power", lambda x, y: x ** y, replace=True)

MB = 1024 ** 2

# Defaults for the probe itself; override on the CLI with probe.<key>=<value>
PROBE_DEFAULTS = {
    'batch_sizes': [8, 16, 32, 48, 64],
    'soak_steps': 5,        # extra real steps per batch size (catches growth past step 1)
    'num_workers': 4,
    'budget_mb': 24564,     # RTX 4090; used only for the fits/doesn't-fit verdict
    'headroom_frac': 0.90,  # call it a fit only below this fraction of budget
}


class _StubTrainer:
    """Minimal stand-in for pl.Trainer so train_val_step's datamodule lookups work."""
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.model = None


def _build(cfg, device):
    """Instantiate the same modules train_diffusion.py builds, on `device`."""
    import common.model.diff.mdm.gaussian_diffusion as mdm_gd
    from common.model.diff.unet import DualUNetModel, UNetModel
    from common.model.gridae import gridae as gridae_module
    from common.model.gridae.old.gridae import GRIDAE as GRIDAEOld
    from common.model.vae.handvae import HandVAE
    from common.model.graspdifftrainer import GraspDiffTrainer
    from common.model.lgcdifftrainer import LGCDiffTrainer

    if cfg.ae.name == 'GRIDAEOld':
        gridae = GRIDAEOld(cfg.ae, obj_1d_feat=True)
    else:
        gridae = getattr(gridae_module, cfg.ae.name)(cfg.ae)
    hand_ae = HandVAE(cfg.hand_ae)

    mdm_cfg = cfg.generator.mdm
    if cfg.generator.model_name == 'dual_latent_diffusion':
        trainer_module, model_class, diffusion_class = (
            GraspDiffTrainer, DualUNetModel, mdm_gd.DualGaussianDiffusion)
    else:
        trainer_module, model_class, diffusion_class = (
            LGCDiffTrainer, UNetModel, mdm_gd.GaussianDiffusion)

    model = model_class(cfg.generator.unet)
    diffusion = diffusion_class(
        timesteps=mdm_cfg.timesteps,
        schedule_cfg=mdm_cfg.schedule_cfg,
        model_mean_type=mdm_gd.ModelMeanType[mdm_cfg.model_mean_type.upper()],
        model_var_type=mdm_gd.ModelVarType[mdm_cfg.model_var_type.upper()],
        rand_t_type=mdm_cfg.rand_t_type,
        rescale_timesteps=mdm_cfg.rescale_timesteps,
        msdf_cfg=cfg.msdf,
    )

    if trainer_module is GraspDiffTrainer:
        pl_model = trainer_module(grid_ae=gridae, model=model, diffusion=diffusion,
                                  hand_ae=hand_ae, cfg=cfg)
    else:
        pl_model = trainer_module(grid_ae=gridae, model=model, diffusion=diffusion, cfg=cfg)

    pl_model = pl_model.to(device).train()
    # train_val_step logs through the Trainer, which we don't have here.
    pl_model.log_dict = lambda *a, **k: None
    pl_model.log = lambda *a, **k: None
    return pl_model


def _measure_bs(pl_model, dm, cfg, bs, device, probe):
    """Run real training steps at batch size `bs`; return a result dict."""
    from common.dataset_utils.datamodules import HOIDatasetModule

    loader = torch.utils.data.DataLoader(
        dm.train_set, batch_size=bs, shuffle=False,
        num_workers=probe.num_workers, collate_fn=HOIDatasetModule.collate_fn,
        drop_last=True,
    )
    opt = torch.optim.AdamW(
        [p for p in pl_model.parameters() if p.requires_grad], lr=cfg.train.lr)

    def one_step(batch, idx):
        batch = pl_model.transfer_batch_to_device(batch, device, 0)
        opt.zero_grad(set_to_none=True)
        loss = pl_model.train_val_step(batch, batch_idx=idx, stage='train')
        loss.backward()
        opt.step()
        return loss

    it = iter(loader)
    try:
        # Warm-up: allocator + autotune settle, optimizer states get allocated.
        one_step(next(it), 1)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        steady = torch.cuda.memory_allocated(device) / MB

        # Measured steps over DIFFERENT batches (n_grids/contact counts vary).
        n_steps = max(1, probe.soak_steps)
        for si in range(n_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            one_step(batch, si + 2)
        torch.cuda.synchronize(device)

        return {
            'bs': bs, 'ok': True, 'steps': n_steps, 'steady': steady,
            'alloc': torch.cuda.max_memory_allocated(device) / MB,
            'resv': torch.cuda.max_memory_reserved(device) / MB,
        }
    except torch.cuda.OutOfMemoryError as e:
        return {'bs': bs, 'ok': False, 'err': str(e).split('\n')[0][:110]}
    finally:
        del opt, loader
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


@hydra.main(config_path="../config", config_name="mlcdiff", version_base=None)
def main(cfg):
    from common.dataset_utils.datamodules import HOIDatasetModule

    torch.set_float32_matmul_precision('medium')
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device available; this probe measures GPU memory.")
    device = torch.device("cuda:0")

    # Merge probe defaults under any probe.* values present in the config/CLI.
    # (CLI overrides need `+probe.x=y` since `probe` isn't in the base configs.)
    user_probe = OmegaConf.select(cfg, 'probe') or OmegaConf.create({})
    probe = OmegaConf.merge(OmegaConf.create(PROBE_DEFAULTS), user_probe)

    # Single GPU, no visualization, no wandb during probing.
    with open_dict(cfg):
        cfg.trainer.devices = 1
        cfg.train.vis_every_n_batches = 10 ** 9
        cfg.val.vis_every_n_batches = 10 ** 9
        cfg.debug = True
        cfg.run_phase = 'train'

    gpu_name = torch.cuda.get_device_name(0)
    total_mb = torch.cuda.get_device_properties(0).total_memory / MB
    print(f"\n[gpu] {gpu_name}  ({total_mb:.0f} MB total)")
    print(f"[cfg] msdf: num_grids={cfg.msdf.num_grids} kernel_size={cfg.msdf.kernel_size} "
          f"scale={cfg.msdf.scale}")
    print(f"[cfg] unet: d_x={cfg.generator.unet.d_x} d_y={cfg.generator.unet.d_y} "
          f"d_model={cfg.generator.unet.d_model} nblocks={cfg.generator.unet.nblocks}")
    print(f"[cfg] precision={cfg.trainer.precision}  lr={cfg.train.lr}  "
          f"loss_weights={OmegaConf.to_container(cfg.train.loss_weights)}")

    pl_model = _build(cfg, device)
    n_total = sum(p.numel() for p in pl_model.parameters())
    n_train = sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    torch.cuda.synchronize(device)
    weights_mb = torch.cuda.memory_allocated(device) / MB
    print(f"[params] total={n_total/1e6:.2f}M  trainable={n_train/1e6:.2f}M")
    print(f"[mem] weights resident: {weights_mb:.0f} MB "
          f"(+ ~{n_train*8/MB:.0f} MB AdamW states once stepped)")

    dm = HOIDatasetModule(cfg)
    dm.prepare_data()
    dm.setup('fit')
    object.__setattr__(pl_model, '_trainer', _StubTrainer(dm))

    budget = float(probe.budget_mb)
    limit = budget * float(probe.headroom_frac)
    results = []
    for bs in list(probe.batch_sizes):
        r = _measure_bs(pl_model, dm, cfg, int(bs), device, probe)
        results.append(r)
        if r['ok']:
            print(f"[bs={r['bs']:>4}] {r['steps']} steps  steady={r['steady']:7.0f}MB  "
                  f"peak_alloc={r['alloc']:8.0f}MB  peak_reserved={r['resv']:8.0f}MB")
        else:
            print(f"[bs={r['bs']:>4}] *** CUDA OOM *** {r['err']}")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print(f"  SINGLE-GPU TRAINING MEMORY   budget={budget:.0f} MB "
          f"(fit threshold {limit:.0f} MB = {probe.headroom_frac:.0%})")
    print("=" * 84)
    print(f"  {'bs':>5} | {'steady':>9} | {'peak alloc':>11} | {'peak reserved':>13} | verdict")
    print("  " + "-" * 80)
    ok_bs = []
    for r in results:
        if not r['ok']:
            print(f"  {r['bs']:>5} | {'OOM':>9} | {'OOM':>11} | {'OOM':>13} | OOM")
            continue
        fits = r['resv'] < limit
        ok_bs.append((r['bs'], fits))
        print(f"  {r['bs']:>5} | {r['steady']:8.0f}M | {r['alloc']:10.0f}M | "
              f"{r['resv']:12.0f}M | {'fits' if fits else 'too tight'}")
    print("=" * 84)

    fitting = [b for b, f in ok_bs if f]
    if fitting:
        print(f"  Largest batch size fitting in {budget:.0f} MB: {max(fitting)}")
    else:
        print(f"  No probed batch size fits in {budget:.0f} MB.")

    # Rough linear extrapolation: reserved ≈ a*bs + b, from measured points
    good = [r for r in results if r['ok']]
    if len(good) >= 2:
        xs = np.array([r['bs'] for r in good], dtype=float)
        ys = np.array([r['resv'] for r in good], dtype=float)
        a, b = np.polyfit(xs, ys, 1)
        print(f"  Fit: peak_reserved ≈ {a:.1f} MB/sample × bs + {b:.0f} MB")
        if a > 0:
            print(f"  → extrapolated max bs under {limit:.0f} MB: {int((limit - b) / a)}")
    print()


if __name__ == "__main__":
    main()
