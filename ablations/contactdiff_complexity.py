"""
Measure GFLOPs, parameter count, and peak GPU memory for the ContactDiff
(point-based ablation) inference pipeline:

    UNetModel (1 denoising step × T)

The model directly denoises per-point contact maps (contact + CSE) conditioned
on object surface geometry (xyz + normals).

Usage:
    conda activate hoi_common
    python ablations/contactdiff_complexity.py
"""
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import OmegaConf
from torchinfo import summary

OmegaConf.register_new_resolver("add", lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver("sub", lambda x, y: x - y, replace=True)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x / y, replace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper helpers
# ─────────────────────────────────────────────────────────────────────────────

class ConditionWrapper(torch.nn.Module):
    """obj_pc (B, 6, N_pts) → obj_feat (B, obj_feat_dim, N_pts)
    obj_pc = concat(xyz, normals), PointNet2seg encoder.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obj_pc):
        local_obj_feat, glob_obj_feat = self.model.obj_feat_net(obj_pc)
        obj_feat = torch.cat(
            [local_obj_feat, glob_obj_feat.repeat(1, 1, local_obj_feat.shape[2])],
            dim=1,
        )
        return obj_feat


class DenoiseWrapper(torch.nn.Module):
    """x_t (B, N_pts, d_x) → x_pred (B, N_pts, d_x)
    Bakes in obj_feat and timestep so torchinfo sees a single tensor input.
    """
    def __init__(self, model, obj_feat, ts):
        super().__init__()
        self.model = model
        self.register_buffer('obj_feat', obj_feat)
        self.register_buffer('ts', ts)

    def forward(self, x_t):
        obj_feat = self.obj_feat.expand(x_t.shape[0], *self.obj_feat.shape[1:])
        return self.model(x_t, self.ts.expand(x_t.shape[0]), obj_feat)


# ─────────────────────────────────────────────────────────────────────────────

def _params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _summary(wrapper, input_data, device, label, depth=4):
    print(f"\n{'='*60}")
    print(f"  torchinfo — {label}")
    print(f"{'='*60}")
    s = summary(
        wrapper,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=depth,
        verbose=1,
        device=device,
    )
    return s.total_mult_adds


@hydra.main(config_path="../config", config_name="contactdiff", version_base=None)
def main(cfg):
    from ablations.point_unet import UNetModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build model ───────────────────────────────────────────────────────────
    model = UNetModel(cfg.generator.unet).to(device).eval()

    # ── Dimensions from config ─────────────────────────────────────────────────
    N     = cfg.data.object_sample    # 1024  (sampled object surface points)
    d_x   = cfg.generator.unet.d_x   # 1 + cse_dim = 5
    T     = cfg.generator.diffusion.timesteps  # 1000
    # obj_pc: xyz (3) + normals (3) = 6 channels
    obj_in_dim = cfg.generator.unet.obj_encoder.in_dim  # 6

    B = 1  # batch size for profiling

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    obj_pc  = torch.randn(B, obj_in_dim, N, device=device)  # (B, 6, N_pts)
    x_noisy = torch.randn(B, N, d_x, device=device)         # (B, N_pts, d_x)
    ts      = torch.zeros(B, dtype=torch.long, device=device)

    # ── 1. Parameter count ────────────────────────────────────────────────────
    total_params, trainable_params = _params(model)

    print(f"\n{'='*60}")
    print("  PARAMETER COUNTS")
    print(f"{'='*60}")
    print(f"  UNetModel (ContactDiff) : {total_params/1e6:.3f} M")
    print(f"  Trainable               : {trainable_params/1e6:.3f} M")

    # ── 2. Object conditioning ────────────────────────────────────────────────
    cond_macs = _summary(
        ConditionWrapper(model).to(device).eval(),
        (obj_pc,), device,
        f"UNet condition  [B,{obj_in_dim},N_pts] → obj_feat  (PointNet2seg)",
    )

    # Compute obj_feat for the denoising wrapper
    with torch.no_grad():
        local_obj_feat, glob_obj_feat = model.obj_feat_net(obj_pc)
        obj_feat = torch.cat(
            [local_obj_feat, glob_obj_feat.repeat(1, 1, local_obj_feat.shape[2])],
            dim=1,
        )

    # ── 3. Single denoising step ──────────────────────────────────────────────
    step_macs = _summary(
        DenoiseWrapper(model, obj_feat[:1], ts[:1]).to(device).eval(),
        (x_noisy,), device,
        f"UNet single step  [B,N_pts,{d_x}] → x_pred",
    )

    # ── 4. Peak GPU memory ────────────────────────────────────────────────────
    peak_mb_cond = peak_mb_denoise = None
    if device.type == "cuda":
        # Conditioning: encode object point cloud once
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            local_f, glob_f = model.obj_feat_net(obj_pc)
            _ = torch.cat([local_f, glob_f.repeat(1, 1, local_f.shape[2])], dim=1)
        peak_mb_cond = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        # Denoising: full reverse pass (T steps)
        cond_dict = obj_feat
        x = torch.randn(B, N, d_x, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                ts_t = torch.full((B,), t, dtype=torch.long, device=device)
                x = model(x, ts_t, cond_dict)
        peak_mb_denoise = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    # ── 5. Summary ────────────────────────────────────────────────────────────
    gflops = lambda macs: 2 * macs / 1e9

    print(f"\n{'='*60}")
    print(f"  INFERENCE SUMMARY  (B=1, N={N} surface pts, d_x={d_x}, T={T} steps)")
    print(f"{'='*60}")
    print(f"  UNetModel (ContactDiff) : {total_params/1e6:.3f} M params")
    print()
    print("  ── GFLOPs per inference call ───────────────────────────")
    print(f"  Obj conditioning         : {gflops(cond_macs):.4f} GFLOPs")
    print(f"  1 denoise step           : {gflops(step_macs):.4f} GFLOPs")
    print(f"  {T} steps total           : {gflops(step_macs * T):.2f} GFLOPs")
    total_inf_gflops = gflops(cond_macs) + gflops(step_macs * T)
    print(f"  ───────────────────────────────────────────────────────")
    print(f"  Total inference GFLOPs   : {total_inf_gflops:.2f}")
    print()
    if peak_mb_cond is not None:
        print("  ── Peak GPU memory ─────────────────────────────────────")
        print(f"  Conditioning            : {peak_mb_cond:.1f} MB")
        print(f"  Denoising ({T} steps)   : {peak_mb_denoise:.1f} MB")
    else:
        print("  Peak GPU mem   : N/A (no CUDA)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
