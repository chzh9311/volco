"""
Measure GFLOPs, parameter count, and peak GPU memory for the VolumeVAE model.

Usage:
    conda activate hoi_common
    python scripts/model_complexity.py
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


class EncodeWrapper(torch.nn.Module):
    """Wraps model.encode so torchinfo can trace it with a single tensor input."""
    def __init__(self, model, obj_sdf):
        super().__init__()
        self.model = model
        self.register_buffer('obj_sdf', obj_sdf)

    def forward(self, x):
        posterior, _, _ = self.model.encode(x, self.obj_sdf.expand(x.shape[0], *self.obj_sdf.shape[1:]))
        return posterior.mode()


class DecodeWrapper(torch.nn.Module):
    """Wraps model.decode so torchinfo can trace it with a single tensor input."""
    def __init__(self, model, obj_cond):
        super().__init__()
        self.model = model
        self.obj_cond = obj_cond  # list of tensors, not nn.Parameters

    def forward(self, z):
        return self.model.decode(z, obj_cond=self.obj_cond)


@hydra.main(config_path="../config", config_name="volume_vae", version_base=None)
def main(cfg):
    from common.model.volume_vae import volume_vae as volume_vae_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = getattr(volume_vae_module, cfg.ae.name)
    model = model_class(cfg.ae).to(device)
    model.eval()

    K = cfg.msdf.kernel_size          # 8
    in_dim = cfg.ae.in_dim            # 1 + hand_cse_dim = 5
    obj_in_dim = cfg.ae.obj_in_dim    # 1
    B = 1

    x       = torch.randn(B, in_dim,     K, K, K, device=device)
    obj_sdf = torch.randn(B, obj_in_dim, K, K, K, device=device)

    # ── 1. Parameter count ────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  Total parameters    : {total_params:,}  ({total_params/1e6:.2f} M)")
    print(f"  Trainable parameters: {trainable_params:,}  ({trainable_params/1e6:.2f} M)")

    # ── 2. torchinfo — full forward pass ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("  torchinfo summary — full forward pass (encode + decode)")
    print(f"{'='*60}")
    full_summary = summary(
        model,
        input_data=(x, obj_sdf),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        row_settings=["var_names"],
        depth=4,
        verbose=1,
        device=device,
    )
    total_macs = full_summary.total_mult_adds
    print(f"\n  Total MACs      : {total_macs:,}")
    print(f"  GFLOPs (≈2×MACs): {2 * total_macs / 1e9:.4f}")

    # ── 3. torchinfo — encode / decode via wrappers ───────────────────────────
    with torch.no_grad():
        posterior, _, obj_cond = model.encode(x, obj_sdf)
        z = posterior.mode()

    enc_wrapper = EncodeWrapper(model, obj_sdf).to(device).eval()
    dec_wrapper = DecodeWrapper(model, obj_cond).to(device).eval()

    print(f"\n{'='*60}")
    print("  torchinfo — encode wrapper")
    print(f"{'='*60}")
    enc_summary = summary(
        enc_wrapper,
        input_data=(x,),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
        verbose=1,
        device=device,
    )

    print(f"\n{'='*60}")
    print("  torchinfo — decode wrapper")
    print(f"{'='*60}")
    dec_summary = summary(
        dec_wrapper,
        input_data=(z,),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=4,
        verbose=1,
        device=device,
    )

    enc_macs = enc_summary.total_mult_adds
    dec_macs = dec_summary.total_mult_adds

    # ── 4. Peak GPU memory ────────────────────────────────────────────────────
    peak_mb = peak_mb_large = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(x, obj_sdf)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        B_large   = 256
        x_l       = torch.randn(B_large, in_dim,     K, K, K, device=device)
        obj_sdf_l = torch.randn(B_large, obj_in_dim, K, K, K, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(x_l, obj_sdf_l)
        peak_mb_large = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Model          : {cfg.ae.name}")
    print(f"  Kernel size K  : {K}")
    print(f"  in_dim         : {in_dim}  (contact + CSE)")
    print(f"  feat_dim       : {cfg.ae.feat_dim}")
    print(f"  Parameters     : {total_params/1e6:.3f} M  (trainable: {trainable_params/1e6:.3f} M)")
    print(f"  GFLOPs (B=1)   : {2 * total_macs / 1e9:.4f}")
    print(f"    encode       : {2 * enc_macs / 1e9:.4f} GFLOPs")
    print(f"    decode       : {2 * dec_macs / 1e9:.4f} GFLOPs")
    if peak_mb is not None:
        print(f"  Peak GPU mem   : {peak_mb:.1f} MB  (B=1) / {peak_mb_large:.1f} MB  (B=256)")
    else:
        print("  Peak GPU mem   : N/A (no CUDA)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
