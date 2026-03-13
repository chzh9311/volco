"""
Measure GFLOPs, parameter count, and peak GPU memory for the full
dual-latent diffusion inference pipeline:

    HandVAE (encode/decode)  +  DualUNetModel (1 denoising step × T)  +  GRIDAE (decode)

Usage:
    conda activate hoi_common
    python scripts/diffusion_complexity.py
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
# Wrapper helpers (needed so torchinfo can trace a single-tensor forward)
# ─────────────────────────────────────────────────────────────────────────────

class HandVAEEncodeWrapper(torch.nn.Module):
    """hand_verts (B,3,N_verts) → latent z (B, latent_dim)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        posterior = self.model.encode(x)
        return posterior.mode()


class HandVAEDecodeWrapper(torch.nn.Module):
    """latent z (B, latent_dim) → hand vertices (B, N_verts, 3)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        _, handV, _ = self.model.decode(z)
        return handV


class UNetConditionWrapper(torch.nn.Module):
    """obj_pc (B, K^3+3, N_pts) → obj_feat (B, obj_feat_dim, N_pts)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obj_pc):
        local_obj_feat, glob_obj_feat = self.model.obj_feat_net(obj_pc)
        obj_feat = torch.cat(
            [local_obj_feat, glob_obj_feat.unsqueeze(2).expand(-1, -1, local_obj_feat.shape[2])],
            dim=1,
        )
        return obj_feat


class UNetDenoiseWrapper(torch.nn.Module):
    """
    x_t (B, d_y + d_x*n_pts) → x_pred (B, d_y + d_x*n_pts)
    Bakes in obj_feat and timestep so torchinfo sees a single tensor input.
    """
    def __init__(self, model, obj_feat, ts):
        super().__init__()
        self.model = model
        self.register_buffer('obj_feat', obj_feat)
        self.register_buffer('ts', ts)

    def forward(self, x_t):
        cond = {'obj_feat': self.obj_feat.expand(x_t.shape[0], *self.obj_feat.shape[1:])}
        x_pred, _ = self.model(x_t, self.ts.expand(x_t.shape[0]), cond, is_train=False)
        return x_pred


class GRIDAEEncodeWrapper(torch.nn.Module):
    """x (B*N, in_dim, K, K, K) → z (B*N, latent_dim)"""
    def __init__(self, model, obj_sdf):
        super().__init__()
        self.model = model
        self.register_buffer('obj_sdf', obj_sdf)

    def forward(self, x):
        obj_sdf = self.obj_sdf.expand(x.shape[0], *self.obj_sdf.shape[1:])
        posterior, _, _ = self.model.encode(x, obj_sdf)
        return posterior.mode()


class GRIDAEDecodeWrapper(torch.nn.Module):
    """z (B*N, latent_dim) → recon_grid (B*N, c, K, K, K)"""
    def __init__(self, model, obj_cond):
        super().__init__()
        self.model = model
        self.obj_cond = obj_cond

    def forward(self, z):
        return self.model.decode(z, obj_cond=self.obj_cond)


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


@hydra.main(config_path="../config", config_name="mlcdiff", version_base=None)
def main(cfg):
    from common.model.gridae import gridae as gridae_module
    from common.model.diff.unet import DualUNetModel
    from common.model.vae.handvae import HandVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build models ──────────────────────────────────────────────────────────
    gridae_cls = getattr(gridae_module, cfg.ae.name)
    gridae = gridae_cls(cfg.ae).to(device).eval()

    hand_ae = HandVAE(cfg.hand_ae).to(device).eval()

    unet = DualUNetModel(cfg.generator.unet).to(device).eval()

    # ── Dimensions from config ─────────────────────────────────────────────────
    K       = cfg.msdf.kernel_size      # 8
    N       = cfg.msdf.num_grids        # 128  (sparse MSDF grid points)
    d_x     = cfg.generator.unet.d_x   # 128  (GRIDAE latent dim)
    d_y     = cfg.generator.unet.d_y   # 16   (HandVAE latent dim)
    n_pts   = cfg.generator.unet.n_pts  # 128
    T       = cfg.generator.diffusion.timesteps  # 1000
    N_verts = 778                       # MANO vertex count

    B = 1  # batch size for profiling

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    # HandVAE
    hand_verts  = torch.randn(B, 3, N_verts, device=device)   # (B, 3, N_verts)
    hand_latent = torch.randn(B, d_y, device=device)

    # GRIDAE (per grid)
    obj_sdf     = torch.randn(B * N, 1, K, K, K, device=device)  # (B*N, 1, K, K, K)
    lg_contact  = torch.randn(B * N, cfg.ae.in_dim, K, K, K, device=device)

    # DualUNet
    # obj_pc is obj_msdf permuted: (B, N_pts, K^3+3) → (B, K^3+3, N_pts) = (B, 515, N_pts)
    obj_in_dim = K ** 3 + 3  # 515: MSDF grid values (K^3) + centre xyz (3)
    obj_pc  = torch.randn(B, obj_in_dim, n_pts, device=device)  # (B, 515, N_pts)
    x_noisy = torch.randn(B, d_y + d_x * n_pts, device=device)
    ts      = torch.zeros(B, dtype=torch.long, device=device)  # t = 0 representative

    # ── 1. Parameter counts ────────────────────────────────────────────────────
    gridae_total, gridae_train = _params(gridae)
    hand_ae_total, hand_ae_train = _params(hand_ae)
    unet_total, unet_train = _params(unet)
    total_params = gridae_total + hand_ae_total + unet_total
    total_train  = gridae_train + hand_ae_train + unet_train

    print(f"\n{'='*60}")
    print("  PARAMETER COUNTS")
    print(f"{'='*60}")
    print(f"  GRIDAE        : {gridae_total/1e6:.3f} M")
    print(f"  HandVAE       : {hand_ae_total/1e6:.3f} M")
    print(f"  DualUNetModel : {unet_total/1e6:.3f} M")
    print(f"  ─────────────────────────────")
    print(f"  Total         : {total_params/1e6:.3f} M  (trainable: {total_train/1e6:.3f} M)")

    # ── 2. HandVAE — encode ───────────────────────────────────────────────────
    enc_hand_macs = _summary(
        HandVAEEncodeWrapper(hand_ae).to(device).eval(),
        (hand_verts,), device, "HandVAE encode  [B,3,778] → z [B,16]",
    )

    # ── 3. HandVAE — decode ───────────────────────────────────────────────────
    dec_hand_macs = _summary(
        HandVAEDecodeWrapper(hand_ae).to(device).eval(),
        (hand_latent,), device, "HandVAE decode  [B,16] → handV [B,778,3]",
    )

    # ── 4. DualUNet — object conditioning ─────────────────────────────────────
    unet_cond_macs = _summary(
        UNetConditionWrapper(unet).to(device).eval(),
        (obj_pc,), device, f"DualUNet condition  [B,{obj_in_dim},N_pts] → obj_feat",
    )

    # Compute obj_feat for the denoising wrapper
    with torch.no_grad():
        local_obj_feat, glob_obj_feat = unet.obj_feat_net(obj_pc)
        obj_feat = torch.cat(
            [local_obj_feat, glob_obj_feat.unsqueeze(2).expand(-1, -1, local_obj_feat.shape[2])],
            dim=1,
        )

    # ── 5. DualUNet — single denoising step ───────────────────────────────────
    unet_step_macs = _summary(
        UNetDenoiseWrapper(unet, obj_feat[:1], ts[:1]).to(device).eval(),
        (x_noisy,), device,
        f"DualUNet single step  [B, {d_y}+{d_x}×{n_pts}] → x_pred",
    )

    # ── 6. GRIDAE — encode (per-grid) ─────────────────────────────────────────
    # Encode a single grid first to get obj_cond for the decode wrapper
    with torch.no_grad():
        posterior_g1, _, obj_cond1 = gridae.encode(lg_contact[:1], obj_sdf[:1])
        z_grid1 = posterior_g1.mode()
        # Also encode all grids to get full z for GPU memory measurement
        posterior_g, _, obj_cond = gridae.encode(lg_contact, obj_sdf)
        z_grid = posterior_g.mode()

    gridae_enc_macs = _summary(
        GRIDAEEncodeWrapper(gridae, obj_sdf[:1]).to(device).eval(),
        (lg_contact[:1],), device,
        f"GRIDAE encode  [B*N,{cfg.ae.in_dim},K,K,K] → z [B*N,{d_x}]  (per grid)",
    )

    # ── 7. GRIDAE — decode (per-grid) ─────────────────────────────────────────
    gridae_dec_macs = _summary(
        GRIDAEDecodeWrapper(gridae, obj_cond1).to(device).eval(),
        (z_grid1,), device,
        f"GRIDAE decode  [B*N,{d_x}] → grid [B*N,c,K,K,K]  (per grid)",
    )

    # Scale per-grid MACs to full batch of N grids
    gridae_enc_macs_full = gridae_enc_macs * N
    gridae_dec_macs_full = gridae_dec_macs * N

    # ── 8. Peak GPU memory ────────────────────────────────────────────────────
    peak_mb_encode = peak_mb_denoise = peak_mb_decode = None
    if device.type == "cuda":
        # Encoding phase: hand_ae.encode + gridae.encode per grid
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = hand_ae.encode(hand_verts)
            _ = gridae.encode(lg_contact, obj_sdf)
        peak_mb_encode = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        # Denoising: one full reverse pass (T steps)
        cond_dict = {'obj_feat': obj_feat}
        cat_noise = torch.randn(B, d_y + d_x * n_pts, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            x = cat_noise
            for t in range(T - 1, -1, -1):
                ts_t = torch.full((B,), t, dtype=torch.long, device=device)
                x, _ = unet(x, ts_t, cond_dict, is_train=False)
        peak_mb_denoise = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        # Decoding phase: hand_ae.decode + gridae.decode per grid
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = hand_ae.decode(hand_latent)
            _ = gridae.decode(z_grid, obj_cond)
        peak_mb_decode = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    # ── 9. Summary ────────────────────────────────────────────────────────────
    gflops = lambda macs: 2 * macs / 1e9

    print(f"\n{'='*60}")
    print("  INFERENCE SUMMARY  (B=1, N=128 grids, T=1000 steps)")
    print(f"{'='*60}")
    print(f"  GRIDAE        : {gridae_total/1e6:.3f} M params")
    print(f"  HandVAE       : {hand_ae_total/1e6:.3f} M params")
    print(f"  DualUNetModel : {unet_total/1e6:.3f} M params")
    print(f"  Total         : {total_params/1e6:.3f} M params")
    print()
    print("  ── GFLOPs per inference call ───────────────────────────")
    print(f"  HandVAE encode           : {gflops(enc_hand_macs):.4f} GFLOPs")
    print(f"  HandVAE decode           : {gflops(dec_hand_macs):.4f} GFLOPs")
    print(f"  DualUNet obj conditioning : {gflops(unet_cond_macs):.4f} GFLOPs")
    print(f"  DualUNet 1 denoise step  : {gflops(unet_step_macs):.4f} GFLOPs")
    print(f"  DualUNet {T} steps total  : {gflops(unet_step_macs * T):.2f} GFLOPs")
    print(f"  GRIDAE encode  (N grids) : {gflops(gridae_enc_macs_full):.4f} GFLOPs")
    print(f"  GRIDAE decode  (N grids) : {gflops(gridae_dec_macs_full):.4f} GFLOPs")
    total_inf_gflops = (
        gflops(enc_hand_macs) + gflops(dec_hand_macs) +
        gflops(unet_cond_macs) + gflops(unet_step_macs * T) +
        gflops(gridae_enc_macs_full) + gflops(gridae_dec_macs_full)
    )
    print(f"  ───────────────────────────────────────────────────────")
    print(f"  Total inference GFLOPs   : {total_inf_gflops:.2f}")
    print()
    if peak_mb_encode is not None:
        print("  ── Peak GPU memory ─────────────────────────────────────")
        print(f"  Encoding phase  : {peak_mb_encode:.1f} MB")
        print(f"  Denoising ({T} t) : {peak_mb_denoise:.1f} MB")
        print(f"  Decoding phase  : {peak_mb_decode:.1f} MB")
    else:
        print("  Peak GPU mem   : N/A (no CUDA)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
