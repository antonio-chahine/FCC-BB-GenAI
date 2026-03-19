#!/usr/bin/env python3
"""
Unified particle generation pipeline with DDPM.

Changes vs base:
  - Cosine beta schedule (replaces linear)
  - asinh transform on beta_x, beta_y, and y position (sharp peak at 0 spread)
  - beta_z uses arctanh unsquash directly (already maps +-1 to +-inf, no logit needed)
  - t^2 timestep sampling (biases toward low-t for better peak recovery)
  - Z-plane discrete decomposition (z split into plane index + residual)
  - OneCycleLR scheduler
  - Z-plane embedding + head in ParticleDenoiser

Modes:
    train    - Train the diffusion model
    sample   - Generate synthetic events from trained model
    evaluate - Compare real vs generated distributions

Usage:
    python particle_diffusion_new.py train --data_path guineapig_raw_trimmed.npy --outdir results
    python particle_diffusion_new.py sample --outdir results --n_events 1000
    python particle_diffusion_new.py evaluate --real_path guineapig_raw_trimmed.npy --gen_path results/generated_events.npy
"""

import os
import math
import argparse
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

# The two interaction-point z-planes in nm (+-32.4 mm)
Z_PLANES = np.array([-32400000.0, 32400000.0], dtype=np.float32)

# asinh scale for beta_x and beta_y  (~std of raw unsquashed beta components)
ASINH_SCALE_XY = 0.1

# asinh scale for y position (~std of raw y values in nm)
ASINH_SCALE_Y = 10000.0


# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    return idx[n_val:], idx[:n_val]


def make_cosine_beta_schedule(T: int, s: float = 0.008, device: str = "cpu"):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = torch.arange(T + 1, device=device, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0, 0.999).float()
    alphas   = 1.0 - betas
    acp      = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    return betas, alphas, acp, acp_prev


def make_linear_gamma_schedule(T: int, g0: float, g1: float, device: str):
    return torch.linspace(g0, g1, T, device=device)


def charge_balance_loss(pdg_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Soft charge balance loss.
    Real data has exactly N(e-) == N(e+) per event.
    Penalises squared expected charge imbalance, differentiable through softmax.
    pdg_logits: (B, K, 2)  index 0=e-, index 1=e+
    mask:       (B, K) bool
    """
    probs    = torch.softmax(pdg_logits, dim=-1)
    p_eminus = probs[:, :, 0] * mask
    p_eplus  = probs[:, :, 1] * mask
    imbalance = (p_eminus - p_eplus).sum(dim=1)
    return (imbalance ** 2).mean()


def q_sample_pdg(pdg0: torch.Tensor, t: torch.Tensor, gammas: torch.Tensor,
                 n_classes: int, mask: torch.Tensor):
    B, K = pdg0.shape
    g = gammas[t].view(B, 1)
    u = torch.rand((B, K), device=pdg0.device)
    flip = (u < g) & mask
    pdg_t = pdg0.clone()
    if flip.any():
        pdg_t[flip] = torch.randint(0, n_classes, (int(flip.sum()),), device=pdg0.device)
    return pdg_t


def q_sample_zplane(zplane0: torch.Tensor, t: torch.Tensor, gammas: torch.Tensor,
                    n_zplane: int, mask: torch.Tensor):
    """Same discrete-corruption process as PDG but for z-plane index."""
    B, K = zplane0.shape
    g = gammas[t].view(B, 1)
    u = torch.rand((B, K), device=zplane0.device)
    flip = (u < g) & mask
    zplane_t = zplane0.clone()
    if flip.any():
        zplane_t[flip] = torch.randint(0, n_zplane, (int(flip.sum()),), device=zplane0.device)
    return zplane_t


# ------------------------------------------------------------------
# Beta transforms
# ------------------------------------------------------------------
def beta_unsquash_np(beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse of tanh-squash: maps beta in (-1,1)^3 to u in R^3."""
    beta = np.asarray(beta, dtype=np.float32)
    bmag = np.linalg.norm(beta, axis=1, keepdims=True)
    bmag = np.clip(bmag, 0.0, 1.0 - eps)
    bhat = beta / (bmag + 1e-12)
    s = bmag / (1.0 - eps)
    umag = np.arctanh(np.clip(s, 0.0, 1.0 - 1e-7))
    return umag * bhat


def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat


def asinh_transform_xy(u_xy: np.ndarray, scale: float = ASINH_SCALE_XY) -> np.ndarray:
    """asinh(u / scale) -- spreads the sharp peak at 0 for beta_x, beta_y."""
    return np.arcsinh(u_xy / scale).astype(np.float32)


def asinh_inverse_xy(v_xy: np.ndarray, scale: float = ASINH_SCALE_XY) -> np.ndarray:
    return (np.sinh(v_xy) * scale).astype(np.float32)


def asinh_transform_y(y: np.ndarray, scale: float = ASINH_SCALE_Y) -> np.ndarray:
    """asinh(y / scale) -- spreads the sharp peak at 0 for y position."""
    return np.arcsinh(y / scale).astype(np.float32)


def asinh_inverse_y(v: np.ndarray, scale: float = ASINH_SCALE_Y) -> np.ndarray:
    return (np.sinh(v) * scale).astype(np.float32)


# ------------------------------------------------------------------
# Z-plane decomposition
# ------------------------------------------------------------------
def assign_z_plane(z_arr: np.ndarray):
    """
    Split z into (plane_idx, residual).
    plane_idx in {0, 1}  (index into Z_PLANES)
    residual  = z - Z_PLANES[plane_idx]
    """
    dists = np.abs(z_arr[:, None] - Z_PLANES[None, :])
    plane_idx = np.argmin(dists, axis=1).astype(np.int64)
    residual  = z_arr - Z_PLANES[plane_idx]
    return plane_idx, residual.astype(np.float32)


def reconstruct_z(plane_idx: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return (Z_PLANES[plane_idx] + residual).astype(np.float32)


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    data_path: str = "guineapig_raw_trimmed.npy"
    outdir: str = "new_41"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    max_particles: int = 1300
    min_particles: int = 1
    keep_fraction: float = 1.0

    # Cosine schedule (replaces linear beta_start/beta_end)
    T: int = 500
    cosine_s: float = 0.0307688247430154

    d_model: int = 512
    nhead: int = 8
    num_layers: int = 5
    dropout: float = 0.2378997883128378

    batch_size: int = 16
    lr: float = 0.0005244136133080424
    epochs: int = 50
    num_workers: int = 8
    grad_clip: float = 2.083015934534223
    seed: int = 123

    frac_range: float = 0.60
    me: float = 0.00051099895069  # GeV

    feat_dim: int = 7   # [logE, u_x_asinh, u_y_asinh, u_z, x, y_asinh, z_residual]
    n_pdg: int = 2
    n_zplane: int = 2
    lambda_pdg: float = 0.13406787694932354
    lambda_zplane: float = 0.47236743859637587
    lambda_charge: float = 0.011241862095793064

    gamma_start: float = 0.0003570478920909981
    gamma_end: float = 0.06130682222763452

    n_events: int = 500
    sample_batch_size: int = 16

    pct_start: float = 0.0907023547660844
    div_factor: float = 23.92275706162204


# ============================================================
# DATASET
# ============================================================
class MCPDataset(Dataset):
    """
    Input rows (7 cols):
        [E_signed, betax, betay, betaz, x, y, z]

    Continuous features stored (7 cols):
        [logE, asinh(u_x/s), asinh(u_y/s), u_z, x, asinh(y/s_y), z_residual]

        beta_z: uses arctanh unsquash (u_z) directly -- already maps +-1 to +-inf
    """

    def __init__(self, path, max_particles=1300, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)

        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]

        self.pdg_to_idx = {11: 0, -11: 1}
        self.idx_to_pdg = {0: 11, 1: -11}

        events_cont   = []
        events_pdg    = []
        events_zplane = []

        for ev in raw:
            if ev is None:
                continue
            ev = np.asarray(ev)
            if ev.ndim != 2 or ev.shape[1] < 7:
                continue
            if len(ev) < min_particles:
                continue

            ev = ev.astype(np.float32)

            E_signed = ev[:, 0]
            beta     = ev[:, 1:4]   # (N,3): beta_x, beta_y, beta_z
            pos      = ev[:, 4:7]   # (N,3): x, y, z

            pdg     = np.where(E_signed >= 0.0, 11, -11)
            pdg_idx = np.where(pdg == 11, 0, 1).astype(np.int64)

            Eabs = np.maximum(np.abs(E_signed), 1e-12)
            logE = np.log(Eabs)

            # unsquash all three beta components via arctanh
            u = beta_unsquash_np(beta)              # (N,3)

            # asinh transform on u_x, u_y (peaks at 0)
            ux_t = asinh_transform_xy(u[:, 0:1])   # (N,1)
            uy_t = asinh_transform_xy(u[:, 1:2])   # (N,1)

            # u_z passed directly -- arctanh already maps beta_z +-1 -> +-inf
            uz_t = u[:, 2:3]                        # (N,1)

            # z-plane decomposition
            zplane_idx, z_residual = assign_z_plane(pos[:, 2])

            x_pos = pos[:, 0:1]
            y_pos = asinh_transform_y(pos[:, 1:2])  # asinh on y (peak at 0)

            cont = np.concatenate(
                [logE[:, None], ux_t, uy_t, uz_t, x_pos, y_pos, z_residual[:, None]],
                axis=1
            ).astype(np.float32)   # (N, 7)

            events_cont.append(cont)
            events_pdg.append(pdg_idx)
            events_zplane.append(zplane_idx)

        if len(events_cont) == 0:
            raise RuntimeError("No events -- check .npy contains (K,7) arrays.")

        self.events_cont   = events_cont
        self.events_pdg    = events_pdg
        self.events_zplane = events_zplane
        self.max_particles = max_particles
        self.feat_dim      = 7

        all_feats = np.concatenate(events_cont, axis=0)
        self.feat_mean = all_feats.mean(axis=0).astype(np.float32)
        self.feat_std  = np.maximum(all_feats.std(axis=0), 1e-6).astype(np.float32)

        self.multiplicities = np.array([len(ev) for ev in events_cont], dtype=np.int64)

    def __len__(self):
        return len(self.events_cont)

    def __getitem__(self, idx):
        cont   = self.events_cont[idx]
        pdg    = self.events_pdg[idx]
        zplane = self.events_zplane[idx]

        N    = len(cont)
        Kmax = self.max_particles

        if N <= Kmax:
            chosen = np.arange(N)
        else:
            chosen = torch.randperm(N)[:Kmax].numpy()

        cont   = cont[chosen]
        pdg    = pdg[chosen]
        zplane = zplane[chosen]
        K = cont.shape[0]

        cont_norm = (cont - self.feat_mean) / self.feat_std

        x0      = np.zeros((Kmax, self.feat_dim), dtype=np.float32)
        pdg0    = np.zeros((Kmax,), dtype=np.int64)
        zplane0 = np.zeros((Kmax,), dtype=np.int64)
        mask    = np.zeros((Kmax,), dtype=np.bool_)

        x0[:K]      = cont_norm
        pdg0[:K]    = pdg
        zplane0[:K] = zplane
        mask[:K]    = True

        return (torch.from_numpy(x0),
                torch.from_numpy(pdg0),
                torch.from_numpy(zplane0),
                torch.from_numpy(mask))


# ============================================================
# MODEL COMPONENTS
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor):
        device = t.device
        half   = self.d_model // 2
        freqs  = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ParticleDenoiser(nn.Module):
    def __init__(self, d_model=256, nhead=2, num_layers=3, dropout=0.1,
                 n_pdg=2, n_zplane=2):
        super().__init__()
        self.d_model = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.mom_proj = nn.Linear(7, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, 7)

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        self.pdg_emb    = nn.Embedding(n_pdg,    d_model)
        self.zplane_emb = nn.Embedding(n_zplane, d_model)

        self.pdg_head     = nn.Linear(d_model, n_pdg)
        self.zplane_head  = nn.Linear(d_model, n_zplane)

        self.skip_alpha = 0.2

    def forward(self, x_t, t, pdg_t, zplane_t, mask):
        B, K, _ = x_t.shape

        t_emb = self.time_emb(t)
        t_emb = self.t_mlp(t_emb).unsqueeze(1).expand(B, K, self.d_model)

        mom_emb    = self.mom_proj(x_t)
        pdg_emb    = self.pdg_emb(pdg_t.clamp(0, self.pdg_emb.num_embeddings - 1))
        zplane_emb = self.zplane_emb(zplane_t.clamp(0, self.zplane_emb.num_embeddings - 1))

        h = t_emb + mom_emb + pdg_emb + zplane_emb

        K_event = mask.sum(dim=1)
        k = torch.log(K_event.float().clamp(min=1)).unsqueeze(-1)
        k_emb = self.k_mlp(k).unsqueeze(1)
        h = h + k_emb

        h_in = h
        h = self.transformer(h, src_key_padding_mask=~mask)
        h = h + self.skip_alpha * h_in
        h = h * mask.unsqueeze(-1)

        eps_hat       = self.output(h) * mask.unsqueeze(-1)
        pdg_logits    = self.pdg_head(h)
        zplane_logits = self.zplane_head(h)

        return eps_hat, pdg_logits, zplane_logits


# ============================================================
# DDPM WRAPPER
# ============================================================
class DDPM:
    def __init__(self, model, T, device, cosine_s=0.008):
        self.model  = model
        self.T      = T
        self.device = device

        betas, alphas, acp, acp_prev = make_cosine_beta_schedule(T, s=cosine_s, device=device)

        self.betas    = betas
        self.alphas   = alphas
        self.acp      = acp
        self.acp_prev = acp_prev

        self.sqrt_acp           = torch.sqrt(acp)
        self.sqrt_1m_acp        = torch.sqrt(1.0 - acp)
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise

    def p_sample(self, x_t, t, pdg_t, zplane_t, mask):
        B = x_t.shape[0]

        eps_hat, pdg_logits, zplane_logits = self.model(x_t, t, pdg_t, zplane_t, mask)

        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)

        mu  = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat
        )
        var = self.posterior_variance[t].view(B, 1, 1)
        z   = torch.zeros_like(x_t) if t[0].item() == 0 else torch.randn_like(x_t)

        x_prev = (mu + torch.sqrt(var) * z) * mask.unsqueeze(-1)
        return x_prev, pdg_logits, zplane_logits

    @torch.no_grad()
    def sample(self, mask, pdg_init, zplane_init):
        B, K   = mask.shape
        x      = torch.randn((B, K, 7), device=self.device) * mask.unsqueeze(-1)
        pdg    = pdg_init.clone()
        zplane = zplane_init.clone()

        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)

            x, pdg_logits, zplane_logits = self.p_sample(x, t, pdg, zplane, mask)

            pdg_probs    = torch.softmax(pdg_logits, dim=-1)
            pdg_samp     = torch.multinomial(pdg_probs.view(-1, pdg_probs.size(-1)), 1).view(B, K)
            pdg          = torch.where(mask, pdg_samp, pdg)

            zp_probs     = torch.softmax(zplane_logits, dim=-1)
            zp_samp      = torch.multinomial(zp_probs.view(-1, zp_probs.size(-1)), 1).view(B, K)
            zplane       = torch.where(mask, zp_samp, zplane)

        return x, pdg, zplane


# ============================================================
# TRAINING
# ============================================================
def train(args):
    cfg = CFG()

    print("Using device:", cfg.device)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    if args.data_path:     cfg.data_path     = args.data_path
    if args.outdir:        cfg.outdir        = args.outdir
    if args.max_particles: cfg.max_particles = args.max_particles
    if args.epochs:        cfg.epochs        = args.epochs
    if args.batch_size:    cfg.batch_size    = args.batch_size
    if args.T:             cfg.T             = args.T
    if args.seed:          cfg.seed          = args.seed

    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    val_frac = 0.1
    ds_full  = MCPDataset(cfg.data_path, max_particles=cfg.max_particles,
                          min_particles=cfg.min_particles, keep_fraction=cfg.keep_fraction)

    meta_path = os.path.join(cfg.outdir, "meta.pt")
    if getattr(args, "resume", False) and os.path.exists(meta_path):
        meta_old = torch.load(meta_path, map_location="cpu")
        if "train_idx" in meta_old and "val_idx" in meta_old:
            train_idx = np.asarray(meta_old["train_idx"])
            val_idx   = np.asarray(meta_old["val_idx"])
        else:
            train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)
    else:
        train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)

    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    model = ParticleDenoiser(
        d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
        n_pdg=cfg.n_pdg, n_zplane=cfg.n_zplane,
    ).to(cfg.device)

    ddpm   = DDPM(model, cfg.T, cfg.device, cosine_s=cfg.cosine_s)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(dl_train),
        pct_start=cfg.pct_start, anneal_strategy='cos',
        div_factor=cfg.div_factor, final_div_factor=1e4,
    )

    start_epoch  = 0
    train_losses = []
    val_losses   = []

    ckpt_path = os.path.join(cfg.outdir, "ckpt_last.pt")
    if getattr(args, "resume", False) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt and ckpt["opt"] is not None:
            opt.load_state_dict(ckpt["opt"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

        tl_path = os.path.join(cfg.outdir, "train_losses.npy")
        vl_path = os.path.join(cfg.outdir, "val_losses.npy")
        if os.path.exists(tl_path) and os.path.exists(vl_path):
            train_losses = list(np.load(tl_path).astype(float))
            val_losses   = list(np.load(vl_path).astype(float))
        print(f"Resuming from {ckpt_path} at epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    meta = {
        "multiplicities": ds_full.multiplicities,
        "feat_mean":      ds_full.feat_mean,
        "feat_std":       ds_full.feat_std,
        "feat_dim":       ds_full.feat_dim,
        "me":             cfg.me,
        "n_pdg":          cfg.n_pdg,
        "n_zplane":       cfg.n_zplane,
        "idx_to_pdg":     ds_full.idx_to_pdg,
        "max_particles":  cfg.max_particles,
        "T":              cfg.T,
        "cosine_s":       cfg.cosine_s,
        "d_model":        cfg.d_model,
        "nhead":          cfg.nhead,
        "num_layers":     cfg.num_layers,
        "dropout":        cfg.dropout,
        "val_frac":       val_frac,
        "seed":           cfg.seed,
        "n_events":       len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events":   len(val_idx),
        "train_idx":      train_idx,
        "val_idx":        val_idx,
        "asinh_scale_xy": ASINH_SCALE_XY,
        "asinh_scale_y":  ASINH_SCALE_Y,
        "z_planes":       Z_PLANES,
    }
    if not os.path.exists(meta_path):
        torch.save(meta, meta_path)

    for epoch in range(start_epoch, cfg.epochs):
        # ---- TRAIN ----
        model.train()
        total_train = 0.0
        n_train = 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [train]", leave=False)
        for x0, pdg0, zplane0, mask in pbar:
            x0      = x0.to(cfg.device)
            pdg0    = pdg0.to(cfg.device)
            zplane0 = zplane0.to(cfg.device)
            mask    = mask.to(cfg.device)

            B = x0.shape[0]

            # t^2 sampling -- bias toward low-t for better peak recovery
            u = torch.rand(B, device=cfg.device)
            t = (u ** 2 * cfg.T).long().clamp(0, cfg.T - 1)

            noise    = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t      = ddpm.q_sample(x0, t, noise)
            pdg_t    = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
            zplane_t = q_sample_zplane(zplane0, t, gammas, cfg.n_zplane, mask)

            eps_hat, pdg_logits, zplane_logits = model(x_t, t, pdg_t, zplane_t, mask)

            mse         = (eps_hat - noise).pow(2).sum(dim=-1)
            diff_loss   = (mse * mask).sum() / mask.sum().clamp(min=1)
            pdg_loss    = F.cross_entropy(pdg_logits[mask], pdg0[mask])
            zplane_loss = F.cross_entropy(zplane_logits[mask], zplane0[mask])
            c_loss      = charge_balance_loss(pdg_logits, mask)
            loss        = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_zplane * zplane_loss + cfg.lambda_charge * c_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()

            total_train += loss.item()
            n_train     += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = total_train / max(n_train, 1)

        # ---- VALIDATION ----
        model.eval()
        total_val = 0.0
        n_val = 0

        with torch.no_grad():
            pbarv = tqdm(dl_val, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [val]", leave=False)
            for x0, pdg0, zplane0, mask in pbarv:
                x0      = x0.to(cfg.device)
                pdg0    = pdg0.to(cfg.device)
                zplane0 = zplane0.to(cfg.device)
                mask    = mask.to(cfg.device)

                B = x0.shape[0]

                u = torch.rand(B, device=cfg.device)
                t = (u ** 2 * cfg.T).long().clamp(0, cfg.T - 1)

                noise    = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t      = ddpm.q_sample(x0, t, noise)
                pdg_t    = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
                zplane_t = q_sample_zplane(zplane0, t, gammas, cfg.n_zplane, mask)

                eps_hat, pdg_logits, zplane_logits = model(x_t, t, pdg_t, zplane_t, mask)

                mse         = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss   = (mse * mask).sum() / mask.sum().clamp(min=1)
                pdg_loss    = F.cross_entropy(pdg_logits[mask], pdg0[mask])
                zplane_loss = F.cross_entropy(zplane_logits[mask], zplane0[mask])
                c_loss      = charge_balance_loss(pdg_logits, mask)
                loss        = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_zplane * zplane_loss + cfg.lambda_charge * c_loss

                total_val += loss.item()
                n_val     += 1
                pbarv.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = total_val / max(n_val, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(cfg.outdir, "val_losses.npy"),   np.array(val_losses))

        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict(),
             "scheduler": scheduler.state_dict(), "epoch": epoch,
             "train_loss": train_loss, "val_loss": val_loss},
            os.path.join(cfg.outdir, "ckpt_last.pt"),
        )

    np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cfg.outdir, "val_losses.npy"),   np.array(val_losses))
    print("Training complete. Outputs saved to:", cfg.outdir)


# ============================================================
# SAMPLING
# ============================================================
def load_meta_and_model(outdir: str, device: str):
    meta = torch.load(os.path.join(outdir, "meta.pt"), map_location="cpu")

    model = ParticleDenoiser(
        d_model=meta["d_model"], nhead=meta["nhead"],
        num_layers=meta["num_layers"], dropout=meta["dropout"],
        n_pdg=int(meta.get("n_pdg", 2)),
        n_zplane=int(meta.get("n_zplane", 2)),
    ).to(device)

    ckpt = torch.load(os.path.join(outdir, "ckpt_last.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(model=model, T=int(meta["T"]), device=device,
                cosine_s=float(meta.get("cosine_s", 0.008)))
    return meta, model, ddpm


def _decode_batch(x_norm_batch, pdg_idx_batch, zplane_idx_batch, Ks, meta):
    """
    Convert normalised model outputs to physical units.
    Returns list of (K_i, 8) arrays: [pdg, E, betax, betay, betaz, x, y, z]
    """
    mean        = np.asarray(meta["feat_mean"], dtype=np.float32)
    std         = np.asarray(meta["feat_std"],  dtype=np.float32)
    idx_to_pdg  = meta["idx_to_pdg"]
    asinh_scale    = float(meta.get("asinh_scale_xy", ASINH_SCALE_XY))
    asinh_scale_y  = float(meta.get("asinh_scale_y",  ASINH_SCALE_Y))

    events = []
    for i, K in enumerate(Ks):
        x_i      = x_norm_batch[i, :K]
        pdg_i    = pdg_idx_batch[i, :K]
        zplane_i = zplane_idx_batch[i, :K]

        cont = x_i * std + mean
        # cols: [logE, u_x_asinh, u_y_asinh, u_z, x, y_asinh, z_residual]
        logE       = cont[:, 0]
        ux_asinh   = cont[:, 1]
        uy_asinh   = cont[:, 2]
        u_z        = cont[:, 3]
        x_pos      = cont[:, 4]
        y_asinh    = cont[:, 5]
        z_residual = cont[:, 6]

        E  = np.exp(logE)

        # invert asinh for u_x, u_y, then squash to beta_x, beta_y
        ux = asinh_inverse_xy(ux_asinh, scale=asinh_scale)
        uy = asinh_inverse_xy(uy_asinh, scale=asinh_scale)
        u_xy    = np.stack([ux, uy, np.zeros_like(ux)], axis=1)
        beta_xy = beta_squash_np(u_xy)
        betax   = beta_xy[:, 0]
        betay   = beta_xy[:, 1]

        # u_z -> beta_z via tanh squash (inverse of arctanh unsquash)
        bz = np.tanh(u_z).astype(np.float32)

        # invert asinh for y position
        y_pos = asinh_inverse_y(y_asinh, scale=asinh_scale_y)

        # reconstruct z from plane + residual
        z_pos = reconstruct_z(zplane_i, z_residual)

        pdg = np.array([idx_to_pdg[int(j)] for j in pdg_i], dtype=np.int64)

        out = np.stack(
            [pdg.astype(np.float32), E, betax, betay, bz, x_pos, y_pos, z_pos],
            axis=1
        ).astype(np.float32)
        events.append(out)

    return events


def sample_batch(meta: dict, ddpm: DDPM, device: str, batch_size: int):
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    Kmax     = int(meta["max_particles"])
    n_pdg    = int(meta.get("n_pdg", 2))
    n_zplane = int(meta.get("n_zplane", 2))

    Ks = np.clip(np.random.choice(multiplicities, size=batch_size, replace=True), 1, Kmax)

    mask_np = np.zeros((batch_size, Kmax), dtype=np.bool_)
    for i, K in enumerate(Ks):
        mask_np[i, :K] = True
    mask_t = torch.from_numpy(mask_np).to(device)

    pdg_init    = torch.randint(0, n_pdg,    (batch_size, Kmax), device=device) * mask_t.long()
    zplane_init = torch.randint(0, n_zplane, (batch_size, Kmax), device=device) * mask_t.long()

    with torch.no_grad():
        x_norm, pdg_idx, zplane_idx = ddpm.sample(mask_t, pdg_init, zplane_init)

    return _decode_batch(
        x_norm.cpu().numpy(), pdg_idx.cpu().numpy(), zplane_idx.cpu().numpy(), Ks, meta
    )


def sample_single(meta: dict, ddpm: DDPM, device: str):
    return sample_batch(meta, ddpm, device, batch_size=1)[0]


def sample(args):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    meta, model, ddpm = load_meta_and_model(args.outdir, device)

    events = []
    n_done = 0
    for _ in tqdm(range(math.ceil(args.n_events / args.sample_batch_size)),
                  desc="Generating batches"):
        bs = min(args.sample_batch_size, args.n_events - n_done)
        events.extend(sample_batch(meta, ddpm, device, bs))
        n_done += bs

    out_path = os.path.join(args.outdir, "generated_events.npy")
    np.save(out_path, np.array(events, dtype=object))
    print("Saved:", out_path)


# ============================================================
# EVALUATION
# ============================================================
try:
    from scipy.stats import wasserstein_distance
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import corner as _corner
    HAVE_CORNER = True
except Exception:
    HAVE_CORNER = False


def plot_corner_overlay(real_dict, gen_dict, keys, labels, outpath, title="",
                        max_points=40000, q_lo=0.01, q_hi=0.99, seed=123, bins=40):
    cols_r = [np.asarray(real_dict[k], dtype=np.float64) for k in keys]
    cols_g = [np.asarray(gen_dict[k],  dtype=np.float64) for k in keys]
    R = np.stack(cols_r, axis=1)
    G = np.stack(cols_g, axis=1)
    R = R[np.all(np.isfinite(R), axis=1)]
    G = G[np.all(np.isfinite(G), axis=1)]
    if R.shape[0] == 0 or G.shape[0] == 0:
        return

    rng = np.random.default_rng(seed)
    if R.shape[0] > max_points:
        R = R[rng.choice(R.shape[0], size=max_points, replace=False)]
    if G.shape[0] > max_points:
        G = G[rng.choice(G.shape[0], size=max_points, replace=False)]

    ranges = []
    for d in range(len(keys)):
        z  = np.concatenate([R[:, d], G[:, d]])
        ranges.append((np.quantile(z, q_lo), np.quantile(z, q_hi)))

    fig = _corner.corner(R, labels=labels, range=ranges, bins=bins, smooth=1.0,
                         plot_density=True, plot_contours=True, fill_contours=True,
                         levels=(0.68, 0.95), color="C0", label_kwargs={"fontsize": 11})
    _corner.corner(G, fig=fig, range=ranges, bins=bins, smooth=1.0,
                   plot_density=False, plot_contours=True, fill_contours=False,
                   levels=(0.68, 0.95), color="C1")
    if title:
        fig.suptitle(title, y=1.02)
    savefig(fig, outpath)


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white", "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "legend.frameon": True, "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc", "axes.edgecolor": "black",
        "axes.linewidth": 1.0, "axes.grid": False,
    })


def savefig(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def load_events(path: str):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unrecognized format in {path}")


def sanitize_event(ev, me=0.000511):
    ev = np.asarray(ev)

    if ev.ndim == 2 and ev.shape[1] >= 8:
        pdg   = ev[:, 0].astype(np.int64, copy=False)
        Eabs  = np.abs(ev[:, 1].astype(np.float64, copy=False))
        betax = ev[:, 2].astype(np.float64, copy=False)
        betay = ev[:, 3].astype(np.float64, copy=False)
        betaz = ev[:, 4].astype(np.float64, copy=False)
        x = ev[:, 5].astype(np.float64, copy=False)
        y = ev[:, 6].astype(np.float64, copy=False)
        z = ev[:, 7].astype(np.float64, copy=False)
        beta     = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)
        pvec     = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
        E_signed = np.where(pdg == -11, -Eabs, Eabs)
        return (pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz)

    if ev.ndim == 2 and ev.shape[1] >= 7:
        E_signed = ev[:, 0].astype(np.float64, copy=False)
        betax = ev[:, 1].astype(np.float64, copy=False)
        betay = ev[:, 2].astype(np.float64, copy=False)
        betaz = ev[:, 3].astype(np.float64, copy=False)
        x = ev[:, 4].astype(np.float64, copy=False)
        y = ev[:, 5].astype(np.float64, copy=False)
        z = ev[:, 6].astype(np.float64, copy=False)
        pdg      = np.where(E_signed >= 0.0, 11, -11).astype(np.int64)
        Eabs     = np.abs(E_signed)
        beta     = np.stack([betax, betay, betaz], axis=1)
        beta_mag = np.linalg.norm(beta, axis=1)
        pvec     = Eabs[:, None] * beta
        px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
        return (pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz)

    empty = np.array([], dtype=np.float64)
    return (empty.astype(np.int64),
            empty, empty, empty, empty, empty, empty,
            empty, empty, empty, empty, empty, empty)


def extract_species(events, pdgs=None, me=0.000511):
    mult = np.zeros(len(events), dtype=np.int64)
    px_list, py_list, pz_list = [], [], []
    E_list, Esigned_list, bmag_list = [], [], []
    x_list, y_list, z_list = [], [], []
    bx_list, by_list, bz_list = [], [], []

    for i, ev in enumerate(events):
        pdg, px, py, pz, Eabs, E_signed, bmag, x, y, z, betax, betay, betaz = \
            sanitize_event(ev, me=me)

        if pdgs is None:
            sel = np.ones(len(px), dtype=bool)
        else:
            sel = np.zeros(len(px), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)

        mult[i] = int(np.sum(sel))
        if np.any(sel):
            px_list.append(px[sel]); py_list.append(py[sel]); pz_list.append(pz[sel])
            E_list.append(Eabs[sel]); Esigned_list.append(E_signed[sel])
            bmag_list.append(bmag[sel])
            bx_list.append(betax[sel]); by_list.append(betay[sel]); bz_list.append(betaz[sel])
            if x.size:
                x_list.append(x[sel]); y_list.append(y[sel]); z_list.append(z[sel])

    def cat(lst): return np.concatenate(lst) if lst else np.array([], dtype=np.float64)

    px_all = cat(px_list); py_all = cat(py_list); pz_all = cat(pz_list)
    p_all  = np.sqrt(px_all**2 + py_all**2 + pz_all**2)
    pt_all = np.sqrt(px_all**2 + py_all**2)

    return {
        "mult": mult,
        "px": px_all, "py": py_all, "pz": pz_all, "p": p_all, "pt": pt_all,
        "E": cat(E_list), "E_abs": cat(E_list),
        "E_signed": cat(Esigned_list), "beta_mag": cat(bmag_list),
        "x": cat(x_list), "y": cat(y_list), "z": cat(z_list),
        "betax": cat(bx_list), "betay": cat(by_list), "betaz": cat(bz_list),
    }


def peak_centered_range(x, y, bins=400, frac=0.995, min_width=1e-12,
                        q_lo=0.001, q_hi=0.999):
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size: vals.append(arr)
    if not vals: return None

    z = np.concatenate(vals)
    if z.size < 2: return None

    zmin = float(np.quantile(z, q_lo))
    zmax = float(np.quantile(z, q_hi))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or (zmax - zmin) < min_width:
        return None

    counts, edges = np.histogram(z, bins=bins, range=(zmin, zmax))
    tot = counts.sum()
    if tot == 0: return None

    i0   = int(np.argmax(counts))
    lo_i = hi_i = i0
    cum  = int(counts[i0])
    target = frac * tot

    while cum < target and (lo_i > 0 or hi_i < len(counts) - 1):
        left  = counts[lo_i - 1] if lo_i > 0 else -1
        right = counts[hi_i + 1] if hi_i < len(counts) - 1 else -1
        if left >= right and lo_i > 0:
            lo_i -= 1; cum += counts[lo_i]
        elif hi_i < len(counts) - 1:
            hi_i += 1; cum += counts[hi_i]
        else:
            break

    return (float(edges[lo_i]), float(edges[hi_i + 1]))


def robust_range(x, y, q_lo=0.005, q_hi=0.995):
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size: vals.append(arr)
    if not vals: return None

    z = np.concatenate(vals)
    if z.size < 2: return None

    lo = float(np.quantile(z, q_lo))
    hi = float(np.quantile(z, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    return (lo, hi)


def clamp_to_range(arr, lo, hi):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= lo) & (arr <= hi)]


def kl_divergence_from_counts(p_counts, q_counts, eps=1e-12):
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    p = np.clip(p / (p.sum() + eps), eps, None)
    q = np.clip(q / (q.sum() + eps), eps, None)
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d(x, y):
    if len(x) == 0 or len(y) == 0: return np.nan
    if HAVE_SCIPY: return float(wasserstein_distance(x, y))
    xs = np.sort(np.asarray(x, dtype=np.float64))
    ys = np.sort(np.asarray(y, dtype=np.float64))
    q  = np.linspace(0.0, 1.0, 400)
    xq = np.interp(q, np.linspace(0.0, 1.0, len(xs)), xs)
    yq = np.interp(q, np.linspace(0.0, 1.0, len(ys)), ys)
    return float(np.mean(np.abs(xq - yq)))


def plot_multiplicity(real_mult, gen_mult, outpath, species_name, n_real, n_gen,
                      bins=50, logy=False):
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8), constrained_layout=True)
    rng = robust_range(real_mult, gen_mult, q_lo=0.0, q_hi=1.0)
    if rng is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); savefig(fig, outpath); return
    lo, hi = rng
    ax.hist(clamp_to_range(real_mult, lo, hi), bins=bins, range=(lo, hi),
            density=True, alpha=0.55, label="Real")
    ax.hist(clamp_to_range(gen_mult, lo, hi),  bins=bins, range=(lo, hi),
            density=True, histtype="step", linewidth=1.8, label="Generated")
    ax.set_title(f"Multiplicity of {species_name} | real={n_real} gen={n_gen}", pad=10)
    ax.set_xlabel(f"Multiplicity N({species_name}) per event")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    if logy: ax.set_yscale("log")
    savefig(fig, outpath)


def two_panel_dist(real, gen, outpath, xlabel, title, species_name, bins=80,
                   ratio_min_count=10, frac_range=0.80, fixed_range=None):
    fig = plt.figure(figsize=(7.6, 6.2), constrained_layout=True)
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(labelbottom=False)

    if fixed_range is not None:
        lo, hi = fixed_range
    elif xlabel.startswith("|E|"):
        z  = np.concatenate([real, gen])
        z  = z[np.isfinite(z)]
        lo = 0.0
        hi = float(np.quantile(z, frac_range))
        if not np.isfinite(hi) or hi <= lo:
            hi = float(np.quantile(z, 0.99))
    else:
        rng = peak_centered_range(real, gen, bins=600, frac=frac_range)
        if rng is None:
            ax_top.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_top.transAxes)
            ax_top.set_axis_off(); ax_bot.set_axis_off()
            savefig(fig, outpath); return
        lo, hi = rng

    real_use = clamp_to_range(real, lo, hi)
    gen_use  = clamp_to_range(gen,  lo, hi)

    if real_use.size == 0 or gen_use.size == 0:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax_top.transAxes)
        ax_top.set_axis_off(); ax_bot.set_axis_off()
        savefig(fig, outpath); return

    r_counts, edges = np.histogram(real_use, bins=bins, range=(lo, hi), density=False)
    g_counts, _     = np.histogram(gen_use,  bins=bins, range=(lo, hi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ax_top.hist(real_use, bins=bins, range=(lo, hi), density=True,
                alpha=0.55, label="Real Data")
    ax_top.hist(gen_use,  bins=bins, range=(lo, hi), density=True,
                histtype="step", linewidth=1.8, label="Generated Data")
    ax_top.set_title(f"{title} -- {species_name}", pad=10)
    ax_top.set_ylabel("Density", labelpad=10)
    ax_bot.set_ylabel("Frac. diff.", labelpad=10)
    ax_top.legend(loc="upper left")

    kl = kl_divergence_from_counts(r_counts, g_counts)
    wd = wasserstein_1d(real_use, gen_use)
    ax_top.text(0.98, 0.95, f"KL: {kl:.4f}\nW1: {wd:.4f}",
                transform=ax_top.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white",
                          edgecolor="#bbbbbb", alpha=0.95))

    r_prob   = r_counts / max(r_counts.sum(), 1)
    g_prob   = g_counts / max(g_counts.sum(), 1)
    mask_r   = r_counts >= ratio_min_count
    frac     = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac_err = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac[mask_r]     = (g_prob[mask_r] - r_prob[mask_r]) / r_prob[mask_r]
    frac_err[mask_r] = 1.0 / np.sqrt(r_counts[mask_r])

    ax_bot.axhline(0.0, linewidth=1.0, color='black')
    ax_bot.axhspan(-0.1, 0.1, color="gray", alpha=0.15, zorder=0)
    ax_bot.errorbar(centers[mask_r], frac[mask_r], yerr=frac_err[mask_r],
                    fmt="o", markersize=3, linewidth=1.0, capsize=0)
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylim(-1.0, 1.0)
    ax_bot.text(0.98, 0.85, "+-10% band", transform=ax_bot.transAxes,
                ha="right", va="center", fontsize=10)
    savefig(fig, outpath)


def evaluate(args):
    setup_style()
    cfg = CFG()

    if args.outdir is None:
        args.outdir = os.path.dirname(args.gen_path)
    os.makedirs(args.outdir, exist_ok=True)

    real_events = load_events(args.real_path)
    gen_events  = load_events(args.gen_path)
    n_real = len(real_events)
    n_gen  = len(gen_events)
    print(f"Loaded {n_real} real events and {n_gen} generated events")

    species_list = [
        {"name": "e-",  "pdgs": [11],  "tag": "eminus"},
        {"name": "e+",  "pdgs": [-11], "tag": "eplus"},
        {"name": "all", "pdgs": None,  "tag": "all"},
    ]

    plot_keys_all = [
        ("E_signed", "E (signed) [GeV]"),
        ("betax", r"$\beta_x$"), ("betay", r"$\beta_y$"), ("betaz", r"$\beta_z$"),
        ("x", "x [nm]"), ("y", "y [nm]"), ("z", "z [nm]"),
        ("px", "p_x [GeV]"), ("py", "p_y [GeV]"), ("pz", "p_z [GeV]"),
        ("pt", "p_T [GeV]"), ("p", "|p| [GeV]"),
    ]
    plot_keys_charge = [
        ("E_abs", "|E| [GeV]"),
        ("betax", r"$\beta_x$"), ("betay", r"$\beta_y$"), ("betaz", r"$\beta_z$"),
        ("x", "x [nm]"), ("y", "y [nm]"), ("z", "z [nm]"),
        ("px", "p_x [GeV]"), ("py", "p_y [GeV]"), ("pz", "p_z [GeV]"),
        ("pt", "p_T [GeV]"), ("p", "|p| [GeV]"),
    ]

    for sp in species_list:
        print(f"Processing species: {sp['name']}")
        real_sp = extract_species(real_events, sp["pdgs"], me=cfg.me)
        gen_sp  = extract_species(gen_events,  sp["pdgs"], me=cfg.me)

        corner_sets = [
            (["px", "py", "pz"],         ["p_x", "p_y", "p_z"],         "p_xyz"),
            (["betax", "betay", "betaz"], [r"$\beta_x$", r"$\beta_y$", r"$\beta_z$"], "beta_xyz"),
        ]
        if real_sp["x"].size and gen_sp["x"].size:
            corner_sets.append((["x", "y", "z"], ["x", "y", "z"], "xyz"))

        for keys, labels, tag2 in corner_sets:
            if any(real_sp[k].size == 0 or gen_sp[k].size == 0 for k in keys):
                continue
            plot_corner_overlay(
                real_dict=real_sp, gen_dict=gen_sp,
                keys=keys, labels=labels,
                outpath=os.path.join(args.outdir, f"corner_{tag2}_{sp['tag']}.png"),
                title=f"Corner: {tag2} -- {sp['name']} | real={n_real} gen={n_gen}",
                max_points=30000, q_lo=0.01, q_hi=0.99, seed=cfg.seed,
            )

        plot_multiplicity(
            real_sp["mult"], gen_sp["mult"],
            outpath=os.path.join(args.outdir, f"multiplicity_{sp['tag']}.png"),
            species_name=sp["name"], n_real=n_real, n_gen=n_gen,
            bins=args.mult_bins, logy=False,
        )

        plot_keys = plot_keys_all if sp["tag"] == "all" else plot_keys_charge
        for key, xlabel in plot_keys:
            if key not in real_sp or key not in gen_sp:
                continue
            if real_sp[key].size == 0 or gen_sp[key].size == 0:
                continue

            frac  = cfg.frac_range
            if key in ("x", "y", "z"):       frac = 0.98
            if key in ("E_signed", "E_abs"): frac = 0.4
            fixed = (-1.0, 1.0) if key == "betaz" else None

            two_panel_dist(
                real_sp[key], gen_sp[key],
                outpath=os.path.join(args.outdir, f"{key}_{sp['tag']}.png"),
                xlabel=xlabel,
                title=f"Comparison of {key} | real={n_real} gen={n_gen}",
                species_name=sp["name"],
                bins=args.mom_bins,
                ratio_min_count=args.ratio_min_count,
                frac_range=frac,
                fixed_range=fixed,
            )

    print(f"Evaluation complete. Plots saved to: {args.outdir}/")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG()

    parser = argparse.ArgumentParser(
        description="Particle diffusion model - train, sample, or evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--data_path',     type=str)
    train_parser.add_argument('--outdir',        type=str)
    train_parser.add_argument('--max_particles', type=int)
    train_parser.add_argument('--epochs',        type=int)
    train_parser.add_argument('--batch_size',    type=int)
    train_parser.add_argument('--T',             type=int)
    train_parser.add_argument('--seed',          type=int)
    train_parser.add_argument('--resume', action='store_true')

    sample_parser = subparsers.add_parser('sample')
    sample_parser.add_argument('--outdir',            type=str, default=cfg.outdir)
    sample_parser.add_argument('--n_events',          type=int, default=cfg.n_events)
    sample_parser.add_argument('--sample_batch_size', type=int, default=cfg.sample_batch_size)

    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--real_path',       type=str, default=cfg.data_path)
    eval_parser.add_argument('--gen_path',        type=str,
                             default=os.path.join(cfg.outdir, "generated_events.npy"))
    eval_parser.add_argument('--outdir',          type=str,
                             default=os.path.join(cfg.outdir, "plots"))
    eval_parser.add_argument('--include_all',     action='store_true')
    eval_parser.add_argument('--mult_bins',       type=int, default=50)
    eval_parser.add_argument('--mom_bins',        type=int, default=80)
    eval_parser.add_argument('--ratio_min_count', type=int, default=10)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'evaluate':
        from particle_diffusion_eval import evaluate
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()