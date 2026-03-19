#!/usr/bin/env python3
"""
GNN-based particle generation pipeline with DDPM.

Replaces the Transformer denoiser with an EdgeConv GNN denoiser.
Edges: fully connected within each event (all particles talk to all others).
Everything else kept identical to particle_diffusion_new.py:
  - Cosine beta schedule
  - asinh transform on beta_x, beta_y (SCALE=0.1), and y position
  - beta_z uses arctanh unsquash directly
  - t^2 timestep sampling
  - Z-plane discrete decomposition
  - OneCycleLR scheduler

Requires: torch_geometric

Usage:
    python particle_diffusion_gnn.py train --data_path guineapig_raw_trimmed.npy --outdir gnn_results
    python particle_diffusion_gnn.py sample --outdir gnn_results --n_events 1000
    python particle_diffusion_gnn.py evaluate --real_path guineapig_raw_trimmed.npy --gen_path gnn_results/generated_events.npy
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

from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_geometric.data import Data, Batch

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
Z_PLANES = np.array([-32400000.0, 32400000.0], dtype=np.float32)

# Corrected: set to ~std of unsquashed beta_x/beta_y (~0.44 measured)
ASINH_SCALE_XY = 0.1

# asinh scale for y position
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
    steps = torch.arange(T + 1, device=device, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0, 0.999).float()
    alphas  = 1.0 - betas
    acp     = torch.cumprod(alphas, dim=0)
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


def q_sample_pdg(pdg0, t, gammas, n_classes, mask):
    B, K = pdg0.shape
    g = gammas[t].view(B, 1)
    u = torch.rand((B, K), device=pdg0.device)
    flip = (u < g) & mask
    pdg_t = pdg0.clone()
    if flip.any():
        pdg_t[flip] = torch.randint(0, n_classes, (int(flip.sum()),), device=pdg0.device)
    return pdg_t


def q_sample_zplane(zplane0, t, gammas, n_zplane, mask):
    B, K = zplane0.shape
    g = gammas[t].view(B, 1)
    u = torch.rand((B, K), device=zplane0.device)
    flip = (u < g) & mask
    zplane_t = zplane0.clone()
    if flip.any():
        zplane_t[flip] = torch.randint(0, n_zplane, (int(flip.sum()),), device=zplane0.device)
    return zplane_t


# ------------------------------------------------------------------
# Feature transforms
# ------------------------------------------------------------------
def beta_unsquash_np(beta, eps=1e-6):
    beta = np.asarray(beta, dtype=np.float32)
    bmag = np.linalg.norm(beta, axis=1, keepdims=True)
    bmag = np.clip(bmag, 0.0, 1.0 - eps)
    bhat = beta / (bmag + 1e-12)
    s = bmag / (1.0 - eps)
    umag = np.arctanh(np.clip(s, 0.0, 1.0 - 1e-7))
    return umag * bhat


def beta_squash_np(u, eps=1e-6):
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat


def asinh_transform_xy(u_xy, scale=ASINH_SCALE_XY):
    return np.arcsinh(u_xy / scale).astype(np.float32)


def asinh_inverse_xy(v_xy, scale=ASINH_SCALE_XY):
    return (np.sinh(v_xy) * scale).astype(np.float32)


def asinh_transform_y(y, scale=ASINH_SCALE_Y):
    return np.arcsinh(y / scale).astype(np.float32)


def asinh_inverse_y(v, scale=ASINH_SCALE_Y):
    return (np.sinh(v) * scale).astype(np.float32)


def assign_z_plane(z_arr):
    dists = np.abs(z_arr[:, None] - Z_PLANES[None, :])
    plane_idx = np.argmin(dists, axis=1).astype(np.int64)
    residual  = z_arr - Z_PLANES[plane_idx]
    return plane_idx, residual.astype(np.float32)


def reconstruct_z(plane_idx, residual):
    return (Z_PLANES[plane_idx] + residual).astype(np.float32)


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    data_path: str = "guineapig_raw_trimmed.npy"
    outdir: str    = "new_42"
    device: str    = "cuda" if torch.cuda.is_available() else "cpu"

    max_particles: int  = 1300
    min_particles: int  = 1
    keep_fraction: float = 1.0

    T: int         = 500
    cosine_s: float = 0.008

    # GNN hyperparameters
    hidden: int    = 256
    n_layers: int  = 4
    dropout: float = 0.1
    k_neighbors: int = 16   # for kNN edge construction (used only if fully_connected=False)
    fully_connected: bool = True

    batch_size: int  = 8    # smaller than transformer due to graph overhead
    lr: float        = 1e-3
    epochs: int      = 100
    num_workers: int = 8
    grad_clip: float = 2.0
    seed: int        = 123

    feat_dim: int   = 7
    n_pdg: int      = 2
    n_zplane: int   = 2
    lambda_pdg: float    = 0.1
    lambda_zplane: float = 0.5
    lambda_charge: float = 0.011241862095793064

    gamma_start: float = 0.001
    gamma_end: float   = 0.15

    n_events: int         = 500
    sample_batch_size: int = 8

    pct_start: float  = 0.1
    div_factor: float = 10.0

    me: float      = 0.00051099895069
    frac_range: float = 0.60


# ============================================================
# DATASET  (identical to particle_diffusion_new)
# ============================================================
class MCPDataset(Dataset):
    def __init__(self, path, max_particles=1300, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)
        if keep_fraction < 1.0:
            raw = raw[:int(len(raw) * keep_fraction)]

        self.pdg_to_idx = {11: 0, -11: 1}
        self.idx_to_pdg = {0: 11, 1: -11}

        events_cont, events_pdg, events_zplane = [], [], []

        for ev in raw:
            if ev is None: continue
            ev = np.asarray(ev)
            if ev.ndim != 2 or ev.shape[1] < 7: continue
            if len(ev) < min_particles: continue
            ev = ev.astype(np.float32)

            E_signed = ev[:, 0]
            beta     = ev[:, 1:4]
            pos      = ev[:, 4:7]

            pdg_idx = np.where(E_signed >= 0.0, 0, 1).astype(np.int64)
            Eabs    = np.maximum(np.abs(E_signed), 1e-12)
            logE    = np.log(Eabs)

            u    = beta_unsquash_np(beta)
            ux_t = asinh_transform_xy(u[:, 0:1], scale=ASINH_SCALE_XY)
            uy_t = asinh_transform_xy(u[:, 1:2], scale=ASINH_SCALE_XY)
            uz_t = u[:, 2:3]

            zplane_idx, z_residual = assign_z_plane(pos[:, 2])
            x_pos = pos[:, 0:1]
            y_pos = asinh_transform_y(pos[:, 1:2])

            cont = np.concatenate(
                [logE[:, None], ux_t, uy_t, uz_t, x_pos, y_pos, z_residual[:, None]], axis=1
            ).astype(np.float32)

            events_cont.append(cont)
            events_pdg.append(pdg_idx)
            events_zplane.append(zplane_idx)

        if not events_cont:
            raise RuntimeError("No events loaded.")

        self.events_cont   = events_cont
        self.events_pdg    = events_pdg
        self.events_zplane = events_zplane
        self.max_particles = max_particles
        self.feat_dim      = 7

        all_feats       = np.concatenate(events_cont, axis=0)
        self.feat_mean  = all_feats.mean(axis=0).astype(np.float32)
        self.feat_std   = np.maximum(all_feats.std(axis=0), 1e-6).astype(np.float32)
        self.multiplicities = np.array([len(e) for e in events_cont], dtype=np.int64)

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
            chosen = np.random.choice(N, Kmax, replace=False)

        cont   = cont[chosen]
        pdg    = pdg[chosen]
        zplane = zplane[chosen]
        K      = cont.shape[0]

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
# GRAPH UTILITIES
# ============================================================
def build_fully_connected_edges(K: int, device):
    """Return edge_index for a fully connected graph of K nodes."""
    src = torch.arange(K, device=device).repeat_interleave(K)
    dst = torch.arange(K, device=device).repeat(K)
    # remove self-loops
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def batch_to_pyg(x_t, pdg_t, zplane_t, mask, device):
    """
    Convert padded batch tensors to a PyG Batch object.
    x_t:      (B, Kmax, feat_dim)
    pdg_t:    (B, Kmax)
    zplane_t: (B, Kmax)
    mask:     (B, Kmax) bool

    Returns a PyG Batch where each graph is one event (real particles only).
    Also returns (batch_idx_list, K_list) for reassembling outputs.
    """
    graphs = []
    K_list = []

    for i in range(x_t.shape[0]):
        m  = mask[i]            # (Kmax,) bool
        K  = int(m.sum().item())
        K_list.append(K)

        xi      = x_t[i, :K]       # (K, feat_dim)
        pdg_i   = pdg_t[i, :K]     # (K,)
        zpl_i   = zplane_t[i, :K]  # (K,)

        edge_index = build_fully_connected_edges(K, device)

        # Concatenate PDG and zplane embeddings as extra node features
        # (passed as integers; model will embed them)
        data = Data(
            x=xi,
            edge_index=edge_index,
            pdg=pdg_i,
            zplane=zpl_i,
            num_nodes=K,
        )
        graphs.append(data)

    return Batch.from_data_list(graphs), K_list


# ============================================================
# GNN DENOISER
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, t):
        device = t.device
        half   = self.d_model // 2
        freqs  = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class EdgeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_ch, out_ch),
            nn.LayerNorm(out_ch),
            nn.SiLU(),
        )
        self.conv = EdgeConv(mlp, aggr='mean')
        # residual projection if dims differ
        self.proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index) + self.proj(x)


class GNNDenoiser(nn.Module):
    """
    EdgeConv GNN denoiser.

    At each denoising step:
      1. Project node features to hidden dim
      2. Add time embedding, PDG embedding, zplane embedding
      3. Stack N EdgeConv blocks (fully connected edges = all-pairs message passing)
      4. Per-node output head for epsilon prediction
      5. Global pool -> PDG and zplane classification heads
    """
    def __init__(self, feat_dim=7, hidden=256, n_layers=4, dropout=0.1,
                 n_pdg=2, n_zplane=2):
        super().__init__()
        self.hidden   = hidden
        self.feat_dim = feat_dim

        # Input projection
        self.input_proj = nn.Linear(feat_dim, hidden)

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        # PDG and zplane embeddings
        self.pdg_emb    = nn.Embedding(n_pdg,    hidden)
        self.zplane_emb = nn.Embedding(n_zplane, hidden)

        # Multiplicity embedding (log K -> hidden)
        self.k_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        # EdgeConv blocks
        self.blocks = nn.ModuleList([
            EdgeConvBlock(hidden, hidden, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.eps_head    = nn.Linear(hidden, feat_dim)
        self.pdg_head    = nn.Linear(hidden, n_pdg)
        self.zplane_head = nn.Linear(hidden, n_zplane)

    def forward(self, batch, t_per_event):
        """
        batch:        PyG Batch (node features, edge_index, batch vector)
        t_per_event:  (B,) timestep per event in batch

        Returns:
            eps_hat:       (total_nodes, feat_dim)
            pdg_logits:    (total_nodes, n_pdg)
            zplane_logits: (total_nodes, n_zplane)
            K_list:        list of ints (nodes per event)
        """
        x          = batch.x           # (N_total, feat_dim)
        edge_index = batch.edge_index  # (2, E_total)
        b_vec      = batch.batch       # (N_total,) event index per node
        pdg_idx    = batch.pdg
        zplane_idx = batch.zplane

        # Project features
        h = self.input_proj(x)

        # Time embedding: expand from per-event to per-node
        t_emb = self.time_mlp(self.time_emb(t_per_event))  # (B, hidden)
        h = h + t_emb[b_vec]

        # PDG and zplane embeddings
        h = h + self.pdg_emb(pdg_idx.clamp(0, self.pdg_emb.num_embeddings - 1))
        h = h + self.zplane_emb(zplane_idx.clamp(0, self.zplane_emb.num_embeddings - 1))

        # Multiplicity embedding: log(K) per event, expanded to nodes
        K_per_event = torch.bincount(b_vec, minlength=t_per_event.shape[0]).float()
        log_k = torch.log(K_per_event.clamp(min=1)).unsqueeze(-1)  # (B, 1)
        h = h + self.k_mlp(log_k)[b_vec]

        # EdgeConv blocks
        for block in self.blocks:
            h = block(h, edge_index)

        eps_hat       = self.eps_head(h)
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

        self.sqrt_acp    = torch.sqrt(acp)
        self.sqrt_1m_acp = torch.sqrt(1.0 - acp)
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise

    @torch.no_grad()
    def sample(self, mask, pdg_init, zplane_init, sample_steps=None):
        B, K   = mask.shape
        x      = torch.randn((B, K, self.model.feat_dim), device=self.device) * mask.unsqueeze(-1)
        pdg    = pdg_init.clone()
        zplane = zplane_init.clone()

        if sample_steps is None or sample_steps >= self.T:
            timesteps = list(reversed(range(self.T)))
        else:
            indices   = np.linspace(0, self.T - 1, sample_steps, dtype=int)
            timesteps = list(reversed(indices.tolist()))

        for ti in timesteps:
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)

            # Build PyG batch for this step
            pyg_batch, K_list = batch_to_pyg(x, pdg, zplane, mask, self.device)
            pyg_batch = pyg_batch.to(self.device)

            eps_hat_flat, pdg_logits_flat, zplane_logits_flat = \
                self.model(pyg_batch, t)

            # Reassemble flat node outputs back to (B, K, ...)
            eps_hat       = torch.zeros_like(x)
            pdg_logits    = torch.zeros(B, K, self.model.pdg_head.out_features,
                                        device=self.device)
            zplane_logits = torch.zeros(B, K, self.model.zplane_head.out_features,
                                        device=self.device)

            ptr = 0
            for i, Ki in enumerate(K_list):
                eps_hat[i, :Ki]       = eps_hat_flat[ptr:ptr+Ki]
                pdg_logits[i, :Ki]    = pdg_logits_flat[ptr:ptr+Ki]
                zplane_logits[i, :Ki] = zplane_logits_flat[ptr:ptr+Ki]
                ptr += Ki

            # DDPM reverse step
            beta_t  = self.betas[t].view(B, 1, 1)
            alpha_t = self.alphas[t].view(B, 1, 1)
            acp_t   = self.acp[t].view(B, 1, 1)

            mu  = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat
            )
            var = self.posterior_variance[t].view(B, 1, 1)
            z   = torch.zeros_like(x) if ti == 0 else torch.randn_like(x)
            x   = (mu + torch.sqrt(var) * z) * mask.unsqueeze(-1)

            # Sample discrete labels
            pdg_probs = torch.softmax(pdg_logits, dim=-1)
            pdg_samp  = torch.multinomial(
                pdg_probs.view(-1, pdg_probs.size(-1)), 1
            ).view(B, K)
            pdg = torch.where(mask, pdg_samp, pdg)

            zp_probs = torch.softmax(zplane_logits, dim=-1)
            zp_samp  = torch.multinomial(
                zp_probs.view(-1, zp_probs.size(-1)), 1
            ).view(B, K)
            zplane = torch.where(mask, zp_samp, zplane)

        return x, pdg, zplane


# ============================================================
# TRAINING
# ============================================================
def train(args):
    cfg = CFG()
    if args.data_path:     cfg.data_path     = args.data_path
    if args.outdir:        cfg.outdir        = args.outdir
    if args.max_particles: cfg.max_particles = args.max_particles
    if args.epochs:        cfg.epochs        = args.epochs
    if args.batch_size:    cfg.batch_size    = args.batch_size
    if args.T:             cfg.T             = args.T
    if args.seed:          cfg.seed          = args.seed

    print(f"Device: {cfg.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    ds_full = MCPDataset(cfg.data_path, max_particles=cfg.max_particles,
                         min_particles=cfg.min_particles, keep_fraction=cfg.keep_fraction)

    train_idx, val_idx = split_indices(len(ds_full), 0.1, cfg.seed)
    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    model = GNNDenoiser(
        feat_dim=cfg.feat_dim, hidden=cfg.hidden, n_layers=cfg.n_layers,
        dropout=cfg.dropout, n_pdg=cfg.n_pdg, n_zplane=cfg.n_zplane,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GNN parameters: {n_params:,}")

    ddpm   = DDPM(model, cfg.T, cfg.device, cosine_s=cfg.cosine_s)
    gammas = make_linear_gamma_schedule(cfg.T, cfg.gamma_start, cfg.gamma_end, cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(dl_train),
        pct_start=cfg.pct_start, anneal_strategy='cos',
        div_factor=cfg.div_factor, final_div_factor=1e4,
    )

    meta = {
        "multiplicities": ds_full.multiplicities,
        "feat_mean": ds_full.feat_mean, "feat_std": ds_full.feat_std,
        "feat_dim": ds_full.feat_dim, "me": cfg.me,
        "n_pdg": cfg.n_pdg, "n_zplane": cfg.n_zplane,
        "idx_to_pdg": ds_full.idx_to_pdg,
        "max_particles": cfg.max_particles,
        "T": cfg.T, "cosine_s": cfg.cosine_s,
        "hidden": cfg.hidden, "n_layers": cfg.n_layers, "dropout": cfg.dropout,
        "asinh_scale_xy": ASINH_SCALE_XY, "asinh_scale_y": ASINH_SCALE_Y,
        "z_planes": Z_PLANES, "train_idx": train_idx, "val_idx": val_idx,
    }
    torch.save(meta, os.path.join(cfg.outdir, "meta.pt"))

    train_losses, val_losses = [], []

    for epoch in range(cfg.epochs):
        # ---- TRAIN ----
        model.train()
        total_train, n_train = 0.0, 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1:03d}/{cfg.epochs} [train]", leave=False)
        for x0, pdg0, zplane0, mask in pbar:
            x0      = x0.to(cfg.device)
            pdg0    = pdg0.to(cfg.device)
            zplane0 = zplane0.to(cfg.device)
            mask    = mask.to(cfg.device)
            B       = x0.shape[0]

            # t^2 sampling
            u = torch.rand(B, device=cfg.device)
            t = (u ** 2 * cfg.T).long().clamp(0, cfg.T - 1)

            noise    = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t      = ddpm.q_sample(x0, t, noise)
            pdg_t    = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
            zplane_t = q_sample_zplane(zplane0, t, gammas, cfg.n_zplane, mask)

            # Build PyG batch
            pyg_batch, K_list = batch_to_pyg(x_t, pdg_t, zplane_t, mask, cfg.device)
            pyg_batch = pyg_batch.to(cfg.device)

            eps_hat_flat, pdg_logits_flat, zplane_logits_flat = model(pyg_batch, t)

            # Reassemble to padded tensors for loss computation
            eps_hat       = torch.zeros_like(x0)
            pdg_logits    = torch.zeros(B, x0.shape[1], cfg.n_pdg,    device=cfg.device)
            zplane_logits = torch.zeros(B, x0.shape[1], cfg.n_zplane, device=cfg.device)

            ptr = 0
            for i, Ki in enumerate(K_list):
                eps_hat[i, :Ki]       = eps_hat_flat[ptr:ptr+Ki]
                pdg_logits[i, :Ki]    = pdg_logits_flat[ptr:ptr+Ki]
                zplane_logits[i, :Ki] = zplane_logits_flat[ptr:ptr+Ki]
                ptr += Ki

            mse         = (eps_hat - noise).pow(2).sum(dim=-1)
            diff_loss   = (mse * mask).sum() / mask.sum().clamp(min=1)
            pdg_loss    = F.cross_entropy(pdg_logits[mask], pdg0[mask])
            zplane_loss = F.cross_entropy(zplane_logits[mask], zplane0[mask])
            c_loss      = charge_balance_loss(pdg_logits, mask)
            loss        = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_zplane * zplane_loss + cfg.lambda_charge * c_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()

            total_train += loss.item()
            n_train     += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = total_train / max(n_train, 1)

        # ---- VALIDATION ----
        model.eval()
        total_val, n_val = 0.0, 0

        with torch.no_grad():
            for x0, pdg0, zplane0, mask in dl_val:
                x0      = x0.to(cfg.device)
                pdg0    = pdg0.to(cfg.device)
                zplane0 = zplane0.to(cfg.device)
                mask    = mask.to(cfg.device)
                B       = x0.shape[0]

                u = torch.rand(B, device=cfg.device)
                t = (u ** 2 * cfg.T).long().clamp(0, cfg.T - 1)

                noise    = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t      = ddpm.q_sample(x0, t, noise)
                pdg_t    = q_sample_pdg(pdg0, t, gammas, cfg.n_pdg, mask)
                zplane_t = q_sample_zplane(zplane0, t, gammas, cfg.n_zplane, mask)

                pyg_batch, K_list = batch_to_pyg(x_t, pdg_t, zplane_t, mask, cfg.device)
                pyg_batch = pyg_batch.to(cfg.device)

                eps_hat_flat, pdg_logits_flat, zplane_logits_flat = model(pyg_batch, t)

                eps_hat       = torch.zeros_like(x0)
                pdg_logits    = torch.zeros(B, x0.shape[1], cfg.n_pdg,    device=cfg.device)
                zplane_logits = torch.zeros(B, x0.shape[1], cfg.n_zplane, device=cfg.device)

                ptr = 0
                for i, Ki in enumerate(K_list):
                    eps_hat[i, :Ki]       = eps_hat_flat[ptr:ptr+Ki]
                    pdg_logits[i, :Ki]    = pdg_logits_flat[ptr:ptr+Ki]
                    zplane_logits[i, :Ki] = zplane_logits_flat[ptr:ptr+Ki]
                    ptr += Ki

                mse         = (eps_hat - noise).pow(2).sum(dim=-1)
                diff_loss   = (mse * mask).sum() / mask.sum().clamp(min=1)
                pdg_loss    = F.cross_entropy(pdg_logits[mask], pdg0[mask])
                zplane_loss = F.cross_entropy(zplane_logits[mask], zplane0[mask])
                c_loss      = charge_balance_loss(pdg_logits, mask)
                loss        = diff_loss + cfg.lambda_pdg * pdg_loss + cfg.lambda_zplane * zplane_loss + cfg.lambda_charge * c_loss

                total_val += loss.item()
                n_val     += 1

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

    print(f"Training complete. Outputs saved to: {cfg.outdir}")


# ============================================================
# SAMPLING
# ============================================================
def load_meta_and_model(outdir, device):
    meta  = torch.load(os.path.join(outdir, "meta.pt"), map_location="cpu")
    model = GNNDenoiser(
        feat_dim=int(meta["feat_dim"]),
        hidden=int(meta["hidden"]),
        n_layers=int(meta["n_layers"]),
        dropout=float(meta["dropout"]),
        n_pdg=int(meta.get("n_pdg", 2)),
        n_zplane=int(meta.get("n_zplane", 2)),
    ).to(device)
    ckpt = torch.load(os.path.join(outdir, "ckpt_last.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ddpm = DDPM(model, int(meta["T"]), device, cosine_s=float(meta.get("cosine_s", 0.008)))
    return meta, model, ddpm


def _decode_batch(x_norm_batch, pdg_idx_batch, zplane_idx_batch, Ks, meta):
    mean       = np.asarray(meta["feat_mean"], dtype=np.float32)
    std        = np.asarray(meta["feat_std"],  dtype=np.float32)
    idx_to_pdg = meta["idx_to_pdg"]
    scale_xy   = float(meta.get("asinh_scale_xy", ASINH_SCALE_XY))
    scale_y    = float(meta.get("asinh_scale_y",  ASINH_SCALE_Y))

    events = []
    for i, K in enumerate(Ks):
        x_i      = x_norm_batch[i, :K]
        pdg_i    = pdg_idx_batch[i, :K]
        zplane_i = zplane_idx_batch[i, :K]

        cont = x_i * std + mean
        logE       = cont[:, 0]
        ux_asinh   = cont[:, 1]
        uy_asinh   = cont[:, 2]
        u_z        = cont[:, 3]
        x_pos      = cont[:, 4]
        y_asinh    = cont[:, 5]
        z_residual = cont[:, 6]

        E  = np.exp(logE)
        ux = asinh_inverse_xy(ux_asinh, scale=scale_xy)
        uy = asinh_inverse_xy(uy_asinh, scale=scale_xy)
        u_xy    = np.stack([ux, uy, np.zeros_like(ux)], axis=1)
        beta_xy = beta_squash_np(u_xy)
        betax   = beta_xy[:, 0]
        betay   = beta_xy[:, 1]
        bz      = np.tanh(u_z).astype(np.float32)
        y_pos   = asinh_inverse_y(y_asinh, scale=scale_y)
        z_pos   = reconstruct_z(zplane_i, z_residual)
        pdg     = np.array([idx_to_pdg[int(j)] for j in pdg_i], dtype=np.int64)

        out = np.stack(
            [pdg.astype(np.float32), E, betax, betay, bz, x_pos, y_pos, z_pos], axis=1
        ).astype(np.float32)
        events.append(out)

    return events


def sample_batch(meta, ddpm, device, batch_size, sample_steps=None):
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
        x_norm, pdg_idx, zplane_idx = ddpm.sample(mask_t, pdg_init, zplane_init,
                                                    sample_steps=sample_steps)

    return _decode_batch(
        x_norm.cpu().numpy(), pdg_idx.cpu().numpy(), zplane_idx.cpu().numpy(), Ks, meta
    )


def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta, model, ddpm = load_meta_and_model(args.outdir, device)

    events, n_done = [], 0
    for _ in tqdm(range(math.ceil(args.n_events / args.sample_batch_size)),
                  desc="Generating"):
        bs = min(args.sample_batch_size, args.n_events - n_done)
        events.extend(sample_batch(meta, ddpm, device, bs))
        n_done += bs

    out_path = os.path.join(args.outdir, "generated_events.npy")
    np.save(out_path, np.array(events, dtype=object))
    print(f"Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG()
    parser = argparse.ArgumentParser(description="GNN particle diffusion model")
    subparsers = parser.add_subparsers(dest='mode')

    tp = subparsers.add_parser('train')
    tp.add_argument('--data_path',     type=str)
    tp.add_argument('--outdir',        type=str)
    tp.add_argument('--max_particles', type=int)
    tp.add_argument('--epochs',        type=int)
    tp.add_argument('--batch_size',    type=int)
    tp.add_argument('--T',             type=int)
    tp.add_argument('--seed',          type=int)

    sp = subparsers.add_parser('sample')
    sp.add_argument('--outdir',            type=str, default=cfg.outdir)
    sp.add_argument('--n_events',          type=int, default=cfg.n_events)
    sp.add_argument('--sample_batch_size', type=int, default=cfg.sample_batch_size)

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()