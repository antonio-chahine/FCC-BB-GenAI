#!/usr/bin/env python3
"""
Unified particle generation pipeline with DDPM.

Modes:
    train    - Train the diffusion model
    sample   - Generate synthetic events from trained model
    evaluate - Compare real vs generated distributions

Usage:
    python particle_diffusion.py train --data_path mc_gen1.npy --outdir results
    python particle_diffusion.py sample --outdir results --n_events 1000
    python particle_diffusion.py evaluate --real_path mc_gen1.npy --gen_path results/generated_events.npy
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


# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed: int):
    """Make runs reproducible."""
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


def make_linear_beta_schedule(T: int, beta_start: float, beta_end: float, device: str):
    """Create DDPM linear noise schedule."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    return betas, alphas, acp, acp_prev


# ============================================================
# CONFIG
# ============================================================
@dataclass
class CFG:
    data_path: str = "mc_gen1.npy"
    outdir: str = "mc_gen1_model_diffusingpdg_2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    max_particles: int = 1050
    min_particles: int = 1
    keep_fraction: float = 1.0
    
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    
    d_model = 192
    nhead = 6
    num_layers = 4
    dropout = 0.1
    
    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 50
    num_workers: int = 2
    grad_clip: float = 1.0
    seed: int = 123

    frac_range = 0.80

# ============================================================
# DATASET
# ============================================================
class MCPDataset(Dataset):
    def __init__(self, path, max_particles=64, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)
        
        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]
        
        events = []
        for ev in raw:
            if ev is None:
                continue
            if len(ev) < min_particles:
                continue
            events.append(ev)
        
        if len(events) == 0:
            raise RuntimeError("No events left after filtering")
        
        self.events = events
        self.max_particles = max_particles
        
        # Build PDG vocabulary
        all_pdgs = np.concatenate([ev[:, 0].astype(np.int64) for ev in events], axis=0)
        uniq_pdgs = np.unique(all_pdgs)
        
        self.pdg_to_id = {int(p): i for i, p in enumerate(uniq_pdgs.tolist())}
        self.id_to_pdg = {i: int(p) for p, i in self.pdg_to_id.items()}
        self.vocab_size = len(self.pdg_to_id)
        
        # Momentum normalisation (per PDG)
        all_mom = np.concatenate([ev[:, 1:4].astype(np.float32) for ev in events], axis=0)
        
        self.mom_mean_by_id = np.zeros((self.vocab_size, 3), dtype=np.float32)
        self.mom_std_by_id  = np.zeros((self.vocab_size, 3), dtype=np.float32)
        
        for pdg_code in uniq_pdgs.tolist():
            pdg_code = int(pdg_code)
            sel = (all_pdgs == pdg_code)
            m = all_mom[sel]
            
            mean = m.mean(axis=0).astype(np.float32)
            std = m.std(axis=0).astype(np.float32)
            std = np.maximum(std, 1e-6)

            vid = self.pdg_to_id[pdg_code]
            self.mom_mean_by_id[vid] = mean
            self.mom_std_by_id[vid]  = std
        
        self.multiplicities = np.array([len(ev) for ev in events], dtype=np.int64)
        self.pdg_pool = all_pdgs
    
    def __len__(self):
        return len(self.events)
    
    def __getitem__(self, idx):
        ev = self.events[idx]
        N = int(len(ev))
        Kmax = self.max_particles
        
        if N <= Kmax:
            chosen = np.arange(N)
        else:
            chosen = torch.randperm(N)[:Kmax].numpy()
        
        rows = ev[chosen]
        K = rows.shape[0]
        
        pdg_codes = rows[:, 0].astype(np.int64)
        mom = rows[:, 1:4].astype(np.float32)

        # per-PDG momentum normalisation (keep as you already do)
        pdg_ids_real = np.array([self.pdg_to_id[int(x)] for x in pdg_codes], dtype=np.int64)
        mean = self.mom_mean_by_id[pdg_ids_real]
        std  = self.mom_std_by_id[pdg_ids_real]
        mom_norm = (mom - mean) / std

        # charge: -1 for e- (PDG 11), +1 for e+ (PDG -11)
        charge = np.where(pdg_codes == 11, -1.0, +1.0).astype(np.float32)

        pdg_ids = np.full((Kmax,), self.vocab_size, dtype=np.int64)
        x0      = np.zeros((Kmax, 4), dtype=np.float32)
        mask    = np.zeros((Kmax,), dtype=np.bool_)

        pdg_ids[:K]   = pdg_ids_real
        x0[:K, :3]    = mom_norm
        x0[:K,  3]    = charge
        mask[:K]      = True

        return torch.from_numpy(pdg_ids), torch.from_numpy(x0), torch.from_numpy(mask), torch.tensor(K, dtype=torch.long)



# ============================================================
# MODEL COMPONENTS
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.d_model // 2
        
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ParticleDenoiser(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.mom_proj = nn.Linear(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output   = nn.Linear(d_model, 4)

        # Continuous multiplicity conditioning (no buckets)
        self.k_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )


    def forward(self, x_t, t, mask, K_event):
        B, K, _ = x_t.shape

        t_emb = self.time_emb(t)          # (B, d_model)
        t_emb = self.t_mlp(t_emb)         # (B, d_model)
        t_emb = t_emb.unsqueeze(1).expand(B, K, self.d_model)

        mom_emb = self.mom_proj(x_t)

        h = t_emb + mom_emb
        src_key_padding_mask = ~mask

        # Continuous multiplicity conditioning: embed log(K)
        k = torch.log(K_event.float().clamp(min=1)).unsqueeze(-1)  # (B,1)
        k_emb = self.k_mlp(k).unsqueeze(1)                         # (B,1,d_model)
        h = h + k_emb                                              # broadcast to (B,K,d_model)


        h_in = h
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h = h + h_in

        eps_hat = self.output(h)
        eps_hat = eps_hat * mask.unsqueeze(-1)
        return eps_hat

        


# ============================================================
# DDPM WRAPPER
# ============================================================
class DDPM:
    def __init__(self, model, T, beta_start, beta_end, device):
        self.model = model
        self.T = T
        self.device = device
        
        betas, alphas, acp, acp_prev = make_linear_beta_schedule(T, beta_start, beta_end, device)
        
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev
        
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_1m_acp = torch.sqrt(1.0 - acp)
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)
    
    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise
    
    def p_sample(self, x_t, t, mask, K_event):
        eps_hat = self.model(x_t, t, mask, K_event)
        B = x_t.shape[0]

        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)

        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat)
        var = self.posterior_variance[t].view(B, 1, 1)

        if t[0].item() == 0:
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)

        x_prev = mu + torch.sqrt(var) * z
        return x_prev * mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(self, mask, K_event):
        B, K = mask.shape
        x = torch.randn((B, K, 4), device=self.device) * mask.unsqueeze(-1)

        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, mask, K_event)

        return x


# ============================================================
# TRAINING
# ============================================================
def train(args):
    cfg = CFG()
    
    if args.data_path:
        cfg.data_path = args.data_path
    if args.outdir:
        cfg.outdir = args.outdir
    if args.max_particles:
        cfg.max_particles = args.max_particles
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.T:
        cfg.T = args.T
    if args.seed:
        cfg.seed = args.seed
    
    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    
    val_frac = 0.1
    
    ds_full = MCPDataset(
        cfg.data_path,
        max_particles=cfg.max_particles,
        min_particles=cfg.min_particles,
        keep_fraction=cfg.keep_fraction,
    )
    
    train_idx, val_idx = split_indices(len(ds_full), val_frac, cfg.seed)
    
    ds_train = torch.utils.data.Subset(ds_full, train_idx)
    ds_val   = torch.utils.data.Subset(ds_full, val_idx)
    
    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    
    model = ParticleDenoiser(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    
    ddpm = DDPM(model, cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    meta = {
        "pdg_to_id": ds_full.pdg_to_id,
        "id_to_pdg": ds_full.id_to_pdg,
        "vocab_size": ds_full.vocab_size,
        "multiplicities": ds_full.multiplicities,
        "pdg_pool": ds_full.pdg_pool,
        "mom_mean_by_id": ds_full.mom_mean_by_id,
        "mom_std_by_id": ds_full.mom_std_by_id,
        "max_particles": cfg.max_particles,
        "T": cfg.T,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events": len(val_idx),
    }
    torch.save(meta, os.path.join(cfg.outdir, "meta.pt"))
    
    train_losses = []
    val_losses = []
    
    for epoch in range(cfg.epochs):
        # TRAIN
        model.train()
        total_train = 0.0
        n_train = 0
        
        for pdg_id, x0, mask, K_event in dl_train:
            K_event = K_event.to(cfg.device)
            pdg_id = pdg_id.to(cfg.device)
            x0     = x0.to(cfg.device)
            mask   = mask.to(cfg.device)
            
            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)
            
            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t = ddpm.q_sample(x0, t, noise)
            
            eps_hat = model(x_t, t, mask, K_event)
                        
            mse = (eps_hat - noise).pow(2)
            mse[..., 3] *= 2.0   # upweight charge channel
            mse = mse.sum(dim=-1)
            masked = mse[mask]
            loss = masked.mean() if masked.numel() > 0 else torch.tensor(0.0, device=cfg.device)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            
            total_train += loss.item()
            n_train += 1
        
        train_loss = total_train / max(n_train, 1)
        
        # VALIDATION
        model.eval()
        total_val = 0.0
        n_val = 0
        
        with torch.no_grad():
            for pdg_id, x0, mask, K_event in dl_val:
                K_event = K_event.to(cfg.device)

                pdg_id = pdg_id.to(cfg.device)
                x0     = x0.to(cfg.device)
                mask   = mask.to(cfg.device)
                
                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=cfg.device)
                
                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t = ddpm.q_sample(x0, t, noise)
                
                eps_hat = model(x_t, t, mask, K_event)
                                
                mse = (eps_hat - noise).pow(2)
                mse[..., 3] *= 2.0
                mse = mse.sum(dim=-1)
                loss = mse[mask].mean()
                
                total_val += loss.item()
                n_val += 1
        
        val_loss = total_val / max(n_val, 1)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | train={train_loss:.6f} | val={val_loss:.6f}")
        
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            os.path.join(cfg.outdir, "ckpt_last.pt"),
        )
    
    np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cfg.outdir, "val_losses.npy"), np.array(val_losses))
    
    print("Training complete. Outputs saved to:", cfg.outdir)


# ============================================================
# SAMPLING
# ============================================================
def load_meta_and_model(outdir: str, device: str):
    meta_path = os.path.join(outdir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu")
    
    model = ParticleDenoiser(
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(device)


    
    ckpt_path = os.path.join(outdir, "ckpt_last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(
        model=model,
        T=int(meta["T"]),
        beta_start=float(meta["beta_start"]),
        beta_end=float(meta["beta_end"]),
        device=device,
    )

    
    return meta, model, ddpm


def sample_one_event(meta: dict, ddpm: DDPM, device: str):
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    K = int(np.random.choice(multiplicities))

    Kmax = int(meta["max_particles"])
    K = max(1, min(K, Kmax))



    mask = np.zeros((Kmax,), dtype=np.bool_)
    mask[:K] = True
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # one diffusion sample only
    with torch.no_grad():
        K_event_t = torch.tensor([K], device=device, dtype=torch.long)
        x_gen = ddpm.sample(mask_t, K_event_t)[0].detach().cpu().numpy()  # (Kmax,4)

    x_gen = x_gen[:K]
    mom_norm = x_gen[:, :3]

    q_hat = x_gen[:, 3]  # continuous "charge score"

    # No forced balance: assign by sign
    # Convention: q<0 => e- (11), q>=0 => e+ (-11)
    pdg_vals = np.where(q_hat < 0.0, 11, -11).astype(np.int64)


    # (optional) make charge exactly ±1 if you want to store/debug it
    # charge = np.empty(K, dtype=np.float32)
    # charge[order[:half]] = -1.0
    # charge[order[half:]] = +1.0

    # denormalise momenta using per-PDG stats
    pdg_to_id = meta["pdg_to_id"]
    pdg_ids = np.array([pdg_to_id[int(p)] for p in pdg_vals], dtype=np.int64)

    mean = meta["mom_mean_by_id"][pdg_ids]
    std  = meta["mom_std_by_id"][pdg_ids]
    mom  = mom_norm * std + mean

    event_rows = np.zeros((K, 4), dtype=np.float32)
    event_rows[:, 0] = pdg_vals.astype(np.float32)
    event_rows[:, 1:4] = mom.astype(np.float32)
    return event_rows




def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = args.outdir
    n_events = args.n_events
    
    meta, model, ddpm = load_meta_and_model(outdir, device)
    
    from tqdm import trange
    events = []
    for _ in trange(n_events, desc="Generating events"):
        events.append(sample_one_event(meta, ddpm, device))
    
    out_path = os.path.join(outdir, "generated_events.npy")
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


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
    })


def savefig(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_events(path: str):
    arr = np.load(path, allow_pickle=True)
    
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]
    
    raise ValueError(f"Unrecognized format in {path}")


def sanitize_event(ev):
    if ev is None or (isinstance(ev, np.ndarray) and ev.size == 0):
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float64),
                np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    
    ev = np.asarray(ev)
    if ev.ndim != 2 or ev.shape[1] < 4:
        raise ValueError(f"Each event must be (K,4+). Got shape={ev.shape}")
    
    pdg = ev[:, 0].astype(np.int64, copy=False)
    px  = ev[:, 1].astype(np.float64, copy=False)
    py  = ev[:, 2].astype(np.float64, copy=False)
    pz  = ev[:, 3].astype(np.float64, copy=False)
    return pdg, px, py, pz


def extract_species(events, pdgs=None):
    mult = np.zeros(len(events), dtype=np.int64)
    px_list, py_list, pz_list = [], [], []
    
    for i, ev in enumerate(events):
        pdg, px, py, pz = sanitize_event(ev)
        
        if pdgs is None:
            sel = np.ones_like(pdg, dtype=bool)
        else:
            sel = np.zeros_like(pdg, dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)
        
        mult[i] = int(np.sum(sel))
        if np.any(sel):
            px_list.append(px[sel])
            py_list.append(py[sel])
            pz_list.append(pz[sel])
    
    if px_list:
        px_all = np.concatenate(px_list)
        py_all = np.concatenate(py_list)
        pz_all = np.concatenate(pz_list)
    else:
        px_all = np.array([], dtype=np.float64)
        py_all = np.array([], dtype=np.float64)
        pz_all = np.array([], dtype=np.float64)
    
    p_all = np.sqrt(px_all**2 + py_all**2 + pz_all**2)
    pt_all = np.sqrt(px_all**2 + py_all**2)
    
    return {
        "mult": mult, "px": px_all, "py": py_all,
        "pz": pz_all, "p": p_all, "pt": pt_all,
    }


def peak_centered_range(x, y, bins=400, frac=0.995, min_width=1e-12):
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return None
    
    z = np.concatenate(vals)
    if z.size < 2:
        return None
    
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or (zmax - zmin) < min_width:
        return None
    
    counts, edges = np.histogram(z, bins=bins, range=(zmin, zmax))
    tot = counts.sum()
    if tot == 0:
        return None
    
    i0 = int(np.argmax(counts))
    lo_i = hi_i = i0
    cum = int(counts[i0])
    
    target = frac * tot
    while cum < target and (lo_i > 0 or hi_i < len(counts) - 1):
        if lo_i > 0 and hi_i < len(counts) - 1:
            if counts[lo_i - 1] > counts[hi_i + 1]:
                lo_i -= 1
                cum += counts[lo_i]
            else:
                hi_i += 1
                cum += counts[hi_i]
        elif lo_i > 0:
            lo_i -= 1
            cum += counts[lo_i]
        else:
            hi_i += 1
            cum += counts[hi_i]
    
    lo = float(edges[lo_i])
    hi = float(edges[hi_i + 1])
    return (lo, hi)


def robust_range(x, y, q_lo=0.005, q_hi=0.995):
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return None
    
    z = np.concatenate(vals)
    if z.size < 2:
        return None
    
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
    
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if HAVE_SCIPY:
        return float(wasserstein_distance(x, y))
    
    xs = np.sort(np.asarray(x, dtype=np.float64))
    ys = np.sort(np.asarray(y, dtype=np.float64))
    q = np.linspace(0.0, 1.0, 400)
    xq = np.interp(q, np.linspace(0.0, 1.0, len(xs)), xs)
    yq = np.interp(q, np.linspace(0.0, 1.0, len(ys)), ys)
    return float(np.mean(np.abs(xq - yq)))


def plot_multiplicity(real_mult, gen_mult, outpath, species_name, n_real, n_gen, bins=50, logy=False):
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8), constrained_layout=True)
    
    rng = robust_range(real_mult, gen_mult, q_lo=0.0, q_hi=1.0)
    if rng is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        savefig(fig, outpath)
        return
    
    lo, hi = rng
    real_mult = clamp_to_range(real_mult, lo, hi)
    gen_mult  = clamp_to_range(gen_mult,  lo, hi)
    
    ax.hist(real_mult, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Real")
    ax.hist(gen_mult,  bins=bins, range=(lo, hi), density=True, histtype="step",
            linewidth=1.8, label="Generated")
    
    ax.set_title(
        f"Multiplicity of {species_name} | real={n_real} gen={n_gen}",
        pad=10
    )    
    ax.set_xlabel(f"Multiplicity N({species_name}) per event")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    if logy:
        ax.set_yscale("log")
    
    savefig(fig, outpath)


def two_panel_dist(real, gen, outpath, xlabel, title, species_name, bins=80, ratio_min_count=10, frac_range=0.80):
    fig = plt.figure(figsize=(7.6, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(labelbottom=False)
    
    rng = peak_centered_range(real, gen, bins=600, frac=frac_range)
    if rng is None:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()
        savefig(fig, outpath)
        return
    
    lo, hi = rng
    
    real_use = clamp_to_range(real, lo, hi)
    gen_use  = clamp_to_range(gen,  lo, hi)
    
    if real_use.size == 0 or gen_use.size == 0:
        ax_top.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()
        savefig(fig, outpath)
        return
    
    r_counts, edges = np.histogram(real_use, bins=bins, range=(lo, hi), density=False)
    g_counts, _     = np.histogram(gen_use,  bins=bins, range=(lo, hi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    ax_top.hist(real_use, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Real Data")
    ax_top.hist(gen_use,  bins=bins, range=(lo, hi), density=True, histtype="step",
                linewidth=1.8, label="Generated Data")
    
    ax_top.set_title(f"{title} — {species_name}", pad=10)
    ax_top.set_ylabel("Density", labelpad=10)
    ax_bot.set_ylabel("Frac. diff.", labelpad=10)
    ax_top.legend(loc="upper left")
    
    kl = kl_divergence_from_counts(r_counts, g_counts)
    wd = wasserstein_1d(real_use, gen_use)
    
    ax_top.text(
        0.98, 0.95, f"KL: {kl:.4f}\nW1: {wd:.4f}",
        transform=ax_top.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )
    
    r_sum = r_counts.sum()
    g_sum = g_counts.sum()
    
    r_prob = r_counts / max(r_sum, 1)
    g_prob = g_counts / max(g_sum, 1)
    
    mask = r_counts >= ratio_min_count
    
    frac = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac[mask] = (g_prob[mask] - r_prob[mask]) / r_prob[mask]
    
    frac_err = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac_err[mask] = 1.0 / np.sqrt(r_counts[mask])
    
    ax_bot.axhline(0.0, linewidth=1.0, color='black')
    ax_bot.axhspan(-0.1, 0.1, color="gray", alpha=0.15, zorder=0)
    
    ax_bot.errorbar(centers[mask], frac[mask], yerr=frac_err[mask],
                    fmt="o", markersize=3, linewidth=1.0, capsize=0)
    
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylim(-1.0, 1.0)
    
    ax_bot.text(0.98, 0.85, "±10% band", transform=ax_bot.transAxes,
                ha="right", va="center", fontsize=10)
    
    savefig(fig, outpath)


def evaluate(args):
    setup_style()
    cfg = CFG()  # reads CFG.frac_range

    
    # If no outdir specified, use the directory of the generated data
    if args.outdir is None:
        args.outdir = os.path.dirname(args.gen_path)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    real_events = load_events(args.real_path)
    gen_events  = load_events(args.gen_path)
    
    n_real = len(real_events)
    n_gen  = len(gen_events)
    
    print(f"Loaded {n_real} real events and {n_gen} generated events")
    
    species_list = [
    
        {"name": "e−", "pdgs": [11],      "tag": "eminus"},
        {"name": "e+", "pdgs": [-11],     "tag": "eplus"},
    ]
    
    if args.include_all:
        species_list.append({"name": "all", "pdgs": None, "tag": "all"})
    
    for sp in species_list:
        print(f"Processing species: {sp['name']}")
        real_sp = extract_species(real_events, sp["pdgs"])
        gen_sp  = extract_species(gen_events,  sp["pdgs"])
        
        plot_multiplicity(
            real_sp["mult"], gen_sp["mult"],
            outpath=os.path.join(args.outdir, f"multiplicity_{sp['tag']}.png"),
            species_name=sp["name"], n_real=n_real, n_gen=n_gen,
            bins=args.mult_bins, logy=False,
        )
        
        for key in ["px", "py", "pz", "pt"]:
            two_panel_dist(
                real_sp[key], gen_sp[key],
                outpath=os.path.join(args.outdir, f"{key}_{sp['tag']}.png"),
                xlabel=("p_T" if key == "pt" else key),
                title=f"Comparison of {key} | real={n_real} gen={n_gen}",
                species_name=sp["name"],
                bins=args.mom_bins,
                ratio_min_count=args.ratio_min_count,
                frac_range=cfg.frac_range,
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
        epilog="""
Examples:
  Train:    python particle_diffusion.py train --data_path mc_gen1.npy --outdir results --epochs 100
  Sample:   python particle_diffusion.py sample --outdir results --n_events 1000
  Evaluate: python particle_diffusion.py evaluate --real_path mc_gen1.npy --gen_path results/generated_events.npy
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train the diffusion model')
    train_parser.add_argument('--data_path', type=str, help='Path to training data (.npy)')
    train_parser.add_argument('--outdir', type=str, help='Output directory')
    train_parser.add_argument('--max_particles', type=int, help='Max particles per event')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--T', type=int, help='Diffusion steps')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Sample
    sample_parser = subparsers.add_parser('sample', help='Generate synthetic events')
    sample_parser.add_argument('--outdir', type=str, default=cfg.outdir, help='Model directory')
    sample_parser.add_argument('--n_events', type=int, default=1247, help='Number of events')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate distributions')
    eval_parser.add_argument('--real_path', type=str, default=cfg.data_path, help='Real data path')
    eval_parser.add_argument('--gen_path', type=str, default=os.path.join(cfg.outdir, "generated_events.npy"), help='Generated data path')
    eval_parser.add_argument('--outdir', type=str, default=os.path.join(cfg.outdir, "plots"), help='Output directory (default: same as gen_path)')
    eval_parser.add_argument('--include_all', action='store_true', help='Include all species')
    eval_parser.add_argument('--mult_bins', type=int, default=50, help='Multiplicity bins')
    eval_parser.add_argument('--mom_bins', type=int, default=80, help='Momentum bins')
    eval_parser.add_argument('--ratio_min_count', type=int, default=10, help='Min counts for ratio')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
