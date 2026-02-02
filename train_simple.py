#!/usr/bin/env python3
"""
A minimal, heavily-commented DDPM (diffusion model) that generates particle momenta.

DATA FORMAT (input .npy):
    - numpy object array called "events"
    - events[i] is a numpy array of shape (Ni, 4):
          [pdg, px, py, pz]
      where Ni is the number of particles in event i.

WHAT THE MODEL LEARNS:
    - We treat each event as a sequence of up to Kmax particles.
    - For each particle we want to model its momentum vector (px, py, pz) (3 numbers).
    - We condition the model on:
          (a) particle PDG ID (categorical token)
          (b) diffusion timestep t
    - The model predicts the Gaussian noise ε that was added at timestep t.

KEY IDEA:
    Training:
        1) Take the real (normalised) momenta x0
        2) Sample random timestep t
        3) Add noise -> x_t
        4) Predict noise eps_hat = model(x_t, t, pdg_id)
        5) MSE between eps_hat and true noise (only on real particles, not padding)

    Sampling:
        1) Start from pure noise x_T
        2) Repeatedly apply the reverse step down to t=0
        3) Output is in *normalised* momentum units; you later denormalise.
"""

import os
import math
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset


def split_events(n_events: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    event_ids = np.arange(n_events)
    rng.shuffle(event_ids)

    n_val = int(round(val_frac * n_events))
    val_events = set(event_ids[:n_val].tolist())
    train_events = set(event_ids[n_val:].tolist())
    return train_events, val_events


def run_epoch(ddpm, model, loader, device, opt=None, grad_clip=None):
    """
    One epoch over loader.
    - If opt is provided: training (with grads)
    - If opt is None: validation (no grads)
    Returns mean loss.
    """
    is_train = opt is not None
    model.train(is_train)

    total = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for pdg_id, x0, mask in loader:
            pdg_id = pdg_id.to(device)     # (B,K)
            x0     = x0.to(device)         # (B,K,3)
            mask   = mask.to(device)       # (B,K)

            B = x0.shape[0]
            t = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)

            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t = ddpm.q_sample(x0, t, noise)

            eps_hat = model(x_t, t, pdg_id, mask)

            mse_per_token = (eps_hat - noise).pow(2).sum(dim=-1)  # (B,K)
            loss = mse_per_token[mask].mean()

            if is_train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            total += float(loss.item())
            n_batches += 1

    return total / max(n_batches, 1)


# ============================================================
# 1) CONFIG
# ============================================================
@dataclass
class CFG:
    # Files / output
    data_path: str = "mc_gen1.npy"
    outdir: str = "mc_gen1_model_simple"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Event handling
    max_particles: int = 256   # pad/truncate each event to this length Kmax
    min_particles: int = 1    # drop events with fewer than this
    keep_fraction: float = 1.0  # use first fraction of dataset (debugging)

    # Diffusion schedule
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Transformer denoiser
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 256
    lr: float = 2e-4
    epochs: int = 100
    num_workers: int = 2
    grad_clip: float = 1.0
    seed: int = 123


# ============================================================
# 2) SMALL UTILITIES
# ============================================================
def set_seed(seed: int):
    """Make runs reproducible."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_linear_beta_schedule(T: int, beta_start: float, beta_end: float, device: str):
    """
    Create the standard DDPM linear noise schedule and useful derived arrays.

    betas[t]   : how much noise we add at step t
    alphas[t]  = 1 - betas[t]
    acp[t]     = prod_{i<=t} alphas[i]  (alpha cumulative product)
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)   # (T,)
    alphas = 1.0 - betas                                             # (T,)
    acp = torch.cumprod(alphas, dim=0)                               # (T,)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])   # (T,)
    return betas, alphas, acp, acp_prev


# ============================================================
# 3) DATASET
# ============================================================
class MCPDataset(Dataset):
    """
    Chunked dataset: uses ALL particles from ALL events.

    Each dataset item is a chunk of up to Kmax particles from one event.
    Large events produce multiple items; last chunk is padded.
    """

    def __init__(self, path, max_particles=64, min_particles=1, keep_fraction=1.0):
        raw = np.load(path, allow_pickle=True)
        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]

        # ---- filter events
        events = []
        for ev in raw:
            if ev is None:
                continue
            if len(ev) < min_particles:
                continue
            events.append(ev)

        if len(events) == 0:
            raise RuntimeError("No events left after filtering; check data/min_particles.")

        self.events = events
        self.max_particles = int(max_particles)

        # ---- vocab from ALL particles
        all_pdgs = np.concatenate([ev[:, 0].astype(np.int64) for ev in events], axis=0)
        uniq_pdgs = np.unique(all_pdgs)

        self.pdg_to_id = {int(p): i for i, p in enumerate(uniq_pdgs.tolist())}
        self.id_to_pdg = {i: int(p) for p, i in self.pdg_to_id.items()}
        self.vocab_size = len(self.pdg_to_id)

        # ---- per-PDG normalisation from ALL particles
        all_mom = np.concatenate([ev[:, 1:4].astype(np.float32) for ev in events], axis=0)

        self.min_mom_std = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        self.mom_mean_by_id = np.zeros((self.vocab_size, 3), dtype=np.float32)
        self.mom_std_by_id  = np.zeros((self.vocab_size, 3), dtype=np.float32)

        for pdg_code in uniq_pdgs.tolist():
            pdg_code = int(pdg_code)
            sel = (all_pdgs == pdg_code)
            m = all_mom[sel]

            mean = m.mean(axis=0).astype(np.float32)
            std  = np.maximum(m.std(axis=0).astype(np.float32), self.min_mom_std)

            vid = self.pdg_to_id[pdg_code]
            self.mom_mean_by_id[vid] = mean
            self.mom_std_by_id[vid]  = std

        # ---- for later sampling scripts
        self.multiplicities = np.array([len(ev) for ev in events], dtype=np.int64)
        self.pdg_pool = all_pdgs

        # ---- build chunk index: EVERY particle belongs to exactly one chunk
        # chunk = (event_idx, start, end)
        self.chunks = []
        K = self.max_particles
        for ei, ev in enumerate(events):
            N = len(ev)
            for s in range(0, N, K):
                e = min(s + K, N)
                self.chunks.append((ei, s, e))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ei, s, e = self.chunks[idx]
        ev = self.events[ei]

        rows = ev[s:e]                 # (Kchunk,4), Kchunk<=Kmax
        Kchunk = rows.shape[0]
        Kmax = self.max_particles

        pdg_codes = rows[:, 0].astype(np.int64)
        mom = rows[:, 1:4].astype(np.float32)

        # PDG -> token id
        pdg_ids_real = np.array([self.pdg_to_id[int(x)] for x in pdg_codes], dtype=np.int64)

        # normalise per PDG id
        mean = self.mom_mean_by_id[pdg_ids_real]
        std  = self.mom_std_by_id[pdg_ids_real]
        mom_norm = (mom - mean) / std

        # pad to Kmax
        pdg_ids = np.zeros((Kmax,), dtype=np.int64)
        x0      = np.zeros((Kmax, 3), dtype=np.float32)
        mask    = np.zeros((Kmax,), dtype=np.bool_)

        pdg_ids[:Kchunk] = pdg_ids_real
        x0[:Kchunk]      = mom_norm
        mask[:Kchunk]    = True

        return torch.from_numpy(pdg_ids), torch.from_numpy(x0), torch.from_numpy(mask)




# ============================================================
# 4) MODEL COMPONENTS
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    """
    Classic sinusoidal embedding used in Transformers / diffusion models.

    Input:  t of shape (B,)
    Output: embedding of shape (B, d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.d_model // 2

        # frequencies
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )

        # angles: (B, half)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        # sin/cos: (B, 2*half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        # if d_model is odd, pad one extra dimension
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ParticleDenoiser(nn.Module):
    """
    Per-particle denoiser (shared MLP):
      eps_hat_i = f(x_t_i, pdg_i, t)

    Inputs:
        x_t    : (B,K,3)
        t      : (B,)
        pdg_id : (B,K)
        mask   : (B,K)
    Output:
        eps_hat: (B,K,3)
    """
    def __init__(self, vocab_size, d_model=128, hidden=256, n_layers=3, dropout=0.1):
        super().__init__()

        self.pdg_emb = nn.Embedding(vocab_size, d_model)
        self.t_emb = SinusoidalTimeEmbedding(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        in_dim = 3 + d_model + d_model  # x_t (3) + pdg_emb (d_model) + t_emb (d_model)

        layers = []
        dim = in_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            dim = hidden

        layers += [nn.Linear(dim, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, pdg_id, mask):
        B, K, _ = x_t.shape

        pdg_e = self.pdg_emb(pdg_id)          # (B,K,d)
        t_e = self.t_mlp(self.t_emb(t))       # (B,d)
        t_e = t_e.unsqueeze(1).expand(B, K, -1)  # (B,K,d)

        h = torch.cat([x_t, pdg_e, t_e], dim=-1) # (B,K, 3+2d)
        eps_hat = self.net(h)                    # (B,K,3)

        return eps_hat * mask.unsqueeze(-1)



# ============================================================
# 5) DDPM OBJECT (FORWARD + REVERSE STEPS)
# ============================================================
class DDPM:
    """
    Wraps the diffusion math.
    - q_sample: forward process (add noise)
    - p_sample: reverse step using model prediction
    - sample  : full reverse chain
    """

    def __init__(self, model, T, beta_start, beta_end, device):
        self.model = model
        self.T = T
        self.device = device

        betas, alphas, acp, acp_prev = make_linear_beta_schedule(T, beta_start, beta_end, device)
        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev

        # Precompute square roots used all the time
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_1m_acp = torch.sqrt(1.0 - acp)

        # Posterior variance for reverse transitions:
        # Var[q(x_{t-1} | x_t, x0)]
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion formula:
            x_t = sqrt(acp[t]) * x0 + sqrt(1-acp[t]) * noise

        Shapes:
            x0    : (B,K,3)
            t     : (B,)
            noise : (B,K,3)
        """
        B = x0.shape[0]
        a = self.sqrt_acp[t].view(B, 1, 1)
        b = self.sqrt_1m_acp[t].view(B, 1, 1)
        return a * x0 + b * noise

    def p_sample(self, x_t, t, pdg_id, mask):
        """
        One reverse step from x_t -> x_{t-1}.

        We use the DDPM mean:
            mu = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-acp_t) * eps_hat)

        Then sample:
            x_{t-1} = mu + sqrt(var) * z     (z ~ N(0, I), except z=0 at t=0)
        """
        B = x_t.shape[0]

        # Predict epsilon using the model
        eps_hat = self.model(x_t, t, pdg_id, mask)

        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)

        # Reverse mean
        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat)

        # Reverse variance
        var = self.posterior_variance[t].view(B, 1, 1)

        # Add stochasticity except at the final step t=0
        if t[0].item() == 0:
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)

        x_prev = mu + torch.sqrt(var) * z

        # Force padding tokens to remain zero
        return x_prev * mask.unsqueeze(-1)

    @torch.no_grad()
    def sample(self, pdg_id, mask):
        """
        Full reverse chain to produce x0_hat (normalised momenta).

        Inputs:
            pdg_id : (B,K)
            mask   : (B,K)
        Output:
            x      : (B,K,3) in normalised units
        """
        B, K = pdg_id.shape

        # Start from pure noise
        x = torch.randn((B, K, 3), device=self.device) * mask.unsqueeze(-1)

        # Step backwards from T-1 down to 0
        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t, pdg_id, mask)

        return x


# ============================================================
# 6) TRAINING
# ============================================================
def train():
    cfg = CFG()
    set_seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)

    val_frac = 0.10  # change if you want

    # ---- Build ONE dataset containing ALL events/chunks
    #      (We’ll split by event ID, then select chunks accordingly.)
    ds = MCPDataset(
        cfg.data_path,
        max_particles=cfg.max_particles,
        min_particles=cfg.min_particles,
        keep_fraction=cfg.keep_fraction,
    )

    n_events = len(ds.events)
    train_events, val_events = split_events(n_events, val_frac=val_frac, seed=cfg.seed)

    # ---- Convert event split -> chunk indices split
    # ds.chunks[i] = (event_idx, start, end)
    train_chunk_idx = [i for i, (ei, _, _) in enumerate(ds.chunks) if ei in train_events]
    val_chunk_idx   = [i for i, (ei, _, _) in enumerate(ds.chunks) if ei in val_events]

    if len(train_chunk_idx) == 0 or len(val_chunk_idx) == 0:
        raise RuntimeError(
            f"Bad split: train chunks={len(train_chunk_idx)}, val chunks={len(val_chunk_idx)}. "
            f"Try changing val_frac or check dataset size."
        )

    ds_train = Subset(ds, train_chunk_idx)
    ds_val   = Subset(ds, val_chunk_idx)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model
    model = ParticleDenoiser(
        vocab_size=ds.vocab_size,
        d_model=cfg.d_model,
        hidden=256,
        n_layers=3,
        dropout=cfg.dropout,
    ).to(cfg.device)

    ddpm = DDPM(model, cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ---- Save meta for sampling
    meta = {
        "pdg_to_id": ds.pdg_to_id,
        "id_to_pdg": ds.id_to_pdg,
        "vocab_size": ds.vocab_size,
        "multiplicities": ds.multiplicities,
        "pdg_pool": ds.pdg_pool,
        "mom_mean_by_id": ds.mom_mean_by_id,
        "mom_std_by_id": ds.mom_std_by_id,
        "min_mom_std": ds.min_mom_std,
        "max_particles": cfg.max_particles,
        "T": cfg.T,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "d_model": cfg.d_model,
        "dropout": cfg.dropout,
        "hidden": 256,
        "n_layers": 3,
        # split info (useful for reproducibility)
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": n_events,
        "n_train_events": len(train_events),
        "n_val_events": len(val_events),
        "n_train_chunks": len(train_chunk_idx),
        "n_val_chunks": len(val_chunk_idx),
    }
    torch.save(meta, os.path.join(cfg.outdir, "meta.pt"))

    best_val = float("inf")

    train_losses = []
    val_losses = []
    for epoch in range(cfg.epochs):


        train_loss = run_epoch(ddpm, model, dl_train, cfg.device, opt=opt, grad_clip=cfg.grad_clip)
        val_loss   = run_epoch(ddpm, model, dl_val, cfg.device, opt=None)

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(cfg.outdir, "ckpt_last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(cfg.outdir, "ckpt_best.pt"))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)


    np.save(os.path.join(cfg.outdir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(cfg.outdir, "val_losses.npy"), np.array(val_losses))

    print("Training complete. Outputs saved to:", cfg.outdir)
    print("Best val loss:", best_val)


if __name__ == "__main__":
    train()
