#!/usr/bin/env python3
"""
generates particle momenta.

DATA FORMAT (input .npy):
    - numpy object array called events
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
from sklearn.model_selection import train_test_split

def split_indices(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    return idx[n_val:], idx[:n_val]

# ============================================================
# 1) CONFIG
# ============================================================
@dataclass
class CFG:
    # Files / output
    data_path: str = "mc_gen1.npy"
    outdir: str = "mc_gen1_model_400steps"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Event handling
    max_particles: int = 1050   # pad/truncate each event to this length Kmax
    min_particles: int = 1    # drop events with fewer than this
    keep_fraction: float = 1.0  # use first fraction of dataset (debugging)

    # Diffusion schedule
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Transformer denoiser
    d_model = 192
    nhead = 6
    num_layers = 4
    dropout = 0.1


    # Training
    batch_size: int = 64
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
    Loads events and returns padded sequences.

    For each event we output:
        pdg_ids : (Kmax,)  torch.long
        x0      : (Kmax,3) torch.float  (NORMALISED momenta)
        mask    : (Kmax,)  torch.bool   True for real particles, False for padding
    """

    def __init__(self, path, max_particles=64, min_particles=1, keep_fraction=1.0):
        # Load object array of events
        raw = np.load(path, allow_pickle=True)

        # (Optional) use only a fraction (useful for quick debugging)
        if keep_fraction < 1.0:
            raw = raw[: int(len(raw) * keep_fraction)]

        # Filter out empty / too-small events
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
        self.max_particles = max_particles

        # ------------------------------------------------------------
        # Build PDG vocabulary: map PDG code -> integer token id [0..V-1]
        # ------------------------------------------------------------
        all_pdgs = np.concatenate([ev[:, 0].astype(np.int64) for ev in events], axis=0)
        uniq_pdgs = np.unique(all_pdgs)

        self.pdg_to_id = {int(p): i for i, p in enumerate(uniq_pdgs.tolist())}
        self.id_to_pdg = {i: int(p) for p, i in self.pdg_to_id.items()}
        self.vocab_size = len(self.pdg_to_id)

        # ------------------------------------------------------------
        # Momentum normalisation (per PDG):
        # For each PDG token, compute mean/std of (px,py,pz),
        # then normalise momenta as (p - mean) / std.
        # ------------------------------------------------------------
        all_mom = np.concatenate([ev[:, 1:4].astype(np.float32) for ev in events], axis=0)

        # Safety: don't allow extremely tiny std
        self.min_mom_std = np.array([0.05, 0.05, 0.05], dtype=np.float32)

        self.mom_mean_by_id = np.zeros((self.vocab_size, 3), dtype=np.float32)
        self.mom_std_by_id  = np.zeros((self.vocab_size, 3), dtype=np.float32)

        for pdg_code in uniq_pdgs.tolist():
            pdg_code = int(pdg_code)

            # Select rows for this PDG
            sel = (all_pdgs == pdg_code)
            m = all_mom[sel]  # (Np,3)

            mean = m.mean(axis=0).astype(np.float32)
            std  = m.std(axis=0).astype(np.float32)
            std  = np.maximum(std, self.min_mom_std)

            vid = self.pdg_to_id[pdg_code]
            self.mom_mean_by_id[vid] = mean
            self.mom_std_by_id[vid]  = std

        # Useful for later scripts
        self.multiplicities = np.array([len(ev) for ev in events], dtype=np.int64)
        self.pdg_pool = all_pdgs  # raw PDG codes (for simple sampling)

        self.seed = 1234
        self.rng = np.random.default_rng(self.seed)


    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        ev = self.events[idx]
        N = int(len(ev))
        Kmax = self.max_particles

        # --- Option 1: choose a random subset if event is larger than Kmax
        if N <= Kmax:
            chosen = np.arange(N)
        else:
            # worker-safe alternative:
            chosen = torch.randperm(N)[:Kmax].numpy()

        rows = ev[chosen]  # (K,4) where K=min(N,Kmax)
        K = rows.shape[0]

        pdg_codes = rows[:, 0].astype(np.int64)     # (K,)
        mom = rows[:, 1:4].astype(np.float32)       # (K,3)

        # Map PDG codes -> vocab token ids
        pdg_ids_real = np.array([self.pdg_to_id[int(x)] for x in pdg_codes], dtype=np.int64)

        # Normalise momenta per PDG token id
        mean = self.mom_mean_by_id[pdg_ids_real]    # (K,3)
        std  = self.mom_std_by_id[pdg_ids_real]     # (K,3)
        mom_norm = (mom - mean) / std               # (K,3)

        # Pad to Kmax
        pdg_ids = np.zeros((Kmax,), dtype=np.int64)
        x0      = np.zeros((Kmax, 3), dtype=np.float32)
        mask    = np.zeros((Kmax,), dtype=np.bool_)

        pdg_ids[:K] = pdg_ids_real
        x0[:K]      = mom_norm
        mask[:K]    = True

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
    Transformer denoiser that predicts epsilon (noise) for each particle's momentum.

    Inputs:
        x_t    : (B,K,3) noisy momenta at timestep t
        t      : (B,)    timestep indices
        pdg_id : (B,K)   PDG token ids
        mask   : (B,K)   True where real particle, False where padding

    Output:
        eps_hat: (B,K,3)
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()

        # Convert momentum 3-vector -> model dimension
        self.x_proj = nn.Linear(3, d_model)

        # Learnable embedding for PDG tokens
        self.pdg_emb = nn.Embedding(vocab_size, d_model)

        # Time embedding + small MLP for flexibility
        self.t_emb = SinusoidalTimeEmbedding(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder (self-attention)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Predict epsilon back in momentum space (3 numbers)
        self.out = nn.Linear(d_model, 3)

    def forward(self, x_t, t, pdg_id, mask):
        # Token embedding = momentum projection + PDG embedding
        h = self.x_proj(x_t) + self.pdg_emb(pdg_id)   # (B,K,d)

        # Add time embedding to every token
        te = self.t_mlp(self.t_emb(t))                # (B,d)
        h = h + te.unsqueeze(1)                       # (B,K,d)

        # Transformer uses "key_padding_mask=True" to mean "ignore this token"
        key_padding_mask = ~mask                      # padding positions are True here
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        return self.out(h)                            # (B,K,3)


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


    # ---- model + diffusion wrapper + optimizer
    model = ParticleDenoiser(
        vocab_size=ds_full.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)


    ddpm = DDPM(model, cfg.T, cfg.beta_start, cfg.beta_end, cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # ---- save metadata needed for later sampling/denorm scripts
    meta = {
        "pdg_to_id": ds_full.pdg_to_id,
        "id_to_pdg": ds_full.id_to_pdg,
        "vocab_size": ds_full.vocab_size,
        "multiplicities": ds_full.multiplicities,
        "pdg_pool": ds_full.pdg_pool,
        "mom_mean_by_id": ds_full.mom_mean_by_id,
        "mom_std_by_id": ds_full.mom_std_by_id,
        "min_mom_std": ds_full.min_mom_std,

        "max_particles": cfg.max_particles,
        "T": cfg.T,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,

        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,

        # (optional but useful)
        "val_frac": val_frac,
        "seed": cfg.seed,
        "n_events": len(ds_full),
        "n_train_events": len(train_idx),
        "n_val_events": len(val_idx),
    }
    torch.save(meta, os.path.join(cfg.outdir, "meta.pt"))


    # ---- training loop
    train_losses = []
    val_losses = []

    for epoch in range(cfg.epochs):
        # ---------------- TRAIN ----------------
        model.train()
        total_train = 0.0
        n_train = 0

        for pdg_id, x0, mask in dl_train:
            pdg_id = pdg_id.to(cfg.device)
            x0     = x0.to(cfg.device)
            mask   = mask.to(cfg.device)

            B = x0.shape[0]
            t = torch.randint(0, cfg.T, (B,), device=cfg.device)

            noise = torch.randn_like(x0) * mask.unsqueeze(-1)
            x_t = ddpm.q_sample(x0, t, noise)

            eps_hat = model(x_t, t, pdg_id, mask)

            mse = (eps_hat - noise).pow(2).sum(dim=-1)
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

        # ---------------- VALIDATION ----------------
        model.eval()
        total_val = 0.0
        n_val = 0

        with torch.no_grad():
            for pdg_id, x0, mask in dl_val:
                pdg_id = pdg_id.to(cfg.device)
                x0     = x0.to(cfg.device)
                mask   = mask.to(cfg.device)

                B = x0.shape[0]
                t = torch.randint(0, cfg.T, (B,), device=cfg.device)

                noise = torch.randn_like(x0) * mask.unsqueeze(-1)
                x_t = ddpm.q_sample(x0, t, noise)

                eps_hat = model(x_t, t, pdg_id, mask)

                mse = (eps_hat - noise).pow(2).sum(dim=-1)
                loss = mse[mask].mean()

                total_val += loss.item()
                n_val += 1

        val_loss = total_val / max(n_val, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f}"
        )

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


if __name__ == "__main__":
    train()
