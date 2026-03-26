#!/usr/bin/env python3
"""
generate_pairs.py
-----------------
Standalone sampler: loads a trained diffusion model and writes one
output_XXXXX.pair file per event.

Data pipeline audit
-------------------
Original .pairs files (Jan Eysermans format):
  col 0 : energy [GeV]   positive=e-, negative=e+
  col 1 : betax          dimensionless
  col 2 : betay          dimensionless
  col 3 : betaz          dimensionless
  col 4 : xpos  [mm]
  col 5 : ypos  [mm]
  col 6 : zpos  [mm]
  col 7 : process        (kept in .npy but NOT used by model)
  col 8-10 : unused

The loading script (extra_data.npy builder) does:
  d = np.loadtxt(f)[:, :8]   --> keeps cols 0-7

MCPDataset loads this .npy and reads ev[:, :7]  --> cols 0-6 only:
  ev[:, 0]   = E_signed  (GeV, signed)
  ev[:, 1:4] = beta      (dimensionless)
  ev[:, 4:7] = pos       (mm, NO unit conversion applied)

MCPDataset then transforms for training:
  PDG index : 0 if E_signed >= 0 (e-), 1 if E_signed < 0 (e+)  --> discrete head
  logE      : log(|E_signed|)                                    --> continuous head col 0
  u         : beta_unsquash(beta)                                --> continuous head cols 1-3
  pos       : stored as-is in mm                                 --> continuous head cols 4-6

So positions are in mm throughout. No unit conversion needed on decode.

Decode procedure
----------------
ddpm.sample() returns two SEPARATE tensors:
  x_norm   (B, Kmax, 7) : normalised [logE, u_x, u_y, u_z, x, y, z]  -- continuous head
  pdg_idx  (B, Kmax)    : class 0=e- / 1=e+                           -- discrete head

decode_batch():
  1. Denormalise x_norm  --> logE, u, pos_mm
  2. |E|   = exp(logE)                   from continuous head (always positive)
  3. beta  = beta_squash(u)              from continuous head
  4. sign  = +1 if pdg_idx==0, -1 if 1  from discrete head
  5. energy = sign * |E|                 recombined

Output .pair row (11 cols):
  energy  betax  betay  betaz  xpos  ypos  zpos  -99  -99  -99  -99

Usage
-----
  python generate_pairs.py \\
      --outdir            /path/to/model_dir    \\
      --pairsdir          /path/to/output_pairs \\
      --n_events          50000                 \\
      --sample_batch_size 32                    \\
      --start_index       0
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================
# Noise schedule  (identical to training script)
# ============================================================

def make_cosine_beta_schedule(T: int, s: float = 0.008, device: str = "cpu"):
    steps = torch.arange(T + 1, device=device) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0, 0.999)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    return betas, alphas, acp, acp_prev


# ============================================================
# beta_squash  (identical to training script)
# Inverse of beta_unsquash: maps unbounded u --> |beta|<1 via tanh
# ============================================================

def beta_squash_np(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    umag = np.linalg.norm(u, axis=1, keepdims=True)
    uhat = u / (umag + 1e-12)
    s = np.tanh(umag)
    return (1.0 - eps) * s * uhat


# ============================================================
# Model  (identical to particle_diffusion.py)
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor):
        device = t.device
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ParticleDenoiser(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1, n_pdg=2):
        super().__init__()
        self.d_model  = d_model
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.mom_proj = nn.Linear(7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output   = nn.Linear(d_model, 7)
        self.t_mlp    = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.k_mlp    = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pdg_emb  = nn.Embedding(n_pdg, d_model)
        self.pdg_head = nn.Linear(d_model, n_pdg)
        self.skip_alpha = 0.2

    def forward(self, x_t, t, pdg_t, mask):
        B, K, _ = x_t.shape
        t_emb   = self.t_mlp(self.time_emb(t)).unsqueeze(1).expand(B, K, self.d_model)
        mom_emb = self.mom_proj(x_t)
        pdg_emb = self.pdg_emb(pdg_t.clamp(0, self.pdg_emb.num_embeddings - 1))
        h = t_emb + mom_emb + pdg_emb
        K_event = mask.sum(dim=1)
        k_emb   = self.k_mlp(
            torch.log(K_event.float().clamp(min=1)).unsqueeze(-1)
        ).unsqueeze(1)
        h = h + k_emb
        h_in = h
        h = self.transformer(h, src_key_padding_mask=~mask)
        h = (h + self.skip_alpha * h_in) * mask.unsqueeze(-1)
        eps_hat    = self.output(h) * mask.unsqueeze(-1)
        pdg_logits = self.pdg_head(h)   # (B, K, n_pdg)
        return eps_hat, pdg_logits


# ============================================================
# DDPM  (identical to particle_diffusion.py)
# ============================================================

class DDPM:
    def __init__(self, model, T, device, cosine_s=0.008):
        self.model  = model
        self.T      = T
        self.device = device
        betas, alphas, acp, acp_prev = make_cosine_beta_schedule(
            T, s=cosine_s, device=device
        )
        self.betas    = betas
        self.alphas   = alphas
        self.acp      = acp
        self.acp_prev = acp_prev
        self.posterior_variance = betas * (1.0 - acp_prev) / (1.0 - acp)

    def p_sample(self, x_t, t, pdg_t, mask):
        B = x_t.shape[0]
        eps_hat, pdg_logits = self.model(x_t, t, pdg_t, mask)
        beta_t  = self.betas[t].view(B, 1, 1)
        alpha_t = self.alphas[t].view(B, 1, 1)
        acp_t   = self.acp[t].view(B, 1, 1)
        mu  = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_hat
        )
        var = self.posterior_variance[t].view(B, 1, 1)
        z   = torch.zeros_like(x_t) if t[0].item() == 0 else torch.randn_like(x_t)
        x_prev = (mu + torch.sqrt(var) * z) * mask.unsqueeze(-1)
        return x_prev, pdg_logits

    @torch.no_grad()
    def sample(self, mask, pdg_init):
        """
        Full reverse diffusion.

        Returns
        -------
        x   : (B, Kmax, 7)  normalised [log|E|, u_x, u_y, u_z, x_mm, y_mm, z_mm]
        pdg : (B, Kmax)     class index  0=e-  1=e+   <-- SEPARATE from x
        """
        B, K = mask.shape
        x   = torch.randn((B, K, 7), device=self.device) * mask.unsqueeze(-1)
        pdg = pdg_init.clone()

        for ti in reversed(range(self.T)):
            t = torch.full((B,), ti, device=self.device, dtype=torch.long)
            x, logits = self.p_sample(x, t, pdg, mask)
            probs = torch.softmax(logits, dim=-1)
            samp  = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, K)
            pdg   = torch.where(mask, samp, pdg)

        return x, pdg


# ============================================================
# Model loading
# ============================================================

def load_meta_and_model(outdir: str, device: str):
    meta_path = os.path.join(outdir, "meta.pt")
    ckpt_path = os.path.join(outdir, "ckpt_last.pt")
    for p in (meta_path, ckpt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    meta  = torch.load(meta_path, map_location="cpu")
    model = ParticleDenoiser(
        d_model    = meta["d_model"],
        nhead      = meta["nhead"],
        num_layers = meta["num_layers"],
        dropout    = meta["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(
        model    = model,
        T        = int(meta["T"]),
        device   = device,
        cosine_s = float(meta.get("cosine_s", 0.008)),
    )
    return meta, ddpm


# ============================================================
# Decode: model output --> .pairs rows
# ============================================================

def decode_batch(x_norm_batch, pdg_idx_batch, Ks, meta):
    """
    Convert normalised model outputs to physical .pairs columns.

    Column layout verified against the data pipeline:
      Training data loaded as ev[:, :7]:
        col 0 = E_signed [GeV]  --> model stores log|E| in col 0
        col 1-3 = beta          --> model stores beta_unsquash(beta) in cols 1-3
        col 4-6 = pos [mm]      --> model stores pos as-is in cols 4-6 (already mm)

    PDG and energy magnitude are SEPARATE model outputs:
      x_norm_batch  encodes log|E| in col 0 -- energy magnitude only
      pdg_idx_batch encodes species class   -- determines energy sign

    Parameters
    ----------
    x_norm_batch  : (B, Kmax, 7)
    pdg_idx_batch : (B, Kmax)    integers 0=e- / 1=e+
    Ks            : (B,)         true multiplicity per event
    meta          : dict from meta.pt

    Returns
    -------
    list of (K_i, 11) float32 arrays:
      [energy_GeV, betax, betay, betaz, xpos_mm, ypos_mm, zpos_mm, -99, -99, -99, -99]
    """
    mean       = np.asarray(meta["feat_mean"], dtype=np.float32)  # shape (7,)
    std        = np.asarray(meta["feat_std"],  dtype=np.float32)  # shape (7,)
    idx_to_pdg = meta["idx_to_pdg"]   # {0: 11, 1: -11}

    dummy = np.full((1, 4), -99.0, dtype=np.float32)

    events = []
    for i, K in enumerate(Ks):
        # ----------------------------------------------------------
        # Continuous head output
        # x_norm cols: [log|E|,  u_x, u_y, u_z,  x_mm, y_mm, z_mm]
        # ----------------------------------------------------------
        cont = x_norm_batch[i, :K] * std + mean   # denormalise --> physical units

        logE  = cont[:, 0]          # log|E|, so E in GeV
        u     = cont[:, 1:4]        # pre-squash velocity  (K, 3)
        pos   = cont[:, 4:7]        # position in mm  (K, 3)  -- no conversion needed

        E_abs = np.exp(logE)        # |E| in GeV, always positive
        beta  = beta_squash_np(u)   # dimensionless velocity  (K, 3)

        # ----------------------------------------------------------
        # Discrete PDG head output  (independent of continuous head)
        # 0 --> e- (pdg 11)  --> energy sign +1
        # 1 --> e+ (pdg -11) --> energy sign -1
        # ----------------------------------------------------------
        pdg_codes = np.array(
            [idx_to_pdg[int(j)] for j in pdg_idx_batch[i, :K]],
            dtype=np.int64
        )
        sign   = np.where(pdg_codes == 11, 1.0, -1.0).astype(np.float32)
        energy = sign * E_abs       # signed energy in GeV  (K,)

        # ----------------------------------------------------------
        # Assemble: [energy, betax, betay, betaz, x, y, z, -99x4]
        # ----------------------------------------------------------
        rows = np.concatenate([
            energy[:, None],            # (K, 1)
            beta,                       # (K, 3)
            pos,                        # (K, 3)  already in mm
            np.tile(dummy, (K, 1)),     # (K, 4)  dummy columns
        ], axis=1).astype(np.float32)   # (K, 11)

        events.append(rows)

    return events


# ============================================================
# Batch sampler
# ============================================================

def sample_batch(meta, ddpm, device, batch_size):
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    Kmax  = int(meta["max_particles"])
    n_pdg = int(meta["n_pdg"])

    Ks = np.random.choice(multiplicities, size=batch_size, replace=True)
    Ks = np.clip(Ks, 1, Kmax).astype(np.int64)

    mask_np = np.zeros((batch_size, Kmax), dtype=np.bool_)
    for i, K in enumerate(Ks):
        mask_np[i, :K] = True
    mask_t = torch.from_numpy(mask_np).to(device)

    pdg_init = torch.randint(0, n_pdg, (batch_size, Kmax), device=device)
    pdg_init = pdg_init * mask_t.long()

    with torch.no_grad():
        x_norm, pdg_idx = ddpm.sample(mask_t, pdg_init)

    return decode_batch(
        x_norm.cpu().numpy(),
        pdg_idx.cpu().numpy(),
        Ks, meta,
    )


# ============================================================
# .pairs writer
# ============================================================

def write_pairs_file(filepath: str, particles: np.ndarray):
    """Write one .pair file with one particle per line."""
    with open(filepath, "w") as f:
        for row in particles:
            f.write(" ".join(f"{v:.8g}" for v in row) + "\n")


# ============================================================
# Sanity check on a decoded batch
# ============================================================

def sanity_check(events):
    all_rows = np.concatenate(events, axis=0)
    energies = all_rows[:, 0]
    betas    = all_rows[:, 1:4]
    bmag     = np.linalg.norm(betas, axis=1)

    n_eminus  = int(np.sum(energies > 0))
    n_eplus   = int(np.sum(energies < 0))
    n_zero    = int(np.sum(energies == 0))

    print(f"  Sanity check over {len(all_rows)} particles:")
    print(f"    e-  (energy>0) : {n_eminus}")
    print(f"    e+  (energy<0) : {n_eplus}")
    print(f"    zero energy    : {n_zero}  (should be 0)")
    print(f"    |beta| range   : [{bmag.min():.4f}, {bmag.max():.4f}]  (should be <1)")
    print(f"    |E|   range    : [{np.abs(energies).min():.4e}, {np.abs(energies).max():.4e}] GeV")
    print(f"    dummy col [7]  : {np.unique(all_rows[:, 7])}  (should be [-99])")

    if bmag.max() >= 1.0:
        print("  WARNING: some |beta| >= 1 — check beta_squash")
    if n_zero > 0:
        print("  WARNING: zero-energy particles found")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate .pairs files from a trained particle diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--outdir",            type=str, required=True,
                        help="Directory containing meta.pt and ckpt_last.pt")
    parser.add_argument("--pairsdir",          type=str, default=None,
                        help="Output directory for .pair files "
                             "(default: <outdir>/pairs)")
    parser.add_argument("--n_events",          type=int, default=50000,
                        help="Total number of events to generate")
    parser.add_argument("--sample_batch_size", type=int, default=32,
                        help="Events per GPU batch — increase until OOM")
    parser.add_argument("--start_index",       type=int, default=0,
                        help="Start file numbering at this index "
                             "(for SLURM array jobs to avoid filename collisions)")
    parser.add_argument("--seed",              type=int, default=42,
                        help="Global RNG seed")
    parser.add_argument("--sanity_batches",    type=int, default=2,
                        help="Run sanity check on this many initial batches (0=skip)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    if device == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")

    if args.pairsdir is None:
        args.pairsdir = os.path.join(args.outdir, "pairs")
    os.makedirs(args.pairsdir, exist_ok=True)

    print(f"Loading model from: {args.outdir}")
    meta, ddpm = load_meta_and_model(args.outdir, device)
    print(f"  T             = {meta['T']}")
    print(f"  d_model       = {meta['d_model']}")
    print(f"  nhead         = {meta['nhead']}")
    print(f"  num_layers    = {meta['num_layers']}")
    print(f"  max_particles = {meta['max_particles']}")
    print(f"  idx_to_pdg    = {meta['idx_to_pdg']}")
    print(f"  feat_mean     = {meta['feat_mean']}")
    print(f"  feat_std      = {meta['feat_std']}")
    print(f"  Positions stored in mm (no unit conversion applied)")

    n_events   = args.n_events
    batch_size = args.sample_batch_size
    n_batches  = math.ceil(n_events / batch_size)
    n_done     = 0
    file_idx   = args.start_index

    print(f"\nGenerating {n_events} events "
          f"({n_batches} batches of up to {batch_size}) ...")

    for batch_num in tqdm(range(n_batches), desc="Batches"):
        remaining = n_events - n_done
        bs = min(batch_size, remaining)

        batch_events = sample_batch(meta, ddpm, device, bs)

        # Optional sanity check on first few batches
        if batch_num < args.sanity_batches:
            print(f"\n--- Batch {batch_num} sanity check ---")
            sanity_check(batch_events)

        for particles in batch_events:
            fname = os.path.join(args.pairsdir, f"output_{file_idx:05d}.pair")
            write_pairs_file(fname, particles)
            file_idx += 1

        n_done += bs

    print(f"\nDone. {n_done} .pair files written to: {args.pairsdir}")
    print(f"File range: output_{args.start_index:05d}.pair "
          f"--> output_{file_idx - 1:05d}.pair")


if __name__ == "__main__":
    main()