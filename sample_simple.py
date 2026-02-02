#!/usr/bin/env python3
"""
Generate synthetic MCParticle-like events with a trained PDG-conditioned DDPM (MLP denoiser).

- Loads meta.pt + ckpt_last.pt from training
- Rebuilds ParticleDenoiser with the NEW MLP signature
- Samples variable-length events, denormalises, saves generated_events.npy (dtype=object)
"""

import os
import numpy as np
import torch

from train_simple import ParticleDenoiser, DDPM, CFG


def load_meta_and_model(outdir: str, device: str):
    """
    Load:
      - meta.pt: everything needed to rebuild model + sampling distributions
      - ckpt_last.pt: trained weights

    Returns: (meta_dict, model, ddpm)
    """
    meta_path = os.path.join(outdir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu")

    # --- NEW MLP denoiser hyperparams (fallback safely if not stored)
    d_model = int(meta.get("d_model", 128))
    hidden = int(meta.get("hidden", 256))      # if you didn't store this, defaults to 256
    n_layers = int(meta.get("n_layers", 3))    # if you didn't store this, defaults to 3
    dropout = float(meta.get("dropout", 0.1))

    model = ParticleDenoiser(
        vocab_size=int(meta["vocab_size"]),
        d_model=d_model,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    ckpt_path = os.path.join(outdir, "ckpt_last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
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
    """
    Sample a single event.
    Output: np.ndarray (K,4): [pdg, px, py, pz]
    """
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    K = int(np.random.choice(multiplicities))

    Kmax = int(meta["max_particles"])
    K = max(1, min(K, Kmax))

    pdg_pool = np.asarray(meta["pdg_pool"], dtype=np.int64)
    pdg_vals = np.random.choice(pdg_pool, size=K, replace=True)

    pdg_to_id = meta["pdg_to_id"]
    pdg_ids = np.array([pdg_to_id[int(p)] for p in pdg_vals], dtype=np.int64)

    pdg_pad = np.zeros((Kmax,), dtype=np.int64)
    mask = np.zeros((Kmax,), dtype=np.bool_)
    pdg_pad[:K] = pdg_ids
    mask[:K] = True

    pdg_t = torch.from_numpy(pdg_pad).unsqueeze(0).to(device)     # (1,Kmax)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)       # (1,Kmax)

    with torch.no_grad():
        x_norm = ddpm.sample(pdg_t, mask_t)[0].detach().cpu().numpy()  # (Kmax,3)
    x_norm = x_norm[:K]  # (K,3)

    mom_mean_by_id = meta.get("mom_mean_by_id", None)
    mom_std_by_id = meta.get("mom_std_by_id", None)

    if mom_mean_by_id is not None and mom_std_by_id is not None:
        mean = mom_mean_by_id[pdg_ids]
        std = mom_std_by_id[pdg_ids]
        mom = x_norm * std + mean
    else:
        mom_mean = meta["mom_mean"]
        mom_std = meta["mom_std"]
        mom = x_norm * mom_std + mom_mean

    event_rows = np.zeros((K, 4), dtype=np.float32)
    event_rows[:, 0] = pdg_vals.astype(np.float32)
    event_rows[:, 1:4] = mom.astype(np.float32)
    return event_rows


def main():
    cfg = CFG()
    cfg.outdir = "mc_gen1_model_simple"  # folder containing meta.pt + ckpt_last.pt
    device = cfg.device

    n_events = 1247

    meta, model, ddpm = load_meta_and_model(cfg.outdir, device)

    from tqdm import trange
    events = []
    for _ in trange(n_events, desc="Generating events"):
        events.append(sample_one_event(meta, ddpm, device))

    out_path = os.path.join(cfg.outdir, "generated_events.npy")
    np.save(out_path, np.array(events, dtype=object))
    print("Saved:", out_path)
    print("Format: list of events; each event is (K,4) float32 rows [pdg, px, py, pz]")


if __name__ == "__main__":
    main()
