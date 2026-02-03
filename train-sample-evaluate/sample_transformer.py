#!/usr/bin/env python3
"""
Generate synthetic MCParticle-like events with a trained PDG-conditioned DDPM.

What this script does:
- Loads training metadata (meta.pt) and the last checkpoint (ckpt_last.pt)
- Recreates the ParticleDenoiser model with the same hyperparameters as training
- Uses the DDPM sampler to generate momenta (px, py, pz) for a randomly chosen number of particles K per event
- Samples PDG codes from the empirical PDG pool saved during training
- Denormalises momenta back to physical units using stats stored in meta.pt
- Saves a numpy object array: list of events, each event is an array of shape (K, 4) with [pdg, px, py, pz]
"""

import os
import numpy as np
import torch

from train_transformer import ParticleDenoiser, DDPM, CFG


def load_meta_and_model(outdir: str, device: str):
    """
    Load:
      - meta.pt: contains everything needed to rebuild the model and sampling distributions
      - ckpt_last.pt: contains trained model weights

    Returns: (meta_dict, model, ddpm)
    """
    # --- Load metadata saved at training time (always load meta on CPU)
    meta_path = os.path.join(outdir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu")

    # --- Rebuild the model EXACTLY as in training (use hyperparams from meta)
    model = ParticleDenoiser(
        vocab_size=meta["vocab_size"],
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(device)

    # --- Load weights
    ckpt_path = os.path.join(outdir, "ckpt_last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Wrap in DDPM helper (same diffusion schedule as training)
    ddpm = DDPM(
        model=model,
        T=int(meta["T"]),
        beta_start=0.0,   # unused
        beta_end=0.0,     # unused
        device=device,
    )


    return meta, model, ddpm


def sample_one_event(meta: dict, ddpm: DDPM, device: str):
    """
    Sample a single event.

    Output:
      event_rows: np.ndarray of shape (K, 4) with columns [pdg, px, py, pz]
    """
    # ---------- 1) Choose how many particles K the event should have ----------
    # During training you stored an empirical distribution of multiplicities,
    # so we sample K from that list.
    multiplicities = np.asarray(meta["multiplicities"], dtype=np.int64)
    K = int(np.random.choice(multiplicities))

    # Clip K to what the network supports (it was trained with fixed max_particles)
    Kmax = int(meta["max_particles"])
    K = max(1, min(K, Kmax))

    # ---------- 2) Sample PDG codes ----------
    # You also stored an empirical pool of PDG codes seen in training.
    pdg_pool = np.asarray(meta["pdg_pool"], dtype=np.int64)
    pdg_vals = np.random.choice(pdg_pool, size=K, replace=True)  # actual PDG codes, length K

    # Convert PDG codes -> token IDs used by the embedding lookup
    pdg_to_id = meta["pdg_to_id"]
    pdg_ids = np.array([pdg_to_id[int(p)] for p in pdg_vals], dtype=np.int64)  # length K

    # ---------- 3) Build padded tokens + mask ----------
    # Model expects fixed-length sequences of length Kmax
    # - pdg_pad: shape (Kmax,) integer token IDs
    # - mask:    shape (Kmax,) bool indicating which positions are real (True) vs padding (False)
    pdg_pad = np.zeros((Kmax,), dtype=np.int64)
    mask = np.zeros((Kmax,), dtype=np.bool_)

    pdg_pad[:K] = pdg_ids
    mask[:K] = True

    # Add batch dimension (1, Kmax) and move to device
    pdg_t = torch.from_numpy(pdg_pad).unsqueeze(0).to(device)   # (1, Kmax)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)     # (1, Kmax)

    # ---------- 4) DDPM sample in *normalised* momentum space ----------
    # ddpm.sample returns something like (B, Kmax, 3) in the normalised space.
    # We take batch 0 and then keep only the first K rows (real particles).
    with torch.no_grad():
        x_norm = ddpm.sample(pdg_t, mask_t)[0].detach().cpu().numpy()  # (Kmax, 3)
    x_norm = x_norm[:K]  # (K, 3)

    # ---------- 5) Denormalise to physical (px, py, pz) ----------
    # Prefer per-PDG stats if available, else fall back to global stats.
    mom_mean_by_id = meta.get("mom_mean_by_id", None)
    mom_std_by_id = meta.get("mom_std_by_id", None)

    if mom_mean_by_id is not None and mom_std_by_id is not None:
        # pdg_ids is length K, so this indexes out mean/std for each particle
        mean = mom_mean_by_id[pdg_ids]  # (K, 3)
        std = mom_std_by_id[pdg_ids]    # (K, 3)
        mom = x_norm * std + mean       # (K, 3)
    else:
        # Older meta: use global stats
        mom_mean = meta["mom_mean"]     # (3,)
        mom_std = meta["mom_std"]       # (3,)
        mom = x_norm * mom_std + mom_mean

    # ---------- 6) Pack into output format ----------
    # Each row is: [pdg, px, py, pz]
    event_rows = np.zeros((K, 4), dtype=np.float32)
    event_rows[:, 0] = pdg_vals.astype(np.float32)
    event_rows[:, 1:4] = mom.astype(np.float32)

    return event_rows


def main():
    # ---- Config: keep this tiny and obvious
    cfg = CFG()
    cfg.outdir = "mc_gen1_model_changes"  # where to load model from
    device = cfg.device

    # ---- How many events to generate
    n_events = 1247

    # ---- Load everything needed for sampling
    meta, model, ddpm = load_meta_and_model(cfg.outdir, device)

    # ---- Generate a list of events (variable K, so we save as object array)
    from tqdm import trange

    # ...

    events = []
    for _ in trange(n_events, desc="Generating events"):
        events.append(sample_one_event(meta, ddpm, device))

    # ---- Save
    # Using dtype=object because each event has different K (ragged array)
    out_path = os.path.join(cfg.outdir, "generated_events.npy")
    np.save(out_path, np.array(events, dtype=object))
    print("Saved:", out_path)
    print("Format: list of events; each event is (K,4) float32 rows [pdg, px, py, pz]")


if __name__ == "__main__":
    main()
