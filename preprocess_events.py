#!/usr/bin/env python3
"""
preprocess_events.py

Run ONCE to preprocess real + generated event files into a cache of
ready-to-plot numpy arrays.  evaluate_steps.py will then skip all
extraction/sanitising and load the cache directly.

What this does:
  1. Loads every raw .npy file (real + all gen steps).
  2. Sanitises events (handles guineapig-format and explicit-PDG format).
  3. Splits into species: e− (pdg=11), e+ (pdg=-11), all.
  4. Filters outliers via configurable IQR / quantile cuts.
  5. Saves one .npy cache file per (source × species) in --cache_dir.

Output files:
  <cache_dir>/real_all.npy
  <cache_dir>/real_eminus.npy
  <cache_dir>/real_eplus.npy
  <cache_dir>/gen_{step}steps_all.npy
  <cache_dir>/gen_{step}steps_eminus.npy
  <cache_dir>/gen_{step}steps_eplus.npy

Each file is a dict saved as a .npy (np.save with allow_pickle=True) with keys:
  mult, px, py, pz, p, pt, E_abs, E_signed, beta_mag,
  x, y, z, betax, betay, betaz
  + metadata: n_events_raw, n_events_kept, filter_params, source_path

Usage:
  python preprocess_events.py \\
    --real_path guineapig_raw_trimmed.npy \\
    --gen_dir   results_nosquash \\
    --steps     25 50 100 150 200 \\
    --cache_dir preprocessed_cache

Then pass --cache_dir to evaluate_steps.py (see updated loader section).
"""

from __future__ import annotations
import os
import argparse
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


# ──────────────────────────────────────────────
# Raw loading
# ──────────────────────────────────────────────
def load_events(path: str) -> list:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 7:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unrecognised format in {path}")


# ──────────────────────────────────────────────
# Sanitise a single event → common arrays
# ──────────────────────────────────────────────
def sanitize_event(ev):
    ev = np.asarray(ev, dtype=np.float64)
    if ev.ndim != 2 or ev.shape[0] == 0:
        empty = np.array([], dtype=np.float64)
        return (empty.astype(np.int64),) + (empty,) * 12

    if ev.shape[1] >= 8:                     # explicit PDG format
        pdg      = ev[:, 0].astype(np.int64)
        Eabs     = np.abs(ev[:, 1])
        betax, betay, betaz = ev[:, 2], ev[:, 3], ev[:, 4]
        x, y, z  = ev[:, 5], ev[:, 6], ev[:, 7]
        E_signed = np.where(pdg == -11, -Eabs, Eabs)

    elif ev.shape[1] >= 7:                   # guineapig format
        E_signed = ev[:, 0]
        betax, betay, betaz = ev[:, 1], ev[:, 2], ev[:, 3]
        x, y, z  = ev[:, 4], ev[:, 5], ev[:, 6]
        pdg      = np.where(E_signed >= 0, 11, -11).astype(np.int64)
        Eabs     = np.abs(E_signed)

    else:
        empty = np.array([], dtype=np.float64)
        return (empty.astype(np.int64),) + (empty,) * 12

    beta_mag = np.sqrt(betax**2 + betay**2 + betaz**2)
    px, py, pz = Eabs * betax, Eabs * betay, Eabs * betaz
    return (pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz)


# ──────────────────────────────────────────────
# Outlier filtering
# ──────────────────────────────────────────────
def iqr_bounds(arr, k=5.0):
    """Return (lo, hi) = median ± k*IQR.  NaN-safe."""
    a = arr[np.isfinite(arr)]
    if a.size < 4:
        return -np.inf, np.inf
    q25, q75 = np.percentile(a, 25), np.percentile(a, 75)
    iqr = q75 - q25
    return q25 - k * iqr, q75 + k * iqr


def quantile_bounds(arr, q_lo=0.0005, q_hi=0.9995):
    a = arr[np.isfinite(arr)]
    if a.size < 4:
        return -np.inf, np.inf
    return float(np.quantile(a, q_lo)), float(np.quantile(a, q_hi))


def filter_particles(
    pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz,
    iqr_k=5.0,
    q_lo=0.0005,
    q_hi=0.9995,
    min_energy=0.0,
    max_beta=1.0,
):
    """
    Build a boolean mask that removes unphysical / extreme-outlier particles.
    Criteria applied:
      - Energy > min_energy  (default 0, i.e. keep all positive)
      - beta_mag <= max_beta (physical: nothing faster than light)
      - beta_mag > 0         (remove stopped particles / NaN rows)
      - Each of betax, betay, betaz, x, y, z, px, py, pz within IQR bounds
      - E_abs within quantile bounds
    """
    n = len(pdg)
    mask = np.ones(n, dtype=bool)

    # Physical cuts
    mask &= np.isfinite(Eabs) & (Eabs > min_energy)
    mask &= np.isfinite(beta_mag) & (beta_mag > 0) & (beta_mag <= max_beta)

    # Per-variable outlier removal
    for arr in (betax, betay, betaz, px, py, pz):
        lo, hi = iqr_bounds(arr[mask], k=iqr_k)
        mask &= (arr >= lo) & (arr <= hi)

    # Spatial: wider quantile cut (positions can be legitimately spread)
    for arr in (x, y, z):
        lo, hi = quantile_bounds(arr[mask], q_lo=q_lo, q_hi=q_hi)
        mask &= (arr >= lo) & (arr <= hi)

    # Energy: quantile cut
    lo, hi = quantile_bounds(Eabs[mask], q_lo=q_lo, q_hi=q_hi)
    mask &= (Eabs >= lo) & (Eabs <= hi)

    return mask


# ──────────────────────────────────────────────
# Extract + filter one species from raw events
# ──────────────────────────────────────────────
def extract_and_filter(events, pdgs=None, filter_cfg=None):
    """
    Returns a dict of ready-to-plot arrays for one species.
    filter_cfg: dict of kwargs forwarded to filter_particles, or None to skip.
    """
    fc = filter_cfg or {}

    n_events = len(events)
    mult_raw  = np.zeros(n_events, dtype=np.int64)
    mult_kept = np.zeros(n_events, dtype=np.int64)

    cols = {k: [] for k in (
        "px","py","pz","E_abs","E_signed","beta_mag",
        "x","y","z","betax","betay","betaz"
    )}

    for i, ev in enumerate(events):
        (pdg, px, py, pz, Eabs, E_signed, bmag,
         x, y, z, betax, betay, betaz) = sanitize_event(ev)

        if len(pdg) == 0:
            continue

        # Species selection
        if pdgs is None:
            sel = np.ones(len(pdg), dtype=bool)
        else:
            sel = np.zeros(len(pdg), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)

        mult_raw[i] = int(sel.sum())
        if not sel.any():
            continue

        # Slice to species
        args_sp = (pdg[sel], px[sel], py[sel], pz[sel],
                   Eabs[sel], E_signed[sel], bmag[sel],
                   x[sel] if x.size else np.array([]),
                   y[sel] if y.size else np.array([]),
                   z[sel] if z.size else np.array([]),
                   betax[sel], betay[sel], betaz[sel])

        # Outlier filter
        if filter_cfg is not None:
            fmask = filter_particles(*args_sp, **fc)
            args_sp = tuple(a[fmask] if a.size else a for a in args_sp)
            mult_kept[i] = int(fmask.sum())
        else:
            mult_kept[i] = mult_raw[i]

        _, px_, py_, pz_, Eabs_, E_signed_, bmag_, x_, y_, z_, bx_, by_, bz_ = args_sp

        cols["px"].append(px_);  cols["py"].append(py_);  cols["pz"].append(pz_)
        cols["E_abs"].append(Eabs_);  cols["E_signed"].append(E_signed_)
        cols["beta_mag"].append(bmag_)
        cols["betax"].append(bx_);  cols["betay"].append(by_);  cols["betaz"].append(bz_)
        if x_.size:
            cols["x"].append(x_);  cols["y"].append(y_);  cols["z"].append(z_)

    def cat(lst):
        return np.concatenate(lst) if lst else np.array([], dtype=np.float64)

    result = {k: cat(v) for k, v in cols.items()}
    result["mult_raw"]  = mult_raw
    result["mult_kept"] = mult_kept
    # "mult" used by evaluate_steps = kept particles per event
    result["mult"]      = mult_kept

    px_ = result["px"]; py_ = result["py"]; pz_ = result["pz"]
    result["p"]  = np.sqrt(px_**2 + py_**2 + pz_**2)
    result["pt"] = np.sqrt(px_**2 + py_**2)

    # Summary stats (handy for sanity checks)
    result["n_events"] = n_events
    result["n_particles_raw"]  = int(mult_raw.sum())
    result["n_particles_kept"] = int(mult_kept.sum())

    return result


# ──────────────────────────────────────────────
# Worker (top-level for pickling)
# ──────────────────────────────────────────────
def _worker(task):
    path, pdgs, tag, label, filter_cfg = task
    events = load_events(path)
    result = extract_and_filter(events, pdgs=pdgs, filter_cfg=filter_cfg)
    result["source_path"] = path
    result["label"]       = label
    return label, tag, result


# ──────────────────────────────────────────────
# Normalisation (optional, applied after filter)
# ──────────────────────────────────────────────
def compute_normalisation(cache_dict, keys=None):
    """
    Compute per-variable (mean, std) across ALL species combined for a
    given source label.  Returns a stats dict you can save alongside
    the cache for use in training/inference.
    """
    if keys is None:
        keys = ["px","py","pz","p","pt","E_abs","E_signed",
                "beta_mag","x","y","z","betax","betay","betaz"]
    stats = {}
    for k in keys:
        vals = np.concatenate([d[k] for d in cache_dict.values()
                               if k in d and len(d[k])])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stats[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                        "min":  float(vals.min()),  "max": float(vals.max()),
                        "q01":  float(np.quantile(vals, 0.01)),
                        "q99":  float(np.quantile(vals, 0.99))}
        else:
            stats[k] = {"mean": 0.0, "std": 1.0}
    return stats


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw event .npy files into a plotting-ready cache."
    )
    parser.add_argument("--real_path",  required=True,
                        help="Path to real events .npy file")
    parser.add_argument("--gen_dir",    required=True,
                        help="Directory containing generated_events_{step}steps.npy")
    parser.add_argument("--steps",      type=int, nargs="+", default=[25, 50, 100, 150, 200])
    parser.add_argument("--cache_dir",  default="preprocessed_cache",
                        help="Where to write the processed .npy files")
    parser.add_argument("--workers",    type=int, default=None,
                        help="Parallel workers (default: cpu count)")

    # Filter options
    parser.add_argument("--no_filter",  action="store_true",
                        help="Skip outlier filtering entirely")
    parser.add_argument("--iqr_k",      type=float, default=5.0,
                        help="IQR multiplier for momentum/beta outlier cuts (default 5)")
    parser.add_argument("--q_lo",       type=float, default=0.0005,
                        help="Lower quantile for energy/position cuts (default 0.0005)")
    parser.add_argument("--q_hi",       type=float, default=0.9995,
                        help="Upper quantile for energy/position cuts (default 0.9995)")
    parser.add_argument("--min_energy", type=float, default=0.0,
                        help="Minimum particle energy to keep in GeV (default 0)")
    parser.add_argument("--max_beta",   type=float, default=1.0,
                        help="Maximum |beta| to keep (default 1.0, i.e. physical)")

    # Normalisation
    parser.add_argument("--save_norm",  action="store_true",
                        help="Compute and save normalisation stats for real data")

    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    filter_cfg = None if args.no_filter else {
        "iqr_k":      args.iqr_k,
        "q_lo":       args.q_lo,
        "q_hi":       args.q_hi,
        "min_energy": args.min_energy,
        "max_beta":   args.max_beta,
    }

    species_list = [
        {"name": "e−",  "pdgs": [11],  "tag": "eminus"},
        {"name": "e+",  "pdgs": [-11], "tag": "eplus"},
        {"name": "all", "pdgs": None,  "tag": "all"},
    ]

    # Validate gen files exist
    gen_paths = {}
    for s in args.steps:
        p = os.path.join(args.gen_dir, f"generated_events_{s}steps.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")
        gen_paths[s] = p

    # Build task list
    tasks = []
    for sp in species_list:
        tasks.append((args.real_path, sp["pdgs"], sp["tag"], "real", filter_cfg))
    for s, p in gen_paths.items():
        for sp in species_list:
            tasks.append((p, sp["pdgs"], sp["tag"], f"gen_{s}steps", filter_cfg))

    n_workers = args.workers if args.workers is not None else min(len(tasks), os.cpu_count() or 1)
    print(f"Preprocessing {len(tasks)} tasks with {n_workers} workers …")
    print(f"Filter config: {filter_cfg or 'DISABLED'}\n")

    t0 = time.time()
    results = {}   # (label, tag) → dict

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_worker, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                label, tag, data = fut.result()
                results[(label, tag)] = data
                done += 1
                pct = data["n_particles_kept"] / max(data["n_particles_raw"], 1) * 100
                print(f"  [{done:3d}/{len(tasks)}]  {label:20s}  {tag:7s}  "
                      f"particles kept: {data['n_particles_kept']:>8,} / "
                      f"{data['n_particles_raw']:>8,}  ({pct:.1f}%)")
    else:
        for i, t in enumerate(tasks):
            label, tag, data = _worker(t)
            results[(label, tag)] = data
            pct = data["n_particles_kept"] / max(data["n_particles_raw"], 1) * 100
            print(f"  [{i+1:3d}/{len(tasks)}]  {label:20s}  {tag:7s}  "
                  f"particles kept: {data['n_particles_kept']:>8,} / "
                  f"{data['n_particles_raw']:>8,}  ({pct:.1f}%)")

    # Save cache files
    print(f"\nSaving cache to: {args.cache_dir}")
    saved = []
    for (label, tag), data in results.items():
        fname = f"{label}_{tag}.npy"
        fpath = os.path.join(args.cache_dir, fname)
        np.save(fpath, data, allow_pickle=True)
        saved.append(fpath)
        print(f"  Saved → {fname}")

    # Optional normalisation stats (real data only)
    if args.save_norm:
        real_cache = {tag: results[("real", sp["tag"])]
                      for sp in species_list
                      for tag in [sp["tag"]]
                      if ("real", sp["tag"]) in results}
        stats = compute_normalisation(real_cache)
        stats_path = os.path.join(args.cache_dir, "normalisation_stats.npy")
        np.save(stats_path, stats, allow_pickle=True)
        print(f"\nNormalisation stats → {stats_path}")
        print("  Variables:")
        for k, v in stats.items():
            print(f"    {k:12s}  mean={v['mean']:+.4e}  std={v['std']:.4e}  "
                  f"range=[{v['min']:.3e}, {v['max']:.3e}]")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.  {len(saved)} files written to: {args.cache_dir}")
    print("\n── How to use with evaluate_steps.py ──────────────────────────────")
    print(f"  python evaluate_steps.py \\")
    print(f"    --cache_dir {args.cache_dir} \\")
    print(f"    --steps {' '.join(str(s) for s in args.steps)} \\")
    print(f"    --outdir your_plot_dir")
    print("────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
