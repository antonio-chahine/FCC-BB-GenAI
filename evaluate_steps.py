#!/usr/bin/env python3
"""
evaluate_steps.py  (optimised + consolidated-panel edition)

Changes vs. original:
  1. Speed:
     - Pre-extract ALL species×steps in one parallel pass (ProcessPoolExecutor).
     - Extraction and metric computation are parallelised across species/steps.
     - Array work is vectorised; no redundant reloads.
  2. Consolidated panel plots:
     - Four panel groups per species: Energy, Beta, Position, Momentum.
     - Each panel group = one figure with subplots (one sub-plot per component)
       + a shared fractional-difference row at the bottom.
     - Individual per-variable PNGs are no longer generated (saves I/O time).

Example:
  python evaluate_steps.py \\
    --real_path guineapig_raw_trimmed.npy \\
    --gen_dir results_nosquash \\
    --steps 25 50 100 150 200 \\
    --outdir results_nosquash/plots_steps \\
    --step_for_corner 200 \\
    --step_for_mult 200
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# ──────────────────────────────────────────────
# Styling / IO
# ──────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.grid": False,
    })


def savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_events(path: str):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 7:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"Unrecognised format in {path}")


# ──────────────────────────────────────────────
# Physics helpers
# ──────────────────────────────────────────────
def sanitize_event(ev, me=0.000511):
    ev = np.asarray(ev, dtype=np.float64)

    if ev.ndim == 2 and ev.shape[1] >= 8:
        pdg    = ev[:, 0].astype(np.int64)
        Eabs   = np.abs(ev[:, 1])
        betax, betay, betaz = ev[:, 2], ev[:, 3], ev[:, 4]
        x, y, z = ev[:, 5], ev[:, 6], ev[:, 7]
        E_signed = np.where(pdg == -11, -Eabs, Eabs)
    elif ev.ndim == 2 and ev.shape[1] >= 7:
        E_signed = ev[:, 0]
        betax, betay, betaz = ev[:, 1], ev[:, 2], ev[:, 3]
        x, y, z = ev[:, 4], ev[:, 5], ev[:, 6]
        pdg  = np.where(E_signed >= 0.0, 11, -11).astype(np.int64)
        Eabs = np.abs(E_signed)
    else:
        empty = np.array([], dtype=np.float64)
        return (empty.astype(np.int64),) + (empty,) * 11

    beta_mag = np.sqrt(betax**2 + betay**2 + betaz**2)
    pvec = Eabs[:, None] * np.stack([betax, betay, betaz], axis=1)
    px, py, pz = pvec[:, 0], pvec[:, 1], pvec[:, 2]
    return (pdg, px, py, pz, Eabs, E_signed, beta_mag, x, y, z, betax, betay, betaz)


def extract_species(events, pdgs=None, me=0.000511):
    """Extract particle arrays for one species from a list of events."""
    n = len(events)
    mult = np.zeros(n, dtype=np.int64)
    cols = {k: [] for k in ("px","py","pz","E_abs","E_signed","beta_mag",
                             "x","y","z","betax","betay","betaz")}

    for i, ev in enumerate(events):
        (pdg, px, py, pz, Eabs, E_signed, bmag,
         x, y, z, betax, betay, betaz) = sanitize_event(ev, me=me)

        if pdgs is None:
            sel = np.ones(len(px), dtype=bool)
        else:
            sel = np.zeros(len(px), dtype=bool)
            for code in pdgs:
                sel |= (pdg == code)

        mult[i] = int(sel.sum())
        if sel.any():
            cols["px"].append(px[sel]);   cols["py"].append(py[sel]);   cols["pz"].append(pz[sel])
            cols["E_abs"].append(Eabs[sel]);    cols["E_signed"].append(E_signed[sel])
            cols["beta_mag"].append(bmag[sel])
            cols["betax"].append(betax[sel]);   cols["betay"].append(betay[sel]); cols["betaz"].append(betaz[sel])
            if x.size:
                cols["x"].append(x[sel]); cols["y"].append(y[sel]); cols["z"].append(z[sel])

    def cat(lst):
        return np.concatenate(lst) if lst else np.array([], dtype=np.float64)

    result = {k: cat(v) for k, v in cols.items()}
    result["mult"] = mult
    px_ = result["px"]; py_ = result["py"]; pz_ = result["pz"]
    result["p"]  = np.sqrt(px_**2 + py_**2 + pz_**2)
    result["pt"] = np.sqrt(px_**2 + py_**2)
    return result


# ──────────────────────────────────────────────
# Parallelism helpers  (top-level so picklable)
# ──────────────────────────────────────────────
def _extract_task(args_tuple):
    """Worker: loads events and extracts one (species, step) pair."""
    path, pdgs, me, step, tag = args_tuple
    events = load_events(path)
    return tag, step, extract_species(events, pdgs, me)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def kl_divergence_from_counts(p_counts, q_counts, eps=1e-12):
    p = np.asarray(p_counts, dtype=np.float64); q = np.asarray(q_counts, dtype=np.float64)
    p /= p.sum() + eps;  q /= q.sum() + eps
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d(x, y):
    if not len(x) or not len(y): return np.nan
    if HAVE_SCIPY: return float(wasserstein_distance(x, y))
    xs = np.sort(x.astype(np.float64)); ys = np.sort(y.astype(np.float64))
    q  = np.linspace(0., 1., 400)
    return float(np.mean(np.abs(
        np.interp(q, np.linspace(0., 1., len(xs)), xs) -
        np.interp(q, np.linspace(0., 1., len(ys)), ys)
    )))


def _finite(arr):
    a = np.asarray(arr, dtype=np.float64)
    return a[np.isfinite(a)]


def choose_range(real, gens, key, frac_range=0.80, q_lo=0.005, q_hi=0.995):
    r = _finite(real)
    gcat = np.concatenate([_finite(g) for g in gens if len(g)]) if gens else np.array([])
    z = np.concatenate([r, gcat])
    if z.size < 2: return None

    if key in ("E_abs", "E_signed"):
        lo = 0.0 if key == "E_abs" else float(np.quantile(z, q_lo))
        hi = float(np.quantile(z, frac_range))
        if not np.isfinite(hi) or hi <= lo:
            hi = float(np.quantile(z, 0.99))
        return (lo, hi)

    lo = float(np.quantile(z, q_lo))
    hi = float(np.quantile(z, q_hi))
    return (lo, hi) if (np.isfinite(lo) and np.isfinite(hi) and hi > lo) else None


def hist_prob(arr, bins, lo, hi):
    a = _finite(arr); a = a[(a >= lo) & (a <= hi)]
    counts, edges = np.histogram(a, bins=bins, range=(lo, hi))
    prob = counts / max(counts.sum(), 1)
    return prob, counts, 0.5 * (edges[:-1] + edges[1:])


# ──────────────────────────────────────────────
# Consolidated panel plot
# ──────────────────────────────────────────────
PANEL_GROUPS = {
    # group_tag : (title, [(key, xlabel, frac_range, logy), ...])
    "energy": ("Energy", [
        ("E_abs",    "|E| [GeV]",    0.40, True),
        ("E_signed", "E_signed [GeV]", 0.40, True),
    ]),
    "beta": (r"$\beta$ components", [
        ("betax", r"$\beta_x$", 0.60, False),
        ("betay", r"$\beta_y$", 0.60, False),
        ("betaz", r"$\beta_z$", 0.60, False),
    ]),
    "position": ("Position", [
        ("x", "x [nm]", 0.98, False),
        ("y", "y [nm]", 0.98, False),
        ("z", "z [nm]", 0.98, False),
    ]),
    "momentum": ("Momentum", [
        ("px",  "p_x [GeV]", 0.60, False),
        ("py",  "p_y [GeV]", 0.60, False),
        ("pz",  "p_z [GeV]", 0.60, False),
        ("pt",  "p_T [GeV]", 0.60, False),
        ("p",   "|p| [GeV]", 0.60, False),
    ]),
}

# Which panel groups each species gets
SPECIES_PANELS = {
    "all":    ["energy", "beta", "position", "momentum"],
    "eminus": ["energy", "beta", "position", "momentum"],
    "eplus":  ["energy", "beta", "position", "momentum"],
}

# For "all" species, use E_signed; for charged species, use E_abs
ENERGY_KEY_BY_TAG = {
    "all":    "E_signed",
    "eminus": "E_abs",
    "eplus":  "E_abs",
}


def plot_panel_group(
    group_title,
    vars_info,          # [(key, xlabel, frac_range, logy), ...]
    real_sp,
    gen_by_step,        # {step: sp_dict}
    steps,
    outpath,
    species_name,
    bins=120,
    ratio_min_count=10,
):
    """
    One figure per panel group.  Layout:
      top row  : distribution subplots (one per variable)
      bottom row: fractional-difference subplots (shared x with top)
    """
    # Filter variables that have data
    valid = []
    for key, xlabel, frac_range, logy in vars_info:
        real_arr = real_sp.get(key, np.array([]))
        if _finite(real_arr).size == 0:
            continue
        gens = [gen_by_step[s].get(key, np.array([])) for s in steps]
        if all(_finite(g).size == 0 for g in gens):
            continue
        valid.append((key, xlabel, frac_range, logy))

    if not valid:
        return

    ncols = len(valid)
    fig_w = max(5.5, 3.2 * ncols)
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(fig_w, 6.4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        constrained_layout=True,
    )
    if ncols == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"{group_title} — {species_name}", y=1.02)

    for col, (key, xlabel, frac_range, logy) in enumerate(valid):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.tick_params(labelbottom=False)

        real_arr = real_sp.get(key, np.array([]))
        gens_arr = [gen_by_step[s].get(key, np.array([])) for s in steps]
        rng = choose_range(real_arr, gens_arr, key=key, frac_range=frac_range)

        if rng is None:
            ax_top.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax_top.transAxes, fontsize=8)
            ax_top.set_axis_off(); ax_bot.set_axis_off()
            continue

        lo, hi = rng
        real_use = _finite(real_arr); real_use = real_use[(real_use >= lo) & (real_use <= hi)]
        ax_top.hist(real_use, bins=bins, range=(lo, hi), density=True,
                    alpha=0.50, label="Simulated", color="C0")

        r_prob, r_counts, centers = hist_prob(real_arr, bins, lo, hi)

        within_bounds_pcts = []   # [(step, pct_within_10pct, color), ...]
        for idx, s in enumerate(steps):
            color = f"C{idx+1}"
            g = gen_by_step[s].get(key, np.array([]))
            g_use = _finite(g); g_use = g_use[(g_use >= lo) & (g_use <= hi)]
            ax_top.hist(g_use, bins=bins, range=(lo, hi), density=True,
                        histtype="step", linewidth=1.5, label=f"Step {s}", color=color)

            g_prob, g_counts, _ = hist_prob(g, bins, lo, hi)
            mask = r_counts >= ratio_min_count
            frac = np.full_like(r_prob, np.nan)
            frac[mask] = (g_prob[mask] - r_prob[mask]) / np.maximum(r_prob[mask], 1e-12)
            ax_bot.plot(centers[mask], frac[mask], linewidth=1.2, color=color)

            # track % of valid bins within ±10%
            frac_valid = frac[mask & np.isfinite(frac)]
            if frac_valid.size:
                pct_in = 100.0 * np.mean(np.abs(frac_valid) <= 0.10)
                within_bounds_pcts.append((s, pct_in, color))

        ax_top.set_title(key, pad=6, fontsize=10)
        ax_top.set_ylabel("Density")
        if logy:
            ax_top.set_yscale("log")
        if col == 0:
            ax_top.legend(loc="upper right", ncol=1, fontsize=7)

        ax_bot.axhline(0.0, linewidth=0.8, color="black")
        ax_bot.axhline( 0.1, linestyle="--", linewidth=0.8, alpha=0.6, color="grey")
        ax_bot.axhline(-0.1, linestyle="--", linewidth=0.8, alpha=0.6, color="grey")
        ax_bot.set_ylabel("Frac Diff", fontsize=8)
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_ylim(-1.0, 1.0)

        # Annotate % within ±10% — stacked in top-right corner of frac-diff subplot
        for rank, (s, pct, color) in enumerate(within_bounds_pcts):
            ax_bot.text(
                0.98, 0.97 - rank * 0.20,
                f"S{s}: {pct:.0f}% within ±10%",
                transform=ax_bot.transAxes,
                fontsize=6, color=color, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.6),
            )

    savefig(fig, outpath)


# ──────────────────────────────────────────────
# Multiplicity + corner (unchanged logic, speed tweaks)
# ──────────────────────────────────────────────
def plot_multiplicity(real_mult, gen_mult, outpath, species_name, step, bins=60):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    r = _finite(real_mult); g = _finite(gen_mult)
    if not r.size or not g.size:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off(); savefig(fig, outpath); return

    lo = float(min(r.min(), g.min())); hi = float(max(r.max(), g.max()))
    ax.hist(r, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Simulated")
    ax.hist(g, bins=bins, range=(lo, hi), density=True, histtype="step",
            linewidth=1.8, label=f"Generated (Step {step})")
    ax.set_title(f"Multiplicity: {species_name}", pad=10)
    ax.set_xlabel(f"N({species_name}) per event"); ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.95,
            f"Simulated: mean={r.mean():.2f}, std={r.std():.2f}\n"
            f"Generated: mean={g.mean():.2f}, std={g.std():.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#bbb", alpha=0.95))
    savefig(fig, outpath)


def corner_overlay(real_dict, gen_dict, keys, labels, outpath, title="",
                   bins=40, q_lo=0.01, q_hi=0.99, max_points=30000, seed=123):
    if not HAVE_CORNER: return
    R = np.stack([_finite(real_dict[k]) for k in keys], axis=1)
    G = np.stack([_finite(gen_dict[k]) for k in keys], axis=1)
    R = R[np.all(np.isfinite(R), axis=1)]; G = G[np.all(np.isfinite(G), axis=1)]
    if not R.shape[0] or not G.shape[0]: return

    rng = np.random.default_rng(seed)
    if R.shape[0] > max_points: R = R[rng.choice(R.shape[0], max_points, replace=False)]
    if G.shape[0] > max_points: G = G[rng.choice(G.shape[0], max_points, replace=False)]

    ranges = [(float(np.quantile(np.concatenate([R[:, d], G[:, d]]), q_lo)),
               float(np.quantile(np.concatenate([R[:, d], G[:, d]]), q_hi)))
              for d in range(len(keys))]

    fig = _corner.corner(R, labels=labels, range=ranges, bins=bins, smooth=1.0,
                         plot_density=True, plot_contours=True, fill_contours=True,
                         levels=(0.68, 0.95), color="C0", label_kwargs={"fontsize": 11})
    _corner.corner(G, fig=fig, range=ranges, bins=bins, smooth=1.0,
                   plot_density=False, plot_contours=True, fill_contours=False,
                   levels=(0.68, 0.95), color="C1")
    if title: fig.suptitle(title, y=1.02)
    savefig(fig, outpath)


# ──────────────────────────────────────────────
# Metrics vs steps
# ──────────────────────────────────────────────
def compute_metrics_vs_step(real_arr, gen_by_step, steps, key, bins=120, frac_range=0.80):
    gens = [gen_by_step[s] for s in steps]
    rng = choose_range(real_arr, gens, key=key, frac_range=frac_range)
    if rng is None: return None
    lo, hi = rng

    kl_list, w1_list = [], []
    for s in steps:
        g = gen_by_step[s]
        r_ = _finite(real_arr); r_ = r_[(r_ >= lo) & (r_ <= hi)]
        g_ = _finite(g);        g_ = g_[(g_ >= lo) & (g_ <= hi)]
        if not r_.size or not g_.size:
            kl_list.append(np.nan); w1_list.append(np.nan); continue
        rc, _ = np.histogram(r_, bins=bins, range=(lo, hi))
        gc, _ = np.histogram(g_, bins=bins, range=(lo, hi))
        kl_list.append(kl_divergence_from_counts(rc, gc))
        w1_list.append(wasserstein_1d(r_, g_))

    return {"lo": lo, "hi": hi, "steps": np.array(steps, int),
            "kl": np.array(kl_list), "w1": np.array(w1_list)}


def plot_metrics_grid(metrics_dict, outpath, title, ylog_kl=True):
    vars_ = list(metrics_dict.keys())
    if not vars_: return
    n = len(vars_)
    fig, axes = plt.subplots(n, 2, figsize=(10.0, max(3.0, 1.2*n)), constrained_layout=True)
    if n == 1: axes = axes.reshape(1, 2)
    for i, v in enumerate(vars_):
        m = metrics_dict[v]; x = m["steps"]
        for j, (metric, label) in enumerate([("kl", "KL ↓"), ("w1", "W1 ↓")]):
            ax = axes[i, j]
            ax.plot(x, m[metric], marker="o", linewidth=1.6)
            ax.set_ylabel(v, fontsize=8); ax.set_xlabel("Steps"); ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.2)
            if j == 0 and ylog_kl: ax.set_yscale("log")
    fig.suptitle(title, y=1.02)
    savefig(fig, outpath)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    setup_style()

    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path",       default=None, help="Not needed if --cache_dir is set")
    parser.add_argument("--gen_dir",         default=None, help="Not needed if --cache_dir is set")
    parser.add_argument("--steps",           type=int, nargs="+", default=[25, 50, 100, 150, 200])
    parser.add_argument("--outdir",          required=True)
    parser.add_argument("--step_for_mult",   type=int, default=200)
    parser.add_argument("--step_for_corner", type=int, default=200)
    parser.add_argument("--me",              type=float, default=0.00051099895069)
    parser.add_argument("--bins",            type=int,   default=120)
    parser.add_argument("--ratio_min_count", type=int,   default=10)
    parser.add_argument("--no_metric_curves", action="store_true")
    parser.add_argument("--workers",         type=int, default=None,
                        help="Parallel workers for loading/extraction (default: cpu_count)")
    parser.add_argument("--cache_dir",       type=str, default=None,
                        help="Path to preprocessed cache from preprocess_events.py. "
                             "If set, skips all raw loading and extraction — much faster.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    species_list = [
        {"name": "e−",  "pdgs": [11],   "tag": "eminus"},
        {"name": "e+",  "pdgs": [-11],  "tag": "eplus"},
        {"name": "all", "pdgs": None,   "tag": "all"},
    ]

    real_by_species     = {}
    gen_by_species_step = {sp["tag"]: {} for sp in species_list}

    # ── Fast path: load from preprocessed cache ──────────────────────────
    if args.cache_dir:
        print(f"Loading from cache: {args.cache_dir}")
        for sp in species_list:
            tag = sp["tag"]
            fpath = os.path.join(args.cache_dir, f"real_{tag}.npy")
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Cache file missing: {fpath}\n"
                    f"Run preprocess_events.py first with --cache_dir {args.cache_dir}"
                )
            real_by_species[tag] = np.load(fpath, allow_pickle=True).item()
            print(f"  Loaded real/{tag}  ({real_by_species[tag]['n_particles_kept']:,} particles)")

            for s in args.steps:
                fpath = os.path.join(args.cache_dir, f"gen_{s}steps_{tag}.npy")
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Cache file missing: {fpath}")
                gen_by_species_step[tag][s] = np.load(fpath, allow_pickle=True).item()
            print(f"  Loaded gen/{tag}  steps={args.steps}")

        print("Cache loaded. Generating plots …")

    # ── Slow path: extract from raw files on the fly ─────────────────────
    else:
        if not args.real_path or not args.gen_dir:
            raise ValueError("Either --cache_dir OR both --real_path and --gen_dir are required.")

        tasks = []
        for sp in species_list:
            tasks.append((args.real_path, sp["pdgs"], args.me, "real", sp["tag"]))

        for s in args.steps:
            p = os.path.join(args.gen_dir, f"generated_events_{s}steps.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing: {p}")
            for sp in species_list:
                tasks.append((p, sp["pdgs"], args.me, s, sp["tag"]))

        print(f"Running {len(tasks)} extraction tasks "
              f"({'parallel' if (args.workers or 1) > 1 else 'serial'}) …")
        print("  Tip: run preprocess_events.py once and use --cache_dir for much faster startup.")

        n_workers = args.workers if args.workers is not None else min(len(tasks), os.cpu_count() or 1)

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_extract_task, t): t for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    tag, step, sp_dict = fut.result()
                    if step == "real":
                        real_by_species[tag] = sp_dict
                    else:
                        gen_by_species_step[tag][step] = sp_dict
                    done += 1
                    if done % 5 == 0 or done == len(tasks):
                        print(f"  extracted {done}/{len(tasks)}")
        else:
            for i, t in enumerate(tasks):
                tag, step, sp_dict = _extract_task(t)
                if step == "real":
                    real_by_species[tag] = sp_dict
                else:
                    gen_by_species_step[tag][step] = sp_dict
                if (i+1) % 5 == 0 or (i+1) == len(tasks):
                    print(f"  extracted {i+1}/{len(tasks)}")

        print("Extraction complete.  Generating plots …")

    # ── Panel plots ──────────────────────────────────────────────────────
    for sp in species_list:
        tag  = sp["tag"]
        name = sp["name"]
        real_sp    = real_by_species[tag]
        gen_by_step = gen_by_species_step[tag]   # {step: sp_dict}

        for group_tag in SPECIES_PANELS.get(tag, []):
            group_title, vars_info = PANEL_GROUPS[group_tag]

            # For "all" species, swap E_abs → E_signed in the energy panel
            if group_tag == "energy" and tag == "all":
                vars_info_use = [(k.replace("E_abs", "E_signed"), xl, fr, ly)
                                 for k, xl, fr, ly in vars_info]
            else:
                vars_info_use = vars_info

            outpath = os.path.join(args.outdir, f"{group_tag}_{tag}_panel.png")
            plot_panel_group(
                group_title=group_title,
                vars_info=vars_info_use,
                real_sp=real_sp,
                gen_by_step={s: gen_by_step[s] for s in args.steps},
                steps=args.steps,
                outpath=outpath,
                species_name=name,
                bins=args.bins,
                ratio_min_count=args.ratio_min_count,
            )
            print(f"  → {outpath}")

    # ── Metric curves ────────────────────────────────────────────────────
    if not args.no_metric_curves:
        for sp in species_list:
            tag  = sp["tag"]
            name = sp["name"]
            real_sp    = real_by_species[tag]
            gen_by_step = gen_by_species_step[tag]

            all_vars = []
            for group_tag in SPECIES_PANELS.get(tag, []):
                _, vars_info = PANEL_GROUPS[group_tag]
                for key, xlabel, frac_range, logy in vars_info:
                    if key not in [v[0] for v in all_vars]:
                        all_vars.append((key, xlabel, frac_range))

            metrics_by_var = {}
            for key, xlabel, frac_range in all_vars:
                real_arr = real_sp.get(key, np.array([]))
                g_by_s   = {s: gen_by_step[s].get(key, np.array([])) for s in args.steps}
                if _finite(real_arr).size == 0: continue
                if all(_finite(g_by_s[s]).size == 0 for s in args.steps): continue

                m = compute_metrics_vs_step(real_arr, g_by_s, args.steps, key,
                                            bins=args.bins, frac_range=frac_range)
                if m is not None:
                    metrics_by_var[key] = m

            if metrics_by_var:
                npz_path = os.path.join(args.outdir, f"metrics_vs_steps_{tag}.npz")
                pack = {}
                for k, m in metrics_by_var.items():
                    pack[f"{k}_steps"] = m["steps"]
                    pack[f"{k}_kl"]    = m["kl"]
                    pack[f"{k}_w1"]    = m["w1"]
                    pack[f"{k}_lohi"]  = np.array([m["lo"], m["hi"]])
                np.savez(npz_path, **pack)
                print(f"  Saved metrics → {npz_path}")

                plot_metrics_grid(
                    metrics_by_var,
                    outpath=os.path.join(args.outdir, f"metrics_curves_{tag}.png"),
                    title=f"Metric improvement vs diffusion steps — {name}",
                )
                print(f"  → metrics_curves_{tag}.png")

    # ── Multiplicity ─────────────────────────────────────────────────────
    s_mult = args.step_for_mult
    for sp in species_list:
        tag = sp["tag"]
        if s_mult in gen_by_species_step[tag]:
            outpath = os.path.join(args.outdir, f"multiplicity_{tag}_step{s_mult}.png")
            plot_multiplicity(
                real_by_species[tag]["mult"],
                gen_by_species_step[tag][s_mult]["mult"],
                outpath, sp["name"], step=s_mult,
            )
            print(f"  → {outpath}")

    # ── Corner plots ─────────────────────────────────────────────────────
    s_corner = args.step_for_corner
    for sp in species_list:
        tag = sp["tag"]
        if not HAVE_CORNER or s_corner not in gen_by_species_step[tag]:
            continue
        real_sp = real_by_species[tag]
        gen_sp  = gen_by_species_step[tag][s_corner]

        corner_sets = [
            (["px","py","pz"],           ["p_x [GeV]","p_y [GeV]","p_z [GeV]"],        "p_xyz"),
            (["betax","betay","betaz"],  [r"$\beta_x$",r"$\beta_y$",r"$\beta_z$"],      "beta_xyz"),
        ]
        if real_sp["x"].size and gen_sp["x"].size:
            corner_sets.append((["x","y","z"], ["x [nm]","y [nm]","z [nm]"], "xyz"))

        for keys, labels, tag2 in corner_sets:
            if any(real_sp[k].size == 0 or gen_sp[k].size == 0 for k in keys):
                continue
            outpath = os.path.join(args.outdir, f"corner_{tag2}_{tag}_step{s_corner}.png")
            corner_overlay(real_sp, gen_sp, keys, labels, outpath,
                           title=f"Corner (Step {s_corner}): {tag2} — {sp['name']}")
            print(f"  → {outpath}")

    print(f"\nAll done.  Plots in: {args.outdir}")


if __name__ == "__main__":
    main()
