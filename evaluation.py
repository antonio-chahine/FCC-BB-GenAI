#!/usr/bin/env python3
"""
Evaluate real vs generated MCParticle event samples.

Expected input format (both REAL_PATH and GEN_PATH):
- .npy containing either:
  (A) object array of length N, each element is (K,4+) with columns [pdg, px, py, pz, ...]
  (B) numeric array shaped (N, K, 4+) with [pdg, px, py, pz, ...]

Outputs:
- Multiplicity histograms per species
- 2-panel distributions: top = density overlay, bottom = (gen-real)/real using COUNTS with min-count cut
- KL divergence (hist-based) and Wasserstein distance (1D)

This rewrite focuses plots around the histogram "bulk" via robust quantile-based ranges
(so outliers don't stretch the x-axis). Optionally clamps data to the plotted window
so the ratio + metrics reflect what you see.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS (edit these)
# ============================================================
REAL_PATH = "mc_gen1.npy"
GEN_PATH  = "/work/submit/anton100/msci-project/FCC-BB-GenAI/mc_gen1_model/generated_events.npy"
OUTDIR    = "eval_plots_mc_gen1_model"

# Optional metadata for titles
TITLE_KMAX  = None
TITLE_STEPS = None

# What to plot
INCLUDE_ALL_SPECIES = False
PHOTON_MULT_LOGY    = False

# Histogram binning
MULT_BINS = 50
MOM_BINS  = 80

# Ratio panel stability
RATIO_MIN_REAL_COUNT = 10   # bins with fewer real counts are hidden in ratio panel
RATIO_YLIM = (-1.0, 1.0)

# Log-momentum settings (not used in this script, kept for easy extension)
LOGP_EPS = 1e-12

# Save controls
SAVE_DPI        = 200
SAVE_PAD_INCHES = 0.06

# ============================================================
# NEW: "focus around the histogram" controls
# ============================================================
# Use quantiles to define x-range so extreme tails/outliers don't dominate.
# Examples:
#   0.005/0.995 -> central 99%
#   0.01 /0.99  -> central 98% (tighter)
RANGE_QLO = 0.005
RANGE_QHI = 0.995

# If True: drop values outside [lo,hi] before hist/counts/ratio/metrics.
# If False: still plot using [lo,hi], but counts/metrics include out-of-range values
# (less consistent with what you see).
CLAMP_TO_PLOTTED_RANGE = True


# ============================================================
# Optional SciPy for Wasserstein
# ============================================================
try:
    from scipy.stats import wasserstein_distance
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ============================================================
# Styling
# ============================================================
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


def savefig(fig, path):
    fig.savefig(
        path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# ============================================================
# Loading
# ============================================================
def load_events(path: str):
    arr = np.load(path, allow_pickle=True)

    # Case A: object array (N,) each element an event
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return list(arr)

    # Case B: numeric array (N,K,4+) or (N,K,>=4)
    if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] >= 4:
        return [arr[i] for i in range(arr.shape[0])]

    raise ValueError(
        f"Unrecognized format in {path}\n"
        f"Got type={type(arr)} shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}\n"
        "Expected object-array of events or numeric array shaped (N,K,>=4)."
    )


def sanitize_event(ev):
    """Return (pdg, px, py, pz) as 1D arrays. Empty -> all empty arrays."""
    if ev is None:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    ev = np.asarray(ev)
    if ev.size == 0:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))

    if ev.ndim != 2 or ev.shape[1] < 4:
        raise ValueError(f"Each event must be (K,4+) with [pdg, px, py, pz]. Got shape={ev.shape}")

    pdg = ev[:, 0].astype(np.int64, copy=False)
    px  = ev[:, 1].astype(np.float64, copy=False)
    py  = ev[:, 2].astype(np.float64, copy=False)
    pz  = ev[:, 3].astype(np.float64, copy=False)
    return pdg, px, py, pz


def extract_species(events, pdgs=None):
    """
    For given PDGs (list) or pdgs=None for all:
    Returns dict with:
      mult: (Nevents,) multiplicity per event
      px,py,pz,p: flattened arrays over selected particles
    """
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
        "mult": mult,
        "px": px_all,
        "py": py_all,
        "pz": pz_all,
        "p":  p_all,
        "pt": pt_all,

    }


# ============================================================
# Ranges: "focus around histogram"
# ============================================================
def robust_range(x, y, q_lo=RANGE_QLO, q_hi=RANGE_QHI, min_width=1e-12):
    """
    Range spanning both arrays based on quantiles (ignores extreme tails/outliers).
    Returns (lo, hi) or None if degenerate / empty.
    """
    vals = []
    for arr in (x, y):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return None

    z = np.concatenate(vals)
    lo = float(np.quantile(z, q_lo))
    hi = float(np.quantile(z, q_hi))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi - lo < min_width:
        # fallback: use min/max finite range
        lo2, hi2 = finite_range(x, y) or (None, None)
        if lo2 is None:
            return None
        return float(lo2), float(hi2)
    return lo, hi


def finite_range(x, y):
    """Range spanning both arrays (finite values)."""
    lo = np.inf
    hi = -np.inf
    for arr in (x, y):
        if len(arr) == 0:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        lo = min(lo, float(np.min(arr)))
        hi = max(hi, float(np.max(arr)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    return lo, hi


def clamp_to_range(arr, lo, hi):
    """Keep only values within [lo, hi]."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= lo) & (arr <= hi)]


# ============================================================
# Metrics
# ============================================================
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

    # fallback approximate Wasserstein-1 by quantile interpolation
    xs = np.sort(np.asarray(x, dtype=np.float64))
    ys = np.sort(np.asarray(y, dtype=np.float64))
    q = np.linspace(0.0, 1.0, 400)
    xq = np.interp(q, np.linspace(0.0, 1.0, len(xs)), xs)
    yq = np.interp(q, np.linspace(0.0, 1.0, len(ys)), ys)
    return float(np.mean(np.abs(xq - yq)))


# ============================================================
# Plot helpers
# ============================================================
def header_text(n_real, n_gen):
    bits = [f"real={n_real}", f"gen={n_gen}"]
    if TITLE_KMAX is not None:
        bits.append(f"Kmax={TITLE_KMAX}")
    if TITLE_STEPS is not None:
        bits.append(f"steps={TITLE_STEPS}")
    return " | ".join(bits)


def plot_multiplicity(real_mult, gen_mult, outpath, species_name, n_real, n_gen, logy=False):
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.8), constrained_layout=True)

    # multiplicities typically don't have crazy outliers,
    # but we still apply robust range for consistency
    rng = robust_range(real_mult, gen_mult, q_lo=0.0, q_hi=1.0)
    if rng is None:
        ax.text(0.5, 0.5, "No/degenerate data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        savefig(fig, outpath)
        return

    lo, hi = rng
    if CLAMP_TO_PLOTTED_RANGE:
        real_mult = clamp_to_range(real_mult, lo, hi)
        gen_mult  = clamp_to_range(gen_mult,  lo, hi)

    ax.hist(real_mult, bins=MULT_BINS, range=(lo, hi), density=True, alpha=0.55, label="Real")
    ax.hist(gen_mult,  bins=MULT_BINS, range=(lo, hi), density=True, histtype="step",
            linewidth=1.8, label="Generated")

    ax.set_title(f"Multiplicity | {header_text(n_real, n_gen)}", pad=10)
    ax.set_xlabel(f"Multiplicity N({species_name}) per event")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    if logy:
        ax.set_yscale("log")

    savefig(fig, outpath)


def two_panel_dist(real, gen, outpath, xlabel, title, bins):
    """
    Top: density overlay (real filled, gen step)
    Bottom: (gen-real)/real using COUNTS with min-count cut + error bars ~ 1/sqrt(N_real)

    Uses robust quantile x-range to "focus around the histogram".
    Optionally clamps to plotted window so metrics + ratio match what you see.
    """
    fig = plt.figure(figsize=(7.6, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.tick_params(labelbottom=False)

    rng = robust_range(real, gen, q_lo=RANGE_QLO, q_hi=RANGE_QHI)
    if rng is None:
        ax_top.text(0.5, 0.5, "No/degenerate data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()
        savefig(fig, outpath)
        return

    lo, hi = rng

    real_use = np.asarray(real, dtype=np.float64)
    gen_use  = np.asarray(gen,  dtype=np.float64)

    if CLAMP_TO_PLOTTED_RANGE:
        real_use = clamp_to_range(real_use, lo, hi)
        gen_use  = clamp_to_range(gen_use,  lo, hi)
    else:
        # still drop non-finite
        real_use = real_use[np.isfinite(real_use)]
        gen_use  = gen_use[np.isfinite(gen_use)]

    if real_use.size == 0 or gen_use.size == 0 or not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        ax_top.text(0.5, 0.5, "No/degenerate data", ha="center", va="center", transform=ax_top.transAxes)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()
        savefig(fig, outpath)
        return

    # counts for metrics + ratio (within plotted window)
    r_counts, edges = np.histogram(real_use, bins=bins, range=(lo, hi), density=False)
    g_counts, _     = np.histogram(gen_use,  bins=bins, range=(lo, hi), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # top density overlay
    ax_top.hist(real_use, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="Real Data")
    ax_top.hist(gen_use,  bins=bins, range=(lo, hi), density=True, histtype="step",
                linewidth=1.8, label="Generated Data")

    ax_top.set_title(title, pad=10)
    ax_top.set_ylabel("Density")
    ax_top.legend(loc="upper left")

    # metrics (computed on the same window you're plotting)
    kl = kl_divergence_from_counts(r_counts, g_counts)
    wd = wasserstein_1d(real_use, gen_use)

    ax_top.text(
        0.98, 0.95,
        f"KL: {kl:.4f}\nW1: {wd:.4f}",
        transform=ax_top.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#bbbbbb", alpha=0.95),
    )

       # --- normalise counts so ratio compares SHAPES, not sample size ---
    r_sum = r_counts.sum()
    g_sum = g_counts.sum()

    # probabilities per bin
    r_prob = r_counts / max(r_sum, 1)
    g_prob = g_counts / max(g_sum, 1)

    minc = int(RATIO_MIN_REAL_COUNT)
    mask = r_counts >= minc  # still use real *counts* for stability

    frac = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac[mask] = (g_prob[mask] - r_prob[mask]) / r_prob[mask]

    # uncertainty (rough): 1/sqrt(N_real_bin) still reasonable as a stability indicator
    frac_err = np.full_like(r_prob, np.nan, dtype=np.float64)
    frac_err[mask] = 1.0 / np.sqrt(r_counts[mask])

    ax_bot.axhline(0.0, linewidth=1.0)
    ax_bot.axhspan(
        -0.1, 0.1,
        color="gray",
        alpha=0.15,
        zorder=0
    )


    ax_bot.errorbar(
        centers[mask], frac[mask], yerr=frac_err[mask],
        fmt="o", markersize=3, linewidth=1.0, capsize=0
    )

    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("Frac. diff.")
    ax_bot.set_ylim(*RATIO_YLIM)

    ax_bot.text(
        0.98, 0.85,
        "±10% band",
        transform=ax_bot.transAxes,
        ha="right",
        va="center",
        fontsize=10
    )



    savefig(fig, outpath)


# ============================================================
# Main
# ============================================================
def main():
    setup_style()
    os.makedirs(OUTDIR, exist_ok=True)

    real_events = load_events(REAL_PATH)
    gen_events  = load_events(GEN_PATH)

    n_real = len(real_events)
    n_gen  = len(gen_events)

    species_list = [
        {"name": "e±", "pdgs": [11, -11], "tag": "eplus_eminus"},
        {"name": "e−", "pdgs": [11],      "tag": "eminus"},
        {"name": "e+", "pdgs": [-11],     "tag": "eplus"},
    ]
    if INCLUDE_ALL_SPECIES:
        species_list.append({"name": "all", "pdgs": None, "tag": "all"})

    for sp in species_list:
        real_sp = extract_species(real_events, sp["pdgs"])
        gen_sp  = extract_species(gen_events,  sp["pdgs"])

        # multiplicity
        mult_logy = (sp["tag"] == "photon" and PHOTON_MULT_LOGY)
        plot_multiplicity(
            real_sp["mult"],
            gen_sp["mult"],
            outpath=os.path.join(OUTDIR, f"multiplicity_{sp['tag']}.png"),
            species_name=sp["name"],
            n_real=n_real,
            n_gen=n_gen,
            logy=mult_logy,
        )

        # px, py, pz, pt
        for key in ["px", "py", "pz", "pt"]:
            two_panel_dist(
                real_sp[key],
                gen_sp[key],
                outpath=os.path.join(OUTDIR, f"{key}_{sp['tag']}.png"),
                xlabel=("p_T" if key == "pt" else key),
                title=f"Comparison of {key} | {header_text(n_real, n_gen)}",
                bins=MOM_BINS,
            )

    print(f"Saved plots to: {OUTDIR}/")
    print(f"Focused range quantiles: [{RANGE_QLO}, {RANGE_QHI}]")
    print(f"Clamp to plotted range: {CLAMP_TO_PLOTTED_RANGE}")


if __name__ == "__main__":
    main()
