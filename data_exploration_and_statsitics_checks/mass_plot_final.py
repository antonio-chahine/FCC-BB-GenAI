import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor":    "white",
    "savefig.facecolor":   "white",
    "axes.facecolor":      "white",
    "axes.titlesize":      15,
    "axes.labelsize":      13,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     11,
    "legend.frameon":      True,
    "legend.facecolor":    "white",
    "legend.edgecolor":    "#cccccc",
    "axes.edgecolor":      "black",
    "axes.linewidth":      1.2,
    "axes.grid":           False,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    5,
    "ytick.major.size":    5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

C_SIM  = "#2166AC"
C_GEN  = "#D6604D"
C_LINE = "#1a1a1a"
C_MREF = "#2ca02c"

def savefig(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

# ── Load data ──────────────────────────────────────────────────────────────────
simulated_events = np.load(
    "/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy",
    allow_pickle=True)
generated_events = np.load(
    "/work/submit/anton100/msci-project/FCC-BB-GenAI/new_10/generated_events.npy",
    allow_pickle=True)

m_e = 0.000511   # GeV

def extract_masses(events, is_generated=False):
    all_m, all_m2, all_beta2 = [], [], []
    tot = 0

    for ev in events:
        ev = np.asarray(ev, dtype=np.float64)
        if ev.ndim != 2:
            continue
        if is_generated and ev.shape[1] >= 5:
            E    = np.abs(ev[:, 1])
            beta = ev[:, 2:5]
        elif not is_generated and ev.shape[1] >= 4:
            E    = np.abs(ev[:, 0])
            beta = ev[:, 1:4]
        else:
            continue
        beta2 = (beta**2).sum(axis=1)
        m2    = E**2 - E**2 * beta2
        tot  += len(m2)
        m2c   = np.clip(m2, 0.0, None)
        all_m.append(np.sqrt(m2c))
        all_m2.append(m2)
        all_beta2.append(beta2)

    all_m     = np.concatenate(all_m)
    all_m2    = np.concatenate(all_m2)
    all_beta2 = np.concatenate(all_beta2)

    counts, edges = np.histogram(all_m, bins=2000,
                                  range=(0, np.quantile(all_m, 0.999)))
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts)+1])

    stats = dict(
        n_events=len(events), n_particles=tot,
        mean=float(all_m.mean()),
        median=float(np.median(all_m)),
        peak=float(peak),
        std=float(all_m.std()),
    )
    return all_m, all_m2, all_beta2, stats


sim_m, sim_m2, sim_b2, sim_s = extract_masses(simulated_events, is_generated=False)
gen_m, gen_m2, gen_b2, gen_s = extract_masses(generated_events, is_generated=True)


# ── Figure 1: Full distribution (log y) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

hi_q = float(np.quantile(np.concatenate([sim_m, gen_m]), 0.999))
bins_full = np.linspace(0, hi_q, 250)

ax.hist(sim_m, bins=bins_full, density=True, alpha=0.55,
        color=C_SIM, label="Simulated", log=True)
ax.hist(gen_m, bins=bins_full, density=True, histtype="step",
        linewidth=2.0, color=C_GEN, label="Generated", log=True)

ax.axvline(m_e, color=C_MREF, linestyle="--", linewidth=1.6,
           label=r"$m_e$ = 0.510999 MeV")

for s, col in [(sim_s, C_SIM), (gen_s, C_GEN)]:
    ax.axvline(s["peak"], color=col, linestyle=":", linewidth=1.4, alpha=0.8)

annot = (
    f"Peak mass:\n"
    f"  Sim: {sim_s['peak']*1e3:.6f} MeV\n"
    f"  Gen: {gen_s['peak']*1e3:.6f} MeV\n"
    f"  $m_e$:  0.510999 MeV"
)
ax.text(0.97, 0.97, annot, transform=ax.transAxes, va="top", ha="right",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                               edgecolor="#888888", alpha=0.92))

ax.set_xlabel("Mass $m$ [GeV]")
ax.set_ylabel("Density (log scale)")
ax.set_title("Per-particle mass distribution")
ax.legend(loc="upper center")

savefig(fig, "mass_distribution_2.png")
print("Saved mass_distribution.png")


# ── Figure 2: Zoom near electron mass — linear + ratio panel ─────────────────
zoom_lo, zoom_hi = 0.0, 0.002

fig2 = plt.figure(figsize=(9, 7))
gs2  = fig2.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
ax2  = fig2.add_subplot(gs2[0])
ax2r = fig2.add_subplot(gs2[1], sharex=ax2)
ax2.tick_params(labelbottom=False)

bins_zoom = np.linspace(zoom_lo, zoom_hi, 200)
r_cnt, _ = np.histogram(sim_m, bins=bins_zoom, density=False)
g_cnt, _ = np.histogram(gen_m, bins=bins_zoom, density=False)
ctrs = 0.5 * (bins_zoom[:-1] + bins_zoom[1:])

bin_width = bins_zoom[1] - bins_zoom[0]
r_dens = r_cnt / (r_cnt.sum() * bin_width) if r_cnt.sum() > 0 else r_cnt.astype(float)
g_dens = g_cnt / (g_cnt.sum() * bin_width) if g_cnt.sum() > 0 else g_cnt.astype(float)

ax2.bar(ctrs, r_dens, width=bin_width, alpha=0.55, color=C_SIM, label="Simulated")
ax2.step(np.append(bins_zoom[:-1], bins_zoom[-1]),
         np.append(g_dens, g_dens[-1]),
         where="post", linewidth=2.0, color=C_GEN, label="Generated")

ax2.axvline(m_e, color=C_MREF, linestyle="--", linewidth=1.8,
            label=rf"$m_e$ = {m_e*1e3:.3f} MeV")

for s, col in [(sim_s, C_SIM), (gen_s, C_GEN)]:
    ax2.axvline(s["peak"], color=col, linestyle=":", linewidth=1.5, alpha=0.8)

# single clean annotation
annot2 = (
    f"Peak mass:\n"
    f"  Sim: {sim_s['peak']*1e3:.3f} MeV\n"
    f"  Gen: {gen_s['peak']*1e3:.3f} MeV\n"
    f"  $m_e$:  0.511 MeV"
)
ax2.text(0.97, 0.97, annot2, transform=ax2.transAxes, va="top", ha="right",
         fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                edgecolor="#888888", alpha=0.92))

ax2.set_ylabel("Density")
ax2.set_title(r"Mass near $m_e$ — zoom (0–2 MeV)")
ax2.legend(loc="upper left", fontsize=10)
ax2.set_xlim(zoom_lo, zoom_hi)

# ratio panel
mask = r_cnt >= 5
r_p  = r_cnt / max(r_cnt.sum(), 1)
g_p  = g_cnt / max(g_cnt.sum(), 1)
frac = np.where(mask, (g_p - r_p) / np.where(r_p > 0, r_p, np.nan), np.nan)

ax2r.axhline(0, color=C_LINE, linewidth=1.2)
ax2r.axhspan(-0.1, 0.1, color="#AAAAAA", alpha=0.18)
ax2r.plot(ctrs[mask], frac[mask], "o", markersize=3.5, color=C_GEN)
ax2r.set_ylim(-1, 1)
ax2r.set_ylabel("Fract. diff.", fontsize=11)
ax2r.yaxis.set_major_locator(MaxNLocator(nbins=3))

ax2.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*1e3:.1f}"))
ax2r.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*1e3:.1f}"))
ax2r.set_xlabel("Mass [MeV]")

savefig(fig2, "mass_distribution_zoom.png")
print("Saved mass_distribution_zoom.png")
