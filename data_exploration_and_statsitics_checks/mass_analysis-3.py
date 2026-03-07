import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

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

def extract_masses(events, label="", is_generated=False):
    all_m, all_m2, all_beta2 = [], [], []
    neg_m2 = 0
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
        p2    = E**2 * beta2
        m2    = E**2 - p2

        tot += len(m2)
        neg_m2 += int(np.sum(m2 < -1e-6))

        m2c = np.clip(m2, 0.0, None)
        all_m.append(np.sqrt(m2c))
        all_m2.append(m2)
        all_beta2.append(beta2)

    all_m    = np.concatenate(all_m)
    all_m2   = np.concatenate(all_m2)
    all_beta2= np.concatenate(all_beta2)

    counts, edges = np.histogram(all_m, bins=2000,
                                  range=(0, np.quantile(all_m, 0.999)))
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts)+1])

    stats = dict(
        n_events=len(events), n_particles=tot,
        neg_m2=neg_m2,
        mean=float(all_m.mean()),
        median=float(np.median(all_m)),
        peak=float(peak),
        std=float(all_m.std()),
        p1=float(np.percentile(all_m, 1)),
        p99=float(np.percentile(all_m, 99)),
        rel_err_std=float(((all_m - m_e)/m_e).std()),
    )
    return all_m, all_m2, all_beta2, stats


sim_m, sim_m2, sim_b2, sim_s = extract_masses(simulated_events, "Simulated", is_generated=False)
gen_m, gen_m2, gen_b2, gen_s = extract_masses(generated_events, "Generated", is_generated=True)


# ── Figure 1: Full distribution — no side boxes, single in-plot metric ────────
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

hi_q = float(np.quantile(np.concatenate([sim_m, gen_m]), 0.999))
bins_full = np.linspace(0, hi_q, 250)

ax.hist(sim_m, bins=bins_full, density=True, alpha=0.55,
        color=C_SIM, label="Simulated", log=True)
ax.hist(gen_m, bins=bins_full, density=True, histtype="step",
        linewidth=2.0, color=C_GEN, label="Generated", log=True)

ax.axvline(m_e, color=C_MREF, linestyle="--", linewidth=1.6,
           label=r"$m_e$ = 0.511 MeV")

# dotted peak lines
for s, col in [(sim_s, C_SIM), (gen_s, C_GEN)]:
    ax.axvline(s["peak"], color=col, linestyle=":", linewidth=1.4, alpha=0.8)

# ── Single in-plot annotation: peak mass ──────────────────────────────────────
annot = (
    f"Peak mass:\n"
    f"  Sim: {sim_s['peak']*1e3:.3f} MeV\n"
    f"  Gen: {gen_s['peak']*1e3:.3f} MeV\n"
    f"  $m_e$:  0.511 MeV"
)
ax.text(0.97, 0.97, annot,
        transform=ax.transAxes, va="top", ha="right", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#888888", alpha=0.92))

ax.set_xlabel("Mass $m$ [GeV]")
ax.set_ylabel("Density (log scale)")
ax.set_title("Per-particle mass distribution")
ax.legend(loc="upper center")

savefig(fig, "/mnt/user-data/outputs/mass_distribution.png")
print("Saved mass_distribution.png")



# ── Figure 2: Zoom near electron mass — linear + ratio panel ─────────────────
zoom_lo, zoom_hi = 0.0, 0.002   # 0 – 2 MeV

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
# manually normalise — avoids numpy RuntimeWarning when empty bins exist
r_dens = r_cnt / (r_cnt.sum() * bin_width) if r_cnt.sum() > 0 else r_cnt.astype(float)
g_dens = g_cnt / (g_cnt.sum() * bin_width) if g_cnt.sum() > 0 else g_cnt.astype(float)
ax2.bar(ctrs, r_dens, width=bin_width, alpha=0.55, color=C_SIM, label="Simulated")
ax2.step(np.append(bins_zoom[:-1], bins_zoom[-1]),
         np.append(g_dens, g_dens[-1]),
         where="post", linewidth=2.0, color=C_GEN, label="Generated")

# electron mass reference + peak markers
ax2.axvline(m_e, color=C_MREF, linestyle="--", linewidth=1.8,
            label=rf"$m_e$ = {m_e*1e3:.3f} MeV")

for s, col, lbl in [(sim_s, C_SIM, "Sim. peak"), (gen_s, C_GEN, "Gen. peak")]:
    ax2.axvline(s["peak"], color=col, linestyle=":", linewidth=1.5,
                label=f"{lbl} = {s['peak']*1e3:.4f} MeV")

# stats box
box_txt = (
    f"Simulated  peak = {sim_s['peak']*1e3:.4f} MeV\n"
    f"           mean = {sim_s['mean']*1e3:.4f} MeV\n"
    f"           std  = {sim_s['std']*1e3:.4f} MeV\n\n"
    f"Generated  peak = {gen_s['peak']*1e3:.4f} MeV\n"
    f"           mean = {gen_s['mean']*1e3:.4f} MeV\n"
    f"           std  = {gen_s['std']*1e3:.4f} MeV\n\n"
    f"$m_e$ (true) = {m_e*1e3:.3f} MeV"
)
ax2.text(0.97, 0.97, box_txt, transform=ax2.transAxes,
         ha="right", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                   edgecolor="#888888", alpha=0.95))

ax2.set_ylabel("Density")
ax2.set_title(r"Mass near $m_e$ — peak zoom (0 – 2 MeV)")
ax2.legend(loc="upper left", fontsize=10)
ax2.set_xlim(zoom_lo, zoom_hi)

# ratio panel
mask = r_cnt >= 5
r_p = r_cnt / max(r_cnt.sum(), 1)
g_p = g_cnt / max(g_cnt.sum(), 1)
frac = np.where(mask, (g_p - r_p) / np.where(r_p > 0, r_p, np.nan), np.nan)

ax2r.axhline(0, color=C_LINE, linewidth=1.2)
ax2r.axhspan(-0.1, 0.1, color="#AAAAAA", alpha=0.18)
ax2r.plot(ctrs[mask], frac[mask], "o", markersize=3.5, color=C_GEN)
ax2r.set_ylim(-1, 1)
ax2r.set_ylabel("Fract. diff.", fontsize=11)
ax2r.set_xlabel("Mass $m$ [GeV]")
ax2r.yaxis.set_major_locator(MaxNLocator(nbins=3))

# x-axis in MeV for readability
ax2.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*1e3:.1f}"))
ax2r.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*1e3:.1f}"))
ax2r.set_xlabel("Mass [MeV]")
ax2.set_xlabel("")

savefig(fig2, "mass_distribution_zoom.png")
print("Saved mass_distribution_zoom.png")


# ── Figure 3: Relative mass error (sim + gen) ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5), constrained_layout=True)

rel_sim = (sim_m - m_e) / m_e
rel_gen = (gen_m - m_e) / m_e

bins_rel = np.linspace(-0.5, 0.5, 200)

ax3.hist(rel_sim, bins=bins_rel, density=True, alpha=0.55,
         color=C_SIM, label=f"Simulated  (σ={rel_sim.std():.4f})")
ax3.hist(rel_gen, bins=bins_rel, density=True, histtype="step",
         linewidth=2.0, color=C_GEN,
         label=f"Generated  (σ={rel_gen.std():.4f})")

ax3.axvline(0, color=C_MREF, linestyle="--", linewidth=1.6, label="True $m_e$")

# Gaussian fit overlay (simulated)
mu_s, sig_s = float(rel_sim.mean()), float(rel_sim.std())
mu_g, sig_g = float(rel_gen.mean()), float(rel_gen.std())
x_fit = np.linspace(-0.5, 0.5, 500)
ax3.plot(x_fit, norm.pdf(x_fit, mu_s, sig_s), color=C_SIM,
         linewidth=1.8, linestyle="-.", alpha=0.8, label="Gaussian fit (sim)")
ax3.plot(x_fit, norm.pdf(x_fit, mu_g, sig_g), color=C_GEN,
         linewidth=1.8, linestyle="-.", alpha=0.8, label="Gaussian fit (gen)")

stats_box = (
    f"Simulated : μ={mu_s:+.4f},  σ={sig_s:.4f}\n"
    f"Generated : μ={mu_g:+.4f},  σ={sig_g:.4f}"
)
ax3.text(0.02, 0.97, stats_box, transform=ax3.transAxes,
         va="top", ha="left", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                   edgecolor="#888888", alpha=0.95))

ax3.set_xlabel(r"$(m - m_e)\,/\,m_e$")
ax3.set_ylabel("Density")
ax3.set_title("Relative mass error")
ax3.set_xlim(-0.5, 0.5)
ax3.legend(loc="upper right")

savefig(fig3, "mass_relative_error.png")
print("Saved mass_relative_error.png")
