#!/usr/bin/env python3
"""
Combined evaluation script:
  - Beta / energy / position / momentum plots
  - Per-particle mass distribution (3×1 vertical, log scale, E²−p² formula)
  - Charge asymmetry per event
  - Closest e+/e- pair opening angles
  - Per-event summed momentum components

Distribution plots: THREE zoom levels arranged VERTICALLY (3 rows × 1 col)
  row 0 — full range
  row 1 — slight zoom (~5 % of full width around peak)
  row 2 — tight zoom  (~0.5 % of full width around peak)
Each zoom level has its own histogram + fractional-difference ratio panel.

Separate simulated-only multiplicity plot also produced.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# ── Palette (matches photo style) ────────────────────────────────────────────
C_REAL  = "grey"       # simulated: grey filled
C_GEN   = "blue"       # generated: blue step
C_BAND  = "#AAAAAA"    # +-10% ratio band
C_ZERO  = "#333333"    # ratio zero line
C_MREF  = "red"        # reference line (m_e etc.)
C_GLINE = "black"      # generated mean/annotation lines

# ── Global style (matches photo) ──────────────────────────────────────────────
def setup_style():
    matplotlib.rcParams.update({
        "figure.facecolor":    "white",
        "savefig.facecolor":   "white",
        "axes.facecolor":      "white",
        "axes.titlesize":      15,
        "axes.labelsize":      14,
        "xtick.labelsize":     14,
        "ytick.labelsize":     14,
        "legend.fontsize":     12,
        "legend.frameon":      True,
        "legend.facecolor":    "white",
        "legend.edgecolor":    "#cccccc",
        "legend.framealpha":   0.95,
        "axes.edgecolor":      "black",
        "axes.linewidth":      1.0,
        "axes.grid":           False,
        "lines.linewidth":     2.0,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.titlepad":       8,
        "axes.labelpad":       4,
    })


def savefig(fig, path, dpi=200):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.15, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  {path}")


# ── Stats helpers ─────────────────────────────────────────────────────────────
def robust_range(x, y, q_lo=0.005, q_hi=0.995):
    z = np.concatenate([np.asarray(x, dtype=np.float64),
                        np.asarray(y, dtype=np.float64)])
    z = z[np.isfinite(z)]
    if z.size < 2:
        return None
    lo, hi = float(np.quantile(z, q_lo)), float(np.quantile(z, q_hi))
    return (lo, hi) if (np.isfinite(lo) and np.isfinite(hi) and lo < hi) else None


def clamp_to_range(arr, lo, hi):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr[(arr >= lo) & (arr <= hi)]


def _ratio_panel(ax, centers, r_counts, g_counts, ratio_min_count, show_ylabel):
    mask   = r_counts >= ratio_min_count
    r_prob = r_counts / max(r_counts.sum(), 1)
    g_prob = g_counts / max(g_counts.sum(), 1)
    frac   = np.where(mask,
                      (g_prob - r_prob) / np.where(r_prob > 0, r_prob, np.nan),
                      np.nan)
    ax.axhline(0.0, color=C_ZERO, linewidth=1.2)
    ax.axhspan(-0.10, 0.10, color=C_BAND, alpha=0.18, zorder=0)
    ax.plot(centers[mask], frac[mask], color=C_GEN, linewidth=1.5, zorder=2)
    ax.set_ylim(-1.0, 1.0)
    if show_ylabel:
        ax.set_ylabel("Fract.\ndiff.", fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.text(0.98, 0.80, r"$\pm$10% band", transform=ax.transAxes,
            ha="right", va="center", fontsize=9, color="#555555")


def _peak_zoom_range(arr, zoom_frac=0.01):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    counts, edges = np.histogram(arr, bins=600)
    i = int(np.argmax(counts))
    peak_centre = 0.5 * (edges[i] + edges[i + 1])
    full = arr.max() - arr.min()
    half = zoom_frac * full * 0.5
    if half <= 0:
        return None
    return (peak_centre - half, peak_centre + half)


# ── Core: single variable — VERTICAL (full + slight zoom + tight zoom) ────────
def plot_single_three_zoom(real, gen, outpath, xlabel, title, species_name,
                           n_real=0, n_gen=0, bins=80, ratio_min_count=10,
                           fixed_range=None,
                           slight_zoom_frac=0.05,
                           tight_zoom_frac=0.005,
                           tight_zoom_range=None):
    """
    Three zoom levels arranged VERTICALLY (3 rows × 1 col).

    tight_zoom_range : (lo, hi) tuple to explicitly set the tight zoom x-limits.
                       If None, the range is computed automatically from
                       tight_zoom_frac via _peak_zoom_range.
    """
    real = np.asarray(real, dtype=np.float64); real = real[np.isfinite(real)]
    gen  = np.asarray(gen,  dtype=np.float64); gen  = gen[np.isfinite(gen)]
    if real.size == 0 or gen.size == 0:
        return

    if fixed_range is not None:
        lo, hi = fixed_range
    else:
        rng = robust_range(real, gen)
        if rng is None:
            return
        lo, hi = rng

    real_full = clamp_to_range(real, lo, hi)
    gen_full  = clamp_to_range(gen,  lo, hi)
    # Use unclamped data for zoom range so zoom_frac is relative to full data extent
    combined_full = np.concatenate([real[np.isfinite(real)], gen[np.isfinite(gen)]])

    slight_rng = _peak_zoom_range(combined_full, zoom_frac=slight_zoom_frac)

    # tight zoom: explicit range takes priority over auto-computed one
    if tight_zoom_range is not None:
        tight_rng = tight_zoom_range
    else:
        tight_rng = _peak_zoom_range(combined_full, zoom_frac=tight_zoom_frac)

    rows = [("Full range",   lo, hi, real_full, gen_full)]
    for rng_z, label_z in [(slight_rng, "Slight zoom"), (tight_rng, "Tight zoom")]:
        if rng_z is not None:
            zlo, zhi = rng_z
            r_z = clamp_to_range(real, zlo, zhi)
            g_z = clamp_to_range(gen,  zlo, zhi)
            if r_z.size > 0 and g_z.size > 0:
                rows.append((label_z, zlo, zhi, r_z, g_z))

    n_rows = len(rows)
    nr = n_real if n_real else real_full.size
    ng = n_gen  if n_gen  else gen_full.size

    fig = plt.figure(figsize=(7.0 * n_rows, 5.5))
    fig.suptitle(
        f"{title}  —  {species_name}  |  "
        f"Simulated: {nr:,}  ·  Generated: {ng:,}",
        fontsize=15, y=1.03,
    )

    outer = gridspec.GridSpec(1, n_rows, figure=fig,
                              left=0.06, right=0.98,
                              top=0.92, bottom=0.12,
                              wspace=0.28)

    for col_idx, (sub_title, clo, chi, r_data, g_data) in enumerate(rows):
        edges = np.linspace(clo, chi, bins + 1)
        r_cnt, _ = np.histogram(r_data, bins=edges)
        g_cnt, _ = np.histogram(g_data, bins=edges)
        ctrs = 0.5 * (edges[:-1] + edges[1:])

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col_idx],
            height_ratios=[3, 1], hspace=0.08)
        ax_h = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax_h)
        ax_h.tick_params(labelbottom=False, bottom=True)

        ax_h.hist(r_data, bins=edges, density=True, alpha=0.6,
                  color=C_REAL,
                  label="Simulated" if col_idx == 0 else "_nolegend_")
        ax_h.hist(g_data, bins=edges, density=True, histtype="step",
                  linewidth=2.2, color=C_GEN,
                  label="Generated" if col_idx == 0 else "_nolegend_")

        ax_h.set_title(sub_title, fontsize=15, pad=5)
        ax_h.set_ylabel("Density" if col_idx == 0 else "", fontsize=15)
        if col_idx == 0:
            ax_h.legend(loc="upper right", fontsize=13)

        _ratio_panel(ax_r, ctrs, r_cnt, g_cnt,
                     ratio_min_count, show_ylabel=(col_idx == 0))
        ax_r.set_xlabel(xlabel, fontsize=13)

    fig.align_ylabels()
    savefig(fig, outpath)


# ── Multiplicity: overlay + simulated-only ────────────────────────────────────
def plot_multiplicity(real_mult, gen_mult, outpath, species_name,
                      n_real, n_gen, bins=50, logy=False, figsize=(8, 5.0)):
    """Overlay plot: Simulated vs Generated."""
    rm = np.asarray(real_mult, dtype=np.float64); rm = rm[np.isfinite(rm)]
    gm = np.asarray(gen_mult,  dtype=np.float64); gm = gm[np.isfinite(gm)]
    rng = robust_range(rm, gm, q_lo=0.0, q_hi=1.0)
    if rng is None:
        return
    lo, hi = rng
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(rm, bins=bins, range=(lo, hi), density=True, alpha=0.6, color=C_REAL,
            label=f"Simulated  (μ={rm.mean():.1f}, σ={rm.std():.1f})")
    ax.hist(gm, bins=bins, range=(lo, hi), density=True,
            histtype="step", linewidth=2.2, color=C_GEN,
            label=f"Generated  (μ={gm.mean():.1f}, σ={gm.std():.1f})")
    ax.set_title(
        f"Multiplicity — {species_name}  |  "
        f"Simulated: {n_real:,}  ·  Generated: {n_gen:,}",
        pad=8, fontsize=13)
    ax.set_xlabel(f"N({species_name}) per event", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    if logy:
        ax.set_yscale("log")
    savefig(fig, outpath)


def plot_multiplicity_simulated_only(real_mult, outpath, species_name, n_real, bins=50, logy=False):
    """Compact simulated-only multiplicity plot, suitable for slide inset."""
    rm = np.asarray(real_mult, dtype=np.float64); rm = rm[np.isfinite(rm)]
    if rm.size == 0:
        return
    lo = float(rm.min()); hi = float(rm.max())

    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    ax.hist(rm, bins=bins, range=(lo, hi), density=True,
            alpha=0.6, color=C_REAL, edgecolor="none")

    ax.text(0.97, 0.95,
            rf"$\mu={rm.mean():.1f}$,  $\sigma={rm.std():.1f}$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#333333")

    ax.set_title(f"Multiplicity — Simulated", fontsize=12, pad=6)
    ax.set_xlabel(f"N({species_name}) per event", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if logy:
        ax.set_yscale("log")
    savefig(fig, outpath)


def _plot_single_panel(real, gen, outpath, xlabel, title,
                       bins=80, ratio_min_count=10, fixed_range=None,
                       figsize=(6, 7)):
    """Single histogram + ratio panel, no zoom levels."""
    real = np.asarray(real, dtype=np.float64); real = real[np.isfinite(real)]
    gen  = np.asarray(gen,  dtype=np.float64); gen  = gen[np.isfinite(gen)]
    if real.size == 0 or gen.size == 0:
        return
    if fixed_range is not None:
        lo, hi = fixed_range
    else:
        rng = robust_range(real, gen)
        if rng is None: return
        lo, hi = rng

    r_data = clamp_to_range(real, lo, hi)
    g_data = clamp_to_range(gen,  lo, hi)
    edges  = np.linspace(lo, hi, bins + 1)
    r_cnt, _ = np.histogram(r_data, bins=edges)
    g_cnt, _ = np.histogram(g_data, bins=edges)
    ctrs = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.01)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1],
                           hspace=0.08, left=0.08, right=0.98,
                           top=0.95, bottom=0.10)
    ax_h = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax_h)
    ax_h.tick_params(labelbottom=False)

    ax_h.hist(r_data, bins=edges, density=True, alpha=0.6,
              color=C_REAL, label="Simulated")
    ax_h.hist(g_data, bins=edges, density=True, histtype="step",
              linewidth=2.2, color=C_GEN, label="Generated")
    ax_h.set_ylabel("Density", fontsize=14)
    ax_h.legend(loc="upper right", fontsize=12)

    _ratio_panel(ax_r, ctrs, r_cnt, g_cnt, ratio_min_count, show_ylabel=True)
    ax_r.set_xlabel(xlabel, fontsize=14)
    ax_r.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))
    fig.align_ylabels()
    savefig(fig, outpath)


# ── Energy ────────────────────────────────────────────────────────────────────
def plot_energy(real_E, gen_E, outpath, species_name,
                bins=80, ratio_min_count=10, n_real=0, n_gen=0):
    plot_single_three_zoom(
        real_E, gen_E, outpath=outpath,
        xlabel="E (signed) [GeV]",
        title="E (signed)",
        species_name=species_name,
        n_real=n_real, n_gen=n_gen,
        bins=bins, ratio_min_count=ratio_min_count,
        slight_zoom_frac=0.05,
        tight_zoom_frac=0.001,
    )


def plot_energy_from_mass(real_E, gen_beta, gen_E_sign, outpath, species_name,
                          me=0.000511, bins=80, ratio_min_count=10,
                          n_real=0, n_gen=0):
    gen_beta  = np.asarray(gen_beta,   dtype=np.float64)
    gen_Esign = np.asarray(gen_E_sign, dtype=np.float64)
    gen_Eabs  = np.abs(gen_Esign)
    beta_mag  = np.linalg.norm(gen_beta, axis=1)
    p_mag     = gen_Eabs * beta_mag
    E_proper  = np.sqrt(p_mag**2 + me**2)
    sign      = np.sign(gen_Esign)
    sign[sign == 0] = 1.0
    gen_E_recomputed = sign * E_proper

    plot_single_three_zoom(
        real_E, gen_E_recomputed, outpath=outpath,
        xlabel=r"E (from $m_e$) [GeV]",
        title=r"E recomputed via $\sqrt{|\mathbf{p}|^2 + m_e^2}$",
        species_name=species_name,
        n_real=n_real, n_gen=n_gen,
        bins=bins, ratio_min_count=ratio_min_count,
        slight_zoom_frac=0.05,
        tight_zoom_frac=0.001,
    )


# ── 3-component grouped subplot (VERTICAL layout) ─────────────────────────────
def three_panel_three_zoom(real_dict, gen_dict,
                            keys, xlabels, outpath,
                            title, species_name, n_real=0, n_gen=0,
                            bins=80, ratio_min_count=10,
                            fixed_ranges=None,
                            slight_zoom_frac=0.05,
                            tight_zoom_frac=0.005):
    assert len(keys) == 3
    if fixed_ranges is None:
        fixed_ranges = [None, None, None]

    nr_str = f"{n_real:,}" if n_real else "?"
    ng_str = f"{n_gen:,}"  if n_gen  else "?"
    full_title = (f"{title} — {species_name} | "
                  f"Simulated: {nr_str} · Generated: {ng_str}")

    row_defs = [
        ("Full range",   None),
        ("Slight zoom",  slight_zoom_frac),
        ("Tight zoom",   tight_zoom_frac),
    ]
    n_zoom = len(row_defs)
    n_cols = 3

    fig = plt.figure(figsize=(18, 7.5 * n_zoom))
    fig.suptitle(full_title, fontsize=20, y=1.002)

    outer = gridspec.GridSpec(n_zoom, n_cols, figure=fig,
                              left=0.06, right=0.98,
                              top=0.97, bottom=0.04,
                              wspace=0.28, hspace=0.50)

    for zoom_idx, (row_label, zoom_frac) in enumerate(row_defs):
        for col, (key, xlabel, frange) in enumerate(
                zip(keys, xlabels, fixed_ranges)):

            real = np.asarray(real_dict.get(key, []), dtype=np.float64)
            gen  = np.asarray(gen_dict.get(key,  []), dtype=np.float64)
            real = real[np.isfinite(real)]; gen = gen[np.isfinite(gen)]

            if frange is not None:
                lo, hi = frange
            else:
                rng = robust_range(real, gen)
                if rng is None:
                    continue
                lo, hi = rng

            real_full = clamp_to_range(real, lo, hi)
            gen_full  = clamp_to_range(gen,  lo, hi)
            combined  = np.concatenate([real_full, gen_full])

            inner = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer[zoom_idx, col],
                height_ratios=[3, 1], hspace=0.08)
            ax_h = fig.add_subplot(inner[0])
            ax_r = fig.add_subplot(inner[1], sharex=ax_h)
            ax_h.tick_params(labelbottom=False)

            if zoom_frac is None:
                r_data, g_data = real_full, gen_full
                edges = np.linspace(lo, hi, bins + 1)
            else:
                zrng = _peak_zoom_range(combined, zoom_frac=zoom_frac)
                if zrng is None:
                    ax_h.set_axis_off(); ax_r.set_axis_off(); continue
                zlo, zhi = zrng
                r_data = clamp_to_range(real, zlo, zhi)
                g_data = clamp_to_range(gen,  zlo, zhi)
                if r_data.size == 0 or g_data.size == 0:
                    ax_h.set_axis_off(); ax_r.set_axis_off(); continue
                edges = np.linspace(zlo, zhi, bins + 1)

            r_cnt, _ = np.histogram(r_data, bins=edges)
            g_cnt, _ = np.histogram(g_data, bins=edges)
            ctrs = 0.5 * (edges[:-1] + edges[1:])

            first_cell = (zoom_idx == 0 and col == 0)
            ax_h.hist(r_data, bins=edges, density=True, alpha=0.6,
                      color=C_REAL, label="Simulated" if first_cell else "_nolegend_")
            ax_h.hist(g_data, bins=edges, density=True, histtype="step",
                      linewidth=2.2, color=C_GEN, label="Generated" if first_cell else "_nolegend_")

            ax_h.set_title(xlabel if zoom_idx > 0 else f"{row_label}  ·  {xlabel}",
                           fontsize=17, pad=5)

            if col == 0:
                ax_h.set_ylabel("Density", fontsize=16)
            if first_cell:
                ax_h.legend(loc="upper right", fontsize=15)

            _ratio_panel(ax_r, ctrs, r_cnt, g_cnt,
                         ratio_min_count, show_ylabel=(col == 0))
            ax_r.set_xlabel(xlabel, fontsize=15)

    savefig(fig, outpath)


# ── Mass: single plot, log scale, E²−p² formula ──────────────────────────────
def plot_mass(simulated_events, generated_events, outdir,
              n_real=0, n_gen=0, bins=150):
    """
    Single plot: reconstructed mass m = sqrt(max(E² - p², 0)), all particles.
    Log-scale y-axis. No ratio panel. Shows m_e reference, sim peak, gen peak.
    """
    ME_PDG = 0.00051099895069  # GeV — full precision

    def extract_m(events, is_generated, pdg_sel=None):
        E_list, p_list = [], []
        for ev in events:
            ev = np.asarray(ev, dtype=np.float64)
            if ev.ndim != 2: continue
            if is_generated and ev.shape[1] >= 5:
                pdgs = ev[:, 0]; E = np.abs(ev[:, 1]); beta = ev[:, 2:5]
            elif not is_generated and ev.shape[1] >= 4:
                E = np.abs(ev[:, 0]); beta = ev[:, 1:4]
                pdgs = np.where(ev[:, 0] >= 0, 11, -11).astype(float)
            else:
                continue
            if pdg_sel is not None:
                mask = np.isin(pdgs, pdg_sel); E = E[mask]; beta = beta[mask]
            p = np.linalg.norm(E[:, None] * beta, axis=1)
            E_list.append(E); p_list.append(p)
        if not E_list: return np.array([])
        E_all = np.concatenate(E_list); p_all = np.concatenate(p_list)
        E_max = float(np.quantile(E_all, 0.99)); p_max = float(np.quantile(p_all, 0.99))
        sel = (E_all <= E_max) & (p_all <= p_max)
        return np.sqrt(np.clip(E_all[sel]**2 - p_all[sel]**2, 0, None))

    def find_peak(m_arr, bins_arr):
        counts, edges = np.histogram(m_arr, bins=bins_arr)
        i = int(np.argmax(counts))
        return 0.5 * (edges[i] + edges[i+1])

    nr = n_real or len(simulated_events)
    ng = n_gen  or len(generated_events)

    real_m = extract_m(simulated_events, False)
    gen_m  = extract_m(generated_events, True)
    if real_m.size == 0 or gen_m.size == 0: return

    hi_q  = 0.0009
    edges = np.linspace(0, hi_q, bins + 1)

    bw     = edges[1] - edges[0]
    r_cnt, _ = np.histogram(real_m, bins=edges)
    g_cnt, _ = np.histogram(gen_m,  bins=edges)
    r_dens = r_cnt / (r_cnt.sum() * bw + 1e-30)
    g_dens = g_cnt / (g_cnt.sum() * bw + 1e-30)
    ctrs   = 0.5 * (edges[:-1] + edges[1:])

    sim_peak = find_peak(real_m, edges)
    gen_peak = find_peak(gen_m,  edges)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    fig.suptitle(
        f"Reconstructed Electron Mass  |  Simulated: {nr:,}  ·  Generated: {ng:,}",
        fontsize=14,
    )

    ax.bar(ctrs, r_dens, width=bw, alpha=0.6, color=C_REAL, label="Simulated")
    ax.step(np.append(edges[:-1], edges[-1]),
            np.append(g_dens, g_dens[-1]),
            where="post", linewidth=2.2, color=C_GEN, label="Generated")

    ax.axvline(ME_PDG, color=C_MREF, linestyle="--", linewidth=1.8,
               label=rf"$m_e$ = {ME_PDG*1e3:.6f} MeV")
    ax.axvline(sim_peak, color=C_REAL, linestyle=":", linewidth=1.6,
               label=f"Sim peak = {sim_peak*1e3:.4f} MeV")
    ax.axvline(gen_peak, color=C_GEN, linestyle=":", linewidth=1.6,
               label=f"Gen peak = {gen_peak*1e3:.4f} MeV")

    ax.set_yscale("log")
    ax.set_xlabel(r"Reconstructed $m$ [GeV]", fontsize=14)
    ax.set_ylabel("Density (log)", fontsize=14)
    ax.set_xlim(0, hi_q)
    ax.legend(loc="upper right", fontsize=13)

    for arr, lbl in [(real_m, "Sim"), (gen_m, "Gen")]:
        peak = find_peak(arr, edges)
        print(f"    {lbl}  peak={peak*1e3:.4f} MeV  mean={arr.mean()*1e3:.4f} MeV  std={arr.std()*1e3:.4f} MeV")

    savefig(fig, os.path.join(outdir, "mass.png"))


# ── Per-event summed momentum comparison ──────────────────────────────────────
def plot_momentum_comparison(real_events, gen_events, outdir,
                             n_real=0, n_gen=0,
                             bins=80, ratio_min_count=10):
    print("  Computing per-event summed momenta …")

    def sum_p_real(ev):
        ev = np.asarray(ev, dtype=np.float64)
        E    = np.abs(ev[:, 0])
        beta = ev[:, 1:4]
        return (E[:, None] * beta).sum(axis=0)

    def sum_p_gen(ev):
        ev = np.asarray(ev, dtype=np.float64)
        E    = ev[:, 1]
        beta = ev[:, 2:5]
        return (E[:, None] * beta).sum(axis=0)

    real_sum = np.array([sum_p_real(ev) for ev in real_events])
    gen_sum  = np.array([sum_p_gen(ev)  for ev in gen_events])

    r_px, r_py, r_pz = real_sum[:, 0], real_sum[:, 1], real_sum[:, 2]
    g_px, g_py, g_pz = gen_sum[:, 0],  gen_sum[:, 1],  gen_sum[:, 2]

    nr = n_real or len(real_events)
    ng = n_gen  or len(gen_events)

    components = [
        (r_px, g_px, r"$\sum p_x$ [GeV]", "sum_px"),
        (r_py, g_py, r"$\sum p_y$ [GeV]", "sum_py"),
        (r_pz, g_pz, r"$\sum p_z$ [GeV]", "sum_pz"),
    ]

    for r_data, g_data, xlabel, tag in components:
        plot_single_three_zoom(
            r_data, g_data,
            outpath=os.path.join(outdir, f"momentum_{tag}.png"),
            xlabel=xlabel,
            title="Per-event summed momentum",
            species_name="all",
            n_real=nr, n_gen=ng,
            bins=bins, ratio_min_count=ratio_min_count,
            slight_zoom_frac=0.15,
            tight_zoom_frac=0.03,
        )

    r_pT   = np.sqrt(r_px**2 + r_py**2)
    g_pT   = np.sqrt(g_px**2 + g_py**2)
    r_pmag = np.sqrt(r_px**2 + r_py**2 + r_pz**2)
    g_pmag = np.sqrt(g_px**2 + g_py**2 + g_pz**2)

    for r_data, g_data, xlabel, tag in [
        (r_pT,   g_pT,   r"$p_T$ [GeV]", "sum_pT"),
        (r_pmag, g_pmag, r"$|\sum\mathbf{p}|$ [GeV]", "sum_pmag"),
    ]:
        plot_single_three_zoom(
            r_data, g_data,
            outpath=os.path.join(outdir, f"momentum_{tag}.png"),
            xlabel=xlabel,
            title="Per-event summed momentum",
            species_name="all",
            n_real=nr, n_gen=ng,
            bins=bins, ratio_min_count=ratio_min_count,
            slight_zoom_frac=0.15,
            tight_zoom_frac=0.03,
        )


# ── Opening angle ─────────────────────────────────────────────────────────────
def _find_closest_positron_opening_angles(electrons_p, positrons_p):
    if len(electrons_p) == 0 or len(positrons_p) == 0:
        return np.array([])
    e_norms = np.linalg.norm(electrons_p, axis=1, keepdims=True)
    p_norms = np.linalg.norm(positrons_p, axis=1, keepdims=True)
    e_valid = (e_norms[:, 0] > 0); p_valid = (p_norms[:, 0] > 0)
    electrons_p = electrons_p[e_valid]; positrons_p = positrons_p[p_valid]
    e_norms = e_norms[e_valid]; p_norms = p_norms[p_valid]
    if len(electrons_p) == 0 or len(positrons_p) == 0:
        return np.array([])
    e_hat = electrons_p / e_norms; p_hat = positrons_p / p_norms
    dots = np.clip(e_hat @ p_hat.T, -1.0, 1.0)
    return np.arccos(dots).min(axis=1)


def compute_opening_angles_from_events(events):
    all_angles = []
    for ev in events:
        ev = np.asarray(ev, dtype=np.float64)
        if ev.ndim != 2 or ev.shape[1] < 5:
            continue
        pdgs = ev[:, 0]
        E    = np.abs(ev[:, 1])
        beta = ev[:, 2:5]
        p_e  = E[pdgs ==  11, None] * beta[pdgs ==  11]
        p_ep = E[pdgs == -11, None] * beta[pdgs == -11]
        angles = _find_closest_positron_opening_angles(p_e, p_ep)
        if angles.size > 0:
            all_angles.append(angles)
    return np.concatenate(all_angles) if all_angles else np.array([])


def compute_opening_angles_from_species(sp_eminus, sp_eplus):
    def iter_event_p(sp):
        px, py, pz = (np.asarray(sp[k], dtype=np.float64) for k in ("px","py","pz"))
        mult = np.asarray(sp["mult"], dtype=int)
        idx = 0
        for m in mult:
            yield (np.column_stack([px[idx:idx+m], py[idx:idx+m], pz[idx:idx+m]])
                   if m > 0 else np.empty((0, 3)))
            idx += m

    all_angles = []
    for p_e, p_ep in zip(iter_event_p(sp_eminus), iter_event_p(sp_eplus)):
        angles = _find_closest_positron_opening_angles(p_e, p_ep)
        if angles.size > 0:
            all_angles.append(angles)
    return np.concatenate(all_angles) if all_angles else np.array([])


def plot_opening_angles(real_sp_eminus, real_sp_eplus,
                        gen_events, outdir, n_real=0, n_gen=0,
                        bins=80, ratio_min_count=10):
    print("  Computing opening angles …")
    real_angles = compute_opening_angles_from_species(real_sp_eminus, real_sp_eplus)
    gen_angles  = compute_opening_angles_from_events(gen_events)
    print(f"  Real: {len(real_angles):,} angles,  Generated: {len(gen_angles):,} angles")
    if real_angles.size == 0 or gen_angles.size == 0:
        print("  WARNING: no opening angles — skipping."); return

    real_deg = np.degrees(real_angles)
    gen_deg  = np.degrees(gen_angles)

    zoom_cols = [
        (0.0, 180.0, "Full range  (0–180°)"),
        (0.0,  30.0, "Slight zoom  (0–30°)"),
        (0.0,   5.0, "Tight zoom  (0–5°)"),
    ]

    nr, ng = n_real or len(real_deg), n_gen or len(gen_deg)

    fig = plt.figure(figsize=(9, 5.0 * len(zoom_cols)))
    fig.suptitle(
        r"Closest $e^+$/$e^-$ pair opening angle  —  all events"
        f"  |  Simulated: {nr:,}  ·  Generated: {ng:,}",
        fontsize=14, y=0.995,
    )
    outer = gridspec.GridSpec(len(zoom_cols), 1, figure=fig,
                              left=0.08, right=0.98,
                              top=0.94, bottom=0.04, hspace=0.22)

    for row_idx, (lo, hi, sub_title) in enumerate(zoom_cols):
        r_data = clamp_to_range(real_deg, lo, hi)
        g_data = clamp_to_range(gen_deg,  lo, hi)
        edges  = np.linspace(lo, hi, bins + 1)
        r_cnt, _ = np.histogram(r_data, bins=edges)
        g_cnt, _ = np.histogram(g_data, bins=edges)
        ctrs = 0.5 * (edges[:-1] + edges[1:])

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row_idx],
            height_ratios=[3, 1], hspace=0.08)
        ax_h = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax_h)
        ax_h.tick_params(labelbottom=False, bottom=True)

        ax_h.hist(r_data, bins=edges, density=True, alpha=0.6, color=C_REAL,
                  label="Simulated" if row_idx == 0 else "_nolegend_")
        ax_h.hist(g_data, bins=edges, density=True, histtype="step",
                  linewidth=2.2, color=C_GEN,
                  label="Generated" if row_idx == 0 else "_nolegend_")
        ax_h.set_title(sub_title, fontsize=17, pad=5)
        ax_h.set_ylabel("Density", fontsize=16)
        ax_h.set_xlim(lo, hi)
        if row_idx == 0:
            ax_h.legend(loc="upper right", fontsize=16)

        _ratio_panel(ax_r, ctrs, r_cnt, g_cnt, ratio_min_count, show_ylabel=True)
        if row_idx == len(zoom_cols) - 1:
            ax_r.set_xlabel(r"$\theta$ [deg]", fontsize=14)
        else:
            ax_r.set_xlabel("")
        ax_r.set_xlim(lo, hi)

    fig.align_ylabels()
    savefig(fig, os.path.join(outdir, "opening_angle.png"))


# ── Charge asymmetry ──────────────────────────────────────────────────────────
def plot_charge_asymmetry(generated_events, outdir):
    charge_diff, total_mult = [], []
    for ev in generated_events:
        ev = np.asarray(ev)
        pdgs = ev[:, 0]
        charge_diff.append(np.sum(pdgs == 11) - np.sum(pdgs == -11))
        total_mult.append(len(pdgs))

    charge_diff = np.array(charge_diff)
    total_mult  = np.array(total_mult)
    mu = charge_diff.mean(); sigma = charge_diff.std()
    lo = float(np.quantile(charge_diff, 0.01))
    hi = float(np.quantile(charge_diff, 0.99))
    charge_diff = charge_diff[(charge_diff >= lo) & (charge_diff <= hi)]
    unique_vals = np.unique(charge_diff)
    bins_edges  = np.append(unique_vals - 0.5, unique_vals[-1] + 0.5)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.hist(charge_diff, bins=bins_edges, rwidth=1.0, edgecolor="black",
            color=C_GEN, alpha=0.75)
    dummy = Line2D([], [], linestyle="none")
    ax.legend([dummy],
              [rf"$\langle N \rangle = {total_mult.mean():.2f}$" "\n"
               rf"$\mu = {mu:.3f}$" "\n"
               rf"$\sigma = {sigma:.3f}$"],
              fontsize=19)
    ax.set_xlabel(r"$N_{e^-} - N_{e^+}$", fontsize=20)
    ax.set_ylabel("Number of events", fontsize=14)
    ax.set_title("Charge Asymmetry Per Event — Generated", fontsize=22)
    savefig(fig, os.path.join(outdir, "charge_asymmetry.png"))


# ── Corner plots (same colours) ───────────────────────────────────────────────
def plot_corner_overlay(real_dict, gen_dict, keys, labels, outpath,
                        title="", max_points=40000,
                        q_lo=0.01, q_hi=0.99, seed=123, bins=40):
    try:
        import corner as _corner
    except ImportError:
        print("  corner not available — skipping corner plot"); return

    cols_r = [np.asarray(real_dict[k], dtype=np.float64) for k in keys]
    cols_g = [np.asarray(gen_dict[k],  dtype=np.float64) for k in keys]
    R = np.stack(cols_r, axis=1); G = np.stack(cols_g, axis=1)
    R = R[np.all(np.isfinite(R), axis=1)]; G = G[np.all(np.isfinite(G), axis=1)]
    if R.shape[0] == 0 or G.shape[0] == 0:
        return

    rng = np.random.default_rng(seed)
    if R.shape[0] > max_points:
        R = R[rng.choice(R.shape[0], size=max_points, replace=False)]
    if G.shape[0] > max_points:
        G = G[rng.choice(G.shape[0], size=max_points, replace=False)]

    ranges = []
    for d in range(len(keys)):
        z = np.concatenate([R[:, d], G[:, d]])
        ranges.append((float(np.quantile(z, q_lo)), float(np.quantile(z, q_hi))))

    fig = _corner.corner(R, labels=labels, range=ranges, bins=bins, smooth=1.0,
                         plot_density=True, plot_contours=True, fill_contours=True,
                         levels=(0.68, 0.95), color=C_REAL,
                         label_kwargs={"fontsize": 16})
    _corner.corner(G, fig=fig, range=ranges, bins=bins, smooth=1.0,
                   plot_density=False, plot_contours=True, fill_contours=False,
                   levels=(0.68, 0.95), color=C_GEN)
    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    savefig(fig, outpath)


# ── Main evaluate function ────────────────────────────────────────────────────
try:
    from particle_diffusion_new_data_discretepdgconditioning import (
        load_events, extract_species, CFG,
    )
    _HAVE_MAIN = True
except ImportError:
    _HAVE_MAIN = False


def evaluate(args):
    if not _HAVE_MAIN:
        raise RuntimeError("particle_diffusion_new_data_discretepdgconditioning.py "
                           "must be importable.")
    setup_style()
    cfg = CFG()

    if args.outdir is None:
        args.outdir = os.path.dirname(args.gen_path)
    os.makedirs(args.outdir, exist_ok=True)

    real_events = load_events(args.real_path)
    gen_events  = load_events(args.gen_path)
    n_real = len(real_events); n_gen = len(gen_events)
    print(f"Loaded {n_real:,} real  /  {n_gen:,} generated events")

    mom_bins       = getattr(args, "mom_bins", 80)
    mult_bins      = getattr(args, "mult_bins", 50)
    ratio_min      = getattr(args, "ratio_min_count", 10)

    species_list = [
        {"name": "all",  "pdgs": None,  "tag": "all"},
        {"name": "e⁻",   "pdgs": [11],  "tag": "eminus"},
        {"name": "e⁺",   "pdgs": [-11], "tag": "eplus"},
    ]

    for sp in species_list:
        print(f"\n── {sp['name']} ──")
        real_sp = extract_species(real_events, sp["pdgs"], me=cfg.me)
        gen_sp  = extract_species(gen_events,  sp["pdgs"], me=cfg.me)
        tag = sp["tag"]

        # 1) Multiplicity: overlay
        plot_multiplicity(
            real_sp["mult"], gen_sp["mult"],
            outpath=os.path.join(args.outdir, f"multiplicity_{tag}.png"),
            species_name=sp["name"], n_real=n_real, n_gen=n_gen,
            bins=mult_bins,
            figsize=(8, 5.0),
        )
        # 1b) Simulated-only multiplicity
        plot_multiplicity_simulated_only(
            real_sp["mult"],
            outpath=os.path.join(args.outdir, f"multiplicity_simulated_{tag}.png"),
            species_name=sp["name"], n_real=n_real,
            bins=mult_bins,
        )

        # 2) Energy
        e_key = "E_signed"
        if real_sp[e_key].size and gen_sp[e_key].size:
            plot_energy(
                real_sp[e_key], gen_sp[e_key],
                outpath=os.path.join(args.outdir, f"energy_{tag}.png"),
                species_name=sp["name"],
                bins=mom_bins, ratio_min_count=ratio_min,
                n_real=n_real, n_gen=n_gen,
            )
            # 2b) Fixed-range single panel: -0.01 to 0.01 GeV
            _plot_single_panel(
                real_sp[e_key], gen_sp[e_key],
                outpath=os.path.join(args.outdir, f"energy_zoom_{tag}.png"),
                xlabel="E (signed) [GeV]",
                title=("E (signed) peak zoom  -  " + sp["name"] + f"  |  Simulated: {n_real:,}  Generated: {n_gen:,}"),
                bins=mom_bins, ratio_min_count=ratio_min,
                fixed_range=(-0.01, 0.01),
            )

        # 3) Beta components
        for key, xlabel in [("betax", r"$\beta_x$"),
                             ("betay", r"$\beta_y$"),
                             ("betaz", r"$\beta_z$")]:
            if real_sp[key].size and gen_sp[key].size:
                plot_single_three_zoom(
                    real_sp[key], gen_sp[key],
                    outpath=os.path.join(args.outdir, f"{key}_{tag}.png"),
                    xlabel=xlabel, title=r"$\beta$ component",
                    species_name=sp["name"], n_real=n_real, n_gen=n_gen,
                    bins=mom_bins, ratio_min_count=ratio_min,
                    fixed_range=(-1, 1),
                    slight_zoom_frac=0.15, tight_zoom_frac=0.003,
                    tight_zoom_range=(-0.0025, 0.0025),  # explicit tight x-limits for beta
                )

        # 4) Momentum grouped
        if all(real_sp[k].size and gen_sp[k].size for k in ("px", "py", "pz")):
            three_panel_three_zoom(
                real_sp, gen_sp,
                keys=["px", "py", "pz"],
                xlabels=[r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"],
                outpath=os.path.join(args.outdir, f"p_xyz_{tag}.png"),
                title="Momentum components", species_name=sp["name"],
                n_real=n_real, n_gen=n_gen, bins=mom_bins,
            )

        # 5) Position — x,y in um; z in mm
        for key, xlabel, scale in [("x", "x [μm]", 1e-3), ("y", "y [μm]", 1e-3)]:
            if real_sp[key].size and gen_sp[key].size:
                plot_single_three_zoom(
                    real_sp[key] * scale, gen_sp[key] * scale,
                    outpath=os.path.join(args.outdir, f"pos_{key}_{tag}.png"),
                    xlabel=xlabel, title=f"Position {key}",
                    species_name=sp["name"], n_real=n_real, n_gen=n_gen,
                    bins=mom_bins, ratio_min_count=ratio_min,
                    slight_zoom_frac=0.15, tight_zoom_frac=0.005,
                )
        # z — full range + slight zoom (tight zoom not useful for discrete peaks)
        if real_sp["z"].size and gen_sp["z"].size:
            r_z = real_sp["z"] * 1e-6; g_z = gen_sp["z"] * 1e-6
            plot_single_three_zoom(
                r_z, g_z,
                outpath=os.path.join(args.outdir, f"pos_z_{tag}.png"),
                xlabel="z [mm]", title="Position z",
                species_name=sp["name"], n_real=n_real, n_gen=n_gen,
                bins=mom_bins, ratio_min_count=ratio_min,
                slight_zoom_frac=0.02, tight_zoom_frac=0.0,
            )

        # 6) Corner plots
        corner_sets = [
            (["px", "py", "pz"],
             [r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"],
             "p_xyz", r"$p_x, p_y, p_z$"),
            (["betax", "betay", "betaz"],
             [r"$\beta_x$", r"$\beta_y$", r"$\beta_z$"],
             "beta_xyz", r"$\beta_x, \beta_y, \beta_z$"),
        ]
        if real_sp["x"].size and gen_sp["x"].size:
            corner_sets.append((["x", "y", "z"], ["x [nm]", "y [nm]", "z [nm]"],
                                "xyz", "x, y, z"))
        for keys, labels, ctag, ctitle in corner_sets:
            if any(real_sp[k].size == 0 or gen_sp[k].size == 0 for k in keys):
                continue
            plot_corner_overlay(
                real_sp, gen_sp, keys=keys, labels=labels,
                outpath=os.path.join(args.outdir, f'corner_{ctag}_{tag}.png'),
                title=ctitle + '  -  ' + sp['name'] + f'  |  Simulated: {n_real:,}  Generated: {n_gen:,}',
                max_points=30000, q_lo=0.01, q_hi=0.99, seed=cfg.seed,
            )

    # ── Mass distribution ─────────────────────────────────────────────────────
    print("\n── Mass distribution ──")
    plot_mass(real_events, gen_events, outdir=args.outdir,
              n_real=n_real, n_gen=n_gen, bins=150)

    # ── Charge asymmetry ──────────────────────────────────────────────────────
    print("\n── Charge asymmetry ──")
    plot_charge_asymmetry(gen_events, outdir=args.outdir)

    # ── Opening angles ────────────────────────────────────────────────────────
    print("\n── Opening angles ──")
    real_sp_eminus = extract_species(real_events, [11],  me=cfg.me)
    real_sp_eplus  = extract_species(real_events, [-11], me=cfg.me)
    plot_opening_angles(
        real_sp_eminus, real_sp_eplus, gen_events,
        outdir=args.outdir, n_real=n_real, n_gen=n_gen,
        bins=mom_bins, ratio_min_count=ratio_min,
    )

    # ── Per-event summed momentum ─────────────────────────────────────────────
    print("\n── Per-event summed momentum ──")
    plot_momentum_comparison(
        real_events, gen_events,
        outdir=args.outdir, n_real=n_real, n_gen=n_gen,
        bins=mom_bins, ratio_min_count=ratio_min,
    )

    print(f"\nDone. Plots saved to  {args.outdir}/")


evaluate_improved = evaluate


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined particle evaluation plots")
    parser.add_argument("--real_path",       required=True)
    parser.add_argument("--gen_path",        required=True)
    parser.add_argument("--outdir",          default=None)
    parser.add_argument("--mom_bins",        type=int, default=80)
    parser.add_argument("--mult_bins",       type=int, default=50)
    parser.add_argument("--ratio_min_count", type=int, default=10)
    args = parser.parse_args()
    setup_style()
    evaluate(args)