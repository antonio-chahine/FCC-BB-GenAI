#!/usr/bin/env python3
"""
Combined evaluation script:
  - particle_diffusion_eval.py  (beta/energy/position/momentum plots)
  - mass_plot_final.py          (per-particle mass distribution)
  - check_charge.py             (charge asymmetry per event)
  - opening_angle_plot.py       (closest e+/e- pair opening angles)
  - momentum_comparison.py      (per-event summed momentum components)  ← NEW

All plots with distributions now show THREE zoom levels:
  col 0 — full range
  col 1 — slight zoom (~5 % of full width around peak)
  col 2 — tight zoom  (~0.5 % of full width around peak)
Each zoom level has its own histogram + fractional-difference ratio panel.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# ── Palette ───────────────────────────────────────────────────────────────────
C_REAL = "#2166AC"
C_GEN  = "#D6604D"
C_BAND = "#AAAAAA"
C_ZERO = "#333333"
C_MREF = "#2ca02c"

# ── Global style ──────────────────────────────────────────────────────────────
def setup_style():
    matplotlib.rcParams.update({
        "figure.facecolor":    "white",
        "savefig.facecolor":   "white",
        "axes.facecolor":      "white",
        "axes.titlesize":      17,
        "axes.labelsize":      14,
        "xtick.labelsize":     18,
        "ytick.labelsize":     18,
        "legend.fontsize":     20,
        "legend.frameon":      True,
        "legend.facecolor":    "white",
        "legend.edgecolor":    "#cccccc",
        "legend.framealpha":   0.95,
        "axes.edgecolor":      "black",
        "axes.linewidth":      1.2,
        "axes.grid":           False,
        "lines.linewidth":     2.0,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.major.size":    5,
        "ytick.major.size":    5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.titlepad":       10,
        "axes.labelpad":       6,
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
    ax.plot(centers[mask], frac[mask], "o", markersize=3,
            color=C_GEN, linewidth=0, zorder=2)
    ax.set_ylim(-1.0, 1.0)
    if show_ylabel:
        ax.set_ylabel("Fract.\ndiff.", fontsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))


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


# ── Core: single variable — full + slight zoom + tight zoom ───────────────────
def plot_single_three_zoom(real, gen, outpath, xlabel, title, species_name,
                           n_real=0, n_gen=0, bins=80, ratio_min_count=10,
                           fixed_range=None,
                           slight_zoom_frac=0.05,
                           tight_zoom_frac=0.005):
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
    combined  = np.concatenate([real_full, gen_full])

    slight_rng = _peak_zoom_range(combined, zoom_frac=slight_zoom_frac)
    tight_rng  = _peak_zoom_range(combined, zoom_frac=tight_zoom_frac)

    cols = [(lo, hi, real_full, gen_full, "Full range", xlabel)]
    for rng_z, label_z in [(slight_rng, "Slight zoom"), (tight_rng, "Tight zoom")]:
        if rng_z is not None:
            zlo, zhi = rng_z
            r_z = clamp_to_range(real, zlo, zhi)
            g_z = clamp_to_range(gen,  zlo, zhi)
            if r_z.size > 0 and g_z.size > 0:
                cols.append((zlo, zhi, r_z, g_z, label_z, f"{xlabel} (zoom)"))

    n_cols = len(cols)
    nr = n_real if n_real else len(real_full)
    ng = n_gen  if n_gen  else len(gen_full)

    fig = plt.figure(figsize=(8.5 * n_cols + 0.5, 8.5))
    fig.suptitle(
        f"{title}  —  {species_name} | "
        f"Simulated: {nr:,}  ·  Generated: {ng:,}",
        fontsize=21, y=1.06
    )
    fig.subplots_adjust(top=0.88, left=0.07, right=0.98,
                        bottom=0.09, wspace=0.30)
    outer = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.30)

    for col_idx, (clo, chi, r_data, g_data, sub_title, xl) in enumerate(cols):
        edges = np.linspace(clo, chi, bins + 1)
        r_cnt, _ = np.histogram(r_data, bins=edges)
        g_cnt, _ = np.histogram(g_data, bins=edges)
        ctrs = 0.5 * (edges[:-1] + edges[1:])

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col_idx],
            height_ratios=[3, 1], hspace=0.08)
        ax_h = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax_h)
        ax_h.tick_params(labelbottom=False)

        ax_h.hist(r_data, bins=edges, density=True, alpha=0.55,
                  color=C_REAL,
                  label="Simulated" if col_idx == 0 else "_nolegend_")
        ax_h.hist(g_data, bins=edges, density=True, histtype="step",
                  linewidth=2.2, color=C_GEN,
                  label="Generated" if col_idx == 0 else "_nolegend_")

        ax_h.set_title(sub_title, fontsize=19, pad=5)
        ax_h.set_ylabel("Density" if col_idx == 0 else "", fontsize=20)
        if col_idx == 0:
            ax_h.legend(loc="upper right", fontsize=18)

        _ratio_panel(ax_r, ctrs, r_cnt, g_cnt,
                     ratio_min_count, show_ylabel=(col_idx == 0))
        ax_r.set_xlabel(xl, fontsize=18)

    savefig(fig, outpath)


# ── Multiplicity ──────────────────────────────────────────────────────────────
def plot_multiplicity(real_mult, gen_mult, outpath, species_name,
                      n_real, n_gen, bins=50, logy=False):
    rm = np.asarray(real_mult, dtype=np.float64); rm = rm[np.isfinite(rm)]
    gm = np.asarray(gen_mult,  dtype=np.float64); gm = gm[np.isfinite(gm)]
    rng = robust_range(rm, gm, q_lo=0.0, q_hi=1.0)
    if rng is None:
        return
    lo, hi = rng
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.hist(rm, bins=bins, range=(lo, hi), density=True, alpha=0.55,
            color=C_REAL,
            label=f"Simulated  (μ={rm.mean():.1f}, σ={rm.std():.1f})")
    ax.hist(gm, bins=bins, range=(lo, hi), density=True,
            histtype="step", linewidth=2.2, color=C_GEN,
            label=f"Generated  (μ={gm.mean():.1f}, σ={gm.std():.1f})")
    ax.set_title(
        f"Multiplicity — {species_name} | "
        f"Simulated: {n_real:,} events, Generated: {n_gen:,} events", pad=10)
    ax.set_xlabel(f"N({species_name}) per event")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    if logy:
        ax.set_yscale("log")
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
        xlabel="E (from $m_e$) [GeV]",
        title=r"E recomputed via $\sqrt{|\mathbf{p}|^2 + m_e^2}$",
        species_name=species_name,
        n_real=n_real, n_gen=n_gen,
        bins=bins, ratio_min_count=ratio_min_count,
        slight_zoom_frac=0.05,
        tight_zoom_frac=0.001,
    )


# ── 3-component grouped subplot ───────────────────────────────────────────────
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
                  f"Simulated: {nr_str} events, Generated: {ng_str} events")

    row_defs = [
        ("Full range",   None),
        ("Slight zoom",  slight_zoom_frac),
        ("Tight zoom",   tight_zoom_frac),
    ]
    n_rows = len(row_defs)
    n_cols = 3

    fig = plt.figure(figsize=(22, 6.5 * n_rows))
    fig.suptitle(full_title, fontsize=22, y=1.001)

    outer = gridspec.GridSpec(1, n_cols, figure=fig,
                              left=0.06, right=0.98,
                              top=0.955, bottom=0.05, wspace=0.28)

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
            n_rows * 2, 1, subplot_spec=outer[col],
            height_ratios=[3, 1] * n_rows,
            hspace=0.08)

        for row_idx, (row_label, zoom_frac) in enumerate(row_defs):
            ax_h = fig.add_subplot(inner[row_idx * 2])
            ax_r_row = fig.add_subplot(inner[row_idx * 2 + 1], sharex=ax_h)
            ax_h.tick_params(labelbottom=False)

            if zoom_frac is None:
                r_data, g_data = real_full, gen_full
                edges = np.linspace(lo, hi, bins + 1)
            else:
                zrng = _peak_zoom_range(combined, zoom_frac=zoom_frac)
                if zrng is None:
                    ax_h.set_axis_off(); ax_r_row.set_axis_off(); continue
                zlo, zhi = zrng
                r_data = clamp_to_range(real, zlo, zhi)
                g_data = clamp_to_range(gen,  zlo, zhi)
                if r_data.size == 0 or g_data.size == 0:
                    ax_h.set_axis_off(); ax_r_row.set_axis_off(); continue
                edges = np.linspace(zlo, zhi, bins + 1)

            r_cnt, _ = np.histogram(r_data, bins=edges)
            g_cnt, _ = np.histogram(g_data, bins=edges)
            ctrs = 0.5 * (edges[:-1] + edges[1:])

            lbl_sim = "Simulated" if (col == 0 and row_idx == 0) else "_nolegend_"
            lbl_gen = "Generated" if (col == 0 and row_idx == 0) else "_nolegend_"

            ax_h.hist(r_data, bins=edges, density=True, alpha=0.55,
                      color=C_REAL, label=lbl_sim)
            ax_h.hist(g_data, bins=edges, density=True, histtype="step",
                      linewidth=2.2, color=C_GEN, label=lbl_gen)

            if row_idx == 0:
                ax_h.set_title(xlabel, fontsize=20, pad=6)
                if col == 0:
                    ax_h.legend(loc="upper right", fontsize=17)

            if col == 0:
                ax_h.set_ylabel(f"{row_label}\nDensity", fontsize=17)
            else:
                ax_h.set_ylabel("")

            _ratio_panel(ax_r_row, ctrs, r_cnt, g_cnt,
                         ratio_min_count, show_ylabel=(col == 0))

            if row_idx == n_rows - 1:
                ax_r_row.set_xlabel(xlabel, fontsize=18)

    savefig(fig, outpath)


# ── Per-event summed momentum comparison ──────────────────────────────────────
def plot_momentum_comparison(real_events, gen_events, outdir,
                             n_real=0, n_gen=0,
                             bins=80, ratio_min_count=10):
    """
    Compare per-event summed momentum (px, py, pz) between real and generated.

    Real format:      [E_signed, betax, betay, betaz, x, y, z]  (7 cols)
    Generated format: [pdg, E, betax, betay, betaz, x, y, z]    (8 cols)

    p = E * beta  (relativistic: p = γm·β, E = γm  →  p = E·β)
    For real data E is always positive (sign encodes PDG), so we use abs().
    """
    print("  Computing per-event summed momenta …")

    def sum_p_real(ev):
        ev = np.asarray(ev, dtype=np.float64)
        E    = np.abs(ev[:, 0])        # col 0: E_signed  → take abs
        beta = ev[:, 1:4]              # cols 1,2,3: betax, betay, betaz
        return (E[:, None] * beta).sum(axis=0)   # (3,)

    def sum_p_gen(ev):
        ev = np.asarray(ev, dtype=np.float64)
        E    = ev[:, 1]                # col 1: E (always positive)
        beta = ev[:, 2:5]              # cols 2,3,4: betax, betay, betaz
        return (E[:, None] * beta).sum(axis=0)   # (3,)

    real_sum = np.array([sum_p_real(ev) for ev in real_events])   # (N_real, 3)
    gen_sum  = np.array([sum_p_gen(ev)  for ev in gen_events])    # (N_gen,  3)

    r_px, r_py, r_pz = real_sum[:, 0], real_sum[:, 1], real_sum[:, 2]
    g_px, g_py, g_pz = gen_sum[:, 0],  gen_sum[:, 1],  gen_sum[:, 2]

    nr = n_real or len(real_events)
    ng = n_gen  or len(gen_events)

    components = [
        (r_px, g_px, r"$\sum p_x$ [GeV]", "sum_px"),
        (r_py, g_py, r"$\sum p_y$ [GeV]", "sum_py"),
        (r_pz, g_pz, r"$\sum p_z$ [GeV]", "sum_pz"),
    ]

    # ── Main 3-panel overlay (matches the style of the uploaded image) ────────
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle(
        f"Per-event summed momentum components: Real vs Generated  |  "
        f"Simulated: {nr:,}  ·  Generated: {ng:,}",
        fontsize=16,
    )

    for ax, (r_data, g_data, xlabel, _) in zip(axs, components):
        rng = robust_range(r_data, g_data)
        if rng is None:
            continue
        lo, hi = rng
        ax.hist(r_data, bins=bins, range=(lo, hi), density=True,
                alpha=0.55, color=C_REAL, label="Real")
        ax.hist(g_data, bins=bins, range=(lo, hi), density=True,
                histtype="step", linewidth=2.2, color=C_GEN, label="Generated")
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Density", fontsize=14)
        ax.legend(fontsize=13)

        # Print summary stats
        for arr, lbl in [(r_data, "Real"), (g_data, "Gen ")]:
            arr = arr[np.isfinite(arr)]
            print(f"    {lbl}  {xlabel:20s}  mean={arr.mean():.5f}  "
                  f"std={arr.std():.5f}  "
                  f"[1%,50%,99%]={np.percentile(arr,[1,50,99])}")

    savefig(fig, os.path.join(outdir, "momentum_sum_components.png"))

    # ── Per-component with ratio panels ──────────────────────────────────────
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

    # ── Derived: pT and |p| ───────────────────────────────────────────────────
    r_pT   = np.sqrt(r_px**2 + r_py**2)
    g_pT   = np.sqrt(g_px**2 + g_py**2)
    r_pmag = np.sqrt(r_px**2 + r_py**2 + r_pz**2)
    g_pmag = np.sqrt(g_px**2 + g_py**2 + g_pz**2)

    for r_data, g_data, xlabel, tag in [
        (r_pT,   g_pT,   r"$p_T = \sqrt{(\sum p_x)^2+(\sum p_y)^2}$ [GeV]", "sum_pT"),
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
    angles = []
    if len(electrons_p) == 0 or len(positrons_p) == 0:
        return np.array(angles)
    e_norms = np.linalg.norm(electrons_p, axis=1, keepdims=True)
    p_norms = np.linalg.norm(positrons_p, axis=1, keepdims=True)
    e_valid = (e_norms[:, 0] > 0)
    p_valid = (p_norms[:, 0] > 0)
    electrons_p = electrons_p[e_valid]
    positrons_p = positrons_p[p_valid]
    e_norms = e_norms[e_valid]
    p_norms = p_norms[p_valid]
    if len(electrons_p) == 0 or len(positrons_p) == 0:
        return np.array(angles)
    e_hat = electrons_p / e_norms
    p_hat = positrons_p / p_norms
    dots = np.clip(e_hat @ p_hat.T, -1.0, 1.0)
    angles_mat = np.arccos(dots)
    min_angles = angles_mat.min(axis=1)
    return min_angles


def _extract_momenta_from_events(events, pdg_code, is_generated=False):
    per_event = []
    for ev in events:
        ev = np.asarray(ev, dtype=np.float64)
        if ev.ndim != 2:
            continue
        if is_generated:
            if ev.shape[1] < 5:
                continue
            pdgs  = ev[:, 0]
            mask  = (pdgs == pdg_code)
            E     = np.abs(ev[mask, 1])
            beta  = ev[mask, 2:5]
            p     = E[:, None] * beta
        else:
            if ev.shape[1] < 4:
                continue
            E     = np.abs(ev[:, 0])
            beta  = ev[:, 1:4]
            p     = E[:, None] * beta
        per_event.append(p)
    return per_event


def compute_opening_angles_from_events(events):
    all_angles = []
    for ev in events:
        ev = np.asarray(ev, dtype=np.float64)
        if ev.ndim != 2 or ev.shape[1] < 5:
            continue
        pdgs    = ev[:, 0]
        mask_e  = (pdgs ==  11)
        mask_ep = (pdgs == -11)
        E       = np.abs(ev[:, 1])
        beta    = ev[:, 2:5]
        p_e     = E[mask_e,  None] * beta[mask_e]
        p_ep    = E[mask_ep, None] * beta[mask_ep]
        angles  = _find_closest_positron_opening_angles(p_e, p_ep)
        if angles.size > 0:
            all_angles.append(angles)
    return np.concatenate(all_angles) if all_angles else np.array([])


def compute_opening_angles_from_species(sp_eminus, sp_eplus):
    def iter_event_p(sp):
        px   = np.asarray(sp["px"],   dtype=np.float64)
        py   = np.asarray(sp["py"],   dtype=np.float64)
        pz   = np.asarray(sp["pz"],   dtype=np.float64)
        mult = np.asarray(sp["mult"], dtype=int)
        idx  = 0
        for m in mult:
            if m > 0:
                yield np.column_stack([px[idx:idx+m],
                                       py[idx:idx+m],
                                       pz[idx:idx+m]])
            else:
                yield np.empty((0, 3))
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
        print("  WARNING: no opening angles computed — skipping plot.")
        return

    real_deg = np.degrees(real_angles)
    gen_deg  = np.degrees(gen_angles)

    zoom_cols = [
        (0.0,  180.0, "Full range  (0–180°)"),
        (0.0,   30.0, "Slight zoom  (0–30°)"),
        (0.0,    5.0, "Tight zoom  (0–5°)"),
    ]

    nr, ng = n_real or len(real_deg), n_gen or len(gen_deg)
    fig = plt.figure(figsize=(8.5 * 3 + 0.5, 8.5))
    fig.suptitle(
        r"Closest $e^+$/$e^-$ pair opening angle  —  all events | "
        f"Simulated: {nr:,}  ·  Generated: {ng:,}",
        fontsize=21, y=1.06,
    )
    fig.subplots_adjust(top=0.88, left=0.07, right=0.98, bottom=0.09, wspace=0.30)
    outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30)

    for col_idx, (lo, hi, sub_title) in enumerate(zoom_cols):
        r_data = clamp_to_range(real_deg, lo, hi)
        g_data = clamp_to_range(gen_deg,  lo, hi)

        edges = np.linspace(lo, hi, bins + 1)
        r_cnt, _ = np.histogram(r_data, bins=edges)
        g_cnt, _ = np.histogram(g_data, bins=edges)
        ctrs = 0.5 * (edges[:-1] + edges[1:])

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col_idx],
            height_ratios=[3, 1], hspace=0.08)
        ax_h = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax_h)
        ax_h.tick_params(labelbottom=False)

        ax_h.hist(r_data, bins=edges, density=True, alpha=0.55, color=C_REAL,
                  label="Simulated" if col_idx == 0 else "_nolegend_")
        ax_h.hist(g_data, bins=edges, density=True, histtype="step",
                  linewidth=2.2, color=C_GEN,
                  label="Generated" if col_idx == 0 else "_nolegend_")

        ax_h.set_title(sub_title, fontsize=19, pad=5)
        ax_h.set_ylabel("Density" if col_idx == 0 else "", fontsize=20)
        ax_h.set_xlim(lo, hi)
        if col_idx == 0:
            ax_h.legend(loc="upper right", fontsize=18)

        _ratio_panel(ax_r, ctrs, r_cnt, g_cnt, ratio_min_count,
                     show_ylabel=(col_idx == 0))
        ax_r.set_xlabel(r"$\theta$ [deg]", fontsize=18)
        ax_r.set_xlim(lo, hi)

    savefig(fig, os.path.join(outdir, "opening_angle.png"))


# ── Mass ──────────────────────────────────────────────────────────────────────
def plot_mass(simulated_events, generated_events, outdir):
    m_e = 0.000511

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
        peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
        stats = dict(n_events=len(events), n_particles=tot,
                     mean=float(all_m.mean()), median=float(np.median(all_m)),
                     peak=float(peak), std=float(all_m.std()))
        return all_m, all_m2, all_beta2, stats

    sim_m, sim_m2, sim_b2, sim_s = extract_masses(simulated_events, is_generated=False)
    gen_m, gen_m2, gen_b2, gen_s = extract_masses(generated_events, is_generated=True)

    hi_q     = float(np.quantile(np.concatenate([sim_m, gen_m]), 0.999))
    fmt_mev3 = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*1e3:.1f}")

    fig1, ax1 = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bins_full = np.linspace(0, hi_q, 250)
    ax1.hist(sim_m, bins=bins_full, density=True, alpha=0.55,
             color=C_REAL, label="Simulated", log=True)
    ax1.hist(gen_m, bins=bins_full, density=True, histtype="step",
             linewidth=2.0, color=C_GEN, label="Generated", log=True)
    ax1.axvline(m_e, color=C_MREF, linestyle="--", linewidth=1.6,
                label=rf"$m_e$ = {m_e*1e3:.6f} MeV")
    ax1.axvline(sim_s["peak"], color=C_REAL, linestyle=":", linewidth=1.4, alpha=0.8,
                label=f"Sim peak = {sim_s['peak']*1e3:.6f} MeV")
    ax1.axvline(gen_s["peak"], color=C_GEN,  linestyle=":", linewidth=1.4, alpha=0.8,
                label=f"Gen peak = {gen_s['peak']*1e3:.6f} MeV")
    ax1.set_xlabel("Mass $m$ [GeV]")
    ax1.set_ylabel("Density (log scale)")
    ax1.set_title("Per-particle mass distribution")
    ax1.legend(loc="upper center")
    savefig(fig1, os.path.join(outdir, "mass_full.png"))

    zoom_configs = [
        ("Slight zoom (0–2 MeV)",        0.0,     0.002,   200, fmt_mev3, "mass_slight_zoom.png"),
        ("Tight zoom (around $m_e$)",     0.00045, 0.00057, 150, fmt_mev3, "mass_tight_zoom.png"),
    ]
    for sub_title, zlo, zhi, nbins, x_fmt, fname in zoom_configs:
        bins_z = np.linspace(zlo, zhi, nbins)
        r_cnt, _ = np.histogram(sim_m, bins=bins_z, density=False)
        g_cnt, _ = np.histogram(gen_m, bins=bins_z, density=False)
        ctrs = 0.5 * (bins_z[:-1] + bins_z[1:])
        bw   = bins_z[1] - bins_z[0]
        r_dens = r_cnt / (r_cnt.sum() * bw + 1e-30)
        g_dens = g_cnt / (g_cnt.sum() * bw + 1e-30)

        fig2 = plt.figure(figsize=(9, 7))
        gs2  = fig2.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax_h = fig2.add_subplot(gs2[0])
        ax_r = fig2.add_subplot(gs2[1], sharex=ax_h)
        ax_h.tick_params(labelbottom=False)
        ax_h.bar(ctrs, r_dens, width=bw, alpha=0.55, color=C_REAL, label="Simulated")
        ax_h.step(np.append(bins_z[:-1], bins_z[-1]),
                  np.append(g_dens, g_dens[-1]),
                  where="post", linewidth=2.0, color=C_GEN, label="Generated")
        ax_h.axvline(m_e, color=C_MREF, linestyle="--", linewidth=1.8,
                     label=rf"$m_e$ = {m_e*1e3:.3f} MeV")
        ax_h.axvline(sim_s["peak"], color=C_REAL, linestyle=":", linewidth=1.5, alpha=0.8,
                     label=f"Sim peak = {sim_s['peak']*1e3:.3f} MeV")
        ax_h.axvline(gen_s["peak"], color=C_GEN,  linestyle=":", linewidth=1.5, alpha=0.8,
                     label=f"Gen peak = {gen_s['peak']*1e3:.3f} MeV")
        ax_h.set_ylabel("Density")
        ax_h.set_title(rf"Mass near $m_e$ — {sub_title}")
        ax_h.legend(loc="upper right", fontsize=18)
        ax_h.set_xlim(zlo, zhi)
        mask = r_cnt >= 5
        r_p  = r_cnt / max(r_cnt.sum(), 1)
        g_p  = g_cnt / max(g_cnt.sum(), 1)
        frac = np.where(mask, (g_p - r_p) / np.where(r_p > 0, r_p, np.nan), np.nan)
        ax_r.axhline(0, color="#1a1a1a", linewidth=1.2)
        ax_r.axhspan(-0.1, 0.1, color="#AAAAAA", alpha=0.18)
        ax_r.plot(ctrs[mask], frac[mask], "o", markersize=3.5, color=C_GEN)
        ax_r.set_ylim(-1, 1)
        ax_r.set_ylabel("Fract. diff.", fontsize=17)
        ax_r.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_h.xaxis.set_major_formatter(x_fmt)
        ax_r.xaxis.set_major_formatter(x_fmt)
        ax_r.set_xlabel("Mass [MeV]")
        savefig(fig2, os.path.join(outdir, fname))


# ── Charge asymmetry ──────────────────────────────────────────────────────────
def plot_charge_asymmetry(generated_events, outdir):
    charge_diff = []
    total_mult  = []
    for ev in generated_events:
        ev = np.asarray(ev)
        pdgs = ev[:, 0]
        n_minus = np.sum(pdgs == 11)
        n_plus  = np.sum(pdgs == -11)
        charge_diff.append(n_minus - n_plus)
        total_mult.append(n_minus + n_plus)

    charge_diff = np.array(charge_diff)
    total_mult  = np.array(total_mult)
    avg_mult = total_mult.mean()
    mu       = charge_diff.mean()
    sigma    = charge_diff.std()

    unique_vals = np.unique(charge_diff)
    bins = np.append(unique_vals - 0.5, unique_vals[-1] + 0.5)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.hist(charge_diff, bins=bins, rwidth=1.0, edgecolor="black",
            color=C_GEN, alpha=0.75)
    legend_text = (rf"$\langle N \rangle = {avg_mult:.2f}$" "\n"
                   rf"$\mu = {mu:.3f}$" "\n"
                   rf"$\sigma = {sigma:.3f}$")
    dummy = Line2D([], [], linestyle="none")
    ax.legend([dummy], [legend_text], fontsize=19)
    ax.set_xlabel(r"$N_{e^-} - N_{e^+}$", fontsize=20)
    ax.set_ylabel("Number of events", fontsize=20)
    ax.set_title("Charge Asymmetry Per Event — Generated", fontsize=22)
    savefig(fig, os.path.join(outdir, "charge_asymmetry.png"))


# ── Main evaluate function ────────────────────────────────────────────────────
try:
    from particle_diffusion_new_data_discretepdgconditioning import (
        load_events, extract_species, sanitize_event,
        plot_corner_overlay, CFG,
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
    n_real = len(real_events)
    n_gen  = len(gen_events)
    print(f"Loaded {n_real:,} real  /  {n_gen:,} generated events")

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

        # 1) Multiplicity
        plot_multiplicity(
            real_sp["mult"], gen_sp["mult"],
            outpath=os.path.join(args.outdir, f"multiplicity_{tag}.png"),
            species_name=sp["name"], n_real=n_real, n_gen=n_gen,
            bins=getattr(args, "mult_bins", 50),
        )

        # 2) Energy
        e_key = "E_signed"
        if real_sp[e_key].size and gen_sp[e_key].size:
            plot_energy(
                real_sp[e_key], gen_sp[e_key],
                outpath=os.path.join(args.outdir, f"energy_{tag}.png"),
                species_name=sp["name"],
                bins=getattr(args, "mom_bins", 80),
                ratio_min_count=getattr(args, "ratio_min_count", 10),
                n_real=n_real, n_gen=n_gen,
            )

        # 2b) Energy recomputed from proper electron mass
        gen_beta_arr = np.column_stack([
            gen_sp["betax"], gen_sp["betay"], gen_sp["betaz"]
        ]) if (gen_sp["betax"].size and gen_sp["betay"].size
               and gen_sp["betaz"].size) else None
        if real_sp[e_key].size and gen_beta_arr is not None and gen_sp[e_key].size:
            plot_energy_from_mass(
                real_sp[e_key], gen_beta_arr, gen_sp[e_key],
                outpath=os.path.join(args.outdir, f"energy_from_mass_{tag}.png"),
                species_name=sp["name"], me=cfg.me,
                bins=getattr(args, "mom_bins", 80),
                ratio_min_count=getattr(args, "ratio_min_count", 10),
                n_real=n_real, n_gen=n_gen,
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
                    bins=getattr(args, "mom_bins", 80),
                    ratio_min_count=getattr(args, "ratio_min_count", 10),
                    fixed_range=(-1, 1),
                )

        # 4) Momentum grouped
        if all(real_sp[k].size and gen_sp[k].size for k in ("px", "py", "pz")):
            three_panel_three_zoom(
                real_sp, gen_sp,
                keys=["px", "py", "pz"],
                xlabels=[r"$p_x$ [GeV]", r"$p_y$ [GeV]", r"$p_z$ [GeV]"],
                outpath=os.path.join(args.outdir, f"p_xyz_{tag}.png"),
                title="Momentum components", species_name=sp["name"],
                n_real=n_real, n_gen=n_gen,
                bins=getattr(args, "mom_bins", 80),
            )

        # 5) Position components
        for key, xlabel in [("x", "x [nm]"), ("y", "y [nm]"), ("z", "z [nm]")]:
            if real_sp[key].size and gen_sp[key].size:
                plot_single_three_zoom(
                    real_sp[key], gen_sp[key],
                    outpath=os.path.join(args.outdir, f"pos_{key}_{tag}.png"),
                    xlabel=xlabel, title=f"Position {key}",
                    species_name=sp["name"], n_real=n_real, n_gen=n_gen,
                    bins=getattr(args, "mom_bins", 80),
                    ratio_min_count=getattr(args, "ratio_min_count", 10),
                )

        # 6) Corner plots
        corner_sets = [
            (["px", "py", "pz"],
             [r"$p_x$", r"$p_y$", r"$p_z$"], r"$p_x, p_y, p_z$"),
            (["betax", "betay", "betaz"],
             [r"$\beta_x$", r"$\beta_y$", r"$\beta_z$"],
             r"$\beta_x, \beta_y, \beta_z$"),
        ]
        if real_sp["x"].size and gen_sp["x"].size:
            corner_sets.append((["x", "y", "z"], ["x", "y", "z"], "x, y, z"))
        try:
            for keys, labels, title_vars in corner_sets:
                if any(real_sp[k].size == 0 or gen_sp[k].size == 0 for k in keys):
                    continue
                plot_corner_overlay(
                    real_dict=real_sp, gen_dict=gen_sp,
                    keys=keys, labels=labels,
                    outpath=os.path.join(args.outdir,
                                         f"corner_{'_'.join(keys)}_{tag}.png"),
                    title=(f"Corner: {title_vars} — {sp['name']} | "
                           f"Simulated: {n_real:,}, Generated: {n_gen:,}"),
                    max_points=30000, q_lo=0.01, q_hi=0.99, seed=cfg.seed,
                )
        except Exception as exc:
            print(f"  corner plots skipped: {exc}")

    # ── Mass distribution ─────────────────────────────────────────────────────
    print("\n── Mass distribution ──")
    plot_mass(real_events, gen_events, outdir=args.outdir)

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
        bins=getattr(args, "mom_bins", 80),
        ratio_min_count=getattr(args, "ratio_min_count", 10),
    )

    # ── Per-event summed momentum comparison ──────────────────────────────────
    print("\n── Per-event summed momentum ──")
    plot_momentum_comparison(
        real_events, gen_events,
        outdir=args.outdir, n_real=n_real, n_gen=n_gen,
        bins=getattr(args, "mom_bins", 80),
        ratio_min_count=getattr(args, "ratio_min_count", 10),
    )

    print(f"\nDone. Plots saved to  {args.outdir}/")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combined particle evaluation plots")
    parser.add_argument("--real_path",  required=True, help="Path to real events .npy")
    parser.add_argument("--gen_path",   required=True, help="Path to generated events .npy")
    parser.add_argument("--outdir",     default=None,  help="Output directory for plots")
    parser.add_argument("--mom_bins",   type=int, default=80)
    parser.add_argument("--mult_bins",  type=int, default=50)
    parser.add_argument("--ratio_min_count", type=int, default=10)
    args = parser.parse_args()

    setup_style()
    evaluate(args)

# Alias so particle_diffusion.py can import either name
evaluate_improved = evaluate