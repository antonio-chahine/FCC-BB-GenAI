#!/usr/bin/env python3
"""
Plot the z boundary spike in simulated GuineaPig data.
Usage:
    python investigate_z_peak.py
    python investigate_z_peak.py --data-path /your/path
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str,
                    default='/work/submit/anton100/msci-project/FCC-BB-GenAI')
args = parser.parse_args()

# ── Load and flatten z ────────────────────────────────────────────────────────
print("Loading...")
raw    = np.load(os.path.join(args.data_path, 'guineapig_raw_trimmed.npy'), allow_pickle=True)
events = list(raw) if raw.dtype == object else [raw[i] for i in range(len(raw))]

all_z = []
for ev in events:
    ev = np.asarray(ev)
    if   ev.ndim == 2 and ev.shape[1] >= 8: all_z.append(ev[:, 7])
    elif ev.ndim == 2 and ev.shape[1] >= 7: all_z.append(ev[:, 6])
all_z = np.concatenate(all_z)
print(f"  {len(events):,} events, {len(all_z):,} particles")

# ── Find boundary ─────────────────────────────────────────────────────────────
z_max    = all_z.max()
n_at_max = (all_z == z_max).sum()
n_at_min = (all_z == all_z.min()).sum()
print(f"  Boundary: {all_z.min():.0f} to {z_max:.0f} (raw units)")
print(f"  Particles at +boundary: {n_at_max:,}  ({100*n_at_max/len(all_z):.2f}%)")
print(f"  Particles at -boundary: {n_at_min:,}  ({100*n_at_min/len(all_z):.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

# ── Left: zoomed out ──────────────────────────────────────────────────────────
zoom_out = 0.02 * z_max
z_out    = all_z[all_z > (z_max - zoom_out)]
axes[0].hist(z_out, bins=200, color='steelblue', alpha=0.8,
             range=(z_max - zoom_out, z_max + zoom_out * 0.01))
axes[0].set_xlabel('z (raw units)')
axes[0].set_ylabel('Count')
axes[0].set_yscale('log')
axes[0].set_title('Zoomed out')

# ── Right: zoomed in ──────────────────────────────────────────────────────────
zoom_in    = 1
background = all_z[(all_z > z_max - zoom_in) & (all_z < z_max)]
spike_n    = n_at_max
bar_width  = 0.5   # narrow spike bar width in raw units

axes[1].hist(background, bins=30, color='steelblue', alpha=0.8,
             range=(z_max - zoom_in, z_max - 1))
axes[1].bar(z_max, spike_n, width=bar_width, color='steelblue', alpha=0.8, align='edge')
axes[1].set_xlim(z_max - zoom_in, z_max + bar_width * 3)
axes[1].set_xlabel('z (raw units)')
axes[1].set_ylabel('Count')
axes[1].set_yscale('log')
axes[1].set_title('Zoomed in')

fig.suptitle('z boundary spike (positive)', fontsize=13)
out = os.path.join(args.data_path, 'z_peak_zoom.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")