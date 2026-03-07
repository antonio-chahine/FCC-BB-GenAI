import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--real',  default='/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy')
parser.add_argument('--gen',   default='/work/submit/anton100/msci-project/FCC-BB-GenAI/new_25_before/generated_events.npy')
parser.add_argument('--outdir', default='.')
args = parser.parse_args()

# ============================================================
# Load
# ============================================================
print(f"Loading real data:      {args.real}")
real_events = np.load(args.real, allow_pickle=True)
print(f"Loading generated data: {args.gen}")
gen_events  = np.load(args.gen,  allow_pickle=True)

# ============================================================
# Momentum extraction  (correct column layout per format)
# ============================================================

def momenta_real(ev):
    """Real format: [E_signed, betax, betay, betaz, x, y, z]
    px = |E| * betax  etc."""
    E    = np.abs(ev[:, 0])
    beta = ev[:, 1:4]
    return (E[:, None] * beta).sum(axis=0)   # (3,)

def momenta_gen(ev):
    """Generated format: [pdg, E, betax, betay, betaz, x, y, z]
    px = E * betax  etc."""
    E    = ev[:, 1]
    beta = ev[:, 2:5]
    return (E[:, None] * beta).sum(axis=0)   # (3,)

real_sum_p = np.array([momenta_real(ev) for ev in real_events])   # (N,3)
gen_sum_p  = np.array([momenta_gen(ev)  for ev in gen_events])    # (M,3)

r_px, r_py, r_pz = real_sum_p.T
g_px, g_py, g_pz = gen_sum_p.T

r_pT    = np.sqrt(r_px**2 + r_py**2)
g_pT    = np.sqrt(g_px**2 + g_py**2)
r_pmag  = np.sqrt(r_px**2 + r_py**2 + r_pz**2)
g_pmag  = np.sqrt(g_px**2 + g_py**2 + g_pz**2)
r_phi   = np.arctan2(r_py, r_px)
g_phi   = np.arctan2(g_py, g_px)
r_dxy   = r_px - r_py
g_dxy   = g_px - g_py

# ============================================================
# Print summaries
# ============================================================

def summarize(r, g, name):
    for arr, label in [(r, "Real"), (g, "Gen ")]:
        arr = arr[np.isfinite(arr)]
        print(f"  {label} {name}: mean={arr.mean():.4f}  std={arr.std():.4f}"
              f"  1/50/99%={np.percentile(arr,[1,50,99])}")

print("\n" + "="*60)
for name, r, g in [("sum_px", r_px, g_px), ("sum_py", r_py, g_py),
                   ("sum_pz", r_pz, g_pz), ("pT",     r_pT, g_pT),
                   ("|sum_p|", r_pmag, g_pmag)]:
    print(f"\n{name}:")
    summarize(r, g, name)

print("\nPhi moments:")
for arr, label in [(r_phi, "Real"), (g_phi, "Gen ")]:
    print(f"  {label}  <cos(phi)>={np.cos(arr).mean():.6g}  <sin(phi)>={np.sin(arr).mean():.6g}")

def quadrant_fracs(px, py, label):
    q1 = np.sum((px >= 0) & (py >= 0))
    q2 = np.sum((px <  0) & (py >= 0))
    q3 = np.sum((px <  0) & (py <  0))
    q4 = np.sum((px >= 0) & (py <  0))
    qs = np.array([q1, q2, q3, q4], dtype=float)
    s  = qs.sum()
    print(f"\nQuadrant fractions ({label}):")
    print(f"  Q1(+,+)={qs[0]/s:.4f}  Q2(-,+)={qs[1]/s:.4f}"
          f"  Q3(-,-)={qs[2]/s:.4f}  Q4(+,-)={qs[3]/s:.4f}")

quadrant_fracs(r_px, r_py, "Real")
quadrant_fracs(g_px, g_py, "Gen")

# Worst events
print("\n--- Worst generated events by |sum_p| ---")
worst = np.argsort(g_pmag)[-10:][::-1]
for idx in worst:
    ev = gen_events[idx]
    print(f"  i={idx:5d}  |sum_p|={g_pmag[idx]:.3f}"
          f"  (px,py,pz)=({g_px[idx]:.3f},{g_py[idx]:.3f},{g_pz[idx]:.3f})"
          f"  mult={len(ev)}")

# ============================================================
# Plotting helpers
# ============================================================

def shared_range(a, b, q_lo=0.01, q_hi=0.99):
    both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    return float(np.quantile(both, q_lo)), float(np.quantile(both, q_hi))

def overlay_hist(ax, real, gen, xlabel, bins=80):
    lo, hi = shared_range(real, gen)
    ax.hist(real, bins=bins, range=(lo, hi), density=True,
            alpha=0.55, label="Real")
    ax.hist(gen,  bins=bins, range=(lo, hi), density=True,
            histtype="step", linewidth=1.8, label="Generated")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

# ============================================================
# Figure 1: per-component momentum distributions
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
overlay_hist(axs[0], r_px, g_px, r"$\sum p_x$ [GeV]")
overlay_hist(axs[1], r_py, g_py, r"$\sum p_y$ [GeV]")
overlay_hist(axs[2], r_pz, g_pz, r"$\sum p_z$ [GeV]")
plt.suptitle("Per-event summed momentum components: Real vs Generated")
plt.tight_layout()
plt.savefig(f"{args.outdir}/momentum_components_overlay.png", dpi=150)
print(f"\nSaved: momentum_components_overlay.png")

# ============================================================
# Figure 2: derived quantities
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(16, 9))

overlay_hist(axs[0,0], r_pT,   g_pT,   r"$p_T = \sqrt{\sum p_x^2 + \sum p_y^2}$ [GeV]")
overlay_hist(axs[0,1], r_pmag, g_pmag, r"$|\sum \mathbf{p}|$ [GeV]")
overlay_hist(axs[0,2], r_dxy,  g_dxy,  r"$\sum p_x - \sum p_y$ [GeV]")

# Phi overlaid
lo_phi, hi_phi = -np.pi, np.pi
axs[1,0].hist(r_phi, bins=72, range=(lo_phi, hi_phi), density=True,
              alpha=0.55, label="Real")
axs[1,0].hist(g_phi, bins=72, range=(lo_phi, hi_phi), density=True,
              histtype="step", linewidth=1.8, label="Generated")
axs[1,0].set_xlabel(r"$\phi = \mathrm{atan2}(\sum p_y, \sum p_x)$")
axs[1,0].set_ylabel("Density")
axs[1,0].legend(fontsize=9)
axs[1,0].grid(True, alpha=0.2)

overlay_hist(axs[1,1], r_px, g_px, r"$\sum p_x$ [GeV] (zoom)")
overlay_hist(axs[1,2], r_py, g_py, r"$\sum p_y$ [GeV] (zoom)")

plt.suptitle("Momentum diagnostics: Real vs Generated")
plt.tight_layout()
plt.savefig(f"{args.outdir}/momentum_diagnostics_overlay.png", dpi=150)
print(f"Saved: momentum_diagnostics_overlay.png")

plt.show()
