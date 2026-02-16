import numpy as np
import matplotlib.pyplot as plt

events = np.load("mc_gen1.npy", allow_pickle=True)

# Vectorised-ish: build arrays of sums
sum_p = np.array([ev[:, 1:4].sum(axis=0) for ev in events])   # shape (N,3)
sum_px, sum_py, sum_pz = sum_p.T

pT = np.sqrt(sum_px**2 + sum_py**2)
p_mag = np.sqrt(sum_px**2 + sum_py**2 + sum_pz**2)

# --- XY symmetry diagnostics ---
phi = np.arctan2(sum_py, sum_px)                 # [-pi, pi]
delta_xy = sum_px - sum_py
abs_delta_xy = np.abs(delta_xy)

# "ratio" only meaningful away from zero; keep a masked version
eps = 1e-9
ratio_xy = sum_px / (sum_py + np.sign(sum_py) * eps)  # avoids blow-up at exactly 0


def summarize(x, name):
    x = np.asarray(x)
    finite = np.isfinite(x)
    x = x[finite]
    print(f"\n{name}:")
    print(f"  N    = {len(x)}")
    print(f"  mean = {x.mean():.4f}, std = {x.std():.4f}")
    print(f"  min  = {x.min():.4f}, max = {x.max():.4f}")
    print(f"  1%/50%/99% = {np.percentile(x,[1,50,99])}")


# ---- Summaries ----
summarize(sum_px, "sum_px")
summarize(sum_py, "sum_py")
summarize(sum_pz, "sum_pz")
summarize(pT, "pT = sqrt(sum_px^2+sum_py^2)")
summarize(p_mag, "|sum_p|")

summarize(delta_xy, "delta_xy = sum_px - sum_py")
summarize(abs_delta_xy, "|delta_xy|")
summarize(phi, "phi = atan2(sum_py, sum_px)")

print("\nPhi moments (should be ~0 if azimuthally symmetric):")
print(f"  <cos(phi)> = {np.cos(phi).mean():.6g}")
print(f"  <sin(phi)> = {np.sin(phi).mean():.6g}")

# Quadrant fractions for (sum_px, sum_py)
q1 = np.sum((sum_px >= 0) & (sum_py >= 0))
q2 = np.sum((sum_px <  0) & (sum_py >= 0))
q3 = np.sum((sum_px <  0) & (sum_py <  0))
q4 = np.sum((sum_px >= 0) & (sum_py <  0))
qs = np.array([q1, q2, q3, q4], dtype=float)
print("\nQuadrant fractions (ideally ~0.25 each):")
if qs.sum() > 0:
    print(f"  Q1(+,+) = {qs[0]/qs.sum():.4f}")
    print(f"  Q2(-,+) = {qs[1]/qs.sum():.4f}")
    print(f"  Q3(-,-) = {qs[2]/qs.sum():.4f}")
    print(f"  Q4(+,-) = {qs[3]/qs.sum():.4f}")

# ---- Plots: distributions ----
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

axs[0,0].hist(sum_px, bins=100)
axs[0,0].set_title("sum_px")

axs[0,1].hist(sum_py, bins=100)
axs[0,1].set_title("sum_py")

axs[0,2].hist(sum_pz, bins=100)
axs[0,2].set_title("sum_pz")

axs[1,0].hist(delta_xy, bins=100)
axs[1,0].set_title("delta_xy = sum_px - sum_py")

axs[1,1].hist(phi, bins=72, range=(-np.pi, np.pi))
axs[1,1].set_title("phi = atan2(sum_py, sum_px)")

axs[1,2].hist(p_mag, bins=100)
axs[1,2].set_title("|sum_p|")

for ax in axs.ravel():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---- Identify worst events ----
k = 10

# (A) worst by total momentum non-closure
worst_pmag = np.argsort(p_mag)[-k:][::-1]
print("\nWorst events by |sum_p|:")
for idx in worst_pmag:
    print(f"  i={idx:6d}  |sum_p|={p_mag[idx]:9.3f}  (px,py,pz)=({sum_px[idx]:.3f},{sum_py[idx]:.3f},{sum_pz[idx]:.3f})  mult={len(events[idx])}")

# (B) worst by x–y imbalance
worst_xy = np.argsort(abs_delta_xy)[-k:][::-1]
print("\nWorst events by |sum_px - sum_py|:")
for idx in worst_xy:
    print(f"  i={idx:6d}  |dxy|={abs_delta_xy[idx]:9.3f}  dxy={delta_xy[idx]: .3f}  phi={phi[idx]: .3f}  (px,py)=({sum_px[idx]:.3f},{sum_py[idx]:.3f})  mult={len(events[idx])}")
