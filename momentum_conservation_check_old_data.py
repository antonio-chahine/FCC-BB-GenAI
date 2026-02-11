import numpy as np
import matplotlib.pyplot as plt

events = np.load("mc_gen1.npy", allow_pickle=True)

# Vectorised-ish: build arrays of sums
sum_p = np.array([ev[:, 1:4].sum(axis=0) for ev in events])   # shape (N,3)
sum_px, sum_py, sum_pz = sum_p.T
pT = np.sqrt(sum_px**2 + sum_py**2)
p_mag = np.sqrt(sum_px**2 + sum_py**2 + sum_pz**2)

def summarize(x, name):
    print(f"\n{name}:")
    print(f"  mean = {x.mean():.4f}, std = {x.std():.4f}")
    print(f"  min  = {x.min():.4f}, max = {x.max():.4f}")
    print(f"  1%/50%/99% = {np.percentile(x,[1,50,99])}")

summarize(sum_px, "sum_px")
summarize(sum_py, "sum_py")
summarize(sum_pz, "sum_pz")
summarize(pT, "pT = sqrt(sum_px^2+sum_py^2)")
summarize(p_mag, "|sum_p|")

# ---- Plots: distributions ----
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0,0].hist(sum_px, bins=100)
axs[0,0].set_title("sum_px")

axs[0,1].hist(sum_py, bins=100)
axs[0,1].set_title("sum_py")

axs[1,0].hist(sum_pz, bins=100)
axs[1,0].set_title("sum_pz")

axs[1,1].hist(p_mag, bins=100)
axs[1,1].set_title("|sum_p|")

for ax in axs.ravel():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---- Identify worst events ----
k = 10
worst = np.argsort(p_mag)[-k:][::-1]
print("\nWorst events by |sum_p|:")
for idx in worst:
    print(f"  i={idx:6d}  |sum_p|={p_mag[idx]:9.3f}  (px,py,pz)=({sum_px[idx]:.3f},{sum_py[idx]:.3f},{sum_pz[idx]:.3f})  mult={len(events[idx])}")
