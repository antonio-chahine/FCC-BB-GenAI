#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DATAFILE = "guineapig_raw_trimmed.npy"

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
events = np.load(DATAFILE, allow_pickle=True)

n_events = len(events)
print(f"Loaded {n_events} events")

# ------------------------------------------------------------
# Event-level statistics
# ------------------------------------------------------------
multiplicities = np.array([len(ev) for ev in events])

print("\nEvent multiplicity stats")
print(f"  min   : {multiplicities.min()}")
print(f"  max   : {multiplicities.max()}")
print(f"  mean  : {multiplicities.mean():.2f}")
print(f"  median: {np.median(multiplicities)}")

# ------------------------------------------------------------
# Flatten particle-level data
# ------------------------------------------------------------
# Shape: (N_total_particles, 7)
particles = np.concatenate(events, axis=0)

print(f"\nTotal particles: {particles.shape[0]}")

px, py, pz, E, vx, vy, vz = particles.T

p = np.sqrt(px**2 + py**2 + pz**2)
pt = np.sqrt(px**2 + py**2)

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
def report_bad(name, arr):
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    print(f"{name:>3}: NaN={n_nan}, inf={n_inf}")

print("\nSanity checks")
report_bad("px", px)
report_bad("py", py)
report_bad("pz", pz)
report_bad("E ", E)
report_bad("vx", vx)
report_bad("vy", vy)
report_bad("vz", vz)

# ------------------------------------------------------------
# Distributions
# ------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
axs = axs.flatten()

axs[0].hist(multiplicities, bins=100)
axs[0].set_title("Event multiplicity")

axs[1].hist(p, bins=200, log=True)
axs[1].set_title("|p|")

axs[2].hist(pt, bins=200, log=True)
axs[2].set_title("pT")

axs[3].hist(E, bins=200, log=True)
axs[3].set_title("Energy")

axs[4].hist(vz, bins=200)
axs[4].set_title("vz")

axs[5].hist(np.sqrt(vx**2 + vy**2), bins=200)
axs[5].set_title("Transverse vertex radius")

for ax in axs:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Extreme values (useful for cuts)
# ------------------------------------------------------------
def summary(name, arr):
    print(
        f"{name:>3}: "
        f"min={arr.min():.3e}, "
        f"max={arr.max():.3e}, "
        f"99%={np.percentile(arr, 99):.3e}"
    )

print("\nExtremes / percentiles")
summary("|p|", p)
summary("pT", pt)
summary("E ", E)
summary("vz", vz)
summary("rT", np.sqrt(vx**2 + vy**2))
