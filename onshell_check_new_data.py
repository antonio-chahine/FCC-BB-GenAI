import numpy as np
import matplotlib.pyplot as plt

events = np.load("guineapig_raw_trimmed.npy", allow_pickle=True)

# collect per-particle masses across all events
all_m = []
all_m2 = []
all_beta2 = []

neg_m2_count = 0
tot_particles = 0

for ev in events:
    E = ev[:, 0]         # (n,)
    beta = ev[:, 1:4]    # (n,3)

    beta2 = (beta**2).sum(axis=1)
    p2 = (E**2) * beta2

    m2 = E**2 - p2       # = E^2 (1 - beta^2)
    tot_particles += len(m2)

    # count negatives (numerical or inconsistent rows)
    neg_m2_count += np.sum(m2 < -1e-6)

    # clip tiny negatives to 0 for sqrt
    m2_clip = np.clip(m2, 0.0, None)
    m = np.sqrt(m2_clip)

    all_m.append(m)
    all_m2.append(m2)
    all_beta2.append(beta2)

all_m = np.concatenate(all_m)
all_m2 = np.concatenate(all_m2)
all_beta2 = np.concatenate(all_beta2)

print("Total particles:", tot_particles)
print("Neg m^2 count (m2 < -1e-6):", int(neg_m2_count))
print("beta^2: min/mean/max =", all_beta2.min(), all_beta2.mean(), all_beta2.max())
print("m^2:    1%/50%/99%   =", np.percentile(all_m2, [1,50,99]))
print("m:      1%/50%/99%   =", np.percentile(all_m,  [1,50,99]))

# plot mass distribution (log y is often helpful)
plt.figure(figsize=(7,4))
plt.hist(all_m, bins=200, log=True)
plt.xlabel("Per-particle mass m")
plt.ylabel("Counts (log)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mass_distribution.png")
plt.show()

# optional: zoom in near electron mass if you expect electrons
plt.figure(figsize=(7,4))
plt.hist(all_m, bins=200, range=(0, 0.01), log=True)
plt.xlabel("m (zoom near 0–0.01)")
plt.ylabel("Counts (log)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mass_distribution_zoom.png")
plt.show()

m_true = 0.000511
rel_err = (all_m - m_true) / m_true

plt.hist(rel_err, bins=200)
plt.xlim(-0.1, 0.1)
plt.title("Relative mass error")
plt.show()

print("std rel err:", rel_err.std())

