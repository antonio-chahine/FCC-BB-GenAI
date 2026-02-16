import numpy as np
import matplotlib.pyplot as plt

# Load events
events = np.load("mc_gen1.npy", allow_pickle=True)

n_events = 0
n_equal = 0
diffs = []

for ev in events:
    if ev is None or len(ev) == 0:
        continue

    pdg = ev[:, 0].astype(int)
    n_minus = np.sum(pdg == 11)
    n_plus  = np.sum(pdg == -11)

    diffs.append(n_minus - n_plus)
    n_events += 1

    if n_minus == n_plus:
        n_equal += 1

diffs = np.array(diffs)

# Print summary
print(f"Total events checked: {n_events}")
print(f"Events with N(e-) == N(e+): {n_equal}")
print(f"Fraction equal: {n_equal / n_events:.6f}")
print()
print("Asymmetry statistics (N(e-) − N(e+)):")
print(f"  Mean = {diffs.mean():.4f}")
print(f"  Std  = {diffs.std():.4f}")
print(f"  Min / Max = {diffs.min()} / {diffs.max()}")

# Plot asymmetry distribution
plt.figure(figsize=(6, 4))
plt.hist(diffs, bins=50)
plt.xlabel("N(e−) − N(e+)")
plt.ylabel("Events")
plt.title("Charge asymmetry per event")
plt.tight_layout()
plt.savefig("asymmetry_distribution.png")
plt.show()
