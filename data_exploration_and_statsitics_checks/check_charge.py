import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

events = np.load(
    "/work/submit/anton100/msci-project/FCC-BB-GenAI/new_14/generated_events.npy",
    allow_pickle=True
)

charge_diff = []
total_mult  = []

for ev in events:
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

plt.figure(figsize=(7,5))

plt.hist(
    charge_diff,
    bins=bins,
    rwidth=1.0,
    edgecolor="black"
)

legend_text = (
    rf"$\langle N \rangle = {avg_mult:.2f}$" "\n"
    rf"$\mu = {mu:.3f}$" "\n"
    rf"$\sigma = {sigma:.3f}$"
)

dummy = Line2D([], [], linestyle='none')
plt.legend([dummy], [legend_text])

plt.xlabel(r"$N_{e^-} - N_{e^+}$")
plt.ylabel("Number of events")
plt.title("Charge Asymmetry Per Event")
plt.tight_layout()
plt.savefig("charge_asymmetry.png")
plt.show()