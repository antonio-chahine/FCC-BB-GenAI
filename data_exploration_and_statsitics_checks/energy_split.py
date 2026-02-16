import numpy as np
events = np.load("guineapig_raw_trimmed_1/generated_events.npy", allow_pickle=True)

n_zero = 0
n_tot = 0
for ev in events:
    E = ev[:,0]
    n_zero += np.sum(np.isclose(E, 0.0))
    n_tot += len(E)

print("Fraction E_signed == 0:", n_zero / n_tot)
