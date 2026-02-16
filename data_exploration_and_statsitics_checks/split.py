#!/usr/bin/env python3
import numpy as np

PATH = "guineapig_raw_trimmed.npy"   # your saved object array
HAS_PROCESS_COL = False             # set True if you saved col7=process (i.e. d[:, :8])

events = np.load(PATH, allow_pickle=True)

deltas = []
negs = []
npos = []
ntot = []
even_mult = []
all_e_sign = []

# if you saved process, this will store per-process stats
by_proc = {1: [], 2: [], 3: []}

for ev in events:
    if ev is None:
        continue
    ev = np.asarray(ev)
    if ev.ndim != 2 or ev.shape[1] < 7:
        continue

    E = ev[:, 0]
    n_em = int(np.sum(E > 0))   # electrons
    n_ep = int(np.sum(E < 0))   # positrons
    n    = int(len(E))

    deltas.append(n_em - n_ep)
    negs.append(n_ep)
    npos.append(n_em)
    ntot.append(n)
    even_mult.append((n % 2) == 0)

    all_e_sign.append(E)

    if HAS_PROCESS_COL and ev.shape[1] >= 8:
        proc = ev[:, 7].astype(int)
        for p in (1, 2, 3):
            sel = (proc == p)
            if np.any(sel):
                Ep = E[sel]
                nem_p = int(np.sum(Ep > 0))
                nep_p = int(np.sum(Ep < 0))
                by_proc[p].append(nem_p - nep_p)

deltas = np.asarray(deltas, dtype=int)
negs   = np.asarray(negs, dtype=int)
npos   = np.asarray(npos, dtype=int)
ntot   = np.asarray(ntot, dtype=int)
even_mult = np.asarray(even_mult, dtype=bool)

E_all = np.concatenate(all_e_sign) if len(all_e_sign) else np.array([])

print("==== Dataset symmetry check ====")
print("N events used:", len(deltas))
print("Total particles:", int(ntot.sum()))
print()

# overall charge balance
total_em = int(npos.sum())
total_ep = int(negs.sum())
print("Total e-:", total_em)
print("Total e+:", total_ep)
print("Total (e- - e+):", total_em - total_ep)
print("Fraction e-:", total_em / max(total_em + total_ep, 1))
print()

# event-by-event symmetry
print("Events with Ne- == Ne+:", int(np.sum(deltas == 0)), f"({np.mean(deltas==0)*100:.2f}%)")
print("Mean(Ne- - Ne+):", float(np.mean(deltas)))
print("Std(Ne- - Ne+):", float(np.std(deltas)))
print("Max |Ne- - Ne+|:", int(np.max(np.abs(deltas))) if len(deltas) else 0)
print()

# even multiplicity
print("Events with even total multiplicity:", int(np.sum(even_mult)), f"({np.mean(even_mult)*100:.2f}%)")
print("Events with odd total multiplicity:", int(np.sum(~even_mult)), f"({np.mean(~even_mult)*100:.2f}%)")
print()

# small histogram of delta values around 0
if len(deltas):
    # show most common delta values
    vals, counts = np.unique(deltas, return_counts=True)
    order = np.argsort(-counts)
    print("Most common (Ne- - Ne+) values:")
    for v, c in zip(vals[order][:15], counts[order][:15]):
        print(f"  {v:>4d}: {c} ({c/len(deltas)*100:.2f}%)")
    print()

# energy sign sanity
if E_all.size:
    print("Energy sign sanity:")
    print("  % E>0:", float(np.mean(E_all > 0)) * 100)
    print("  % E<0:", float(np.mean(E_all < 0)) * 100)
    print("  % E==0:", float(np.mean(E_all == 0)) * 100)
    print()

# per-process symmetry (if available)
if HAS_PROCESS_COL:
    print("Per-process (Ne- - Ne+) summary:")
    for p in (1, 2, 3):
        arr = np.asarray(by_proc[p], dtype=int)
        if arr.size == 0:
            print(f"  process {p}: no entries")
            continue
        print(f"  process {p}: N={arr.size}  frac(delta=0)={np.mean(arr==0)*100:.2f}%  mean={np.mean(arr):.3f}  std={np.std(arr):.3f}  max|.|={np.max(np.abs(arr))}")
