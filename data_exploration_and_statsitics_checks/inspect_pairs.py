#!/usr/bin/env python3
"""
Inspect a single .pairs file to understand the e-/e+ column layout
and check what opening angles actually look like.
"""

import numpy as np
import sys

FILE = (
    "/ceph/submit/data/group/fcc/ee/beam_backgrounds/guineapig/"
    "FCCee_Z_GHC_V25p3_4_FCCee_Z256_2T_grids8/output0_999833.pairs"
)
if len(sys.argv) > 1:
    FILE = sys.argv[1]

d = np.loadtxt(FILE, dtype=np.float64)
print(f"File: {FILE}")
print(f"Shape: {d.shape}  ({d.shape[0]} particles, {d.shape[1]} columns)\n")

# ── Column layout guess ───────────────────────────────────────────────────────
# Based on GuineaPig .pairs format:
# col 0: E (signed: +ve = e-, -ve = e+)  [GeV]
# col 1: βx
# col 2: βy
# col 3: βz
# col 4: x [nm]
# col 5: y [nm]
# col 6: z [nm]
# col 7: process
# col 8,9,10: extra (possibly weights/indices)

E    = d[:, 0]
betax = d[:, 1]
betay = d[:, 2]
betaz = d[:, 3]

mask_eminus = E > 0   # e-
mask_eplus  = E < 0   # e+

eminus = d[mask_eminus]
eplus  = d[mask_eplus]

print(f"N e-  (E > 0): {mask_eminus.sum()}")
print(f"N e+  (E < 0): {mask_eplus.sum()}\n")

# ── Show first 5 of each ─────────────────────────────────────────────────────
print("First 5 e-  [E, βx, βy, βz]:")
for row in eminus[:5]:
    print(f"  E={row[0]:+.4f}  βx={row[1]:+.3e}  βy={row[2]:+.3e}  βz={row[3]:+.6f}")

print("\nFirst 5 e+  [E, βx, βy, βz]:")
for row in eplus[:5]:
    print(f"  E={row[0]:+.4f}  βx={row[1]:+.3e}  βy={row[2]:+.3e}  βz={row[3]:+.6f}")

# ── Compute opening angles between ALL e-/e+ pairs ───────────────────────────
def compute_p(rows):
    E_abs = np.abs(rows[:, 0])
    beta  = rows[:, 1:4]
    return E_abs[:, None] * beta   # p = |E| * β

p_e  = compute_p(eminus)
p_ep = compute_p(eplus)

# Normalise
def normalise(p):
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    return p[valid] / norms[valid], valid

p_e_hat,  valid_e  = normalise(p_e)
p_ep_hat, valid_ep = normalise(p_ep)

# Pairwise dot products → angles
dots = np.clip(p_e_hat @ p_ep_hat.T, -1.0, 1.0)   # (N_e, N_ep)
angles_deg = np.degrees(np.arccos(dots))

# For each electron: closest positron angle
min_angles = angles_deg.min(axis=1)

print(f"\n── Opening angles (3D, closest e+ per e-) ──")
print(f"  min  : {min_angles.min():.4f}°")
print(f"  max  : {min_angles.max():.4f}°")
print(f"  mean : {min_angles.mean():.4f}°")
print(f"  median: {np.median(min_angles):.4f}°")

print(f"\nFirst 10 closest-positron opening angles:")
for i, a in enumerate(min_angles[:10]):
    j = np.argmin(angles_deg[i])
    print(f"  e-[{i}] βz={eminus[i,3]:+.4f}  →  closest e+[{j}] βz={eplus[j,3]:+.4f}  θ={a:.4f}°")

# ── Also check: do pairs tend to have same or opposite βz sign? ──────────────
print(f"\n── βz sign statistics ──")
print(f"  e-  βz > 0: {(eminus[:,3] > 0).sum()}  |  βz < 0: {(eminus[:,3] < 0).sum()}")
print(f"  e+  βz > 0: {(eplus[:,3]  > 0).sum()}  |  βz < 0: {(eplus[:,3]  < 0).sum()}")

# ── What if we use only transverse momentum? ──────────────────────────────────
p_e_T  = compute_p(eminus)[:, :2]   # just px, py
p_ep_T = compute_p(eplus)[:,  :2]

def normalise2d(p):
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    valid = norms[:, 0] > 0
    return p[valid] / norms[valid], valid

p_e_T_hat,  _ = normalise2d(p_e_T)
p_ep_T_hat, _ = normalise2d(p_ep_T)

dots_T = np.clip(p_e_T_hat @ p_ep_T_hat.T, -1.0, 1.0)
angles_T_deg = np.degrees(np.arccos(dots_T))
min_T = angles_T_deg.min(axis=1)

print(f"\n── Transverse-only opening angles (closest e+ per e-) ──")
print(f"  min  : {min_T.min():.4f}°")
print(f"  max  : {min_T.max():.4f}°")
print(f"  mean : {min_T.mean():.4f}°")
print(f"  median: {np.median(min_T):.4f}°")