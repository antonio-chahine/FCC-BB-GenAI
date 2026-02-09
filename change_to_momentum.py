#!/usr/bin/env python3
import argparse
import numpy as np
from tqdm import trange

ME_GEV = 0.000511  # electron mass

def convert_event_raw7_to_mom4(ev, swap_charge=False):
    """
    ev: (K,7) [E_signed, betax, betay, betaz, x, y, z]
    returns: (K,4) [pdg, px, py, pz]
    """
    ev = np.asarray(ev)
    if ev.ndim != 2 or ev.shape[1] < 7 or ev.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    E_signed = ev[:, 0].astype(np.float64)
    beta     = ev[:, 1:4].astype(np.float64)

    # PDG from sign(E)
    pdg = np.where(E_signed > 0, 11, -11).astype(np.int64)
    if swap_charge:
        pdg = -pdg

    Eabs = np.abs(E_signed)

    # (optional but recommended) enforce physical beta
    beta = np.clip(beta, -0.999999, 0.999999)
    bmag = np.linalg.norm(beta, axis=1, keepdims=True)
    beta = beta / np.maximum(bmag, 1.0)

    # Momentum: p = gamma*m*beta = (|E|/m)*m*beta = |E| * beta
    # (equivalent to using gamma=E/m and m=ME_GEV)
    p = (Eabs[:, None] * beta).astype(np.float64)  # GeV

    out = np.zeros((ev.shape[0], 4), dtype=np.float32)
    out[:, 0] = pdg.astype(np.float32)
    out[:, 1:4] = p.astype(np.float32)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input .npy (object array of (K,7) events)")
    ap.add_argument("--out_path", required=True, help="Output .npy (object array of (K,4) events)")
    ap.add_argument("--swap_charge", action="store_true",
                    help="Use this if your file convention has E>0 as e+ instead of e-")
    args = ap.parse_args()

    arr = np.load(args.in_path, allow_pickle=True)
    events = list(arr) if (isinstance(arr, np.ndarray) and arr.dtype == object) else [arr[i] for i in range(arr.shape[0])]

    out_events = []
    for i in trange(len(events), desc="Converting"):
        out_events.append(convert_event_raw7_to_mom4(events[i], swap_charge=args.swap_charge))

    np.save(args.out_path, np.array(out_events, dtype=object))
    print("Saved:", args.out_path)
    print("Example shapes:", events[0].shape, "->", out_events[0].shape)

if __name__ == "__main__":
    main()
