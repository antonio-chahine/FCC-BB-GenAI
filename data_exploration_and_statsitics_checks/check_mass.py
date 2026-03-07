#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config: column indices in each particle row
# real-style: [E_signed, bx, by, bz, x, y, z]
E_I, BX_I, BY_I, BZ_I = 0, 1, 2, 3

# If your generated file still has PDG in col 0, set GEN_HAS_PDG=True and
# set indices accordingly (e.g. [pdg, E, bx, by, bz, x, y, z] -> E=1, bx=2, by=3, bz=4)
GEN_HAS_PDG = True
if GEN_HAS_PDG:
    E_I, BX_I, BY_I, BZ_I = 1, 2, 3, 4


def load_events(path: str) -> np.ndarray:
    assert os.path.exists(path), f"Missing: {path}"
    return np.load(path, allow_pickle=True)


def iter_particles(events_obj: np.ndarray):
    """Yield particle arrays (n_particles, n_features) event by event."""
    for ev in events_obj:
        p = np.asarray(ev)
        if p.size == 0 or p.ndim != 2:
            continue
        yield p


def particle_invariant_mass(p: np.ndarray, e_i: int, bx_i: int, by_i: int, bz_i: int):
    """
    Returns arrays for one event:
      Eabs, beta2, m2, m, physical_mask
    """
    E = p[:, e_i].astype(np.float64, copy=False)
    bx = p[:, bx_i].astype(np.float64, copy=False)
    by = p[:, by_i].astype(np.float64, copy=False)
    bz = p[:, bz_i].astype(np.float64, copy=False)

    Eabs = np.abs(E)
    beta2 = bx*bx + by*by + bz*bz

    # physical if beta^2 < 1 (allow tiny numerical overshoot)
    physical = beta2 < (1.0 + 1e-12)

    # m^2 = E^2 (1 - beta^2); clip negative due to rounding / unphysical beta
    one_minus = 1.0 - beta2
    m2 = (Eabs * Eabs) * one_minus
    m2_clipped = np.maximum(m2, 0.0)
    m = np.sqrt(m2_clipped)

    return Eabs, beta2, m2, m, physical


def compute_mass_stats(events_obj: np.ndarray, e_i: int, bx_i: int, by_i: int, bz_i: int):
    all_m = []
    all_m2 = []
    all_beta2 = []
    n_tot = 0
    n_phys = 0

    for p in iter_particles(events_obj):
        Eabs, beta2, m2, m, physical = particle_invariant_mass(p, e_i, bx_i, by_i, bz_i)
        n = len(Eabs)
        n_tot += n
        n_phys += int(np.sum(physical))

        all_beta2.append(beta2)
        all_m2.append(m2)
        all_m.append(m)

    if n_tot == 0:
        raise RuntimeError("No particles found in file (events empty or wrong format).")

    all_m = np.concatenate(all_m)
    all_m2 = np.concatenate(all_m2)
    all_beta2 = np.concatenate(all_beta2)

    frac_phys = n_phys / n_tot
    return {
        "m": all_m,
        "m2": all_m2,
        "beta2": all_beta2,
        "n_particles": n_tot,
        "frac_beta2_lt_1": frac_phys,
    }


def summarize(arr: np.ndarray, name: str):
    arr = np.asarray(arr)
    qs = np.quantile(arr, [0.001, 0.01, 0.5, 0.99, 0.999])
    print(f"{name}: N={len(arr)} mean={arr.mean():.6g} std={arr.std():.6g} min={arr.min():.6g} max={arr.max():.6g}")
    print(f"  quantiles 0.1/1/50/99/99.9% = {qs}")


def main():
    gen_path = "/work/submit/anton100/msci-project/FCC-BB-GenAI/new_10/generated_events.npy"
    gen = load_events(gen_path)

    print("Loaded gen:", gen.shape, gen.dtype)
    first = np.asarray(gen[0])
    print("Example gen[0] shape:", first.shape)
    print("First particle row:", first[0])

    stats = compute_mass_stats(gen, E_I, BX_I, BY_I, BZ_I)
    print("\n=== Particle-level invariant mass check ===")
    print("Total particles:", stats["n_particles"])
    print("Fraction with beta^2 < 1:", stats["frac_beta2_lt_1"])

    summarize(stats["beta2"], "beta^2")
    summarize(stats["m2"], "m^2")
    summarize(stats["m"], "m")

    # Optional plots
    # (If you don't want plots on batch nodes, comment this block out.)
    plt.figure()
    plt.hist(stats["beta2"], bins=200, density=True)
    plt.xlabel(r"$\beta^2$")
    plt.ylabel("Density")
    plt.title("Generated: beta^2 distribution")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("gen_beta2.png", dpi=200)

    plt.figure()
    # m2 can be negative if unphysical beta; show clipped for visibility
    plt.hist(np.maximum(stats["m2"], 0.0), bins=200, density=True)
    plt.xlabel(r"$m^2$ (clipped at 0)")
    plt.ylabel("Density")
    plt.title("Generated: invariant mass-squared")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("gen_m2.png", dpi=200)

    plt.figure()
    plt.hist(stats["m"], bins=200, density=True)
    plt.xlabel(r"$m$")
    plt.ylabel("Density")
    plt.title("Generated: invariant mass")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("gen_m.png", dpi=200)

    print("\nSaved plots: gen_beta2.png, gen_m2.png, gen_m.png")


if __name__ == "__main__":
    main()