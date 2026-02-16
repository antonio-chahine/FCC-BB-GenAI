#!/usr/bin/env python3
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from podio import root_io

# ----------------------------
# SETTINGS
# ----------------------------
FILES = glob.glob(
    "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"
)

MAX_FILES  = 20        # keep small while testing
MAX_EVENTS = 200       # per file cap (keeps runtime sane)
OUT_PREFIX = "genstatus_compare"   # outputs genstatus_compare_*.png

# Which generatorStatus values to compare
STATUSES = [0, 1]

# ----------------------------
# Helpers
# ----------------------------
def safe_eta(px, py, pz):
    p = math.sqrt(px*px + py*py + pz*pz)
    # eta = 0.5 ln((p+pz)/(p-pz))
    # guard p==|pz| (rare but can happen numerically)
    denom = p - pz
    numer = p + pz
    if denom <= 0 or numer <= 0:
        return None
    return 0.5 * math.log(numer / denom)

def to_np(a):
    return np.array(a, dtype=np.float64)

# ----------------------------
# Accumulators
# ----------------------------
# per status: list of per-event multiplicities
mult = {s: [] for s in STATUSES}

# per status: pooled kinematics
pool = {s: defaultdict(list) for s in STATUSES}
# keys we'll fill: "p", "pt", "px", "py", "pz", "eta", "pdg"
# Note: storing pdg as int, others float

# quick counters
status_value_counter = Counter()
events_seen = 0

# ----------------------------
# Loop
# ----------------------------
for fi, f in enumerate(FILES[:MAX_FILES]):
    print("Reading", f)
    reader = root_io.Reader(f)
    events = reader.get("events")

    for ei, event in enumerate(events):
        if ei >= MAX_EVENTS:
            break

        if "MCParticles" not in event.getAvailableCollections():
            continue

        mcp = event.get("MCParticles")

        # count per-event by status
        per_event_count = {s: 0 for s in STATUSES}

        for p in mcp:
            gs = int(p.getGeneratorStatus())
            status_value_counter[gs] += 1

            if gs not in STATUSES:
                continue

            mom = p.getMomentum()
            px, py, pz = float(mom.x), float(mom.y), float(mom.z)

            pt = math.sqrt(px*px + py*py)
            pp = math.sqrt(px*px + py*py + pz*pz)
            eta = safe_eta(px, py, pz)

            pool[gs]["px"].append(px)
            pool[gs]["py"].append(py)
            pool[gs]["pz"].append(pz)
            pool[gs]["pt"].append(pt)
            pool[gs]["p"].append(pp)
            if eta is not None:
                pool[gs]["eta"].append(eta)
            pool[gs]["pdg"].append(int(p.getPDG()))

            per_event_count[gs] += 1

        for s in STATUSES:
            mult[s].append(per_event_count[s])

        events_seen += 1

print("\n=== Done ===")
print("Events seen:", events_seen)
print("GeneratorStatus values seen (global counts):")
print(status_value_counter)

# ----------------------------
# Print multiplicity summaries
# ----------------------------
def summarize(arr, name):
    a = to_np(arr)
    print(
        f"{name}: mean={a.mean():.2f}, std={a.std():.2f}, "
        f"min={a.min():.0f}, p50={np.median(a):.0f}, p90={np.quantile(a,0.9):.0f}, max={a.max():.0f}"
    )

print("\n=== Multiplicity per event (MCParticles with given generatorStatus) ===")
for s in STATUSES:
    summarize(mult[s], f"genStatus=={s}")

# ----------------------------
# Plotting
# ----------------------------
def overlay_hist(data_a, data_b, label_a, label_b, title, xlabel, outname, bins=120, logy=False, xlim=None):
    a = to_np(data_a)
    b = to_np(data_b)

    plt.figure()
    plt.hist(a, bins=bins, density=True, histtype="step", linewidth=2, label=label_a)
    plt.hist(b, bins=bins, density=True, histtype="step", linewidth=2, label=label_b)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    if logy:
        plt.yscale("log")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()
    print("Saved", outname)

# Multiplicity overlay
overlay_hist(
    mult[0], mult[1],
    "genStatus==0", "genStatus==1",
    "Multiplicity per event", "N particles per event",
    f"{OUT_PREFIX}_mult.png",
    bins=80, logy=True
)

# Momentum overlays
# Use some reasonable x-limits to avoid a few huge outliers dominating view
overlay_hist(pool[0]["p"],  pool[1]["p"],  "genStatus==0", "genStatus==1",
             "Momentum magnitude |p|", "|p|", f"{OUT_PREFIX}_p.png",  bins=160, logy=True)

overlay_hist(pool[0]["pt"], pool[1]["pt"], "genStatus==0", "genStatus==1",
             "Transverse momentum pT", "pT", f"{OUT_PREFIX}_pt.png", bins=160, logy=True)

overlay_hist(pool[0]["pz"], pool[1]["pz"], "genStatus==0", "genStatus==1",
             "Longitudinal momentum pz", "pz", f"{OUT_PREFIX}_pz.png", bins=200, logy=True)

overlay_hist(pool[0]["eta"], pool[1]["eta"], "genStatus==0", "genStatus==1",
             "Pseudorapidity eta", "eta", f"{OUT_PREFIX}_eta.png", bins=160, logy=False, xlim=(-8, 8))

# PDG composition (top few) – printed, not plotted
def top_pdg(pdg_list, k=12):
    c = Counter(pdg_list)
    return c.most_common(k)

print("\n=== Top PDGs (genStatus==0) ===")
print(top_pdg(pool[0]["pdg"]))
print("\n=== Top PDGs (genStatus==1) ===")
print(top_pdg(pool[1]["pdg"]))
