import glob
import numpy as np
from podio import root_io

FILES = glob.glob(
    "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_o2_v05/FCCee_Z_4IP_04may23_FCCee_Z/*.root"
)

OUTFILE = "mcparticles_gen1.npy"

all_events = []  # list of events, each event is (N_i, 4): [pdg, px, py, pz]

for f in FILES:
    print("Reading", f)
    reader = root_io.Reader(f)
    events = reader.get("events")

    for event in events:
        if "MCParticles" not in event.getAvailableCollections():
            all_events.append(np.zeros((0, 4), dtype=np.float32))
            continue

        mcp = event.get("MCParticles")
        rows = []

        for p in mcp:
            if int(p.getGeneratorStatus()) != 1:
                continue

            mom = p.getMomentum()
            rows.append([
                int(p.getPDG()),
                float(mom.x), float(mom.y), float(mom.z)
            ])

        all_events.append(np.array(rows, dtype=np.float32))

np.save(OUTFILE, np.array(all_events, dtype=object))
print("Saved", OUTFILE)
print("Total events:", len(all_events))
