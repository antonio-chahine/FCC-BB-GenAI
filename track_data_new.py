import glob
import numpy as np
from tqdm import tqdm

FILES = glob.glob(
    "/ceph/submit/data/group/fcc/ee/beam_backgrounds/guineapig/"
    "FCCee_Z_GHC_V25p3_4_FCCee_Z256_2T_grids8/*.pairs"
)

FILES = [f for f in FILES if "/output_" in f and "/output0_" not in f]
FILES = sorted(FILES)

OUTFILE = "guineapig_raw_trimmed.npy"

all_events = []
skipped = []

for f in tqdm(FILES, desc="Reading .pairs", unit="event"):
    try:
        d = np.loadtxt(f, dtype=np.float32)

        # keep only raw kinematics + vertex
        d = d[:, :7]

        all_events.append(d)

    except PermissionError:
        skipped.append(f)
        continue

    except OSError as e:
        # catches other filesystem weirdness
        skipped.append(f)
        continue

events = np.array(all_events, dtype=object)
np.save(OUTFILE, events, allow_pickle=True)

print(f"Saved {len(events)} events")
print(f"Skipped {len(skipped)} files")
print("Example event shape:", events[0].shape)
