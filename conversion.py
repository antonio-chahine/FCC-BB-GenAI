import numpy as np
import pyhepmc as hep

NM_TO_MM = 1e-6

def convert_generated_to_hepmc(npy_path, hepmc_path, max_events=None):
    events = np.load(npy_path, allow_pickle=True)

    with hep.open(hepmc_path, "w") as f:
        for iev, ev in enumerate(events):
            if max_events is not None and iev >= max_events:
                break

            ev = np.asarray(ev, dtype=np.float32)

            if ev.ndim != 2 or ev.shape[1] < 8:
                print(f"Skipping event {iev}: bad shape {ev.shape}")
                continue

            evt = hep.GenEvent()
            evt.event_number = iev

            for row in ev:
                pdg   = int(row[0])
                E     = float(row[1])
                betax = float(row[2])
                betay = float(row[3])
                betaz = float(row[4])
                x_nm  = float(row[5])
                y_nm  = float(row[6])
                z_nm  = float(row[7])

                # Your code uses p = E * beta
                px = E * betax
                py = E * betay
                pz = E * betaz

                # Convert nm -> mm for detector simulation
                x_mm = x_nm * NM_TO_MM
                y_mm = y_nm * NM_TO_MM
                z_mm = z_nm * NM_TO_MM

                vtx = hep.GenVertex((x_mm, y_mm, z_mm, 0.0))
                p = hep.GenParticle((px, py, pz, E), pdg, 1)  # status 1 = final state

                vtx.add_particle_out(p)
                evt.add_vertex(vtx)

            f.write(evt)

    print(f"Saved HepMC3 file to {hepmc_path}")


convert_generated_to_hepmc(
    "/work/submit/anton100/msci-project/FCC-BB-GenAI/T_sweep_cosine_charge/T_100/generated_events.npy",
    "/work/submit/anton100/msci-project/FCC-BB-GenAI/T_sweep_cosine_charge/T_100/generated.hepmc3",
    max_events=10
)
    