import numpy as np

f = "/ceph/submit/data/group/fcc/ee/beam_backgrounds/guineapig/FCCee_Z_GHC_V25p3_4_FCCee_Z256_2T_grids8/output0_999833.pairs"

d = np.loadtxt(f, dtype=np.float32)
print("Shape:", d.shape)          # (N_particles, N_columns)
print("N particles:", d.shape[0])
print("N columns:", d.shape[1])
print(d[:5])  