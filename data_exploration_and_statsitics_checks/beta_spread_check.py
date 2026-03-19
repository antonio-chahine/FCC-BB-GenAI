import numpy as np
from particle_diffusion_new import beta_unsquash_np

raw = np.load("guineapig_raw_trimmed.npy", allow_pickle=True)
beta = np.concatenate([ev[:, 1:4] for ev in raw if ev is not None], axis=0).astype(np.float32)

u = beta_unsquash_np(beta)
print("std u_x:", np.std(u[:, 0]))
print("std u_y:", np.std(u[:, 1]))
print("percentiles |u_x|:", np.percentile(np.abs(u[:, 0]), [50, 90, 99]))