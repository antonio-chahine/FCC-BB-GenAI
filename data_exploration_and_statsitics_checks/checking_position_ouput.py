import numpy as np

raw = np.load("guineapig_raw_trimmed.npy", allow_pickle=True)
all_pos = np.concatenate([ev[:, 4:7] for ev in raw if ev is not None], axis=0)
print("x: ", all_pos[:,0].min(), all_pos[:,0].max())
print("y: ", all_pos[:,1].min(), all_pos[:,1].max())
print("z: ", all_pos[:,2].min(), all_pos[:,2].max())