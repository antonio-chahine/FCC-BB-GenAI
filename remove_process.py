import numpy as np

data = np.load("extra_data.npy", allow_pickle=True)
data_7col = np.array([ev[:, :7] for ev in data], dtype=object)
np.save("extra_data_7col.npy", data_7col, allow_pickle=True)
print(f"Done. {len(data_7col)} events, example shape: {data_7col[0].shape}")