import numpy as np
data = np.load("extra_data_7col.npy", allow_pickle=True)
print(data[0][:3])   # first 3 particles of first event