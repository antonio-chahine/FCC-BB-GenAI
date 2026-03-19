import numpy as np
import matplotlib.pyplot as plt

data = np.load("/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy", allow_pickle=True)

all_z = np.concatenate([ev[:, 6] for ev in data if ev is not None])
all_bx = np.concatenate([ev[:, 1] for ev in data if ev is not None])  # col 1 = betax

print(np.percentile(all_z, [0.1, 1, 99, 99.9]))

mask_p = (all_z > 0)
mask_n = (all_z < 0)

plt.hist(all_bx[mask_p], bins=200, alpha=0.5, label='z+')
plt.hist(all_bx[mask_n], bins=200, alpha=0.5, label='z-')
plt.xlabel('β_x')
plt.legend()
plt.title('β_x split by z-plane')
plt.savefig("bx_hist_by_z.png")
plt.show()

data = np.load("/work/submit/anton100/msci-project/FCC-BB-GenAI/guineapig_raw_trimmed.npy", allow_pickle=True)
all_y = np.concatenate([ev[:, 5] for ev in data if ev is not None])
print(np.std(all_y))  # set ASINH_SCALE_Y to this value