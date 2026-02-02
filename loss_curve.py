import os
import numpy as np
import matplotlib.pyplot as plt

outdir = "mc_gen1_model_simple"

train = np.load(os.path.join(outdir, "train_losses.npy"))
val   = np.load(os.path.join(outdir, "val_losses.npy"))

epochs = np.arange(1, len(train) + 1)

plt.figure(figsize=(6,4))
plt.plot(epochs, train, label="Train")
plt.plot(epochs, val, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.yscale("log")   # optional but recommended
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150)
plt.show()
