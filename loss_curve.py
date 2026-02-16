import os
import numpy as np
import matplotlib.pyplot as plt

outdir = "new_4"

train = np.load(os.path.join(outdir, "train_losses.npy"))
val   = np.load(os.path.join(outdir, "val_losses.npy"))

epochs = np.arange(1, len(train) + 1)

plt.figure(figsize=(7, 4.5), dpi=200)

plt.plot(epochs, train, label="Train", linewidth=2.0)
plt.plot(epochs, val,   label="Validation", linewidth=2.0)

plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.yscale("log")

plt.grid(True, which="major", alpha=0.3)
plt.grid(True, which="minor", alpha=0.15)

plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(os.path.join(outdir, "loss_curve.pdf"))
plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=300)
plt.show()
