import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
#idx= 0 -> Δ(21) => -Δ(21) = Δ(12) | the data has unfortunately been generated opposite to the paper, ez fix but I can't regenerate it all
#in the short amount of time left so we just use -
#idx= 1 -> Δ(10) => -Δ(10) = Δ(01)

Delta_scatter_x = 1  # which Δ index (e.g., 0 or 1)
Delta_scatter_y = 0
Delta_hist = 1
bins = 40
pixel4_method = "ignore"  # "direct", "avg", or "ignore"
alpha = 0.5


datasets = {
    "Raise1k (real)": np.load(f"../results/raise1k_SREC_{pixel4_method}.npz")["delta_vals"],
    #"Midjourney": np.load(f"../results/mj5_SREC_{pixel4_method}.npz")["delta_vals"],
    #"DALL-E2": np.load(f"../results/dalle2_SREC_{pixel4_method}.npz")["delta_vals"],
    #"GLIDE": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    #"SDXL": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    "openimgs (real)": np.load(f"../results/openimgs_SREC_{pixel4_method}.npz")["delta_vals"],
    "imagenet": np.load(f"../results/imagenet_SREC_{pixel4_method}.npz")["delta_vals"],
    "progan_real": np.load(f"../results/progan_real_SREC_{pixel4_method}.npz")["delta_vals"],
    #"progan_fake": np.load(f"../results/progan_fake2_SREC_{pixel4_method}.npz")["delta_vals"],
    #"S3GAN": np.load(f"../results/s3gan_fake_SREC_{pixel4_method}.npz")["delta_vals"]

}

colors = {
    "Raise1k (real)": "purple",
    "Midjourney": "green",
    "DALL-E2": "blue",
    "GLIDE": "orange",
    "SDXL": "red",
    "openimgs (real)": "black",
    "imagenet": "brown",
    "progan_real": "cyan",
    "progan_fake": "magenta",
    "S3GAN": "yellow"
}

delta_labels = {0: r"$\Delta^{1,2}$", 1: r"$\Delta^{0,1}$"}


plt.figure(figsize=(12,4))
# ---  Scatter plot ---
plt.subplot(1,2,1)
for name, Delta_vals in datasets.items():
    mask = np.isfinite(Delta_vals[:,Delta_scatter_x]) & np.isfinite(Delta_vals[:,Delta_scatter_y])
    x = -Delta_vals[:,Delta_scatter_x][mask]   # flip sign
    y = -Delta_vals[:,Delta_scatter_y][mask]   # flip sign

    if len(x) > 0:
        plt.scatter(x, y, alpha=alpha, color=colors[name], label=name, edgecolor='w', s=20)

plt.xlabel(delta_labels.get(Delta_scatter_x, f"Δ({Delta_scatter_x})"))
plt.ylabel(delta_labels.get(Delta_scatter_y, f"Δ({Delta_scatter_y})"))
plt.title(f"{delta_labels.get(Delta_scatter_y, f'Δ({Delta_scatter_y})')} vs {delta_labels.get(Delta_scatter_x, f'Δ({Delta_scatter_x})')}")
plt.legend()

# ---  Histogram ---
plt.subplot(1,2,2)
for name, Delta_vals in datasets.items():
    delta_clean = -Delta_vals[:,Delta_hist][np.isfinite(Delta_vals[:,Delta_hist])]  # flip sign

    if len(delta_clean) == 0:
        continue

    plt.hist(delta_clean, bins=bins, alpha=alpha, density=True, color=colors[name], label=name)

plt.xlabel(delta_labels.get(Delta_hist, f"Δ({Delta_hist})"))
plt.ylabel("Density")
plt.title(f"Histogram of {delta_labels.get(Delta_hist, f'Δ({Delta_hist})')}")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
