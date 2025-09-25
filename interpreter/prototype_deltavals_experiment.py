import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Delta_hist = (1, 0)  # compare |Δ(1)| + |Δ(0)|
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
    "progan_fake": np.load(f"../results/progan_fake2_SREC_{pixel4_method}.npz")["delta_vals"],
    # "S3GAN": np.load(f"../results/s3gan_fake_SREC_{pixel4_method}.npz")["delta_vals"]
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

plt.figure(figsize=(12, 4))

# --- Scatter plot ---
plt.subplot(1, 2, 1)
for name, Delta_vals in datasets.items():
    mask = np.isfinite(Delta_vals[:, 0]) & np.isfinite(Delta_vals[:, 1])
    summed = (np.abs(Delta_vals[:, 1]) + np.abs(Delta_vals[:, 0]))[mask]

    if len(summed) > 0:
        plt.scatter(range(len(summed)), summed, alpha=alpha, color=colors[name], label=name, s=20)

plt.xlabel("Sample index")
plt.ylabel(r"$|\Delta^{0,1}| + |\Delta^{1,2}|$")
plt.title("Sum of absolute values: Δ(1) + Δ(0)")
plt.legend()

# --- Histogram ---
plt.subplot(1, 2, 2)
for name, Delta_vals in datasets.items():
    mask = np.isfinite(Delta_vals[:, 0]) & np.isfinite(Delta_vals[:, 1])
    summed = (np.abs(Delta_vals[:, 1]) + np.abs(Delta_vals[:, 0]))[mask]

    if len(summed) == 0:
        continue

    plt.hist(summed, bins=bins, alpha=alpha, density=True, color=colors[name], label=name)

plt.xlabel(r"$|\Delta^{0,1}| + |\Delta^{1,2}|$")
plt.ylabel("Density")
plt.title("Histogram of |Δ(1)| + |Δ(0)|")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
