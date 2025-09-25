import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
D_scatter_x = 0   # Which D index (e.g. 0, 1, 2)
D_scatter_y = 2
D_hist = 2
bins = 40
pixel4_method = "ignore"  # "ignore", "avg", or "direct"
alpha = 0.5


datasets = {
    "Raise1k (real)": np.load(f"../results/raise1k_SREC_{pixel4_method}.npz")["D_vals"],
    #"Midjourney": np.load(f"../results/mj5_SREC_{pixel4_method}.npz")["D_vals"],
    #"DALL-E2": np.load(f"../results/dalle2_SREC_{pixel4_method}.npz")["D_vals"],
    #"GLIDE": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["D_vals"],
    #"SDXL": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["D_vals"],
    "openimgs (real)": np.load(f"../results/openimgs_SREC_{pixel4_method}.npz")["D_vals"],
    "imagenet": np.load(f"../results/imagenet_SREC_{pixel4_method}.npz")["D_vals"],
    "progan_real": np.load(f"../results/progan_real_SREC_{pixel4_method}.npz")["D_vals"],
    "progan_fake": np.load(f"../results/progan_fake2_SREC_{pixel4_method}.npz")["D_vals"],
    "S3GAN": np.load(f"../results/s3gan_fake_SREC_{pixel4_method}.npz")["D_vals"]

}

colors = {
    "Raise1k (real)": "#8172b3",
    "Midjourney": "#55a868",
    "DALL-E2": "#4c72b0",
    "GLIDE": "#dd8452",
    "SDXL": "#c44e52",
    "openimgs (real)": "black",
    "imagenet": "brown",
    "progan_real": "cyan",
    "progan_fake": "magenta",
    "S3GAN": "yellow"
}

plt.figure(figsize=(12,4))

# --- Scatter plot ---
plt.subplot(1,2,1)
for name, D_vals in datasets.items():

    mask = np.isfinite(D_vals[:,D_scatter_x]) & np.isfinite(D_vals[:,D_scatter_y])
    x = D_vals[:,D_scatter_x][mask]
    y = D_vals[:,D_scatter_y][mask]

    if len(x) > 0:
        plt.scatter(x, y, alpha=alpha, color=colors[name], label=name, edgecolor='w', s=20)


plt.xlabel(f"$D^{{({2 - D_scatter_x})}}$")
plt.ylabel(f"$D^{{({2 - D_scatter_y})}}$")
plt.title(f"$D^{{({2 - D_scatter_y})}}$ vs $D^{{({2 - D_scatter_x})}}$")
plt.legend()

# --- Histogram ---
plt.subplot(1,2,2)
for name, D_vals in datasets.items():
    D_clean = D_vals[:,D_hist][np.isfinite(D_vals[:,D_hist])]

    if len(D_clean) == 0:
        continue

    # Add tiny noise if constant to avoid divide-by-zero in density
    #if np.all(D_clean == D_clean[0]):
    #    D_clean = D_clean + 1e-12 * np.random.randn(len(D_clean))

    plt.hist(D_clean, bins=bins, alpha=alpha, density=True, color=colors[name], label=name)


plt.xlabel(f"$D^{{({2 - D_hist})}}$")
plt.ylabel("Density")
plt.title(f"Histogram of $D^{{({2 - D_hist})}}$")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
