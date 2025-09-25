import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
Delta_scatter_x = 1   # which Δ index (e.g., 0 or 1)
Delta_scatter_y = 0
Delta_hist = 1
bins = 40
pixel4_method = "ignore"  # "direct", "avg", or "ignore"
alpha=0.5


datasets = {
    "Raise1k (real)": np.load(f"../results/raise1k_SREC_{pixel4_method}.npz")["delta_vals"],
    "Midjourney": np.load(f"../results/mj5_SREC_{pixel4_method}.npz")["delta_vals"],
    "DALL-E2": np.load(f"../results/dalle2_SREC_{pixel4_method}.npz")["delta_vals"],
    "GLIDE": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    "SDXL": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    "openimgs (real)": np.load(f"../results/openimgs_SREC_{pixel4_method}.npz")["delta_vals"]
}

colors = {
    "Raise1k (real)": "purple",
    "Midjourney": "green",
    "DALL-E2": "blue",
    "GLIDE": "orange",
    "SDXL": "red",
    "openimgs (real)": "black"
}

plt.figure(figsize=(12,4))

# ---  Scatter plot ---
plt.subplot(1,2,1)
for name, Delta_vals in datasets.items():
    mask = np.isfinite(Delta_vals[:,Delta_scatter_x]) & np.isfinite(Delta_vals[:,Delta_scatter_y])
    x = np.abs(Delta_vals[:,Delta_scatter_x][mask])
    y = np.abs(Delta_vals[:,Delta_scatter_y][mask])

    if len(x) > 0:
        plt.scatter(x, y, alpha=alpha, color=colors[name], label=name, edgecolor='w', s=20)

plt.xlabel(f"|Δ({Delta_scatter_x})|")
plt.ylabel(f"|Δ({Delta_scatter_y})|")
plt.title(f"|Δ({Delta_scatter_y})| vs |Δ({Delta_scatter_x})|")
plt.legend()

# ---  Histogram ---
plt.subplot(1,2,2)
for name, Delta_vals in datasets.items():
    delta_clean = np.abs(Delta_vals[:,Delta_hist][np.isfinite(Delta_vals[:,Delta_hist])])

    if len(delta_clean) == 0:
        continue

    # Add tiny noise if constant to avoid divide-by-zero in density
    #if np.all(delta_clean == delta_clean[0]):
    #    delta_clean = delta_clean + 1e-12 * np.random.randn(len(delta_clean))

    plt.hist(delta_clean, bins=bins, alpha=alpha, density=True, color=colors[name], label=name)

plt.xlabel(f"|Δ({Delta_hist})|")
plt.ylabel("Density")
plt.title(f"Histogram of |Δ({Delta_hist})|")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()
