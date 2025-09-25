import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parameters ---
pixel4_method = "ignore"

datasets = {
    "Raise1k (real)": f"../results/raise1k_SREC_{pixel4_method}.npz",
    "Midjourney": f"../results/mj5_SREC_{pixel4_method}.npz",
    "DALL-E2": f"../results/dalle2_SREC_{pixel4_method}.npz",
    "GLIDE": f"../results/glide_SREC_{pixel4_method}.npz",
    "SDXL": f"../results/glide_SREC_{pixel4_method}.npz",
    "openimgs (real)": f"../results/openimgs_SREC_{pixel4_method}.npz",
    "imagenet": f"../results/imagenet_SREC_{pixel4_method}.npz",
    "progan_real": f"../results/progan_real_SREC_{pixel4_method}.npz",
    "S3GAN": f"../results/s3gan_fake_SREC_{pixel4_method}.npz",
}

# Which sets are real vs synthetic
real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "S3GAN"]


all_datasets = {}
for name, path in datasets.items():
    npz = np.load(path)
    D_vals = npz["D_vals"]
    delta_vals = npz["delta_vals"]
    feats = np.concatenate([D_vals, delta_vals], axis=1)
    feats = feats[np.all(np.isfinite(feats), axis=1)]
    all_datasets[name] = feats

real_data, fake_data = [], []
for name, feats in all_datasets.items():
    target_list = real_data if name in real_sets else fake_data
    for row in feats:
        target_list.append({
            "dataset": name,
            "kind": "Real" if name in real_sets else "Fake",
            "D2": row[0], "D1": row[1], "D0": row[2],
            "Δ(1,2)": row[3], "Δ(0,1)": row[4]
        })

real_df = pd.DataFrame(real_data)
fake_df = pd.DataFrame(fake_data)
fake_df = fake_df.sample(n=len(real_df), random_state=0)
df = pd.concat([real_df, fake_df], ignore_index=True)


# --- Correlation matrix ---
feature_cols = ["D2", "D1", "D0", "Δ(1,2)", "Δ(0,1)"]
X = df[feature_cols]
corr_matrix = X.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

latex_labels = {
    "D2": r"$D^{(2)}$",
    "D1": r"$D^{(1)}$",
    "D0": r"$D^{(0)}$",
    "Δ(1,2)": r"$\Delta^{(1,2)}$",
    "Δ(0,1)": r"$\Delta^{(0,1)}$"
}


# --- Heatmap ---
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.8,
    square=True,
    annot_kws={"size": 16}  # font size
)


ax.set_xticklabels([latex_labels[c] for c in feature_cols], rotation=45, ha="right", fontsize=16)
ax.set_yticklabels([latex_labels[c] for c in feature_cols], rotation=0, fontsize=16)


ax.set_title("Feature Correlation Matrix (Balanced Real vs Fake)", fontsize=22, pad=20)


cbar = ax.collections[0].colorbar
cbar.set_label("Correlation", fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.show()
