import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Parameters ---
pixel4_method = "ignore"   # "ignore", "avg", "direct"

datasets = {
    "Raise1k (real)": f"../results/raise1k_SREC_{pixel4_method}.npz",
    "Midjourney": f"../results/mj5_SREC_{pixel4_method}.npz",
    "DALL-E2": f"../results/dalle2_SREC_{pixel4_method}.npz",
    "GLIDE": f"../results/glide_SREC_{pixel4_method}.npz",
    "SDXL": f"../results/glide_SREC_{pixel4_method}.npz",  # double-check path
    "openimgs (real)": f"../results/openimgs_SREC_{pixel4_method}.npz",
    "imagenet": f"../results/imagenet_SREC_{pixel4_method}.npz",
    "progan_real": f"../results/progan_real_SREC_{pixel4_method}.npz",
    "S3GAN": f"../results/s3gan_fake_SREC_{pixel4_method}.npz",
}

real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "S3GAN"]


all_datasets = {}

for name, path in datasets.items():
    npz = np.load(path)
    D_vals = npz["D_vals"]       # shape (N, 3)
    delta_vals = npz["delta_vals"]  # shape (N, 2)

    feats = np.concatenate([D_vals, delta_vals], axis=1)  # shape (N, 5)
    feats = feats[np.all(np.isfinite(feats), axis=1)]     # filter NaN/Inf

    all_datasets[name] = feats


real_data = []
fake_data = []

for name, feats in all_datasets.items():
    if name in real_sets:
        for row in feats:
            real_data.append({
                "dataset": name,
                "kind": "Real",
                "D2": row[0],
                "D1": row[1],
                "D0": row[2],
                "Δ(1,2)": row[3],
                "Δ(0,1)": row[4],
            })
    else:
        for row in feats:
            fake_data.append({
                "dataset": name,
                "kind": "Fake",
                "D2": row[0],
                "D1": row[1],
                "D0": row[2],
                "Δ(1,2)": row[3],
                "Δ(0,1)": row[4],
            })


# Balance: match total fake to total real
real_df = pd.DataFrame(real_data)
fake_df = pd.DataFrame(fake_data)

n_real = len(real_df)
print("Total real samples:", n_real)

fake_df = fake_df.sample(n=n_real, random_state=0)


df = pd.concat([real_df, fake_df], ignore_index=True)
print("Balanced data:", df.shape)
print(df['kind'].value_counts())

# --- Pairplot ---
pairplot = sns.pairplot(
    df,
    vars=["D2", "D1", "D0", "Δ(1,2)", "Δ(0,1)"],
    hue="kind",
    diag_kind="kde",
    corner=True,
    plot_kws=dict(alpha=0.3, s=20, edgecolor="none")
)


latex_labels = {
    "D2": r"$D^{(2)}$",
    "D1": r"$D^{(1)}$",
    "D0": r"$D^{(0)}$",
    "Δ(1,2)": r"$\Delta^{(1,2)}$",
    "Δ(0,1)": r"$\Delta^{(0,1)}$"
}


for ax, var in zip(pairplot.diag_axes, pairplot.x_vars):
    ax.set_xlabel(latex_labels.get(var, var), fontsize=14)
    ax.set_ylabel(latex_labels.get(var, var), fontsize=14, rotation=0, labelpad=40)
    ax.tick_params(labelsize=12)


for i, row_var in enumerate(pairplot.y_vars):
    for j, col_var in enumerate(pairplot.x_vars[:i+1]):  # only lower triangle
        ax = pairplot.axes[i, j]
        if ax is not None:
            ax.set_xlabel(latex_labels.get(col_var, col_var), fontsize=14)
            ax.set_ylabel(latex_labels.get(row_var, row_var), fontsize=14)
            ax.tick_params(labelsize=12)


if pairplot._legend is not None:
    pairplot._legend.remove()

plt.subplots_adjust(top=0.92, right=0.92)
plt.suptitle("Pairplot of SReC Features (Balanced Real vs Fake)", fontsize=16, y=1.02)
plt.show()
