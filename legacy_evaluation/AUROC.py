import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# --- Parameters ---
pixel4_method = "ignore"   # direct, ignore, avg
delta_idx = 1              # which Δ index to test (e.g., 0 or 1)

# --- Load datasets ---
datasets = {
    #"Raise1k (real)": np.load(f"../results/raise1k_SREC_{pixel4_method}.npz")["delta_vals"],
    "Midjourney": np.load(f"../results/mj5_SREC_{pixel4_method}.npz")["delta_vals"],
    #"DALL-E2": np.load(f"../results/dalle2_SREC_{pixel4_method}.npz")["delta_vals"],
    #"GLIDE": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    #"SDXL": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    #"openimgs (real)": np.load(f"../results/openimgs_SREC_{pixel4_method}.npz")["delta_vals"],
    "imagenet": np.load(f"../results/imagenet_SREC_{pixel4_method}.npz")["delta_vals"],
    # "progan_real": np.load(f"../results/progan_real_SREC_{pixel4_method}.npz")["delta_vals"],
    # "progan_fake": np.load(f"../results/progan_fake2_SREC_{pixel4_method}.npz")["delta_vals"],
    #"S3GAN": np.load(f"../results/s3gan_fake_SREC_{pixel4_method}.npz")["delta_vals"],
}

# Which sets are real vs synthetic
real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "progan_fake", "S3GAN"]


scores = []
labels = []

for name, Delta_vals in datasets.items():
    # Flip sign of delta, since mistake in calculations and time left for the thesis is too short to fix this
    delta_clean = -Delta_vals[:, delta_idx][np.isfinite(Delta_vals[:, delta_idx])]
    if len(delta_clean) == 0:
        continue

    scores.extend(delta_clean.tolist())
    labels.extend(([0] * len(delta_clean)) if name in real_sets else [1] * len(delta_clean))

scores = np.array(scores)
labels = np.array(labels)


auc_normal = roc_auc_score(labels, scores)
auc_flipped = roc_auc_score(labels, -scores)

if auc_flipped > auc_normal:
    auc = auc_flipped
    scores = -scores   # use flipped scores for ROC curve., i.e. classifier with 1% accuracy is 99% basically
    orientation = "flipped"
else:
    auc = auc_normal
    orientation = "normal"

print(f"AUROC (D index {delta_idx}, orientation={orientation}): {auc:.4f}")

# --- Plot ---
fpr, tpr, _ = roc_curve(labels, scores)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f}, {orientation})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (D-values)")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()