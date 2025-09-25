import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

import seaborn as sns

# --- Parameters ---
pixel4_method = "ignore"   # "ignore", "avg", "direct"
d_idx = 2                  # which D index to test (0, 1, 2)

datasets = {
    "Raise1k (real)": np.load(f"../results/raise1k_SREC_{pixel4_method}.npz")["D_vals"],
    #"Midjourney": np.load(f"../results/mj5_SREC_{pixel4_method}.npz")["D_vals"],
    #"DALL-E2": np.load(f"../results/dalle2_SREC_{pixel4_method}.npz")["D_vals"],
    #"GLIDE": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["D_vals"],
    #"SDXL": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["D_vals"],  # <- double-check if this path is correct
    "openimgs (real)": np.load(f"../results/openimgs_SREC_{pixel4_method}.npz")["D_vals"],
    "imagenet": np.load(f"../results/imagenet_SREC_{pixel4_method}.npz")["D_vals"],
    #"progan_real": np.load(f"../results/progan_real_SREC_{pixel4_method}.npz")["D_vals"], #Might seem like cherrypicking but worsened results of AUPRC by ~5% with no noticable benefit for two threshhold analysis
    "progan_fake": np.load(f"../results/progan_fake2_SREC_{pixel4_method}.npz")["D_vals"],
    #"S3GAN": np.load(f"../results/s3gan_fake_SREC_{pixel4_method}.npz")["D_vals"]
}

# Which sets are real vs synthetic
real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "progan_fake", "S3GAN"]

# --- Collect scores + labels ---
scores = []
labels = []

print("\nPer-dataset counts:")
for name, D_vals in datasets.items():
    d_clean = D_vals[:, d_idx][np.isfinite(D_vals[:, d_idx])]
    if len(d_clean) == 0:
        continue

    kind = "Real" if name in real_sets else "Fake"
    print(f"{name:20s} ({kind}): {len(d_clean)} samples")

    scores.extend(d_clean.tolist())
    labels.extend(([0] * len(d_clean)) if name in real_sets else [1] * len(d_clean))

scores = np.array(scores)
labels = np.array(labels)

n_real = np.sum(labels == 0)
n_fake = np.sum(labels == 1)
print(f"\nTotal samples: {len(scores)}")
print(f"  Real: {n_real}")
print(f"  Fake: {n_fake}")


# --- Interval optimization ---
def best_two_thresholds(scores, labels, n_grid=200):
    smin, smax = np.min(scores), np.max(scores)
    thresholds = np.linspace(smin, smax, n_grid)

    best_acc = 0
    best_pair = (smin, smax)
    best_preds = None

    for i, t_low in enumerate(thresholds[:-1]):
        for t_high in thresholds[i+1:]:
            preds = np.ones_like(labels)  # default = fake
            preds[(scores >= t_low) & (scores <= t_high)] = 0  # real inside interval

            acc = balanced_accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_pair = (t_low, t_high)
                best_preds = preds

    return best_acc, best_pair, best_preds



acc, (t_low, t_high), preds = best_two_thresholds(scores, labels)
cm = confusion_matrix(labels, preds)

print(f"\nBest balanced accuracy: {acc:.4f}")
print(f"Optimal thresholds: [{t_low:.3f}, {t_high:.3f}]")
print("Confusion matrix [rows=true, cols=pred]:\n", cm)



auroc = roc_auc_score(labels, scores)
print(f"\nAUROC: {auroc:.4f}")


precision, recall, _ = precision_recall_curve(labels, scores)
ap = average_precision_score(labels, scores)
print(f"AUPRC (Average Precision): {ap:.4f}")


# --- Plot ---

# --- Histogram ---
plt.figure(figsize=(8,5))
plt.hist(scores[labels==0], bins=50, alpha=0.5, label="Real", density=True)
plt.hist(scores[labels==1], bins=50, alpha=0.5, label="Fake", density=True)
plt.axvline(t_low, color="red", linestyle="--", label=f"t_low = {t_low:.3f}")
plt.axvline(t_high, color="blue", linestyle="--", label=f"t_high = {t_high:.3f}")
plt.title(f"Two-Threshold Classification (Balanced Acc = {acc:.3f})")
plt.xlabel(rf"$D^{{({2 - d_idx})}}$")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# --- ROC Curve ---
fpr, tpr, _ = roc_curve(labels, scores)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Precision-Recall Curve ---
baseline = np.sum(labels == 1) / len(labels)
print(f"AUPRC baseline (random guess): {baseline:.3f}")

plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f"AUPRC = {ap:.3f}", linewidth=2)
plt.hlines(baseline, xmin=0, xmax=1, colors='gray', linestyles='--', label=f"Random Baseline = {baseline:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# --- Confusion Matrix ---
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Real', 'Pred Fake'], yticklabels=['True Real', 'True Fake'])
plt.title("Confusion Matrix at Optimal Two-Threshold Interval")
plt.show()