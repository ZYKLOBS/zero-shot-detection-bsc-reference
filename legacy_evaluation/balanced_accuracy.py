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
pixel4_method = "ignore" # "ignore", "avg", "direct"
delta_idx = 1 # which Δ index to test (e.g., 0 or 1)

# --- Load datasets (same as your code) ---
datasets = {
    "Raise1k (real)": np.load(f"../results/raise1k_SREC_{pixel4_method}.npz")["delta_vals"],
    #"Midjourney": np.load(f"../results/mj5_SREC_{pixel4_method}.npz")["delta_vals"],
    #"DALL-E2": np.load(f"../results/dalle2_SREC_{pixel4_method}.npz")["delta_vals"],
    #"GLIDE": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    #"SDXL": np.load(f"../results/glide_SREC_{pixel4_method}.npz")["delta_vals"],
    "openimgs (real)": np.load(f"../results/openimgs_SREC_{pixel4_method}.npz")["delta_vals"],
    "imagenet": np.load(f"../results/imagenet_SREC_{pixel4_method}.npz")["delta_vals"],
    #"progan_real": np.load(f"../results/progan_real_SREC_{pixel4_method}.npz")["delta_vals"],
    "progan_fake": np.load(f"../results/progan_fake2_SREC_{pixel4_method}.npz")["delta_vals"],
    #"S3GAN": np.load(f"../results/s3gan_fake_SREC_{pixel4_method}.npz")["delta_vals"],
}

real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "progan_fake", "S3GAN"]

scores = []
labels = []

for name, D_vals in datasets.items():
    d_clean = D_vals[:, delta_idx][np.isfinite(D_vals[:, delta_idx])]
    if len(d_clean) == 0:
        continue
    scores.extend(d_clean.tolist())
    labels.extend(([0]*len(d_clean)) if name in real_sets else [1]*len(d_clean))

scores = np.array(scores)
labels = np.array(labels)
n_real = np.sum(labels == 0)
n_fake = np.sum(labels == 1)

print("Real:", n_real)
print("Fake:", n_fake)

# --- Interval optimization ---
def best_two_thresholds(scores, labels, n_grid=200):
    smin, smax = np.min(scores), np.max(scores)
    thresholds = np.linspace(smin, smax, n_grid)
    best_acc = 0
    best_pair = (smin, smax)
    best_preds = None
    for i, t_low in enumerate(thresholds[:-1]):
        for t_high in thresholds[i+1:]:
            preds = np.ones_like(labels)
            preds[(scores >= t_low) & (scores <= t_high)] = 0
            acc = balanced_accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_pair = (t_low, t_high)
                best_preds = preds
    return best_acc, best_pair, best_preds

acc, (t_low, t_high), preds = best_two_thresholds(scores, labels)
cm = confusion_matrix(labels, preds)

auroc = roc_auc_score(labels, scores)
precision, recall, _ = precision_recall_curve(labels, scores)
ap = average_precision_score(labels, scores)
baseline_ap = n_real / (n_real + n_fake)

# --- Plot  ---
font_large = 18
font_legend = 16
tick_size = 14

# --- Histogram ---
plt.figure(figsize=(10,6))
plt.hist(scores[labels==0], bins=50, alpha=0.5, label="Real", density=True)
plt.hist(scores[labels==1], bins=50, alpha=0.5, label="Fake", density=True)
plt.axvline(t_low, color="red", linestyle="--", label=f"t_low = {t_low:.3f}")
plt.axvline(t_high, color="blue", linestyle="--", label=f"t_high = {t_high:.3f}")

delta_label = r"$\Delta^{(0,1)}$" if delta_idx==1 else r"$\Delta^{(1,2)}$"

plt.title(f"Two-Threshold Classification (Balanced Acc = {acc:.3f})", fontsize=font_large)
plt.xlabel(delta_label, fontsize=font_large)
plt.ylabel("Density", fontsize=font_large)
plt.legend(fontsize=font_legend)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- ROC Curve ---
plt.figure(figsize=(8,8))
fpr, tpr, _ = roc_curve(labels, scores)
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}", linewidth=2)
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate", fontsize=font_large)
plt.ylabel("True Positive Rate", fontsize=font_large)
plt.title("ROC Curve", fontsize=font_large)
plt.legend(fontsize=font_legend)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Precision-Recall Curve ---
plt.figure(figsize=(8,8))
plt.plot(recall, precision, label=f"AUPRC = {ap:.3f}", linewidth=2)
plt.hlines(baseline_ap, 0, 1, colors="gray", linestyles="--", label=f"Baseline = {baseline_ap:.3f}")
plt.xlabel("Recall", fontsize=font_large)
plt.ylabel("Precision", fontsize=font_large)
plt.title("Precision-Recall Curve", fontsize=font_large)
plt.legend(fontsize=font_legend)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Confusion Matrix ---
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Real','Pred Fake'],
            yticklabels=['True Real','True Fake'], annot_kws={"size":font_large})
plt.title("Confusion Matrix at Optimal Two-Threshold Interval", fontsize=font_large)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.tight_layout()
plt.show()
