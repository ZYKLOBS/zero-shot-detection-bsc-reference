import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

# --- Parameters ---

pixel4_method = "ignore" # "ignore", "avg", "direct"
delta_idx = 1  # which Δ index to test (e.g., 0 or 1)

datasets = {
    "Raise1k (real)": f"../results/raise1k_SREC_{pixel4_method}.npz",
    #"Midjourney": f"../results/mj5_SREC_{pixel4_method}.npz",
    #"DALL-E2": f"../results/dalle2_SREC_{pixel4_method}.npz",
    #"GLIDE": f"../results/glide_SREC_{pixel4_method}.npz",
    #"SDXL": f"../results/glide_SREC_{pixel4_method}.npz",
    "openimgs (real)": f"../results/openimgs_SREC_{pixel4_method}.npz",
    "imagenet": f"../results/imagenet_SREC_{pixel4_method}.npz",
    #"progan_real": f"../results/progan_real_SREC_{pixel4_method}.npz",
    "progan_fake": f"../results/progan_fake2_SREC_{pixel4_method}.npz",
    #"S3GAN": f"../results/s3gan_fake_SREC_{pixel4_method}.npz",
}

# Which sets are real vs synthetic
real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "S3GAN", "progan_fake"]


all_data = []
for name, path in datasets.items():
    npz = np.load(path)
    feats = npz["delta_vals"]
    feats = feats[np.all(np.isfinite(feats), axis=1)]  # remove NaN/Inf
    kind = 0 if name in real_sets else 1  # 0 = real, 1 = fake
    for row in feats:
        all_data.append({"Δ(1,2)": row[0], "Δ(0,1)": row[1], "label": kind})

df = pd.DataFrame(all_data)


# Train/test split
X = df[["Δ(1,2)", "Δ(0,1)"]].values
y = df["label"].values


X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=115, stratify=y
)


scores = X_test[:, delta_idx]
labels = y_test


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

# --- Metrics ---
auroc = roc_auc_score(labels, scores)
precision, recall, _ = precision_recall_curve(labels, scores)
ap = average_precision_score(labels, scores)
n_real = np.sum(labels == 0)
n_fake = np.sum(labels == 1)
baseline_ap = n_real / (n_real + n_fake)

# --- Plot ---
font_large = 18
font_legend = 16
tick_size = 14
delta_label = r"$\Delta^{(0,1)}$" if delta_idx == 1 else r"$\Delta^{(1,2)}$"

# --- Histogram ---
plt.figure(figsize=(10,6))
plt.hist(scores[labels==0], bins=50, alpha=0.5, label="Real", density=True)
plt.hist(scores[labels==1], bins=50, alpha=0.5, label="Fake", density=True)
plt.axvline(t_low, color="red", linestyle="--", label=f"t_low = {t_low:.3f}")
plt.axvline(t_high, color="blue", linestyle="--", label=f"t_high = {t_high:.3f}")
plt.title(f"Two-Threshold Classification (Balanced Acc = {acc:.3f})", fontsize=font_large)
plt.xlabel(delta_label, fontsize=font_large)
plt.ylabel("Density", fontsize=font_large)
plt.legend(fontsize=font_legend)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- ROC ---
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

# --- PR Curve ---
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Real','Pred Fake'],
            yticklabels=['True Real','True Fake'],
            annot_kws={"size":font_large})
plt.title("Confusion Matrix at Optimal Two-Threshold Interval", fontsize=font_large)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.tight_layout()
plt.show()
