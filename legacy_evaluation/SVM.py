import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Parameters
# -------------------
pixel4_method = "ignore"

datasets = {
    "Raise1k (real)": f"../results/raise1k_SREC_{pixel4_method}.npz",
    #"Midjourney": f"../results/mj5_SREC_{pixel4_method}.npz",
    #"DALL-E2": f"../results/dalle2_SREC_{pixel4_method}.npz",
    #"GLIDE": f"../results/glide_SREC_{pixel4_method}.npz",
    #"SDXL": f"../results/glide_SREC_{pixel4_method}.npz",
    "openimgs (real)": f"../results/openimgs_SREC_{pixel4_method}.npz",
    "imagenet": f"../results/imagenet_SREC_{pixel4_method}.npz",
    #"progan_real": f"../results/progan_real_SREC_{pixel4_method}.npz",
    "progan_fake": f"../results/progan_fake_SREC_{pixel4_method}.npz",
    #"S3GAN": f"../results/s3gan_fake_SREC_{pixel4_method}.npz",
}

real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "S3GAN", "progan_fake"]


all_data = []
for name, path in datasets.items():
    npz = np.load(path)
    feats = np.concatenate([npz["D_vals"], npz["delta_vals"]], axis=1)
    feats = feats[np.all(np.isfinite(feats), axis=1)]  # remove NaN/Inf

    kind = 0 if name in real_sets else 1  # 0=Real, 1=Fake
    for row in feats:
        all_data.append({
            "D2": row[0], "D1": row[1], "D0": row[2],
            "Δ(1,2)": row[3], "Δ(0,1)": row[4],
            "label": kind
        })

df = pd.DataFrame(all_data)
print(df.shape)

feature_cols = ["D2", "D1", "D0", "Δ(1,2)", "Δ(0,1)"]
X = df[feature_cols].values
y = df["label"].values

# Normalize
X = MinMaxScaler().fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=115, stratify=y
)


# --- Train SVM ---
model = SVC(kernel="rbf", probability=True, random_state=115)
model.fit(X_train, y_train)

# Predict probabilities
y_scores = model.predict_proba(X_test)[:, 1]

#AUPRC
precision, recall, _ = precision_recall_curve(y_test, y_scores)
auprc = average_precision_score(y_test, y_scores)
print(f"AUPRC (Average Precision): {auprc:.4f}")


# --- Find optimal threshold for balanced accuracy ---
thresholds = np.linspace(0, 1, 200)
best_bal_acc = 0
best_thresh = 0.5

for t in thresholds:
    y_pred_t = (y_scores >= t).astype(int)
    bal_acc_t = balanced_accuracy_score(y_test, y_pred_t)
    if bal_acc_t > best_bal_acc:
        best_bal_acc = bal_acc_t
        best_thresh = t

print(f"Optimal threshold: {best_thresh:.3f}")
print(f"Balanced Accuracy at optimal threshold: {best_bal_acc:.4f}")


# --- Confusion Matrix ---
y_pred_opt = (y_scores >= best_thresh).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Real', 'Pred Fake'],
            yticklabels=['True Real', 'True Fake'])
plt.title(f"Confusion Matrix (Threshold={best_thresh:.3f})", fontsize=14)
plt.tight_layout()
plt.show()


# --- AUPRC ---
precision, recall, _ = precision_recall_curve(y_test, y_scores)
auprc = average_precision_score(y_test, y_scores)
baseline = np.mean(y_test)  # fraction of positives (fake images)

print(f"AUPRC (Average Precision): {auprc:.4f}")
print(f"Baseline (random chance): {baseline:.4f}")


# --- Plot Precision-Recall Curve ---
plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", linewidth=2)
plt.hlines(baseline, 0, 1, colors="gray", linestyles="--",
           label=f"Baseline = {baseline:.3f}")
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision-Recall Curve (SVM, RBF kernel)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# --- Decision Function ---

decision_scores = model.decision_function(X_test)
plt.figure(figsize=(6,4))
sns.histplot(decision_scores[y_test==0], color="blue", label="Real", kde=True, stat="density")
sns.histplot(decision_scores[y_test==1], color="red", label="Fake", kde=True, stat="density")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Decision function score")
plt.ylabel("Density")
plt.title("SVM Decision Function Distribution")
plt.legend()
plt.tight_layout()
plt.show()


# --- permutation test ---

# Compute permutation importance
permutation_importance_result = permutation_importance(
    model, X_test, y_test, scoring="average_precision", n_repeats=30, random_state=115
)

# Sort features by importance
sorted_idx = permutation_importance_result.importances_mean.argsort()

plt.figure(figsize=(8, 5))

# --- Bar plot ---
plt.barh(
    [feature_cols[i] for i in sorted_idx],
    permutation_importance_result.importances_mean[sorted_idx],
    xerr=permutation_importance_result.importances_std[sorted_idx],
    color='steelblue',
    edgecolor='black',
    alpha=0.85
)


plt.xlabel("Mean Decrease in AUPRC", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.title("Permutation Feature Importance (SVM, RBF kernel, AUPRC)", fontsize=16)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()


plt.savefig("feature_importance.pdf", bbox_inches="tight")
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")

plt.show()


print("Feature importances (mean ± std, decrease in AUPRC):")
for i in sorted_idx:
    print(f"{feature_cols[i]}: {permutation_importance_result.importances_mean[i]:.4f} ± {permutation_importance_result.importances_std[i]:.4f}")



