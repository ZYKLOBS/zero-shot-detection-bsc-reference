import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

real_sets = ["Raise1k (real)", "openimgs (real)", "imagenet", "progan_real"]
fake_sets = ["Midjourney", "DALL-E2", "GLIDE", "SDXL", "S3GAN"]

# -------------------
# Load & prepare data
# -------------------
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


feature_cols = ["D2", "D1", "D0", "Δ(1,2)", "Δ(0,1)"]
X = df[feature_cols].values
y = df["label"].values

# Normalize
X = MinMaxScaler().fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=115, stratify=y #COD ZOMBIES ME BELOVED
)

# --- Train logistic regression ---
#model = LogisticRegression(max_iter=1000)
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# --- AUPRC ---
precision, recall, _ = precision_recall_curve(y_test, y_scores)
auprc = average_precision_score(y_test, y_scores)
print(f"AUPRC (Average Precision): {auprc:.4f}")

# --- Interval optimization ---
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


# --- Precision-Recall Curve ---
plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}", linewidth=2)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
#plt.title("Precision-Recall Curve (Logistic Regression L2)", fontsize=16)
plt.title("Precision-Recall Curve (Logistic Regression L1)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# --- Feature weights plot ---
weights = model.coef_[0]  # logistic regression coefficients
latex_labels = [r"$D^{(2)}$", r"$D^{(1)}$", r"$D^{(0)}$", r"$\Delta^{(1,2)}$", r"$\Delta^{(0,1)}$"]

plt.figure(figsize=(10, 6))
bars = plt.bar(
    range(len(weights)),
    weights,
    color=['red' if w < 0 else 'green' for w in weights]
)
plt.xticks(range(len(weights)), latex_labels, rotation=45, ha='right', fontsize=14)
plt.ylabel('Weight', fontsize=14)
plt.title('Logistic Regression Feature Importance (Weights)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

# Add weight values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.3f}',
        ha='center',
        va='bottom' if height > 0 else 'top',
        fontsize=12
    )

plt.tight_layout()
plt.show()
