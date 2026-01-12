import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt


# Load expression data (ignore GEO header lines)
expr = pd.read_csv(
    "data/GSE45827_series_matrix.txt",
    sep="\t",
    comment="!",
    index_col=0
)

# Transpose: samples × genes
X = expr.T

# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

print("Expression shape:", X.shape)

labels = []

with open("data/GSE45827_series_matrix.txt") as f:
    for line in f:
        if line.startswith("!Sample_characteristics_ch1"):
            values = line.strip().split("\t")[1:]
            for v in values:
                if ("cancer" in v.lower()) or ("tumor" in v.lower()):
                    labels.append(1)
                else:
                    labels.append(0)
            break  # IMPORTANT: stop after first matching line

y = np.array(labels)

print("Labels length:", len(y))
assert X.shape[0] == y.shape[0], \
    f"Mismatch: {X.shape[0]} samples vs {y.shape[0]} labels"

print("Samples and labels aligned")

# Train function
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

selector = SelectKBest(score_func=f_classif, k=200)

X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

print("Selected features:", X_train_sel.shape[1])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver="saga",
    l1_ratio=0.5,   # elastic net
    C=0.5,
    max_iter=10000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model,
    scaler.fit_transform(selector.fit_transform(X, y)),
    y,
    cv=cv,
    scoring="roc_auc"
)

print("CV ROC-AUC scores:", scores)
print("Mean ROC-AUC:", scores.mean())
print("Std ROC-AUC:", scores.std())

#permutation test
from sklearn.model_selection import permutation_test_score

score, perm_scores, pvalue = permutation_test_score(
    model,
    scaler.fit_transform(selector.fit_transform(X, y)),
    y,
    scoring="roc_auc",
    cv=cv,
    n_permutations=100,
    random_state=42
)

print("Permutation ROC-AUC:", score)
print("Permutation p-value:", pvalue)

# leave one out test
from sklearn.model_selection import LeaveOneOut, cross_val_score

loo = LeaveOneOut()

acc_scores = cross_val_score(
    model,
    scaler.fit_transform(selector.fit_transform(X, y)),
    y,
    cv=loo,
    scoring="accuracy"
)

print("LOOCV accuracy:", acc_scores.mean())


selected_genes = X.columns[selector.get_support()]
coef = model.coef_[0]

gene_importance = (
    pd.DataFrame({
        "Gene": selected_genes,
        "Weight": coef
    })
    .sort_values("Weight", ascending=False)
)

gene_importance.head(10)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

# Predict probabilities
y_prob = model.predict_proba(X_test_sel)[:, 1]

# Bootstrap ROC
n_bootstraps = 1000
rng = np.random.RandomState(42)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_test), len(y_test))
    if len(np.unique(y_test[indices])) < 2:
        continue

    fpr, tpr, _ = roc_curve(y_test[indices], y_prob[indices])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

# Statistics
mean_tpr = np.mean(tprs, axis=0)
std_tpr = np.std(tprs, axis=0)
mean_auc = np.mean(aucs)
std_auc = np.std(aucs)

# Plot
plt.figure(figsize=(5, 5))
plt.plot(mean_fpr, mean_tpr, color="black",
         label=f"ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")

plt.fill_between(
    mean_fpr,
    np.maximum(mean_tpr - std_tpr, 0),
    np.minimum(mean_tpr + std_tpr, 1),
    color="gray",
    alpha=0.3,
    label="±1 SD"
)

plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("output/roc_curve.pdf")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Feature selection + scaling
X_all_sel = selector.transform(X)
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all_sel)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_scaled)

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

# Plot
plt.figure(figsize=(5, 5))

plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Normal",
    alpha=0.7,
    edgecolor="black",
    s=50
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Cancer",
    alpha=0.7,
    edgecolor="black",
    s=50
)

plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
plt.title("PCA of Gene Expression (Selected Genes)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("output/pca_plot.pdf")
plt.show()

# -----------------------------------------------------
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve,
    auc
)

# Selected genes from training
train_genes = X.columns[selector.get_support()]

# Training data with selected genes
X_train_sel = X[train_genes]

# Fit scaler on TRAINING data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)

# Fit PCA on TRAINING data only
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

ext_csv = "data/GSE39004_series_matrix.txt"

# load dataset
expr_42568 = pd.read_csv(
    ext_csv,
    sep="\t",
    comment="!",
    index_col=0
)

# transpose to samples × genes
X_ext = expr_42568.T.apply(pd.to_numeric, errors="coerce")
#X_ext = X_ext.fillna(X_ext.mean())

X = X_ext.to_numpy(dtype=float)
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = col_means[inds[1]]

X_ext = pd.DataFrame(X, index=X_ext.index, columns=X_ext.columns)

print("External expression shape:", X_ext.shape)



# --- 1. Extract Sample Characteristics ---
characteristics = []

with open(ext_csv) as f:
    for line in f:
        if line.startswith("!Sample_characteristics_ch1"):
            values = [v.strip().strip('"') for v in line.rstrip().split("\t")[1:]]
            characteristics.append(values)

characteristics = np.array(characteristics)
print("Found characteristic rows:", characteristics.shape[0])

# --- 2. Identify tissue type row (robust) ---
sample_group_row = None
best_match_count = 0

for row in characteristics:
    match_count = sum(
        1 for v in row
        if v and v.lower().startswith("tissue type:")
    )
    if match_count > best_match_count:
        best_match_count = match_count
        sample_group_row = row

if sample_group_row is None or best_match_count == 0:
    raise RuntimeError("tissue type row not found in GEO file.")

print(f"Identified tissue type row with {best_match_count} matches")


# --- 3. Convert to binary labels (Tumor=1, Non-tumor=0) ---
cancer_groups = {"tumor"}
normal_groups = {"non-tumor"}

y_ext = []
valid_samples = []

sample_ids = X_ext.index.tolist()

for i, v in enumerate(sample_group_row):
    if not v or ":" not in v:
        continue

    group = v.split(":", 1)[1].strip().lower()

    if group in cancer_groups:
        y_ext.append(1)
        valid_samples.append(sample_ids[i])
    elif group in normal_groups:
        y_ext.append(0)
        valid_samples.append(sample_ids[i])
    # else: exclude other categories

y_ext = np.array(y_ext)

print("External labels (0=Normal, 1=Tumor):", np.bincount(y_ext))
print("Unique labels:", np.unique(y_ext))

assert len(np.unique(y_ext)) == 2, "Only one class found in external labels!"

# --- 4. Subset expression data to match labels ---
X_ext = X_ext.loc[valid_samples]

# --- 5. Align genes with training data ---
X_ext_common = X_ext.reindex(columns=train_genes, fill_value=0)

print("External data shape after gene alignment:", X_ext_common.shape)
print("Scaler expects features:", len(train_genes))

# --- 6. Scale using training scaler ---
X_ext_scaled = scaler.transform(X_ext_common)

print("External data successfully scaled.")

from collections import Counter

raw_groups = []

for v in sample_group_row:
    if v and ":" in v:
        raw_groups.append(v.split(":", 1)[1].strip().lower())

print(Counter(raw_groups))

assert X_ext_scaled.shape[0] == len(y_ext)
print("Sanity check passed.")
print("Final external feature matrix shape:", X_ext_scaled.shape)
print("Final external labels shape:", y_ext.shape)
print("Label balance (0=Normal, 1=Tumor):", np.bincount(y_ext))

from sklearn.metrics import precision_recall_curve, classification_report

y_ext_prob = model.predict_proba(X_ext_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_ext, y_ext_prob)

# --- Choose threshold with recall >= 0.8 and best precision ---
target_recall = 0.8

valid = np.where(recall[:-1] >= target_recall)[0]

if len(valid) == 0:
    raise RuntimeError("No threshold achieves the target recall.")

best_idx = valid[np.argmax(precision[valid])]
best_threshold = thresholds[best_idx]

print("Chosen threshold (recall ≥ 0.8):", best_threshold)
print("Precision at threshold:", precision[best_idx])
print("Recall at threshold:", recall[best_idx])

y_ext_pred = (y_ext_prob >= best_threshold).astype(int)

print(classification_report(y_ext, y_ext_pred, zero_division=0))

fpr, tpr, _ = roc_curve(y_ext, y_ext_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("External Validation ROC (GSE39004)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

X_ext_pca = pca.transform(X_ext_scaled)

plt.figure(figsize=(5,5))
plt.scatter(
    X_ext_pca[y_ext == 0, 0],
    X_ext_pca[y_ext == 0, 1],
    label="Normal",
    alpha=0.7,
    edgecolor="black"
)
plt.scatter(
    X_ext_pca[y_ext == 1, 0],
    X_ext_pca[y_ext == 1, 1],
    label="Cancer",
    alpha=0.7,
    edgecolor="black"
)

plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
plt.title("PCA External Dataset (GSE39004)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c="black",
    alpha=0.4,
    label="Training"
)

plt.scatter(
    X_ext_pca[:, 0],
    X_ext_pca[:, 1],
    c="red",
    alpha=0.7,
    label="GSE39004"
)

plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
plt.title("PCA: Training vs External Dataset")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()