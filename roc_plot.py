"""
Combined Breast Cancer Gene Expression Analysis
================================================
Trains a single model on GSE45827 and validates it against five external
GEO datasets (GSE10810, GSE39004, GSE42568, GSE61304, GSE65194), producing:

  1. One combined ROC plot  -> output/combined_roc_all_datasets.pdf
     - a pooled ROC curve using every external sample from all 5 datasets
     - individual per-dataset ROC curves overlaid for comparison

  2. One combined PCA plot  -> output/combined_pca_all_datasets.pdf
     - training samples (background) + all 5 external datasets projected
       into the same PCA space, colored by dataset and shaped by class

Expected folder layout (same convention as the original notebooks):
    data/GSE45827_series_matrix.txt   (training set)
    data/GSE10810_series_matrix.txt
    data/GSE39004_series_matrix.txt
    data/GSE42568_series_matrix.txt
    data/GSE61304_series_matrix.txt
    data/GSE65194_series_matrix.txt

Output files are written to ./output/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

os.makedirs("output", exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_expression(path):
    """Load a GEO series matrix file, transpose to samples x genes, impute NaNs.

    Uses numpy-based mean imputation instead of DataFrame.fillna(X.mean()),
    which is extremely slow on wide matrices (tens of thousands of gene
    columns) because pandas performs it as a per-column assignment internally.
    """
    expr = pd.read_csv(path, sep="\t", comment="!", index_col=0)
    X_df = expr.T.apply(pd.to_numeric, errors="coerce")

    arr = X_df.to_numpy(dtype=float, copy=True)
    col_means = np.nanmean(arr, axis=0)
    # Columns that are entirely NaN produce nan means; fall back to 0 for those
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])

    X = pd.DataFrame(arr, index=X_df.index, columns=X_df.columns)
    return X


def get_characteristics_rows(path):
    """Return all '!Sample_characteristics_ch1' rows as a list of lists of strings."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                values = [v.strip().strip('"') for v in line.rstrip().split("\t")[1:]]
                rows.append(values)
    return rows


# ---------------------------------------------------------------------------
# Dataset-specific label extraction
# (kept faithful to the logic used in each original notebook)
# ---------------------------------------------------------------------------

def labels_generic_cancer_keyword(path):
    """Used for training set (GSE45827) and GSE42568:
    first characteristics row, label 1 if 'cancer'/'tumor' keyword present."""
    labels = []
    with open(path) as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                values = line.strip().split("\t")[1:]
                for v in values:
                    labels.append(1 if ("cancer" in v.lower() or "tumor" in v.lower()) else 0)
                break
    return np.array(labels), None  # None = no sample filtering needed


def labels_gse10810(path):
    """GSE10810: 'tumor (t) vs healthy (s):' row, last char T/S."""
    rows = get_characteristics_rows(path)
    diagnosis_row = None
    for row in rows:
        if any("tumor (t) vs healthy (s):" in v.lower() for v in row):
            diagnosis_row = row
            break
    if diagnosis_row is None:
        raise RuntimeError("GSE10810: diagnosis row not found.")
    y = np.array([0 if v.strip()[-1].upper() == "S" else 1 for v in diagnosis_row])
    return y, None


def labels_gse39004(path, X_ext):
    """GSE39004: 'tissue type: tumor / non-tumor' row, with sample filtering."""
    rows = get_characteristics_rows(path)
    sample_group_row, best_match_count = None, 0
    for row in rows:
        match_count = sum(1 for v in row if v and v.lower().startswith("tissue type:"))
        if match_count > best_match_count:
            best_match_count = match_count
            sample_group_row = row
    if sample_group_row is None or best_match_count == 0:
        raise RuntimeError("GSE39004: tissue type row not found.")

    cancer_groups = {"tumor"}
    normal_groups = {"non-tumor"}
    y, valid_samples = [], []
    sample_ids = X_ext.index.tolist()
    for i, v in enumerate(sample_group_row):
        if not v or ":" not in v:
            continue
        group = v.split(":", 1)[1].strip().lower()
        if group in cancer_groups:
            y.append(1)
            valid_samples.append(sample_ids[i])
        elif group in normal_groups:
            y.append(0)
            valid_samples.append(sample_ids[i])
    return np.array(y), valid_samples  # valid_samples used to subset X_ext by index label


def labels_gse61304(path):
    """GSE61304: 'diagnosis:' row (excluding age-at-diagnosis), 'within normal limits' = healthy."""
    rows = get_characteristics_rows(path)
    diagnosis_row = None
    for row in rows:
        matches = [
            v.lower().startswith("diagnosis:")
            and not v.lower().startswith("diagnosis: age")
            and not v.lower().startswith("age at diagnosis:")
            for v in row
        ]
        if sum(matches) > len(row) * 0.5:
            diagnosis_row = row
            break
    if diagnosis_row is None:
        raise RuntimeError("GSE61304: diagnosis row not found.")
    y = np.array([0 if "within normal limits" in v.lower() else 1 for v in diagnosis_row])
    return y, None


def labels_gse65194(path, X_ext):
    """GSE65194: 'sample_group:' row, subtype names = cancer, 'healthy' = normal, else excluded."""
    rows = get_characteristics_rows(path)
    sample_group_row = None
    for row in rows:
        if any(v.lower().startswith("sample_group:") for v in row):
            sample_group_row = row
            break
    if sample_group_row is None:
        raise RuntimeError("GSE65194: sample_group row not found.")

    cancer_groups = {"tnbc", "her2", "luminal a", "luminal b"}
    normal_groups = {"healthy"}
    y, valid_idx = [], []
    for i, v in enumerate(sample_group_row):
        group = v.split(":")[1].strip().lower()
        if group in cancer_groups:
            y.append(1)
            valid_idx.append(i)
        elif group in normal_groups:
            y.append(0)
            valid_idx.append(i)
    return np.array(y), valid_idx  # valid_idx used to subset X_ext by positional index


# ---------------------------------------------------------------------------
# 1. Train on GSE45827
# ---------------------------------------------------------------------------

TRAIN_PATH = "data/GSE45827_series_matrix.txt"

print("Loading training data (GSE45827)...")
X = load_expression(TRAIN_PATH)
y, _ = labels_generic_cancer_keyword(TRAIN_PATH)

assert X.shape[0] == y.shape[0], f"Mismatch: {X.shape[0]} samples vs {y.shape[0]} labels"
print("Training expression shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

selector = SelectKBest(score_func=f_classif, k=200)
X_train_sel = selector.fit_transform(X_train, y_train)
train_genes = X.columns[selector.get_support()]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)

model = LogisticRegression(
    solver="saga", l1_ratio=0.5, C=0.5, max_iter=10000, random_state=RANDOM_STATE
)
model.fit(X_train_scaled, y_train)
print("Model trained on", X_train_scaled.shape[0], "samples,", X_train_scaled.shape[1], "features.")

# PCA fit on all training data (selected genes), used as the common projection space
X_all_sel = selector.transform(X)
X_all_scaled = scaler.fit_transform(X_all_sel)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_all_scaled)
pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100

# Refit scaler on the training data only (used consistently for external transforms)
scaler = StandardScaler()
X_train_scaled_for_ext = scaler.fit_transform(X[train_genes])


# ---------------------------------------------------------------------------
# 2. Validate on each external dataset
# ---------------------------------------------------------------------------

EXTERNAL_DATASETS = {
    "GSE10810": ("data/GSE10810_series_matrix.txt", labels_gse10810),
    "GSE39004": ("data/GSE39004_series_matrix.txt", labels_gse39004),
    "GSE42568": ("data/GSE42568_series_matrix.txt", labels_generic_cancer_keyword),
    "GSE61304": ("data/GSE61304_series_matrix.txt", labels_gse61304),
    "GSE65194": ("data/GSE65194_series_matrix.txt", labels_gse65194),
}

all_y_true = []
all_y_prob = []
all_dataset_tags = []
per_dataset_results = {}   # name -> (fpr, tpr, auc)
per_dataset_pca = {}       # name -> (X_pca, y)

for name, (path, label_fn) in EXTERNAL_DATASETS.items():
    print(f"\nProcessing external dataset {name}...")
    X_ext = load_expression(path)

    # Each dataset's label function has a slightly different signature/behavior
    if label_fn is labels_generic_cancer_keyword:
        y_ext, _ = label_fn(path)
    elif label_fn is labels_gse39004 or label_fn is labels_gse65194:
        y_ext, valid = label_fn(path, X_ext)
        if label_fn is labels_gse39004:
            X_ext = X_ext.loc[valid]          # valid = sample IDs
        else:
            X_ext = X_ext.iloc[valid]         # valid = positional indices
    else:
        y_ext, _ = label_fn(path)

    if X_ext.shape[0] != len(y_ext):
        print(f"  Warning: {name} sample/label mismatch "
              f"({X_ext.shape[0]} vs {len(y_ext)}); truncating to match.")
        n = min(X_ext.shape[0], len(y_ext))
        X_ext = X_ext.iloc[:n]
        y_ext = y_ext[:n]

    # Align genes with training set, scale with training scaler
    X_ext_common = X_ext.reindex(columns=train_genes, fill_value=0)
    X_ext_scaled = scaler.transform(X_ext_common)

    # Predict
    y_ext_prob = model.predict_proba(X_ext_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_ext, y_ext_prob)
    roc_auc = auc(fpr, tpr)
    per_dataset_results[name] = (fpr, tpr, roc_auc)

    all_y_true.append(y_ext)
    all_y_prob.append(y_ext_prob)
    all_dataset_tags.extend([name] * len(y_ext))

    # PCA projection into training PCA space
    X_ext_pca = pca.transform(X_ext_scaled)
    per_dataset_pca[name] = (X_ext_pca, y_ext)

    print(f"  {name}: n={len(y_ext)}, AUC={roc_auc:.3f}")


# ---------------------------------------------------------------------------
# 3. Combined ROC plot (pooled across all external datasets + per-dataset overlays)
# ---------------------------------------------------------------------------

y_true_all = np.concatenate(all_y_true)
y_prob_all = np.concatenate(all_y_prob)

fpr_pooled, tpr_pooled, _ = roc_curve(y_true_all, y_prob_all)
auc_pooled = auc(fpr_pooled, tpr_pooled)

plt.figure(figsize=(6, 6))
plt.plot(
    fpr_pooled, tpr_pooled, color="black", linewidth=2.5,
    label=f"Pooled (all datasets), AUC = {auc_pooled:.2f}"
)

colors = plt.cm.tab10.colors
for (name, (fpr, tpr, roc_auc)), color in zip(per_dataset_results.items(), colors):
    plt.plot(fpr, tpr, color=color, alpha=0.8, linewidth=1.2,
              label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All External Validation Datasets")
plt.legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("output/combined_roc_all_datasets.pdf")
plt.show()
print("\nSaved combined ROC plot -> output/combined_roc_all_datasets.pdf")
print(f"Pooled AUC across all {len(y_true_all)} external samples: {auc_pooled:.3f}")


# ---------------------------------------------------------------------------
# 4. Combined PCA plot (training background + all external datasets)
# ---------------------------------------------------------------------------

plt.figure(figsize=(7, 6))

# Training samples as light gray background
plt.scatter(
    X_train_pca[:, 0], X_train_pca[:, 1],
    c="lightgray", alpha=0.5, s=30, label="Training (GSE45827)"
)

markers = ["o", "s", "^", "D", "v"]
for (name, (X_ext_pca, y_ext)), color, marker in zip(per_dataset_pca.items(), colors, markers):
    plt.scatter(
        X_ext_pca[y_ext == 0, 0], X_ext_pca[y_ext == 0, 1],
        facecolor="none", edgecolor=color, marker=marker, s=45, alpha=0.8,
        label=f"{name} - Normal"
    )
    plt.scatter(
        X_ext_pca[y_ext == 1, 0], X_ext_pca[y_ext == 1, 1],
        facecolor=color, edgecolor="black", marker=marker, s=45, alpha=0.8,
        label=f"{name} - Cancer"
    )

plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
plt.title("PCA - Training vs All External Datasets")
plt.legend(frameon=False, fontsize=7, loc="best", ncol=1)
plt.tight_layout()
plt.savefig("output/combined_pca_all_datasets.pdf")
plt.show()
print("Saved combined PCA plot -> output/combined_pca_all_datasets.pdf")