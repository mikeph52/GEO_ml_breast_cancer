import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
            break

y = np.array(labels)

print("Labels length:", len(y))

assert X.shape[0] == y.shape[0], \
    f"Mismatch: {X.shape[0]} samples vs {y.shape[0]} labels"

print("Samples and labels aligned")

# Train function
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

model = LogisticRegression(
    solver="saga",
    l1_ratio=0.5,   # elastic net
    C=0.5,
    max_iter=10000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

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

import os
os.makedirs("output", exist_ok=True)

X.to_csv("output/expression_matrix.csv")

labels_df = pd.DataFrame({
    "sample_id": X.index,
    "label": y
})
labels_df.to_csv("output/labels.csv", index=False)

X_selected = X[selected_genes]
X_selected.to_csv("output/expression_selected_genes.csv")

gene_importance.to_csv("output/important_genes.csv", index=False)

from sklearn.metrics import roc_curve, auc

def rocauc():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

from sklearn.decomposition import PCA

def pca_plot():
    X_all_sel = selector.transform(X)
    X_all_scaled = scaler.transform(X_all_sel)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all_scaled)

    plt.figure()
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label="Normal")
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Cancer")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (Selected Genes)")
    plt.legend()
    plt.show()

rocauc()
pca_plot()
