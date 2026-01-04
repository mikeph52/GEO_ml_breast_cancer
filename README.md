# Breast Cancer Gene Expression Predictor
by mikep_

This project demonstrates a complete **beginner-to-intermediate bioinformatics machine learning workflow** using publicly available gene expression data from GEO.  
The goal is to classify **cancer vs normal samples** based on transcriptomic profiles and to practice proper evaluation, validation, and interpretation.

---

## 📌 Project Overview

- **Dataset:** GEO accession `GSE45827`
- **Task:** Binary classification (Cancer vs Normal)
- **Data type:** Microarray gene expression
- **Samples:** 155
- **Genes:** ~30,000
- **Model:** Logistic Regression
- **Language:** Python

This project covers:
- Data loading and preprocessing
- Label extraction from GEO metadata
- Feature filtering
- Model training
- Cross-validation
- Permutation testing
- Visualization (PCA, ROC curve)
- Exporting results for reproducibility

---

## 📂 Repository Structure

├── data/
│ └── GSE45827_series_matrix.txt
├── output/
│ ├── expression_matrix.csv
│ ├── expression_selected_genes.csv
│ ├── labels.csv
│ ├── important_genes.csv
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
├── gene_expression_ml.ipynb
├── README.md

---

## 🔬 Dataset Description

- Source: **Gene Expression Omnibus (GEO)**
- Platform: Microarray
- Labels were extracted from `!Sample_characteristics_ch1` metadata
- Samples containing keywords such as **"cancer"** or **"tumor"** were labeled as cancer

---

## 🧪 Methods

### 1️⃣ Preprocessing
- Transposed expression matrix to `samples × genes`
- Converted all values to numeric
- Missing values filled using gene-wise means

### 2️⃣ Feature Selection
- Removed low-variance genes using `VarianceThreshold`
- Reduced dimensionality from ~30,000 to ~200 genes

### 3️⃣ Model
- Logistic Regression
- Evaluated using:
  - Train/test split
  - Stratified k-fold cross-validation
  - Permutation testing

---

## 📊 Results

| Metric | Value |
|------|------|
| Accuracy | 1.00 |
| ROC-AUC | 1.00 |
| CV ROC-AUC | 1.00 ± 0.00 |
| Permutation Test p-value | ~0.01 |

**Interpretation:**  
The classifier perfectly separates cancer and normal samples, indicating a strong biological signal in gene expression data. Permutation testing confirms that performance is statistically significant and not due to chance.

---

## 📈 Visualizations

- **PCA plot** shows clear separation between cancer and normal samples
- **ROC curve** demonstrates near-perfect classification performance

---

## ⚠️ Notes on Evaluation

- ROC-AUC is **not defined for Leave-One-Out CV** in imbalanced datasets
- Accuracy was used instead for LOOCV
- This behavior is expected and reflects known statistical limitations

---

## 🧬 Key Outputs

- `important_genes.csv`: genes ranked by logistic regression coefficients
- `expression_selected_genes.csv`: reduced feature matrix
- Exported train/test splits for reproducibility

---

## 🚀 How to Run

```bash
conda create -n bio_ml python=3.11
conda activate bio_ml
pip install pandas numpy scikit-learn matplotlib

