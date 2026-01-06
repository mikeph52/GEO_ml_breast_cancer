# Breast Cancer Gene Expression Predictor
by mikeph_

This project demonstrates a complete beginner-to-intermediate bioinformatics machine learning workflow using publicly available gene expression data from GEO.  
The goal is to classify **cancer vs normal samples** based on transcriptomic profiles and to practice proper evaluation, validation, and interpretation.

---

## Project Overview

- **Dataset:** GEO accession `GSE45827`
- **Task:** Binary classification (Cancer vs Normal)
- **Data type:** Microarray gene expression
- **Samples:** 155
- **Genes:** ~30,000
- **Model:** Logistic Regression
- **Language:** Python

---

## Dataset Description

- Source: **Gene Expression Omnibus (GEO)**
- Platform: Microarray
- Labels were extracted from `!Sample_characteristics_ch1` metadata
- Samples containing keywords such as **"cancer"** or **"tumor"** were labeled as cancer

---

## Methods

### Preprocessing
- Transposed expression matrix to `samples × genes`
- Converted all values to numeric
- Missing values filled using gene-wise means

### Feature Selection
- Removed low-variance genes using `VarianceThreshold`
- Reduced dimensionality from ~30,000 to ~200 genes

### Model
- Logistic Regression
- Evaluated using:
  - Train/test split
  - Stratified k-fold cross-validation
  - Permutation testing

---

## Results
### On Training Dataset:

| Metric | Value |
|------|------|
| Accuracy | 1.00 |
| ROC-AUC | 1.00 |
| CV ROC-AUC | 1.00 ± 0.00 |
| Permutation Test p-value | ~0.01 |

**Interpretation:**  
The classifier perfectly separates cancer and normal samples, indicating a strong biological signal in gene expression data. Permutation testing confirms that performance is statistically significant and not due to chance.

---

##  Visualizations

- **PCA plot** shows clear separation between cancer and normal samples
- **ROC curve** demonstrates near-perfect classification performance

---

## ⚠️ Notes on Evaluation

- ROC-AUC is **not defined for Leave-One-Out CV** in imbalanced datasets
- Accuracy was used instead for LOOCV
- This behavior is expected and reflects known statistical limitations

---

## Key Outputs

- `important_genes.csv`: genes ranked by logistic regression coefficients
- `expression_selected_genes.csv`: reduced feature matrix
- Exported train/test splits for reproducibility

---
## Results on external datasets
### GSE42568:

  | Class        | Precision | Recall | F1‑score | Support |
  | ------------ | --------- | ------ | -------- | ------- |
  | 0            | 0.61      | 0.82   | 0.70     | 17      |
  | 1            | 0.97      | 0.91   | 0.94     | 104     |
  | Macro avg    | 0.79      | 0.87   | 0.82     | 121     |
  | Weighted avg | 0.92      | 0.90   | 0.91     | 121     |

- ### ROC-AUC
  ![](results_photos/GSE42568-roc.png)

- ### PCA
  ![](results_photos/GSE42568-pca.png)

- ### PCA Training Data vs GSE42568
  ![](results_photos/trainvsGSE42568.png)

### GSE61304:

[GEO NCBI](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE61304)

- ### Characteristics
  ```
  Found characteristic rows: 11
  Selected diagnosis row index: 1
  Example diagnosis values: ['diagnosis: Adenocarcinoma of breast, ductal'
  'diagnosis: Adenocarcinoma of breast, ductal, mucinous'
  'diagnosis: Within normal limits'
  'diagnosis: Adenocarcinoma of breast, ductal'
  'diagnosis: Adenocarcinoma of breast, ductal']
  External labels (0=Normal, 1=Cancer): [ 4 58]
  Common genes: 200
  ```

  | Class        | Precision | Recall | F1‑score | Support |
  | ------------ | --------- | ------ | -------- | ------- |
  | 0            | 0.00      | 0.00   | 0.00     | 4       |
  | 1            | 0.93      | 0.86   | 0.89     | 58      |
  | Macro avg    | 0.46      | 0.43   | 0.45     | 62      |
  | Weighted avg | 0.87      | 0.81   | 0.84     | 62      |


- ### ROC-AUC
  ![](results_photos/GSE61304-roc.png)

- ### PCA
  ![](results_photos/GSE61304-pca.png)

- ### PCA Training Data vs GSE61304
  ![](results_photos/trainvsGSE61304.png)

---
## How to Run

```bash
conda create -n bio_ml python=3.11
conda activate bio_ml
pip install pandas numpy scikit-learn matplotlib

