# Gene Expression–Based Breast Cancer Classifier
by mikeph_

A bioinformatics machine learning workflow using publicly available gene expression data from GEO.  
The goal is to classify cancer vs normal samples based on transcriptomic profiles and to practice proper evaluation, validation, and interpretation.

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

For more information visit [_Results_](docs/results.md)
---

## How to Run

### Install depedencies

```bash
pip install depedencies.txt
```

### Download datasets

Because of the file limits of github, the datasets are available [here](https://drive.google.com/file/d/1E3CJD42XcQLKKZkrxrXNmvxlvBsWBYUH/view?usp=sharing).
