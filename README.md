# HDB Resale Price Prediction using Random Forest Regression

Predicting Singapore HDB resale flat prices using Random Forest Regression on ~170,000 transactions from 2017–2025, sourced from [data.gov.sg](https://data.gov.sg).

---
## Table of Contents
- [Overview](#overview)
- [Dataset](#Dataset)
- [Methodology](#methodology)
  - [Data Cleaning](#1-Data-Cleaning)
  - [Feature Engineering](#2-Feature-Engineering)
  - [Exploratory Data Analysis](#3-Exploratory-data-analysis)
  - [Modelling](#4-modelling)
  - [Hyperparameter Tuning](#5-hyperparameter-tuning)
- [Feature Importance](#feature-importance)
- [Model Iteration and Diagnostics](#model-iteration--diagnostics)
- [Limitations and Improvements](#limitations--future-improvements)
- [Requirements](#Requirements)


---

## Overview

This project aims to predict HDB resale prices using Random Forest Regression, estimating prices using location, flat type, remaining lease, floor level etc. An end-to-end regression pipeline is built to predict resale prices, covering data cleaning, exploratory data analysis, feature engineering, model training, diagnostics and hyperparameter tuning.

---

## Dataset

- **Source:** HDB Resale Flat Prices, data.gov.sg
- **Retrieved:** October 2025
- **Coverage:** January 2017 – October 2025
- **Size:** ~170,000 transactions
- **Features include:** Town, flat type, storey range, floor area (sqm), flat model, remaining lease, lease commencement date, resale price

---

## Methodology

### 1. Data Cleaning
- Checked for missing values (none found) and removed duplicate rows

### 2. Feature Engineering
Raw fields were converted into model-usable numeric features:

| Raw Field | Engineered Feature | Description |
|-----------|-------------------|-------------|
| `storey_range` (e.g. "07 TO 09") | `average_storey` | Midpoint of storey range |
| `remaining_lease` (e.g. "61 years 04 months") | `remaining_lease_in_months` | Total lease remaining in months |
| `month` (datetime) | `month_numeric` | Months elapsed since Jan 2017 |
| `month` (datetime) | `year` | Calendar year of transaction |

### 3. Exploratory Data Analysis
- Resale prices are right-skewed (skewness ≈ 0.95), driven by a small number of premium flats in prime locations
  <img width="518" height="405" alt="image" src="https://github.com/user-attachments/assets/488e0f96-673e-402c-8212-8ea869691ff3" />

- Remaining lease distribution / Lease Commence Year shows distinct peaks corresponding to major BTO construction periods in the 1980s, 1990s, and 2010s.
  <img width="529" height="402" alt="image" src="https://github.com/user-attachments/assets/8f19d5d6-c12f-4d91-aa94-caa7637c4e06" />

- Bukit Merah consistently appears in the top 5 towns by average resale price across most flat types (red dot on graph represents sale counts)
  <img width="992" height="1001" alt="image" src="https://github.com/user-attachments/assets/152c2113-9d26-47ef-85f6-a0148d12cbea" />

- Correlation matrix revealed two highly correlated feature pairs (`lease_commence_date` / `remaining_lease_in_months` at 0.99, and `year` / `month_numeric` at 0.99) — both pairs were retained after empirical testing showed including both improved model performance
  <img width="801" height="704" alt="image" src="https://github.com/user-attachments/assets/af803269-11b9-4532-b109-2fdb26bd105d" />


### 4. Modelling

**Algorithm:** Random Forest Regressor  
**Encoding:** Target Encoding for categorical features (`town`, `flat_type`, `flat_model`) — each category is replaced by its mean resale price from the training set, preserving meaningful price information without creating high-dimensional dummy columns  
**Train/test split:** Temporal — earliest ~80% of transactions used for training, most recent used ~20% for testing. This mirrors real-world usage where the model predicts future prices from historical data.

**Final feature set:**
```python
features = ['town', 'flat_type', 'average_storey', 'floor_area_sqm',
            'flat_model', 'lease_commence_date', 'remaining_lease_in_months',
            'month_numeric', 'year']
```



### 5. Hyperparameter Tuning

`RandomizedSearchCV` (3-fold CV, 30 iterations) was used in place of `GridSearchCV` to efficiently search a wider parameter space within a fixed compute budget:

```python
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [8, 10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.5, 0.8, 1]
}
```

**Optimal parameters found:** `n_estimators=300`, `max_depth=None`, `max_features=0.5`, `min_samples_split=2`, `min_samples_leaf=1`

Tuning improved test MAE from $55,613 → $51,722 and test R² from 0.840 → 0.862.

---

## Results

| Metric | Baseline RF | Tuned RF |
|--------|------------|----------|
| MAE    | $52,813.    | $51,722  |
| RMSE   | $74,812    | $72,991  |
| R²     | 0.855      | 0.862    |

Baseline  
<img width="815" height="219" alt="image" src="https://github.com/user-attachments/assets/b5530a54-c469-4784-9a8f-893cb35ae1c6" />  
Tuned  
<img width="815" height="219" alt="image" src="https://github.com/user-attachments/assets/8f1a5d9f-ddd8-40d1-9806-65c11db0f60d" />

*Test set: ~20% of data, temporal split (transactions from approximately mid-2024 onwards)*

The tuned model predicts HDB resale prices within ~$51,700 on average, and explains 86% of the variance in resale prices on unseen data.

---

## Feature Importance

`flat_model` was the strongest predictor of resale price, followed by `town`. `average_storey` and `month_numeric` tied for third, reflecting the combined importance of physical flat characteristics and market timing in determining price.
<img width="573" height="628" alt="image" src="https://github.com/user-attachments/assets/2de3516d-0d18-4cd1-85bb-18c0b24b862d" />

---

### Model Iteration & Diagnostics

Encoder Choice:  
Started with OrdinalEncoder before switching to Target Encoding. Test R² score decreased to 0.58 implying an issue separate from switching encoders. Checks found that the mean resale prices increased over time (~47% between 2017 and 2025), which resulted in the choice to add `month_numeric` and `year` as explicit time features.

Test R² recovered to 0.848 after adding `month_numeric` and `year`.

Two correlated feature pairs (`year`/`month_numeric` and `lease_commence_date`/`remaining_lease_in_months`) with 0.99 correlation were kept in the feature list. After testing, it was determined that test R² score improved when both pairs were included, showing that high correlation between features does not necessarily imply redundancy in a tree-based model.

## Limitations & Future Improvements

1. **Log-transform the target** — resale prices are right-skewed (skewness ≈ 0.95). Training on log-transformed prices and back-transforming predictions could reduce the influence of high-value outliers on model training
2. **Add a validation set** — several modelling decisions (encoding choice, feature additions) were evaluated using the test set, which risks mild optimistic bias in reported metrics. A dedicated validation set would reserve the test set strictly for final evaluation
3. **Additional Features** — proximity to MRT stations, schools, and amenities can be used to further refine accuracy of resale price prediction
4. **Benchmark against gradient boosting** — XGBoost or LightGBM may outperform Random Forest on this dataset


---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn>=1.3
joblib
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```



