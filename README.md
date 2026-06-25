# HDB Resale Price Prediction using Random Forest Regression

Predicting Singapore HDB resale flat prices using Random Forest Regression on ~170,000 transactions from 2017–2025, sourced from [data.gov.sg](https://data.gov.sg).

---
## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Cleaning](###Data-Cleaning)
  - [Feature Engineering](#Feature-Engineering)
  - [Exploratory Data Analysis]
  - [Modelling]
  - [Model Iteration and Diagnostics]
  - [Hyperparameter Tuning]
- [Feature Importance]
- [Limitations and Improvements]







---

## Overview

Singapore's HDB resale market is highly active, with prices influenced by location, flat type, remaining lease, floor level, and broader market trends over time. This project builds an end-to-end regression pipeline to predict resale prices, covering data cleaning, exploratory data analysis, feature engineering, model training, diagnostics, hyperparameter tuning, and model persistence.

---

## Dataset

- **Source:** HDB Resale Flat Prices, data.gov.sg
- **Retrieved:** October 2025
- **Coverage:** January 2017 – October 2025
- **Size:** ~170,000 transactions after deduplication
- **Features include:** Town, flat type, storey range, floor area (sqm), flat model, remaining lease, lease commencement date, resale price

---

## Project Structure

```
├── regression_cleaned.ipynb    # Main notebook
├── final_model.pkl             # Saved trained pipeline (preprocessor + regressor)
└── README.md
```

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
- Remaining lease distribution shows distinct peaks corresponding to major BTO construction periods in the 1980s, 1990s, and post-2015
- Bukit Merah consistently appears in the top 5 towns by average resale price across nearly all flat types
- Correlation matrix revealed two highly correlated feature pairs (`lease_commence_date` / `remaining_lease_in_months` at 0.99, and `year` / `month_numeric` at 0.99) — both pairs were retained after empirical testing showed including both improved model performance

### 4. Modelling

**Algorithm:** Random Forest Regressor  
**Encoding:** Target Encoding for categorical features (`town`, `flat_type`, `flat_model`) — each category is replaced by its mean resale price from the training set, preserving meaningful price information without creating high-dimensional dummy columns  
**Train/test split:** Temporal — earliest ~80% of transactions used for training, most recent ~20% for testing. This mirrors real-world usage where the model predicts future prices from historical data, avoiding lookahead bias that a random split would introduce

**Final feature set:**
```python
features = ['town', 'flat_type', 'average_storey', 'floor_area_sqm',
            'flat_model', 'lease_commence_date', 'remaining_lease_in_months',
            'month_numeric', 'year']
```

### 5. Model Iteration & Diagnostics

Switching from OrdinalEncoder to Target Encoding initially produced a strong train R² (0.94) but poor test R² (0.58) — a larger gap than expected from encoding choice alone. Diagnosis revealed two compounding issues: Target Encoding locks in historical price levels from training data only, and Random Forest cannot extrapolate beyond price ranges seen during training. As resale prices rose ~47% between 2017 and 2025, the model systematically underpredicted test-period prices.

This was resolved by adding `month_numeric` and `year` as explicit time features, allowing the model to learn the upward price trend directly rather than relying on frozen historical encodings. Test R² recovered from 0.58 → 0.83 → 0.848 with each addition.

Notably, both correlated feature pairs (`year`/`month_numeric` and `lease_commence_date`/`remaining_lease_in_months`) were empirically tested rather than dropped based on correlation alone — in both cases, retaining both members of the pair improved test R², demonstrating that high correlation between features does not necessarily imply redundancy in a tree-based model.

### 6. Hyperparameter Tuning

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
| MAE    | $55,613    | $51,722  |
| RMSE   | $78,709    | $72,991  |
| R²     | 0.840      | 0.862    |

*Test set: ~20% of data, temporal split (transactions from approximately mid-2024 onwards)*

**In plain terms:** The tuned model predicts HDB resale prices within ~$51,700 on average, and explains 86% of the variance in resale prices on unseen data.

---

## Feature Importance

`flat_model` was the strongest predictor of resale price, followed by `town`. `average_storey` and `month_numeric` tied for third, reflecting the combined importance of physical flat characteristics and market timing in determining price.

---

## Limitations & Improvements

1. **Log-transform the target** — resale prices are right-skewed (skewness ≈ 0.95). Training on log-transformed prices and back-transforming predictions could reduce the influence of high-value outliers on model training
2. **Add a validation set** — several modelling decisions (encoding choice, feature additions) were evaluated using the test set, which risks mild optimistic bias in reported metrics. A dedicated validation set would reserve the test set strictly for final evaluation
3. **Geospatial features** — proximity to MRT stations, schools, and amenities are known drivers of HDB resale prices and are publicly available via LTA DataMall and OneMap API
4. **Benchmark against gradient boosting** — XGBoost or LightGBM may outperform Random Forest on this dataset, particularly given the temporal distribution shift between training and test periods
5. **Wrap feature engineering into the pipeline** — currently, raw data must be pre-processed (storey range conversion, lease parsing, date features) before being passed to the saved model. Encapsulating these steps in a custom `FunctionTransformer` as the first pipeline step would allow true raw-to-prediction inference

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



