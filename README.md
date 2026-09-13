# 🏦 Bank Customer Churn Prediction — Data Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-success?logo=lightgbm)
![SQL Server](https://img.shields.io/badge/Database-SQL%20Server-CC2927?logo=microsoftsqlserver&logoColor=white)
![Power BI](https://img.shields.io/badge/BI-Power%20BI-F2C811?logo=powerbi&logoColor=black)
![Orange](https://img.shields.io/badge/Data%20Mining-Orange-orange?logo=orangedatamining)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end churn analytics pipeline built as the **Google Colab / Python** stage of a
larger four-part *Bank Management System* project (SQL Server → Orange → **Google Colab**
→ Power BI). This repo covers cleaning, EDA, preprocessing, and model benchmarking on a
real bank credit-card customer dataset.

**[Overview](#-executive-summary) • [Dataset & Cleaning](#1-dataset--cleaning) •
[EDA](#2-exploratory-data-analysis) • [ML Pipeline](#3-preprocessing--modeling) •
[Results](#4-results) • [Project Context](#5-project-context-full-system) •
[Repo Structure](#6-repository-structure)**

---

## 📌 Executive Summary

Customer attrition on a credit-card portfolio was modeled end-to-end:

- Cleaned and typed **10,226 rows × 21 columns** of banking/behavioral data.
- Diagnosed churn drivers with univariate/bivariate EDA and a full correlation study.
- Balanced a **16.1% churn rate** with SMOTE and benchmarked **5 ML algorithms**.
- Tuned the top model (LightGBM) with `RandomizedSearchCV`, cutting missed churners
  (false negatives) from 114 (Logistic Regression baseline) down to **43**.
- Cross-validated the behavioral findings against a parallel **Orange Data Mining**
  workflow (Neural Network, AUC = 0.969) built by a teammate on the same dataset.

> All figures in this README are copied directly from executed notebook output —
> no estimated or rounded-up numbers.

---

## 1. Dataset & Cleaning

- **Source:** `BankChurners_data.csv`
- **Shape after cleaning:** 10,226 rows × 21 columns (2 Naive-Bayes leakage columns dropped on load)
- **Target:** `status` — `1` = Attrited Customer, `0` = Existing Customer
- **Class balance:** 83.9% Existing vs 16.1% Attrited

| Column | Nulls | % | Strategy |
|---|---|---|---|
| education | 1,532 | 15% | Filled with `Unknown` (explicit business label) |
| income | 1,125 | 11% | Filled with `Unknown` (explicit business label) |
| marital | 752 | 7% | Filled with `Unknown` (explicit business label) |

Imputation rules generalize to any dataset: drop columns >40% missing, fall back to mode
(or an explicit `Unknown` label for known business domains) for categoricals, and use
median (skewed, `|skew| > 1`) or mean (near-normal) for numeric columns. Duplicate rows
were removed with `drop_duplicates()`, and `client_id` was confirmed unique before being
excluded from modeling.

---

## 2. Exploratory Data Analysis

![EDA — Customer Profile & Behavioral Distributions](images/eda_distributions.png)

![EDA — Feature Correlations & Categorical Analysis](images/eda_correlations.png)

**Key findings:**
- `trans_count` and `trans_amt` are the strongest churn signals — attrited customers
  average far fewer transactions and spend less overall.
- `revolving_bal` and `util_ratio` spike toward zero right before churn — customers stop
  using their card before leaving.
- `products_count` separates churners clearly: 1–3 products for attrited vs. 3–6 for
  existing customers.
- `credit_limit` and `open_to_buy` are perfectly correlated (r = 1.00) — `open_to_buy`
  was dropped to avoid redundancy.

---

## 3. Preprocessing & Modeling

**Pipeline:** drop `client_id` → encode target (0/1) → label/ordinal/one-hot encode
categoricals (20 → 24 columns) → log-transform skewed features → stratified 75/25 split
(`X_train` (7,595, 22) / `X_test` (2,532, 22)) → `StandardScaler` fit on train only →
**SMOTE** on the training set.

| SMOTE | Existing | Attrited | Total |
|---|---|---|---|
| Before | 6,375 | 1,220 | 7,595 |
| After | 6,375 | 6,375 | 12,750 |

Five models were trained on the SMOTE-balanced training set and evaluated on the
untouched (imbalanced) test set.

---

## 4. Results

### Benchmark comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision | FN (missed churners) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8839 | 0.6195 | 0.7199 | 0.6659 | 0.9155 | 0.7163 | 114 |
| Decision Tree | 0.9155 | 0.7198 | 0.7764 | 0.7470 | 0.8593 | 0.5948 | 91 |
| Random Forest | 0.9518 | 0.8626 | 0.8329 | 0.8475 | 0.9843 | 0.9265 | 68 |
| XGBoost | 0.9648 | 0.9098 | 0.8673 | 0.8881 | 0.9918 | 0.9612 | 54 |
| LightGBM | 0.9696 | 0.9188 | 0.8894 | 0.9039 | 0.9912 | 0.9608 | 45 |
| **LightGBM (Tuned)** 🏆 | 0.9613 | 0.8687 | **0.8943** | 0.8814 | 0.9889 | 0.9470 | **43** |

**Tuning:** `RandomizedSearchCV` — 50 candidates × 5-fold CV, optimizing **recall**:

```text
Best Parameters: {'subsample': 0.8, 'num_leaves': 70, 'n_estimators': 300,
                   'min_child_samples': 10, 'max_depth': 10,
                   'learning_rate': 0.01, 'colsample_bytree': 0.6}
Best Recall (CV): 0.9691
```

The tuned model was selected as the **production candidate** because it minimizes missed
churners (43 FN vs. 45 for untuned LightGBM) — the metric that matters most for a
retention use case — even though its point numbers on other metrics are marginally lower
than the untuned model's. This is a deliberate precision/recall trade-off, not a
free upgrade.

### Top predictive features (LightGBM Tuned)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `trans_amt_log` | 3,232 |
| 2 | `trans_count` | 2,055 |
| 3 | `amt_change_q4_q1_log` | 2,028 |
| 4 | `products_count` | 1,853 |
| 5 | `inactive_months` | 1,492 |
| 6 | `revolving_bal` | 1,486 |
| 7 | `ct_change_q4_q1_log` | 1,439 |
| 8 | `contacts_count` | 1,382 |
| 9 | `age` | 1,185 |
| 10 | `credit_limit_log` | 1,107 |

### Business recommendations

1. Flag customers with fewer than ~40 transactions/year for early retention outreach.
2. Monitor `revolving_bal` dropping to 0 — a strong signal of imminent churn.
3. Cross-sell additional products to customers holding only 1–2 bank products.
4. Investigate customers with 5+ contacts in 12 months — often a dissatisfaction signal.
5. Re-engage customers after 3+ inactive months with targeted offers.


![Entity-Relationship Diagram](images/erd_diagram.png)

![Power BI — Bank Customer Churn Analysis dashboard](images/powerbi_dashboard_cover.png)

| Tier | Tool | Highlights (from the project presentation) |
|---|---|---|
| Database | SQL Server | 7 normalized tables, full FK referential integrity, IDENTITY PKs; 10 analytical queries (customer statements, branch performance, dormant accounts); AML fraud rule (`AVG + 3×STDEV` per branch) flagged 3 seeded suspicious transactions (95K, 88K, 75K EGP) |
| Data Mining | Orange | Best model: **Neural Network** — AUC = 0.969, CA = 93.8%, MCC = 0.764, 77.6% of churners caught at a 3.1% false-positive rate; top feature `Total_Trans_Ct` (χ² = 1166), consistent with `trans_count`/`trans_amt_log` topping the LightGBM importance list above |
| Modeling | Google Colab (this repo) | See [Results](#4-results) above |
| BI | Power BI | Overview KPIs (customers, churn rate, avg transaction), churn deep-dive by products/age/education/gender, inactivity × contact-count risk heatmap, interactive slicers |






