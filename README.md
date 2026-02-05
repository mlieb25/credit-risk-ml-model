# Credit Risk ML Model – Home Credit Style (US Immigrant \& Student Loans)

This project implements an end‑to‑end **consumer credit risk model** inspired by the *Home Credit – Credit Risk Model Stability* competition. The focus is on building a **production‑grade, temporally stable model** for **US immigrants and international students** using real‑world style loan application data.

The repository demonstrates the full lifecycle of a credit risk model:

- Exploratory Data Analysis (EDA) on a massive, multi‑table dataset
- Data reduction and risk‑focused aggregation
- Master feature merging and memory optimization
- Time‑aware train/test splitting and preprocessing
- Baseline (no‑ML) benchmarks
- Gradient boosting model selection (LightGBM) with temporal CV
- Final evaluation and prediction driver (SHAP) analysis

***

## 1. Repository Structure

```text
Credit-Risk-ML-Model/
├── 01_analyze_individual_tables.py
├── 02_eda_tier1_base_static.py
├── 03_eda_tier2_bureau_applprev.py
├── 04_eda_tier3_person.py
├── 05_reduce_person_to_unique_cases.py
├── 06_reduce_person_coapplicant.py
├── 07_reduce_bureau_aggregation.py
├── 08_reduce_applprev_aggregation.py
├── 09_master_merge.py
├── 10_optimize_dataset.py
├── 11_sort_and_split_data.py
├── 12_preprocess_pipeline.py
├── 13_baseline_no_ml.py
├── 14_train_model_selection.py
├── 15_evaluate_final_model.py
├── 16_visualize_prediction_drivers.py
└── README.md
```

The raw Home Credit–style CSV/Parquet files are expected in a local data folder (paths are configurable inside each script).

***

## 2. Project Objective

The objective is to build an **Alternative Stability Engine (ASE)** that:

- Predicts the **probability of default (PD)** for consumer loans
- Safely **expands approvals** for thin‑file US immigrants and international students
- Maintains **default rates below ~4%** with **stable performance over time**
- Provides **explainable** decisions suitable for regulatory and business review

***

## 3. End‑to‑End Pipeline Overview

The pipeline is deliberately broken into small, composable scripts. You can run them step‑by‑step or adapt individual stages into your own workflow.

### 3.1 EDA and Understanding the Dataset

**01_analyze_individual_tables.py**

- Scans the training data directory for all CSV files.
- For each table, reports:
    - Row/column counts and memory usage
    - Presence and cardinality of `case_id`
    - Column types and missing value percentages
- Writes:
    - `[table]_column_stats.csv` – per‑column stats
    - `[table]_report.txt` – human‑readable summary
    - `all_tables_summary.csv` and `master_summary.txt` – cross‑table summary

**02_eda_tier1_base_static.py**

- Focuses on core tables:
    - `train_base.csv` (application + target)
    - `train_static_0_0.csv` (core static features)
- Analyzes:
    - Target distribution and class imbalance
    - Temporal structure (by `WEEK_NUM`, `MONTH`, `date_decision`)
    - Missing values and column‑level statistics
- Generates plots:
    - Target distribution
    - Application volume and default rate over time

**03_eda_tier2_bureau_applprev.py**

- Explores:
    - Credit bureau (external history)
    - Previous applications (internal history)
- Quantifies:
    - One‑to‑many cardinality per `case_id`
    - Distribution of debt, delinquency, and loan counts
- Informs aggregation design (worst case, totals, and counts).

**04_eda_tier3_person.py**

- Analyzes the `person` table:
    - Primary vs co‑applicants
    - Demographics and employment attributes
- Confirms that a single `case_id` can have multiple person rows and motivates later reductions.

***

### 3.2 Reduction and Risk‑Focused Aggregation

**05_reduce_person_to_unique_cases.py**

- Reduces the `person` table to a **single primary applicant** per `case_id`.
- Keeps key demographic and employment features.
- Outputs: `train_person_primary.csv` (1 row per case).

**06_reduce_person_coapplicant.py**

- Extracts **co‑applicants** (non‑primary persons).
- Aggregates to one co‑applicant record per `case_id` (when present).
- Prefixes all features with `coapp_` to avoid collisions.
- Outputs: `train_person_coapplicant.csv`.

**07_reduce_bureau_aggregation.py**

- Loads and concatenates multiple depth‑1 bureau files.
- Groups by `case_id` and computes risk‑focused aggregations:
    - **DPD (days past due)**: `max`, `mean`, `std`, `count_nonzero`
    - **Debt/Amounts**: `max`, `sum`, `mean`, `count_positive`
    - **Contract counts**: `sum`, `max`, `mean`
    - **Dates**: `min`, `max` + derived recency/span features
    - **Categoricals**: `mode`, `nunique`
- Flattens multi‑index columns and prefixes with `bureau_`.
- Outputs: `train_bureau_aggregated.csv`.

**08_reduce_applprev_aggregation.py**

- Loads and concatenates previous application files.
- Groups by `case_id` and aggregates:
    - **Numeric**: `max`, `mean`, `min`, `sum`, `count`
    - **Categorical**: `mode`, `nunique`
    - **Dates**: `min`, `max` + recency features
- Produces summary features such as total number of previous apps and refused apps.
- Outputs: `train_applprev_aggregated.csv`.

***

### 3.3 Master Merge and Dataset Optimization

**09_master_merge.py**

- Loads:
    - `train_base.csv`
    - Static files (e.g., `train_static_0_0.csv`, `train_static_0_1.csv`, `train_static_cb_0.csv`)
    - `train_person_primary.csv`
    - `train_person_coapplicant.csv`
    - `train_applprev_aggregated.csv`
    - `train_bureau_aggregated.csv`
- Performs sequential **left joins** on `case_id`:
    - Base + Static → + Primary Person → + Co‑Applicant → + ApplPrev → + Bureau
- Checks:
    - Row count consistency with base
    - Co‑applicant coverage
    - Feature counts by source (base/static/person/coapp/applprev/bureau)
- Outputs: `final_train_merged.csv`.

**10_optimize_dataset.py**

- Loads `final_train_merged.csv`.
- Steps:
    - Drops columns that are 100% missing.
    - Downcasts `float64` → `float32`, and `int64` → smaller integer types.
    - Converts `date_decision` to `datetime`.
    - Saves compressed `final_train_merged.parquet`.
- Prints:
    - Memory and file size reduction statistics.

***

### 3.4 Time‑Aware Split and Preprocessing

**11_sort_and_split_data.py**

- Loads `final_train_merged.parquet`.
- Sorts strictly by `date_decision`.
- Splits into:
    - `train.parquet` (first ~80% by time)
    - `test.parquet` (last ~20% by time)
- Ensures no future leakage from test into train.

**12_preprocess_pipeline.py**

- Loads `train.parquet` and `test.parquet`.
- Separates features and target:
    - Drops metadata (`case_id`, `date_decision`, `MONTH`, `WEEK_NUM`).
    - Keeps `target` as label.
- Builds a `ColumnTransformer` with:
    - **Numeric pipeline**: median imputation → `StandardScaler`.
    - **Categorical pipeline**: constant 'MISSING' imputation → `OneHotEncoder` with `min_frequency=0.01`.
- Fits on training data **only**, then transforms both train and test.
- Saves:
    - `X_train_processed.parquet`, `X_test_processed.parquet`
    - `y_train.parquet`, `y_test.parquet`
    - `preprocessing_pipeline.joblib` for reuse in inference.

***

### 3.5 Baseline (No‑ML) Benchmarks

**13_baseline_no_ml.py**

- Loads `final_train_merged.csv`.
- Computes target distribution and class imbalance.
- Creates three baselines:

1. **Majority Class** – always predicts the dominant class.
2. **Random Guessing** – stratified random predictions.
3. **Simple Rule‑Based Model** – uses intuitive rules on a few key features (e.g., bureau overdues, income vs loan amount, previous refusals).
- Evaluates each baseline using:
    - Accuracy, Precision, Recall, F1
    - ROC‑AUC and Gini (where applicable)
    - Confusion Matrix and qualitative interpretation

***

### 3.6 Model Training, Selection, and Evaluation

**14_train_model_selection.py**

- Loads `X_train_processed.parquet` and `y_train.parquet`.
- Sets up `TimeSeriesSplit` (5 folds) to respect temporal ordering.
- Configures a `LGBMClassifier` with:
    - `objective='binary'`, `metric='auc'`
    - `class_weight='balanced'` for imbalanced target
- Runs `RandomizedSearchCV` over a regularization‑focused hyperparameter space:
    - `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`
    - `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`
- Saves:
    - Best model → `best_model_lgbm.joblib`
    - Feature importances → `feature_importance.csv`

**15_evaluate_final_model.py**

- Loads `X_test_processed.parquet`, `y_test.parquet`, and `best_model_lgbm.joblib`.
- Computes:
    - ROC‑AUC and Gini coefficient
    - Classification report and confusion matrix
    - Probability distribution plots by class
- Generates plots in `evaluation_plots/`:
    - ROC curve
    - Confusion matrix heatmap
    - Probability distribution of predicted PDs
- Optionally computes **SHAP** feature importances and exports `shap_feature_importances.csv`.

***

### 3.7 Prediction Driver Visualization

**16_visualize_prediction_drivers.py**

- Loads the trained LightGBM model (from `/models/best_model_lgbm.joblib`).
- Loads a large sample of the final merged dataset (Parquet).
- Identifies the target column and feature names expected by the model.
- Produces:
    - Feature importance charts
    - SHAP‑based views (if SHAP is installed and compatible)
- Helps answer: *“Why did the model flag this applicant as high risk?”*.

***

## 4. How to Run the Pipeline

From the project root, in a Python 3.9+ environment with required libraries installed:

```bash
# 1. EDA (optional but recommended)
python 01_analyze_individual_tables.py
python 02_eda_tier1_base_static.py
python 03_eda_tier2_bureau_applprev.py
python 04_eda_tier3_person.py

# 2. Reductions & Aggregations
python 05_reduce_person_to_unique_cases.py
python 06_reduce_person_coapplicant.py
python 07_reduce_bureau_aggregation.py
python 08_reduce_applprev_aggregation.py

# 3. Master Merge & Optimization
python 09_master_merge.py
python 10_optimize_dataset.py

# 4. Time‑Aware Split & Preprocessing
python 11_sort_and_split_data.py
python 12_preprocess_pipeline.py

# 5. Baselines & Modeling
python 13_baseline_no_ml.py
python 14_train_model_selection.py

# 6. Evaluation & Explainability
python 15_evaluate_final_model.py
python 16_visualize_prediction_drivers.py
```

Adjust paths inside scripts as needed to point to your local data directory.

***

## 5. Requirements

Core libraries used:

- `pandas`, `numpy` for data manipulation
- `scikit-learn` for preprocessing, baselines, CV, and metrics
- `lightgbm` for gradient boosting model
- `matplotlib`, `seaborn` for visualizations
- `joblib` for model and pipeline persistence
- `pyarrow` for Parquet I/O
- `shap` (optional) for explainability

You can capture them in a `requirements.txt` similar to:

```text
pandas
numpy
scikit-learn
lightgbm
matplotlib
seaborn
joblib
pyarrow
shap
```


***

## 6. Future Work

- Add API or Streamlit app for real‑time scoring.
- Extend feature engineering with macro‑economic and cash‑flow data.
- Add fairness and disparate impact analysis modules.