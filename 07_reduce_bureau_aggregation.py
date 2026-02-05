"""
Reduce Credit Bureau Dataset to Unique Case IDs - Risk-Focused Aggregation

Concatenates all depth-1 bureau files, then aggregates to case-level with emphasis on:
- DPD features: max (worst behavior), mean, std, count_nonzero
- Debt features: max, sum, mean, count_positive
- Contract features: sum, max, nunique
- Date features: max, min, recency

Files combined:
- train_credit_bureau_a_1_0.csv
- train_credit_bureau_a_1_1.csv
- train_credit_bureau_a_1_2.csv
- train_credit_bureau_a_1_3.csv 

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model")
DATA_PATH = BASE_PATH / "home-credit-credit-risk-model-stability" / "csv_files" / "train"

print("="*80)
print("REDUCE CREDIT BUREAU DATASET - RISK-FOCUSED AGGREGATION")
print("="*80)

# List of depth-1 bureau files to concatenate
bureau_files = [
    'train_credit_bureau_a_1_0.csv',
    'train_credit_bureau_a_1_1.csv',
    'train_credit_bureau_a_1_2.csv',
    'train_credit_bureau_a_1_3.csv'
]

print("\n[Step 1] Loading and concatenating depth-1 bureau files...\n")

dfs_to_concat = []
for file in bureau_files:
    file_path = DATA_PATH / file
    if file_path.exists():
        print(f"Loading {file}...")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"  Rows: {df.shape[0]:,}")
        print(f"  Columns: {df.shape[1]}")
        print(f"  Unique case_ids: {df['case_id'].nunique():,}")
        dfs_to_concat.append(df)
    else:
        print(f"{file} not found - skipping")

if len(dfs_to_concat) == 0:
    print("\nERROR: No bureau files found!")
    exit()

# Concatenate all bureau files
print(f"\nConcatenating {len(dfs_to_concat)} bureau file(s)...")
train_bureau = pd.concat(dfs_to_concat, ignore_index=True)

print(f"\nCombined dataset:")
print(f"  Shape: {train_bureau.shape}")
print(f"  Rows: {train_bureau.shape[0]:,}")
print(f"  Columns: {train_bureau.shape[1]}")
print(f"  Unique case_ids: {train_bureau['case_id'].nunique():,}")
print(f"  Average rows per case_id: {train_bureau.shape[0] / train_bureau['case_id'].nunique():.2f}")
print(f"  Memory: {train_bureau.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Cardinality distribution
print(f"\nBureau records per case_id distribution:")
bureau_per_case = train_bureau.groupby('case_id').size()
print(f"  Min: {bureau_per_case.min()}")
print(f"  Mean: {bureau_per_case.mean():.2f}")
print(f"  Median: {bureau_per_case.median():.0f}")
print(f"  Max: {bureau_per_case.max()}")
print(f"  75th percentile: {bureau_per_case.quantile(0.75):.0f}")
print(f"  90th percentile: {bureau_per_case.quantile(0.90):.0f}")

# Identify column types and feature categories
print("\n[Step 2] Categorizing features for risk-focused aggregation...")

exclude_cols = ['case_id', 'num_group1']
feature_cols = [col for col in train_bureau.columns if col not in exclude_cols]

# Categorize features by suffix and content
dpd_features = []  # Days Past Due - payment behavior
debt_features = []  # Outstanding debt amounts
contract_count_features = []  # Counts of contracts/installments
amount_features = []  # Credit amounts, limits, installments
date_features = []  # Date columns
rate_features = []  # Interest rates
categorical_features = []  # Categorical codes

for col in feature_cols:
    col_lower = col.lower()
    
    # Date columns (suffix _D)
    if col.endswith('_D'):
        date_features.append(col)
    # DPD features (suffix _P or contains 'dpd')
    elif col.endswith('_P') or 'dpd' in col_lower or 'overdue' in col_lower:
        if train_bureau[col].dtype in ['int64', 'float64']:
            dpd_features.append(col)
        else:
            categorical_features.append(col)
    # Debt/Amount features (suffix _A)
    elif col.endswith('_A'):
        if 'debt' in col_lower or 'overdue' in col_lower or 'outstanding' in col_lower:
            debt_features.append(col)
        else:
            amount_features.append(col)
    # Count features (suffix _L, contains 'number' or 'count')
    elif col.endswith('_L'):
        if 'numberof' in col_lower or 'count' in col_lower or train_bureau[col].dtype in ['int64', 'float64']:
            contract_count_features.append(col)
        elif 'rate' in col_lower:
            rate_features.append(col)
        else:
            categorical_features.append(col)
    # Categorical features (suffix _M, _T)
    elif col.endswith('_M') or col.endswith('_T'):
        categorical_features.append(col)
    # Remaining numeric
    elif train_bureau[col].dtype in ['int64', 'float64']:
        amount_features.append(col)
    else:
        categorical_features.append(col)

print(f"  DPD/Payment features: {len(dpd_features)}")
print(f"  Debt features: {len(debt_features)}")
print(f"  Contract count features: {len(contract_count_features)}")
print(f"  Amount features: {len(amount_features)}")
print(f"  Rate features: {len(rate_features)}")
print(f"  Date features: {len(date_features)}")
print(f"  Categorical features: {len(categorical_features)}")

# Build risk-focused aggregation dictionary
print("\n[Step 3] Building risk-focused aggregation dictionary...")
agg_dict = {}

# DPD features: max (worst behavior), mean, std, count of non-zero
for col in dpd_features:
    agg_dict[col] = [
        'max',  # Worst payment delay
        'mean',  # Average delay
        'std',  # Payment consistency
        lambda x: (x > 0).sum()  # Count of times late
    ]

# Debt features: max (peak exposure), sum (total), mean, count of positive values
for col in debt_features:
    agg_dict[col] = [
        'max',  # Peak debt
        'sum',  # Total debt
        'mean',  # Average debt
        lambda x: (x > 0).sum()  # Count of contracts with debt
    ]

# Contract count features: sum (total experience), max
for col in contract_count_features:
    agg_dict[col] = ['sum', 'max', 'mean']

# Amount features: max, mean, sum
for col in amount_features:
    agg_dict[col] = ['max', 'mean', 'sum']

# Rate features: mean, max, min
for col in rate_features:
    agg_dict[col] = ['mean', 'max', 'min']

# Date features: max (most recent), min (earliest)
for col in date_features:
    agg_dict[col] = ['max', 'min']

# Categorical features: mode, nunique
for col in categorical_features:
    agg_dict[col] = [
        lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan,
        'nunique'
    ]

total_aggs = sum(len(v) for v in agg_dict.values())
print(f"  Total features to aggregate: {len(agg_dict)}")
print(f"  Estimated output features: ~{total_aggs}")

# Perform aggregation
print("\n[Step 4] Performing aggregation by case_id...")
print("  This may take 2-3 minutes for large concatenated dataset...")

train_bureau_agg = train_bureau.groupby('case_id').agg(agg_dict).reset_index()

# Flatten multi-level column names
print("\n[Step 5] Flattening column names...")
new_cols = ['case_id']

for col in train_bureau_agg.columns[1:]:
    if isinstance(col, tuple):
        # Handle lambda function names
        agg_func = col[1]
        if '<lambda>' in str(agg_func):
            # Determine which lambda based on column type
            if col[0] in dpd_features or col[0] in debt_features:
                # Check if it's the count lambda (4th or 4th aggregation)
                idx = list(agg_dict[col[0]]).index(agg_func)
                if idx == 3:  # Last aggregation is count
                    agg_func = 'count_nonzero' if col[0] in dpd_features else 'count_positive'
                else:
                    agg_func = 'mode'
            else:
                agg_func = 'mode'
        new_col_name = f"bureau_{col[0]}_{agg_func}"
        new_cols.append(new_col_name)
    else:
        new_cols.append(col)

train_bureau_agg.columns = new_cols

print(f"  Flattened to {len(new_cols)} columns")

# Add derived features
print("\n[Step 6] Creating derived features...")

# Total number of bureau records per case
train_bureau_agg['bureau_total_records'] = train_bureau.groupby('case_id').size().values

# Credit history span (if date columns available)
if 'bureau_dateofcredstart_181D_min' in train_bureau_agg.columns and 'bureau_dateofcredstart_181D_max' in train_bureau_agg.columns:
    try:
        train_bureau_agg['bureau_dateofcredstart_181D_min_dt'] = pd.to_datetime(
            train_bureau_agg['bureau_dateofcredstart_181D_min'], errors='coerce'
        )
        train_bureau_agg['bureau_dateofcredstart_181D_max_dt'] = pd.to_datetime(
            train_bureau_agg['bureau_dateofcredstart_181D_max'], errors='coerce'
        )
        
        # Credit history span in days
        train_bureau_agg['bureau_credit_history_span_days'] = (
            train_bureau_agg['bureau_dateofcredstart_181D_max_dt'] - 
            train_bureau_agg['bureau_dateofcredstart_181D_min_dt']
        ).dt.days
        
        # Drop temporary datetime columns
        train_bureau_agg.drop(['bureau_dateofcredstart_181D_min_dt', 'bureau_dateofcredstart_181D_max_dt'], 
                             axis=1, inplace=True)
        print("  ✓ Created credit history span feature")
    except:
        print("  ⚠ Could not create credit history span (date parsing issue)")

# Recency features
if 'bureau_lastupdate_1112D_max' in train_bureau_agg.columns:
    try:
        train_bureau_agg['bureau_lastupdate_1112D_max_dt'] = pd.to_datetime(
            train_bureau_agg['bureau_lastupdate_1112D_max'], errors='coerce'
        )
        reference_date = pd.to_datetime('2019-01-01')  # Adjust based on your data
        train_bureau_agg['bureau_days_since_last_update'] = (
            reference_date - train_bureau_agg['bureau_lastupdate_1112D_max_dt']
        ).dt.days
        train_bureau_agg.drop('bureau_lastupdate_1112D_max_dt', axis=1, inplace=True)
        print("  ✓ Created recency feature: days_since_last_update")
    except:
        print("  ⚠ Could not create recency feature (date parsing issue)")

print(f"  Total derived features: 1-3")

# Summary statistics
print("\n[Step 7] Aggregated dataset summary...")
print(f"\nAggregated dataset:")
print(f"  Shape: {train_bureau_agg.shape}")
print(f"  Rows: {train_bureau_agg.shape[0]:,}")
print(f"  Columns: {train_bureau_agg.shape[1]}")
print(f"  Unique case_ids: {train_bureau_agg['case_id'].nunique():,}")
print(f"  Memory: {train_bureau_agg.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Check for duplicates
duplicates = train_bureau_agg['case_id'].duplicated().sum()
print(f"  Duplicate case_ids: {duplicates}")

if duplicates == 0:
    print("  ✓ SUCCESS: Each case_id appears exactly once!")
else:
    print(f"  ✗ WARNING: Found {duplicates} duplicate case_ids")

# Missing value summary
print(f"\nMissing values in aggregated dataset:")
missing_pct = (train_bureau_agg.isnull().sum() / len(train_bureau_agg) * 100).sort_values(ascending=False)
top_missing = missing_pct[missing_pct > 0].head(10)
if len(top_missing) > 0:
    print("  Top 10 columns with missing values:")
    for col, pct in top_missing.items():
        print(f"    {col}: {pct:.1f}%")
else:
    print("  No missing values found!")

# Sample of key aggregated features
print(f"\nSample aggregated features (first 3 rows):")
key_cols = ['case_id', 'bureau_total_records']

# Add some key DPD features if they exist
for col in train_bureau_agg.columns:
    if 'dpdmax' in col and 'max' in col and len(key_cols) < 6:
        key_cols.append(col)
    elif 'debtoutstand' in col and 'sum' in col and len(key_cols) < 6:
        key_cols.append(col)

available_key_cols = [col for col in key_cols if col in train_bureau_agg.columns]
if len(available_key_cols) > 1:
    print(train_bureau_agg[available_key_cols].head(3))

# Save the aggregated dataset
print("\n[Step 8] Saving aggregated dataset...")
output_path = BASE_PATH / "train_bureau_aggregated.csv"
train_bureau_agg.to_csv(output_path, index=False)
print(f"  Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")

print("\n" + "="*80)
print("AGGREGATION COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Input files: {len(dfs_to_concat)}")
print(f"  Input rows: {train_bureau.shape[0]:,}")
print(f"  Output rows: {train_bureau_agg.shape[0]:,}")
print(f"  Row reduction: {(1 - train_bureau_agg.shape[0]/train_bureau.shape[0])*100:.1f}%")
print(f"  Input columns: {train_bureau.shape[1]}")
print(f"  Output columns: {train_bureau_agg.shape[1]}")
print(f"  Feature expansion: +{train_bureau_agg.shape[1] - train_bureau.shape[1]} columns")
print(f"  Unique case_ids preserved: {train_bureau_agg['case_id'].nunique():,}")
print(f"\nKey risk metrics captured:")
print(f"  ✓ DPD max (worst payment behavior)")
print(f"  ✓ Debt sum (total exposure)")
print(f"  ✓ Contract counts (credit experience)")
print(f"  ✓ Credit history span")
