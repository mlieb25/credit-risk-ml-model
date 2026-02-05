"""
Reduce Application Previous Dataset to Unique Case IDs - Aggregation Strategy

Concatenates all depth-1 applprev files, then aggregates to case-level summary statistics:
- Numeric features: max, mean, min, sum, count
- Categorical features: mode, nunique
- Date features: max, min, recency

Files combined:
- train_applprev_1_0.csv
- train_applprev_1_1.csv

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
print("REDUCE APPLPREV DATASET TO UNIQUE CASE_IDS - AGGREGATION STRATEGY")
print("="*80)

# List of depth-1 applprev files to concatenate
applprev_files = [
    'train_applprev_1_0.csv',
    'train_applprev_1_1.csv'
]

print("\n[Step 1] Loading and concatenating depth-1 applprev files...\n")

dfs_to_concat = []
for file in applprev_files:
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
    print("\nERROR: No applprev files found!")
    exit()

# Concatenate all applprev files
print(f"\nConcatenating {len(dfs_to_concat)} applprev file(s)...")
train_applprev = pd.concat(dfs_to_concat, ignore_index=True)

print(f"\nCombined dataset:")
print(f"  Shape: {train_applprev.shape}")
print(f"  Rows: {train_applprev.shape[0]:,}")
print(f"  Columns: {train_applprev.shape[1]}")
print(f"  Unique case_ids: {train_applprev['case_id'].nunique():,}")
print(f"  Average rows per case_id: {train_applprev.shape[0] / train_applprev['case_id'].nunique():.2f}")
print(f"  Memory: {train_applprev.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Cardinality distribution
print(f"\nApplications per case_id distribution:")
apps_per_case = train_applprev.groupby('case_id').size()
print(f"  Min: {apps_per_case.min()}")
print(f"  Mean: {apps_per_case.mean():.2f}")
print(f"  Median: {apps_per_case.median():.0f}")
print(f"  Max: {apps_per_case.max()}")
print(f"  75th percentile: {apps_per_case.quantile(0.75):.0f}")
print(f"  90th percentile: {apps_per_case.quantile(0.90):.0f}")

# Identify column types for appropriate aggregation
print("\n[Step 2] Analyzing column types for aggregation strategy...")

# Exclude case_id and num_group1 from aggregation
exclude_cols = ['case_id', 'num_group1']
feature_cols = [col for col in train_applprev.columns if col not in exclude_cols]

# Identify numeric, categorical, and date columns
numeric_cols = []
categorical_cols = []
date_cols = []

for col in feature_cols:
    dtype = train_applprev[col].dtype
    
    # Date columns (suffix _D)
    if col.endswith('_D'):
        date_cols.append(col)
    # Numeric columns (suffixes _A, _P, _L that are numeric)
    elif dtype in ['int64', 'float64']:
        numeric_cols.append(col)
    # Categorical columns (suffixes _M, _T, _L that are object/string)
    else:
        categorical_cols.append(col)

print(f"  Numeric columns: {len(numeric_cols)}")
print(f"  Categorical columns: {len(categorical_cols)}")
print(f"  Date columns: {len(date_cols)}")

# Build aggregation dictionary
print("\n[Step 3] Building aggregation dictionary...")
agg_dict = {}

# Numeric aggregations: max, mean, min, sum, count
for col in numeric_cols:
    agg_dict[col] = ['max', 'mean', 'min', 'sum', 'count']

# Categorical aggregations: mode (most frequent), nunique
for col in categorical_cols:
    agg_dict[col] = [lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan, 'nunique']

# Date aggregations: max (most recent), min (earliest)
for col in date_cols:
    agg_dict[col] = ['max', 'min']

print(f"  Total features to aggregate: {len(agg_dict)}")
print(f"  Estimated output features: ~{len(numeric_cols)*5 + len(categorical_cols)*2 + len(date_cols)*2}")

# Perform aggregation
print("\n[Step 4] Performing aggregation by case_id...")
print("  This may take a minute...")

train_applprev_agg = train_applprev.groupby('case_id').agg(agg_dict).reset_index()

# Flatten multi-level column names
print("\n[Step 5] Flattening column names...")
new_cols = ['case_id']

for col in train_applprev_agg.columns[1:]:
    if isinstance(col, tuple):
        # Handle lambda function names
        agg_func = col[1]
        if '<lambda>' in str(agg_func):
            agg_func = 'mode'
        new_col_name = f"applprev_{col[0]}_{agg_func}"
        new_cols.append(new_col_name)
    else:
        new_cols.append(col)

train_applprev_agg.columns = new_cols

print(f"  Flattened to {len(new_cols)} columns")

# Add derived features
print("\n[Step 6] Creating derived features...")

# Count of previous applications
train_applprev_agg['applprev_total_applications'] = train_applprev.groupby('case_id').size().values

# Recency features (days since most recent application)
if 'applprev_creationdate_885D_max' in train_applprev_agg.columns:
    try:
        train_applprev_agg['applprev_creationdate_885D_max_dt'] = pd.to_datetime(
            train_applprev_agg['applprev_creationdate_885D_max'], errors='coerce'
        )
        reference_date = pd.to_datetime('2019-01-01')  # Adjust based on your data
        train_applprev_agg['applprev_days_since_last_application'] = (
            reference_date - train_applprev_agg['applprev_creationdate_885D_max_dt']
        ).dt.days
        # Drop the temporary datetime column
        train_applprev_agg.drop('applprev_creationdate_885D_max_dt', axis=1, inplace=True)
        print("  ✓ Created recency feature: days_since_last_application")
    except:
        print("  ⚠ Could not create recency feature (date parsing issue)")

print(f"  Total derived features: 1-2")

# Summary statistics
print("\n[Step 7] Aggregated dataset summary...")
print(f"\nAggregated dataset:")
print(f"  Shape: {train_applprev_agg.shape}")
print(f"  Rows: {train_applprev_agg.shape[0]:,}")
print(f"  Columns: {train_applprev_agg.shape[1]}")
print(f"  Unique case_ids: {train_applprev_agg['case_id'].nunique():,}")
print(f"  Memory: {train_applprev_agg.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

# Check for duplicates
duplicates = train_applprev_agg['case_id'].duplicated().sum()
print(f"  Duplicate case_ids: {duplicates}")

if duplicates == 0:
    print("  ✓ SUCCESS: Each case_id appears exactly once!")
else:
    print(f"  ✗ WARNING: Found {duplicates} duplicate case_ids")

# Missing value summary
print(f"\nMissing values in aggregated dataset:")
missing_pct = (train_applprev_agg.isnull().sum() / len(train_applprev_agg) * 100).sort_values(ascending=False)
top_missing = missing_pct[missing_pct > 0].head(10)
if len(top_missing) > 0:
    print("  Top 10 columns with missing values:")
    for col, pct in top_missing.items():
        print(f"    {col}: {pct:.1f}%")
else:
    print("  No missing values found!")

# Sample of key aggregated features
print(f"\nSample aggregated features (first 3 rows):")
key_cols = ['case_id', 'applprev_total_applications']
if 'applprev_credamount_590A_max' in train_applprev_agg.columns:
    key_cols.append('applprev_credamount_590A_max')
if 'applprev_credamount_590A_mean' in train_applprev_agg.columns:
    key_cols.append('applprev_credamount_590A_mean')
if 'applprev_annuity_853A_mean' in train_applprev_agg.columns:
    key_cols.append('applprev_annuity_853A_mean')

available_key_cols = [col for col in key_cols if col in train_applprev_agg.columns]
print(train_applprev_agg[available_key_cols].head(3))

# Save the aggregated dataset
print("\n[Step 8] Saving aggregated dataset...")
output_path = BASE_PATH / "train_applprev_aggregated.csv"
train_applprev_agg.to_csv(output_path, index=False)
print(f"  Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")

print("\n" + "="*80)
print("AGGREGATION COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Input files: {len(dfs_to_concat)}")
print(f"  Input rows: {train_applprev.shape[0]:,}")
print(f"  Output rows: {train_applprev_agg.shape[0]:,}")
print(f"  Row reduction: {(1 - train_applprev_agg.shape[0]/train_applprev.shape[0])*100:.1f}%")
print(f"  Input columns: {train_applprev.shape[1]}")
print(f"  Output columns: {train_applprev_agg.shape[1]}")
print(f"  Feature expansion: +{train_applprev_agg.shape[1] - train_applprev.shape[1]} columns")
print(f"  Unique case_ids preserved: {train_applprev_agg['case_id'].nunique():,}")
