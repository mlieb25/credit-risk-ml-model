"""
Master Merge Script - Credit Risk ML Model

Joins the 6 reduced training datasets into a single model-ready file:
1. train_base.csv (Base)
2. train_static_*.csv (All Static files merged)
3. train_person_primary.csv (Primary Applicant)
4. train_person_coapplicant.csv (Co-Applicant)
5. train_applprev_aggregated.csv (Aggregated Previous Apps)
6. train_bureau_aggregated.csv (Aggregated Bureau)

"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model")
DATA_PATH = BASE_PATH / "home-credit-credit-risk-model-stability" / "csv_files" / "train"

print("="*80)
print("MASTER MERGE: CREATING MODEL-READY DATASET")
print("="*80)

def load_and_print(path, name):
    print(f"Loading {name}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape: {df.shape}")
    print(f"  Unique case_ids: {df['case_id'].nunique():,}")
    return df

# 1. Load all datasets
print("\n[Step 1] Loading all training datasets...\n")

# Raw Base
df_base = load_and_print(DATA_PATH / "train_base.csv", "Base Data")

# Load and concatenate all static files
print("\nLoading Static Data files...")
static_files = [
    'train_static_0_0.csv',
    'train_static_0_1.csv',
    'train_static_cb_0.csv'
]

static_dfs = []
for file in static_files:
    file_path = DATA_PATH / file
    if file_path.exists():
        print(f"  Loading {file}...")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"    Shape: {df.shape}")
        print(f"    Unique case_ids: {df['case_id'].nunique():,}")
        static_dfs.append(df)
    else:
        print(f"  {file} not found - skipping")

if len(static_dfs) == 0:
    print("\nERROR: No static files found!")
    exit()

# Merge static files horizontally (on case_id) if multiple exist
if len(static_dfs) == 1:
    df_static = static_dfs[0]
    print(f"\n  Using single static file: {df_static.shape}")
else:
    print(f"\n  Merging {len(static_dfs)} static files horizontally on case_id...")
    df_static = static_dfs[0]
    for i in range(1, len(static_dfs)):
        df_static = df_static.merge(static_dfs[i], on='case_id', how='outer', suffixes=('', f'_static{i}'))
    print(f"  Combined static shape: {df_static.shape}")
    print(f"  Unique case_ids: {df_static['case_id'].nunique():,}")

# Our reduced files (Assumes the previous scripts have been run)
print("\nLoading reduced/aggregated datasets...")
try:
    df_person = load_and_print(BASE_PATH / "train_person_primary.csv", "Primary Applicant Data")
    df_coapp = load_and_print(BASE_PATH / "train_person_coapplicant.csv", "Co-Applicant Data")
    df_applprev = load_and_print(BASE_PATH / "train_applprev_aggregated.csv", "Aggregated Applprev Data")
    df_bureau = load_and_print(BASE_PATH / "train_bureau_aggregated.csv", "Aggregated Bureau Data")
    
    print(f"\n  ✓ All datasets loaded successfully")
    print(f"  Co-applicant coverage: {df_coapp['case_id'].nunique() / df_base['case_id'].nunique() * 100:.1f}% of cases")
    
except FileNotFoundError as e:
    print(f"\nERROR: One or more reduced files not found. {e}")
    print("Please ensure you have run the following scripts first:")
    print("  1. reduce_person_to_unique_cases.py")
    print("  2. reduce_person_coapplicant.py")
    print("  3. reduce_applprev_aggregation.py")
    print("  4. reduce_bureau_aggregation.py")
    exit()

# 2. Sequential Left Joins
print("\n" + "="*80)
print("[Step 2] Performing Sequential Left Joins on case_id")
print("="*80)

print("\nMerging Base + Static...")
df = df_base.merge(df_static, on='case_id', how='left', suffixes=('', '_static'))
print(f"  Current Shape: {df.shape}")

print("\nMerging with Primary Applicant Data...")
df = df.merge(df_person, on='case_id', how='left', suffixes=('', '_person'))
print(f"  Current Shape: {df.shape}")

print("\nMerging with Co-Applicant Data...")
df = df.merge(df_coapp, on='case_id', how='left')
print(f"  Current Shape: {df.shape}")
print(f"  Cases with co-applicant: {df['coapp_personindex_1023L'].notna().sum():,}")
print(f"  Co-applicant prevalence: {df['coapp_personindex_1023L'].notna().sum() / len(df) * 100:.1f}%")

print("\nMerging with Aggregated Applprev Data...")
df = df.merge(df_applprev, on='case_id', how='left')
print(f"  Current Shape: {df.shape}")

print("\nMerging with Aggregated Bureau Data...")
df = df.merge(df_bureau, on='case_id', how='left')
print(f"  Current Shape: {df.shape}")

# 3. Final Cleanup & Summary
print("\n" + "="*80)
print("[Step 3] Final Cleanup & Summary")
print("="*80)

# Verify Row Count (Should match base)
if df.shape[0] == df_base.shape[0]:
    print(f"✓ Row count consistency verified: {df.shape[0]:,} rows")
else:
    print(f"⚠ WARNING: Row count mismatch! Base: {df_base.shape[0]:,}, Merged: {df.shape[0]:,}")

# Drop potential duplicate columns from merges (e.g., duplicated group keys)
cols_to_drop = [c for c in df.columns if '_person' in c and c.replace('_person', '') in df.columns]
if cols_to_drop:
    print(f"Dropping {len(cols_to_drop)} redundant columns...")
    df.drop(columns=cols_to_drop, inplace=True)

print(f"\nFinal Dataset Summary:")
print(f"  Total Rows: {df.shape[0]:,}")
print(f"  Total Columns: {df.shape[1]:,}")
print(f"  Unique case_ids: {df['case_id'].nunique():,}")

if 'target' in df.columns:
    print(f"  Target Rate: {df['target'].mean():.4f}")
    print(f"  Target Distribution:")
    print(f"    Class 0 (No Default): {(df['target'] == 0).sum():,} ({(df['target'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"    Class 1 (Default): {(df['target'] == 1).sum():,} ({(df['target'] == 1).sum()/len(df)*100:.1f}%)")
else:
    print("  Target: Column missing")

# Feature breakdown by source
print(f"\nFeature breakdown by source:")
base_static_cols = len([c for c in df.columns if not c.startswith(('coapp_', 'applprev_', 'bureau_'))])
coapp_cols = len([c for c in df.columns if c.startswith('coapp_')])
applprev_cols = len([c for c in df.columns if c.startswith('applprev_')])
bureau_cols = len([c for c in df.columns if c.startswith('bureau_')])

print(f"  Base + Static + Primary Person: {base_static_cols} columns")
print(f"  Co-Applicant features: {coapp_cols} columns")
print(f"  Applprev aggregated: {applprev_cols} columns")
print(f"  Bureau aggregated: {bureau_cols} columns")
print(f"  Total: {df.shape[1]} columns")

# Missing value overview
print(f"\nMissing value overview (top sources):")
missing_by_source = {
    'Co-Applicant': df[[c for c in df.columns if c.startswith('coapp_')]].isnull().mean().mean() * 100 if coapp_cols > 0 else 0,
    'Bureau': df[[c for c in df.columns if c.startswith('bureau_')]].isnull().mean().mean() * 100 if bureau_cols > 0 else 0,
    'Applprev': df[[c for c in df.columns if c.startswith('applprev_')]].isnull().mean().mean() * 100 if applprev_cols > 0 else 0
}

for source, pct in sorted(missing_by_source.items(), key=lambda x: x[1], reverse=True):
    print(f"  {source}: {pct:.1f}% missing (expected for cases without that data)")

# 4. Save Final Dataset
print("\n[Step 4] Saving final training dataset...")
output_path = BASE_PATH / "final_train_merged.csv"
df.to_csv(output_path, index=False)
print(f"✓ Master merged dataset saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")

print("\n" + "="*80)
print("MASTER MERGE COMPLETE - Ready for Feature Selection & Training!")
print("="*80)
print(f"\nNext steps:")
print(f"  1. Run feature importance analysis")
print(f"  2. Handle missing values (tree models can handle NaNs naturally)")
print(f"  3. Train baseline XGBoost/LightGBM model")
print(f"  4. Feature selection based on importance")
print(f"  5. Hyperparameter tuning")
