"""
Reduce Person Dataset to Co-Applicant Features

Concatenates all depth-1 person files, then filters for first co-applicant 
(personindex_1023L == 1), resulting in co-applicant features per case_id.

Files combined:
- train_person_1.csv
- train_person_2.csv

"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model")
DATA_PATH = BASE_PATH / "home-credit-credit-risk-model-stability" / "csv_files" / "train"

print("="*80)
print("EXTRACT CO-APPLICANT FEATURES FROM PERSON DATASET")
print("="*80)

# List of depth-1 person files to concatenate
person_files = [
    'train_person_1.csv',
    'train_person_2.csv'
]

print("\n[Step 1] Loading and concatenating depth-1 person files...\n")

dfs_to_concat = []
for file in person_files:
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
    print("\nERROR: No person files found!")
    exit()

# Concatenate all person files
print(f"\nConcatenating {len(dfs_to_concat)} person file(s)...")
train_person = pd.concat(dfs_to_concat, ignore_index=True)

orig_rows = train_person.shape[0]
orig_cases = train_person['case_id'].nunique()

print(f"\nCombined dataset:")
print(f"  Shape: {train_person.shape}")
print(f"  Rows: {orig_rows:,}")
print(f"  Columns: {train_person.shape[1]}")
print(f"  Unique case_ids: {orig_cases:,}")
print(f"  Average rows per case_id: {orig_rows/orig_cases:.2f}")

# Check personindex_1023L distribution
print(f"\nPersonindex distribution:")
if 'personindex_1023L' in train_person.columns:
    print(train_person['personindex_1023L'].value_counts().sort_index().head(10))
    print(f"  Missing values: {train_person['personindex_1023L'].isna().sum():,}")
    
    # Count cases with co-applicants
    cases_with_coapplicant = train_person[train_person['personindex_1023L'] == 1]['case_id'].nunique()
    print(f"\n  Cases with co-applicant (personindex == 1): {cases_with_coapplicant:,}")
    print(f"  Co-applicant prevalence: {cases_with_coapplicant/orig_cases*100:.1f}% of all cases")
else:
    print("  WARNING: personindex_1023L column not found!")

# Filter for first co-applicant only (personindex == 1)
print("\n[Step 2] Filtering for personindex_1023L == 1 (first co-applicants only)...")
train_person_coapp = train_person[train_person['personindex_1023L'] == 1].copy()

print(f"\nFiltered dataset:")
print(f"  Shape: {train_person_coapp.shape}")
print(f"  Rows: {train_person_coapp.shape[0]:,}")
print(f"  Columns: {train_person_coapp.shape[1]}")
print(f"  Unique case_ids: {train_person_coapp['case_id'].nunique():,}")

# Verify uniqueness
print("\n[Step 3] Verifying uniqueness...")
duplicates = train_person_coapp['case_id'].duplicated().sum()
print(f"  Duplicate case_ids after filtering: {duplicates}")

if duplicates == 0:
    print("  ✓ SUCCESS: Each case_id appears at most once!")
else:
    print(f"  ✗ WARNING: Found {duplicates} duplicate case_ids")
    print("  Multiple co-applicants detected - keeping first occurrence...")
    # Keep only the first co-applicant per case
    train_person_coapp = train_person_coapp.drop_duplicates(subset='case_id', keep='first')
    print(f"  After deduplication: {train_person_coapp.shape[0]:,} rows")

# Rename columns to indicate co-applicant features
print("\n[Step 4] Renaming columns with 'coapp_' prefix...")

# Keep case_id as is, rename all other columns
columns_to_rename = {col: f'coapp_{col}' for col in train_person_coapp.columns if col != 'case_id'}
train_person_coapp.rename(columns=columns_to_rename, inplace=True)

print(f"  Renamed {len(columns_to_rename)} columns")
print(f"  Sample renamed columns: {list(train_person_coapp.columns[1:6])}")

# Save the co-applicant dataset
print("\n[Step 5] Saving co-applicant dataset...")
output_path = BASE_PATH / "train_person_coapplicant.csv"
train_person_coapp.to_csv(output_path, index=False)
print(f"  Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

print("\n" + "="*80)
print("CO-APPLICANT EXTRACTION COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Input files: {len(dfs_to_concat)}")
print(f"  Input rows: {orig_rows:,}")
print(f"  Output rows (co-applicants): {train_person_coapp.shape[0]:,}")
print(f"  Cases with co-applicant  {train_person_coapp['case_id'].nunique():,}")
print(f"  Cases without co-applicant: {orig_cases - train_person_coapp['case_id'].nunique():,}")
print(f"\nUsage:")
print(f"  Merge this file with your primary applicant ")
print(f"  df.merge(train_person_coapplicant, on='case_id', how='left')")
print(f"  Missing co-applicant features will be NaN (single applicant cases)")
