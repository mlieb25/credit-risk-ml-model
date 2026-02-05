"""
Reduce Person Dataset to Unique Case IDs

Concatenates all depth-1 person files, then filters for primary applicant 
(personindex_1023L == 0), resulting in exactly one row per case_id.

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
print("REDUCE PERSON DATASET TO UNIQUE CASE_IDS")
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
else:
    print("  WARNING: personindex_1023L column not found!")

# Filter for primary applicant only (personindex == 0)
print("\n[Step 2] Filtering for personindex_1023L == 0 (primary applicants only)...")
train_person_primary = train_person[train_person['personindex_1023L'] == 0].copy()

print(f"\nFiltered dataset:")
print(f"  Shape: {train_person_primary.shape}")
print(f"  Rows: {train_person_primary.shape[0]:,}")
print(f"  Columns: {train_person_primary.shape[1]}")
print(f"  Unique case_ids: {train_person_primary['case_id'].nunique():,}")

# Verify uniqueness
print("\n[Step 3] Verifying uniqueness...")
duplicates = train_person_primary['case_id'].duplicated().sum()
print(f"  Duplicate case_ids after filtering: {duplicates}")

if duplicates == 0:
    print("  ✓ SUCCESS: Each case_id appears exactly once!")
else:
    print(f"  ✗ WARNING: Found {duplicates} duplicate case_ids")
    print("\nDuplicate case_ids:")
    dup_cases = train_person_primary[train_person_primary['case_id'].duplicated(keep=False)]['case_id'].unique()
    print(dup_cases[:10])

# Check if we lost any cases
original_cases = train_person['case_id'].nunique()
filtered_cases = train_person_primary['case_id'].nunique()
lost_cases = original_cases - filtered_cases

if lost_cases > 0:
    print(f"\n  NOTE: {lost_cases:,} cases do not have a personindex_1023L == 0 record")
    print(f"  This represents {lost_cases/original_cases*100:.2f}% of original cases")
else:
    print(f"\n  ✓ All {original_cases:,} cases retained in filtered dataset")

# Save the unique case_id dataset
print("\n[Step 4] Saving unique person dataset...")
output_path = BASE_PATH / "train_person_primary.csv"
train_person_primary.to_csv(output_path, index=False)
print(f"  Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

print("\n" + "="*80)
print("REDUCTION COMPLETE")
print("="*80)
print(f"\nSummary:")
print(f"  Input files: {len(dfs_to_concat)}")
print(f"  Input rows: {orig_rows:,}")
print(f"  Output rows: {train_person_primary.shape[0]:,}")
print(f"  Reduction: {(1 - train_person_primary.shape[0]/orig_rows)*100:.1f}%")
print(f"  Unique case_ids maintained: {train_person_primary['case_id'].nunique():,}")
