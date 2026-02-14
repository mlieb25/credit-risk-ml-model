"""
Individual Table Analysis
BAIT 509 Final Project - Home Credit Risk Model

This script analyzes each CSV file in the train folder separately,
producing summary statistics and data profiles for each table.

Author: Mitchell Liebrecht
Date: January 17, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Define paths
BASE_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/home-credit-credit-risk-model-stability")
TRAIN_PATH = BASE_PATH / "csv_files" / "train"
OUTPUT_PATH = BASE_PATH / "table_analysis"

# Create output directory
OUTPUT_PATH.mkdir(exist_ok=True)

print("="*80)
print("INDIVIDUAL TABLE ANALYSIS - HOME CREDIT TRAINING DATA")
print("="*80)

start_time = time.time()

# ============================================================================
# 1. GET LIST OF ALL CSV FILES
# ============================================================================
print("\n[1] Scanning for CSV files...")

csv_files = sorted(list(TRAIN_PATH.glob("*.csv")))
print(f"   Found {len(csv_files)} CSV files")

if len(csv_files) == 0:
    print("   ERROR: No CSV files found")
    exit(1)

# ============================================================================
# 2. ANALYZE EACH FILE
# ============================================================================
print("\n[2] Analyzing each table...\n")

all_table_stats = []

for i, csv_file in enumerate(csv_files, 1):
    table_name = csv_file.stem
    print(f"   [{i}/{len(csv_files)}] {table_name}")
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_file)
        
        # Basic shape info
        n_rows, n_cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        print(f"        Shape: {n_rows:,} rows × {n_cols} columns ({memory_mb:.2f} MB)")
        
        # Check for case_id
        has_case_id = 'case_id' in df.columns
        unique_case_ids = df['case_id'].nunique() if has_case_id else 0
        
        if has_case_id:
            cardinality = "one-to-one" if unique_case_ids == n_rows else "one-to-many"
            print(f"        case_id: {unique_case_ids:,} unique ({cardinality})")
        else:
            print(f"        case_id: NOT FOUND")
        
        # Column types
        dtype_counts = df.dtypes.value_counts().to_dict()
        n_numeric = len(df.select_dtypes(include=[np.number]).columns)
        n_categorical = len(df.select_dtypes(include=['object']).columns)
        n_datetime = len(df.select_dtypes(include=['datetime64']).columns)
        
        print(f"        Columns: {n_numeric} numeric, {n_categorical} categorical, {n_datetime} datetime")
        
        # Missing values
        total_missing = df.isna().sum().sum()
        total_cells = n_rows * n_cols
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
        cols_with_missing = (df.isna().sum() > 0).sum()
        
        print(f"        Missing: {missing_pct:.2f}% overall ({cols_with_missing}/{n_cols} columns)")
        
        # Collect summary stats for this table
        table_stats = {
            'table_name': table_name,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'memory_mb': round(memory_mb, 2),
            'has_case_id': has_case_id,
            'unique_case_ids': unique_case_ids,
            'cardinality': cardinality if has_case_id else 'N/A',
            'n_numeric': n_numeric,
            'n_categorical': n_categorical,
            'n_datetime': n_datetime,
            'missing_pct': round(missing_pct, 2),
            'cols_with_missing': cols_with_missing
        }
        all_table_stats.append(table_stats)
        
        # ====================================================================
        # DETAILED ANALYSIS FOR THIS TABLE
        # ====================================================================
        
        # Column-level statistics
        col_stats = []
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            n_missing = df[col].isna().sum()
            missing_pct_col = (n_missing / n_rows * 100) if n_rows > 0 else 0
            n_unique = df[col].nunique()
            
            stat = {
                'column': col,
                'dtype': col_type,
                'n_missing': n_missing,
                'missing_pct': round(missing_pct_col, 2),
                'n_unique': n_unique,
                'uniqueness_pct': round((n_unique / n_rows * 100), 2) if n_rows > 0 else 0
            }
            
            # Add sample values for categorical columns
            if df[col].dtype == 'object' and n_unique <= 20:
                top_values = df[col].value_counts().head(5).to_dict()
                stat['top_5_values'] = str(top_values)
            elif df[col].dtype in ['int64', 'float64']:
                stat['mean'] = round(df[col].mean(), 2) if not df[col].isna().all() else None
                stat['median'] = round(df[col].median(), 2) if not df[col].isna().all() else None
                stat['std'] = round(df[col].std(), 2) if not df[col].isna().all() else None
                stat['min'] = round(df[col].min(), 2) if not df[col].isna().all() else None
                stat['max'] = round(df[col].max(), 2) if not df[col].isna().all() else None
            
            col_stats.append(stat)
        
        # Save detailed column stats for this table
        col_stats_df = pd.DataFrame(col_stats)
        col_stats_file = OUTPUT_PATH / f"{table_name}_column_stats.csv"
        col_stats_df.to_csv(col_stats_file, index=False)
        
        # Generate a text report for this table
        report_lines = []
        report_lines.append("="*80)
        report_lines.append(f"TABLE ANALYSIS: {table_name}")
        report_lines.append("="*80)
        report_lines.append(f"\nFile: {csv_file.name}")
        report_lines.append(f"Rows: {n_rows:,}")
        report_lines.append(f"Columns: {n_cols}")
        report_lines.append(f"Memory: {memory_mb:.2f} MB")
        
        if has_case_id:
            report_lines.append(f"\ncase_id Statistics:")
            report_lines.append(f"  Unique case_ids: {unique_case_ids:,}")
            report_lines.append(f"  Cardinality: {cardinality}")
            if cardinality == "one-to-many":
                avg_rows_per_case = n_rows / unique_case_ids
                report_lines.append(f"  Average rows per case: {avg_rows_per_case:.2f}")
        
        report_lines.append(f"\nColumn Type Distribution:")
        report_lines.append(f"  Numeric: {n_numeric}")
        report_lines.append(f"  Categorical: {n_categorical}")
        report_lines.append(f"  Datetime: {n_datetime}")
        
        report_lines.append(f"\nMissing Values:")
        report_lines.append(f"  Overall: {missing_pct:.2f}%")
        report_lines.append(f"  Columns affected: {cols_with_missing}/{n_cols}")
        
        # Top columns by missing values
        missing_by_col = df.isna().sum().sort_values(ascending=False)
        missing_by_col = missing_by_col[missing_by_col > 0].head(10)
        if len(missing_by_col) > 0:
            report_lines.append(f"\n  Top 10 columns with missing values:")
            for col, count in missing_by_col.items():
                pct = (count / n_rows * 100)
                report_lines.append(f"    {col}: {count:,} ({pct:.2f}%)")
        
        # Sample data
        report_lines.append(f"\nFirst 3 Rows Sample:")
        report_lines.append(df.head(3).to_string())
        
        report_lines.append("\n" + "="*80)
        
        # Save text report
        report_file = OUTPUT_PATH / f"{table_name}_report.txt"
        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        print(f"        ✓ Saved: {table_name}_column_stats.csv & {table_name}_report.txt")
        
    except Exception as e:
        print(f"        ✗ ERROR: {str(e)}")
        all_table_stats.append({
            'table_name': table_name,
            'error': str(e)
        })
    
    print()

# ============================================================================
# 3. CREATE MASTER SUMMARY
# ============================================================================
print("\n[3] Creating master summary...")

# Save all table stats
all_stats_df = pd.DataFrame(all_table_stats)
all_stats_file = OUTPUT_PATH / "all_tables_summary.csv"
all_stats_df.to_csv(all_stats_file, index=False)
print(f"   ✓ Saved: all_tables_summary.csv")

# Create a master summary report
summary_lines = []
summary_lines.append("="*80)
summary_lines.append("HOME CREDIT TRAINING DATA - MASTER SUMMARY")
summary_lines.append("="*80)
summary_lines.append(f"\nDate: January 17, 2026")
summary_lines.append(f"Total tables analyzed: {len(csv_files)}")
summary_lines.append(f"\nOverall Statistics:")

if 'n_rows' in all_stats_df.columns:
    total_rows = all_stats_df['n_rows'].sum()
    total_cols = all_stats_df['n_cols'].sum()
    total_memory = all_stats_df['memory_mb'].sum()
    avg_missing = all_stats_df['missing_pct'].mean()
    
    summary_lines.append(f"  Total rows across all tables: {total_rows:,}")
    summary_lines.append(f"  Total columns across all tables: {total_cols:,}")
    summary_lines.append(f"  Total memory: {total_memory:.2f} MB")
    summary_lines.append(f"  Average missing %: {avg_missing:.2f}%")
    
    # Tables by size
    summary_lines.append(f"\nLargest Tables (by rows):")
    top_tables = all_stats_df.nlargest(5, 'n_rows')[['table_name', 'n_rows', 'n_cols']]
    for _, row in top_tables.iterrows():
        summary_lines.append(f"  {row['table_name']}: {row['n_rows']:,} rows × {row['n_cols']} cols")
    
    # Tables by columns
    summary_lines.append(f"\nMost Columns:")
    top_cols = all_stats_df.nlargest(5, 'n_cols')[['table_name', 'n_cols']]
    for _, row in top_cols.iterrows():
        summary_lines.append(f"  {row['table_name']}: {row['n_cols']} columns")
    
    # Base table vs others
    base_table = all_stats_df[all_stats_df['table_name'] == 'train_base']
    if len(base_table) > 0:
        base_cases = base_table.iloc[0]['unique_case_ids']
        summary_lines.append(f"\nBase Table:")
        summary_lines.append(f"  train_base.csv: {base_cases:,} unique case_ids")
        
        # Show cardinality of other tables
        summary_lines.append(f"\nTable Cardinality (relative to case_id):")
        for _, row in all_stats_df.iterrows():
            if row['has_case_id']:
                ratio = row['n_rows'] / row['unique_case_ids'] if row['unique_case_ids'] > 0 else 0
                summary_lines.append(f"  {row['table_name']}: {row['cardinality']} (avg {ratio:.2f} rows/case)")
    
    # Missing value issues
    summary_lines.append(f"\nTables with High Missing Values (>50%):")
    high_missing = all_stats_df[all_stats_df['missing_pct'] > 50].sort_values('missing_pct', ascending=False)
    if len(high_missing) > 0:
        for _, row in high_missing.iterrows():
            summary_lines.append(f"  {row['table_name']}: {row['missing_pct']:.2f}%")
    else:
        summary_lines.append(f"  None")

summary_lines.append("\n" + "="*80)
summary_lines.append("OUTPUT FILES GENERATED")
summary_lines.append("="*80)
summary_lines.append(f"\nFor each table:")
summary_lines.append(f"  - [table_name]_column_stats.csv (detailed column statistics)")
summary_lines.append(f"  - [table_name]_report.txt (human-readable summary)")
summary_lines.append(f"\nMaster files:")
summary_lines.append(f"  - all_tables_summary.csv (comparison of all tables)")
summary_lines.append(f"  - master_summary.txt (this file)")
summary_lines.append("\n" + "="*80)

# Save master summary
master_summary_file = OUTPUT_PATH / "master_summary.txt"
with open(master_summary_file, 'w') as f:
    f.write("\n".join(summary_lines))

print(f"   ✓ Saved: master_summary.txt")

# Print to console
print("\n" + "\n".join(summary_lines))

# ============================================================================
# 4. FINAL SUMMARY
# ============================================================================
elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nProcessing time: {elapsed_time:.1f} seconds")
print(f"\nAll outputs saved to: {OUTPUT_PATH}")
print(f"\nGenerated {len(csv_files) * 2 + 2} files:")
print(f"  - {len(csv_files)} × column_stats.csv files")
print(f"  - {len(csv_files)} × report.txt files")
print(f"  - all_tables_summary.csv")
print(f"  - master_summary.txt")
print("\n" + "="*80)
