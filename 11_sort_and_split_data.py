#!/usr/bin/env python3
"""
Time-Series Sort and Split Script
Author: Mitchell Stevens
Date: January 2026

This script prepares the credit risk data for ML training by:
1. Loading the optimized Parquet file
2. Sorting strictly by 'date_decision' (Time-Series requirement)
3. Splitting into Train (80%) and Test (20%) sets based on time
4. Saving separate Parquet files for easy loading later
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def sort_and_split_data(
    input_file="final_train_merged.parquet",
    train_size=0.80,
    target_col="target",
    date_col="date_decision"
):
    print("="*80)
    print("TIME-SERIES DATA PREPARATION")
    print("="*80)
    
    # Setup paths
    base_dir = Path(__file__).parent
    file_path = base_dir / input_file
    
    if not file_path.exists():
        print(f"Error: Input file not found: {file_path}")
        print("Please run 'optimize_dataset.py' or 'convert_csv_to_parquet.py' first.")
        return

    # 1. Load Data
    print(f"\n[1/4] Loading {input_file}...")
    df = pd.read_parquet(file_path)
    print(f"      Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    # 2. Sort by Date (CRITICAL STEP)
    print(f"\n[2/4] Sorting by {date_col}...")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset!")
        
    # Ensure date is datetime type
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    print(f"      Date Range: {min_date.date()} to {max_date.date()}")

    # 3. Time-Series Split
    print(f"\n[3/4] Splitting data (Train: {train_size:.0%}, Test: {1-train_size:.0%})...")
    
    split_idx = int(len(df) * train_size)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Validation checks
    print(f"      Train Set: {len(train_df):,} rows ({train_df[date_col].min().date()} -> {train_df[date_col].max().date()})")
    print(f"      Test Set:  {len(test_df):,} rows ({test_df[date_col].min().date()} -> {test_df[date_col].max().date()})")
    
    # Check for leakage
    if train_df[date_col].max() > test_df[date_col].min():
        print("\nWARNING: DATE OVERLAP DETECTED! Check sorting logic.")
    else:
        print("      ✓ Validation passed: No date overlap between Train and Test.")

    # Check Target Distribution
    train_target = train_df[target_col].mean()
    test_target = test_df[target_col].mean()
    print(f"\n      Default Rate (Target Mean):")
    print(f"      Train: {train_target:.4%} ({train_df[target_col].sum()} defaults)")
    print(f"      Test:  {test_target:.4%} ({test_df[target_col].sum()} defaults)")

    # 4. Save Outputs
    print(f"\n[4/4] Saving split files...")
    train_path = base_dir / "train.parquet"
    test_path = base_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"      Saved: {train_path.name} ({len(train_df):,} rows)")
    print(f"      Saved: {test_path.name} ({len(test_df):,} rows)")
    print("\n" + "="*80)
    print("READY FOR PREPROCESSING")
    print("="*80)
    print("Next steps:")
    print("1. Load 'train.parquet' to fit your Imputers and Scalers.")
    print("2. Transform 'train.parquet' and 'test.parquet' using those fitted objects.")
    print("3. Train your model on the transformed training data.")

if __name__ == "__main__":
    sort_and_split_data()
