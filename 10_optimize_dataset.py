#!/usr/bin/env python3
"""
Dataset Optimization and Size Reduction Script
Author: Mitchell Stevens
Date: January 2026

This script performs "safe" optimizations on the credit risk dataset:
1. Drops columns that are 100% empty (contain no information).
2. Downcasts numeric columns (float64 -> float32) to halve memory usage.
3. Converts the date column to proper datetime format.
4. Saves the result as a highly compressed Parquet file.
"""

import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

def format_bytes(size):
    """Return human readable string for file size"""
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def optimize_dataset(input_file):
    print("=" * 80)
    print(f"STARTING OPTIMIZATION FOR: {input_file.name}")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. READ CSV (Measure Time)
    # ---------------------------------------------------------
    print(f"\n[1/5] Reading CSV file...")
    start_read = time.time()
    
    # Read CSV
    df = pd.read_csv(input_file, low_memory=False)
    
    read_time = time.time() - start_read
    print(f"      Time to read CSV: {read_time:.2f} seconds")
    
    # Initial Stats
    initial_memory = df.memory_usage(deep=True).sum()
    initial_cols = df.shape[1]
    print(f"      Initial Shape:    {df.shape}")
    print(f"      Initial Memory:   {format_bytes(initial_memory)}")

    # ---------------------------------------------------------
    # 2. DROP EMPTY COLUMNS
    # ---------------------------------------------------------
    print(f"\n[2/5] Identifying and dropping empty columns...")
    
    # Calculate null percentage
    null_counts = df.isnull().sum()
    total_rows = len(df)
    
    # Find columns that are 100% null
    empty_cols = null_counts[null_counts == total_rows].index.tolist()
    
    if empty_cols:
        print(f"      Found {len(empty_cols)} columns with 100% missing values.")
        print(f"      Examples: {empty_cols[:5]}...")
        df.drop(columns=empty_cols, inplace=True)
        print(f"      Dropped {len(empty_cols)} columns.")
    else:
        print("      No empty columns found.")

    # ---------------------------------------------------------
    # 3. OPTIMIZE DATA TYPES
    # ---------------------------------------------------------
    print(f"\n[3/5] Optimizing data types (Downcasting)...")
    
    # Float optimization (64 -> 32)
    float_cols = df.select_dtypes(include=['float64']).columns
    print(f"      Downcasting {len(float_cols)} float64 columns to float32...")
    
    # Using float32 instead of float16 for safety/precision balance
    df[float_cols] = df[float_cols].astype('float32')
    
    # Int optimization
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(int_cols) > 0:
        print(f"      Downcasting {len(int_cols)} int64 columns...")
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Date conversion
    if 'date_decision' in df.columns:
        print("      Converting 'date_decision' to datetime...")
        df['date_decision'] = pd.to_datetime(df['date_decision'])

    # ---------------------------------------------------------
    # 4. FINAL STATS
    # ---------------------------------------------------------
    final_memory = df.memory_usage(deep=True).sum()
    final_cols = df.shape[1]
    mem_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    print("\n" + "-" * 40)
    print("OPTIMIZATION RESULTS")
    print("-" * 40)
    print(f"Columns: {initial_cols} -> {final_cols} (Dropped {initial_cols - final_cols})")
    print(f"Memory:  {format_bytes(initial_memory)} -> {format_bytes(final_memory)}")
    print(f"Reduction: {mem_reduction:.1f}%")
    print("-" * 40)

    # ---------------------------------------------------------
    # 5. SAVE TO PARQUET
    # ---------------------------------------------------------
    output_file = input_file.with_suffix('.parquet')
    print(f"\n[5/5] Saving to {output_file.name}...")
    start_save = time.time()
    
    df.to_parquet(output_file, index=False)
    
    save_time = time.time() - start_save
    print(f"      Time to save: {save_time:.2f} seconds")
    
    # File size comparison
    orig_size = os.path.getsize(input_file)
    new_size = os.path.getsize(output_file)
    size_reduction = (orig_size - new_size) / orig_size * 100
    
    print("\n" + "=" * 80)
    print(f"FINAL FILE SIZE COMPARISON")
    print("=" * 80)
    print(f"Original CSV:   {format_bytes(orig_size)}")
    print(f"Optimized Parquet: {format_bytes(new_size)}")
    print(f"Space Saved:    {size_reduction:.1f}%")
    print(f"Total Time:     {time.time() - start_read:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    # Define path
    project_dir = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model")
    csv_file = project_dir / "final_train_merged.csv"
    
    if csv_file.exists():
        optimize_dataset(csv_file)
    else:
        print(f"Error: File not found at {csv_file}")
