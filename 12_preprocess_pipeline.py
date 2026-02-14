#!/usr/bin/env python3
"""
Preprocessing Pipeline Script
Author: Mitchell Liebrecht
Date: January 2026

This script performs the standard ML preprocessing steps:
1. Splits features (X) and target (y).
2. Builds a Scikit-Learn ColumnTransformer pipeline:
   - Numerics: Median Imputation -> Standard Scaling
   - Categoricals: Constant Imputation -> One-Hot Encoding (with rare category handling)
3. Fits the pipeline on Training data ONLY.
4. Transforms both Training and Test data.
5. Saves the processed arrays and the fitted pipeline object (for future inference).
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import gc

def preprocess_data():
    print("="*80)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*80)
    
    base_dir = Path(__file__).parent
    train_path = base_dir / "train.parquet"
    test_path = base_dir / "test.parquet"
    
    # 1. Load Data
    print("\n[1/6] Loading Split Data...")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("train.parquet or test.parquet not found. Run sort_and_split_data.py first.")
        
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    print(f"      Train shape: {train_df.shape}")
    print(f"      Test shape:  {test_df.shape}")

    # 2. Separate Features and Target
    print("\n[2/6] Separating Features (X) and Target (y)...")
    
    # Columns to exclude from features
    # 'case_id' and 'date_decision' are meta-data, not features for the model
    drop_cols = ['case_id', 'date_decision', 'MONTH', 'WEEK_NUM', 'target']
    
    # Ensure all drop_cols exist before dropping
    existing_drop_cols = [c for c in drop_cols if c in train_df.columns]
    
    X_train = train_df.drop(columns=existing_drop_cols)
    y_train = train_df['target']
    
    X_test = test_df.drop(columns=existing_drop_cols)
    y_test = test_df['target']
    
    # Clean up memory
    del train_df, test_df
    gc.collect()

    # 3. Identify Column Types
    print("\n[3/6] Identifying Column Types...")
    
    # Numerical columns
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
    # Categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"      Numerical features:   {len(num_cols)}")
    print(f"      Categorical features: {len(cat_cols)}")

    # 4. Define Pipelines
    print("\n[4/6] Building Scikit-Learn Pipeline...")
    
    # Numeric Pipeline: Impute Median -> Scale
    # using float32 to save memory
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline: Impute 'MISSING' -> OneHotEncode
    # min_frequency=0.01 means categories appearing in <1% of data are grouped as "infrequent"
    # This prevents creating thousands of useless columns
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore', 
            sparse_output=False, 
            min_frequency=0.01,
            dtype=np.float32 
        ))
    ])
    
    # Combine into Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        verbose_feature_names_out=False
    )

    # 5. Fit and Transform
    print("\n[5/6] Fitting Pipeline on Train and Transforming...")
    
    # Fit on Train ONLY
    print("      Fitting on X_train...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform Test
    print("      Transforming X_test...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    print(f"      Final feature count: {len(feature_names)} (expanded from {len(num_cols)+len(cat_cols)})")

    # Convert back to Pandas DataFrames (Optional, but easier to work with)
    # Using float32 to keep size down
    print("      Converting to DataFrames...")
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, dtype=np.float32)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, dtype=np.float32)
    
    # 6. Save Artifacts
    print("\n[6/6] Saving Processed Data and Pipeline...")
    
    # Save Data
    X_train_df.to_parquet(base_dir / "X_train_processed.parquet", index=False)
    X_test_df.to_parquet(base_dir / "X_test_processed.parquet", index=False)
    
    # Save Targets (aligning indices is handled by file separation)
    pd.DataFrame(y_train).to_parquet(base_dir / "y_train.parquet", index=False)
    pd.DataFrame(y_test).to_parquet(base_dir / "y_test.parquet", index=False)
    
    # Save the fitted pipeline object
    pipeline_path = base_dir / "preprocessing_pipeline.joblib"
    joblib.dump(preprocessor, pipeline_path)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Pipeline saved to: {pipeline_path.name}")
    print(f"X_train shape:     {X_train_df.shape}")
    print(f"X_test shape:      {X_test_df.shape}")
    print("="*80)

if __name__ == "__main__":
    preprocess_data()
