#!/usr/bin/env python3
"""
Model Selection & Temporal Cross-Validation Script
Author: Mitchell Stevens
Date: January 2026

This script implements a robust model selection pipeline:
1. Loads processed training data (Parquet).
2. Uses TimeSeriesSplit to respect the temporal nature of the data (no future leakage).
3. Performs Hyperparameter Tuning on LightGBM using RandomizedSearchCV.
4. Focuses on Regularization (L1/L2) to prevent overfitting.
5. Selects and saves the best model.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import gc

def run_model_selection():
    print("="*80)
    print("STARTING MODEL SELECTION PIPELINE")
    print("="*80)
    
    base_dir = Path(__file__).parent
    X_train_path = base_dir / "X_train_processed.parquet"
    y_train_path = base_dir / "y_train.parquet"
    
    # 1. Load Processed Training Data
    print("\n[1/5] Loading Processed Training Data...")
    if not X_train_path.exists():
        raise FileNotFoundError("Processed data not found. Run preprocess_pipeline.py first.")
        
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(y_train_path).iloc[:, 0] # Convert DataFrame to Series
    
    print(f"      X_train shape: {X_train.shape}")
    print(f"      y_train shape: {y_train.shape}")
    print(f"      Target Rate:   {y_train.mean():.4%}")

    # 2. Define Temporal Cross-Validation
    print("\n[2/5] Setting up TimeSeriesSplit...")
    # TimeSeriesSplit creates expanding windows.
    # Fold 1: Train indices [0:k], Test indices [k:2k]
    # Fold 2: Train indices [0:2k], Test indices [2k:3k]
    # ...
    tscv = TimeSeriesSplit(n_splits=5)
    
    print(f"      Defined {tscv.get_n_splits()} temporal folds.")

    # 3. Define Model and Hyperparameter Space
    print("\n[3/5] Configuring LightGBM with Regularization...")
    
    # LightGBM is the industry standard for credit risk
    # We use 'class_weight'='balanced' to handle the default imbalance
    lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        class_weight='balanced',
        n_jobs=-1,
        verbosity=-1
    )
    
    # Hyperparameter Grid focused on Regularization & Capacity
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 70],          # Controls complexity
        'max_depth': [-1, 10, 20],           # Controls depth
        'reg_alpha': [0.0, 0.1, 1.0, 10.0],  # L1 Regularization (Feature Selection)
        'reg_lambda': [0.0, 0.1, 1.0, 10.0], # L2 Regularization
        'subsample': [0.7, 0.8, 0.9],        # Row sampling
        'colsample_bytree': [0.7, 0.8, 0.9]  # Feature sampling
    }

    # 4. Run Randomized Search
    print("\n[4/5] Running RandomizedSearchCV (Temporal)...")
    print("      This may take a few minutes depending on your CPU...")
    
    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=15,               # Number of parameter settings to sample
        scoring='roc_auc',       # Optimize for Area Under Curve
        cv=tscv,                 # Use Time Series Split
        verbose=1,
        random_state=42,
        n_jobs=4                 # Parallel jobs
    )
    
    search.fit(X_train, y_train)
    
    print("\n      Best CV AUC Score: {:.4f}".format(search.best_score_))
    print("      Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"        - {param}: {value}")

    # 5. Save Best Model and Analyze Features
    print("\n[5/5] Saving Best Model and Feature Importance...")
    
    best_model = search.best_estimator_
    
    # Save model
    model_path = base_dir / "best_model_lgbm.joblib"
    joblib.dump(best_model, model_path)
    print(f"      Model saved to: {model_path.name}")
    
    # Extract Feature Importance (L1 regularization naturally selects features)
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    csv_path = base_dir / "feature_importance.csv"
    importance.to_csv(csv_path, index=False)
    
    print("\n      Top 10 Features:")
    print(importance.head(10).to_string(index=False))
    
    # Count zero-importance features (pruned by L1/Trees)
    zero_imp_count = len(importance[importance['importance'] == 0])
    print(f"\n      Features with Zero Importance: {zero_imp_count} (Candidates for removal)")
    
    print("\n" + "="*80)
    print("MODEL SELECTION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_model_selection()
