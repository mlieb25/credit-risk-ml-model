#!/usr/bin/env python3
"""
Final Model Evaluation Script
Author: Mitchell Liebrecht
Date: January 2026

This script evaluates the trained Credit Risk model on the held-out Test Set.
It calculates standard industry metrics and generates visualization plots.

Metrics calculated:
- ROC AUC (Area Under Receiver Operating Characteristic Curve)
- Gini Coefficient (Standard financial metric: 2*AUC - 1)
- Precision, Recall, F1-Score
- Confusion Matrix
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve,
    precision_recall_curve
)

def evaluate_final_model():
    print("="*80)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("="*80)
    
    base_dir = Path(__file__).parent
    
    # 1. Load Artifacts
    print("\n[1/5] Loading Test Data and Model...")
    try:
        X_test = pd.read_parquet(base_dir / "X_test_processed.parquet")
        y_test = pd.read_parquet(base_dir / "y_test.parquet").iloc[:, 0]
        model = joblib.load(base_dir / "best_model_lgbm.joblib")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the previous pipeline scripts first.")
        return

    print(f"      Test Set Size: {len(X_test):,} rows")
    print(f"      Model Type:    {type(model).__name__}")

    # 2. Make Predictions
    print("\n[2/5] Generating Predictions...")
    # Probabilities for class 1 (Default)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Hard class predictions (0 or 1)
    y_pred = model.predict(X_test)

    # 3. Calculate Metrics
    print("\n[3/5] Calculating Performance Metrics...")
    
    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1
    
    print("\n" + "-"*40)
    print(f"KEY PERFORMANCE INDICATORS")
    print("-" * 40)
    print(f"ROC AUC Score:    {auc:.4f} (Target: >0.75)")
    print(f"Gini Coefficient: {gini:.4f} (Target: >0.50)")
    print("-" * 40)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix Summary:")
    print(f"True Negatives (Good Loans Correctly Approved): {tn:,}")
    print(f"False Positives (Good Loans Incorrectly Rejected): {fp:,}")
    print(f"False Negatives (Bad Loans Incorrectly Approved):  {fn:,}  <-- CRITICAL RISK")
    print(f"True Positives  (Bad Loans Correctly Rejected):    {tp:,}")

    # 4. Generate Visualizations
    print("\n[4/5] Generating Plots...")
    
    # Setup plots folder
    plots_dir = base_dir / "evaluation_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # A. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'LightGBM (AUC = {auc:.3f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / "roc_curve.png")
    plt.close()
    
    # B. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(plots_dir / "confusion_matrix.png")
    plt.close()
    
    # C. Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_test==0], bins=50, alpha=0.5, label='Non-Default (0)', density=True, color='green')
    plt.hist(y_prob[y_test==1], bins=50, alpha=0.5, label='Default (1)', density=True, color='red')
    plt.xlabel('Predicted Probability of Default')
    plt.ylabel('Density')
    plt.title('Distribution of Predictions by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / "probability_distribution.png")
    plt.close()

    # 4.E SHAP Feature Importance
    print("\n[5/6] Calculating SHAP Feature Importances and Exporting Feature Importances CSV...")
    try:
        import shap

        # Use a sample for SHAP to limit runtime on large test sets
        shap_sample = X_test
        max_shap_samples = 5000
        if len(X_test) > max_shap_samples:
            shap_sample = X_test.sample(n=max_shap_samples, random_state=42)
            print(f"      SHAP: using sample of {len(shap_sample):,} rows (out of {len(X_test):,}) to speed up computation.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_sample)

        # For binary classifiers shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_vals_for_pos = shap_values[1]
        else:
            shap_vals_for_pos = shap_values

        mean_abs_shap = np.abs(shap_vals_for_pos).mean(axis=0)
        feat_importance_df = pd.DataFrame({
            'feature': shap_sample.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        feat_importance_df['importance_pct'] = 100 * feat_importance_df['mean_abs_shap'] / feat_importance_df['mean_abs_shap'].sum()

        shap_csv_path = base_dir / 'shap_feature_importances.csv'
        feat_importance_df.to_csv(shap_csv_path, index=False)
        print(f"      ✓ SHAP feature importances saved to: {shap_csv_path.name}")

        # Save bar plot of top features
        top_n = 20
        plt.figure(figsize=(10, max(6, top_n * 0.35)))
        sns.barplot(x='mean_abs_shap', y='feature', data=feat_importance_df.head(top_n), palette='viridis')
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top {top_n} Features by Mean Absolute SHAP')
        plt.tight_layout()
        plt.savefig(plots_dir / 'shap_feature_importance.png')
        plt.close()

        # Optionally save raw SHAP values per sample (commented out by default)
        # raw_shap_path = base_dir / 'shap_values.parquet'
        # pd.DataFrame(shap_vals_for_pos, columns=shap_sample.columns).to_parquet(raw_shap_path)

    except Exception as e:
        print(f"      SHAP calculation skipped or failed: {e}")
        print("      To enable SHAP, install the 'shap' package (pip install shap) and ensure the model is tree-based.")

    # 5. Stability Analysis (Basic)
    print("\n[6/6] Checking Probability Stability...")
    print(f"      Mean Predicted Probability: {y_prob.mean():.4f}")
    print(f"      Actual Default Rate:        {y_test.mean():.4f}")
    
    calib_diff = y_prob.mean() - y_test.mean()
    print(f"      Calibration Drift:          {calib_diff:.4f}")
    if abs(calib_diff) > 0.05:
        print("      WARNING: Model may be miscalibrated (probabilities don't match base rates).")
    else:
        print("      ✓ Calibration looks reasonable.")

    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE. Plots saved to: {plots_dir}")
    print("="*80)

if __name__ == "__main__":
    evaluate_final_model()
