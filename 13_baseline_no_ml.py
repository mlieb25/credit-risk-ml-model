"""
Baseline Model Performance - No Machine Learning

Creates simple heuristic-based baselines to establish minimum performance thresholds:
1. Majority Class Baseline - Predict most common class for all cases
2. Random Guessing Baseline - Random predictions based on class distribution
3. Simple Rule-Based Baseline - Basic thresholds on key features

Evaluates on standard credit risk metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (Area Under Receiver Operating Characteristic)
- Gini Coefficient (2*AUC - 1)
- Confusion Matrix

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model")
DATA_FILE = BASE_PATH / "final_train_merged.csv"

print("="*80)
print("BASELINE MODEL EVALUATION - NO MACHINE LEARNING")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA AND PREPARE
# ============================================================================

print("\n[Step 1] Loading final training dataset...")

if not DATA_FILE.exists():
    print(f"\nERROR: {DATA_FILE} not found!")
    print("Please run master_merge.py first to create the final merged dataset.")
    exit()

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"  Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Check for target variable
if 'target' not in df.columns:
    print("\nERROR: 'target' column not found in dataset!")
    exit()

print(f"\n[Step 2] Analyzing target variable distribution...")
target_counts = df['target'].value_counts().sort_index()
print(f"\nTarget Distribution:")
print(f"  Class 0 (No Default): {target_counts[0]:>10,} ({target_counts[0]/len(df)*100:>6.2f}%)")
print(f"  Class 1 (Default):    {target_counts[1]:>10,} ({target_counts[1]/len(df)*100:>6.2f}%)")
print(f"  Imbalance Ratio:      {target_counts[0]/target_counts[1]:>10.2f}:1")

majority_class = target_counts.idxmax()
minority_class = target_counts.idxmin()
baseline_accuracy = target_counts[majority_class] / len(df)

print(f"\n  Majority Class: {majority_class}")
print(f"  If we always predict class {majority_class}, accuracy = {baseline_accuracy:.4f}")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================

print(f"\n[Step 3] Splitting data into train (80%) and test (20%) sets...")

# Use stratified split to maintain class balance
y = df['target']
X_indices = df.index

X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    X_indices, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set:   {len(y_train):>10,} samples")
print(f"    Class 0: {(y_train == 0).sum():>10,} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"    Class 1: {(y_train == 1).sum():>10,} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
print(f"\n  Test set:       {len(y_test):>10,} samples")
print(f"    Class 0: {(y_test == 0).sum():>10,} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")
print(f"    Class 1: {(y_test == 1).sum():>10,} ({(y_test == 1).sum()/len(y_test)*100:.2f}%)")

# ============================================================================
# STEP 4: BASELINE 1 - MAJORITY CLASS PREDICTOR
# ============================================================================

print("\n" + "="*80)
print("BASELINE 1: MAJORITY CLASS PREDICTOR")
print("="*80)
print("\nStrategy: Predict the majority class (Class 0) for ALL cases.")
print("This is the simplest possible baseline.")

# Predict majority class for all test samples
y_pred_majority = np.full(len(y_test), majority_class)

# Calculate metrics
acc_majority = accuracy_score(y_test, y_pred_majority)
prec_majority = precision_score(y_test, y_pred_majority, zero_division=0)
rec_majority = recall_score(y_test, y_pred_majority, zero_division=0)
f1_majority = f1_score(y_test, y_pred_majority, zero_division=0)

# Note: ROC-AUC requires probability scores, not binary predictions
# For majority class, we can't calculate AUC (no discrimination)
print(f"\nPerformance Metrics:")
print(f"  Accuracy:       {acc_majority:.4f}")
print(f"  Precision:      {prec_majority:.4f}")
print(f"  Recall:         {rec_majority:.4f}")
print(f"  F1-Score:       {f1_majority:.4f}")
print(f"  ROC-AUC:        N/A (no probability scores)")
print(f"  Gini:           N/A")

print(f"\nConfusion Matrix:")
cm_majority = confusion_matrix(y_test, y_pred_majority)
print(f"                 Predicted 0    Predicted 1")
print(f"  Actual 0:      {cm_majority[0,0]:>10,}    {cm_majority[0,1]:>10,}")
print(f"  Actual 1:      {cm_majority[1,0]:>10,}    {cm_majority[1,1]:>10,}")

print(f"\nInterpretation:")
print(f"  ✓ Correctly identifies {cm_majority[0,0]:,} non-defaults (True Negatives)")
print(f"  ✗ Misses ALL {cm_majority[1,0]:,} defaults (False Negatives)")
print(f"  This model has 0% recall for the minority class - useless for risk management!")

# ============================================================================
# STEP 5: BASELINE 2 - RANDOM GUESSING
# ============================================================================

print("\n" + "="*80)
print("BASELINE 2: RANDOM GUESSING (STRATIFIED)")
print("="*80)
print("\nStrategy: Randomly assign classes based on training set distribution.")
print("This establishes the 'expected performance by chance' baseline.")

np.random.seed(42)
class_probs = [target_counts[0]/len(df), target_counts[1]/len(df)]
y_pred_random = np.random.choice([0, 1], size=len(y_test), p=class_probs)

# For probability-based metrics, generate random probabilities
y_pred_random_proba = np.random.uniform(0, 1, size=len(y_test))

acc_random = accuracy_score(y_test, y_pred_random)
prec_random = precision_score(y_test, y_pred_random, zero_division=0)
rec_random = recall_score(y_test, y_pred_random, zero_division=0)
f1_random = f1_score(y_test, y_pred_random, zero_division=0)
auc_random = roc_auc_score(y_test, y_pred_random_proba)
gini_random = 2 * auc_random - 1

print(f"\nPerformance Metrics:")
print(f"  Accuracy:       {acc_random:.4f}")
print(f"  Precision:      {prec_random:.4f}")
print(f"  Recall:         {rec_random:.4f}")
print(f"  F1-Score:       {f1_random:.4f}")
print(f"  ROC-AUC:        {auc_random:.4f}")
print(f"  Gini:           {gini_random:.4f}")

print(f"\nConfusion Matrix:")
cm_random = confusion_matrix(y_test, y_pred_random)
print(f"                 Predicted 0    Predicted 1")
print(f"  Actual 0:      {cm_random[0,0]:>10,}    {cm_random[0,1]:>10,}")
print(f"  Actual 1:      {cm_random[1,0]:>10,}    {cm_random[1,1]:>10,}")

print(f"\nInterpretation:")
print(f"  Random guessing performs poorly but at least detects SOME defaults.")
print(f"  AUC ≈ 0.50 means no better than a coin flip.")

# ============================================================================
# STEP 6: BASELINE 3 - SIMPLE RULE-BASED MODEL
# ============================================================================

print("\n" + "="*80)
print("BASELINE 3: SIMPLE RULE-BASED MODEL")
print("="*80)
print("\nStrategy: Use basic business rules on key features to predict default risk.")
print("Rules based on common credit underwriting heuristics.")

# Extract test set data
df_test = df.loc[X_test_idx].copy()

# Define simple rules (adjust column names based on what exists)
print("\nApplying rules:")
print("  Rule 1: High bureau overdue count → Predict Default")
print("  Rule 2: Low income + high loan amount → Predict Default")
print("  Rule 3: Previous refused applications → Predict Default")
print("  Rule 4: Otherwise → Predict No Default")

# Initialize predictions as majority class
y_pred_rules = np.zeros(len(df_test))

# Rule 1: Bureau overdue
if 'bureau_overdue_count' in df_test.columns:
    mask = df_test['bureau_overdue_count'].fillna(0) > 2
    y_pred_rules[mask] = 1
    print(f"    Applied Rule 1 to {mask.sum():,} cases")

# Rule 2: Income-to-loan ratio (if columns exist)
if 'mainoccupationinc_384A' in df_test.columns and 'credamount_770A' in df_test.columns:
    income = df_test['mainoccupationinc_384A'].fillna(df_test['mainoccupationinc_384A'].median())
    loan_amt = df_test['credamount_770A'].fillna(df_test['credamount_770A'].median())
    
    # Simple heuristic: loan > 50% of annual income
    mask = (loan_amt / (income * 12)) > 0.5
    y_pred_rules[mask] = 1
    print(f"    Applied Rule 2 to {mask.sum():,} cases")

# Rule 3: Previous refusals
if 'applprev_refused_count' in df_test.columns:
    mask = df_test['applprev_refused_count'].fillna(0) > 1
    y_pred_rules[mask] = 1
    print(f"    Applied Rule 3 to {mask.sum():,} cases")

# Create probability-like scores for ROC-AUC
# Count how many rules triggered (0-3) and normalize
rule_score = np.zeros(len(df_test))
if 'bureau_overdue_count' in df_test.columns:
    rule_score += (df_test['bureau_overdue_count'].fillna(0) > 2).astype(int)
if 'mainoccupationinc_384A' in df_test.columns and 'credamount_770A' in df_test.columns:
    income = df_test['mainoccupationinc_384A'].fillna(df_test['mainoccupationinc_384A'].median())
    loan_amt = df_test['credamount_770A'].fillna(df_test['credamount_770A'].median())
    rule_score += ((loan_amt / (income * 12)) > 0.5).astype(int)
if 'applprev_refused_count' in df_test.columns:
    rule_score += (df_test['applprev_refused_count'].fillna(0) > 1).astype(int)

y_pred_rules_proba = rule_score / 3.0  # Normalize to 0-1 range

acc_rules = accuracy_score(y_test, y_pred_rules)
prec_rules = precision_score(y_test, y_pred_rules, zero_division=0)
rec_rules = recall_score(y_test, y_pred_rules, zero_division=0)
f1_rules = f1_score(y_test, y_pred_rules, zero_division=0)

# Calculate AUC if we have probability scores
if y_pred_rules_proba.std() > 0:  # Check if scores vary
    auc_rules = roc_auc_score(y_test, y_pred_rules_proba)
    gini_rules = 2 * auc_rules - 1
else:
    auc_rules = None
    gini_rules = None

print(f"\nPerformance Metrics:")
print(f"  Accuracy:       {acc_rules:.4f}")
print(f"  Precision:      {prec_rules:.4f}")
print(f"  Recall:         {rec_rules:.4f}")
print(f"  F1-Score:       {f1_rules:.4f}")
if auc_rules:
    print(f"  ROC-AUC:        {auc_rules:.4f}")
    print(f"  Gini:           {gini_rules:.4f}")
else:
    print(f"  ROC-AUC:        N/A (insufficient score variation)")
    print(f"  Gini:           N/A")

print(f"\nConfusion Matrix:")
cm_rules = confusion_matrix(y_test, y_pred_rules)
print(f"                 Predicted 0    Predicted 1")
print(f"  Actual 0:      {cm_rules[0,0]:>10,}    {cm_rules[0,1]:>10,}")
print(f"  Actual 1:      {cm_rules[1,0]:>10,}    {cm_rules[1,1]:>10,}")

print(f"\nInterpretation:")
print(f"  Simple rules catch {cm_rules[1,1]:,} defaults but also misclassify {cm_rules[0,1]:,} good loans.")
print(f"  Better than random, but still far from optimal.")

# ============================================================================
# STEP 7: COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BASELINE COMPARISON SUMMARY")
print("="*80)

results = pd.DataFrame({
    'Model': ['Majority Class', 'Random Guessing', 'Simple Rules'],
    'Accuracy': [acc_majority, acc_random, acc_rules],
    'Precision': [prec_majority, prec_random, prec_rules],
    'Recall': [rec_majority, rec_random, rec_rules],
    'F1-Score': [f1_majority, f1_random, f1_rules],
    'ROC-AUC': [None, auc_random, auc_rules if auc_rules else None],
    'Gini': [None, gini_random, gini_rules if gini_rules else None]
})

print("\n" + results.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("\n1. MAJORITY CLASS BASELINE:")
print(f"   - Achieves {acc_majority:.2%} accuracy simply by predicting 'No Default' for everyone.")
print(f"   - However, it has 0% recall for defaults - completely useless for risk management.")
print(f"   - This sets the MINIMUM accuracy any ML model must beat.")

print("\n2. RANDOM GUESSING:")
print(f"   - ROC-AUC ≈ 0.50 represents pure chance (no predictive power).")
print(f"   - Any ML model with AUC < 0.55 is essentially worthless.")

print("\n3. SIMPLE RULES:")
if auc_rules and auc_rules > 0.50:
    print(f"   - Shows that basic credit underwriting rules have SOME signal.")
    print(f"   - AUC = {auc_rules:.4f} means the model ranks defaults higher than non-defaults {auc_rules:.1%} of the time.")
    print(f"   - Sets a realistic 'domain expert' baseline that ML should significantly outperform.")
else:
    print(f"   - Rules show marginal improvement but lack sufficient discrimination.")
    print(f"   - Highlights the need for machine learning to capture complex interactions.")

print("\n4. TARGET PERFORMANCE FOR ML MODELS:")
print(f"   - Minimum acceptable AUC: > 0.55 (better than random)")
print(f"   - Good performance: AUC > 0.70")
print(f"   - Excellent performance: AUC > 0.80")
print(f"   - Industry benchmark (Kaggle leaderboard): Check competition metrics")

print("\n5. BUSINESS CONTEXT:")
print(f"   - Default rate = {(y_test == 1).sum()/len(y_test):.2%}")
print(f"   - Missing ONE default could cost 5-10x the profit from approving a good loan.")
print(f"   - Therefore, RECALL (catching defaults) is MORE important than precision.")
print(f"   - ML model should aim for 60-70% recall while maintaining reasonable precision.")

# ============================================================================
# STEP 8: VISUALIZATIONS
# ============================================================================

print("\n[Step 8] Creating comparison visualizations...")

# Figure 1: Metrics Comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Accuracy, Precision, Recall, F1
metrics_df = results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].set_index('Model')
metrics_df.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black')
axes[0].set_title('Baseline Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].set_ylim(0, 1)
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

# Subplot 2: ROC-AUC and Gini
auc_gini_df = results[['Model', 'ROC-AUC', 'Gini']].set_index('Model')
auc_gini_df.plot(kind='bar', ax=axes[1], color=['#9b59b6', '#e67e22'], alpha=0.8, edgecolor='black')
axes[1].set_title('ROC-AUC and Gini Coefficient', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_xlabel('Model', fontsize=12)
axes[1].legend(loc='upper left', fontsize=10)
axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (AUC=0.5)')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
fig1_path = BASE_PATH / "baseline_metrics_comparison.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig1_path}")
plt.close()

# Figure 2: Confusion Matrices
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

for idx, (cm, title) in enumerate([
    (cm_majority, 'Majority Class'),
    (cm_random, 'Random Guessing'),
    (cm_rules, 'Simple Rules')
]):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes2[idx], 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                cbar=False)
    axes2[idx].set_title(f'{title}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes2[idx].set_ylabel('Actual', fontsize=11)
    axes2[idx].set_xlabel('Predicted', fontsize=11)

plt.tight_layout()
fig2_path = BASE_PATH / "baseline_confusion_matrices.png"
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig2_path}")
plt.close()

print("\n" + "="*80)
print("BASELINE EVALUATION COMPLETE")
print("="*80)
print("\nSummary:")
print(f"  - Established 3 baselines: Majority Class, Random, and Simple Rules")
print(f"  - Best baseline performance: {results['Model'].iloc[results['F1-Score'].idxmax()]} (F1={results['F1-Score'].max():.4f})")
if auc_rules:
    print(f"  - Best baseline AUC: {auc_rules:.4f} (Gini: {gini_rules:.4f})")
print(f"\n  Next Step: Train ML models (XGBoost, LightGBM) and aim for AUC > 0.70")
