"""
Credit Risk ML Model - Prediction Drivers Visualization Script
This script creates comprehensive visualizations using your trained LightGBM model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Load the trained model
print("Loading trained LightGBM model...")
model_path = '/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/models/best_model_lgbm.joblib'

# Compatibility shim: some models pickled with older NumPy internals reference
# `numpy._core.multiarray` which may not exist on newer NumPy installs. Create
# a lightweight module alias so unpickling succeeds.
import sys
import types
try:
    # If the legacy module already exists, do nothing
    import numpy._core  # type: ignore
except ModuleNotFoundError:
    try:
        import numpy as _np
        import numpy.core.multiarray as _multiarray
        _shim = types.ModuleType("numpy._core")
        _shim.multiarray = _multiarray
        sys.modules["numpy._core"] = _shim
        sys.modules["numpy._core.multiarray"] = _multiarray
    except Exception:
        # If we cannot create the shim, let the subsequent load raise the original error
        pass

model = joblib.load(model_path)
print(f"✓ Model loaded successfully")
print(f"Model type: {type(model)}")

# Read data
print("\nLoading data...")
file_path = '/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/data/final_train_merged.parquet'
df = pd.read_parquet(file_path, engine='pyarrow')
df = df.head(500000)

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Identify target variable (common names for credit risk)
target_cols = ['default', 'Default', 'target', 'Target', 'bad_loan', 'charged_off', 
               'loan_status', 'is_default', 'default_flag', 'y', 'label']
target_col = None
for col in target_cols:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\nWarning: Could not identify target column. Please set target_col manually.")
    print("Available columns:", df.columns.tolist())
    exit()

print(f"\nTarget variable: {target_col}")
print(f"Target distribution:\n{df[target_col].value_counts()}")

# Get feature names from the model
try:
    # Try to get feature names from the model
    if hasattr(model, 'feature_name_'):
        feature_names = model.feature_name_
    elif hasattr(model, 'feature_names_'):
        feature_names = model.feature_names_
    elif hasattr(model, 'feature_name'):
        feature_names = model.feature_name
    else:
        # If model is wrapped (e.g., in Pipeline), try to access the estimator
        if hasattr(model, 'named_steps'):
            estimator = model.named_steps[list(model.named_steps.keys())[-1]]
            feature_names = estimator.feature_name_
        else:
            # Fall back to using all non-target columns
            feature_names = [col for col in df.columns if col != target_col]
    
    print(f"\nNumber of features in model: {len(feature_names)}")
    
except Exception as e:
    print(f"Note: Using all non-target columns as features")
    feature_names = [col for col in df.columns if col != target_col]

# Get feature importance from the trained model
try:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps'):
        estimator = model.named_steps[list(model.named_steps.keys())[-1]]
        importances = estimator.feature_importances_
    else:
        raise AttributeError("Could not find feature_importances_")
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*60)
    print("Top 20 Features by Importance (from your trained model):")
    print("="*60)
    print(feature_importance.head(20).to_string(index=False))
    
except Exception as e:
    print(f"Error extracting feature importance: {e}")
    print("Please check the model structure.")
    exit()

# Separate features by type
X = df.drop(columns=[target_col])
y = df[target_col]

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nNumerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Create visualizations
output_dir = '/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/visualizations/'
import os
os.makedirs(output_dir, exist_ok=True)

print("\n" + "="*60)
print("Creating visualizations...")
print("="*60)

# 1. Feature Importance Bar Chart
plt.figure(figsize=(12, 8))
top_n = 20
top_features = feature_importance.head(top_n)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title(f'Top {top_n} Feature Importance - Trained LightGBM Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{output_dir}1_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 1_feature_importance.png")
plt.close()

# 2. Distribution of Top Features by Target
top_features_list = feature_importance.head(6)['feature'].tolist()
# Filter to only features that exist in the dataframe
top_features_list = [f for f in top_features_list if f in df.columns][:5]

if top_features_list:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features_list):
        if feature in numerical_cols:
            # Numerical feature - box plot
            data_to_plot = [df[df[target_col] == val][feature].dropna() for val in sorted(df[target_col].unique())]
            bp = axes[idx].boxplot(data_to_plot, labels=sorted(df[target_col].unique()), patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            axes[idx].set_title(f'{feature} Distribution by {target_col}', fontweight='bold')
            axes[idx].set_xlabel(target_col)
            axes[idx].set_ylabel(feature)
            axes[idx].grid(alpha=0.3)
        else:
            # Categorical feature - count plot
            temp_df = df[[feature, target_col]].copy()
            temp_df[target_col] = temp_df[target_col].astype(str)
            cross_tab = pd.crosstab(temp_df[feature], temp_df[target_col], normalize='index') * 100
            cross_tab.plot(kind='bar', ax=axes[idx], stacked=False, color=['steelblue', 'coral'])
            axes[idx].set_title(f'{feature} Distribution by {target_col}', fontweight='bold')
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Percentage (%)')
            axes[idx].legend(title=target_col)
            axes[idx].tick_params(axis='x', rotation=45)
    
    # Remove extra subplot
    if len(top_features_list) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}2_top_features_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 2_top_features_distribution.png")
    plt.close()

# 3. Correlation Heatmap for Top Numerical Features
top_numerical = [f for f in feature_importance.head(15)['feature'] if f in numerical_cols][:10]
if len(top_numerical) >= 2:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[top_numerical + [target_col]].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Heatmap - Top Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 3_correlation_heatmap.png")
    plt.close()

# 4. Default Rate by Top Categorical Features
top_categorical = [f for f in feature_importance.head(10)['feature'] if f in categorical_cols][:3]
if top_categorical:
    fig, axes = plt.subplots(1, len(top_categorical), figsize=(5*len(top_categorical), 5))
    if len(top_categorical) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(top_categorical):
        default_rate = df.groupby(feature)[target_col].mean().sort_values(ascending=False).head(15)
        bars = default_rate.plot(kind='bar', ax=axes[idx], color='coral')
        axes[idx].set_title(f'Default Rate by {feature}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel(feature, fontweight='bold')
        axes[idx].set_ylabel('Default Rate', fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim(0, max(default_rate) * 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}4_default_rate_by_category.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 4_default_rate_by_category.png")
    plt.close()

# 5. Feature Importance Cumulative Plot
plt.figure(figsize=(12, 6))
cumulative_importance = feature_importance['importance'].cumsum()
plt.plot(range(1, len(cumulative_importance)+1), cumulative_importance, 'b-', linewidth=2, label='Cumulative Importance')
plt.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% Threshold')
plt.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% Threshold')
plt.xlabel('Number of Features', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
plt.title('Cumulative Feature Importance - How Many Features Drive Predictions?', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim(0, min(100, len(cumulative_importance)))
plt.tight_layout()
plt.savefig(f'{output_dir}5_cumulative_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 5_cumulative_importance.png")
plt.close()

# 6. Histogram distributions for top numerical features
top_num_features = [f for f in feature_importance.head(15)['feature'] if f in numerical_cols][:6]
if top_num_features:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_num_features):
        for target_val in sorted(df[target_col].unique()):
            subset = df[df[target_col] == target_val][feature].dropna()
            axes[idx].hist(subset, bins=30, alpha=0.6, label=f'{target_col}={target_val}', edgecolor='black')
        
        axes[idx].set_title(f'Distribution: {feature}', fontweight='bold', fontsize=11)
        axes[idx].set_xlabel(feature, fontweight='bold')
        axes[idx].set_ylabel('Frequency', fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}6_numerical_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 6_numerical_distributions.png")
    plt.close()

# 7. Feature Importance by Type (Categorical vs Numerical)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Separate importance by feature type
categorical_importance = feature_importance[feature_importance['feature'].isin(categorical_cols)]
numerical_importance = feature_importance[feature_importance['feature'].isin(numerical_cols)]

# Plot categorical features
if not categorical_importance.empty:
    top_cat = categorical_importance.head(10)
    ax1.barh(range(len(top_cat)), top_cat['importance'], color='coral')
    ax1.set_yticks(range(len(top_cat)))
    ax1.set_yticklabels(top_cat['feature'])
    ax1.set_xlabel('Importance Score', fontweight='bold')
    ax1.set_title('Top 10 Categorical Features', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(alpha=0.3, axis='x')

# Plot numerical features
if not numerical_importance.empty:
    top_num = numerical_importance.head(10)
    ax2.barh(range(len(top_num)), top_num['importance'], color='steelblue')
    ax2.set_yticks(range(len(top_num)))
    ax2.set_yticklabels(top_num['feature'])
    ax2.set_xlabel('Importance Score', fontweight='bold')
    ax2.set_title('Top 10 Numerical Features', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}7_importance_by_type.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 7_importance_by_type.png")
plt.close()

# Save feature importance to CSV
feature_importance.to_csv(f'{output_dir}feature_importance_scores.csv', index=False)
print(f"✓ Saved: feature_importance_scores.csv")

# Calculate metrics for summary
features_for_80 = (cumulative_importance >= 0.8).argmax() + 1
features_for_90 = (cumulative_importance >= 0.9).argmax() + 1

print("\n" + "="*60)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"\nModel Performance Insights:")
print(f"  • Total features in model: {len(feature_importance)}")
print(f"  • Top feature: {feature_importance.iloc[0]['feature']}")
print(f"  • Top feature importance: {feature_importance.iloc[0]['importance']:.4f}")
print(f"  • Features for 80% importance: {features_for_80}")
print(f"  • Features for 90% importance: {features_for_90}")
print(f"  • Categorical features: {len(categorical_importance)}")
print(f"  • Numerical features: {len(numerical_importance)}")
print("\nGenerated Files:")
print("  1. 1_feature_importance.png")
print("  2. 2_top_features_distribution.png")
print("  3. 3_correlation_heatmap.png")
print("  4. 4_default_rate_by_category.png")
print("  5. 5_cumulative_importance.png")
print("  6. 6_numerical_distributions.png")
print("  7. 7_importance_by_type.png")
print("  8. feature_importance_scores.csv")
print("="*60)
