"""
Tier 1 EDA: train_base.csv and train_static Analysis
BAIT 509 Final Project - Home Credit Risk Model

Analyzes:
1. train_base.csv - Target distribution and temporal structure
2. train_static_0_0.csv - Core features, missing values, distributions

Author: Mitchell Liebrecht
Date: January 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/home-credit-credit-risk-model-stability/csv_files/train")
OUTPUT_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/eda_output")

# Create output directory
OUTPUT_PATH.mkdir(exist_ok=True)

# Initialize report lines
report_lines = []

print("="*80)
print("TIER 1 EXPLORATORY DATA ANALYSIS")
print("="*80)

report_lines.append("="*80)
report_lines.append("TIER 1 EXPLORATORY DATA ANALYSIS")
report_lines.append("HOME CREDIT RISK MODEL - BAIT 509 Final Project")
report_lines.append(f"Generated: January 17, 2026")
report_lines.append("="*80)

# ============================================================================
# 1. TRAIN_BASE.CSV - THE FOUNDATION
# ============================================================================
print("\n" + "="*80)
print("[1] TRAIN_BASE.CSV ANALYSIS")
print("="*80)

report_lines.append("\n" + "="*80)
report_lines.append("[1] TRAIN_BASE.CSV ANALYSIS")
report_lines.append("="*80)

print("\n[Loading train_base.csv...]")
df_base = pd.read_csv(DATA_PATH / "train_base.csv")
print(f"Loaded: {df_base.shape[0]:,} rows × {df_base.shape[1]} columns")
print(f"Memory: {df_base.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

report_lines.append(f"\nDataset: train_base.csv")
report_lines.append(f"Rows: {df_base.shape[0]:,}")
report_lines.append(f"Columns: {df_base.shape[1]}")
report_lines.append(f"Memory: {df_base.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n[Column Overview]")
print(df_base.dtypes)
print("\nFirst 5 rows:")
print(df_base.head())

# ----------------------------------------------------------------------------
# TARGET DISTRIBUTION
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("TARGET DISTRIBUTION ANALYSIS")
print("-"*80)

report_lines.append("\n" + "-"*80)
report_lines.append("TARGET DISTRIBUTION ANALYSIS")
report_lines.append("-"*80)

if 'target' in df_base.columns:
    print("\n[Overall Target Distribution]")
    target_counts = df_base['target'].value_counts().sort_index()
    target_pct = df_base['target'].value_counts(normalize=True).sort_index() * 100
    
    for val in target_counts.index:
        count = target_counts[val]
        pct = target_pct[val]
        label = "Non-Default" if val == 0 else "Default"
        print(f"  Target = {val} ({label}): {count:,} ({pct:.2f}%)")
        report_lines.append(f"  Target = {val} ({label}): {count:,} ({pct:.2f}%)")
    
    # Calculate imbalance ratio
    if len(target_counts) == 2:
        imbalance_ratio = target_counts.max() / target_counts.min()
        print(f"\n  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"  Default Rate: {target_pct[1]:.2f}%")
        report_lines.append(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        report_lines.append(f"Default Rate: {target_pct[1]:.2f}%")
        
        # PLOT 1: Target Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        axes[0].bar(['Non-Default', 'Default'], target_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Target Distribution (Counts)', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, (idx, val) in enumerate(target_counts.items()):
            axes[0].text(i, val + max(target_counts)*0.01, f'{val:,}\n({val/len(df_base)*100:.1f}%)', 
                        ha='center', va='bottom', fontsize=11)
        
        # Percentage plot
        axes[1].bar(['Non-Default', 'Default'], target_pct.values, color=['#2ecc71', '#e74c3c'])
        axes[1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1].set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, (idx, val) in enumerate(target_pct.items()):
            axes[1].text(i, val + max(target_pct)*0.01, f'{val:.2f}%', 
                        ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / '1_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 1_target_distribution.png")
    
    # Check for missing target values
    missing_target = df_base['target'].isna().sum()
    if missing_target > 0:
        print(f"\n  WARNING: {missing_target:,} missing target values ({missing_target/len(df_base)*100:.2f}%)")
    
    # Target by WEEK_NUM
    print("\n[Target Distribution by Week]")
    if 'WEEK_NUM' in df_base.columns:
        weekly_stats = df_base.groupby('WEEK_NUM')['target'].agg([
            ('count', 'count'),
            ('default_rate', 'mean'),
            ('defaults', 'sum')
        ]).reset_index()
        weekly_stats['default_rate'] = weekly_stats['default_rate'] * 100
        
        print(f"\n  Weeks in dataset: {df_base['WEEK_NUM'].min()} to {df_base['WEEK_NUM'].max()}")
        print(f"  Total unique weeks: {df_base['WEEK_NUM'].nunique()}")
        print(f"\n  Sample of weekly statistics (first 10 weeks):")
        print(weekly_stats.head(10).to_string(index=False))
        
        print(f"\n  Default rate stability across weeks:")
        print(f"    Mean default rate: {weekly_stats['default_rate'].mean():.2f}%")
        print(f"    Std dev: {weekly_stats['default_rate'].std():.2f}%")
        print(f"    Min: {weekly_stats['default_rate'].min():.2f}% (Week {weekly_stats.loc[weekly_stats['default_rate'].idxmin(), 'WEEK_NUM']})")
        print(f"    Max: {weekly_stats['default_rate'].max():.2f}% (Week {weekly_stats.loc[weekly_stats['default_rate'].idxmax(), 'WEEK_NUM']})")
        
        report_lines.append(f"\nWeeks in dataset: {df_base['WEEK_NUM'].min()} to {df_base['WEEK_NUM'].max()}")
        report_lines.append(f"Total unique weeks: {df_base['WEEK_NUM'].nunique()}")
        report_lines.append(f"Mean default rate: {weekly_stats['default_rate'].mean():.2f}%")
        report_lines.append(f"Std dev: {weekly_stats['default_rate'].std():.2f}%")
        
        # PLOT 2: Application Volume Over Time
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Volume by week
        axes[0].plot(weekly_stats['WEEK_NUM'], weekly_stats['count'], linewidth=2, color='#3498db')
        axes[0].fill_between(weekly_stats['WEEK_NUM'], weekly_stats['count'], alpha=0.3, color='#3498db')
        axes[0].set_xlabel('Week Number', fontsize=12)
        axes[0].set_ylabel('Application Count', fontsize=12)
        axes[0].set_title('Application Volume Over Time (Weekly)', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Default rate by week
        axes[1].plot(weekly_stats['WEEK_NUM'], weekly_stats['default_rate'], linewidth=2, color='#e74c3c')
        axes[1].axhline(y=weekly_stats['default_rate'].mean(), color='black', linestyle='--', 
                       linewidth=1.5, label=f"Mean: {weekly_stats['default_rate'].mean():.2f}%")
        axes[1].fill_between(weekly_stats['WEEK_NUM'], weekly_stats['default_rate'], alpha=0.3, color='#e74c3c')
        axes[1].set_xlabel('Week Number', fontsize=12)
        axes[1].set_ylabel('Default Rate (%)', fontsize=12)
        axes[1].set_title('Default Rate Stability Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / '2_temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 2_temporal_trends.png")
    
    # Target by MONTH
    print("\n[Target Distribution by Month]")
    if 'MONTH' in df_base.columns:
        monthly_stats = df_base.groupby('MONTH')['target'].agg([
            ('count', 'count'),
            ('default_rate', 'mean')
        ]).reset_index()
        monthly_stats['default_rate'] = monthly_stats['default_rate'] * 100
        
        print(f"\n  Months in dataset: {df_base['MONTH'].min()} to {df_base['MONTH'].max()}")
        print(f"  Sample of monthly statistics:")
        print(monthly_stats.head(10).to_string(index=False))

else:
    print("\n  WARNING: 'target' column not found in train_base.csv")

# ----------------------------------------------------------------------------
# TEMPORAL COVERAGE
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("TEMPORAL COVERAGE ANALYSIS")
print("-"*80)

if 'date_decision' in df_base.columns:
    print("\n[Date Decision Analysis]")
    df_base['date_decision_parsed'] = pd.to_datetime(df_base['date_decision'], errors='coerce')
    
    min_date = df_base['date_decision_parsed'].min()
    max_date = df_base['date_decision_parsed'].max()
    date_range = (max_date - min_date).days
    
    print(f"  Date Range: {min_date.date()} to {max_date.date()}")
    print(f"  Span: {date_range} days ({date_range/365.25:.1f} years)")
    
    report_lines.append(f"\nDate Range: {min_date.date()} to {max_date.date()}")
    report_lines.append(f"Span: {date_range} days ({date_range/365.25:.1f} years)")
    
    # Check for missing dates
    missing_dates = df_base['date_decision_parsed'].isna().sum()
    if missing_dates > 0:
        print(f"  WARNING: {missing_dates:,} unparseable dates")
    
    # Application volume over time
    print("\n[Application Volume Over Time]")
    daily_volume = df_base.groupby('date_decision').size()
    print(f"  Average daily applications: {daily_volume.mean():.0f}")
    print(f"  Min daily: {daily_volume.min()} (Date: {daily_volume.idxmin()})")
    print(f"  Max daily: {daily_volume.max()} (Date: {daily_volume.idxmax()})")
    print(f"  Std dev: {daily_volume.std():.0f}")

if 'WEEK_NUM' in df_base.columns:
    print("\n[Week Number Analysis]")
    weekly_volume = df_base.groupby('WEEK_NUM').size()
    
    print(f"  Total weeks: {df_base['WEEK_NUM'].nunique()}")
    print(f"  Average applications per week: {weekly_volume.mean():.0f}")
    print(f"  Min weekly: {weekly_volume.min()} (Week {weekly_volume.idxmin()})")
    print(f"  Max weekly: {weekly_volume.max()} (Week {weekly_volume.idxmax()})")
    
    # Check for gaps
    all_weeks = set(range(df_base['WEEK_NUM'].min(), df_base['WEEK_NUM'].max() + 1))
    present_weeks = set(df_base['WEEK_NUM'].unique())
    missing_weeks = sorted(all_weeks - present_weeks)
    
    if missing_weeks:
        print(f"\n  WARNING: {len(missing_weeks)} missing weeks in sequence")
        if len(missing_weeks) <= 10:
            print(f"    Missing: {missing_weeks}")
    else:
        print(f"\n  ✓ No gaps in week sequence")

# ----------------------------------------------------------------------------
# MISSING VALUES
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("MISSING VALUES - TRAIN_BASE")
print("-"*80)

report_lines.append("\n" + "-"*80)
report_lines.append("MISSING VALUES - TRAIN_BASE")
report_lines.append("-"*80)

missing_base = df_base.isna().sum()
missing_base_pct = (missing_base / len(df_base) * 100).round(2)
missing_summary = pd.DataFrame({
    'column': df_base.columns,
    'missing_count': missing_base,
    'missing_pct': missing_base_pct
})
missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_pct', ascending=False)

if len(missing_summary) > 0:
    print(f"\n  Columns with missing values: {len(missing_summary)}/{len(df_base.columns)}")
    print("\n  Missing value details:")
    print(missing_summary.to_string(index=False))
else:
    print("\n  ✓ No missing values in train_base.csv")

# ----------------------------------------------------------------------------
# KEY STATISTICS
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("KEY STATISTICS - TRAIN_BASE")
print("-"*80)

print(f"\n  Unique case_id: {df_base['case_id'].nunique():,}")
print(f"  Total rows: {len(df_base):,}")
print(f"  Rows per case_id: {len(df_base) / df_base['case_id'].nunique():.2f}")

report_lines.append(f"\nUnique case_id: {df_base['case_id'].nunique():,}")
report_lines.append(f"Total rows: {len(df_base):,}")
report_lines.append(f"Rows per case_id: {len(df_base) / df_base['case_id'].nunique():.2f}")

if df_base['case_id'].nunique() != len(df_base):
    print(f"\n  WARNING: Duplicate case_ids detected!")
    dupe_count = len(df_base) - df_base['case_id'].nunique()
    print(f"    Duplicate rows: {dupe_count:,}")

# ============================================================================
# 2. TRAIN_STATIC_0_0.CSV - CORE FEATURES
# ============================================================================
print("\n\n" + "="*80)
print("[2] TRAIN_STATIC_0_0.CSV ANALYSIS")
print("="*80)

report_lines.append("\n\n" + "="*80)
report_lines.append("[2] TRAIN_STATIC_0_0.CSV ANALYSIS")
report_lines.append("="*80)

print("\n[Loading train_static_0_0.csv...]")
df_static = pd.read_csv(DATA_PATH / "train_static_0_0.csv")
print(f"Loaded: {df_static.shape[0]:,} rows × {df_static.shape[1]} columns")
print(f"Memory: {df_static.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

report_lines.append(f"\nDataset: train_static_0_0.csv")
report_lines.append(f"Rows: {df_static.shape[0]:,}")
report_lines.append(f"Columns: {df_static.shape[1]}")
report_lines.append(f"Memory: {df_static.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n[Cardinality Check]")
print(f"  Unique case_id: {df_static['case_id'].nunique():,}")
print(f"  Total rows: {len(df_static):,}")
if df_static['case_id'].nunique() == len(df_static):
    print(f"  ✓ One-to-one relationship with case_id")
else:
    print(f"  WARNING: Not one-to-one relationship")
    print(f"  Average rows per case_id: {len(df_static) / df_static['case_id'].nunique():.2f}")

# ----------------------------------------------------------------------------
# MISSING VALUE PATTERNS
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("MISSING VALUE PATTERNS - TRAIN_STATIC_0_0")
print("-"*80)

missing_static = df_static.isna().sum()
missing_static_pct = (missing_static / len(df_static) * 100).round(2)
missing_static_df = pd.DataFrame({
    'column': df_static.columns,
    'missing_count': missing_static,
    'missing_pct': missing_static_pct
})
missing_static_df = missing_static_df[missing_static_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

overall_missing_pct = (df_static.isna().sum().sum() / (len(df_static) * len(df_static.columns)) * 100)
print(f"\n  Total columns: {len(df_static.columns)}")
print(f"  Columns with missing values: {len(missing_static_df)}")
print(f"  Overall missing percentage: {overall_missing_pct:.2f}%")

report_lines.append(f"\nTotal columns: {len(df_static.columns)}")
report_lines.append(f"Columns with missing values: {len(missing_static_df)}")
report_lines.append(f"Overall missing percentage: {overall_missing_pct:.2f}%")

print("\n[Top 30 Columns by Missing Percentage]")
print(missing_static_df.head(30).to_string(index=False))

print("\n[Missing Value Threshold Analysis]")
thresholds = [10, 25, 50, 75, 90]
for thresh in thresholds:
    count = (missing_static_df['missing_pct'] > thresh).sum()
    print(f"  Columns with >{thresh}% missing: {count}")
    report_lines.append(f"  Columns with >{thresh}% missing: {count}")

# Identify columns with <50% missing (usable for Phase 2)
usable_cols = missing_static_df[missing_static_df['missing_pct'] < 50]['column'].tolist()
print(f"\n  Columns with <50% missing (Phase 2 candidates): {len(usable_cols)}")
report_lines.append(f"\nColumns with <50% missing (Phase 2 candidates): {len(usable_cols)}")

# PLOT 3: Missing Value Pattern
fig, ax = plt.subplots(figsize=(12, 8))
top_missing = missing_static_df.head(40)
ax.barh(range(len(top_missing)), top_missing['missing_pct'], color='#e74c3c', alpha=0.7)
ax.set_yticks(range(len(top_missing)))
ax.set_yticklabels(top_missing['column'], fontsize=8)
ax.set_xlabel('Missing Percentage (%)', fontsize=12)
ax.set_title('Top 40 Features by Missing Percentage (train_static_0_0)', fontsize=14, fontweight='bold')
ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='50% threshold')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / '3_missing_values.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 3_missing_values.png")

# ----------------------------------------------------------------------------
# FEATURE DISTRIBUTIONS
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("FEATURE DISTRIBUTIONS - KEY NUMERIC FEATURES")
print("-"*80)

# Identify key Phase 2 features
key_features = ['annuity_780A', 'currdebt_22A', 'credamount_770A', 'credamount_590A', 
                'avgoutstandbalancel6m_4187114A', 'avgdbddpdlast24m_3658932P', 
                'dpdmax_139P', 'amtinstpaidbefduel24m_4187115A']

available_features = [f for f in key_features if f in df_static.columns]

if available_features:
    print(f"\n[Analyzing {len(available_features)} key features]")
    
    # PLOT 4: Feature Distributions
    n_features = min(len(available_features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, feat in enumerate(available_features[:n_features]):
        series = df_static[feat].dropna()
        if len(series) > 0:
            axes[idx].hist(series, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel(feat, fontsize=9)
            axes[idx].set_ylabel('Frequency', fontsize=9)
            axes[idx].set_title(f'{feat}\n(n={len(series):,})', fontsize=10, fontweight='bold')
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / '4_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 4_feature_distributions.png")
    
    for feat in available_features:
        print(f"\n  {feat}:")
        series = df_static[feat].dropna()
        
        if len(series) > 0:
            print(f"    Count: {len(series):,} (non-null)")
            print(f"    Mean: {series.mean():.2f}")
            print(f"    Median: {series.median():.2f}")
            print(f"    Std: {series.std():.2f}")
            print(f"    Min: {series.min():.2f}")
            print(f"    Max: {series.max():.2f}")
            print(f"    25th percentile: {series.quantile(0.25):.2f}")
            print(f"    75th percentile: {series.quantile(0.75):.2f}")
            
            # Check for outliers
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()
            print(f"    Outliers (1.5×IQR): {outliers:,} ({outliers/len(series)*100:.2f}%)")
                
            # Store for report
            report_lines.append(f"\n  {feat}:")
            report_lines.append(f"    Count: {len(series):,}")
            report_lines.append(f"    Mean: {series.mean():.2f}, Median: {series.median():.2f}")
            report_lines.append(f"    Range: {series.min():.2f} to {series.max():.2f}")
            
            # Check for negative values (data quality)
            if (series < 0).any():
                neg_count = (series < 0).sum()
                print(f"    WARNING: {neg_count:,} negative values detected")
            
            # Check for zeros
            zero_count = (series == 0).sum()
            if zero_count > 0:
                print(f"    Zero values: {zero_count:,} ({zero_count/len(series)*100:.2f}%)")
        else:
            print(f"    All values are missing")
else:
    print("\n  None of the key Phase 2 features found in this file")

# ----------------------------------------------------------------------------
# CORRELATIONS WITH TARGET (REQUIRES MERGE)
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("CORRELATIONS WITH TARGET")
print("-"*80)

print("\n[Merging train_static with train_base for correlation analysis...]")
if 'target' in df_base.columns:
    merged = df_base[['case_id', 'target']].merge(df_static, on='case_id', how='inner')
    print(f"  Merged dataset: {len(merged):,} rows")
    
    # Calculate correlations for numeric columns only
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['case_id', 'target']]
    
    if numeric_cols:
        print(f"\n[Computing correlations for {len(numeric_cols)} numeric features...]")
        correlations = merged[numeric_cols + ['target']].corr()['target'].drop('target').sort_values(ascending=False)
        
        print("\n  Top 20 Positive Correlations with Target (Higher = More Default Risk):")
        print(correlations.head(20).to_string())
        
        print("\n  Top 20 Negative Correlations with Target (Lower = Less Default Risk):")
        print(correlations.tail(20).to_string())
        
        report_lines.append("\nTop 10 Positive Correlations with Target:")
        for feat, corr in correlations.head(10).items():
            report_lines.append(f"  {feat}: {corr:.4f}")
        
        report_lines.append("\nTop 10 Negative Correlations with Target:")
        for feat, corr in correlations.tail(10).items():
            report_lines.append(f"  {feat}: {corr:.4f}")
        
        # PLOT 5: Top Correlations
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top positive
        top_pos = correlations.head(20)
        axes[0].barh(range(len(top_pos)), top_pos.values, color='#e74c3c', alpha=0.7)
        axes[0].set_yticks(range(len(top_pos)))
        axes[0].set_yticklabels(top_pos.index, fontsize=9)
        axes[0].set_xlabel('Correlation with Target', fontsize=12)
        axes[0].set_title('Top 20 Positive Correlations (High Risk)', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Top negative
        top_neg = correlations.tail(20).sort_values()
        axes[1].barh(range(len(top_neg)), top_neg.values, color='#2ecc71', alpha=0.7)
        axes[1].set_yticks(range(len(top_neg)))
        axes[1].set_yticklabels(top_neg.index, fontsize=9)
        axes[1].set_xlabel('Correlation with Target', fontsize=12)
        axes[1].set_title('Top 20 Negative Correlations (Low Risk)', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / '5_target_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 5_target_correlations.png")
        
        # Flag highly correlated features (multicollinearity check)
        print("\n[Checking for Multicollinearity - Top Correlated Feature Pairs]")
        corr_matrix = merged[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(column, row, corr_matrix[column][row]) 
                           for column in upper_tri.columns 
                           for row in upper_tri.index 
                           if corr_matrix[column][row] > 0.9]
        
        if high_corr_pairs:
            print(f"\n  Found {len(high_corr_pairs)} feature pairs with correlation >0.9:")
            for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:20], 1):
                print(f"    {i}. {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print("\n  ✓ No highly correlated feature pairs (>0.9) detected")
    else:
        print("\n  No numeric features available for correlation")
else:
    print("\n  Cannot compute correlations: target variable not found")

# ----------------------------------------------------------------------------
# DATA TYPE SUMMARY
# ----------------------------------------------------------------------------
print("\n" + "-"*80)
print("DATA TYPE SUMMARY - TRAIN_STATIC_0_0")
print("-"*80)

dtype_counts = df_static.dtypes.value_counts()
print("\n  Column type distribution:")
for dtype, count in dtype_counts.items():
    print(f"    {dtype}: {count} columns")

numeric_cols_static = df_static.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_static.select_dtypes(include=['object']).columns.tolist()

print(f"\n  Numeric columns: {len(numeric_cols_static)}")
print(f"  Categorical columns: {len(categorical_cols)}")

if categorical_cols:
    print("\n  Sample categorical columns (first 10):")
    for col in categorical_cols[:10]:
        n_unique = df_static[col].nunique()
        print(f"    {col}: {n_unique} unique values")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("TIER 1 EDA SUMMARY")
print("="*80)

report_lines.append("\n\n" + "="*80)
report_lines.append("TIER 1 EDA SUMMARY")
report_lines.append("="*80)

print("\n[train_base.csv]")
report_lines.append("\n[train_base.csv]")
if 'target' in df_base.columns:
    default_rate = df_base['target'].mean() * 100
    print(f"  ✓ Target variable identified: {default_rate:.2f}% default rate")
    print(f"  ✓ Temporal coverage: {df_base['WEEK_NUM'].nunique()} weeks" if 'WEEK_NUM' in df_base.columns else "")
    report_lines.append(f"  ✓ Target variable identified: {default_rate:.2f}% default rate")
    if 'WEEK_NUM' in df_base.columns:
        report_lines.append(f"  ✓ Temporal coverage: {df_base['WEEK_NUM'].nunique()} weeks")
print(f"  ✓ {df_base['case_id'].nunique():,} unique cases")
report_lines.append(f"  ✓ {df_base['case_id'].nunique():,} unique cases")

print("\n[train_static_0_0.csv]")
report_lines.append("\n[train_static_0_0.csv]")
print(f"  ✓ {len(df_static.columns)} total features")
print(f"  ✓ {len(usable_cols)} features with <50% missing (Phase 2 ready)")
report_lines.append(f"  ✓ {len(df_static.columns)} total features")
report_lines.append(f"  ✓ {len(usable_cols)} features with <50% missing (Phase 2 ready)")
if available_features:
    print(f"  ✓ Key features analyzed: {', '.join(available_features[:3])}...")
    report_lines.append(f"  ✓ Key features analyzed: {', '.join(available_features[:3])}...")

print("\n[Outputs Generated]")
print(f"  1. 1_target_distribution.png")
print(f"  2. 2_temporal_trends.png")
print(f"  3. 3_missing_values.png")
print(f"  4. 4_feature_distributions.png")
print(f"  5. 5_target_correlations.png")
print(f"  6. tier1_eda_report.txt")

report_lines.append("\n[Next Steps]")
report_lines.append("  1. Proceed to train_person_1.csv analysis (Tier 1.3)")
report_lines.append("  2. Build Phase 1 baseline model with 3-5 features")
report_lines.append("  3. Aggregate credit bureau tables for Phase 2")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)

report_lines.append("\n" + "="*80)
report_lines.append("EDA COMPLETE")
report_lines.append("="*80)

# Save the report
report_file = OUTPUT_PATH / 'tier1_eda_report.txt'
with open(report_file, 'w') as f:
    f.write('\n'.join(report_lines))

print(f"\n✓ Summary report saved to: {report_file}")
print(f"✓ All plots saved to: {OUTPUT_PATH}")
print(f"\nFiles created:")
for file in sorted(OUTPUT_PATH.glob('*.png')):
    print(f"  - {file.name}")
print(f"  - tier1_eda_report.txt")
