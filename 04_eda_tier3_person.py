"""
Tier 1.3 EDA: train_person_1.csv Analysis
BAIT 509 Final Project - Home Credit Risk Model

Analyzes: Demographics, Income, and Employment (Conditions & Capacity)
Author: Mitchell Stevens
Date: January 17, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths
DATA_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/home-credit-credit-risk-model-stability/csv_files/train")
OUTPUT_PATH = Path("/Users/mitchellstevens/Desktop/Projects/Credit Risk ML Model/eda_output")
OUTPUT_PATH.mkdir(exist_ok=True)

report_lines = []

def log(text):
    print(text)
    report_lines.append(text)

log("="*80)
log("TIER 1.3 EDA - train_person_1.csv")
log("="*80)

# 1. Load Data
try:
    # Read first 500k rows to avoid memory issues and ensure speed
    df = pd.read_csv(DATA_PATH / "train_person_1.csv", low_memory=False)
    log(f"\nLoaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
except Exception as e:
    log(f"Error loading  {e}")
    exit()

# 2. Cardinality Analysis
if 'case_id' in df.columns:
    ppc = df.groupby('case_id').size()
    log(f"\nCardinality Analysis:")
    log(f"  Unique cases: {df['case_id'].nunique():,}")
    log(f"  Avg persons per case: {ppc.mean():.2f}")
    
    # Visualization 11: Cardinality
    plt.figure(figsize=(10, 6))
    ppc.value_counts().sort_index().head(10).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Persons per Application (Top 10)', fontsize=14)
    plt.xlabel('Number of Persons')
    plt.ylabel('Count of Cases')
    plt.savefig(OUTPUT_PATH / '11_person_cardinality.png')
    plt.close()

# 3. Demographic Analysis (Conditions)
log("\nDemographic Analysis (Conditions):")

# Age calculation (assuming D suffix means date-like or day count)
if 'birth_259D' in df.columns:
    # In this dataset, some D columns are date strings, others are offsets. 
    # Attempt parsing as date first.
    df['birth_dt'] = pd.to_datetime(df['birth_259D'], errors='coerce')
    
    # If parsing failed (all NaT), check if it's numeric offsets (common in some versions)
    if df['birth_dt'].isnull().all():
        log("  Note: birth_259D appears numeric, treating as day offset from reference.")
        # Home Credit typically uses negative days from application. 
        # Using a proxy age calculation if numeric.
        df['age'] = np.abs(pd.to_numeric(df['birth_259D'], errors='coerce')) / 365.25
    else:
        # Reference date for static age calculation
        df['age'] = (pd.to_datetime('2019-07-01') - df['birth_dt']).dt.days / 365.25
        
    df_age = df[(df['age'] >= 18) & (df['age'] <= 100)]
    if not df_age.empty:
        log(f"  Mean age: {df_age['age'].mean():.1f}")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_age['age'], bins=30, kde=True, color='purple')
        plt.title('Age Distribution of Applicants', fontsize=14)
        plt.savefig(OUTPUT_PATH / '12_age_distribution.png')
        plt.close()

# Categoricals
cats = {'education_927M': 'Education', 'familystate_447L': 'Family Status', 
        'housingtype_772L': 'Housing', 'gender_992L': 'Gender'}
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (col, label) in enumerate(cats.items()):
    if col in df.columns:
        counts = df[col].value_counts().head(8)
        if not counts.empty:
            counts.plot(kind='barh', ax=axes[i], color='teal', alpha=0.7)
            axes[i].set_title(label)
            log(f"  {label} top value: {counts.index[0]} ({counts.iloc[0]/len(df)*100:.1f}%)")

plt.tight_layout()
plt.savefig(OUTPUT_PATH / '13_demographics.png')
plt.close()

# 4. Employment & Income (Capacity)
log("\nEmployment & Income Analysis (Capacity):")

# FIX: Handling categorical 'MORE_FIVE' in employment tenure
if 'empl_employedtotal_800L' in df.columns:
    log("  Analyzing Employment Tenure (empl_employedtotal_800L):")
    # Check data type
    col_type = df['empl_employedtotal_800L'].dtype
    
    if col_type == 'object':
        log("  Note: Column is categorical (contains strings like 'MORE_FIVE')")
        emp_counts = df['empl_employedtotal_800L'].value_counts()
        log(f"  Top categories:\n{emp_counts.head(5).to_string()}")
        
        plt.figure(figsize=(10, 6))
        emp_counts.head(10).plot(kind='bar', color='orange', edgecolor='black')
        plt.title('Employment Tenure Categories', fontsize=14)
        plt.xticks(rotation=45)
        plt.savefig(OUTPUT_PATH / '14_employment_tenure.png')
        plt.close()
    else:
        # Numeric case
        tenure = df['empl_employedtotal_800L'].dropna()
        log(f"  Median employment tenure: {tenure.median():.1f} years")
        plt.figure(figsize=(10, 6))
        sns.histplot(tenure, bins=30, color='orange')
        plt.title('Employment Tenure (Years)')
        plt.savefig(OUTPUT_PATH / '14_employment_tenure.png')
        plt.close()

# 5. Missing Values
missing = df.isnull().mean() * 100
log("\nTop Missing Columns:")
log(missing.sort_values(ascending=False).head(10).to_string())

# Save Report
with open(OUTPUT_PATH / "tier1_person_report.txt", "w") as f:
    f.write("\n".join(report_lines))

log("\n" + "="*80)
log("EDA COMPLETE - Outputs saved to eda_output/")
log("="*80)
