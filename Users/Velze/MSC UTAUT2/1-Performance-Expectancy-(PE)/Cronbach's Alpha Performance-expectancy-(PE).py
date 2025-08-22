import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def calculate_cronbachs_alpha(data):
    """
    Calculate Cronbach's Alpha for a dataset

    Parameters:
    data (DataFrame): DataFrame with items as columns

    Returns:
    float: Cronbach's Alpha value
    """
    # Number of items
    k = data.shape[1]

    # Variance of each item
    item_variances = data.var(axis=0, ddof=1)

    # Total variance (sum of all items)
    total_scores = data.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    # Sum of item variances
    sum_item_variances = item_variances.sum()

    # Cronbach's Alpha formula
    alpha = (k / (k - 1)) * (1 - (sum_item_variances / total_variance))

    return alpha


def calculate_item_total_correlations(data):
    """
    Calculate corrected item-total correlations

    Parameters:
    data (DataFrame): DataFrame with items as columns

    Returns:
    dict: Dictionary with item names as keys and correlations as values
    """
    correlations = {}

    for column in data.columns:
        # Calculate total score excluding the current item (corrected)
        other_items = data.drop(columns=[column])
        other_total = other_items.sum(axis=1)

        # Calculate correlation between item and corrected total
        correlation, _ = pearsonr(data[column], other_total)
        correlations[column] = correlation

    return correlations


# Read the Excel file
print("Reading Performance Expectancy data...")
df = pd.read_excel('performance expect PE.xlsx')

print(f"Dataset loaded successfully!")
print(f"Number of participants: {len(df)}")
print(f"Number of PE items: {len(df.columns)}")
print(f"Items: {list(df.columns)}")
print()

# Convert Likert scale responses to numerical values
likert_mapping = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neither Agree/Disagree': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

print("Converting text responses to numerical values...")
print("Mapping:")
for text, num in likert_mapping.items():
    print(f"  {text} → {num}")
print()

# Apply the mapping to all PE columns
df_numeric = df.copy()
for col in ['PE1', 'PE2', 'PE3', 'PE4', 'PE5']:
    df_numeric[col] = df[col].map(likert_mapping)

# Verify conversion
print("Sample of converted data:")
print(df_numeric.head())
print()

# Check for any unmapped values (should be none if data is clean)
missing_values = df_numeric.isnull().sum().sum()
if missing_values > 0:
    print(f"Warning: {missing_values} values could not be mapped!")
else:
    print("✓ All values successfully converted to numerical format")
print()

# Calculate descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(df_numeric.describe())
print()

# Calculate Cronbach's Alpha
alpha = calculate_cronbachs_alpha(df_numeric)

print("=== RELIABILITY ANALYSIS ===")
print(f"Cronbach's Alpha: {alpha:.4f}")
print()

# Interpret Cronbach's Alpha
if alpha >= 0.9:
    interpretation = "Excellent"
elif alpha >= 0.8:
    interpretation = "Good"
elif alpha >= 0.7:
    interpretation = "Acceptable"
elif alpha >= 0.6:
    interpretation = "Questionable"
else:
    interpretation = "Poor"

print(f"Reliability Interpretation: {interpretation}")
print()

# Calculate item-total correlations
item_correlations = calculate_item_total_correlations(df_numeric)

print("=== CORRECTED ITEM-TOTAL CORRELATIONS ===")
for item, correlation in item_correlations.items():
    print(f"{item}: {correlation:.4f}")
print()

# Provide interpretation for item-total correlations
print("=== ITEM-TOTAL CORRELATION INTERPRETATION ===")
print("Generally acceptable range: 0.30 - 0.70")
print("Items with correlations < 0.30 may be problematic")
print("Items with correlations > 0.70 may be redundant")
print()

for item, correlation in item_correlations.items():
    if correlation < 0.30:
        status = "⚠️  Low (consider review)"
    elif correlation > 0.70:
        status = "📊 High (check for redundancy)"
    else:
        status = "✓ Good"
    print(f"{item}: {correlation:.4f} - {status}")

print()
print("=== SUMMARY ===")
print(f"• Sample size: {len(df)} participants")
print(f"• Number of items: {len(df.columns)}")
print(f"• Cronbach's Alpha: {alpha:.4f} ({interpretation})")
print(f"• Scale range: 1-5 (Likert scale)")
print(f"• Mean scale score: {df_numeric.sum(axis=1).mean():.2f}")
print(f"• Scale score range: {df_numeric.sum(axis=1).min()}-{df_numeric.sum(axis=1).max()}")