import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def convert_likert_to_numeric(response):
    """Convert text Likert responses to numeric values"""
    mapping = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neither Agree/Disagree': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }
    return mapping.get(response, np.nan)


def calculate_cronbach_alpha(data):
    """
    Calculate Cronbach's Alpha for reliability analysis

    Parameters:
    data: DataFrame with items as columns and responses as rows

    Returns:
    alpha: Cronbach's Alpha coefficient
    """
    # Convert to numpy array for calculations
    data_array = data.values

    # Number of items
    k = data_array.shape[1]

    # Calculate variance for each item
    item_variances = np.var(data_array, axis=0, ddof=1)

    # Calculate variance of sum scores
    sum_scores = np.sum(data_array, axis=1)
    total_variance = np.var(sum_scores, ddof=1)

    # Cronbach's Alpha formula
    alpha = (k / (k - 1)) * (1 - (np.sum(item_variances) / total_variance))

    return alpha


def calculate_item_total_correlations(data):
    """
    Calculate corrected item-total correlations
    (correlation between each item and sum of other items)

    Parameters:
    data: DataFrame with items as columns

    Returns:
    correlations: Dictionary with item-total correlations
    """
    correlations = {}

    for col in data.columns:
        # Sum of all other items (excluding current item)
        other_items = data.drop(columns=[col])
        other_items_sum = other_items.sum(axis=1)

        # Calculate correlation between current item and sum of others
        corr, _ = pearsonr(data[col], other_items_sum)
        correlations[col] = corr

    return correlations


def alpha_if_deleted(data):
    """
    Calculate Cronbach's Alpha if each item is deleted

    Parameters:
    data: DataFrame with items as columns

    Returns:
    alpha_deleted: Dictionary with alpha values if item deleted
    """
    alpha_deleted = {}

    for col in data.columns:
        # Create dataset without current item
        reduced_data = data.drop(columns=[col])
        # Calculate alpha for reduced dataset
        alpha_val = calculate_cronbach_alpha(reduced_data)
        alpha_deleted[col] = alpha_val

    return alpha_deleted


def main():
    """Main function to perform Cronbach's Alpha analysis"""

    print("=" * 65)
    print("CRONBACH'S ALPHA ANALYSIS - FACILITATING CONDITIONS (FC)")
    print("=" * 65)

    # Load the Excel file - try different possible filenames
    print("Loading data...")

    import os

    # Show current working directory and files
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    print("Files in current directory:")
    for file in os.listdir(current_dir):
        if file.endswith(('.xlsx', '.xls')):
            print(f"  📊 {file}")

    # Try different possible file names for Facilitating Conditions
    possible_files = [
        'facilitationconditions FC.xlsx',
        'facilitating conditions FC.xlsx',
        'facilitating-conditions-FC.xlsx',
        'facilitationconditions (FC).xlsx',
        'Facilitating Conditions FC.xlsx',
        'FC.xlsx',
        'facilitationconditions FC.xls'
    ]

    df = None
    filename_used = None

    for filename in possible_files:
        try:
            if os.path.exists(filename):
                df = pd.read_excel(filename)
                filename_used = filename
                print(f"✓ Data loaded successfully from '{filename}': {len(df)} participants")
                break
        except Exception as e:
            print(f"Tried '{filename}': {e}")
            continue

    if df is None:
        print("\n❌ Could not load Excel file. Please ensure:")
        print("1. The Excel file is in the same folder as this Python script")
        print("2. The file name matches exactly (check spelling and spaces)")
        print("3. The file is not open in Excel (close it if it is)")
        print("\nFound Excel files in directory:")
        excel_files = [f for f in os.listdir(current_dir) if f.endswith(('.xlsx', '.xls'))]
        if excel_files:
            for f in excel_files:
                print(f"  - {f}")
            print(f"\nTry renaming your file to: 'facilitationconditions FC.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract FC columns
    fc_columns = ['FC1', 'FC2', 'FC3', 'FC4']

    # Check if all FC columns exist
    missing_cols = [col for col in fc_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    fc_data = df[fc_columns].copy()

    print(f"✓ Found {len(fc_columns)} Facilitating Conditions items")

    # Display item descriptions
    print("\nFacilitating Conditions Items:")
    print("FC1: I have the necessary resources (hardware, internet connection) to use GameFi platforms")
    print("FC2: I have the knowledge necessary to use GameFi platforms (blockchain, cryptocurrency basics)")
    print("FC3: I can get help from others when I have difficulties using GameFi platforms")
    print("FC4: The costs associated with GameFi platforms are reasonable (gas fees, initial investment)")

    print("\nOriginal response distribution:")
    for col in fc_columns:
        print(f"{col}: {fc_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in fc_columns:
        fc_data[col] = fc_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = fc_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        fc_data = fc_data.dropna()

    print(f"✓ Final dataset: {len(fc_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 65)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 65)
    print(fc_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 65)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 65)

    alpha = calculate_cronbach_alpha(fc_data)
    print(f"Cronbach's Alpha: {alpha:.4f}")

    # Interpret alpha value
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

    print(f"Reliability Level: {interpretation}")

    # Calculate corrected item-total correlations
    print(f"\nNumber of Items: {len(fc_columns)}")
    print(f"Number of Cases: {len(fc_data)}")

    item_total_corr = calculate_item_total_correlations(fc_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(fc_data)

    for item in fc_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 65)
    print("ITEM ANALYSIS")
    print("=" * 65)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in fc_columns:
        mean_val = fc_data[col].mean()
        std_val = fc_data[col].std()
        var_val = fc_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Facilitating Conditions specific interpretation
    print(f"\n" + "=" * 65)
    print("FACILITATING CONDITIONS SCALE EVALUATION")
    print("=" * 65)

    # Calculate mean scores for interpretation
    overall_mean = fc_data.mean().mean()
    print(f"Overall Facilitating Conditions Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        fc_level = "Strong facilitating conditions for GameFi adoption"
    elif overall_mean >= 3.0:
        fc_level = "Moderate facilitating conditions for GameFi adoption"
    else:
        fc_level = "Weak facilitating conditions for GameFi adoption"

    print(f"Interpretation: {fc_level}")

    # Check for differences between facilitating condition types
    print(f"\nFacilitating Conditions Breakdown:")
    print(f"FC1 (Resources - hardware/internet): {fc_data['FC1'].mean():.3f}")
    print(f"FC2 (Knowledge - blockchain/crypto): {fc_data['FC2'].mean():.3f}")
    print(f"FC3 (Support - help from others): {fc_data['FC3'].mean():.3f}")
    print(f"FC4 (Affordability - reasonable costs): {fc_data['FC4'].mean():.3f}")

    # Identify strengths and barriers
    fc_means = {
        'FC1': fc_data['FC1'].mean(),
        'FC2': fc_data['FC2'].mean(),
        'FC3': fc_data['FC3'].mean(),
        'FC4': fc_data['FC4'].mean()
    }

    strongest_fc = max(fc_means, key=fc_means.get)
    weakest_fc = min(fc_means, key=fc_means.get)

    print(f"\nStrongest Facilitating Condition: {strongest_fc} (M = {fc_means[strongest_fc]:.3f})")
    print(f"Weakest Facilitating Condition: {weakest_fc} (M = {fc_means[weakest_fc]:.3f})")

    # Interpretation guidelines
    print(f"\n" + "=" * 65)
    print("INTERPRETATION GUIDELINES")
    print("=" * 65)
    print("Cronbach's Alpha Interpretation:")
    print("• α ≥ 0.9  : Excellent reliability")
    print("• α ≥ 0.8  : Good reliability")
    print("• α ≥ 0.7  : Acceptable reliability")
    print("• α ≥ 0.6  : Questionable reliability")
    print("• α < 0.6  : Poor reliability")
    print("\nCorrected Item-Total Correlations:")
    print("• r ≥ 0.3  : Good item discrimination")
    print("• r < 0.3  : Consider removing item")
    print("\nFacilitating Conditions Scale Interpretation:")
    print("• Mean ≥ 4.0 : Strong facilitating conditions")
    print("• Mean 3.0-3.9 : Moderate facilitating conditions")
    print("• Mean < 3.0 : Weak facilitating conditions")
    print("\nFC Components:")
    print("• FC1: Technical resources (hardware/internet)")
    print("• FC2: Knowledge resources (blockchain/crypto understanding)")
    print("• FC3: Social resources (support from others)")
    print("• FC4: Financial resources (cost considerations)")


# Run the analysis
if __name__ == "__main__":
    main()