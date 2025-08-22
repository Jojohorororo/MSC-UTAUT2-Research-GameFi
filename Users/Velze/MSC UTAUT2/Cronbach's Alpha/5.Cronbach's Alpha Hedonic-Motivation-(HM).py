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

    print("=" * 60)
    print("CRONBACH'S ALPHA ANALYSIS - HEDONIC MOTIVATION (HM)")
    print("=" * 60)

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

    # Try different possible file names for Hedonic Motivation
    possible_files = [
        'hedonisticmotivation HM.xlsx',
        'hedonic motivation HM.xlsx',
        'hedonic-motivation-HM.xlsx',
        'hedonisticmotivation (HM).xlsx',
        'Hedonic Motivation HM.xlsx',
        'HM.xlsx',
        'hedonisticmotivation HM.xls'
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
            print(f"\nTry renaming your file to: 'hedonisticmotivation HM.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract HM columns
    hm_columns = ['HM1', 'HM2', 'HM3', 'HM4']

    # Check if all HM columns exist
    missing_cols = [col for col in hm_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    hm_data = df[hm_columns].copy()

    print(f"✓ Found {len(hm_columns)} Hedonic Motivation items")

    # Display item descriptions
    print("\nHedonic Motivation Items:")
    print("HM1: Using GameFi platforms is enjoyable")
    print("HM2: I find playing GameFi games to be entertaining")
    print("HM3: The gameplay elements in GameFi platforms are fun, independent of potential financial rewards")
    print("HM4: Using GameFi platforms makes gaming more exciting for me")

    print("\nOriginal response distribution:")
    for col in hm_columns:
        print(f"{col}: {hm_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in hm_columns:
        hm_data[col] = hm_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = hm_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        hm_data = hm_data.dropna()

    print(f"✓ Final dataset: {len(hm_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(hm_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 60)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 60)

    alpha = calculate_cronbach_alpha(hm_data)
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
    print(f"\nNumber of Items: {len(hm_columns)}")
    print(f"Number of Cases: {len(hm_data)}")

    item_total_corr = calculate_item_total_correlations(hm_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(hm_data)

    for item in hm_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 60)
    print("ITEM ANALYSIS")
    print("=" * 60)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in hm_columns:
        mean_val = hm_data[col].mean()
        std_val = hm_data[col].std()
        var_val = hm_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Hedonic Motivation specific interpretation
    print(f"\n" + "=" * 60)
    print("HEDONIC MOTIVATION SCALE EVALUATION")
    print("=" * 60)

    # Calculate mean scores for interpretation
    overall_mean = hm_data.mean().mean()
    print(f"Overall Hedonic Motivation Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        hm_level = "High hedonic motivation toward GameFi"
    elif overall_mean >= 3.0:
        hm_level = "Moderate hedonic motivation toward GameFi"
    else:
        hm_level = "Low hedonic motivation toward GameFi"

    print(f"Interpretation: {hm_level}")

    # Check for differences between hedonic motivation aspects
    print(f"\nHedonic Motivation Breakdown:")
    print(f"HM1 (General enjoyment): {hm_data['HM1'].mean():.3f}")
    print(f"HM2 (Entertainment value): {hm_data['HM2'].mean():.3f}")
    print(f"HM3 (Intrinsic fun - independent of rewards): {hm_data['HM3'].mean():.3f}")
    print(f"HM4 (Excitement enhancement): {hm_data['HM4'].mean():.3f}")

    # Special analysis for intrinsic vs extrinsic motivation
    intrinsic_fun = hm_data['HM3'].mean()  # Fun independent of financial rewards
    extrinsic_items = hm_data[['HM1', 'HM2', 'HM4']].mean().mean()  # General enjoyment items

    print(f"\nIntrinsic vs Extrinsic Motivation Analysis:")
    print(f"Intrinsic Fun (HM3 - independent of rewards): {intrinsic_fun:.3f}")
    print(f"General Hedonic Motivation (HM1,HM2,HM4): {extrinsic_items:.3f}")

    if intrinsic_fun > extrinsic_items:
        motivation_type = "Participants are motivated by intrinsic fun more than general enjoyment"
    elif abs(intrinsic_fun - extrinsic_items) < 0.1:
        motivation_type = "Balanced intrinsic and general hedonic motivation"
    else:
        motivation_type = "General enjoyment exceeds intrinsic fun motivation"

    print(f"Interpretation: {motivation_type}")

    # Identify strongest and weakest hedonic aspects
    hm_means = {
        'HM1': hm_data['HM1'].mean(),
        'HM2': hm_data['HM2'].mean(),
        'HM3': hm_data['HM3'].mean(),
        'HM4': hm_data['HM4'].mean()
    }

    strongest_hm = max(hm_means, key=hm_means.get)
    weakest_hm = min(hm_means, key=hm_means.get)

    print(f"\nStrongest Hedonic Aspect: {strongest_hm} (M = {hm_means[strongest_hm]:.3f})")
    print(f"Weakest Hedonic Aspect: {weakest_hm} (M = {hm_means[weakest_hm]:.3f})")

    # Interpretation guidelines
    print(f"\n" + "=" * 60)
    print("INTERPRETATION GUIDELINES")
    print("=" * 60)
    print("Cronbach's Alpha Interpretation:")
    print("• α ≥ 0.9  : Excellent reliability")
    print("• α ≥ 0.8  : Good reliability")
    print("• α ≥ 0.7  : Acceptable reliability")
    print("• α ≥ 0.6  : Questionable reliability")
    print("• α < 0.6  : Poor reliability")
    print("\nCorrected Item-Total Correlations:")
    print("• r ≥ 0.3  : Good item discrimination")
    print("• r < 0.3  : Consider removing item")
    print("\nHedonic Motivation Scale Interpretation:")
    print("• Mean ≥ 4.0 : High hedonic motivation")
    print("• Mean 3.0-3.9 : Moderate hedonic motivation")
    print("• Mean < 3.0 : Low hedonic motivation")
    print("\nHM Components:")
    print("• HM1: General enjoyment of GameFi platforms")
    print("• HM2: Entertainment value of GameFi games")
    print("• HM3: Intrinsic fun (independent of financial rewards)")
    print("• HM4: Excitement enhancement through GameFi")
    print("\nResearch Note:")
    print("• HM3 is particularly valuable for measuring intrinsic vs extrinsic motivation")
    print("• Compare HM3 to other items to understand pure gaming enjoyment")


# Run the analysis
if __name__ == "__main__":
    main()