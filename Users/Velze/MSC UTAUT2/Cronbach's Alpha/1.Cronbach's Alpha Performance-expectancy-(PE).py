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
    print("CRONBACH'S ALPHA ANALYSIS - PERFORMANCE EXPECTANCY (PE)")
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

    # Try different possible file names for Performance Expectancy
    possible_files = [
        'performance expect PE.xlsx',
        'performance expectancy PE.xlsx',
        'performance-expect-PE.xlsx',
        'performance expect (PE).xlsx',
        'Performance Expect PE.xlsx',
        'PE.xlsx',
        'performance expect PE.xls'
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
            print(f"\nTry renaming your file to: 'performance expect PE.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract PE columns
    pe_columns = ['PE1', 'PE2', 'PE3', 'PE4', 'PE5']

    # Check if all PE columns exist
    missing_cols = [col for col in pe_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    pe_data = df[pe_columns].copy()

    print(f"✓ Found {len(pe_columns)} Performance Expectancy items")

    # Display item descriptions
    print("\nPerformance Expectancy Items:")
    print("PE1: Using GameFi platforms improves my ability to earn financial rewards through gaming")
    print("PE2: GameFi platforms enhance my opportunities to own and trade digital assets (NFTs, tokens)")
    print("PE3: Play-to-earn models in GameFi provide greater financial benefits compared to traditional gaming")
    print("PE4: Using GameFi gives me greater ownership and control over my in-game assets")
    print("PE5: Overall, I find GameFi platforms useful for achieving both gaming enjoyment and financial benefits")

    print("\nOriginal response distribution:")
    for col in pe_columns:
        print(f"{col}: {pe_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in pe_columns:
        pe_data[col] = pe_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = pe_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        pe_data = pe_data.dropna()

    print(f"✓ Final dataset: {len(pe_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(pe_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 60)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 60)

    alpha = calculate_cronbach_alpha(pe_data)
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
    print(f"\nNumber of Items: {len(pe_columns)}")
    print(f"Number of Cases: {len(pe_data)}")

    item_total_corr = calculate_item_total_correlations(pe_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(pe_data)

    for item in pe_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 60)
    print("ITEM ANALYSIS")
    print("=" * 60)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in pe_columns:
        mean_val = pe_data[col].mean()
        std_val = pe_data[col].std()
        var_val = pe_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Performance Expectancy specific interpretation
    print(f"\n" + "=" * 60)
    print("PERFORMANCE EXPECTANCY SCALE EVALUATION")
    print("=" * 60)

    # Calculate mean scores for interpretation
    overall_mean = pe_data.mean().mean()
    print(f"Overall Performance Expectancy Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        pe_level = "High performance expectancy (Strong perceived utility of GameFi)"
    elif overall_mean >= 3.0:
        pe_level = "Moderate performance expectancy (Neutral to positive utility perception)"
    else:
        pe_level = "Low performance expectancy (Poor perceived utility of GameFi)"

    print(f"Interpretation: {pe_level}")

    # Check for differences between performance expectancy dimensions
    print(f"\nPerformance Expectancy Breakdown:")
    print(f"PE1 (Financial rewards earning): {pe_data['PE1'].mean():.3f}")
    print(f"PE2 (Digital asset ownership/trading): {pe_data['PE2'].mean():.3f}")
    print(f"PE3 (Play-to-earn benefits): {pe_data['PE3'].mean():.3f}")
    print(f"PE4 (Asset ownership/control): {pe_data['PE4'].mean():.3f}")
    print(f"PE5 (Overall dual utility): {pe_data['PE5'].mean():.3f}")

    # Performance analysis insights
    print(f"\nPerformance Perception Analysis:")

    # Identify strongest and weakest performance expectancy aspects
    pe_means = {
        'PE1': pe_data['PE1'].mean(),
        'PE2': pe_data['PE2'].mean(),
        'PE3': pe_data['PE3'].mean(),
        'PE4': pe_data['PE4'].mean(),
        'PE5': pe_data['PE5'].mean()
    }

    strongest_pe = max(pe_means, key=pe_means.get)
    weakest_pe = min(pe_means, key=pe_means.get)

    print(f"Strongest Performance Aspect: {strongest_pe} (M = {pe_means[strongest_pe]:.3f})")
    print(f"Weakest Performance Aspect: {weakest_pe} (M = {pe_means[weakest_pe]:.3f})")

    # Performance categories analysis
    financial_earning = pe_data['PE1'].mean()  # Financial rewards
    asset_ownership = (pe_data['PE2'].mean() + pe_data['PE4'].mean()) / 2  # Asset-related
    comparative_benefit = pe_data['PE3'].mean()  # vs traditional gaming
    overall_utility = pe_data['PE5'].mean()  # Dual purpose utility

    print(f"\nPerformance Dimension Analysis:")
    print(f"Financial Earning Utility: {financial_earning:.3f}")
    print(f"Asset Ownership Utility: {asset_ownership:.3f}")
    print(f"Comparative Gaming Benefit: {comparative_benefit:.3f}")
    print(f"Overall Dual Utility: {overall_utility:.3f}")

    # GameFi utility insights
    print(f"\nGameFi Utility Profile:")
    if overall_utility > financial_earning and overall_utility > asset_ownership:
        utility_insight = "Balanced dual-purpose utility (gaming + financial) drives GameFi appeal"
    elif financial_earning > asset_ownership:
        utility_insight = "Financial earning potential is primary utility driver"
    elif asset_ownership > financial_earning:
        utility_insight = "Digital asset ownership/control is primary utility driver"
    else:
        utility_insight = "Balanced utility perception across financial and ownership dimensions"

    print(f"Key Insight: {utility_insight}")

    # Compare GameFi vs traditional gaming
    print(f"\nGameFi vs Traditional Gaming Analysis:")
    if comparative_benefit >= 4.0:
        gaming_comparison = "Strong perceived superiority over traditional gaming"
    elif comparative_benefit >= 3.5:
        gaming_comparison = "Moderate advantage over traditional gaming"
    elif comparative_benefit >= 3.0:
        gaming_comparison = "Slight advantage over traditional gaming"
    else:
        gaming_comparison = "Questionable advantage over traditional gaming"

    print(f"Comparative Performance: {gaming_comparison}")

    # Utility barrier analysis
    print(f"\nUtility Adoption Analysis:")
    if overall_mean >= 3.5:
        utility_barrier = "LOW - Strong perceived utility supports adoption"
    elif overall_mean >= 3.0:
        utility_barrier = "MODERATE - Mixed utility perceptions"
    else:
        utility_barrier = "HIGH - Poor utility perception may hinder adoption"

    print(f"Utility Barrier Level: {utility_barrier}")

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
    print("\nPerformance Expectancy Scale Interpretation:")
    print("• Mean ≥ 4.0 : High utility (strong perceived usefulness)")
    print("• Mean 3.0-3.9 : Moderate utility (neutral to positive usefulness)")
    print("• Mean < 3.0 : Low utility (poor perceived usefulness)")
    print("\nPE Components:")
    print("• PE1: Financial earning ability (play-to-earn effectiveness)")
    print("• PE2: Digital asset opportunities (NFT/token ownership)")
    print("• PE3: Comparative gaming benefits (vs traditional games)")
    print("• PE4: Asset ownership/control (digital property rights)")
    print("• PE5: Dual utility effectiveness (gaming + financial value)")
    print("\nGameFi Research Implications:")
    print("• Low PE scores indicate perceived utility barriers to adoption")
    print("• High PE scores suggest strong performance expectations support adoption")
    print("• Compare PE components to identify specific utility strengths/weaknesses")
    print("• PE3 shows GameFi positioning relative to traditional gaming")
    print("• PE5 captures the unique dual-purpose value proposition of GameFi")


# Run the analysis
if __name__ == "__main__":
    main()