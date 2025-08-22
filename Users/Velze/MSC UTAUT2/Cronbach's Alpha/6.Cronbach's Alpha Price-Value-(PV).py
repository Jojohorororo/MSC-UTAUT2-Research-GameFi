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

    print("=" * 55)
    print("CRONBACH'S ALPHA ANALYSIS - PRICE VALUE (PV)")
    print("=" * 55)

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

    # Try different possible file names for Price Value
    possible_files = [
        'pricevalue PV.xlsx',
        'price value PV.xlsx',
        'price-value-PV.xlsx',
        'pricevalue (PV).xlsx',
        'Price Value PV.xlsx',
        'PV.xlsx',
        'pricevalue PV.xls'
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
            print(f"\nTry renaming your file to: 'pricevalue PV.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract PV columns
    pv_columns = ['PV1', 'PV2', 'PV3']

    # Check if all PV columns exist
    missing_cols = [col for col in pv_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return

    pv_data = df[pv_columns].copy()

    print(f"✓ Found {len(pv_columns)} Price Value items")

    # Display item descriptions
    print("\nPrice Value Items:")
    print("PV1: GameFi platforms are reasonably priced considering their benefits")
    print("PV2: The financial investment required (gas fees, NFT purchases, etc.) provides good value for money")
    print("PV3: The potential financial returns from GameFi platforms justify the initial costs of participation")

    print("\nOriginal response distribution:")
    for col in pv_columns:
        print(f"{col}: {pv_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in pv_columns:
        pv_data[col] = pv_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = pv_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        pv_data = pv_data.dropna()

    print(f"✓ Final dataset: {len(pv_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 55)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 55)
    print(pv_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 55)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 55)

    alpha = calculate_cronbach_alpha(pv_data)
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
    print(f"\nNumber of Items: {len(pv_columns)}")
    print(f"Number of Cases: {len(pv_data)}")

    item_total_corr = calculate_item_total_correlations(pv_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(pv_data)

    for item in pv_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 55)
    print("ITEM ANALYSIS")
    print("=" * 55)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in pv_columns:
        mean_val = pv_data[col].mean()
        std_val = pv_data[col].std()
        var_val = pv_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Price Value specific interpretation
    print(f"\n" + "=" * 55)
    print("PRICE VALUE SCALE EVALUATION")
    print("=" * 55)

    # Calculate mean scores for interpretation
    overall_mean = pv_data.mean().mean()
    print(f"Overall Price Value Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        pv_level = "High price value perception (GameFi seen as good value)"
    elif overall_mean >= 3.0:
        pv_level = "Moderate price value perception (neutral to positive value)"
    else:
        pv_level = "Low price value perception (GameFi seen as poor value)"

    print(f"Interpretation: {pv_level}")

    # Check for differences between price value dimensions
    print(f"\nPrice Value Breakdown:")
    print(f"PV1 (Reasonable pricing vs benefits): {pv_data['PV1'].mean():.3f}")
    print(f"PV2 (Investment value for money): {pv_data['PV2'].mean():.3f}")
    print(f"PV3 (ROI justification): {pv_data['PV3'].mean():.3f}")

    # Economic analysis insights
    print(f"\nEconomic Perception Analysis:")

    # Identify strongest and weakest price value aspects
    pv_means = {
        'PV1': pv_data['PV1'].mean(),
        'PV2': pv_data['PV2'].mean(),
        'PV3': pv_data['PV3'].mean()
    }

    strongest_pv = max(pv_means, key=pv_means.get)
    weakest_pv = min(pv_means, key=pv_means.get)

    print(f"Strongest Price Value Aspect: {strongest_pv} (M = {pv_means[strongest_pv]:.3f})")
    print(f"Weakest Price Value Aspect: {weakest_pv} (M = {pv_means[weakest_pv]:.3f})")

    # Economic barriers vs enablers analysis
    print(f"\nEconomic Adoption Analysis:")
    if overall_mean >= 3.5:
        economic_barrier = "LOW - Price value is generally perceived as positive"
    elif overall_mean >= 3.0:
        economic_barrier = "MODERATE - Mixed perceptions of price value"
    else:
        economic_barrier = "HIGH - Price value seen as significant barrier"

    print(f"Economic Barrier Level: {economic_barrier}")

    # Compare different value perceptions
    general_pricing = pv_data['PV1'].mean()  # Reasonable pricing
    investment_value = pv_data['PV2'].mean()  # Value for money
    roi_expectation = pv_data['PV3'].mean()  # Return justification

    print(f"\nValue Perception Insights:")
    if general_pricing > investment_value and general_pricing > roi_expectation:
        insight = "Participants see GameFi as reasonably priced but question investment value/returns"
    elif investment_value > roi_expectation:
        insight = "Good value for money perception, but ROI expectations may be unrealistic"
    elif roi_expectation > investment_value:
        insight = "Strong ROI confidence but concerns about immediate value for money"
    else:
        insight = "Balanced price value perceptions across all dimensions"

    print(f"Key Insight: {insight}")

    # Interpretation guidelines
    print(f"\n" + "=" * 55)
    print("INTERPRETATION GUIDELINES")
    print("=" * 55)
    print("Cronbach's Alpha Interpretation:")
    print("• α ≥ 0.9  : Excellent reliability")
    print("• α ≥ 0.8  : Good reliability")
    print("• α ≥ 0.7  : Acceptable reliability")
    print("• α ≥ 0.6  : Questionable reliability")
    print("• α < 0.6  : Poor reliability")
    print("\nCorrected Item-Total Correlations:")
    print("• r ≥ 0.3  : Good item discrimination")
    print("• r < 0.3  : Consider removing item")
    print("\nPrice Value Scale Interpretation:")
    print("• Mean ≥ 4.0 : High price value (good value perception)")
    print("• Mean 3.0-3.9 : Moderate price value (neutral to positive)")
    print("• Mean < 3.0 : Low price value (poor value perception)")
    print("\nPV Components:")
    print("• PV1: General pricing reasonableness (cost vs benefits)")
    print("• PV2: Investment value (value for money assessment)")
    print("• PV3: ROI expectations (returns justify costs)")
    print("\nEconomic Research Implications:")
    print("• Low PV scores indicate price/cost barriers to adoption")
    print("• High PV scores suggest economic factors support adoption")
    print("• Compare PV components to identify specific economic concerns")


# Run the analysis
if __name__ == "__main__":
    main()