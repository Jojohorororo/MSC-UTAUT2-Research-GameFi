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
    print("CRONBACH'S ALPHA ANALYSIS - ECONOMIC MOTIVATION (EM)")
    print("GameFi-Specific Extensions")
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

    # Try different possible file names for Economic Motivation
    possible_files = [
        'economicMotivation EM GameFi Specific Extensions.xlsx',
        'economic motivation EM GameFi Specific Extensions.xlsx',
        'economicMotivation EM.xlsx',
        'economic motivation EM.xlsx',
        'economicMotivation (EM) GameFi Specific Extensions.xlsx',
        'Economic Motivation EM GameFi Specific Extensions.xlsx',
        'EM.xlsx',
        'economicMotivation EM GameFi Specific Extensions.xls'
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
            print(f"\nTry renaming your file to: 'economicMotivation EM GameFi Specific Extensions.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract EM columns
    em_columns = ['EM1', 'EM2', 'EM3']

    # Check if all EM columns exist
    missing_cols = [col for col in em_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    em_data = df[em_columns].copy()

    print(f"✓ Found {len(em_columns)} Economic Motivation items")

    # Display item descriptions
    print("\nEconomic Motivation Items (GameFi-Specific Extensions):")
    print("EM1: Earning cryptocurrency or NFTs is my primary motivation for using GameFi platforms")
    print("EM2: I would stop using a GameFi platform if the financial rewards significantly decreased")
    print("EM3: The financial reward aspect is more important to me than the gameplay in GameFi platforms")

    print("\nOriginal response distribution:")
    for col in em_columns:
        print(f"{col}: {em_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in em_columns:
        em_data[col] = em_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = em_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        em_data = em_data.dropna()

    print(f"✓ Final dataset: {len(em_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 65)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 65)
    print(em_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 65)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 65)

    alpha = calculate_cronbach_alpha(em_data)
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
    print(f"\nNumber of Items: {len(em_columns)}")
    print(f"Number of Cases: {len(em_data)}")

    item_total_corr = calculate_item_total_correlations(em_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(em_data)

    for item in em_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 65)
    print("ITEM ANALYSIS")
    print("=" * 65)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in em_columns:
        mean_val = em_data[col].mean()
        std_val = em_data[col].std()
        var_val = em_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Economic Motivation specific interpretation
    print(f"\n" + "=" * 65)
    print("ECONOMIC MOTIVATION SCALE EVALUATION")
    print("=" * 65)

    # Calculate mean scores for interpretation
    overall_mean = em_data.mean().mean()
    print(f"Overall Economic Motivation Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        em_level = "Strong economic motivation (Primarily financially-driven GameFi usage)"
    elif overall_mean >= 3.5:
        em_level = "Moderate-high economic motivation (Significant financial focus)"
    elif overall_mean >= 3.0:
        em_level = "Moderate economic motivation (Balanced financial-gaming motivation)"
    else:
        em_level = "Weak economic motivation (Gaming-focused over financial rewards)"

    print(f"Interpretation: {em_level}")

    # Check for differences between economic motivation dimensions
    print(f"\nEconomic Motivation Breakdown:")
    print(f"EM1 (Primary crypto/NFT motivation): {em_data['EM1'].mean():.3f}")
    print(f"EM2 (Financial reward dependency): {em_data['EM2'].mean():.3f}")
    print(f"EM3 (Financial > gameplay priority): {em_data['EM3'].mean():.3f}")

    # Economic motivation analysis insights
    print(f"\nEconomic Motivation Analysis:")

    # Identify strongest and weakest economic motivation aspects
    em_means = {
        'EM1': em_data['EM1'].mean(),
        'EM2': em_data['EM2'].mean(),
        'EM3': em_data['EM3'].mean()
    }

    strongest_em = max(em_means, key=em_means.get)
    weakest_em = min(em_means, key=em_means.get)

    print(f"Strongest Economic Aspect: {strongest_em} (M = {em_means[strongest_em]:.3f})")
    print(f"Weakest Economic Aspect: {weakest_em} (M = {em_means[weakest_em]:.3f})")

    # Economic motivation dimensions analysis
    primary_motivation = em_data['EM1'].mean()  # Primary crypto/NFT earning motivation
    reward_dependency = em_data['EM2'].mean()  # Financial reward dependency
    financial_priority = em_data['EM3'].mean()  # Financial > gameplay priority

    print(f"\nEconomic Dimension Analysis:")
    print(f"Primary Economic Motivation: {primary_motivation:.3f}")
    print(f"Financial Reward Dependency: {reward_dependency:.3f}")
    print(f"Financial Priority over Gameplay: {financial_priority:.3f}")

    # Economic motivation insights
    print(f"\nEconomic Motivation Profile:")
    if primary_motivation > financial_priority and primary_motivation > reward_dependency:
        motivation_insight = "Primary economic motivation but balanced approach (not purely financial)"
    elif reward_dependency > financial_priority:
        motivation_insight = "High financial dependency but gameplay still valued"
    elif financial_priority > primary_motivation:
        motivation_insight = "Strong financial prioritization over gaming experience"
    else:
        motivation_insight = "Balanced economic motivation across all dimensions"

    print(f"Key Insight: {motivation_insight}")

    # Economic vs gaming balance analysis
    print(f"\nExtrinsic vs Intrinsic Motivation Analysis:")
    if overall_mean >= 4.0:
        motivation_balance = "Highly extrinsic (primarily financially motivated)"
    elif overall_mean >= 3.5:
        motivation_balance = "Moderately extrinsic (significant financial focus)"
    elif overall_mean >= 3.0:
        motivation_balance = "Balanced extrinsic-intrinsic (mixed financial-gaming motivation)"
    else:
        motivation_balance = "Primarily intrinsic (gaming-focused over financial)"

    print(f"Motivation Balance: {motivation_balance}")

    # Financial dependency assessment
    print(f"\nFinancial Dependency Assessment:")
    if reward_dependency >= 4.0:
        dependency_level = "High dependency (would abandon platforms with reduced rewards)"
    elif reward_dependency >= 3.5:
        dependency_level = "Moderate dependency (sensitive to reward changes)"
    elif reward_dependency >= 3.0:
        dependency_level = "Some dependency (influenced by financial rewards)"
    else:
        dependency_level = "Low dependency (not heavily influenced by financial changes)"

    print(f"Reward Dependency Level: {dependency_level}")

    # Economic motivation segmentation
    print(f"\nEconomic Motivation Segmentation:")
    high_economic = len(em_data[em_data.mean(axis=1) >= 4.0])
    moderate_economic = len(em_data[(em_data.mean(axis=1) >= 3.0) & (em_data.mean(axis=1) < 4.0)])
    low_economic = len(em_data[em_data.mean(axis=1) < 3.0])

    print(f"High Economic Motivation (≥4.0): {high_economic} ({(high_economic / len(em_data) * 100):.1f}%)")
    print(
        f"Moderate Economic Motivation (3.0-3.9): {moderate_economic} ({(moderate_economic / len(em_data) * 100):.1f}%)")
    print(f"Low Economic Motivation (<3.0): {low_economic} ({(low_economic / len(em_data) * 100):.1f}%)")

    # GameFi sustainability implications
    print(f"\nGameFi Sustainability Analysis:")
    high_economic_percentage = (high_economic / len(em_data)) * 100

    if high_economic_percentage >= 50:
        sustainability_insight = "High economic dependency - platform sustainability tied to financial rewards"
    elif high_economic_percentage >= 30:
        sustainability_insight = "Moderate economic dependency - mixed sustainability factors"
    else:
        sustainability_insight = "Low economic dependency - platform sustainability less tied to financial rewards"

    print(f"Platform Sustainability: {sustainability_insight}")

    # Economic motivation vs adoption relationship
    print(f"\nEconomic Adoption Implications:")
    if overall_mean >= 3.5:
        adoption_implication = "Strong economic motivations support adoption IF financial rewards are attractive"
    elif overall_mean >= 3.0:
        adoption_implication = "Moderate economic focus - adoption depends on balanced value proposition"
    else:
        adoption_implication = "Weak economic focus - adoption more dependent on gaming quality than rewards"

    print(f"Adoption Driver: {adoption_implication}")

    # Crypto/NFT specific analysis
    print(f"\nCrypto/NFT Earning Analysis:")
    if primary_motivation >= 4.0:
        crypto_focus = "Strong crypto/NFT earning focus - primary motivation for GameFi engagement"
    elif primary_motivation >= 3.5:
        crypto_focus = "Moderate crypto/NFT earning focus - significant but not exclusive motivation"
    elif primary_motivation >= 3.0:
        crypto_focus = "Some crypto/NFT earning interest - balanced with other motivations"
    else:
        crypto_focus = "Limited crypto/NFT earning focus - secondary to gaming experience"

    print(f"Crypto/NFT Focus: {crypto_focus}")

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
    print("\nEconomic Motivation Scale Interpretation:")
    print("• Mean ≥ 4.0 : Strong economic motivation (primarily financial focus)")
    print("• Mean 3.5-3.9 : Moderate-high economic motivation (significant financial focus)")
    print("• Mean 3.0-3.4 : Moderate economic motivation (balanced financial-gaming)")
    print("• Mean < 3.0 : Weak economic motivation (gaming-focused)")
    print("\nEM Components:")
    print("• EM1: Primary crypto/NFT motivation (earning as main driver)")
    print("• EM2: Financial reward dependency (sensitivity to reward changes)")
    print("• EM3: Financial priority over gameplay (extrinsic vs intrinsic balance)")
    print("\nGameFi Economic Research Implications:")
    print("• High EM scores indicate extrinsically motivated users")
    print("• EM2 shows platform loyalty sensitivity to reward structures")
    print("• EM3 reveals user value prioritization (financial vs gaming)")
    print("• Strong economic motivation requires sustainable reward systems")
    print("• Low economic motivation suggests intrinsic gaming appeal")
    print("• Economic segmentation helps target different user motivations")
    print("• High economic dependency may predict churn if rewards decrease")


# Run the analysis
if __name__ == "__main__":
    main()