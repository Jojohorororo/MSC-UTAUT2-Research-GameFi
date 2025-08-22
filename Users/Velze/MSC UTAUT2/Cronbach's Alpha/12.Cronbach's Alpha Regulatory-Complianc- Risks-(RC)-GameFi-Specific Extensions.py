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
        if reduced_data.shape[1] > 1:
            alpha_val = calculate_cronbach_alpha(reduced_data)
        else:
            alpha_val = np.nan  # Cannot calculate alpha with only 1 item
        alpha_deleted[col] = alpha_val

    return alpha_deleted


def main():
    """Main function to perform Cronbach's Alpha analysis"""

    print("=" * 70)
    print("CRONBACH'S ALPHA ANALYSIS - REGULATORY & COMPLIANCE RISKS (RC)")
    print("GameFi-Specific Extensions")
    print("=" * 70)

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

    # Try different possible file names for Regulatory and Compliance Risks
    possible_files = [
        'RegulatoryComplianceRisks RC GameFi-Specific Extensions.xlsx',
        'RegulatoryComplianceRisks RC GameFiSpecific Extensions.xlsx',
        'Regulatory Compliance Risks RC GameFi-Specific Extensions.xlsx',
        'Regulatory Compliance Risks RC GameFiSpecific Extensions.xlsx',
        'regulatoryComplianceRisks RC GameFi-Specific Extensions.xlsx',
        'regulatory compliance risks RC GameFi-Specific Extensions.xlsx',
        'RegulatoryComplianceRisks RC.xlsx',
        'Regulatory Compliance Risks RC.xlsx',
        'RegulatoryComplianceRisks (RC) GameFi-Specific Extensions.xlsx',
        'RC.xlsx',
        'RegulatoryComplianceRisks RC GameFi-Specific Extensions.xls'
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
            print(f"\nTry renaming your file to: 'RegulatoryComplianceRisks RC GameFi-Specific Extensions.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract RC columns
    rc_columns = ['RC1', 'RC2']

    # Check if all RC columns exist
    missing_cols = [col for col in rc_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    rc_data = df[rc_columns].copy()

    print(f"✓ Found {len(rc_columns)} Regulatory and Compliance Risk items")

    # Display item descriptions
    print("\nRegulatory and Compliance Risk Items (GameFi-Specific Extensions):")
    print("RC1: Unclear regulations around cryptocurrency and NFTs make me hesitant to participate in GameFi")
    print("RC2: I am concerned about potential tax and legal implications from GameFi activities")

    print("\nOriginal response distribution:")
    for col in rc_columns:
        print(f"{col}: {rc_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in rc_columns:
        rc_data[col] = rc_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = rc_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        rc_data = rc_data.dropna()

    print(f"✓ Final dataset: {len(rc_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(rc_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 70)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 70)

    alpha = calculate_cronbach_alpha(rc_data)
    print(f"Cronbach's Alpha: {alpha:.4f}")

    # Note about 2-item scales
    print("Note: This is a 2-item scale. Cronbach's Alpha for 2-item scales")
    print("is equivalent to the Spearman-Brown stepped-up reliability.")

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
    print(f"\nNumber of Items: {len(rc_columns)}")
    print(f"Number of Cases: {len(rc_data)}")

    item_total_corr = calculate_item_total_correlations(rc_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(rc_data)

    for item in rc_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        if pd.isna(alpha_del):
            alpha_del_str = "N/A (1 item)"
        else:
            alpha_del_str = f"{alpha_del:.4f}"
        print(f"{item:<6} {corr:>15.4f} {alpha_del_str:>15}")

    # Calculate inter-item correlation for 2-item scale
    inter_item_corr = rc_data['RC1'].corr(rc_data['RC2'])
    print(f"\nInter-item Correlation (RC1-RC2): {inter_item_corr:.4f}")
    print("Note: For 2-item scales, inter-item correlation should be ≥ 0.3")

    # Item-level analysis
    print(f"\n" + "=" * 70)
    print("ITEM ANALYSIS")
    print("=" * 70)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in rc_columns:
        mean_val = rc_data[col].mean()
        std_val = rc_data[col].std()
        var_val = rc_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Regulatory and Compliance Risk specific interpretation
    print(f"\n" + "=" * 70)
    print("REGULATORY & COMPLIANCE RISK SCALE EVALUATION")
    print("=" * 70)

    # Calculate mean scores for interpretation
    overall_mean = rc_data.mean().mean()
    print(f"Overall Regulatory & Compliance Risk Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        rc_level = "High regulatory risk concern (Strong regulatory barriers to adoption)"
    elif overall_mean >= 3.5:
        rc_level = "Moderate-high regulatory risk concern (Significant regulatory concerns)"
    elif overall_mean >= 3.0:
        rc_level = "Moderate regulatory risk concern (Some regulatory awareness)"
    else:
        rc_level = "Low regulatory risk concern (Limited regulatory concerns)"

    print(f"Interpretation: {rc_level}")

    # Check for differences between regulatory risk dimensions
    print(f"\nRegulatory & Compliance Risk Breakdown:")
    print(f"RC1 (Regulatory uncertainty - crypto/NFT rules): {rc_data['RC1'].mean():.3f}")
    print(f"RC2 (Legal/tax compliance concerns): {rc_data['RC2'].mean():.3f}")

    # Regulatory risk analysis insights
    print(f"\nRegulatory & Compliance Risk Analysis:")

    # Identify strongest and weakest regulatory risk aspects
    rc_means = {
        'RC1': rc_data['RC1'].mean(),
        'RC2': rc_data['RC2'].mean()
    }

    strongest_rc = max(rc_means, key=rc_means.get)
    weakest_rc = min(rc_means, key=rc_means.get)

    print(f"Higher Regulatory Concern: {strongest_rc} (M = {rc_means[strongest_rc]:.3f})")
    print(f"Lower Regulatory Concern: {weakest_rc} (M = {rc_means[weakest_rc]:.3f})")

    # Regulatory dimensions analysis
    regulatory_uncertainty = rc_data['RC1'].mean()  # Unclear regulations
    legal_compliance = rc_data['RC2'].mean()  # Tax/legal implications

    print(f"\nRegulatory Dimension Analysis:")
    print(f"Regulatory Uncertainty (Crypto/NFT Rules): {regulatory_uncertainty:.3f}")
    print(f"Legal/Tax Compliance Concerns: {legal_compliance:.3f}")

    # Regulatory profile insights
    print(f"\nRegulatory Risk Profile Analysis:")
    if regulatory_uncertainty > legal_compliance:
        regulatory_insight = "Regulatory uncertainty is greater concern than compliance - Need clearer rules"
    elif legal_compliance > regulatory_uncertainty:
        regulatory_insight = "Legal/tax compliance is greater concern than uncertainty - Need compliance guidance"
    else:
        regulatory_insight = "Balanced regulatory concerns across uncertainty and compliance dimensions"

    print(f"Key Insight: {regulatory_insight}")

    # Regulatory barrier assessment
    print(f"\nRegulatory Barrier Assessment:")
    if overall_mean >= 4.0:
        regulatory_barrier = "HIGH - Strong regulatory concerns may significantly hinder adoption"
    elif overall_mean >= 3.5:
        regulatory_barrier = "MODERATE-HIGH - Regulatory concerns may limit adoption"
    elif overall_mean >= 3.0:
        regulatory_barrier = "MODERATE - Some regulatory awareness but not prohibitive"
    else:
        regulatory_barrier = "LOW - Limited regulatory concerns support adoption"

    print(f"Regulatory Adoption Barrier: {regulatory_barrier}")

    # Regulatory risk segmentation
    print(f"\nRegulatory Risk Segmentation:")
    high_regulatory_risk = len(rc_data[rc_data.mean(axis=1) >= 4.0])
    moderate_regulatory_risk = len(rc_data[(rc_data.mean(axis=1) >= 3.0) & (rc_data.mean(axis=1) < 4.0)])
    low_regulatory_risk = len(rc_data[rc_data.mean(axis=1) < 3.0])

    print(f"High Regulatory Risk (≥4.0): {high_regulatory_risk} ({(high_regulatory_risk / len(rc_data) * 100):.1f}%)")
    print(
        f"Moderate Regulatory Risk (3.0-3.9): {moderate_regulatory_risk} ({(moderate_regulatory_risk / len(rc_data) * 100):.1f}%)")
    print(f"Low Regulatory Risk (<3.0): {low_regulatory_risk} ({(low_regulatory_risk / len(rc_data) * 100):.1f}%)")

    # Regulatory vs Other Risks comparison
    print(f"\nRegulatory vs Other Risk Comparisons:")
    print("Compare with other risk findings:")
    print("- Overall Risk Perception (RP): 4.016 (High general risk awareness)")
    print("- Security Risk (RP2): 4.151 (Highest specific risk concern)")
    print(f"- Regulatory Risk (RC): {overall_mean:.3f}")

    if overall_mean > 4.016:  # Higher than general risk perception
        risk_comparison = "Regulatory risks exceed general risk concerns - Top risk category"
    elif overall_mean > 3.5:
        risk_comparison = "Regulatory risks are significant but not highest risk category"
    else:
        risk_comparison = "Regulatory risks are lower than other risk categories"

    print(f"Risk Hierarchy Position: {risk_comparison}")

    # Regulatory implications for adoption
    print(f"\nRegulatory Adoption Implications:")
    if overall_mean >= 4.0:
        adoption_implication = "High regulatory concerns require policy clarity and compliance support for adoption"
    elif overall_mean >= 3.5:
        adoption_implication = "Moderate regulatory concerns need regulatory guidance and transparency"
    else:
        adoption_implication = "Limited regulatory concerns - current uncertainty acceptable for adoption"

    print(f"Policy Needs: {adoption_implication}")

    # Regulatory strategy analysis
    print(f"\nRegulatory Strategy Analysis:")
    if regulatory_uncertainty >= 4.0:
        regulatory_strategy = "Priority: Advocate for clear crypto/NFT regulatory frameworks"
    elif legal_compliance >= 4.0:
        regulatory_strategy = "Priority: Provide tax and legal compliance guidance to users"
    elif overall_mean >= 3.5:
        regulatory_strategy = "Balanced approach: Both regulatory clarity and compliance support needed"
    else:
        regulatory_strategy = "Current regulatory environment acceptable - monitor for changes"

    print(f"Strategy Priority: {regulatory_strategy}")

    # Regulatory communication needs
    print(f"\nRegulatory Communication Analysis:")
    high_regulatory_percentage = (high_regulatory_risk / len(rc_data)) * 100

    if high_regulatory_percentage >= 50:
        communication_need = "Majority have high regulatory concerns - extensive policy education required"
    elif high_regulatory_percentage >= 30:
        communication_need = "Significant regulatory concerns - targeted regulatory communication needed"
    else:
        communication_need = "Limited regulatory concerns - basic regulatory awareness sufficient"

    print(f"Communication Strategy: {communication_need}")

    # Economic motivation vs regulatory risk analysis
    print(f"\nEconomic Motivation vs Regulatory Risk:")
    print("Note: Compare with Economic Motivation findings:")
    print("- Economic Motivation (EM): 4.129 (Strong economic motivation)")
    print("- 72.9% High Economic Motivation users")
    print(f"- Regulatory Risk (RC): {overall_mean:.3f}")

    if overall_mean >= 3.5:
        eco_reg_insight = "High economic motivation despite regulatory concerns - Economic rewards outweigh regulatory risks"
    else:
        eco_reg_insight = "High economic motivation with manageable regulatory concerns - Favorable risk-reward balance"

    print(f"Economic-Regulatory Balance: {eco_reg_insight}")

    # Interpretation guidelines
    print(f"\n" + "=" * 70)
    print("INTERPRETATION GUIDELINES")
    print("=" * 70)
    print("Cronbach's Alpha Interpretation:")
    print("• α ≥ 0.9  : Excellent reliability")
    print("• α ≥ 0.8  : Good reliability")
    print("• α ≥ 0.7  : Acceptable reliability")
    print("• α ≥ 0.6  : Questionable reliability")
    print("• α < 0.6  : Poor reliability")
    print("\nInter-item Correlation (2-item scales):")
    print("• r ≥ 0.3  : Adequate correlation")
    print("• r < 0.3  : Poor item relationship")
    print("\nRegulatory & Compliance Risk Scale Interpretation:")
    print("• Mean ≥ 4.0 : High regulatory concerns (strong barriers to adoption)")
    print("• Mean 3.5-3.9 : Moderate-high regulatory concerns (significant concerns)")
    print("• Mean 3.0-3.4 : Moderate regulatory concerns (some awareness)")
    print("• Mean < 3.0 : Low regulatory concerns (limited concerns)")
    print("\nRC Components:")
    print("• RC1: Regulatory uncertainty (unclear crypto/NFT regulations)")
    print("• RC2: Legal/tax compliance concerns (potential legal implications)")
    print("\nGameFi Regulatory Research Implications:")
    print("• High RC scores indicate regulatory barriers to mass adoption")
    print("• RC1 shows need for clearer cryptocurrency and NFT regulations")
    print("• RC2 reveals compliance anxiety requiring guidance and support")
    print("• Regulatory vs other risk comparisons show relative barrier importance")
    print("• Economic motivation vs regulatory risk reveals risk-reward calculations")
    print("• High regulatory concerns require policy advocacy and user education")
    print("• Regulatory strategy should address both uncertainty and compliance needs")
    print("• Communication strategies must address regulatory anxiety for adoption")


# Run the analysis
if __name__ == "__main__":
    main()