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
    print("CRONBACH'S ALPHA ANALYSIS - RISK PERCEPTION (RP)")
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

    # Try different possible file names for Risk Perception
    possible_files = [
        'RiskPerception RP Gamefi Specific Extensions.xlsx',
        'Risk Perception RP Gamefi Specific Extensions.xlsx',
        'riskPerception RP Gamefi Specific Extensions.xlsx',
        'risk perception RP Gamefi Specific Extensions.xlsx',
        'RiskPerception RP.xlsx',
        'Risk Perception RP.xlsx',
        'RiskPerception (RP) Gamefi Specific Extensions.xlsx',
        'RP.xlsx',
        'RiskPerception RP Gamefi Specific Extensions.xls'
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
            print(f"\nTry renaming your file to: 'RiskPerception RP Gamefi Specific Extensions.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract RP columns (note: RP4 is named RP3.1 in the Excel file)
    rp_columns = ['RP1', 'RP2', 'RP3', 'RP3.1']

    # Check if all RP columns exist
    missing_cols = [col for col in rp_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    rp_data = df[rp_columns].copy()

    print(f"✓ Found {len(rp_columns)} Risk Perception items")

    # Display item descriptions
    print("\nRisk Perception Items (GameFi-Specific Extensions):")
    print("RP1: I am concerned about the financial risks (volatility, potential losses) associated with GameFi")
    print("RP2: I worry about security vulnerabilities in GameFi platforms (hacks, scams, exploits)")
    print("RP3: I am concerned about the long-term sustainability of GameFi economic models")
    print("RP3.1: The overall uncertainty surrounding GameFi technology and markets is a significant concern for me")

    print("\nOriginal response distribution:")
    for col in rp_columns:
        print(f"{col}: {rp_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in rp_columns:
        rp_data[col] = rp_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = rp_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        rp_data = rp_data.dropna()

    print(f"✓ Final dataset: {len(rp_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 65)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 65)
    print(rp_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 65)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 65)

    alpha = calculate_cronbach_alpha(rp_data)
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
    print(f"\nNumber of Items: {len(rp_columns)}")
    print(f"Number of Cases: {len(rp_data)}")

    item_total_corr = calculate_item_total_correlations(rp_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(rp_data)

    for item in rp_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 65)
    print("ITEM ANALYSIS")
    print("=" * 65)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in rp_columns:
        mean_val = rp_data[col].mean()
        std_val = rp_data[col].std()
        var_val = rp_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Risk Perception specific interpretation
    print(f"\n" + "=" * 65)
    print("RISK PERCEPTION SCALE EVALUATION")
    print("=" * 65)

    # Calculate mean scores for interpretation
    overall_mean = rp_data.mean().mean()
    print(f"Overall Risk Perception Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        rp_level = "High risk perception (Strong concerns about GameFi risks)"
    elif overall_mean >= 3.5:
        rp_level = "Moderate-high risk perception (Significant risk concerns)"
    elif overall_mean >= 3.0:
        rp_level = "Moderate risk perception (Some risk awareness)"
    else:
        rp_level = "Low risk perception (Limited risk concerns)"

    print(f"Interpretation: {rp_level}")

    # Check for differences between risk perception dimensions
    print(f"\nRisk Perception Breakdown:")
    print(f"RP1 (Financial risks - volatility/losses): {rp_data['RP1'].mean():.3f}")
    print(f"RP2 (Security vulnerabilities): {rp_data['RP2'].mean():.3f}")
    print(f"RP3 (Sustainability concerns): {rp_data['RP3'].mean():.3f}")
    print(f"RP3.1 (Technology/market uncertainty): {rp_data['RP3.1'].mean():.3f}")

    # Risk perception analysis insights
    print(f"\nRisk Perception Analysis:")

    # Identify strongest and weakest risk perception aspects
    rp_means = {
        'RP1': rp_data['RP1'].mean(),
        'RP2': rp_data['RP2'].mean(),
        'RP3': rp_data['RP3'].mean(),
        'RP3.1': rp_data['RP3.1'].mean()
    }

    strongest_rp = max(rp_means, key=rp_means.get)
    weakest_rp = min(rp_means, key=rp_means.get)

    print(f"Highest Risk Concern: {strongest_rp} (M = {rp_means[strongest_rp]:.3f})")
    print(f"Lowest Risk Concern: {weakest_rp} (M = {rp_means[weakest_rp]:.3f})")

    # Risk dimensions analysis
    financial_risk = rp_data['RP1'].mean()  # Financial volatility/losses
    security_risk = rp_data['RP2'].mean()  # Security vulnerabilities
    sustainability_risk = rp_data['RP3'].mean()  # Long-term sustainability
    uncertainty_risk = rp_data['RP3.1'].mean()  # Technology/market uncertainty

    print(f"\nRisk Dimension Analysis:")
    print(f"Financial Risk Concerns: {financial_risk:.3f}")
    print(f"Security Risk Concerns: {security_risk:.3f}")
    print(f"Sustainability Risk Concerns: {sustainability_risk:.3f}")
    print(f"Technology/Market Uncertainty: {uncertainty_risk:.3f}")

    # Risk profile insights
    print(f"\nRisk Profile Analysis:")
    if security_risk > financial_risk and security_risk > sustainability_risk:
        risk_insight = "Security vulnerabilities are the primary risk concern"
    elif financial_risk > security_risk and financial_risk > sustainability_risk:
        risk_insight = "Financial volatility/losses are the primary risk concern"
    elif sustainability_risk > financial_risk and sustainability_risk > security_risk:
        risk_insight = "Long-term sustainability is the primary risk concern"
    elif uncertainty_risk > financial_risk and uncertainty_risk > security_risk:
        risk_insight = "Technology/market uncertainty is the primary risk concern"
    else:
        risk_insight = "Balanced risk concerns across all dimensions"

    print(f"Key Insight: {risk_insight}")

    # Risk barrier assessment
    print(f"\nRisk Barrier Assessment:")
    if overall_mean >= 4.0:
        risk_barrier = "HIGH - Strong risk perceptions may significantly hinder adoption"
    elif overall_mean >= 3.5:
        risk_barrier = "MODERATE-HIGH - Risk concerns may limit adoption"
    elif overall_mean >= 3.0:
        risk_barrier = "MODERATE - Some risk awareness but not prohibitive"
    else:
        risk_barrier = "LOW - Limited risk concerns support adoption"

    print(f"Adoption Barrier Level: {risk_barrier}")

    # Risk tolerance segmentation
    print(f"\nRisk Tolerance Segmentation:")
    high_risk_perception = len(rp_data[rp_data.mean(axis=1) >= 4.0])
    moderate_risk_perception = len(rp_data[(rp_data.mean(axis=1) >= 3.0) & (rp_data.mean(axis=1) < 4.0)])
    low_risk_perception = len(rp_data[rp_data.mean(axis=1) < 3.0])

    print(f"High Risk Perception (≥4.0): {high_risk_perception} ({(high_risk_perception / len(rp_data) * 100):.1f}%)")
    print(
        f"Moderate Risk Perception (3.0-3.9): {moderate_risk_perception} ({(moderate_risk_perception / len(rp_data) * 100):.1f}%)")
    print(f"Low Risk Perception (<3.0): {low_risk_perception} ({(low_risk_perception / len(rp_data) * 100):.1f}%)")

    # Risk vs Economic Motivation analysis
    print(f"\nRisk vs Economic Motivation Analysis:")
    print("Note: Compare these results with Economic Motivation (EM) findings:")
    print("- EM Overall Mean: 4.129 (Strong economic motivation)")
    print("- 72.9% High Economic Motivation users")
    print(f"- RP Overall Mean: {overall_mean:.3f}")

    if overall_mean >= 3.5:
        risk_economic_insight = "High risk awareness BUT strong economic motivation - Risk-reward trade-off behavior"
    elif overall_mean >= 3.0:
        risk_economic_insight = "Moderate risk awareness with strong economic motivation - Calculated risk-taking"
    else:
        risk_economic_insight = "Low risk perception supports strong economic motivation - Risk tolerance enables adoption"

    print(f"Risk-Economic Profile: {risk_economic_insight}")

    # GameFi-specific risk implications
    print(f"\nGameFi Risk Management Implications:")

    # Identify primary risk management needs
    if security_risk >= 4.0:
        risk_priority = "Platform security and exploit prevention should be top priority"
    elif financial_risk >= 4.0:
        risk_priority = "Financial risk mitigation (volatility protection) should be top priority"
    elif sustainability_risk >= 4.0:
        risk_priority = "Economic model sustainability demonstration should be top priority"
    elif uncertainty_risk >= 4.0:
        risk_priority = "Technology/market education and transparency should be top priority"
    else:
        risk_priority = "Balanced risk management approach across all areas"

    print(f"Risk Management Priority: {risk_priority}")

    # Adoption implications
    print(f"\nAdoption Risk Implications:")
    high_risk_percentage = (high_risk_perception / len(rp_data)) * 100

    if high_risk_percentage >= 50:
        adoption_implication = "Majority perceive high risks - extensive risk mitigation required for mass adoption"
    elif high_risk_percentage >= 30:
        adoption_implication = "Significant risk concerns - targeted risk management strategies needed"
    else:
        adoption_implication = "Limited risk concerns - current risk levels acceptable for adoption"

    print(f"Adoption Strategy: {adoption_implication}")

    # Risk communication needs
    print(f"\nRisk Communication Analysis:")
    if uncertainty_risk >= 3.5:
        communication_need = "High uncertainty - requires extensive education and transparency initiatives"
    elif sustainability_risk >= 3.5:
        communication_need = "Sustainability concerns - need clear long-term viability demonstrations"
    elif security_risk >= 3.5:
        communication_need = "Security concerns - need robust security communication and assurance"
    elif financial_risk >= 3.5:
        communication_need = "Financial risk concerns - need clear risk disclosure and mitigation strategies"
    else:
        communication_need = "Balanced communication approach addressing all risk dimensions"

    print(f"Communication Strategy: {communication_need}")

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
    print("\nRisk Perception Scale Interpretation:")
    print("• Mean ≥ 4.0 : High risk perception (strong barriers to adoption)")
    print("• Mean 3.5-3.9 : Moderate-high risk perception (significant concerns)")
    print("• Mean 3.0-3.4 : Moderate risk perception (some awareness)")
    print("• Mean < 3.0 : Low risk perception (limited concerns)")
    print("\nRP Components:")
    print("• RP1: Financial risks (volatility, potential losses)")
    print("• RP2: Security vulnerabilities (hacks, scams, exploits)")
    print("• RP3: Sustainability concerns (long-term economic model viability)")
    print("• RP3.1: Technology/market uncertainty (overall GameFi ecosystem risks)")
    print("\nGameFi Risk Research Implications:")
    print("• High RP scores indicate significant adoption barriers")
    print("• RP1 shows sensitivity to financial volatility and loss potential")
    print("• RP2 reveals security trust issues requiring platform assurance")
    print("• RP3 indicates sustainability skepticism affecting long-term commitment")
    print("• RP3.1 shows technology uncertainty requiring education and transparency")
    print("• Risk-Economic Motivation comparison reveals risk-reward trade-off behavior")
    print("• Risk segmentation helps identify different user risk tolerance profiles")
    print("• Primary risk concerns guide targeted risk mitigation strategies")


# Run the analysis
if __name__ == "__main__":
    main()