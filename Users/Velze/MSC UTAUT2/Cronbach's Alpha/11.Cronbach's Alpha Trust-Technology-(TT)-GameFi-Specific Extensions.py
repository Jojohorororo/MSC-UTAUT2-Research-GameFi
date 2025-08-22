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
    print("CRONBACH'S ALPHA ANALYSIS - TRUST IN TECHNOLOGY (TT)")
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

    # Try different possible file names for Trust in Technology
    possible_files = [
        'Trust Technology TT Gamefi Specific Extensions.xlsx',
        'trust technology TT Gamefi Specific Extensions.xlsx',
        'Trust Technology TT.xlsx',
        'trust technology TT.xlsx',
        'TrustTechnology TT Gamefi Specific Extensions.xlsx',
        'Trust Technology (TT) Gamefi Specific Extensions.xlsx',
        'TT.xlsx',
        'Trust Technology TT Gamefi Specific Extensions.xls'
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
            print(f"\nTry renaming your file to: 'Trust Technology TT Gamefi Specific Extensions.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract TT columns
    tt_columns = ['TT1', 'TT2', 'TT3']

    # Check if all TT columns exist
    missing_cols = [col for col in tt_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    tt_data = df[tt_columns].copy()

    print(f"✓ Found {len(tt_columns)} Trust in Technology items")

    # Display item descriptions
    print("\nTrust in Technology Items (GameFi-Specific Extensions):")
    print("TT1: I trust the blockchain technology and smart contracts that power GameFi platforms")
    print("TT2: I feel confident that my digital assets in GameFi are secure and truly owned by me")
    print("TT3: I believe GameFi platform developers are generally trustworthy and transparent")

    print("\nOriginal response distribution:")
    for col in tt_columns:
        print(f"{col}: {tt_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in tt_columns:
        tt_data[col] = tt_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = tt_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        tt_data = tt_data.dropna()

    print(f"✓ Final dataset: {len(tt_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 65)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 65)
    print(tt_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 65)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 65)

    alpha = calculate_cronbach_alpha(tt_data)
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
    print(f"\nNumber of Items: {len(tt_columns)}")
    print(f"Number of Cases: {len(tt_data)}")

    item_total_corr = calculate_item_total_correlations(tt_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(tt_data)

    for item in tt_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 65)
    print("ITEM ANALYSIS")
    print("=" * 65)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in tt_columns:
        mean_val = tt_data[col].mean()
        std_val = tt_data[col].std()
        var_val = tt_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Trust in Technology specific interpretation
    print(f"\n" + "=" * 65)
    print("TRUST IN TECHNOLOGY SCALE EVALUATION")
    print("=" * 65)

    # Calculate mean scores for interpretation
    overall_mean = tt_data.mean().mean()
    print(f"Overall Trust in Technology Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        tt_level = "High trust in technology (Strong confidence in GameFi technology)"
    elif overall_mean >= 3.5:
        tt_level = "Moderate-high trust in technology (Good confidence levels)"
    elif overall_mean >= 3.0:
        tt_level = "Moderate trust in technology (Some confidence)"
    else:
        tt_level = "Low trust in technology (Limited confidence in GameFi technology)"

    print(f"Interpretation: {tt_level}")

    # Check for differences between trust dimensions
    print(f"\nTrust in Technology Breakdown:")
    print(f"TT1 (Blockchain/smart contract trust): {tt_data['TT1'].mean():.3f}")
    print(f"TT2 (Digital asset security confidence): {tt_data['TT2'].mean():.3f}")
    print(f"TT3 (Developer trustworthiness): {tt_data['TT3'].mean():.3f}")

    # Trust analysis insights
    print(f"\nTrust in Technology Analysis:")

    # Identify strongest and weakest trust aspects
    tt_means = {
        'TT1': tt_data['TT1'].mean(),
        'TT2': tt_data['TT2'].mean(),
        'TT3': tt_data['TT3'].mean()
    }

    strongest_tt = max(tt_means, key=tt_means.get)
    weakest_tt = min(tt_means, key=tt_means.get)

    print(f"Highest Trust Aspect: {strongest_tt} (M = {tt_means[strongest_tt]:.3f})")
    print(f"Lowest Trust Aspect: {weakest_tt} (M = {tt_means[weakest_tt]:.3f})")

    # Trust dimensions analysis
    technology_trust = tt_data['TT1'].mean()  # Blockchain/smart contracts
    security_confidence = tt_data['TT2'].mean()  # Digital asset security
    developer_trust = tt_data['TT3'].mean()  # Platform developers

    print(f"\nTrust Dimension Analysis:")
    print(f"Technology Trust (Blockchain/Smart Contracts): {technology_trust:.3f}")
    print(f"Security Confidence (Digital Assets): {security_confidence:.3f}")
    print(f"Developer Trust (Platform Credibility): {developer_trust:.3f}")

    # Trust profile insights
    print(f"\nTrust Profile Analysis:")
    if technology_trust > developer_trust and technology_trust > security_confidence:
        trust_insight = "Technology trust exceeds human/platform trust (tech > people)"
    elif security_confidence > technology_trust and security_confidence > developer_trust:
        trust_insight = "Digital asset security confidence is highest trust dimension"
    elif developer_trust > technology_trust and developer_trust > security_confidence:
        trust_insight = "Developer trustworthiness exceeds technology trust (people > tech)"
    else:
        trust_insight = "Balanced trust across technology, security, and developer dimensions"

    print(f"Key Insight: {trust_insight}")

    # Trust barrier assessment
    print(f"\nTrust Barrier Assessment:")
    if overall_mean >= 4.0:
        trust_barrier = "LOW - High trust supports GameFi adoption"
    elif overall_mean >= 3.5:
        trust_barrier = "MODERATE-LOW - Good trust levels support adoption"
    elif overall_mean >= 3.0:
        trust_barrier = "MODERATE - Some trust concerns may limit adoption"
    else:
        trust_barrier = "HIGH - Low trust may significantly hinder adoption"

    print(f"Trust Adoption Support: {trust_barrier}")

    # Trust segmentation
    print(f"\nTrust Segmentation:")
    high_trust = len(tt_data[tt_data.mean(axis=1) >= 4.0])
    moderate_trust = len(tt_data[(tt_data.mean(axis=1) >= 3.0) & (tt_data.mean(axis=1) < 4.0)])
    low_trust = len(tt_data[tt_data.mean(axis=1) < 3.0])

    print(f"High Trust (≥4.0): {high_trust} ({(high_trust / len(tt_data) * 100):.1f}%)")
    print(f"Moderate Trust (3.0-3.9): {moderate_trust} ({(moderate_trust / len(tt_data) * 100):.1f}%)")
    print(f"Low Trust (<3.0): {low_trust} ({(low_trust / len(tt_data) * 100):.1f}%)")

    # Trust vs Risk Perception analysis
    print(f"\nTrust vs Risk Perception Analysis:")
    print("Note: Compare these results with Risk Perception (RP) findings:")
    print("- RP Overall Mean: 4.016 (High risk perception)")
    print("- RP2 (Security vulnerabilities): 4.151 (Highest risk concern)")
    print(f"- TT Overall Mean: {overall_mean:.3f}")
    print(f"- TT2 (Digital asset security confidence): {security_confidence:.3f}")

    # Calculate trust-risk relationship
    if overall_mean >= 3.5 and overall_mean < 4.016:  # Using RP mean as reference
        trust_risk_insight = "Moderate trust despite high risk awareness - Cautious but willing adoption"
    elif overall_mean >= 4.0:
        trust_risk_insight = "High trust despite high risk perception - Technology confidence overcomes risk concerns"
    elif overall_mean < 3.0:
        trust_risk_insight = "Low trust consistent with high risk perception - Trust and risk barriers align"
    else:
        trust_risk_insight = "Moderate trust and high risk perception - Mixed confidence signals"

    print(f"Trust-Risk Profile: {trust_risk_insight}")

    # Technology adoption readiness
    print(f"\nTechnology Adoption Readiness:")
    if technology_trust >= 4.0:
        tech_readiness = "High blockchain/smart contract confidence supports technology adoption"
    elif technology_trust >= 3.5:
        tech_readiness = "Moderate blockchain/smart contract confidence - good adoption potential"
    elif technology_trust >= 3.0:
        tech_readiness = "Some blockchain/smart contract skepticism - education needed"
    else:
        tech_readiness = "Low blockchain/smart contract confidence - significant trust building required"

    print(f"Blockchain Readiness: {tech_readiness}")

    # Security confidence analysis
    print(f"\nSecurity Confidence Analysis:")
    if security_confidence >= 4.0:
        security_analysis = "High digital asset security confidence despite security risk concerns"
    elif security_confidence >= 3.5:
        security_analysis = "Moderate digital asset security confidence - balanced security perception"
    elif security_confidence >= 3.0:
        security_analysis = "Some digital asset security concerns - matches high security risk perception"
    else:
        security_analysis = "Low digital asset security confidence - aligns with high security risk concerns"

    print(f"Security Trust: {security_analysis}")

    # Developer trust implications
    print(f"\nDeveloper Trust Implications:")
    if developer_trust >= 4.0:
        developer_analysis = "High developer trust supports platform credibility"
    elif developer_trust >= 3.5:
        developer_analysis = "Moderate developer trust - good platform credibility"
    elif developer_trust >= 3.0:
        developer_analysis = "Some developer skepticism - transparency improvements needed"
    else:
        developer_analysis = "Low developer trust - significant credibility building required"

    print(f"Platform Credibility: {developer_analysis}")

    # Trust building strategies
    print(f"\nTrust Building Strategy Analysis:")
    if weakest_tt == 'TT1':
        trust_strategy = "Focus on blockchain/smart contract education and transparency"
    elif weakest_tt == 'TT2':
        trust_strategy = "Focus on security assurance and digital asset protection"
    elif weakest_tt == 'TT3':
        trust_strategy = "Focus on developer transparency and platform governance"
    else:
        trust_strategy = "Balanced trust building across all dimensions"

    print(f"Priority Strategy: {trust_strategy}")

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
    print("\nTrust in Technology Scale Interpretation:")
    print("• Mean ≥ 4.0 : High trust (strong confidence in GameFi technology)")
    print("• Mean 3.5-3.9 : Moderate-high trust (good confidence levels)")
    print("• Mean 3.0-3.4 : Moderate trust (some confidence)")
    print("• Mean < 3.0 : Low trust (limited confidence)")
    print("\nTT Components:")
    print("• TT1: Technology trust (blockchain/smart contract confidence)")
    print("• TT2: Security confidence (digital asset ownership/security trust)")
    print("• TT3: Developer trust (platform credibility and transparency)")
    print("\nGameFi Trust Research Implications:")
    print("• High TT scores indicate technology confidence supports adoption")
    print("• TT1 shows blockchain/smart contract acceptance levels")
    print("• TT2 reveals digital asset security confidence vs risk perception")
    print("• TT3 indicates developer/platform credibility perceptions")
    print("• Trust-Risk comparison reveals confidence vs concern balance")
    print("• Low trust dimensions guide targeted confidence building strategies")
    print("• Technology trust vs human trust patterns inform platform design")
    print("• Security confidence analysis guides security communication strategies")


# Run the analysis
if __name__ == "__main__":
    main()