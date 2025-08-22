import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def convert_ub_to_numeric(response, item_type):
    """Convert Use Behavior responses to numeric values"""
    if item_type == 'frequency':  # UB1 - frequency responses
        frequency_mapping = {
            'Never': 1,
            'Monthly or less': 1,
            'A few times a month': 2,
            'Weekly': 3,
            'Several times a week': 4,
            'Daily': 5
        }
        return frequency_mapping.get(response, np.nan)
    elif item_type == 'hours':  # UB2 - time-based responses
        hours_mapping = {
            '0 hours': 1,
            'Less than 1 hour': 1,
            '1-5 hours': 2,
            '6-10 hours': 3,
            '11-20 hours': 4,
            'More than 20 hours': 5
        }
        return hours_mapping.get(response, np.nan)
    else:
        return np.nan


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

    print("=" * 60)
    print("CRONBACH'S ALPHA ANALYSIS - USE BEHAVIOR (UB)")
    print("Final UTAUT2 + GameFi Extensions Construct")
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

    # Try different possible file names for Use Behavior
    possible_files = [
        'Usebehavior UB.xlsx',
        'Use behavior UB.xlsx',
        'use behavior UB.xlsx',
        'UseBehavior UB.xlsx',
        'Use Behavior UB.xlsx',
        'usebehavior UB.xlsx',
        'Usebehavior (UB).xlsx',
        'UB.xlsx',
        'Usebehavior UB.xls'
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
            print(f"\nTry renaming your file to: 'Usebehavior UB.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract UB columns
    ub_columns = ['UB1', 'UB2']

    # Check if all UB columns exist
    missing_cols = [col for col in ub_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    ub_data = df[ub_columns].copy()

    print(f"✓ Found {len(ub_columns)} Use Behavior items")

    # Display item descriptions
    print("\nUse Behavior Items:")
    print("UB1: How often do you currently use GameFi platforms")
    print("UB2: On average, how many hours per week do you spend on GameFi platforms")

    print("\nOriginal response distribution:")
    for col in ub_columns:
        print(f"{col}: {ub_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    print("UB1 (Frequency): Never/Monthly=1, Few times/month=2, Weekly=3, Several times/week=4, Daily=5")
    print("UB2 (Hours): 0-1 hours=1, 1-5 hours=2, 6-10 hours=3, 11-20 hours=4, 20+ hours=5")
    ub_data['UB1'] = ub_data['UB1'].apply(lambda x: convert_ub_to_numeric(x, 'frequency'))
    ub_data['UB2'] = ub_data['UB2'].apply(lambda x: convert_ub_to_numeric(x, 'hours'))

    # Check for missing values after conversion
    missing_count = ub_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Checking for unconverted responses...")
        for col in ub_columns:
            unconverted = ub_data[col].isnull().sum()
            if unconverted > 0:
                print(f"  {col}: {unconverted} unconverted responses")
                # Show sample of unconverted responses for debugging
                original_col = df[col]
                unconverted_responses = original_col[ub_data[col].isnull()].unique()[:5]
                print(f"    Sample unconverted: {unconverted_responses}")
        print("Removing rows with missing values...")
        ub_data = ub_data.dropna()

    if len(ub_data) == 0:
        print("❌ No complete responses after conversion. Please check response formats.")
        print("Expected formats:")
        print("UB1: 'Never', 'Monthly or less', 'A few times a month', 'Weekly', 'Several times a week', 'Daily'")
        print("UB2: '0 hours', 'Less than 1 hour', '1-5 hours', '6-10 hours', '11-20 hours', 'More than 20 hours'")
        return

    print(f"✓ Final dataset: {len(ub_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(ub_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 60)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 60)

    alpha = calculate_cronbach_alpha(ub_data)
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
    print(f"\nNumber of Items: {len(ub_columns)}")
    print(f"Number of Cases: {len(ub_data)}")

    item_total_corr = calculate_item_total_correlations(ub_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(ub_data)

    for item in ub_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        if pd.isna(alpha_del):
            alpha_del_str = "N/A (1 item)"
        else:
            alpha_del_str = f"{alpha_del:.4f}"
        print(f"{item:<6} {corr:>15.4f} {alpha_del_str:>15}")

    # Calculate inter-item correlation for 2-item scale
    inter_item_corr = ub_data['UB1'].corr(ub_data['UB2'])
    print(f"\nInter-item Correlation (UB1-UB2): {inter_item_corr:.4f}")
    print("Note: For 2-item scales, inter-item correlation should be ≥ 0.3")

    # Item-level analysis
    print(f"\n" + "=" * 60)
    print("ITEM ANALYSIS")
    print("=" * 60)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in ub_columns:
        mean_val = ub_data[col].mean()
        std_val = ub_data[col].std()
        var_val = ub_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Use Behavior specific interpretation
    print(f"\n" + "=" * 60)
    print("USE BEHAVIOR SCALE EVALUATION")
    print("=" * 60)

    # Calculate mean scores for interpretation
    overall_mean = ub_data.mean().mean()
    print(f"Overall Use Behavior Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        ub_level = "High use behavior (Frequent and intensive GameFi usage)"
    elif overall_mean >= 3.5:
        ub_level = "Moderate-high use behavior (Regular GameFi usage)"
    elif overall_mean >= 3.0:
        ub_level = "Moderate use behavior (Some GameFi usage)"
    else:
        ub_level = "Low use behavior (Limited GameFi usage)"

    print(f"Interpretation: {ub_level}")

    # Check for differences between use behavior dimensions
    print(f"\nUse Behavior Breakdown:")
    print(f"UB1 (Usage frequency): {ub_data['UB1'].mean():.3f}")
    print(f"UB2 (Time investment - hours/week): {ub_data['UB2'].mean():.3f}")

    # Use behavior analysis insights
    print(f"\nUse Behavior Analysis:")

    # Identify stronger behavioral pattern
    ub_means = {
        'UB1': ub_data['UB1'].mean(),
        'UB2': ub_data['UB2'].mean()
    }

    stronger_ub = max(ub_means, key=ub_means.get)
    weaker_ub = min(ub_means, key=ub_means.get)

    print(f"Stronger Behavior Pattern: {stronger_ub} (M = {ub_means[stronger_ub]:.3f})")
    print(f"Weaker Behavior Pattern: {weaker_ub} (M = {ub_means[weaker_ub]:.3f})")

    # Behavioral dimensions analysis
    usage_frequency = ub_data['UB1'].mean()  # How often
    time_investment = ub_data['UB2'].mean()  # Hours per week

    print(f"\nBehavioral Dimension Analysis:")
    print(f"Usage Frequency: {usage_frequency:.3f}")
    print(f"Time Investment (Hours/Week): {time_investment:.3f}")

    # Behavioral pattern insights
    print(f"\nBehavioral Pattern Analysis:")
    if usage_frequency > time_investment:
        behavior_insight = "Frequent but brief usage patterns (high frequency, lower time investment)"
    elif time_investment > usage_frequency:
        behavior_insight = "Intensive usage sessions (longer time investment, lower frequency)"
    else:
        behavior_insight = "Balanced usage patterns (consistent frequency and time investment)"

    print(f"Key Insight: {behavior_insight}")

    # Usage intensity assessment
    print(f"\nUsage Intensity Assessment:")
    if overall_mean >= 4.0:
        intensity_level = "HIGH - Heavy GameFi platform usage"
    elif overall_mean >= 3.5:
        intensity_level = "MODERATE-HIGH - Regular GameFi platform usage"
    elif overall_mean >= 3.0:
        intensity_level = "MODERATE - Occasional GameFi platform usage"
    else:
        intensity_level = "LOW - Light GameFi platform usage"

    print(f"Usage Intensity: {intensity_level}")

    # Use behavior segmentation
    print(f"\nUse Behavior Segmentation:")
    high_usage = len(ub_data[ub_data.mean(axis=1) >= 4.0])
    moderate_usage = len(ub_data[(ub_data.mean(axis=1) >= 3.0) & (ub_data.mean(axis=1) < 4.0)])
    low_usage = len(ub_data[ub_data.mean(axis=1) < 3.0])

    print(f"High Usage (≥4.0): {high_usage} ({(high_usage / len(ub_data) * 100):.1f}%)")
    print(f"Moderate Usage (3.0-3.9): {moderate_usage} ({(moderate_usage / len(ub_data) * 100):.1f}%)")
    print(f"Low Usage (<3.0): {low_usage} ({(low_usage / len(ub_data) * 100):.1f}%)")

    # Intention-Behavior Gap Analysis
    print(f"\nINTENTION-BEHAVIOR GAP ANALYSIS")
    print("=" * 60)
    print("Critical UTAUT2 Analysis: Behavioral Intention vs Actual Use Behavior")
    print("\nCompare with Behavioral Intention (BI) findings:")
    print("- Behavioral Intention (BI): 4.092 (Strong adoption intentions)")
    print("- 71.5% High Behavioral Intention users")
    print(f"- Use Behavior (UB): {overall_mean:.3f}")
    print(f"- {(high_usage / len(ub_data) * 100):.1f}% High Use Behavior users")

    # Calculate intention-behavior gap
    intention_behavior_gap = 4.092 - overall_mean
    intention_percentage = 71.5
    behavior_percentage = (high_usage / len(ub_data) * 100)
    percentage_gap = intention_percentage - behavior_percentage

    print(f"\nIntention-Behavior Gap Metrics:")
    print(f"Mean Gap (BI - UB): {intention_behavior_gap:.3f} points")
    print(f"Percentage Gap (High BI - High UB): {percentage_gap:.1f} percentage points")

    if intention_behavior_gap > 0.5:
        gap_interpretation = "SIGNIFICANT intention-behavior gap - Many intend but don't follow through"
    elif intention_behavior_gap > 0.2:
        gap_interpretation = "MODERATE intention-behavior gap - Some conversion challenges"
    else:
        gap_interpretation = "MINIMAL intention-behavior gap - Good intention-to-behavior conversion"

    print(f"Gap Assessment: {gap_interpretation}")

    # Factors influencing behavior conversion
    print(f"\nBehavior Conversion Analysis:")
    print("Potential factors influencing intention-to-behavior conversion:")
    print("- Economic Motivation (EM): 4.129 (Strong economic drive)")
    print("- Risk Perception (RP): 4.016 (High risk awareness)")
    print("- Trust in Technology (TT): 3.433 (Moderate tech trust)")
    print("- Regulatory Concerns (RC): 3.313 (Manageable regulatory worries)")

    if overall_mean < 3.5:
        conversion_analysis = "High intentions but moderate behavior - Risk/trust barriers may limit conversion"
    elif overall_mean >= 3.5:
        conversion_analysis = "Good intention-to-behavior conversion - Motivations overcome barriers"
    else:
        conversion_analysis = "Mixed conversion patterns - Some barriers remain"

    print(f"Conversion Pattern: {conversion_analysis}")

    # UTAUT2 Model Performance
    print(f"\nUTAUT2 MODEL PERFORMANCE ASSESSMENT")
    print("=" * 60)
    print("Final Model Outcome Analysis:")

    # Calculate model success indicators
    if overall_mean >= 3.5 and intention_behavior_gap < 0.5:
        model_success = "HIGH - Strong intentions translate to strong behavior"
    elif overall_mean >= 3.0 and intention_behavior_gap < 0.8:
        model_success = "MODERATE - Good intentions with some behavior conversion"
    else:
        model_success = "MIXED - Intentions don't fully translate to behavior"

    print(f"Model Predictive Success: {model_success}")

    # Practical implications
    print(f"\nPractical Implications:")
    if intention_behavior_gap > 0.5:
        practical_implication = "Focus on removing barriers to convert intentions into actual usage"
    else:
        practical_implication = "Good conversion - focus on maintaining and increasing intentions"

    print(f"Strategy Focus: {practical_implication}")

    # GameFi Platform Usage Insights
    print(f"\nGameFi Platform Usage Insights:")
    high_usage_percentage = (high_usage / len(ub_data)) * 100

    if high_usage_percentage >= 40:
        platform_insight = "Strong user base with high engagement - Platform stickiness achieved"
    elif high_usage_percentage >= 25:
        platform_insight = "Moderate user engagement - Room for usage improvement"
    else:
        platform_insight = "Limited user engagement - Significant usage barriers remain"

    print(f"Platform Performance: {platform_insight}")

    # Final Research Summary
    print(f"\nFINAL UTAUT2 + GAMEFI RESEARCH SUMMARY")
    print("=" * 60)
    print("Complete Model Overview:")
    print(f"1. Performance Expectancy (PE): 0.9297 α - Strong utility perceptions")
    print(f"2. Behavioral Intention (BI): 0.9116 α - {4.092:.3f} mean - Strong adoption intentions")
    print(f"3. Use Behavior (UB): {alpha:.4f} α - {overall_mean:.3f} mean - Actual usage behavior")
    print(f"4. Economic Motivation (EM): 0.8816 α - {4.129:.3f} mean - Primary driver")
    print(f"5. Risk Perception (RP): 0.9066 α - {4.016:.3f} mean - High risk awareness")
    print(f"6. Trust in Technology (TT): 0.8087 α - {3.433:.3f} mean - Moderate tech trust")
    print(f"7. Regulatory Concerns (RC): 0.8947 α - {3.313:.3f} mean - Manageable barriers")

    print(f"\nKey Finding: {gap_interpretation}")
    print(f"User Profile: Sophisticated economic actors who weigh risks against rewards")
    print(
        f"Adoption Pattern: High intentions with {'good' if intention_behavior_gap < 0.5 else 'moderate'} behavioral follow-through")

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
    print("\nInter-item Correlation (2-item scales):")
    print("• r ≥ 0.3  : Adequate correlation")
    print("• r < 0.3  : Poor item relationship")
    print("\nUse Behavior Scale Interpretation:")
    print("• Mean ≥ 4.0 : High usage (frequent and intensive GameFi use)")
    print("• Mean 3.5-3.9 : Moderate-high usage (regular GameFi use)")
    print("• Mean 3.0-3.4 : Moderate usage (some GameFi use)")
    print("• Mean < 3.0 : Low usage (limited GameFi use)")
    print("\nUB Components:")
    print("• UB1: Usage frequency (how often participants use GameFi)")
    print("• UB2: Time investment (hours per week spent on GameFi)")
    print("\nGameFi Use Behavior Research Implications:")
    print("• UB is the ultimate outcome variable in UTAUT2 models")
    print("• Intention-behavior gap analysis reveals conversion effectiveness")
    print("• High UB scores indicate successful GameFi platform adoption")
    print("• UB patterns (frequency vs time) inform platform design strategies")
    print("• Low UB despite high intentions suggests remaining adoption barriers")
    print("• Use behavior segmentation identifies different user engagement levels")
    print("• Behavioral analysis guides retention and engagement strategies")


# Run the analysis
if __name__ == "__main__":
    main()