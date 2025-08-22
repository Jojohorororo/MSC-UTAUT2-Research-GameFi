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
    print("CRONBACH'S ALPHA ANALYSIS - BEHAVIORAL INTENTION (BI)")
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

    # Try different possible file names for Behavioral Intention
    possible_files = [
        'behavioralintention BI.xlsx',
        'behavioral intention BI.xlsx',
        'behavioral-intention-BI.xlsx',
        'behavioralintention (BI).xlsx',
        'Behavioral Intention BI.xlsx',
        'BI.xlsx',
        'behavioralintention BI.xls'
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
            print(f"\nTry renaming your file to: 'behavioralintention BI.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract BI columns
    bi_columns = ['BI1', 'BI2', 'BI3']

    # Check if all BI columns exist
    missing_cols = [col for col in bi_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    bi_data = df[bi_columns].copy()

    print(f"✓ Found {len(bi_columns)} Behavioral Intention items")

    # Display item descriptions
    print("\nBehavioral Intention Items:")
    print("BI1: I intend to use GameFi platforms in the future (within the next 6 months)")
    print("BI2: I plan to use GameFi platforms regularly in the future")
    print("BI3: I intend to recommend GameFi platforms to others through social media and online communities")

    print("\nOriginal response distribution:")
    for col in bi_columns:
        print(f"{col}: {bi_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in bi_columns:
        bi_data[col] = bi_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = bi_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        bi_data = bi_data.dropna()

    print(f"✓ Final dataset: {len(bi_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(bi_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 60)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 60)

    alpha = calculate_cronbach_alpha(bi_data)
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
    print(f"\nNumber of Items: {len(bi_columns)}")
    print(f"Number of Cases: {len(bi_data)}")

    item_total_corr = calculate_item_total_correlations(bi_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(bi_data)

    for item in bi_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 60)
    print("ITEM ANALYSIS")
    print("=" * 60)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in bi_columns:
        mean_val = bi_data[col].mean()
        std_val = bi_data[col].std()
        var_val = bi_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Behavioral Intention specific interpretation
    print(f"\n" + "=" * 60)
    print("BEHAVIORAL INTENTION SCALE EVALUATION")
    print("=" * 60)

    # Calculate mean scores for interpretation
    overall_mean = bi_data.mean().mean()
    print(f"Overall Behavioral Intention Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        bi_level = "Strong behavioral intention (High likelihood of GameFi adoption)"
    elif overall_mean >= 3.5:
        bi_level = "Moderate-high behavioral intention (Good adoption likelihood)"
    elif overall_mean >= 3.0:
        bi_level = "Moderate behavioral intention (Neutral adoption likelihood)"
    else:
        bi_level = "Weak behavioral intention (Low adoption likelihood)"

    print(f"Interpretation: {bi_level}")

    # Check for differences between behavioral intention dimensions
    print(f"\nBehavioral Intention Breakdown:")
    print(f"BI1 (Future usage intention - 6 months): {bi_data['BI1'].mean():.3f}")
    print(f"BI2 (Regular usage plans): {bi_data['BI2'].mean():.3f}")
    print(f"BI3 (Recommendation intention): {bi_data['BI3'].mean():.3f}")

    # Behavioral intention analysis insights
    print(f"\nBehavioral Intention Analysis:")

    # Identify strongest and weakest intention aspects
    bi_means = {
        'BI1': bi_data['BI1'].mean(),
        'BI2': bi_data['BI2'].mean(),
        'BI3': bi_data['BI3'].mean()
    }

    strongest_bi = max(bi_means, key=bi_means.get)
    weakest_bi = min(bi_means, key=bi_means.get)

    print(f"Strongest Intention Aspect: {strongest_bi} (M = {bi_means[strongest_bi]:.3f})")
    print(f"Weakest Intention Aspect: {weakest_bi} (M = {bi_means[weakest_bi]:.3f})")

    # Intention dimensions analysis
    future_usage = bi_data['BI1'].mean()  # 6-month usage intention
    regular_usage = bi_data['BI2'].mean()  # Regular usage plans
    recommendation = bi_data['BI3'].mean()  # Word-of-mouth intention

    print(f"\nIntention Dimension Analysis:")
    print(f"Future Usage Intention (6 months): {future_usage:.3f}")
    print(f"Regular Usage Plans: {regular_usage:.3f}")
    print(f"Recommendation Intention: {recommendation:.3f}")

    # Behavioral prediction insights
    print(f"\nBehavioral Prediction Profile:")
    if regular_usage > future_usage:
        intention_insight = "Strong commitment to sustained usage (regular > future intention)"
    elif future_usage > recommendation:
        intention_insight = "Personal usage intention stronger than social advocacy"
    elif recommendation > future_usage:
        intention_insight = "Social advocacy stronger than personal usage intention"
    else:
        intention_insight = "Balanced behavioral intentions across all dimensions"

    print(f"Key Insight: {intention_insight}")

    # Adoption likelihood assessment
    print(f"\nAdoption Likelihood Assessment:")
    if overall_mean >= 4.0:
        adoption_likelihood = "HIGH - Strong intentions predict likely adoption"
    elif overall_mean >= 3.5:
        adoption_likelihood = "MODERATE-HIGH - Good intentions support adoption"
    elif overall_mean >= 3.0:
        adoption_likelihood = "MODERATE - Mixed intentions, uncertain adoption"
    else:
        adoption_likelihood = "LOW - Weak intentions may hinder adoption"

    print(f"Adoption Likelihood: {adoption_likelihood}")

    # Word-of-mouth analysis
    print(f"\nWord-of-Mouth Analysis:")
    if recommendation >= 4.0:
        wom_strength = "Strong viral potential (high recommendation intentions)"
    elif recommendation >= 3.5:
        wom_strength = "Moderate viral potential (some recommendation intentions)"
    elif recommendation >= 3.0:
        wom_strength = "Limited viral potential (neutral recommendation intentions)"
    else:
        wom_strength = "Low viral potential (weak recommendation intentions)"

    print(f"Viral Marketing Potential: {wom_strength}")

    # Intention-behavior gap analysis
    print(f"\nIntention-Behavior Gap Analysis:")

    # Calculate high intention users
    high_intention_users = len(bi_data[bi_data.mean(axis=1) >= 4.0])
    intention_percentage = (high_intention_users / len(bi_data)) * 100

    print(f"High Intention Users: {high_intention_users}/{len(bi_data)} ({intention_percentage:.1f}%)")

    if intention_percentage >= 50:
        gap_insight = "Majority shows strong intentions - low intention-behavior gap expected"
    elif intention_percentage >= 30:
        gap_insight = "Significant portion shows strong intentions - moderate gap expected"
    else:
        gap_insight = "Limited strong intentions - high intention-behavior gap possible"

    print(f"Gap Analysis: {gap_insight}")

    # Timeline analysis
    print(f"\nTimeline Commitment Analysis:")
    if future_usage >= 4.0:
        timeline_commitment = "Strong 6-month commitment (high near-term adoption probability)"
    elif future_usage >= 3.5:
        timeline_commitment = "Moderate 6-month commitment (good near-term adoption probability)"
    elif future_usage >= 3.0:
        timeline_commitment = "Weak 6-month commitment (uncertain near-term adoption)"
    else:
        timeline_commitment = "Very weak 6-month commitment (low near-term adoption probability)"

    print(f"6-Month Outlook: {timeline_commitment}")

    # Behavioral segmentation
    print(f"\nBehavioral Intention Segmentation:")
    strong_adopters = len(bi_data[bi_data.mean(axis=1) >= 4.0])
    moderate_adopters = len(bi_data[(bi_data.mean(axis=1) >= 3.0) & (bi_data.mean(axis=1) < 4.0)])
    weak_adopters = len(bi_data[bi_data.mean(axis=1) < 3.0])

    print(f"Strong Adopters (≥4.0): {strong_adopters} ({(strong_adopters / len(bi_data) * 100):.1f}%)")
    print(f"Moderate Adopters (3.0-3.9): {moderate_adopters} ({(moderate_adopters / len(bi_data) * 100):.1f}%)")
    print(f"Weak Adopters (<3.0): {weak_adopters} ({(weak_adopters / len(bi_data) * 100):.1f}%)")

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
    print("\nBehavioral Intention Scale Interpretation:")
    print("• Mean ≥ 4.0 : Strong intentions (high adoption likelihood)")
    print("• Mean 3.5-3.9 : Moderate-high intentions (good adoption likelihood)")
    print("• Mean 3.0-3.4 : Moderate intentions (neutral adoption likelihood)")
    print("• Mean < 3.0 : Weak intentions (low adoption likelihood)")
    print("\nBI Components:")
    print("• BI1: Future usage intention (6-month timeline commitment)")
    print("• BI2: Regular usage plans (sustained engagement commitment)")
    print("• BI3: Recommendation intention (word-of-mouth/viral potential)")
    print("\nGameFi Behavioral Research Implications:")
    print("• Strong BI scores predict actual GameFi adoption behavior")
    print("• BI1 shows near-term conversion probability")
    print("• BI2 indicates long-term engagement potential")
    print("• BI3 reveals viral marketing and organic growth potential")
    print("• High BI-Habit correlation suggests behavioral consistency")
    print("• Intention-behavior gap analysis identifies conversion barriers")
    print("• Timeline analysis helps predict adoption velocity")


# Run the analysis
if __name__ == "__main__":
    main()