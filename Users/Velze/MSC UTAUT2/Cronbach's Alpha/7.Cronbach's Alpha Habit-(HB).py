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

    print("=" * 50)
    print("CRONBACH'S ALPHA ANALYSIS - HABIT (HB)")
    print("=" * 50)

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

    # Try different possible file names for Habit
    possible_files = [
        'habit HB.xlsx',
        'habit (HB).xlsx',
        'habit-HB.xlsx',
        'Habit HB.xlsx',
        'HB.xlsx',
        'habit HB.xls'
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
            print(f"\nTry renaming your file to: 'habit HB.xlsx'")
        else:
            print("  No Excel files found in current directory")
        return

    # Extract HB columns
    hb_columns = ['HB1', 'HB2', 'HB3', 'HB4']

    # Check if all HB columns exist
    missing_cols = [col for col in hb_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return

    hb_data = df[hb_columns].copy()

    print(f"✓ Found {len(hb_columns)} Habit items")

    # Display item descriptions
    print("\nHabit Items:")
    print("HB1: Using GameFi platforms has become a habit for me")
    print("HB2: Playing GameFi games is part of my regular routine")
    print("HB3: Using GameFi platforms has become a regular part of my daily/weekly schedule")
    print("HB4: Using GameFi has become natural to me in my gaming routine")

    print("\nOriginal response distribution:")
    for col in hb_columns:
        print(f"{col}: {hb_data[col].value_counts().to_dict()}")

    # Convert text responses to numeric
    print("\nConverting responses to numeric scale (1-5)...")
    for col in hb_columns:
        hb_data[col] = hb_data[col].apply(convert_likert_to_numeric)

    # Check for missing values after conversion
    missing_count = hb_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} responses could not be converted")
        print("Removing rows with missing values...")
        hb_data = hb_data.dropna()

    print(f"✓ Final dataset: {len(hb_data)} complete responses")

    # Display descriptive statistics
    print("\n" + "=" * 50)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 50)
    print(hb_data.describe().round(3))

    # Calculate Cronbach's Alpha
    print("\n" + "=" * 50)
    print("RELIABILITY ANALYSIS RESULTS")
    print("=" * 50)

    alpha = calculate_cronbach_alpha(hb_data)
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
    print(f"\nNumber of Items: {len(hb_columns)}")
    print(f"Number of Cases: {len(hb_data)}")

    item_total_corr = calculate_item_total_correlations(hb_data)

    print(f"\n{'Item':<6} {'Corrected Item-Total':<20} {'Cronbach\'s α if':<15}")
    print(f"{'':6} {'Correlation':<20} {'Item Deleted':<15}")
    print("-" * 45)

    alpha_deleted = alpha_if_deleted(hb_data)

    for item in hb_columns:
        corr = item_total_corr[item]
        alpha_del = alpha_deleted[item]
        print(f"{item:<6} {corr:>15.4f} {alpha_del:>15.4f}")

    # Item-level analysis
    print(f"\n" + "=" * 50)
    print("ITEM ANALYSIS")
    print("=" * 50)

    print(f"{'Item':<6} {'Mean':<8} {'Std Dev':<10} {'Variance':<10}")
    print("-" * 35)

    for col in hb_columns:
        mean_val = hb_data[col].mean()
        std_val = hb_data[col].std()
        var_val = hb_data[col].var()
        print(f"{col:<6} {mean_val:>6.3f} {std_val:>8.3f} {var_val:>8.3f}")

    # Habit specific interpretation
    print(f"\n" + "=" * 50)
    print("HABIT SCALE EVALUATION")
    print("=" * 50)

    # Calculate mean scores for interpretation
    overall_mean = hb_data.mean().mean()
    print(f"Overall Habit Mean: {overall_mean:.3f}")

    if overall_mean >= 4.0:
        habit_level = "Strong habit formation (GameFi use is highly automatic/routine)"
    elif overall_mean >= 3.5:
        habit_level = "Moderate habit formation (GameFi use becoming routine)"
    elif overall_mean >= 3.0:
        habit_level = "Developing habit formation (Some routine GameFi use)"
    else:
        habit_level = "Weak habit formation (GameFi use not yet routine/automatic)"

    print(f"Interpretation: {habit_level}")

    # Check for differences between habit dimensions
    print(f"\nHabit Breakdown:")
    print(f"HB1 (General habit formation): {hb_data['HB1'].mean():.3f}")
    print(f"HB2 (Routine integration): {hb_data['HB2'].mean():.3f}")
    print(f"HB3 (Schedule regularity): {hb_data['HB3'].mean():.3f}")
    print(f"HB4 (Natural gaming integration): {hb_data['HB4'].mean():.3f}")

    # Habit analysis insights
    print(f"\nHabit Formation Analysis:")

    # Identify strongest and weakest habit aspects
    hb_means = {
        'HB1': hb_data['HB1'].mean(),
        'HB2': hb_data['HB2'].mean(),
        'HB3': hb_data['HB3'].mean(),
        'HB4': hb_data['HB4'].mean()
    }

    strongest_hb = max(hb_means, key=hb_means.get)
    weakest_hb = min(hb_means, key=hb_means.get)

    print(f"Strongest Habit Aspect: {strongest_hb} (M = {hb_means[strongest_hb]:.3f})")
    print(f"Weakest Habit Aspect: {weakest_hb} (M = {hb_means[weakest_hb]:.3f})")

    # Habit dimensions analysis
    general_habit = hb_data['HB1'].mean()  # General habit formation
    routine_integration = hb_data['HB2'].mean()  # Regular routine
    schedule_regularity = hb_data['HB3'].mean()  # Daily/weekly schedule
    natural_integration = hb_data['HB4'].mean()  # Natural gaming routine

    print(f"\nHabit Dimension Analysis:")
    print(f"General Habit Formation: {general_habit:.3f}")
    print(f"Routine Integration: {routine_integration:.3f}")
    print(f"Schedule Regularity: {schedule_regularity:.3f}")
    print(f"Natural Gaming Integration: {natural_integration:.3f}")

    # Habit development insights
    print(f"\nHabit Development Profile:")
    if natural_integration > general_habit:
        habit_insight = "GameFi integrates naturally into existing gaming routines (contextual habit)"
    elif routine_integration > schedule_regularity:
        habit_insight = "Routine-driven habit formation stronger than scheduled usage"
    elif schedule_regularity > routine_integration:
        habit_insight = "Scheduled habit formation stronger than spontaneous routine"
    else:
        habit_insight = "Balanced habit formation across all dimensions"

    print(f"Key Insight: {habit_insight}")

    # Automaticity analysis
    print(f"\nAutomaticity Assessment:")
    if overall_mean >= 4.0:
        automaticity = "High automaticity - GameFi use likely unconscious/automatic"
    elif overall_mean >= 3.5:
        automaticity = "Moderate automaticity - GameFi use becoming less effortful"
    elif overall_mean >= 3.0:
        automaticity = "Developing automaticity - Some unconscious GameFi use"
    else:
        automaticity = "Low automaticity - GameFi use still requires conscious effort"

    print(f"Automaticity Level: {automaticity}")

    # Habit strength implications for adoption
    print(f"\nHabit-Based Adoption Analysis:")
    if overall_mean >= 3.5:
        habit_barrier = "LOW - Strong habits support continued GameFi adoption"
    elif overall_mean >= 3.0:
        habit_barrier = "MODERATE - Developing habits may support adoption"
    else:
        habit_barrier = "HIGH - Weak habits may not sustain long-term adoption"

    print(f"Habit Adoption Support: {habit_barrier}")

    # Behavioral psychology insights
    print(f"\nBehavioral Psychology Analysis:")

    # Calculate habit strength indicators
    high_habit_users = len(hb_data[hb_data.mean(axis=1) >= 4.0])
    habit_percentage = (high_habit_users / len(hb_data)) * 100

    print(f"Strong Habit Users: {high_habit_users}/{len(hb_data)} ({habit_percentage:.1f}%)")

    if habit_percentage >= 30:
        psychology_insight = "Significant portion shows strong habit formation - indicates sustainable adoption"
    elif habit_percentage >= 15:
        psychology_insight = "Moderate habit formation - adoption sustainability developing"
    else:
        psychology_insight = "Limited habit formation - may indicate adoption challenges"

    print(f"Psychology Insight: {psychology_insight}")

    # Interpretation guidelines
    print(f"\n" + "=" * 50)
    print("INTERPRETATION GUIDELINES")
    print("=" * 50)
    print("Cronbach's Alpha Interpretation:")
    print("• α ≥ 0.9  : Excellent reliability")
    print("• α ≥ 0.8  : Good reliability")
    print("• α ≥ 0.7  : Acceptable reliability")
    print("• α ≥ 0.6  : Questionable reliability")
    print("• α < 0.6  : Poor reliability")
    print("\nCorrected Item-Total Correlations:")
    print("• r ≥ 0.3  : Good item discrimination")
    print("• r < 0.3  : Consider removing item")
    print("\nHabit Scale Interpretation:")
    print("• Mean ≥ 4.0 : Strong habits (automatic/routine behavior)")
    print("• Mean 3.5-3.9 : Moderate habits (developing automaticity)")
    print("• Mean 3.0-3.4 : Developing habits (some routine behavior)")
    print("• Mean < 3.0 : Weak habits (effortful/conscious behavior)")
    print("\nHB Components:")
    print("• HB1: General habit formation (overall automaticity)")
    print("• HB2: Routine integration (regular behavior patterns)")
    print("• HB3: Schedule regularity (temporal behavior consistency)")
    print("• HB4: Natural gaming integration (contextual automaticity)")
    print("\nGameFi Habit Research Implications:")
    print("• Strong habits indicate sustainable long-term adoption")
    print("• Weak habits may require intervention for continued use")
    print("• Habit formation varies: contextual > routine > scheduled")
    print("• Natural gaming integration shows behavioral stickiness")
    print("• High habit scores predict lower churn/abandonment rates")
    print("• Automaticity reduces cognitive effort and increases retention")


# Run the analysis
if __name__ == "__main__":
    main()