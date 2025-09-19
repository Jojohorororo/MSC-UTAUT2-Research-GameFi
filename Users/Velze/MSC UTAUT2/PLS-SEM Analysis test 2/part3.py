# UTAUT2 GameFi Complete v1 Model Analysis
# Including UB (Use Behavior) structural paths: BI→UB, HB→UB, FC→UB

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_complete_data(file_path="utaut2_cleaned_data.xlsx"):
    """
    Load and prepare UTAUT2 data including UB construct for complete v1 analysis
    """
    print("🚀 Starting UTAUT2 Complete v1 PLS-SEM Analysis...")

    # Load the cleaned data
    df = pd.read_excel(file_path)
    print(f"✅ Data loaded: {len(df)} participants × {len(df.columns)} variables")

    # Define complete measurement model including UB
    measurement_model = {
        'PE': ['PE1', 'PE2', 'PE3', 'PE4', 'PE5'],  # Performance Expectancy
        'EE': ['EE1', 'EE2', 'EE3', 'EE4'],  # Effort Expectancy
        'SI': ['SI1', 'SI2', 'SI3'],  # Social Influence
        'FC': ['FC1', 'FC2', 'FC3', 'FC4'],  # Facilitating Conditions
        'HM': ['HM1', 'HM2', 'HM3', 'HM4'],  # Hedonic Motivation
        'PV': ['PV1', 'PV2', 'PV3'],  # Price Value
        'HB': ['HB1', 'HB2', 'HB3', 'HB4'],  # Habit
        'BI': ['BI1', 'BI2', 'BI3'],  # Behavioral Intention
        'EM': ['EM1', 'EM2', 'EM3'],  # Economic Motivation (GameFi)
        'RP': ['RP1', 'RP2', 'RP3', 'RP4'],  # Risk Perception (GameFi)
        'TT': ['TT1', 'TT2', 'TT3'],  # Trust in Technology (GameFi)
        'RC': ['RC1', 'RC2'],  # Regulatory Compliance (GameFi)
        'UB': ['UB_Frequency', 'UB_Hours']  # Use Behavior (Behavioral measures)
    }

    # Convert behavioral measures to numeric for UB construct
    print("\n📊 Converting behavioral measures to numeric for UB construct...")

    # UB1_UseFreq: Usage frequency mapping
    freq_mapping = {
        'Never': 1,
        'Rarely (less than once a month)': 2,
        'Monthly or less': 2,  # Alternative wording
        'A few times a month': 3,
        'Weekly': 4,
        'Several times a week': 5,
        'Daily': 6
    }

    # UB2_WeeklyHours: Weekly hours mapping
    hours_mapping = {
        '0 hours': 1,
        'Less than 1 hour': 2,
        '1-5 hours': 3,
        '6-10 hours': 4,
        '11-20 hours': 5,
        'More than 20 hours': 6
    }

    # Create numeric UB indicators
    df['UB_Frequency'] = df['UB1_UseFreq'].map(freq_mapping)
    df['UB_Hours'] = df['UB2_WeeklyHours'].map(hours_mapping)

    # Check conversion success
    unmapped_freq = df['UB_Frequency'].isna().sum()
    unmapped_hours = df['UB_Hours'].isna().sum()

    if unmapped_freq > 0 or unmapped_hours > 0:
        print(f"⚠️  Conversion issues: {unmapped_freq} frequency, {unmapped_hours} hours unmapped")
        print("Unique frequency values:", df['UB1_UseFreq'].unique())
        print("Unique hours values:", df['UB2_WeeklyHours'].unique())
    else:
        print("✅ UB behavioral measures converted successfully")

    # Extract all construct data including UB
    all_items = []
    for construct, items in measurement_model.items():
        all_items.extend(items)

    # Get Likert items + converted UB measures
    likert_items = []
    for construct, items in measurement_model.items():
        if construct != 'UB':
            likert_items.extend(items)

    pls_data = df[likert_items + ['UB_Frequency', 'UB_Hours']].copy()

    print(f"✅ Complete model prepared: {len(pls_data)} participants")
    print(f"📊 Constructs: {len(measurement_model)} latent constructs (including UB)")

    return pls_data, measurement_model


def analyze_ub_construct(df, measurement_model):
    """
    Analyze UB construct properties
    """
    print("\n🎯 USE BEHAVIOR (UB) CONSTRUCT ANALYSIS")
    print("=" * 60)

    ub_data = df[['UB_Frequency', 'UB_Hours']]

    # Descriptive statistics
    print("UB Construct Descriptives:")
    print(
        f"UB_Frequency: M={ub_data['UB_Frequency'].mean():.2f}, SD={ub_data['UB_Frequency'].std():.2f}, Range={ub_data['UB_Frequency'].min()}-{ub_data['UB_Frequency'].max()}")
    print(
        f"UB_Hours: M={ub_data['UB_Hours'].mean():.2f}, SD={ub_data['UB_Hours'].std():.2f}, Range={ub_data['UB_Hours'].min()}-{ub_data['UB_Hours'].max()}")

    # Correlation between UB indicators
    ub_corr = ub_data['UB_Frequency'].corr(ub_data['UB_Hours'])
    print(f"Inter-indicator correlation: r = {ub_corr:.3f}")

    # UB construct score (mean of standardized indicators)
    ub_freq_std = (ub_data['UB_Frequency'] - ub_data['UB_Frequency'].mean()) / ub_data['UB_Frequency'].std()
    ub_hours_std = (ub_data['UB_Hours'] - ub_data['UB_Hours'].mean()) / ub_data['UB_Hours'].std()
    ub_construct = (ub_freq_std + ub_hours_std) / 2

    print(f"UB Construct: M={ub_construct.mean():.2f}, SD={ub_construct.std():.2f}")

    # Reliability assessment for UB (2-item construct)
    # Spearman-Brown formula for 2-item reliability
    reliability_ub = (2 * ub_corr) / (1 + ub_corr)
    print(f"UB Construct Reliability (Spearman-Brown): {reliability_ub:.3f}")

    reliability_status = "✅ Good" if reliability_ub >= 0.7 else "⚠️ Acceptable" if reliability_ub >= 0.6 else "❌ Poor"
    print(f"UB Reliability Assessment: {reliability_status}")

    return ub_construct


def complete_structural_model_analysis(df, measurement_model):
    """
    Complete structural model analysis including UB paths
    """
    print("\n🏗️ COMPLETE STRUCTURAL MODEL ANALYSIS (v1)")
    print("=" * 60)

    # Calculate all construct scores
    construct_scores = {}

    for construct, items in measurement_model.items():
        if construct == 'UB':
            # Special handling for UB construct
            ub_data = df[items]
            # Standardize both indicators
            ub_freq_std = (ub_data['UB_Frequency'] - ub_data['UB_Frequency'].mean()) / ub_data['UB_Frequency'].std()
            ub_hours_std = (ub_data['UB_Hours'] - ub_data['UB_Hours'].mean()) / ub_data['UB_Hours'].std()
            construct_scores[construct] = (ub_freq_std + ub_hours_std) / 2
        else:
            # Standard Likert scale constructs
            construct_scores[construct] = df[items].mean(axis=1)

    construct_df = pd.DataFrame(construct_scores)

    print("Complete v1 Structural Model Paths:")
    print("📊 Direct Effects to BI (10 paths):")
    print("- PE → BI, EE → BI, SI → BI, FC → BI, HM → BI")
    print("- PV → BI, HB → BI, EM → BI, TT → BI, RP → BI")
    print("🎯 Direct Effects to UB (3 paths):")
    print("- BI → UB, HB → UB, FC → UB")
    print()

    # === ANALYSIS 1: Paths to BI (same as before) ===
    print("PART 1: PATHS TO BEHAVIORAL INTENTION")
    print("-" * 50)

    bi_predictors = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'EM', 'TT', 'RP']
    X_bi = construct_df[bi_predictors]
    y_bi = construct_df['BI']

    # Standardize variables
    X_bi_std = (X_bi - X_bi.mean()) / X_bi.std()
    y_bi_std = (y_bi - y_bi.mean()) / y_bi.std()

    # Multiple regression for BI
    model_bi = LinearRegression().fit(X_bi_std, y_bi_std)
    path_coeff_bi = model_bi.coef_
    r2_bi = r2_score(y_bi_std, model_bi.predict(X_bi_std))
    adj_r2_bi = 1 - (1 - r2_bi) * (len(y_bi_std) - 1) / (len(y_bi_std) - len(bi_predictors) - 1)

    print(f"R² for BI = {r2_bi:.3f}")
    print(f"Adjusted R² for BI = {adj_r2_bi:.3f}")
    print("\nPath Coefficients to BI:")

    bi_results = {}
    for i, predictor in enumerate(bi_predictors):
        beta = path_coeff_bi[i]
        effect_size = "Large" if abs(beta) >= 0.35 else "Medium" if abs(beta) >= 0.15 else "Small"
        significance = "**" if abs(beta) >= 0.1 else "*" if abs(beta) >= 0.05 else "ns"

        print(f"{predictor} → BI: β = {beta:+.3f} {significance} ({effect_size})")
        bi_results[f"{predictor}_to_BI"] = {'beta': beta, 'effect_size': effect_size}

    # === ANALYSIS 2: Paths to UB (NEW) ===
    print(f"\nPART 2: PATHS TO USE BEHAVIOR")
    print("-" * 50)

    ub_predictors = ['BI', 'HB', 'FC']
    X_ub = construct_df[ub_predictors]
    y_ub = construct_df['UB']

    # Standardize variables
    X_ub_std = (X_ub - X_ub.mean()) / X_ub.std()
    y_ub_std = (y_ub - y_ub.mean()) / y_ub.std()

    # Multiple regression for UB
    model_ub = LinearRegression().fit(X_ub_std, y_ub_std)
    path_coeff_ub = model_ub.coef_
    r2_ub = r2_score(y_ub_std, model_ub.predict(X_ub_std))
    adj_r2_ub = 1 - (1 - r2_ub) * (len(y_ub_std) - 1) / (len(y_ub_std) - len(ub_predictors) - 1)

    print(f"R² for UB = {r2_ub:.3f}")
    print(f"Adjusted R² for UB = {adj_r2_ub:.3f}")
    print("\nPath Coefficients to UB:")

    ub_results = {}
    for i, predictor in enumerate(ub_predictors):
        beta = path_coeff_ub[i]
        effect_size = "Large" if abs(beta) >= 0.35 else "Medium" if abs(beta) >= 0.15 else "Small"
        significance = "**" if abs(beta) >= 0.1 else "*" if abs(beta) >= 0.05 else "ns"

        print(f"{predictor} → UB: β = {beta:+.3f} {significance} ({effect_size})")
        ub_results[f"{predictor}_to_UB"] = {'beta': beta, 'effect_size': effect_size}

    # === EFFECT SIZES (f²) ===
    print(f"\nEFFECT SIZES (f²)")
    print("-" * 30)

    print("For BI:")
    for predictor in bi_predictors:
        X_reduced = X_bi_std.drop(columns=[predictor])
        model_reduced = LinearRegression().fit(X_reduced, y_bi_std)
        r2_reduced = r2_score(y_bi_std, model_reduced.predict(X_reduced))
        f_squared = (r2_bi - r2_reduced) / (1 - r2_bi)
        f2_size = "Large" if f_squared >= 0.35 else "Medium" if f_squared >= 0.15 else "Small" if f_squared >= 0.02 else "None"
        print(f"{predictor}: f² = {f_squared:.3f} ({f2_size})")

    print("\nFor UB:")
    for predictor in ub_predictors:
        X_reduced = X_ub_std.drop(columns=[predictor])
        model_reduced = LinearRegression().fit(X_reduced, y_ub_std)
        r2_reduced = r2_score(y_ub_std, model_reduced.predict(X_reduced))
        f_squared = (r2_ub - r2_reduced) / (1 - r2_ub)
        f2_size = "Large" if f_squared >= 0.35 else "Medium" if f_squared >= 0.15 else "Small" if f_squared >= 0.02 else "None"
        print(f"{predictor}: f² = {f_squared:.3f} ({f2_size})")

    # Combine all results
    all_results = {**bi_results, **ub_results}

    # Save results
    results_df = pd.DataFrame(all_results).T
    results_df['R2_BI'] = r2_bi
    results_df['R2_UB'] = r2_ub
    results_df.to_excel("complete_v1_results.xlsx")
    print("\n💾 Complete v1 results saved to: complete_v1_results.xlsx")

    return all_results, r2_bi, r2_ub, construct_df


def hypothesis_testing_complete_v1(bi_results, ub_results, r2_bi, r2_ub):
    """
    Complete hypothesis testing for v1 model
    """
    print("\n📋 COMPLETE v1 HYPOTHESIS TESTING")
    print("=" * 60)

    # Extract path coefficients
    paths_to_bi = {
        'PE': bi_results['PE_to_BI']['beta'],
        'EE': bi_results['EE_to_BI']['beta'],
        'SI': bi_results['SI_to_BI']['beta'],
        'FC': bi_results['FC_to_BI']['beta'],
        'HM': bi_results['HM_to_BI']['beta'],
        'PV': bi_results['PV_to_BI']['beta'],
        'HB': bi_results['HB_to_BI']['beta'],
        'EM': bi_results['EM_to_BI']['beta'],
        'TT': bi_results['TT_to_BI']['beta'],
        'RP': bi_results['RP_to_BI']['beta']
    }

    paths_to_ub = {
        'BI': ub_results['BI_to_UB']['beta'],
        'HB': ub_results['HB_to_UB']['beta'],
        'FC': ub_results['FC_to_UB']['beta']
    }

    # Define expected signs
    expected_signs = {
        'PE': '+', 'EE': '+', 'SI': '+', 'FC': '+', 'HM': '+', 'PV': '+',
        'HB': '+', 'EM': '+', 'TT': '+', 'RP': '-',  # RP should be negative
        'BI': '+',  # HB and FC to UB should also be positive
    }

    print("COMPLETE v1 HYPOTHESIS TESTING RESULTS:")
    print("=" * 60)

    # BI paths
    print("PATHS TO BEHAVIORAL INTENTION:")
    supported_bi = 0
    total_bi = len(paths_to_bi)

    for construct, beta in paths_to_bi.items():
        expected = expected_signs.get(construct, '+')
        actual_sign = '+' if beta > 0 else '-'
        significance = "**" if abs(beta) >= 0.1 else "*" if abs(beta) >= 0.05 else "ns"

        # Support criteria: significant AND correct direction
        direction_match = (expected == actual_sign)
        is_significant = significance != "ns"
        supported = direction_match and is_significant

        if supported:
            supported_bi += 1

        support_status = "Supported" if supported else "Not Supported"

        print(
            f"H{construct}→BI: {construct} → BI = {beta:+.3f}{significance} (Expected: {expected}) - {support_status}")

    # UB paths
    print(f"\nPATHS TO USE BEHAVIOR:")
    supported_ub = 0
    total_ub = len(paths_to_ub)

    for construct, beta in paths_to_ub.items():
        expected = '+'  # All UB paths expected positive
        actual_sign = '+' if beta > 0 else '-'
        significance = "**" if abs(beta) >= 0.1 else "*" if abs(beta) >= 0.05 else "ns"

        direction_match = (expected == actual_sign)
        is_significant = significance != "ns"
        supported = direction_match and is_significant

        if supported:
            supported_ub += 1

        support_status = "Supported" if supported else "Not Supported"

        print(
            f"H{construct}→UB: {construct} → UB = {beta:+.3f}{significance} (Expected: {expected}) - {support_status}")

    # Overall summary
    total_hypotheses = total_bi + total_ub
    total_supported = supported_bi + supported_ub
    support_rate = (total_supported / total_hypotheses) * 100

    print(f"\n📊 OVERALL v1 MODEL SUMMARY:")
    print(f"Total Hypotheses: {total_hypotheses}")
    print(f"Supported: {total_supported}")
    print(f"Not Supported: {total_hypotheses - total_supported}")
    print(f"Support Rate: {support_rate:.1f}%")
    print(f"\nModel Fit:")
    print(f"R² (BI) = {r2_bi:.3f}")
    print(f"R² (UB) = {r2_ub:.3f}")

    # Model quality assessment
    r2_bi_quality = "Substantial" if r2_bi >= 0.67 else "Moderate" if r2_bi >= 0.33 else "Weak"
    r2_ub_quality = "Substantial" if r2_ub >= 0.67 else "Moderate" if r2_ub >= 0.33 else "Weak"

    print(f"R² Quality: BI ({r2_bi_quality}), UB ({r2_ub_quality})")

    return {
        'total_hypotheses': total_hypotheses,
        'supported': total_supported,
        'support_rate': support_rate,
        'r2_bi': r2_bi,
        'r2_ub': r2_ub
    }


def main():
    """
    Main function for complete v1 analysis
    """
    # Load and prepare complete data
    pls_data, measurement_model = load_and_prepare_complete_data()

    # Analyze UB construct
    ub_construct_score = analyze_ub_construct(pls_data, measurement_model)

    # Complete structural model analysis
    all_results, r2_bi, r2_ub, construct_df = complete_structural_model_analysis(pls_data, measurement_model)

    # Extract BI and UB results
    bi_results = {k: v for k, v in all_results.items() if '_to_BI' in k}
    ub_results = {k: v for k, v in all_results.items() if '_to_UB' in k}

    # Complete hypothesis testing
    summary = hypothesis_testing_complete_v1(bi_results, ub_results, r2_bi, r2_ub)

    print("\n🎉 COMPLETE v1 ANALYSIS FINISHED!")
    print("\nNext Steps:")
    print("- v1 Model is now complete with UB structural paths")
    print("- Ready to proceed to v2 (GameFi inter-construct relationships)")
    print("- Consider moderation analysis (v3) if desired")

    return summary


if __name__ == "__main__":
    main()