import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def calculate_reliability_metrics(data, construct_items):
    """Calculate comprehensive reliability metrics"""
    construct_data = data[construct_items].dropna()

    # Cronbach's Alpha
    n_items = len(construct_items)
    item_variances = construct_data.var(axis=0)
    total_variance = construct_data.sum(axis=1).var()

    cronbach_alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)

    # Composite Reliability (approximation)
    corr_matrix = construct_data.corr()
    avg_inter_item_corr = (corr_matrix.sum().sum() - n_items) / (n_items * (n_items - 1))
    composite_reliability = (n_items * avg_inter_item_corr) / (1 + (n_items - 1) * avg_inter_item_corr)

    # AVE (approximation)
    ave = avg_inter_item_corr / (avg_inter_item_corr + (1 - avg_inter_item_corr) / n_items)

    return cronbach_alpha, composite_reliability, ave


def run_v2a_analysis():
    print("🚀 Starting UTAUT2 v2a Analysis - Complete Core Structural Model...")

    # Load cleaned data
    try:
        data = pd.read_excel('utaut2_cleaned_data.xlsx')
        print(f"✅ Data loaded: {data.shape[0]} participants × {data.shape[1]} variables")
    except FileNotFoundError:
        print("❌ Error: utaut2_cleaned_data.xlsx not found!")
        print("Please run the data cleaning script first.")
        return

    # Define construct items
    construct_items = {
        'PE': ['PE1', 'PE2', 'PE3', 'PE4', 'PE5'],
        'EE': ['EE1', 'EE2', 'EE3', 'EE4'],
        'SI': ['SI1', 'SI2', 'SI3'],
        'FC': ['FC1', 'FC2', 'FC3', 'FC4'],
        'HM': ['HM1', 'HM2', 'HM3', 'HM4'],
        'PV': ['PV1', 'PV2', 'PV3'],
        'HB': ['HB1', 'HB2', 'HB3', 'HB4'],
        'BI': ['BI1', 'BI2', 'BI3'],
        'EM': ['EM1', 'EM2', 'EM3'],
        'RP': ['RP1', 'RP2', 'RP3', 'RP4'],
        'TT': ['TT1', 'TT2', 'TT3'],
        'RC': ['RC1', 'RC2']
    }

    # Create construct scores (mean of items)
    print("\n📊 Computing construct scores...")
    for construct, items in construct_items.items():
        available_items = [item for item in items if item in data.columns]
        if available_items:
            data[construct] = data[available_items].mean(axis=1)
            print(f"✅ {construct}: {len(available_items)} items, M={data[construct].mean():.3f}")

    # Convert behavioral measures for UB construct
    print("\n🔄 Converting behavioral measures for UB construct...")

    # Convert frequency categories to numeric (1-6 scale)
    freq_mapping = {
        'Never': 1,
        'Monthly or less': 2,
        'A few times a month': 3,
        'Weekly': 4,
        'Several times a week': 5,
        'Daily': 6
    }

    # Convert hours categories to numeric (1-6 scale)
    hours_mapping = {
        'Less than 1 hour': 1,
        '1-5 hours': 2,
        '6-10 hours': 3,
        '11-15 hours': 4,
        '16-20 hours': 5,
        'More than 20 hours': 6
    }

    data['UB_Frequency_Num'] = data['UB1_UseFreq'].map(freq_mapping)
    data['UB_Hours_Num'] = data['UB2_WeeklyHours'].map(hours_mapping)

    # Create composite UB score (standardized mean)
    ub_data = data[['UB_Frequency_Num', 'UB_Hours_Num']].dropna()
    scaler = StandardScaler()
    ub_standardized = scaler.fit_transform(ub_data)
    data['UB'] = np.nan
    data.loc[ub_data.index, 'UB'] = np.mean(ub_standardized, axis=1)

    print(f"✅ UB construct: M={data['UB'].mean():.3f}, SD={data['UB'].std():.3f}")

    # v2a STRUCTURAL MODEL - Complete Core Model with RC → BI
    print("\n🗂️ v2a STRUCTURAL MODEL ANALYSIS")
    print("=" * 60)
    print("📋 v2a Model: All v1 paths + RC → BI (complete core model)")

    # Prepare data for structural analysis
    constructs = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'BI', 'EM', 'RP', 'TT', 'RC', 'UB']
    model_data = data[constructs].dropna()

    print(f"✅ Model data: {len(model_data)} complete cases")

    # PART 1: PATHS TO BEHAVIORAL INTENTION (11 paths - added RC → BI)
    print("\nPART 1: PATHS TO BEHAVIORAL INTENTION")
    print("-" * 50)

    # Define predictors for BI (now includes RC)
    bi_predictors = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'EM', 'TT', 'RP', 'RC']

    # Fit regression model for BI
    X_bi = model_data[bi_predictors]
    y_bi = model_data['BI']

    bi_model = LinearRegression().fit(X_bi, y_bi)
    bi_predictions = bi_model.predict(X_bi)

    # Calculate R²
    ss_tot = np.sum((y_bi - np.mean(y_bi)) ** 2)
    ss_res = np.sum((y_bi - bi_predictions) ** 2)
    r2_bi = 1 - (ss_res / ss_tot)
    adjusted_r2_bi = 1 - (1 - r2_bi) * (len(y_bi) - 1) / (len(y_bi) - len(bi_predictors) - 1)

    print(f"R² for BI = {r2_bi:.3f}")
    print(f"Adjusted R² for BI = {adjusted_r2_bi:.3f}")

    # Calculate path coefficients and significance
    print(f"\nv2a Path Coefficients to BI:")

    results_bi = []

    for i, predictor in enumerate(bi_predictors):
        beta = bi_model.coef_[i]

        # Calculate t-statistic and p-value (approximation)
        residuals = y_bi - bi_predictions
        mse = np.sum(residuals ** 2) / (len(y_bi) - len(bi_predictors) - 1)
        var_coef = mse * np.linalg.inv(X_bi.T @ X_bi)[i, i]
        se_coef = np.sqrt(var_coef)
        t_stat = beta / se_coef
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(y_bi) - len(bi_predictors) - 1))

        # Effect size classification
        abs_beta = abs(beta)
        if abs_beta >= 0.35:
            effect_size = "Large"
        elif abs_beta >= 0.15:
            effect_size = "Medium"
        elif abs_beta >= 0.02:
            effect_size = "Small"
        else:
            effect_size = "None"

        # Significance symbols
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"

        print(f"{predictor} → BI: β = {beta:+.3f}{sig} ({effect_size})")

        results_bi.append({
            'Path': f'{predictor}_to_BI',
            'Beta': beta,
            'SE': se_coef,
            'T_stat': t_stat,
            'P_value': p_value,
            'Significance': sig,
            'Effect_Size': effect_size,
            'Expected_Direction': '+' if predictor != 'RP' else '-',
            'R2_BI': r2_bi,
            'R2_UB': None
        })

    # Calculate effect sizes (f²) for BI
    print(f"\nEffect Sizes (f²) for BI:")
    for i, predictor in enumerate(bi_predictors):
        # Remove predictor and calculate R² change
        other_predictors = [p for p in bi_predictors if p != predictor]
        X_reduced = model_data[other_predictors]
        reduced_model = LinearRegression().fit(X_reduced, y_bi)
        reduced_predictions = reduced_model.predict(X_reduced)
        ss_res_reduced = np.sum((y_bi - reduced_predictions) ** 2)
        r2_reduced = 1 - (ss_res_reduced / ss_tot)

        f_squared = (r2_bi - r2_reduced) / (1 - r2_bi)

        if f_squared >= 0.35:
            f_effect = "Large"
        elif f_squared >= 0.15:
            f_effect = "Medium"
        elif f_squared >= 0.02:
            f_effect = "Small"
        else:
            f_effect = "None"

        print(f"{predictor}: f² = {f_squared:.3f} ({f_effect})")

    # PART 2: PATHS TO USE BEHAVIOR (unchanged from v1)
    print("\nPART 2: PATHS TO USE BEHAVIOR")
    print("-" * 50)

    ub_predictors = ['BI', 'HB', 'FC']

    # Fit regression model for UB
    X_ub = model_data[ub_predictors]
    y_ub = model_data['UB']

    ub_model = LinearRegression().fit(X_ub, y_ub)
    ub_predictions = ub_model.predict(X_ub)

    # Calculate R²
    ss_tot_ub = np.sum((y_ub - np.mean(y_ub)) ** 2)
    ss_res_ub = np.sum((y_ub - ub_predictions) ** 2)
    r2_ub = 1 - (ss_res_ub / ss_tot_ub)
    adjusted_r2_ub = 1 - (1 - r2_ub) * (len(y_ub) - 1) / (len(y_ub) - len(ub_predictors) - 1)

    print(f"R² for UB = {r2_ub:.3f}")
    print(f"Adjusted R² for UB = {adjusted_r2_ub:.3f}")

    print(f"\nPath Coefficients to UB:")

    results_ub = []

    for i, predictor in enumerate(ub_predictors):
        beta = ub_model.coef_[i]

        # Calculate t-statistic and p-value
        residuals_ub = y_ub - ub_predictions
        mse_ub = np.sum(residuals_ub ** 2) / (len(y_ub) - len(ub_predictors) - 1)
        var_coef_ub = mse_ub * np.linalg.inv(X_ub.T @ X_ub)[i, i]
        se_coef_ub = np.sqrt(var_coef_ub)
        t_stat_ub = beta / se_coef_ub
        p_value_ub = 2 * (1 - stats.t.cdf(np.abs(t_stat_ub), len(y_ub) - len(ub_predictors) - 1))

        # Effect size classification
        abs_beta = abs(beta)
        if abs_beta >= 0.35:
            effect_size = "Large"
        elif abs_beta >= 0.15:
            effect_size = "Medium"
        elif abs_beta >= 0.02:
            effect_size = "Small"
        else:
            effect_size = "None"

        # Significance symbols
        if p_value_ub < 0.001:
            sig = "***"
        elif p_value_ub < 0.01:
            sig = "**"
        elif p_value_ub < 0.05:
            sig = "*"
        else:
            sig = "ns"

        print(f"{predictor} → UB: β = {beta:+.3f}{sig} ({effect_size})")

        results_ub.append({
            'Path': f'{predictor}_to_UB',
            'Beta': beta,
            'SE': se_coef_ub,
            'T_stat': t_stat_ub,
            'P_value': p_value_ub,
            'Significance': sig,
            'Effect_Size': effect_size,
            'Expected_Direction': '+',
            'R2_BI': r2_bi,
            'R2_UB': r2_ub
        })

    # Combine all results
    all_results = results_bi + results_ub
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_excel('complete_v2a_results.xlsx', index=False)
    print(f"\n💾 v2a results saved to: complete_v2a_results.xlsx")

    # HYPOTHESIS TESTING SUMMARY
    print("\n📋 v2a HYPOTHESIS TESTING RESULTS")
    print("=" * 60)
    print("PATHS TO BEHAVIORAL INTENTION (11 paths):")

    expected_directions = {
        'PE': '+', 'EE': '+', 'SI': '+', 'FC': '+', 'HM': '+',
        'PV': '+', 'HB': '+', 'EM': '+', 'TT': '+', 'RP': '-', 'RC': '+'
    }

    supported_bi = 0
    total_bi = 0

    for result in results_bi:
        predictor = result['Path'].split('_to_')[0]
        expected = expected_directions[predictor]
        actual_direction = '+' if result['Beta'] > 0 else '-'
        is_significant = result['Significance'] != 'ns'

        if expected == actual_direction and is_significant:
            support_status = "Supported"
            supported_bi += 1
        elif expected == actual_direction and not is_significant:
            support_status = "Not Supported (ns)"
        else:
            support_status = "Not Supported"

        total_bi += 1

        print(
            f"H{predictor}→BI: {predictor} → BI = {result['Beta']:+.3f}{result['Significance']} (Expected: {expected}) - {support_status}")

    print(f"\nPATHS TO USE BEHAVIOR (3 paths):")
    supported_ub = 0
    total_ub = 0

    for result in results_ub:
        predictor = result['Path'].split('_to_')[0]
        expected = '+'
        actual_direction = '+' if result['Beta'] > 0 else '-'
        is_significant = result['Significance'] != 'ns'

        if expected == actual_direction and is_significant:
            support_status = "Supported"
            supported_ub += 1
        elif expected == actual_direction and not is_significant:
            support_status = "Not Supported (ns)"
        else:
            support_status = "Not Supported"

        total_ub += 1

        print(
            f"H{predictor}→UB: {predictor} → UB = {result['Beta']:+.3f}{result['Significance']} (Expected: {expected}) - {support_status}")

    # OVERALL MODEL SUMMARY
    total_hypotheses = total_bi + total_ub
    total_supported = supported_bi + supported_ub
    support_rate = (total_supported / total_hypotheses) * 100

    print(f"\n📊 OVERALL v2a MODEL SUMMARY:")
    print(f"Total Hypotheses: {total_hypotheses}")
    print(f"Supported: {total_supported}")
    print(f"Not Supported: {total_hypotheses - total_supported}")
    print(f"Support Rate: {support_rate:.1f}%")
    print(f"\nModel Fit:")
    print(f"R² (BI) = {r2_bi:.3f}")
    print(f"R² (UB) = {r2_ub:.3f}")

    # Compare to v1 results
    print(f"\n🔄 COMPARISON TO v1:")
    print(f"v1 had 10 paths to BI, v2a has 11 paths to BI")
    print(f"Key addition: RC → BI path")
    print(f"v1 R² (BI): ~0.712")
    print(f"v2a R² (BI): {r2_bi:.3f}")
    print(f"R² Change: {r2_bi - 0.712:+.3f}")

    if r2_bi > 0.712:
        print("✅ Adding RC → BI improved model fit!")
    else:
        print("ℹ️ Adding RC → BI had minimal impact on model fit")

    print(f"\n🎉 v2a ANALYSIS COMPLETE!")
    print(f"\nNext Steps:")
    print(f"- Review RC → BI path significance and direction")
    print(f"- Compare v2a vs v1 model performance")
    print(f"- Ready for v2b (GameFi inter-construct relationships)")


if __name__ == "__main__":
    run_v2a_analysis()