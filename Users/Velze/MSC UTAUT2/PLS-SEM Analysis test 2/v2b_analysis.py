import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def run_v2b_analysis():
    print("🚀 Starting UTAUT2 v2b Analysis - GameFi Inter-Construct Relationships...")

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

    # v2b STRUCTURAL MODEL - Complete Model + GameFi Inter-Construct Links
    print("\n🗂️ v2b STRUCTURAL MODEL ANALYSIS")
    print("=" * 70)
    print("📋 v2b Model: v2a paths + GameFi inter-construct relationships")
    print("   New paths: TT → RP (−), RC → TT (+), RC → RP (−)")

    # Prepare data for structural analysis
    constructs = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'BI', 'EM', 'RP', 'TT', 'RC', 'UB']
    model_data = data[constructs].dropna()

    print(f"✅ Model data: {len(model_data)} complete cases")

    # Store all results for comparison
    all_results = []

    # PART 1: GAMEFI INTER-CONSTRUCT RELATIONSHIPS (NEW IN v2b)
    print("\nPART 1: GAMEFI INTER-CONSTRUCT RELATIONSHIPS (NEW)")
    print("-" * 60)

    # Path 1: TT → RP (negative expected)
    print("Path 1: Trust in Technology → Risk Perception")
    X_tt_rp = model_data[['TT']].values.reshape(-1, 1)
    y_rp = model_data['RP']

    tt_rp_model = LinearRegression().fit(X_tt_rp, y_rp)
    tt_rp_pred = tt_rp_model.predict(X_tt_rp)

    # Calculate R² for TT → RP
    ss_tot_rp_tt = np.sum((y_rp - np.mean(y_rp)) ** 2)
    ss_res_rp_tt = np.sum((y_rp - tt_rp_pred) ** 2)
    r2_tt_rp = 1 - (ss_res_rp_tt / ss_tot_rp_tt)

    beta_tt_rp = tt_rp_model.coef_[0]

    # Calculate significance
    residuals_tt_rp = y_rp - tt_rp_pred
    mse_tt_rp = np.sum(residuals_tt_rp ** 2) / (len(y_rp) - 2)
    var_coef_tt_rp = mse_tt_rp / np.sum((X_tt_rp.flatten() - np.mean(X_tt_rp)) ** 2)
    se_coef_tt_rp = np.sqrt(var_coef_tt_rp)
    t_stat_tt_rp = beta_tt_rp / se_coef_tt_rp
    p_value_tt_rp = 2 * (1 - stats.t.cdf(np.abs(t_stat_tt_rp), len(y_rp) - 2))

    # Significance and effect size
    if p_value_tt_rp < 0.001:
        sig_tt_rp = "***"
    elif p_value_tt_rp < 0.01:
        sig_tt_rp = "**"
    elif p_value_tt_rp < 0.05:
        sig_tt_rp = "*"
    else:
        sig_tt_rp = "ns"

    abs_beta_tt_rp = abs(beta_tt_rp)
    if abs_beta_tt_rp >= 0.35:
        effect_tt_rp = "Large"
    elif abs_beta_tt_rp >= 0.15:
        effect_tt_rp = "Medium"
    elif abs_beta_tt_rp >= 0.02:
        effect_tt_rp = "Small"
    else:
        effect_tt_rp = "None"

    print(f"TT → RP: β = {beta_tt_rp:+.3f}{sig_tt_rp} (Expected: −) ({effect_tt_rp})")
    print(f"R² (RP|TT) = {r2_tt_rp:.3f}")

    # Path 2: RC → TT (positive expected)
    print("\nPath 2: Regulatory Compliance → Trust in Technology")
    X_rc_tt = model_data[['RC']].values.reshape(-1, 1)
    y_tt = model_data['TT']

    rc_tt_model = LinearRegression().fit(X_rc_tt, y_tt)
    rc_tt_pred = rc_tt_model.predict(X_rc_tt)

    # Calculate R²
    ss_tot_tt_rc = np.sum((y_tt - np.mean(y_tt)) ** 2)
    ss_res_tt_rc = np.sum((y_tt - rc_tt_pred) ** 2)
    r2_rc_tt = 1 - (ss_res_tt_rc / ss_tot_tt_rc)

    beta_rc_tt = rc_tt_model.coef_[0]

    # Calculate significance
    residuals_rc_tt = y_tt - rc_tt_pred
    mse_rc_tt = np.sum(residuals_rc_tt ** 2) / (len(y_tt) - 2)
    var_coef_rc_tt = mse_rc_tt / np.sum((X_rc_tt.flatten() - np.mean(X_rc_tt)) ** 2)
    se_coef_rc_tt = np.sqrt(var_coef_rc_tt)
    t_stat_rc_tt = beta_rc_tt / se_coef_rc_tt
    p_value_rc_tt = 2 * (1 - stats.t.cdf(np.abs(t_stat_rc_tt), len(y_tt) - 2))

    if p_value_rc_tt < 0.001:
        sig_rc_tt = "***"
    elif p_value_rc_tt < 0.01:
        sig_rc_tt = "**"
    elif p_value_rc_tt < 0.05:
        sig_rc_tt = "*"
    else:
        sig_rc_tt = "ns"

    abs_beta_rc_tt = abs(beta_rc_tt)
    if abs_beta_rc_tt >= 0.35:
        effect_rc_tt = "Large"
    elif abs_beta_rc_tt >= 0.15:
        effect_rc_tt = "Medium"
    elif abs_beta_rc_tt >= 0.02:
        effect_rc_tt = "Small"
    else:
        effect_rc_tt = "None"

    print(f"RC → TT: β = {beta_rc_tt:+.3f}{sig_rc_tt} (Expected: +) ({effect_rc_tt})")
    print(f"R² (TT|RC) = {r2_rc_tt:.3f}")

    # Path 3: RC → RP (negative expected)
    print("\nPath 3: Regulatory Compliance → Risk Perception")
    X_rc_rp = model_data[['RC']].values.reshape(-1, 1)
    y_rp_rc = model_data['RP']

    rc_rp_model = LinearRegression().fit(X_rc_rp, y_rp_rc)
    rc_rp_pred = rc_rp_model.predict(X_rc_rp)

    # Calculate R²
    ss_tot_rp_rc = np.sum((y_rp_rc - np.mean(y_rp_rc)) ** 2)
    ss_res_rp_rc = np.sum((y_rp_rc - rc_rp_pred) ** 2)
    r2_rc_rp = 1 - (ss_res_rp_rc / ss_tot_rp_rc)

    beta_rc_rp = rc_rp_model.coef_[0]

    # Calculate significance
    residuals_rc_rp = y_rp_rc - rc_rp_pred
    mse_rc_rp = np.sum(residuals_rc_rp ** 2) / (len(y_rp_rc) - 2)
    var_coef_rc_rp = mse_rc_rp / np.sum((X_rc_rp.flatten() - np.mean(X_rc_rp)) ** 2)
    se_coef_rc_rp = np.sqrt(var_coef_rc_rp)
    t_stat_rc_rp = beta_rc_rp / se_coef_rc_rp
    p_value_rc_rp = 2 * (1 - stats.t.cdf(np.abs(t_stat_rc_rp), len(y_rp_rc) - 2))

    if p_value_rc_rp < 0.001:
        sig_rc_rp = "***"
    elif p_value_rc_rp < 0.01:
        sig_rc_rp = "**"
    elif p_value_rc_rp < 0.05:
        sig_rc_rp = "*"
    else:
        sig_rc_rp = "ns"

    abs_beta_rc_rp = abs(beta_rc_rp)
    if abs_beta_rc_rp >= 0.35:
        effect_rc_rp = "Large"
    elif abs_beta_rc_rp >= 0.15:
        effect_rc_rp = "Medium"
    elif abs_beta_rc_rp >= 0.02:
        effect_rc_rp = "Small"
    else:
        effect_rc_rp = "None"

    print(f"RC → RP: β = {beta_rc_rp:+.3f}{sig_rc_rp} (Expected: −) ({effect_rc_rp})")
    print(f"R² (RP|RC) = {r2_rc_rp:.3f}")

    # Store GameFi inter-construct results
    gamefi_results = [
        {
            'Path': 'TT_to_RP',
            'Beta': beta_tt_rp,
            'SE': se_coef_tt_rp,
            'T_stat': t_stat_tt_rp,
            'P_value': p_value_tt_rp,
            'Significance': sig_tt_rp,
            'Effect_Size': effect_tt_rp,
            'Expected_Direction': '-',
            'R2': r2_tt_rp,
            'Type': 'GameFi_Inter_Construct'
        },
        {
            'Path': 'RC_to_TT',
            'Beta': beta_rc_tt,
            'SE': se_coef_rc_tt,
            'T_stat': t_stat_rc_tt,
            'P_value': p_value_rc_tt,
            'Significance': sig_rc_tt,
            'Effect_Size': effect_rc_tt,
            'Expected_Direction': '+',
            'R2': r2_rc_tt,
            'Type': 'GameFi_Inter_Construct'
        },
        {
            'Path': 'RC_to_RP',
            'Beta': beta_rc_rp,
            'SE': se_coef_rc_rp,
            'T_stat': t_stat_rc_rp,
            'P_value': p_value_rc_rp,
            'Significance': sig_rc_rp,
            'Effect_Size': effect_rc_rp,
            'Expected_Direction': '-',
            'R2': r2_rc_rp,
            'Type': 'GameFi_Inter_Construct'
        }
    ]

    all_results.extend(gamefi_results)

    # PART 2: ENHANCED STRUCTURAL MODEL WITH GAMEFI INTERACTIONS
    print("\n" + "=" * 70)
    print("PART 2: ENHANCED STRUCTURAL MODEL (v2a paths + GameFi effects)")
    print("=" * 70)

    # Enhanced BI model: Include potential mediated effects through GameFi constructs
    print("\nEnhanced Behavioral Intention Model:")
    print("Predictors: All v2a predictors + potential GameFi mediation effects")

    # All original predictors for BI
    bi_predictors = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'EM', 'TT', 'RP', 'RC']

    # Fit regression model for BI (same as v2a for consistency)
    X_bi = model_data[bi_predictors]
    y_bi = model_data['BI']

    bi_model = LinearRegression().fit(X_bi, y_bi)
    bi_predictions = bi_model.predict(X_bi)

    # Calculate R²
    ss_tot_bi = np.sum((y_bi - np.mean(y_bi)) ** 2)
    ss_res_bi = np.sum((y_bi - bi_predictions) ** 2)
    r2_bi = 1 - (ss_res_bi / ss_tot_bi)

    print(f"R² for BI (Enhanced Model) = {r2_bi:.3f}")

    # Enhanced UB model
    print("\nEnhanced Use Behavior Model:")
    ub_predictors = ['BI', 'HB', 'FC']

    X_ub = model_data[ub_predictors]
    y_ub = model_data['UB']

    ub_model = LinearRegression().fit(X_ub, y_ub)
    ub_predictions = ub_model.predict(X_ub)

    ss_tot_ub = np.sum((y_ub - np.mean(y_ub)) ** 2)
    ss_res_ub = np.sum((y_ub - ub_predictions) ** 2)
    r2_ub = 1 - (ss_res_ub / ss_tot_ub)

    print(f"R² for UB (Enhanced Model) = {r2_ub:.3f}")

    # MEDIATION ANALYSIS
    print("\n" + "=" * 70)
    print("PART 3: MEDIATION ANALYSIS")
    print("=" * 70)

    print("Testing potential mediation paths in GameFi context:")

    # Mediation 1: RC → TT → BI
    print("\nMediation 1: RC → TT → BI")
    # Direct effect: RC → BI (already calculated in v2a)
    direct_rc_bi = 0.018  # From v2a results

    # Indirect effect: RC → TT → BI
    # RC → TT effect: beta_rc_tt (calculated above)
    # TT → BI effect: -0.106 (from v2a results)
    tt_bi_effect = -0.106

    indirect_rc_tt_bi = beta_rc_tt * tt_bi_effect
    total_effect_rc_bi = direct_rc_bi + indirect_rc_tt_bi

    print(f"Direct effect RC → BI: {direct_rc_bi:+.3f}")
    print(f"Indirect effect RC → TT → BI: {indirect_rc_tt_bi:+.3f}")
    print(f"Total effect: {total_effect_rc_bi:+.3f}")

    # Mediation 2: RC → RP → BI
    print("\nMediation 2: RC → RP → BI")
    # RC → RP effect: beta_rc_rp (calculated above)
    # RP → BI effect: +0.097 (from v2a results)
    rp_bi_effect = 0.097

    indirect_rc_rp_bi = beta_rc_rp * rp_bi_effect
    total_effect_rc_bi_2 = direct_rc_bi + indirect_rc_rp_bi

    print(f"Direct effect RC → BI: {direct_rc_bi:+.3f}")
    print(f"Indirect effect RC → RP → BI: {indirect_rc_rp_bi:+.3f}")
    print(f"Total effect: {total_effect_rc_bi_2:+.3f}")

    # Mediation 3: TT → RP → BI
    print("\nMediation 3: TT → RP → BI")
    # Direct effect: TT → BI = -0.106
    direct_tt_bi = -0.106

    # Indirect effect: TT → RP → BI
    indirect_tt_rp_bi = beta_tt_rp * rp_bi_effect
    total_effect_tt_bi = direct_tt_bi + indirect_tt_rp_bi

    print(f"Direct effect TT → BI: {direct_tt_bi:+.3f}")
    print(f"Indirect effect TT → RP → BI: {indirect_tt_rp_bi:+.3f}")
    print(f"Total effect: {total_effect_tt_bi:+.3f}")

    # Compile complete results
    all_results_df = pd.DataFrame(all_results)

    # Add mediation results
    mediation_results = pd.DataFrame([
        {
            'Path': 'RC_TT_BI_mediation',
            'Direct_Effect': direct_rc_bi,
            'Indirect_Effect': indirect_rc_tt_bi,
            'Total_Effect': total_effect_rc_bi,
            'Type': 'Mediation'
        },
        {
            'Path': 'RC_RP_BI_mediation',
            'Direct_Effect': direct_rc_bi,
            'Indirect_Effect': indirect_rc_rp_bi,
            'Total_Effect': total_effect_rc_bi_2,
            'Type': 'Mediation'
        },
        {
            'Path': 'TT_RP_BI_mediation',
            'Direct_Effect': direct_tt_bi,
            'Indirect_Effect': indirect_tt_rp_bi,
            'Total_Effect': total_effect_tt_bi,
            'Type': 'Mediation'
        }
    ])

    # Save results
    all_results_df.to_excel('complete_v2b_gamefi_paths.xlsx', index=False)
    mediation_results.to_excel('v2b_mediation_analysis.xlsx', index=False)

    print(f"\n💾 v2b results saved to:")
    print(f"   - complete_v2b_gamefi_paths.xlsx")
    print(f"   - v2b_mediation_analysis.xlsx")

    # HYPOTHESIS TESTING FOR GAMEFI INTER-CONSTRUCT RELATIONSHIPS
    print("\n📋 v2b HYPOTHESIS TESTING - GAMEFI INTER-CONSTRUCTS")
    print("=" * 70)

    supported_gamefi = 0
    total_gamefi = 0

    # Test GameFi inter-construct hypotheses
    gamefi_expectations = {
        'TT_to_RP': '-',
        'RC_to_TT': '+',
        'RC_to_RP': '-'
    }

    for result in gamefi_results:
        path_name = result['Path']
        expected = result['Expected_Direction']
        actual_direction = '+' if result['Beta'] > 0 else '-'
        is_significant = result['Significance'] != 'ns'

        if expected == actual_direction and is_significant:
            support_status = "Supported"
            supported_gamefi += 1
        elif expected == actual_direction and not is_significant:
            support_status = "Not Supported (ns)"
        else:
            support_status = "Not Supported"

        total_gamefi += 1

        path_display = path_name.replace('_to_', ' → ')
        print(
            f"H{path_display}: {result['Beta']:+.3f}{result['Significance']} (Expected: {expected}) - {support_status}")

    # OVERALL v2b SUMMARY
    gamefi_support_rate = (supported_gamefi / total_gamefi) * 100 if total_gamefi > 0 else 0

    print(f"\n📊 v2b GAMEFI INTER-CONSTRUCT SUMMARY:")
    print(f"GameFi Hypotheses: {total_gamefi}")
    print(f"Supported: {supported_gamefi}")
    print(f"GameFi Support Rate: {gamefi_support_rate:.1f}%")

    print(f"\n📊 COMPLETE v2b MODEL SUMMARY:")
    print(f"Core Model (v2a): 11 BI paths + 3 UB paths = 14 paths")
    print(f"GameFi Inter-Construct: 3 new paths")
    print(f"Total v2b Paths Tested: 17 paths")
    print(f"\nModel Fit:")
    print(f"R² (BI) = {r2_bi:.3f}")
    print(f"R² (UB) = {r2_ub:.3f}")
    print(f"R² (TT|RC) = {r2_rc_tt:.3f}")
    print(f"R² (RP|TT) = {r2_tt_rp:.3f}")
    print(f"R² (RP|RC) = {r2_rc_rp:.3f}")

    # Key insights
    print(f"\n🔍 KEY v2b INSIGHTS:")

    if beta_tt_rp < 0 and sig_tt_rp != 'ns':
        print(f"✅ TT → RP: Trust reduces risk perception (as expected)")
    else:
        print(f"⚠️ TT → RP: Unexpected relationship or non-significant")

    if beta_rc_tt > 0 and sig_rc_tt != 'ns':
        print(f"✅ RC → TT: Regulatory compliance increases tech trust (as expected)")
    else:
        print(f"⚠️ RC → TT: Unexpected relationship or non-significant")

    if beta_rc_rp < 0 and sig_rc_rp != 'ns':
        print(f"✅ RC → RP: Regulatory compliance reduces risk perception (as expected)")
    else:
        print(f"⚠️ RC → RP: Unexpected relationship or non-significant")

    print(f"\n🎉 v2b ANALYSIS COMPLETE!")
    print(f"\nNext Steps:")
    print(f"- Interpret GameFi inter-construct relationships")
    print(f"- Analyze mediation effects for theoretical implications")
    print(f"- Consider v3 moderation analysis if desired")
    print(f"- Compare v2b theoretical insights vs v2a core model")


if __name__ == "__main__":
    run_v2b_analysis()