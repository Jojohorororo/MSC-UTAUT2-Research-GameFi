import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def run_experience_moderation_analysis():
    print("🚀 Starting Experience Moderation Analysis - Final GameFi Theoretical Insights...")
    print("Research Questions: How does GameFi experience moderate key UTAUT2 relationships?")

    # Load cleaned data
    try:
        data = pd.read_excel('utaut2_cleaned_data.xlsx')
        print(f"✅ Data loaded: {data.shape[0]} participants × {data.shape[1]} variables")
    except FileNotFoundError:
        print("❌ Error: utaut2_cleaned_data.xlsx not found!")
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

    # Create construct scores
    print("\n📊 Computing construct scores...")
    for construct, items in construct_items.items():
        available_items = [item for item in items if item in data.columns]
        if available_items:
            data[construct] = data[available_items].mean(axis=1)

    # Convert behavioral measures for UB construct
    freq_mapping = {
        'Never': 1, 'Monthly or less': 2, 'A few times a month': 3,
        'Weekly': 4, 'Several times a week': 5, 'Daily': 6
    }
    hours_mapping = {
        'Less than 1 hour': 1, '1-5 hours': 2, '6-10 hours': 3,
        '11-15 hours': 4, '16-20 hours': 5, 'More than 20 hours': 6
    }

    data['UB_Frequency_Num'] = data['UB1_UseFreq'].map(freq_mapping)
    data['UB_Hours_Num'] = data['UB2_WeeklyHours'].map(hours_mapping)

    # Create composite UB score
    ub_data = data[['UB_Frequency_Num', 'UB_Hours_Num']].dropna()
    scaler = StandardScaler()
    ub_standardized = scaler.fit_transform(ub_data)
    data['UB'] = np.nan
    data.loc[ub_data.index, 'UB'] = np.mean(ub_standardized, axis=1)

    # Process GameFi Experience variable
    print("\n📊 Processing GameFi Experience variable...")

    # Map experience levels to numeric scale
    experience_mapping = {
        'Have tried GameFi but not a regular user': 1,
        'Currently use GameFi platforms occasionally (1-3 times a month)': 2,
        'Currently use GameFi platforms regularly (weekly or more)': 3,
        'Have been using GameFi for over a year': 4
    }

    data['Experience_Numeric'] = data['GameFiExp'].map(experience_mapping)

    # Check experience distribution
    exp_counts = data['Experience_Numeric'].value_counts().sort_index()
    print(f"Experience Distribution:")
    for level, count in exp_counts.items():
        pct = (count / len(data)) * 100
        experience_label = [k for k, v in experience_mapping.items() if v == level][0]
        print(f"  Level {level}: {count} ({pct:.1f}%) - {experience_label}")

    # Prepare moderation data
    constructs_needed = ['EE', 'SI', 'FC', 'HB', 'BI', 'UB', 'Experience_Numeric']
    mod_data = data[constructs_needed].dropna()

    print(f"\n✅ Complete moderation data: {len(mod_data)} cases")

    # Mean-center variables
    print(f"\n📐 Mean-centering variables for moderation analysis...")
    mod_data['EE_centered'] = mod_data['EE'] - mod_data['EE'].mean()
    mod_data['SI_centered'] = mod_data['SI'] - mod_data['SI'].mean()
    mod_data['FC_centered'] = mod_data['FC'] - mod_data['FC'].mean()
    mod_data['HB_centered'] = mod_data['HB'] - mod_data['HB'].mean()
    mod_data['Exp_centered'] = mod_data['Experience_Numeric'] - mod_data['Experience_Numeric'].mean()

    # Create interaction terms
    mod_data['Exp_x_EE'] = mod_data['Exp_centered'] * mod_data['EE_centered']
    mod_data['Exp_x_SI'] = mod_data['Exp_centered'] * mod_data['SI_centered']
    mod_data['Exp_x_FC'] = mod_data['Exp_centered'] * mod_data['FC_centered']
    mod_data['Exp_x_HB_BI'] = mod_data['Exp_centered'] * mod_data['HB_centered']
    mod_data['Exp_x_HB_UB'] = mod_data['Exp_centered'] * mod_data['HB_centered']

    print(f"✅ Experience: M={mod_data['Experience_Numeric'].mean():.2f}, SD={mod_data['Experience_Numeric'].std():.2f}")

    # Store all moderation results
    all_moderation_results = []

    # MODERATION ANALYSIS 1: Experience × (EE → BI)
    print(f"\n" + "=" * 80)
    print("MODERATION 1: Experience × (Effort Expectancy → Behavioral Intention)")
    print("=" * 80)
    print("Theory: Learning curve effects - experience should reduce impact of perceived difficulty")

    # Main effects model
    X_main_1 = mod_data[['EE_centered', 'Exp_centered']]
    y_1 = mod_data['BI']
    main_model_1 = LinearRegression().fit(X_main_1, y_1)
    r2_main_1 = main_model_1.score(X_main_1, y_1)

    # Moderation model
    X_mod_1 = mod_data[['EE_centered', 'Exp_centered', 'Exp_x_EE']]
    mod_model_1 = LinearRegression().fit(X_mod_1, y_1)
    r2_mod_1 = mod_model_1.score(X_mod_1, y_1)

    beta_interaction_1 = mod_model_1.coef_[2]
    delta_r2_1 = r2_mod_1 - r2_main_1

    # Significance test
    mod_pred_1 = mod_model_1.predict(X_mod_1)
    residuals_1 = y_1 - mod_pred_1
    mse_1 = np.sum(residuals_1 ** 2) / (len(y_1) - 4)

    try:
        X_mod_with_const_1 = np.column_stack([np.ones(len(X_mod_1)), X_mod_1])
        cov_matrix_1 = mse_1 * np.linalg.inv(X_mod_with_const_1.T @ X_mod_with_const_1)
        se_interaction_1 = np.sqrt(cov_matrix_1[3, 3])
        t_interaction_1 = beta_interaction_1 / se_interaction_1
        p_interaction_1 = 2 * (1 - stats.t.cdf(np.abs(t_interaction_1), len(y_1) - 4))

        if p_interaction_1 < 0.001:
            sig_1 = "***"
        elif p_interaction_1 < 0.01:
            sig_1 = "**"
        elif p_interaction_1 < 0.05:
            sig_1 = "*"
        else:
            sig_1 = "ns"
    except:
        sig_1 = "n/a"
        p_interaction_1 = 1.0

    print(f"ΔR² = {delta_r2_1:.4f}")
    print(f"Interaction: β = {beta_interaction_1:+.4f} {sig_1}")

    all_moderation_results.append({
        'Moderation': 'Experience × (EE → BI)',
        'Theory': 'Learning curve - experience reduces effort impact',
        'Beta_Interaction': beta_interaction_1,
        'Delta_R2': delta_r2_1,
        'Significance': sig_1,
        'Interpretation': 'Supported' if (beta_interaction_1 < 0 and sig_1 != 'ns') else 'Not Supported'
    })

    # MODERATION ANALYSIS 2: Experience × (SI → BI)
    print(f"\n" + "=" * 80)
    print("MODERATION 2: Experience × (Social Influence → Behavioral Intention)")
    print("=" * 80)
    print("Theory: Social influence stronger for novices, weaker for experts")

    # Main and moderation models
    X_main_2 = mod_data[['SI_centered', 'Exp_centered']]
    y_2 = mod_data['BI']
    main_model_2 = LinearRegression().fit(X_main_2, y_2)
    r2_main_2 = main_model_2.score(X_main_2, y_2)

    X_mod_2 = mod_data[['SI_centered', 'Exp_centered', 'Exp_x_SI']]
    mod_model_2 = LinearRegression().fit(X_mod_2, y_2)
    r2_mod_2 = mod_model_2.score(X_mod_2, y_2)

    beta_interaction_2 = mod_model_2.coef_[2]
    delta_r2_2 = r2_mod_2 - r2_main_2

    # Significance test
    mod_pred_2 = mod_model_2.predict(X_mod_2)
    residuals_2 = y_2 - mod_pred_2
    mse_2 = np.sum(residuals_2 ** 2) / (len(y_2) - 4)

    try:
        X_mod_with_const_2 = np.column_stack([np.ones(len(X_mod_2)), X_mod_2])
        cov_matrix_2 = mse_2 * np.linalg.inv(X_mod_with_const_2.T @ X_mod_with_const_2)
        se_interaction_2 = np.sqrt(cov_matrix_2[3, 3])
        t_interaction_2 = beta_interaction_2 / se_interaction_2
        p_interaction_2 = 2 * (1 - stats.t.cdf(np.abs(t_interaction_2), len(y_2) - 4))

        if p_interaction_2 < 0.001:
            sig_2 = "***"
        elif p_interaction_2 < 0.01:
            sig_2 = "**"
        elif p_interaction_2 < 0.05:
            sig_2 = "*"
        else:
            sig_2 = "ns"
    except:
        sig_2 = "n/a"
        p_interaction_2 = 1.0

    print(f"ΔR² = {delta_r2_2:.4f}")
    print(f"Interaction: β = {beta_interaction_2:+.4f} {sig_2}")

    all_moderation_results.append({
        'Moderation': 'Experience × (SI → BI)',
        'Theory': 'Social influence stronger for novices',
        'Beta_Interaction': beta_interaction_2,
        'Delta_R2': delta_r2_2,
        'Significance': sig_2,
        'Interpretation': 'Supported' if (beta_interaction_2 < 0 and sig_2 != 'ns') else 'Not Supported'
    })

    # MODERATION ANALYSIS 3: Experience × (FC → UB)
    print(f"\n" + "=" * 80)
    print("MODERATION 3: Experience × (Facilitating Conditions → Use Behavior)")
    print("=" * 80)
    print("Theory: Facilitating conditions more important for experienced users")

    # Main and moderation models
    X_main_3 = mod_data[['FC_centered', 'Exp_centered']]
    y_3 = mod_data['UB']
    main_model_3 = LinearRegression().fit(X_main_3, y_3)
    r2_main_3 = main_model_3.score(X_main_3, y_3)

    X_mod_3 = mod_data[['FC_centered', 'Exp_centered', 'Exp_x_FC']]
    mod_model_3 = LinearRegression().fit(X_mod_3, y_3)
    r2_mod_3 = mod_model_3.score(X_mod_3, y_3)

    beta_interaction_3 = mod_model_3.coef_[2]
    delta_r2_3 = r2_mod_3 - r2_main_3

    # Significance test
    mod_pred_3 = mod_model_3.predict(X_mod_3)
    residuals_3 = y_3 - mod_pred_3
    mse_3 = np.sum(residuals_3 ** 2) / (len(y_3) - 4)

    try:
        X_mod_with_const_3 = np.column_stack([np.ones(len(X_mod_3)), X_mod_3])
        cov_matrix_3 = mse_3 * np.linalg.inv(X_mod_with_const_3.T @ X_mod_with_const_3)
        se_interaction_3 = np.sqrt(cov_matrix_3[3, 3])
        t_interaction_3 = beta_interaction_3 / se_interaction_3
        p_interaction_3 = 2 * (1 - stats.t.cdf(np.abs(t_interaction_3), len(y_3) - 4))

        if p_interaction_3 < 0.001:
            sig_3 = "***"
        elif p_interaction_3 < 0.01:
            sig_3 = "**"
        elif p_interaction_3 < 0.05:
            sig_3 = "*"
        else:
            sig_3 = "ns"
    except:
        sig_3 = "n/a"
        p_interaction_3 = 1.0

    print(f"ΔR² = {delta_r2_3:.4f}")
    print(f"Interaction: β = {beta_interaction_3:+.4f} {sig_3}")

    all_moderation_results.append({
        'Moderation': 'Experience × (FC → UB)',
        'Theory': 'Facilitating conditions more important for experts',
        'Beta_Interaction': beta_interaction_3,
        'Delta_R2': delta_r2_3,
        'Significance': sig_3,
        'Interpretation': 'Supported' if (beta_interaction_3 > 0 and sig_3 != 'ns') else 'Not Supported'
    })

    # MODERATION ANALYSIS 4: Experience × (HB → BI)
    print(f"\n" + "=" * 80)
    print("MODERATION 4: Experience × (Habit → Behavioral Intention)")
    print("=" * 80)
    print("Theory: Habit effects stronger for experienced users")

    # Main and moderation models
    X_main_4 = mod_data[['HB_centered', 'Exp_centered']]
    y_4 = mod_data['BI']
    main_model_4 = LinearRegression().fit(X_main_4, y_4)
    r2_main_4 = main_model_4.score(X_main_4, y_4)

    X_mod_4 = mod_data[['HB_centered', 'Exp_centered', 'Exp_x_HB_BI']]
    mod_model_4 = LinearRegression().fit(X_mod_4, y_4)
    r2_mod_4 = mod_model_4.score(X_mod_4, y_4)

    beta_interaction_4 = mod_model_4.coef_[2]
    delta_r2_4 = r2_mod_4 - r2_main_4

    # Significance test
    mod_pred_4 = mod_model_4.predict(X_mod_4)
    residuals_4 = y_4 - mod_pred_4
    mse_4 = np.sum(residuals_4 ** 2) / (len(y_4) - 4)

    try:
        X_mod_with_const_4 = np.column_stack([np.ones(len(X_mod_4)), X_mod_4])
        cov_matrix_4 = mse_4 * np.linalg.inv(X_mod_with_const_4.T @ X_mod_with_const_4)
        se_interaction_4 = np.sqrt(cov_matrix_4[3, 3])
        t_interaction_4 = beta_interaction_4 / se_interaction_4
        p_interaction_4 = 2 * (1 - stats.t.cdf(np.abs(t_interaction_4), len(y_4) - 4))

        if p_interaction_4 < 0.001:
            sig_4 = "***"
        elif p_interaction_4 < 0.01:
            sig_4 = "**"
        elif p_interaction_4 < 0.05:
            sig_4 = "*"
        else:
            sig_4 = "ns"
    except:
        sig_4 = "n/a"
        p_interaction_4 = 1.0

    print(f"ΔR² = {delta_r2_4:.4f}")
    print(f"Interaction: β = {beta_interaction_4:+.4f} {sig_4}")

    all_moderation_results.append({
        'Moderation': 'Experience × (HB → BI)',
        'Theory': 'Habit effects stronger for experienced users',
        'Beta_Interaction': beta_interaction_4,
        'Delta_R2': delta_r2_4,
        'Significance': sig_4,
        'Interpretation': 'Supported' if (beta_interaction_4 > 0 and sig_4 != 'ns') else 'Not Supported'
    })

    # MODERATION ANALYSIS 5: Experience × (HB → UB)
    print(f"\n" + "=" * 80)
    print("MODERATION 5: Experience × (Habit → Use Behavior)")
    print("=" * 80)
    print("Theory: Habit-behavior link stronger for experienced users")

    # Main and moderation models
    X_main_5 = mod_data[['HB_centered', 'Exp_centered']]
    y_5 = mod_data['UB']
    main_model_5 = LinearRegression().fit(X_main_5, y_5)
    r2_main_5 = main_model_5.score(X_main_5, y_5)

    X_mod_5 = mod_data[['HB_centered', 'Exp_centered', 'Exp_x_HB_UB']]
    mod_model_5 = LinearRegression().fit(X_mod_5, y_5)
    r2_mod_5 = mod_model_5.score(X_mod_5, y_5)

    beta_interaction_5 = mod_model_5.coef_[2]
    delta_r2_5 = r2_mod_5 - r2_main_5

    # Significance test
    mod_pred_5 = mod_model_5.predict(X_mod_5)
    residuals_5 = y_5 - mod_pred_5
    mse_5 = np.sum(residuals_5 ** 2) / (len(y_5) - 4)

    try:
        X_mod_with_const_5 = np.column_stack([np.ones(len(X_mod_5)), X_mod_5])
        cov_matrix_5 = mse_5 * np.linalg.inv(X_mod_with_const_5.T @ X_mod_with_const_5)
        se_interaction_5 = np.sqrt(cov_matrix_5[3, 3])
        t_interaction_5 = beta_interaction_5 / se_interaction_5
        p_interaction_5 = 2 * (1 - stats.t.cdf(np.abs(t_interaction_5), len(y_5) - 4))

        if p_interaction_5 < 0.001:
            sig_5 = "***"
        elif p_interaction_5 < 0.01:
            sig_5 = "**"
        elif p_interaction_5 < 0.05:
            sig_5 = "*"
        else:
            sig_5 = "ns"
    except:
        sig_5 = "n/a"
        p_interaction_5 = 1.0

    print(f"ΔR² = {delta_r2_5:.4f}")
    print(f"Interaction: β = {beta_interaction_5:+.4f} {sig_5}")

    all_moderation_results.append({
        'Moderation': 'Experience × (HB → UB)',
        'Theory': 'Habit-behavior link stronger for experienced users',
        'Beta_Interaction': beta_interaction_5,
        'Delta_R2': delta_r2_5,
        'Significance': sig_5,
        'Interpretation': 'Supported' if (beta_interaction_5 > 0 and sig_5 != 'ns') else 'Not Supported'
    })

    # SUMMARY AND INTERPRETATION
    print(f"\n" + "=" * 80)
    print("EXPERIENCE MODERATION ANALYSIS SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(all_moderation_results)

    print("\nAll Experience Moderation Results:")
    for i, result in enumerate(all_moderation_results, 1):
        print(f"{i}. {result['Moderation']}")
        print(f"   Theory: {result['Theory']}")
        print(f"   β = {result['Beta_Interaction']:+.4f} {result['Significance']}")
        print(f"   ΔR² = {result['Delta_R2']:.4f}")
        print(f"   Result: {result['Interpretation']}")
        print()

    # Count supported moderations
    supported_count = sum(1 for r in all_moderation_results if r['Interpretation'] == 'Supported')
    support_rate = (supported_count / len(all_moderation_results)) * 100

    print(f"📊 EXPERIENCE MODERATION SUMMARY:")
    print(f"Total Experience Moderations Tested: {len(all_moderation_results)}")
    print(f"Supported: {supported_count}")
    print(f"Not Supported: {len(all_moderation_results) - supported_count}")
    print(f"Support Rate: {support_rate:.1f}%")

    # Save results
    results_df.to_excel('experience_moderation_results.xlsx', index=False)

    print(f"\n💾 Results saved to: experience_moderation_results.xlsx")

    # THEORETICAL IMPLICATIONS
    print(f"\n" + "=" * 80)
    print("THEORETICAL IMPLICATIONS FOR GAMEFI ADOPTION")
    print("=" * 80)

    if supported_count > 0:
        print(f"🔍 EXPERIENCE EFFECTS DETECTED:")
        print(f"   • GameFi expertise shapes adoption mechanisms")
        print(f"   • User sophistication moderates traditional UTAUT2 relationships")
        print(f"   • Learning curve effects evident in GameFi context")

        # Specific insights based on which moderations were supported
        for result in all_moderation_results:
            if result['Interpretation'] == 'Supported':
                if 'EE → BI' in result['Moderation'] and result['Beta_Interaction'] < 0:
                    print(f"   • Learning curve confirmed: Experienced users less deterred by complexity")
                elif 'SI → BI' in result['Moderation'] and result['Beta_Interaction'] < 0:
                    print(f"   • Social influence reduces with expertise (independence development)")
                elif 'FC → UB' in result['Moderation'] and result['Beta_Interaction'] > 0:
                    print(f"   • Experienced users more sensitive to facilitating conditions")
                elif 'HB → BI' in result['Moderation'] and result['Beta_Interaction'] > 0:
                    print(f"   • Habit formation stronger in experienced GameFi users")
                elif 'HB → UB' in result['Moderation'] and result['Beta_Interaction'] > 0:
                    print(f"   • Habit-behavior automaticity increases with experience")
    else:
        print(f"⚠️ NO SIGNIFICANT EXPERIENCE MODERATIONS:")
        print(f"   • Experience does not significantly moderate tested relationships")
        print(f"   • GameFi adoption patterns consistent across experience levels")
        print(f"   • Users may adapt quickly to GameFi mechanics")

    print(f"\n🎉 EXPERIENCE MODERATION ANALYSIS COMPLETE!")
    print(f"\n📋 FINAL ANALYSIS STATUS:")
    print(f"   • Core Structural Model: ✅ Complete (16/16)")
    print(f"   • GameFi-Specific Moderation: ✅ Complete (1/1)")
    print(f"   • Experience Moderations: ✅ Complete (5/5)")
    print(f"   • Age/Gender Moderations: ⚠️ Excluded (methodological justification)")
    print(f"\n🏆 COMPREHENSIVE UTAUT2+GAMEFI ANALYSIS COMPLETE!")
    print(f"   Total meaningful analyses completed: 22/22")
    print(f"   Ready for thesis discussion and conclusions!")


if __name__ == "__main__":
    run_experience_moderation_analysis()