import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def run_tt_rp_moderation_analysis():
    print("🚀 Starting TT × (RP → BI) Moderation Analysis...")
    print("Research Question: Does Technology Trust moderate the Risk Perception → Behavioral Intention relationship?")

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

    # Prepare moderation analysis data
    moderation_vars = ['TT', 'RP', 'BI']
    mod_data = data[moderation_vars].dropna()

    print(f"✅ Moderation analysis data: {len(mod_data)} complete cases")
    print(f"\nDescriptive Statistics:")
    print(f"Trust in Technology (TT): M={mod_data['TT'].mean():.3f}, SD={mod_data['TT'].std():.3f}")
    print(f"Risk Perception (RP): M={mod_data['RP'].mean():.3f}, SD={mod_data['RP'].std():.3f}")
    print(f"Behavioral Intention (BI): M={mod_data['BI'].mean():.3f}, SD={mod_data['BI'].std():.3f}")

    # Mean-center variables for interaction analysis
    print(f"\n📐 Mean-centering variables for moderation analysis...")
    mod_data['TT_centered'] = mod_data['TT'] - mod_data['TT'].mean()
    mod_data['RP_centered'] = mod_data['RP'] - mod_data['RP'].mean()

    # Create interaction term
    mod_data['TT_x_RP'] = mod_data['TT_centered'] * mod_data['RP_centered']

    print(f"✅ Centered TT: M={mod_data['TT_centered'].mean():.6f}, SD={mod_data['TT_centered'].std():.3f}")
    print(f"✅ Centered RP: M={mod_data['RP_centered'].mean():.6f}, SD={mod_data['RP_centered'].std():.3f}")
    print(f"✅ Interaction TT×RP: M={mod_data['TT_x_RP'].mean():.6f}, SD={mod_data['TT_x_RP'].std():.3f}")

    # MODERATION ANALYSIS
    print(f"\n" + "=" * 70)
    print("MODERATION ANALYSIS: TT × (RP → BI)")
    print("=" * 70)

    # Step 1: Test main effects model (RP → BI, TT → BI)
    print("\nStep 1: Main Effects Model")
    print("Model: BI = β₀ + β₁(RP) + β₂(TT) + ε")

    X_main = mod_data[['RP_centered', 'TT_centered']]
    y_main = mod_data['BI']

    main_model = LinearRegression().fit(X_main, y_main)
    main_pred = main_model.predict(X_main)

    # Calculate R²
    ss_tot_main = np.sum((y_main - np.mean(y_main)) ** 2)
    ss_res_main = np.sum((y_main - main_pred) ** 2)
    r2_main = 1 - (ss_res_main / ss_tot_main)
    adj_r2_main = 1 - (1 - r2_main) * (len(y_main) - 1) / (len(y_main) - X_main.shape[1] - 1)

    print(f"R² (Main Effects) = {r2_main:.4f}")
    print(f"Adjusted R² = {adj_r2_main:.4f}")

    # Main effects coefficients
    beta_rp_main = main_model.coef_[0]
    beta_tt_main = main_model.coef_[1]

    print(f"β₁ (RP → BI): {beta_rp_main:+.4f}")
    print(f"β₂ (TT → BI): {beta_tt_main:+.4f}")

    # Step 2: Test moderation model (add interaction)
    print("\nStep 2: Moderation Model")
    print("Model: BI = β₀ + β₁(RP) + β₂(TT) + β₃(TT×RP) + ε")

    X_mod = mod_data[['RP_centered', 'TT_centered', 'TT_x_RP']]
    y_mod = mod_data['BI']

    mod_model = LinearRegression().fit(X_mod, y_mod)
    mod_pred = mod_model.predict(X_mod)

    # Calculate R²
    ss_res_mod = np.sum((y_mod - mod_pred) ** 2)
    r2_mod = 1 - (ss_res_mod / ss_tot_main)
    adj_r2_mod = 1 - (1 - r2_mod) * (len(y_mod) - 1) / (len(y_mod) - X_mod.shape[1] - 1)

    print(f"R² (Moderation Model) = {r2_mod:.4f}")
    print(f"Adjusted R² = {adj_r2_mod:.4f}")
    print(f"ΔR² = {r2_mod - r2_main:.4f}")

    # Moderation coefficients
    beta_rp_mod = mod_model.coef_[0]
    beta_tt_mod = mod_model.coef_[1]
    beta_interaction = mod_model.coef_[2]

    print(f"β₁ (RP → BI): {beta_rp_mod:+.4f}")
    print(f"β₂ (TT → BI): {beta_tt_mod:+.4f}")
    print(f"β₃ (TT×RP → BI): {beta_interaction:+.4f}")

    # Test significance of interaction
    residuals_mod = y_mod - mod_pred
    mse_mod = np.sum(residuals_mod ** 2) / (len(y_mod) - X_mod.shape[1] - 1)

    # Calculate standard errors and t-statistics
    X_mod_with_const = np.column_stack([np.ones(len(X_mod)), X_mod])
    try:
        cov_matrix = mse_mod * np.linalg.inv(X_mod_with_const.T @ X_mod_with_const)
        se_interaction = np.sqrt(cov_matrix[3, 3])  # SE for interaction term
        t_interaction = beta_interaction / se_interaction
        df_interaction = len(y_mod) - X_mod.shape[1] - 1
        p_interaction = 2 * (1 - stats.t.cdf(np.abs(t_interaction), df_interaction))

        if p_interaction < 0.001:
            sig_interaction = "***"
        elif p_interaction < 0.01:
            sig_interaction = "**"
        elif p_interaction < 0.05:
            sig_interaction = "*"
        else:
            sig_interaction = "ns"

        print(f"\nInteraction Significance Test:")
        print(f"t({df_interaction}) = {t_interaction:.3f}, p = {p_interaction:.4f} {sig_interaction}")

    except np.linalg.LinAlgError:
        print("Warning: Could not calculate interaction significance due to matrix singularity")
        sig_interaction = "n/a"
        p_interaction = 1.0

    # Step 3: Simple slopes analysis
    print(f"\n" + "=" * 70)
    print("SIMPLE SLOPES ANALYSIS")
    print("=" * 70)

    # Define high/low TT levels (±1 SD from mean)
    tt_low = -mod_data['TT_centered'].std()
    tt_high = +mod_data['TT_centered'].std()

    print(f"Low TT (−1 SD): {tt_low:.3f} from mean")
    print(f"High TT (+1 SD): {tt_high:.3f} from mean")

    # Calculate simple slopes: effect of RP on BI at different levels of TT
    # Simple slope = β₁ + β₃ × TT_level
    slope_low_tt = beta_rp_mod + beta_interaction * tt_low
    slope_high_tt = beta_rp_mod + beta_interaction * tt_high

    print(f"\nSimple Slopes:")
    print(f"Effect of RP → BI when TT is LOW:  {slope_low_tt:+.4f}")
    print(f"Effect of RP → BI when TT is HIGH: {slope_high_tt:+.4f}")
    print(f"Difference in slopes: {slope_high_tt - slope_low_tt:+.4f}")

    # Interpret the moderation
    print(f"\n📊 MODERATION INTERPRETATION:")

    if sig_interaction != "ns":
        if beta_interaction > 0:
            print("✅ Significant POSITIVE moderation detected:")
            print("   • Trust in Technology AMPLIFIES the Risk Perception → BI relationship")
            print("   • Higher trust users are MORE sensitive to risk when forming intentions")
            print("   • Supports 'sophisticated risk assessment' interpretation")
        else:
            print("✅ Significant NEGATIVE moderation detected:")
            print("   • Trust in Technology BUFFERS the Risk Perception → BI relationship")
            print("   • Higher trust users are LESS affected by risk when forming intentions")
            print("   • Supports 'trust reduces risk impact' interpretation")
    else:
        print("⚠️ No significant moderation effect:")
        print("   • Trust and Risk operate independently on Behavioral Intention")
        print("   • No evidence that trust moderates risk perception effects")

    # Step 4: Visualization
    print(f"\n📊 Creating moderation visualization...")

    # Create interaction plot
    plt.figure(figsize=(10, 6))

    # Generate prediction lines for visualization
    rp_range = np.linspace(mod_data['RP_centered'].min(), mod_data['RP_centered'].max(), 100)

    # Predictions for low TT
    bi_pred_low_tt = (main_model.intercept_ +
                      beta_rp_mod * rp_range +
                      beta_tt_mod * tt_low +
                      beta_interaction * rp_range * tt_low)

    # Predictions for high TT
    bi_pred_high_tt = (main_model.intercept_ +
                       beta_rp_mod * rp_range +
                       beta_tt_mod * tt_high +
                       beta_interaction * rp_range * tt_high)

    # Convert back to original scale for plotting
    rp_orig = rp_range + mod_data['RP'].mean()

    plt.plot(rp_orig, bi_pred_low_tt, 'r-', linewidth=2, label=f'Low TT (-1 SD)')
    plt.plot(rp_orig, bi_pred_high_tt, 'b-', linewidth=2, label=f'High TT (+1 SD)')

    plt.xlabel('Risk Perception', fontsize=12)
    plt.ylabel('Behavioral Intention', fontsize=12)
    plt.title('TT × RP Moderation Effect on Behavioral Intention', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add interaction significance to plot
    plt.text(0.02, 0.98, f'Interaction: β₃ = {beta_interaction:+.3f} {sig_interaction}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('tt_rp_moderation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    moderation_results = pd.DataFrame([
        {
            'Model': 'Main_Effects',
            'R_squared': r2_main,
            'Adj_R_squared': adj_r2_main,
            'Beta_RP': beta_rp_main,
            'Beta_TT': beta_tt_main,
            'Beta_Interaction': np.nan,
            'Interaction_Significance': np.nan
        },
        {
            'Model': 'Moderation',
            'R_squared': r2_mod,
            'Adj_R_squared': adj_r2_mod,
            'Beta_RP': beta_rp_mod,
            'Beta_TT': beta_tt_mod,
            'Beta_Interaction': beta_interaction,
            'Interaction_Significance': sig_interaction
        }
    ])

    simple_slopes = pd.DataFrame([
        {
            'TT_Level': 'Low (-1 SD)',
            'TT_Value': tt_low,
            'Simple_Slope_RP_to_BI': slope_low_tt,
            'Interpretation': 'Effect of Risk on Intention when Trust is Low'
        },
        {
            'TT_Level': 'High (+1 SD)',
            'TT_Value': tt_high,
            'Simple_Slope_RP_to_BI': slope_high_tt,
            'Interpretation': 'Effect of Risk on Intention when Trust is High'
        }
    ])

    # Save results
    moderation_results.to_excel('tt_rp_moderation_results.xlsx', index=False)
    simple_slopes.to_excel('tt_rp_simple_slopes.xlsx', index=False)

    print(f"\n💾 Results saved:")
    print(f"   • tt_rp_moderation_results.xlsx")
    print(f"   • tt_rp_simple_slopes.xlsx")
    print(f"   • tt_rp_moderation_plot.png")

    # SUMMARY AND THEORETICAL IMPLICATIONS
    print(f"\n" + "=" * 70)
    print("THEORETICAL IMPLICATIONS FOR GAMEFI ADOPTION")
    print("=" * 70)

    if sig_interaction != "ns":
        print(f"🔍 MODERATION DETECTED: Trust in Technology significantly moderates Risk Perception effects")

        if beta_interaction > 0:
            print(f"\n📊 SOPHISTICATED RISK ASSESSMENT PATTERN:")
            print(f"   • High-trust GameFi users show STRONGER risk-intention relationships")
            print(f"   • Supports 'calculated risk-taking' theoretical framework")
            print(f"   • Technology expertise increases risk sensitivity, not risk blindness")
            print(f"   • GameFi adoption involves informed risk-benefit analysis")

        else:
            print(f"\n📊 TRUST BUFFERING PATTERN:")
            print(f"   • High-trust GameFi users show WEAKER risk-intention relationships")
            print(f"   • Technology confidence reduces risk concerns")
            print(f"   • Traditional 'trust reduces perceived risk' mechanism")

    else:
        print(f"🔍 NO MODERATION: Trust and Risk operate independently")
        print(f"   • Trust in Technology does not modify Risk Perception effects")
        print(f"   • Separate pathways to Behavioral Intention")
        print(f"   • Users compartmentalize technology trust vs risk assessment")

    print(f"\n🎉 TT × (RP → BI) MODERATION ANALYSIS COMPLETE!")
    print(f"\nKey Findings:")
    print(f"   • Main model R² = {r2_main:.3f}")
    print(f"   • Moderation model R² = {r2_mod:.3f}")
    print(f"   • ΔR² from interaction = {r2_mod - r2_main:.3f}")
    print(f"   • Interaction effect β₃ = {beta_interaction:+.3f} {sig_interaction}")


if __name__ == "__main__":
    run_tt_rp_moderation_analysis()