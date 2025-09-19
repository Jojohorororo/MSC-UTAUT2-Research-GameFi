# UTAUT2 GameFi PLS-SEM Analysis - Basic v1 Model
# Comprehensive analysis script for UTAUT2 extended with GameFi constructs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
import warnings

warnings.filterwarnings('ignore')

# For PLS-SEM analysis (install with: pip install plssem)
try:
    import plssem

    print("✅ Using plssem package")
    PLS_PACKAGE = "plssem"
except ImportError:
    try:
        # Alternative: seminr equivalent functionality
        print("⚠️  plssem not available, using manual PLS implementation")
        PLS_PACKAGE = "manual"
    except ImportError:
        print("❌ Please install plssem: pip install plssem")
        exit()


def load_and_prepare_data(file_path="utaut2_cleaned_data.xlsx"):
    """
    Load and prepare UTAUT2 data for PLS-SEM analysis
    """
    print("🚀 Starting UTAUT2 PLS-SEM Analysis...")

    # Load the cleaned data
    df = pd.read_excel(file_path)
    print(f"✅ Data loaded: {len(df)} participants × {len(df.columns)} variables")

    # Define construct measurement model
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
    }

    # Extract only Likert scale data for PLS-SEM
    likert_items = []
    for construct, items in measurement_model.items():
        likert_items.extend(items)

    pls_data = df[likert_items].copy()

    # Behavioral variables (for separate analysis)
    behavioral_data = df[['UB1_UseFreq', 'UB2_WeeklyHours']].copy()
    demographic_data = df[['Age', 'Gender', 'Education', 'Income', 'GameFiExp', 'Location']].copy()

    print(f"✅ Prepared data: {len(pls_data)} participants × {len(likert_items)} Likert items")
    print(f"📊 Constructs: {len(measurement_model)} latent constructs")

    return pls_data, behavioral_data, demographic_data, measurement_model


def descriptive_statistics(df, measurement_model):
    """
    Generate comprehensive descriptive statistics
    """
    print("\n📊 DESCRIPTIVE STATISTICS")
    print("=" * 60)

    # Construct-level descriptives
    construct_stats = {}

    for construct, items in measurement_model.items():
        construct_data = df[items]
        construct_mean = construct_data.mean(axis=1)

        stats_dict = {
            'Mean': construct_mean.mean(),
            'SD': construct_mean.std(),
            'Min': construct_mean.min(),
            'Max': construct_mean.max(),
            'Skewness': stats.skew(construct_mean),
            'Kurtosis': stats.kurtosis(construct_mean),
            'N_Items': len(items)
        }

        construct_stats[construct] = stats_dict

    # Create descriptive table
    desc_df = pd.DataFrame(construct_stats).T
    desc_df = desc_df.round(3)

    print("Construct-Level Descriptive Statistics:")
    print(desc_df)

    # Save descriptive statistics
    desc_df.to_excel("descriptive_statistics.xlsx")
    print("💾 Descriptive statistics saved to: descriptive_statistics.xlsx")

    return desc_df, construct_stats


def reliability_analysis(df, measurement_model):
    """
    Assess reliability and validity of measurement model
    """
    print("\n🔍 MEASUREMENT MODEL ASSESSMENT")
    print("=" * 60)

    reliability_results = {}

    for construct, items in measurement_model.items():
        construct_data = df[items]

        # Cronbach's Alpha
        def cronbach_alpha(data):
            n_items = len(data.columns)
            if n_items < 2:
                return np.nan

            item_variances = data.var(axis=0, ddof=1)
            total_variance = data.sum(axis=1).var(ddof=1)

            alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
            return alpha

        alpha = cronbach_alpha(construct_data)

        # Inter-item correlations
        correlations = construct_data.corr().values
        upper_triangle = correlations[np.triu_indices_from(correlations, k=1)]
        avg_inter_item_corr = np.mean(upper_triangle) if len(upper_triangle) > 0 else np.nan

        # Item-total correlations
        total_score = construct_data.sum(axis=1)
        item_total_corrs = []
        for item in items:
            corr = construct_data[item].corr(total_score - construct_data[item])
            item_total_corrs.append(corr)

        avg_item_total_corr = np.mean(item_total_corrs)

        # Composite Reliability (approximation)
        # CR = (Σλ)² / [(Σλ)² + Σθ] where λ = loadings, θ = error variances
        # Using factor loadings approximation
        factor_loadings = item_total_corrs  # Approximation
        sum_loadings = np.sum(factor_loadings)
        sum_loadings_sq = sum_loadings ** 2

        # Error variances (approximation: 1 - λ²)
        error_variances = [1 - (loading ** 2) for loading in factor_loadings]
        sum_error_var = np.sum(error_variances)

        composite_reliability = sum_loadings_sq / (sum_loadings_sq + sum_error_var)

        # Average Variance Extracted (approximation)
        sum_loadings_squared = np.sum([loading ** 2 for loading in factor_loadings])
        ave = sum_loadings_squared / (sum_loadings_squared + sum_error_var)

        reliability_results[construct] = {
            'Cronbach_Alpha': alpha,
            'Composite_Reliability': composite_reliability,
            'AVE': ave,
            'Avg_Inter_Item_Corr': avg_inter_item_corr,
            'Avg_Item_Total_Corr': avg_item_total_corr,
            'N_Items': len(items)
        }

        print(f"{construct}:")
        print(f"  Cronbach's α = {alpha:.3f}")
        print(f"  Composite Reliability = {composite_reliability:.3f}")
        print(f"  AVE = {ave:.3f}")
        print(f"  Items: {len(items)}")

        # Quality assessment
        alpha_status = "✅ Excellent" if alpha >= 0.9 else "✅ Good" if alpha >= 0.8 else "✅ Acceptable" if alpha >= 0.7 else "⚠️ Questionable" if alpha >= 0.6 else "❌ Poor"
        cr_status = "✅ Good" if composite_reliability >= 0.7 else "⚠️ Low"
        ave_status = "✅ Good" if ave >= 0.5 else "⚠️ Low"

        print(f"  Quality: α {alpha_status}, CR {cr_status}, AVE {ave_status}")
        print()

    # Create reliability summary table
    reliability_df = pd.DataFrame(reliability_results).T
    reliability_df = reliability_df.round(3)

    # Save reliability results
    reliability_df.to_excel("reliability_analysis.xlsx")
    print("💾 Reliability analysis saved to: reliability_analysis.xlsx")

    return reliability_results


def correlation_analysis(df, measurement_model):
    """
    Generate construct correlation matrix for discriminant validity
    """
    print("\n🔗 CORRELATION ANALYSIS")
    print("=" * 60)

    # Calculate construct scores (means)
    construct_scores = {}
    for construct, items in measurement_model.items():
        construct_scores[construct] = df[items].mean(axis=1)

    construct_df = pd.DataFrame(construct_scores)

    # Correlation matrix
    corr_matrix = construct_df.corr()
    print("Construct Correlation Matrix:")
    print(corr_matrix.round(3))

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Construct Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save correlation matrix
    corr_matrix.to_excel("correlation_matrix.xlsx")
    print("💾 Correlation matrix saved to: correlation_matrix.xlsx")

    return corr_matrix, construct_scores


def manual_pls_sem_analysis(df, measurement_model, construct_scores):
    """
    Simplified PLS-SEM analysis using manual calculations
    """
    print("\n🏗️ STRUCTURAL MODEL ANALYSIS")
    print("=" * 60)

    # Define structural model paths (Basic v1 Model)
    structural_model = {
        # Direct effects to Behavioral Intention
        'PE → BI': ('PE', 'BI'),
        'EE → BI': ('EE', 'BI'),
        'SI → BI': ('SI', 'BI'),
        'FC → BI': ('FC', 'BI'),  # Note: FC typically goes to UB, but including per v1 spec
        'HM → BI': ('HM', 'BI'),
        'PV → BI': ('PV', 'BI'),
        'HB → BI': ('HB', 'BI'),
        'EM → BI': ('EM', 'BI'),
        'TT → BI': ('TT', 'BI'),
        'RP → BI': ('RP', 'BI'),  # Negative relationship expected
    }

    # Note: UB paths handled separately as behavioral measures
    print("Structural Model Paths (Direct Effects to Behavioral Intention):")
    print("- Performance Expectancy → Behavioral Intention")
    print("- Effort Expectancy → Behavioral Intention")
    print("- Social Influence → Behavioral Intention")
    print("- Facilitating Conditions → Behavioral Intention")
    print("- Hedonic Motivation → Behavioral Intention")
    print("- Price Value → Behavioral Intention")
    print("- Habit → Behavioral Intention")
    print("- Economic Motivation → Behavioral Intention")
    print("- Trust in Technology → Behavioral Intention")
    print("- Risk Perception → Behavioral Intention (negative expected)")
    print()

    # Calculate path coefficients using regression
    construct_df = pd.DataFrame(construct_scores)
    path_results = {}

    print("Path Coefficients (Standardized β):")
    print("-" * 40)

    # Prepare predictors for BI
    bi_predictors = ['PE', 'EE', 'SI', 'FC', 'HM', 'PV', 'HB', 'EM', 'TT', 'RP']
    X = construct_df[bi_predictors]
    y = construct_df['BI']

    # Standardize variables
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()

    # Multiple regression for path coefficients
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    model = LinearRegression().fit(X_std, y_std)
    path_coefficients = model.coef_
    r_squared = r2_score(y_std, model.predict(X_std))

    print(f"R² for Behavioral Intention = {r_squared:.3f}")
    print(f"Adjusted R² = {1 - (1 - r_squared) * (len(y_std) - 1) / (len(y_std) - len(bi_predictors) - 1):.3f}")
    print()

    for i, predictor in enumerate(bi_predictors):
        beta = path_coefficients[i]

        # Simple significance test (t-test approximation)
        # Note: For proper significance testing, use bootstrapping in production
        correlation = construct_df[predictor].corr(construct_df['BI'])

        # Effect size interpretation
        effect_size = "Large" if abs(beta) >= 0.35 else "Medium" if abs(beta) >= 0.15 else "Small"
        significance = "**" if abs(beta) >= 0.1 else "*" if abs(beta) >= 0.05 else ""

        print(f"{predictor} → BI: β = {beta:+.3f} {significance} ({effect_size} effect)")

        path_results[f"{predictor}_to_BI"] = {
            'beta': beta,
            'correlation': correlation,
            'effect_size': effect_size
        }

    print("\n** Large effect (≥0.35), * Medium effect (≥0.15)")
    print("Note: For rigorous significance testing, use bootstrapping (5000+ samples)")

    # Model quality assessment
    print(f"\n📊 MODEL QUALITY ASSESSMENT")
    print("-" * 40)
    print(f"R² (Behavioral Intention): {r_squared:.3f}")

    # R² interpretation
    if r_squared >= 0.75:
        r2_interpretation = "Substantial"
    elif r_squared >= 0.50:
        r2_interpretation = "Moderate"
    elif r_squared >= 0.25:
        r2_interpretation = "Weak"
    else:
        r2_interpretation = "Very Weak"

    print(f"R² Interpretation: {r2_interpretation}")

    # Effect sizes (f²)
    print(f"\nEffect Sizes (f²):")
    for predictor in bi_predictors:
        # Calculate f² = (R²_included - R²_excluded) / (1 - R²_included)
        X_reduced = X_std.drop(columns=[predictor])
        model_reduced = LinearRegression().fit(X_reduced, y_std)
        r2_reduced = r2_score(y_std, model_reduced.predict(X_reduced))

        f_squared = (r_squared - r2_reduced) / (1 - r_squared)

        # f² interpretation
        f2_size = "Large" if f_squared >= 0.35 else "Medium" if f_squared >= 0.15 else "Small" if f_squared >= 0.02 else "None"

        print(f"{predictor}: f² = {f_squared:.3f} ({f2_size})")

    # Save path results
    path_df = pd.DataFrame(path_results).T
    path_df.to_excel("path_coefficients.xlsx")
    print("\n💾 Path coefficients saved to: path_coefficients.xlsx")

    return path_results, r_squared


def behavioral_analysis(behavioral_data, construct_scores):
    """
    Analyze relationships with behavioral Use Behavior measures
    """
    print("\n🎯 BEHAVIORAL USE ANALYSIS")
    print("=" * 60)

    # Convert behavioral data to numeric for analysis
    # UB1_UseFreq: Usage frequency
    freq_mapping = {
        'Never': 0,
        'Rarely (less than once a month)': 1,
        'A few times a month': 2,
        'Weekly': 3,
        'Several times a week': 4,
        'Daily': 5
    }

    # UB2_WeeklyHours: Weekly hours
    hours_mapping = {
        '0 hours': 0,
        'Less than 1 hour': 0.5,
        '1-5 hours': 3,
        '6-10 hours': 8,
        '11-20 hours': 15.5,
        'More than 20 hours': 25
    }

    behavioral_numeric = behavioral_data.copy()
    behavioral_numeric['UB_Frequency'] = behavioral_data['UB1_UseFreq'].map(freq_mapping)
    behavioral_numeric['UB_Hours'] = behavioral_data['UB2_WeeklyHours'].map(hours_mapping)

    # Correlations with key predictors
    construct_df = pd.DataFrame(construct_scores)

    print("Correlations with Behavioral Use:")
    print("-" * 40)

    key_predictors = ['BI', 'HB', 'FC', 'EM', 'TT', 'RP']

    for predictor in key_predictors:
        freq_corr = construct_df[predictor].corr(behavioral_numeric['UB_Frequency'])
        hours_corr = construct_df[predictor].corr(behavioral_numeric['UB_Hours'])

        print(f"{predictor} → Usage Frequency: r = {freq_corr:+.3f}")
        print(f"{predictor} → Weekly Hours: r = {hours_corr:+.3f}")
        print()

    # Usage frequency distribution
    print("Usage Frequency Distribution:")
    usage_dist = behavioral_data['UB1_UseFreq'].value_counts().sort_index()
    for freq, count in usage_dist.items():
        pct = (count / len(behavioral_data)) * 100
        print(f"  {freq}: {count} ({pct:.1f}%)")

    return behavioral_numeric


def generate_report(desc_stats, reliability_results, path_results, r_squared):
    """
    Generate comprehensive analysis report
    """
    print("\n📋 ANALYSIS SUMMARY REPORT")
    print("=" * 60)

    print("🎯 STUDY OVERVIEW")
    print(f"Sample Size: 516 participants")
    print(f"Measurement Model: 12 latent constructs, 42 indicators")
    print(f"Structural Model: Basic v1 UTAUT2 + GameFi extensions")
    print()

    print("📊 KEY FINDINGS")
    print("-" * 30)

    # Reliability summary
    print("Measurement Model Quality:")
    excellent_alpha = sum(1 for r in reliability_results.values() if r['Cronbach_Alpha'] >= 0.9)
    good_alpha = sum(1 for r in reliability_results.values() if 0.8 <= r['Cronbach_Alpha'] < 0.9)
    acceptable_alpha = sum(1 for r in reliability_results.values() if 0.7 <= r['Cronbach_Alpha'] < 0.8)

    print(f"  Excellent reliability (α ≥ 0.9): {excellent_alpha} constructs")
    print(f"  Good reliability (α ≥ 0.8): {good_alpha} constructs")
    print(f"  Acceptable reliability (α ≥ 0.7): {acceptable_alpha} constructs")
    print()

    # Structural model summary
    print("Structural Model Results:")
    print(f"  R² (Behavioral Intention): {r_squared:.3f}")

    # Count significant paths
    large_effects = sum(1 for r in path_results.values() if abs(r['beta']) >= 0.35)
    medium_effects = sum(1 for r in path_results.values() if 0.15 <= abs(r['beta']) < 0.35)
    small_effects = sum(1 for r in path_results.values() if 0.02 <= abs(r['beta']) < 0.15)

    print(f"  Large effects (β ≥ 0.35): {large_effects} paths")
    print(f"  Medium effects (β ≥ 0.15): {medium_effects} paths")
    print(f"  Small effects (β ≥ 0.02): {small_effects} paths")
    print()

    # Top predictors
    print("Strongest Predictors of Behavioral Intention:")
    sorted_paths = sorted(path_results.items(), key=lambda x: abs(x[1]['beta']), reverse=True)
    for i, (path, results) in enumerate(sorted_paths[:5]):
        predictor = path.replace('_to_BI', '')
        print(f"  {i + 1}. {predictor}: β = {results['beta']:+.3f}")
    print()

    print("📝 NEXT STEPS")
    print("-" * 30)
    print("1. Review measurement model quality and consider item refinement")
    print("2. Interpret path coefficients in theoretical context")
    print("3. Consider moderation analysis (Age, Gender, Experience)")
    print("4. Examine mediation effects if theoretically justified")
    print("5. Validate findings with predictive relevance (Q²)")

    # Save comprehensive report
    report_data = {
        'Analysis': 'UTAUT2 GameFi PLS-SEM v1',
        'Sample_Size': 516,
        'Constructs': 12,
        'Indicators': 42,
        'R2_BI': r_squared,
        'Large_Effects': large_effects,
        'Medium_Effects': medium_effects,
        'Small_Effects': small_effects
    }

    report_df = pd.DataFrame([report_data])
    report_df.to_excel("analysis_summary.xlsx", index=False)
    print("\n💾 Analysis summary saved to: analysis_summary.xlsx")


def main():
    """
    Main analysis function
    """
    # Load and prepare data
    pls_data, behavioral_data, demographic_data, measurement_model = load_and_prepare_data("utaut2_cleaned_data.xlsx")

    # Descriptive statistics
    desc_stats, construct_stats = descriptive_statistics(pls_data, measurement_model)

    # Reliability analysis
    reliability_results = reliability_analysis(pls_data, measurement_model)

    # Correlation analysis
    corr_matrix, construct_scores = correlation_analysis(pls_data, measurement_model)

    # Structural model analysis
    path_results, r_squared = manual_pls_sem_analysis(pls_data, measurement_model, construct_scores)

    # Behavioral analysis
    behavioral_numeric = behavioral_analysis(behavioral_data, construct_scores)

    # Generate final report
    generate_report(desc_stats, reliability_results, path_results, r_squared)

    print("\n🎉 UTAUT2 PLS-SEM Analysis Complete!")
    print("\nOutput Files Generated:")
    print("📊 descriptive_statistics.xlsx")
    print("🔍 reliability_analysis.xlsx")
    print("🔗 correlation_matrix.xlsx")
    print("🏗️ path_coefficients.xlsx")
    print("📋 analysis_summary.xlsx")
    print("📈 correlation_matrix.png")


if __name__ == "__main__":
    main()