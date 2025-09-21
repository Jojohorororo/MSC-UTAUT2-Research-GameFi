"""
Factor Analysis for UTAUT2 + GameFi Extensions Study
====================================================
This script performs comprehensive factor analysis on 42 Likert items across 12 constructs:
- 8 UTAUT2 constructs (PE, EE, SI, FC, HM, PV, HB, BI)
- 4 GameFi extensions (EM, RP, TT, RC)

Requirements: pip install pandas numpy matplotlib seaborn factor-analyzer scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("husl")


def load_data(filepath):
    """Load and prepare data for factor analysis"""
    print("🚀 LOADING UTAUT2 + GAMEFI DATA FOR FACTOR ANALYSIS")
    print("=" * 60)

    # Load the Excel file
    df = pd.read_excel(filepath)
    print(f"✅ Data loaded: {df.shape[0]} participants × {df.shape[1]} variables")

    # Define the 42 Likert items for factor analysis
    likert_items = [
        # UTAUT2 Original Constructs (30 items)
        'PE1', 'PE2', 'PE3', 'PE4', 'PE5',  # Performance Expectancy (5)
        'EE1', 'EE2', 'EE3', 'EE4',  # Effort Expectancy (4)
        'SI1', 'SI2', 'SI3',  # Social Influence (3)
        'FC1', 'FC2', 'FC3', 'FC4',  # Facilitating Conditions (4)
        'HM1', 'HM2', 'HM3', 'HM4',  # Hedonic Motivation (4)
        'PV1', 'PV2', 'PV3',  # Price Value (3)
        'HB1', 'HB2', 'HB3', 'HB4',  # Habit (4)
        'BI1', 'BI2', 'BI3',  # Behavioral Intention (3)
        # GameFi Extensions (12 items)
        'EM1', 'EM2', 'EM3',  # Economic Motivation (3)
        'RP1', 'RP2', 'RP3', 'RP4',  # Risk Perception (4)
        'TT1', 'TT2', 'TT3',  # Trust in Technology (3)
        'RC1', 'RC2'  # Regulatory Compliance (2)
    ]

    # Extract factor analysis data
    fa_data = df[likert_items].copy()
    print(f"✅ Factor analysis dataset: {fa_data.shape[0]} × {fa_data.shape[1]} items")

    # Check for missing values
    missing_count = fa_data.isnull().sum().sum()
    print(f"✅ Missing values: {missing_count}")

    if missing_count > 0:
        print("⚠️  Handling missing values with median imputation...")
        fa_data = fa_data.fillna(fa_data.median())

    # Define theoretical construct mapping
    construct_mapping = {
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

    return fa_data, construct_mapping, likert_items


def perform_factorability_tests(data):
    """Test if data is suitable for factor analysis"""
    print("\n📊 FACTORABILITY ASSESSMENT")
    print("-" * 40)

    # Kaiser-Meyer-Olkin Test
    kmo_all, kmo_model = calculate_kmo(data)
    print(f"Kaiser-Meyer-Olkin (KMO) Test:")
    print(f"  Overall KMO: {kmo_model:.4f}")

    if kmo_model >= 0.9:
        kmo_interpretation = "Excellent (≥0.9)"
    elif kmo_model >= 0.8:
        kmo_interpretation = "Good (≥0.8)"
    elif kmo_model >= 0.7:
        kmo_interpretation = "Adequate (≥0.7)"
    elif kmo_model >= 0.6:
        kmo_interpretation = "Mediocre (≥0.6)"
    else:
        kmo_interpretation = "Poor (<0.6)"

    print(f"  Interpretation: {kmo_interpretation}")

    # Bartlett's Test of Sphericity
    chi_square_value, p_value = calculate_bartlett_sphericity(data)
    print(f"\nBartlett's Test of Sphericity:")
    print(f"  Chi-square: {chi_square_value:.2f}")
    print(f"  p-value: {p_value:.2e}")

    if p_value < 0.05:
        print(f"  Interpretation: ✅ Significant (p < 0.05) - Factor analysis appropriate")
    else:
        print(f"  Interpretation: ❌ Non-significant (p ≥ 0.05) - Factor analysis questionable")

    return kmo_model, p_value


def determine_optimal_factors(data, max_factors=15):
    """Determine optimal number of factors using multiple criteria"""
    print("\n📈 DETERMINING OPTIMAL NUMBER OF FACTORS")
    print("-" * 45)

    # Eigenvalue analysis
    fa_eigen = FactorAnalyzer(n_factors=data.shape[1], rotation=None)
    fa_eigen.fit(data)
    eigenvalues = fa_eigen.get_eigenvalues()[0]

    # Kaiser criterion (eigenvalues > 1)
    kaiser_factors = sum(eigenvalues > 1)
    print(f"Kaiser Criterion (eigenvalues > 1): {kaiser_factors} factors")

    # Plot scree plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(eigenvalues[:max_factors]) + 1), eigenvalues[:max_factors], 'bo-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Kaiser Criterion (λ=1)')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot - Eigenvalues')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Variance explained
    plt.subplot(1, 2, 2)
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100
    plt.plot(range(1, len(cumulative_variance[:max_factors]) + 1), cumulative_variance[:max_factors], 'go-',
             linewidth=2)
    plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60% Variance')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='70% Variance')
    plt.xlabel('Factor Number')
    plt.ylabel('Cumulative Variance Explained (%)')
    plt.title('Cumulative Variance Explained')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('factor_analysis_scree_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Variance explained by first 12 factors: {cumulative_variance[11]:.1f}%")

    return kaiser_factors, eigenvalues


def run_confirmatory_factor_analysis(data, construct_mapping, n_factors=12):
    """Run CFA to test the theoretical 12-factor model"""
    print(f"\n🔍 CONFIRMATORY FACTOR ANALYSIS ({n_factors} FACTORS)")
    print("-" * 50)

    # Fit the factor model
    fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin', method='ml')
    fa.fit(data)

    # Get factor loadings
    loadings = fa.loadings_
    loadings_df = pd.DataFrame(loadings,
                               index=data.columns,
                               columns=[f'Factor_{i + 1}' for i in range(n_factors)])

    print("✅ Factor analysis completed")
    print(f"Variance explained: {fa.get_factor_variance()[2][-1] * 100:.1f}%")

    return fa, loadings_df


def analyze_factor_loadings(loadings_df, construct_mapping, threshold=0.6):
    """Analyze factor loading patterns against theoretical model"""
    print("\n📋 FACTOR LOADING ANALYSIS")
    print("-" * 35)

    # Find highest loading for each item
    item_assignments = {}
    problematic_items = []

    for item in loadings_df.index:
        max_loading = loadings_df.loc[item].abs().max()
        factor_assignment = loadings_df.loc[item].abs().idxmax()

        # Check cross-loadings
        sorted_loadings = loadings_df.loc[item].abs().sort_values(ascending=False)
        primary_loading = sorted_loadings.iloc[0]
        secondary_loading = sorted_loadings.iloc[1]

        item_assignments[item] = {
            'factor': factor_assignment,
            'primary_loading': primary_loading,
            'secondary_loading': secondary_loading,
            'cross_loading_issue': (secondary_loading > 0.3 and primary_loading < 0.7)
        }

        if primary_loading < threshold:
            problematic_items.append(f"{item}: low loading ({primary_loading:.3f})")
        elif secondary_loading > 0.3:
            problematic_items.append(
                f"{item}: cross-loading (primary: {primary_loading:.3f}, secondary: {secondary_loading:.3f})")

    # Create factor loading heatmap
    plt.figure(figsize=(16, 12))
    mask = np.abs(loadings_df) < 0.3  # Hide loadings below 0.3
    sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                mask=mask, cbar_kws={'label': 'Factor Loading'})
    plt.title('Factor Loading Matrix (|loading| ≥ 0.3 shown)', fontsize=14, fontweight='bold')
    plt.xlabel('Factors')
    plt.ylabel('Items')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print construct-factor alignment
    print("\nConstruct-Factor Alignment Analysis:")
    print("=" * 40)

    for construct, items in construct_mapping.items():
        print(f"\n{construct} Construct:")
        construct_loadings = []
        for item in items:
            if item in loadings_df.index:
                max_loading = loadings_df.loc[item].abs().max()
                factor_assignment = loadings_df.loc[item].abs().idxmax()
                construct_loadings.append((item, factor_assignment, max_loading))
                print(f"  {item}: {factor_assignment} (loading: {max_loading:.3f})")

        # Check if items cluster on same factor
        factors = [x[1] for x in construct_loadings]
        if len(set(factors)) == 1:
            print(f"  ✅ All items load on {factors[0]}")
        else:
            print(f"  ⚠️  Items spread across factors: {set(factors)}")

    if problematic_items:
        print(f"\n⚠️  Items requiring attention ({len(problematic_items)}):")
        for item in problematic_items:
            print(f"  • {item}")
    else:
        print("\n✅ All items meet loading criteria!")

    return item_assignments


def calculate_construct_validity(data, construct_mapping, loadings_df):
    """Calculate convergent and discriminant validity measures"""
    print("\n🎯 CONSTRUCT VALIDITY ASSESSMENT")
    print("-" * 40)

    validity_results = {}

    # Calculate Average Variance Extracted (AVE) for each construct
    for construct, items in construct_mapping.items():
        construct_items = [item for item in items if item in data.columns]

        if len(construct_items) >= 2:
            # Get loadings for this construct's items on their primary factor
            loadings = []
            for item in construct_items:
                primary_factor = loadings_df.loc[item].abs().idxmax()
                loading = loadings_df.loc[item, primary_factor]
                loadings.append(loading ** 2)

            ave = np.mean(loadings)

            # Calculate construct reliability (Composite Reliability)
            sum_loadings = sum(np.sqrt(loading) for loading in loadings)
            sum_squared_loadings = sum(loading for loading in loadings)
            error_variance = len(construct_items) - sum_squared_loadings

            composite_reliability = (sum_loadings ** 2) / ((sum_loadings ** 2) + error_variance)

            validity_results[construct] = {
                'items': len(construct_items),
                'ave': ave,
                'composite_reliability': composite_reliability,
                'ave_sqrt': np.sqrt(ave)
            }

            print(f"{construct}:")
            print(f"  Items: {len(construct_items)}")
            print(f"  AVE: {ave:.3f} {'✅' if ave >= 0.5 else '❌'} ({'Good' if ave >= 0.5 else 'Poor'})")
            print(
                f"  Composite Reliability: {composite_reliability:.3f} {'✅' if composite_reliability >= 0.7 else '❌'}")

    return validity_results


def create_factor_summary_table(fa, loadings_df, construct_mapping):
    """Create comprehensive factor analysis summary table"""
    print("\n📊 FACTOR ANALYSIS SUMMARY TABLE")
    print("-" * 40)

    # Variance explained by each factor
    variance_explained = fa.get_factor_variance()

    summary_data = []
    for i, factor in enumerate(loadings_df.columns):
        # Find items with highest loadings on this factor
        factor_loadings = loadings_df[factor].abs().sort_values(ascending=False)
        top_items = factor_loadings.head(5)

        # Try to identify which construct this factor represents
        likely_construct = "Unknown"
        for construct, items in construct_mapping.items():
            construct_items_on_factor = sum(1 for item in items if item in top_items.index[:3])
            if construct_items_on_factor >= 2:
                likely_construct = construct
                break

        summary_data.append({
            'Factor': factor,
            'Eigenvalue': variance_explained[0][i],
            'Variance_Explained_%': variance_explained[1][i] * 100,
            'Cumulative_Variance_%': variance_explained[2][i] * 100,
            'Likely_Construct': likely_construct,
            'Top_Loading_Items': ', '.join(top_items.head(3).index),
            'Highest_Loading': top_items.iloc[0]
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format='%.3f'))

    # Save summary table
    summary_df.to_csv('factor_analysis_summary.csv', index=False)
    print("\n💾 Summary table saved as 'factor_analysis_summary.csv'")

    return summary_df


def generate_final_report(kmo_value, bartlett_p, fa, loadings_df, construct_mapping, validity_results):
    """Generate comprehensive factor analysis report"""
    print("\n" + "=" * 60)
    print("🎯 FACTOR ANALYSIS - FINAL REPORT")
    print("=" * 60)

    variance_explained = fa.get_factor_variance()[2][-1] * 100

    report = f"""
FACTOR ANALYSIS RESULTS SUMMARY
{'=' * 40}

DATA OVERVIEW:
• Sample Size: {fa.loadings_.shape[0]} items from 516 participants
• Ratio: 12.3 participants per item (Excellent)
• Missing Data: Handled appropriately

FACTORABILITY ASSESSMENT:
• KMO Test: {kmo_value:.4f} ({
    'Excellent' if kmo_value >= 0.9 else
    'Good' if kmo_value >= 0.8 else
    'Adequate' if kmo_value >= 0.7 else
    'Mediocre' if kmo_value >= 0.6 else 'Poor'
    })
• Bartlett's Test: p = {bartlett_p:.2e} ({'Significant' if bartlett_p < 0.05 else 'Non-significant'})

FACTOR STRUCTURE:
• Number of Factors Extracted: 12 (theoretical model)
• Total Variance Explained: {variance_explained:.1f}%
• Rotation Method: Oblimin (oblique)
• Extraction Method: Maximum Likelihood

CONSTRUCT VALIDITY:
"""

    for construct, results in validity_results.items():
        report += f"• {construct}: AVE = {results['ave']:.3f}, CR = {results['composite_reliability']:.3f}\n"

    report += f"""
RECOMMENDATIONS:
"""

    if kmo_value >= 0.8 and bartlett_p < 0.05:
        report += "✅ Data is excellent for factor analysis\n"
    elif kmo_value >= 0.7:
        report += "✅ Data is adequate for factor analysis\n"
    else:
        report += "⚠️ Data quality concerns for factor analysis\n"

    if variance_explained >= 60:
        report += f"✅ Good variance explanation ({variance_explained:.1f}%)\n"
    else:
        report += f"⚠️ Consider additional factors ({variance_explained:.1f}% explained)\n"

    poor_ave_constructs = [k for k, v in validity_results.items() if v['ave'] < 0.5]
    if poor_ave_constructs:
        report += f"⚠️ Low AVE constructs: {', '.join(poor_ave_constructs)}\n"
    else:
        report += "✅ All constructs meet AVE criteria (≥0.5)\n"

    print(report)

    # Save report to file
    with open('factor_analysis_report.txt', 'w') as f:
        f.write(report)
    print("💾 Full report saved as 'factor_analysis_report.txt'")


def main():
    """Main function to run complete factor analysis"""
    print("🚀 UTAUT2 + GAMEFI FACTOR ANALYSIS PIPELINE")
    print("=" * 50)
    print("Analyzing 42 Likert items across 12 theoretical constructs")
    print("Study: Technology Acceptance in GameFi Platforms\n")

    # File path - UPDATE THIS TO YOUR FILE LOCATION
    filepath = 'utaut2_cleaned_data.xlsx'  # Update this path!

    try:
        # Step 1: Load and prepare data
        fa_data, construct_mapping, likert_items = load_data(filepath)

        # Step 2: Test data suitability
        kmo_value, bartlett_p = perform_factorability_tests(fa_data)

        # Step 3: Determine optimal factors
        kaiser_factors, eigenvalues = determine_optimal_factors(fa_data)

        # Step 4: Run confirmatory factor analysis
        fa, loadings_df = run_confirmatory_factor_analysis(fa_data, construct_mapping, n_factors=12)

        # Step 5: Analyze loading patterns
        item_assignments = analyze_factor_loadings(loadings_df, construct_mapping)

        # Step 6: Assess construct validity
        validity_results = calculate_construct_validity(fa_data, construct_mapping, loadings_df)

        # Step 7: Create summary tables
        summary_df = create_factor_summary_table(fa, loadings_df, construct_mapping)

        # Step 8: Generate final report
        generate_final_report(kmo_value, bartlett_p, fa, loadings_df, construct_mapping, validity_results)

        # Save detailed results
        loadings_df.to_csv('detailed_factor_loadings.csv')
        fa_data.to_csv('factor_analysis_data.csv', index=False)

        print(f"\n🎉 FACTOR ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Files generated:")
        print("   • factor_analysis_scree_plot.png")
        print("   • factor_loadings_heatmap.png")
        print("   • factor_analysis_summary.csv")
        print("   • detailed_factor_loadings.csv")
        print("   • factor_analysis_report.txt")
        print("   • factor_analysis_data.csv")

    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{filepath}'")
        print("Please update the 'filepath' variable with the correct path to your Excel file.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your data file and try again.")


if __name__ == "__main__":
    main()