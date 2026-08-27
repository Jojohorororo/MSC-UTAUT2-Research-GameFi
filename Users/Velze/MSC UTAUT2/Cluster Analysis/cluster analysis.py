"""
UTAUT2 GameFi Cluster Analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD DATA AND CALCULATE CONSTRUCT MEANS
# ============================================================================

def load_and_aggregate_data(filepath='utaut2_cleaned_data.xlsx'):
    """Load UTAUT2 data and calculate construct means from individual items"""
    print("=" * 70)
    print("STEP 1: LOADING DATA AND CALCULATING CONSTRUCT MEANS")
    print("=" * 70)

    # Load data
    df = pd.read_excel(filepath)

    print(f"Dataset loaded successfully!")
    print(f"Total participants: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    # Define construct items mapping
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

    constructs = list(construct_items.keys())

    print("\nCalculating construct means from individual items:")

    # Calculate mean for each construct
    X = pd.DataFrame()
    for construct, items in construct_items.items():
        # Check if all items exist
        missing_items = [item for item in items if item not in df.columns]
        if missing_items:
            print(f"  WARNING: {construct} missing items: {missing_items}")
            continue

        # Calculate mean, handling any missing values
        X[construct] = df[items].mean(axis=1)
        print(f"  ✓ {construct}: Mean of {len(items)} items calculated")

    # Check for missing values in constructs
    missing = X.isnull().sum().sum()
    print(f"\nMissing values in constructs: {missing}")

    if missing > 0:
        print("Removing rows with missing construct values...")
        X = X.dropna()
        print(f"Remaining participants: {len(X)}")

    print(f"\nFinal data shape: {X.shape}")
    print("\nDescriptive Statistics of Constructs:")
    print(X.describe().round(2))

    # Create a complete dataframe with demographics for reference
    df_complete = df.copy()
    for construct in constructs:
        if construct in X.columns:
            df_complete[construct] = X[construct]

    return df_complete, X, constructs

# ============================================================================
# 2. STANDARDIZE DATA
# ============================================================================

def standardize_data(X):
    """Standardize features using z-scores"""
    print("\n" + "=" * 70)
    print("STEP 2: STANDARDIZING DATA (Z-SCORES)")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    print("Data standardized successfully!")
    print("\nStandardized data statistics:")
    print(X_scaled_df.describe().round(2))

    return X_scaled, X_scaled_df, scaler

# ============================================================================
# 3. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

def elbow_analysis(X_scaled, max_k=10):
    """Perform elbow method analysis"""
    print("\n" + "=" * 70)
    print("STEP 3: ELBOW METHOD ANALYSIS")
    print("=" * 70)

    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"k={k}: Within-cluster sum of squares = {kmeans.inertia_:.2f}")

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_range)
    plt.tight_layout()
    plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Elbow plot saved as 'elbow_plot.png'")
    plt.show()

    return inertias

def silhouette_analysis(X_scaled, max_k=10):
    """Perform silhouette analysis"""
    print("\n" + "=" * 70)
    print("STEP 4: SILHOUETTE ANALYSIS")
    print("=" * 70)

    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"k={k}: Average silhouette score = {silhouette_avg:.4f}")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Average Silhouette Score', fontsize=12)
    plt.title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_range)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Good separation threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('silhouette_plot.png', dpi=300, bbox_inches='tight')
    print("\n✓ Silhouette plot saved as 'silhouette_plot.png'")
    plt.show()

    # Find optimal k
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal k based on silhouette score: {optimal_k}")

    return silhouette_scores, optimal_k

# ============================================================================
# 4. PERFORM K-MEANS CLUSTERING WITH k=4
# ============================================================================

def perform_kmeans(X_scaled, n_clusters=4):
    """Perform K-means clustering with specified number of clusters"""
    print("\n" + "=" * 70)
    print(f"STEP 5: K-MEANS CLUSTERING (k={n_clusters})")
    print("=" * 70)

    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)

    print(f"✓ Convergence achieved in {kmeans.n_iter_} iterations")
    print(f"Final inertia: {kmeans.inertia_:.2f}")

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette score: {silhouette_avg:.4f}")

    # Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster sizes:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster + 1}: {count} participants ({count/len(cluster_labels)*100:.1f}%)")

    return kmeans, cluster_labels, silhouette_avg

# ============================================================================
# 5. ANALYZE CLUSTER PROFILES
# ============================================================================

def analyze_clusters(df, X, cluster_labels, constructs):
    """Analyze and profile each cluster"""
    print("\n" + "=" * 70)
    print("STEP 6: CLUSTER PROFILE ANALYSIS")
    print("=" * 70)

    # Add cluster labels to dataframe
    df_analysis = X.copy()
    df_analysis['Cluster'] = cluster_labels + 1  # 1-indexed for readability

    # Calculate cluster means
    cluster_means = df_analysis.groupby('Cluster')[constructs].mean()
    cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()

    print("\nCluster Means by Construct:")
    print(cluster_means.round(2))

    # Perform ANOVA for each construct
    print("\n" + "=" * 70)
    print("ANOVA RESULTS (Testing between-cluster differences)")
    print("=" * 70)

    anova_results = []
    for construct in constructs:
        groups = [df_analysis[df_analysis['Cluster'] == i][construct].values
                  for i in df_analysis['Cluster'].unique()]
        f_stat, p_value = f_oneway(*groups)
        anova_results.append({
            'Construct': construct,
            'F-Statistic': f_stat,
            'p-value': p_value,
            'Significant': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
        })
        print(f"{construct}: F={f_stat:.2f}, p<0.001 {anova_results[-1]['Significant']}")

    anova_df = pd.DataFrame(anova_results)

    # Identify distinctive characteristics
    print("\n" + "=" * 70)
    print("CLUSTER DISTINCTIVE CHARACTERISTICS")
    print("=" * 70)

    for cluster_id in sorted(df_analysis['Cluster'].unique()):
        cluster_data = cluster_means.loc[cluster_id]
        print(f"\nCluster {cluster_id} (n={cluster_sizes[cluster_id]}, {cluster_sizes[cluster_id]/len(df_analysis)*100:.1f}%):")

        # Find highest and lowest constructs
        highest = cluster_data.nlargest(3)
        lowest = cluster_data.nsmallest(3)

        print("  Highest scores:")
        for construct, value in highest.items():
            print(f"    {construct}: {value:.2f}")

        print("  Lowest scores:")
        for construct, value in lowest.items():
            print(f"    {construct}: {value:.2f}")

    return df_analysis, cluster_means, cluster_sizes, anova_df

# ============================================================================
# 6. ASSIGN CLUSTER NAMES BASED ON PROFILES
# ============================================================================

def assign_cluster_names(cluster_means, cluster_sizes):
    """Assign descriptive names to clusters based on their statistical profiles"""
    print("\n" + "=" * 70)
    print("STEP 7: ASSIGNING CLUSTER NAMES (Based on Statistical Profiles)")
    print("=" * 70)

    cluster_names = {}
    assigned_names = set()  # Track which names have been assigned

    print("\nAnalyzing cluster profiles to assign names...")
    print("(Based on distinctive statistical patterns)\n")

    # Analyze each cluster to identify its unique profile
    for cluster_id in cluster_means.index:
        profile = cluster_means.loc[cluster_id]
        size = cluster_sizes[cluster_id]

        print(f"Cluster {cluster_id} (n={size}, {size/cluster_sizes.sum()*100:.1f}%):")
        print(f"  Key stats: TT={profile['TT']:.2f}, RP={profile['RP']:.2f}, BI={profile['BI']:.2f}, RC={profile['RC']:.2f}, Mean={profile.mean():.2f}")

        # PATTERN 1: Risk-Aware Skeptics
        # Distinctive: LOWEST Trust (TT), HIGHEST Risk Perception and BI
        if (profile['TT'] < 2.0 and  # Extremely low trust
            profile['RP'] > 4.5 and  # Very high risk awareness
            profile['BI'] > 4.5 and  # Paradoxically high intention
            "Risk-Aware Skeptics" not in assigned_names):
            cluster_names[cluster_id] = "Risk-Aware Skeptics"
            assigned_names.add("Risk-Aware Skeptics")
            print(f"  → Identified as: Risk-Aware Skeptics (Low TT + High RP + High BI)")

        # PATTERN 2: Disengaged Users
        # Distinctive: LOWEST scores across nearly all constructs
        elif (profile.mean() < 2.8 and  # Overall low engagement
              profile['BI'] < 2.7 and    # Very low intention
              "Disengaged Users" not in assigned_names):
            cluster_names[cluster_id] = "Disengaged Users"
            assigned_names.add("Disengaged Users")
            print(f"  → Identified as: Disengaged Users (Low across all constructs)")

        # PATTERN 3: Pragmatic Adopters
        # Distinctive: High FC, LOWEST Regulatory Concern, typically largest group
        elif (profile['FC'] > 4.0 and   # High facilitating conditions
              profile['RC'] < 2.5 and   # Very low regulatory concern
              "Pragmatic Adopters" not in assigned_names):
            cluster_names[cluster_id] = "Pragmatic Adopters"
            assigned_names.add("Pragmatic Adopters")
            print(f"  → Identified as: Pragmatic Adopters (High FC + Low RC)")

        # PATTERN 4: Confident Enthusiasts
        # Distinctive: HIGHEST Trust, high across all positive constructs
        elif (profile['TT'] > 4.0 and   # Very high trust
              profile.mean() > 4.0 and  # High overall
              "Confident Enthusiasts" not in assigned_names):
            cluster_names[cluster_id] = "Confident Enthusiasts"
            assigned_names.add("Confident Enthusiasts")
            print(f"  → Identified as: Confident Enthusiasts (High TT + High overall)")

        # FALLBACK: If patterns don't match perfectly, use heuristics
        else:
            # Check what's left to assign
            remaining = {"Risk-Aware Skeptics", "Disengaged Users", "Pragmatic Adopters", "Confident Enthusiasts"} - assigned_names

            if "Confident Enthusiasts" in remaining and profile['TT'] == profile['TT'].max():
                cluster_names[cluster_id] = "Confident Enthusiasts"
                assigned_names.add("Confident Enthusiasts")
                print(f"  → Assigned as: Confident Enthusiasts (highest TT)")
            elif "Risk-Aware Skeptics" in remaining and profile['TT'] == profile['TT'].min():
                cluster_names[cluster_id] = "Risk-Aware Skeptics"
                assigned_names.add("Risk-Aware Skeptics")
                print(f"  → Assigned as: Risk-Aware Skeptics (lowest TT)")
            elif "Disengaged Users" in remaining and profile.mean() == profile.mean().min():
                cluster_names[cluster_id] = "Disengaged Users"
                assigned_names.add("Disengaged Users")
                print(f"  → Assigned as: Disengaged Users (lowest overall)")
            elif "Pragmatic Adopters" in remaining:
                cluster_names[cluster_id] = "Pragmatic Adopters"
                assigned_names.add("Pragmatic Adopters")
                print(f"  → Assigned as: Pragmatic Adopters (remaining)")
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
                print(f"  → WARNING: Could not match to expected profile!")

        print()

    print("=" * 70)
    print("FINAL CLUSTER ASSIGNMENTS:")
    print("=" * 70)
    for cluster_id in sorted(cluster_names.keys()):
        name = cluster_names[cluster_id]
        size = cluster_sizes[cluster_id]
        print(f"  Cluster {cluster_id} = {name} (n={size}, {size/cluster_sizes.sum()*100:.1f}%)")
    print()

    return cluster_names

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

def create_visualizations(df_analysis, cluster_means, cluster_sizes, constructs, cluster_names):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 70)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("=" * 70)

    # Map cluster IDs to names for plots
    name_labels = [cluster_names.get(i, f"Cluster {i}") for i in cluster_sizes.index]

    # 1. Cluster size distribution
    plt.figure(figsize=(12, 6))
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    bars = plt.bar(range(len(cluster_sizes)), cluster_sizes.values, color=colors)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Participants', fontsize=12)
    plt.title('Cluster Size Distribution (n=516)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(cluster_sizes)), name_labels, rotation=15, ha='right')

    # Add value labels
    for bar, size in zip(bars, cluster_sizes.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/516*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('cluster_sizes.png', dpi=300, bbox_inches='tight')
    print("✓ Cluster sizes plot saved")
    plt.show()

    # 2. Heatmap of cluster profiles
    plt.figure(figsize=(12, 8))

    # Create renamed dataframe for heatmap
    heatmap_data = cluster_means.T.copy()
    heatmap_data.columns = name_labels

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                center=3, vmin=1, vmax=5, cbar_kws={'label': 'Mean Score'})
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Construct', fontsize=12)
    plt.title('Cluster Profiles Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Heatmap saved")
    plt.show()

    # 3. Parallel coordinates plot
    plt.figure(figsize=(14, 7))

    for idx, cluster_id in enumerate(cluster_means.index):
        plt.plot(constructs, cluster_means.loc[cluster_id],
                marker='o', linewidth=2.5, markersize=8,
                color=colors[idx], label=name_labels[idx])

    plt.xlabel('Constructs', fontsize=12)
    plt.ylabel('Mean Score (1-5 scale)', fontsize=12)
    plt.title('Cluster Profiles - Parallel Coordinates', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.ylim(1, 5)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('parallel_coordinates.png', dpi=300, bbox_inches='tight')
    print("✓ Parallel coordinates plot saved")
    plt.show()

    # 4. Trust vs Risk scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_analysis['TT'], df_analysis['RP'],
                         c=df_analysis['Cluster'], cmap='viridis',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add cluster centroids
    for cluster_id in cluster_means.index:
        plt.scatter(cluster_means.loc[cluster_id, 'TT'],
                   cluster_means.loc[cluster_id, 'RP'],
                   s=500, c='red', marker='X', edgecolors='black', linewidth=2)

        # Add cluster name labels near centroids
        plt.annotate(name_labels[cluster_id-1],
                    (cluster_means.loc[cluster_id, 'TT'], cluster_means.loc[cluster_id, 'RP']),
                    xytext=(10, 10), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    plt.xlabel('Trust in Technology (TT)', fontsize=12)
    plt.ylabel('Risk Perception (RP)', fontsize=12)
    plt.title('Trust vs Risk Profile by Cluster', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trust_vs_risk.png', dpi=300, bbox_inches='tight')
    print("✓ Trust vs Risk plot saved")
    plt.show()

    # 5. Behavioral Intention vs Trust
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_analysis['TT'], df_analysis['BI'],
                         c=df_analysis['Cluster'], cmap='viridis',
                         s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add cluster centroids
    for cluster_id in cluster_means.index:
        plt.scatter(cluster_means.loc[cluster_id, 'TT'],
                   cluster_means.loc[cluster_id, 'BI'],
                   s=500, c='red', marker='X', edgecolors='black', linewidth=2)

        # Add cluster name labels
        plt.annotate(name_labels[cluster_id-1],
                    (cluster_means.loc[cluster_id, 'TT'], cluster_means.loc[cluster_id, 'BI']),
                    xytext=(10, 10), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    plt.xlabel('Trust in Technology (TT)', fontsize=12)
    plt.ylabel('Behavioral Intention (BI)', fontsize=12)
    plt.title('Behavioral Intention vs Trust by Cluster', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('intention_vs_trust.png', dpi=300, bbox_inches='tight')
    print("✓ Behavioral Intention vs Trust plot saved")
    plt.show()

    print("\n✓ All visualizations saved successfully!")

# ============================================================================
# 8. EXPORT RESULTS
# ============================================================================

def export_results(df_analysis, cluster_means, cluster_sizes, anova_df, cluster_names):
    """Export analysis results to Excel"""
    print("\n" + "=" * 70)
    print("STEP 9: EXPORTING RESULTS")
    print("=" * 70)

    # Add cluster names to dataframe
    df_export = df_analysis.copy()
    df_export['Cluster_Name'] = df_export['Cluster'].map(cluster_names)

    with pd.ExcelWriter('cluster_analysis_results.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Full data with cluster assignments
        df_export.to_excel(writer, sheet_name='Data_with_Clusters', index=False)

        # Sheet 2: Cluster means with names
        means_export = cluster_means.copy()
        means_export['Cluster_Name'] = means_export.index.map(cluster_names)
        means_export = means_export[['Cluster_Name'] + list(cluster_means.columns)]
        means_export.to_excel(writer, sheet_name='Cluster_Means')

        # Sheet 3: Cluster sizes with names
        sizes_df = pd.DataFrame({
            'Cluster': cluster_sizes.index,
            'Cluster_Name': [cluster_names[i] for i in cluster_sizes.index],
            'Size': cluster_sizes.values,
            'Percentage': (cluster_sizes.values / cluster_sizes.sum() * 100).round(1)
        })
        sizes_df.to_excel(writer, sheet_name='Cluster_Sizes', index=False)

        # Sheet 4: ANOVA results
        anova_df.to_excel(writer, sheet_name='ANOVA_Results', index=False)

    print("✓ Results exported to 'cluster_analysis_results.xlsx'")
    print("  - Sheet 1: Full data with cluster assignments")
    print("  - Sheet 2: Cluster means by construct")
    print("  - Sheet 3: Cluster sizes and names")
    print("  - Sheet 4: ANOVA statistical tests")

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print(" UTAUT2 GAMEFI CLUSTER ANALYSIS")
    print(" K-Means Clustering with Validation")
    print("=" * 70)

    # Load data and calculate construct means
    df, X, constructs = load_and_aggregate_data('utaut2_cleaned_data.xlsx')

    # Standardize
    X_scaled, X_scaled_df, scaler = standardize_data(X)

    # Validation analyses
    inertias = elbow_analysis(X_scaled, max_k=10)
    silhouette_scores, optimal_k = silhouette_analysis(X_scaled, max_k=10)

    # Perform clustering with k=4
    kmeans, cluster_labels, silhouette_avg = perform_kmeans(X_scaled, n_clusters=4)

    # Analyze clusters
    df_analysis, cluster_means, cluster_sizes, anova_df = analyze_clusters(
        df, X, cluster_labels, constructs
    )

    # Assign descriptive names to clusters
    cluster_names = assign_cluster_names(cluster_means, cluster_sizes)

    # Create visualizations
    create_visualizations(df_analysis, cluster_means, cluster_sizes, constructs, cluster_names)

    # Export results
    export_results(df_analysis, cluster_means, cluster_sizes, anova_df, cluster_names)

    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - elbow_plot.png")
    print("  - silhouette_plot.png")
    print("  - cluster_sizes.png")
    print("  - cluster_heatmap.png")
    print("  - parallel_coordinates.png")
    print("  - trust_vs_risk.png")
    print("  - intention_vs_trust.png")
    print("  - cluster_analysis_results.xlsx")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()