import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def create_colab_chart(response_counts, question_name, mean_score, std_score):
    """
    Create chart optimized for PyCharm
    """
    # Data for the chart
    categories = ['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree']
    counts = [response_counts[1], response_counts[2], response_counts[3], response_counts[4], response_counts[5]]

    # Colors  Google Forms style
    colors = ['#4285f4', '#ea4335', '#fbbc04', '#34a853', '#9c27b0']

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(categories, counts, color=colors, alpha=0.85, edgecolor='white', linewidth=2)

    # Customize the chart
    plt.title(f'Response Distribution\n{question_name}\nMean: {mean_score:.2f} | SD: {std_score:.2f}',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Response Categories', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Responses', fontsize=12, fontweight='bold')

    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total) * 100
        if height > 20:  # Only show percentage if bar is tall enough
            plt.text(bar.get_x() + bar.get_width() / 2., height / 2,
                     f'{percentage:.1f}%', ha='center', va='center',
                     color='white', fontweight='bold', fontsize=11)

    # Styling
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().set_facecolor('#f8f9fa')

    # Add a subtle border
    for spine in plt.gca().spines.values():
        spine.set_color('#ddd')
        spine.set_linewidth(1)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85)
    plt.show()


def create_distribution_histogram(response_counts, question_name, mean_score, std_score):
    """
    Create histogram with normal distribution overlay
    """
    # Reconstruct raw data from counts
    raw_data = []
    for score, count in response_counts.items():
        raw_data.extend([score] * count)

    raw_data = np.array(raw_data)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Histogram with normal overlay
    ax1.hist(raw_data, bins=np.arange(0.5, 6.5, 1), alpha=0.7, color='skyblue',
             edgecolor='black', density=True, label='Actual Data')

    # Overlay normal distribution
    x = np.linspace(1, 5, 100)
    normal_curve = stats.norm.pdf(x, mean_score, std_score)
    ax1.plot(x, normal_curve, 'r-', linewidth=3, label=f'Normal Distribution\n(μ={mean_score:.2f}, σ={std_score:.2f})')

    ax1.set_xlabel('Response Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Response Distribution vs Normal Curve', fontweight='bold')
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xticklabels(['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Frequency histogram
    counts = [response_counts[i] for i in range(1, 6)]
    colors = ['#4285f4', '#ea4335', '#fbbc04', '#34a853', '#9c27b0']
    bars = ax2.bar(range(1, 6), counts, color=colors, alpha=0.8, edgecolor='black')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    ax2.set_xlabel('Response Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Response Frequency Distribution', fontweight='bold')
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels(['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree'])
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Distribution Analysis: {question_name}', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85)
    plt.show()

    # Statistical tests
    print(f"Distribution Analysis:")
    print(f"   • Skewness: {stats.skew(raw_data):.3f} (0 = symmetric, <0 = left-skewed, >0 = right-skewed)")
    print(f"   • Kurtosis: {stats.kurtosis(raw_data):.3f} (0 = normal, >0 = peaked, <0 = flat)")

    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(raw_data)
    print(f"   • Normality Test (Shapiro-Wilk): p = {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        print(f"     ** Data appears normally distributed (p > 0.05)")
    else:
        print(f"     ** Data may not be normally distributed (p ≤ 0.05)")


def create_box_plot(response_counts, question_name, mean_score, std_score):
    """
    Create box plot showing quartiles and outliers
    """
    #  raw data from counts
    raw_data = []
    for score, count in response_counts.items():
        raw_data.extend([score] * count)

    raw_data = np.array(raw_data)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left plot: Vertical box plot -  matplotlib deprecation
    box_parts = ax1.boxplot(raw_data, patch_artist=True, tick_labels=['Responses'])

    # Color the box
    box_parts['boxes'][0].set_facecolor('lightblue')
    box_parts['boxes'][0].set_alpha(0.7)

    # Add statistical annotations
    q1, median, q3 = np.percentile(raw_data, [25, 50, 75])
    iqr = q3 - q1

    #  avoid overlap
    y_positions = [q1, median, q3, mean_score]
    y_positions.sort()

    #  spacing between label = min
    min_spacing = 0.15
    for i in range(1, len(y_positions)):
        if y_positions[i] - y_positions[i - 1] < min_spacing:
            y_positions[i] = y_positions[i - 1] + min_spacing

    #  mapping of original values
    pos_map = {}
    sorted_vals = [q1, median, q3, mean_score]
    sorted_vals.sort()
    for i, val in enumerate(sorted_vals):
        pos_map[val] = y_positions[i]

    #  annotations  positioning
    ax1.annotate(f'Q1: {q1:.2f}', xy=(1, q1), xytext=(1.4, pos_map.get(q1, q1)),
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6),
                 va='center', fontsize=10)

    ax1.annotate(f'Median: {median:.2f}', xy=(1, median), xytext=(1.4, pos_map.get(median, median)),
                 arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
                 va='center', fontweight='bold', fontsize=10, color='blue')

    ax1.annotate(f'Q3: {q3:.2f}', xy=(1, q3), xytext=(1.4, pos_map.get(q3, q3)),
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6),
                 va='center', fontsize=10)

    ax1.annotate(f'Mean: {mean_score:.2f}', xy=(1, mean_score), xytext=(1.4, pos_map.get(mean_score, mean_score)),
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                 va='center', color='red', fontweight='bold', fontsize=10)

    #  mean = red diamond
    ax1.scatter(1, mean_score, marker='D', color='red', s=100, zorder=5, label='Mean')

    ax1.set_ylabel('Response Score', fontweight='bold')
    ax1.set_title('Box Plot Analysis', fontweight='bold')
    ax1.set_ylim(0.5, 5.5)
    ax1.set_xlim(0.5, 2.0)
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.set_yticklabels(['Strongly\nDisagree', 'Disagree', 'Neutral', 'Agree', 'Strongly\nAgree'])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Statistics summary
    ax2.axis('off')

    # statistics summary box
    stats_text = f"""Box Plot Statistics

Position Measures:
   • Minimum: {np.min(raw_data):.2f}
   • Q1 (25th percentile): {q1:.2f}
   • Median (50th percentile): {median:.2f}
   • Q3 (75th percentile): {q3:.2f}
   • Maximum: {np.max(raw_data):.2f}

Spread Measures:
   • Mean: {mean_score:.2f}
   • Standard Deviation: {std_score:.2f}
   • Interquartile Range (IQR): {iqr:.2f}
   • Range: {np.max(raw_data) - np.min(raw_data):.2f}

Key Insights:
   • 50% of responses: {q1:.2f} - {q3:.2f}
   • Middle 50% spread: {iqr:.2f} points
   • Distribution: {'Symmetric' if abs(mean_score - median) < 0.1 else 'Right-skewed' if mean_score > median else 'Left-skewed'}"""

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", alpha=0.9, edgecolor='#dee2e6'),
             fontsize=11, verticalalignment='top', fontfamily='Arial')

    # Add a small violin plot at the bottom right for distribution shape
    ax2_violin = fig.add_axes([0.7, 0.15, 0.25, 0.15])
    violin_parts = ax2_violin.violinplot(raw_data, vert=False, positions=[0],
                                         showmeans=True, showmedians=True)

    # Color the violin
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.6)

    ax2_violin.set_xlabel('Response Score', fontsize=9)
    ax2_violin.set_title('Distribution Shape', fontsize=10, fontweight='bold')
    ax2_violin.set_xlim(0.5, 5.5)
    ax2_violin.set_xticks([1, 2, 3, 4, 5])
    ax2_violin.set_yticks([])
    ax2_violin.grid(True, alpha=0.3)

    plt.suptitle(f'Box Plot Analysis: {question_name}', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.9)
    plt.show()

    # Enhanced interpretations
    print(f"Box Plot Insights:")
    if abs(mean_score - median) < 0.1:
        print(f"   ** Mean ≈ Median ({mean_score:.2f} ≈ {median:.2f}): Symmetric distribution")
        distribution_type = "symmetric"
    elif mean_score > median:
        print(f"   ** Mean > Median ({mean_score:.2f} > {median:.2f}): Right-skewed (more high scores)")
        distribution_type = "right-skewed"
    else:
        print(f"   ** Mean < Median ({mean_score:.2f} < {median:.2f}): Left-skewed (more low scores)")
        distribution_type = "left-skewed"

    # Enhanced variability interpretation
    if iqr < 1.0:
        variability = "Low variability - responses are quite consistent"
    elif iqr < 2.0:
        variability = "Moderate variability - reasonable spread in responses"
    else:
        variability = "High variability - wide range of opinions"

    print(f"   • IQR = {iqr:.2f}: {variability}")
    print(f"   • 50% of responses fall between {q1:.2f} and {q3:.2f}")
    print(f"   • Distribution shape: {distribution_type}")

    # Outlier detection
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = raw_data[(raw_data < lower_fence) | (raw_data > upper_fence)]

    if len(outliers) > 0:
        print(f"   ** Potential outliers detected: {len(outliers)} responses outside normal range")
    else:
        print(f"   ** No outliers detected - all responses within expected range")


def analyze_survey_question(strongly_disagree, disagree, neutral, agree, strongly_agree,
                            question_name="Survey Question", show_charts=True):
    """
    Complete survey analysis function
    """
    # Store the counts
    response_counts = {
        1: strongly_disagree,
        2: disagree,
        3: neutral,
        4: agree,
        5: strongly_agree
    }

    # Calculate total responses
    total_responses = sum(response_counts.values())

    # Calculate mean (weighted average)
    weighted_sum = (1 * strongly_disagree +
                    2 * disagree +
                    3 * neutral +
                    4 * agree +
                    5 * strongly_agree)

    mean_score = weighted_sum / total_responses if total_responses > 0 else 0

    # Calculate standard deviation
    variance = (((1 - mean_score) ** 2 * strongly_disagree) +
                ((2 - mean_score) ** 2 * disagree) +
                ((3 - mean_score) ** 2 * neutral) +
                ((4 - mean_score) ** 2 * agree) +
                ((5 - mean_score) ** 2 * strongly_agree)) / total_responses

    std_score = np.sqrt(variance)

    # Display results
    print(f"ANALYSIS: {question_name}")
    print("=" * 60)

    print(f"\nRESULTS:")
    print(f"Mean: {mean_score:.2f}")
    print(f"SD: {std_score:.2f}")
    print(f"Total Responses: {total_responses}")

    print(f"\nResponse Distribution:")
    scale_labels = {1: "Strongly Disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly Agree"}

    for score in [1, 2, 3, 4, 5]:
        count = response_counts[score]
        percentage = (count / total_responses) * 100 if total_responses > 0 else 0
        print(f"  {score} ({scale_labels[score]}): {count} responses ({percentage:.1f}%)")

    # Calculate agree vs disagree
    total_disagree = strongly_disagree + disagree
    total_agree = agree + strongly_agree
    disagree_pct = (total_disagree / total_responses) * 100
    agree_pct = (total_agree / total_responses) * 100
    neutral_pct = (neutral / total_responses) * 100

    print(f"\nSUMMARY:")
    print(f"  Total Disagree: {total_disagree} ({disagree_pct:.1f}%)")
    print(f"  Neutral: {neutral} ({neutral_pct:.1f}%)")
    print(f"  Total Agree: {total_agree} ({agree_pct:.1f}%)")

    # Interpretation
    print(f"\nINTERPRETATION:")
    if mean_score < 2.5:
        interpretation = "MOSTLY DISAGREE - People generally disagree"
    elif mean_score < 3.0:
        interpretation = "LEAN DISAGREE - People tend to disagree"
    elif mean_score < 3.5:
        interpretation = "SLIGHTLY POSITIVE - Mixed feelings, slightly above neutral"
    elif mean_score < 4.0:
        interpretation = "MOSTLY AGREE - People generally agree"
    else:
        interpretation = "STRONGLY AGREE - High agreement"

    print(interpretation)

    # Create visualizations
    if show_charts:
        print(f"\nCREATING VISUALIZATIONS...")
        print("=" * 60)

        # 1. Basic bar chart
        print("\n1. Basic Response Distribution Chart")
        create_colab_chart(response_counts, question_name, mean_score, std_score)

        # 2. Distribution histogram
        print("\n2. Distribution Curve/Histogram")
        create_distribution_histogram(response_counts, question_name, mean_score, std_score)

        # 3. Box plot - NOW FIXED!
        print("\n3. Box Plot Analysis (FIXED - No More Overlapping!)")
        create_box_plot(response_counts, question_name, mean_score, std_score)

        print(f"\nAll visualizations created successfully!")

    return {
        'mean': mean_score,
        'std': std_score,
        'total': total_responses,
        'counts': response_counts,
        'interpretation': interpretation
    }


# Quick-use function for easy analysis
def quick_analysis(strongly_disagree, disagree, neutral, agree, strongly_agree, question_name="Question",
                   show_charts=True):
    """
    Quick one-line analysis function with all visualizations
    """
    return analyze_survey_question(strongly_disagree, disagree, neutral, agree, strongly_agree, question_name,
                                   show_charts)


# Test the code
if __name__ == "__main__":
    print("TESTING CLEAN SURVEY ANALYSIS CODE")
    print("=" * 50)

    # change arguments from line 403-409 to change topic
    # Test with Performance Expectancy data
    result = analyze_survey_question(
        strongly_disagree=13,
        disagree=19,
        neutral=114,
        agree=220,
        strongly_agree=150,
        question_name="Using GameFi gives me greater ownership and control over my in-game assets.)",
        show_charts=True
    )

    print(f"\nAnalysis Complete")
    print(f"Quick Summary: {result['interpretation']}")
    print(f"Mean: {result['mean']:.2f} | SD: {result['std']:.2f}")

    # change arguments from line 403-409 to change topic
