import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def create_user_behavior_pie_chart(data_counts, categories, question_name, chart_colors=None):
    """
    Create pie chart for user behavior data

    Parameters:
    data_counts: list of counts for each category
    categories: list of category labels
    question_name: title for the chart
    chart_colors: optional list of colors (if None, uses default palette)
    """

    # Calculate total responses
    total_responses = sum(data_counts)

    # Calculate percentages
    percentages = [(count / total_responses) * 100 for count in data_counts]

    # Default color palette if none provided
    if chart_colors is None:
        chart_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf']

    # Create the pie chart
    plt.figure(figsize=(12, 8))

    # Create pie chart with custom styling
    wedges, texts, autotexts = plt.pie(data_counts,
                                       labels=categories,
                                       colors=chart_colors[:len(categories)],
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       explode=[0.05 if i == data_counts.index(max(data_counts)) else 0 for i in
                                                range(len(data_counts))],
                                       shadow=True,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})

    # Customize the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    # Customize category labels
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    plt.title(f'{question_name}\nTotal Responses: {total_responses}',
              fontsize=14, fontweight='bold', pad=20)

    # Add a legend with counts and percentages
    legend_labels = [f'{cat}: {count} ({pct:.1f}%)'
                     for cat, count, pct in zip(categories, data_counts, percentages)]

    plt.legend(wedges, legend_labels,
               title="Response Breakdown",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1),
               fontsize=10)

    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nUSER BEHAVIOR ANALYSIS: {question_name}")
    print("=" * 60)
    print(f"Total Responses: {total_responses}")
    print(f"\nDetailed Breakdown:")

    for i, (category, count, percentage) in enumerate(zip(categories, data_counts, percentages)):
        print(f"  {i + 1}. {category}: {count} responses ({percentage:.1f}%)")

    # Find the most common response
    max_index = data_counts.index(max(data_counts))
    print(f"\nMost Common Response: {categories[max_index]} ({percentages[max_index]:.1f}%)")

    return {
        'total_responses': total_responses,
        'data_counts': data_counts,
        'categories': categories,
        'percentages': percentages,
        'most_common': categories[max_index]
    }


def create_usage_frequency_chart(never, monthly_or_less, few_times_month, weekly, several_times_week, daily):
    """
    Specific function for GameFi usage frequency analysis
    """
    categories = ['Never', 'Monthly or less', 'A few times a month', 'Weekly', 'Several times a week', 'Daily']
    data_counts = [never, monthly_or_less, few_times_month, weekly, several_times_week, daily]

    # Use colors similar to the original charts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

    question_name = "How often do you currently use GameFi platforms?"

    return create_user_behavior_pie_chart(data_counts, categories, question_name, colors)


def create_time_spent_chart(zero_hours, less_than_1, one_to_5, six_to_10, eleven_to_20, more_than_20):
    """
    Specific function for time spent analysis
    """
    categories = ['0 hours', 'Less than 1 hour', '1-5 hours', '6-10 hours', '11-20 hours', 'More than 20 hours']
    data_counts = [zero_hours, less_than_1, one_to_5, six_to_10, eleven_to_20, more_than_20]

    # Use different colors for time categories
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']

    question_name = "On average, how many hours per week do you spend on GameFi platforms?"

    return create_user_behavior_pie_chart(data_counts, categories, question_name, colors)


def create_custom_behavior_chart(data_counts, categories, question_name, colors=None):
    """
    General function for any custom user behavior data
    """
    return create_user_behavior_pie_chart(data_counts, categories, question_name, colors)


def analyze_user_engagement(usage_freq_data, time_spent_data):
    """
    Combined analysis of usage frequency and time spent
    """
    print("COMPREHENSIVE USER ENGAGEMENT ANALYSIS")
    print("=" * 60)

    # Analyze usage frequency
    freq_result = create_usage_frequency_chart(*usage_freq_data)

    print("\n" + "=" * 60)

    # Analyze time spent
    time_result = create_time_spent_chart(*time_spent_data)

    # Calculate engagement scores
    print("\n" + "=" * 60)
    print("ENGAGEMENT INSIGHTS:")

    # Usage frequency score (0-5 scale)
    freq_weights = [0, 1, 2, 3, 4, 5]
    freq_score = sum(count * weight for count, weight in zip(usage_freq_data, freq_weights)) / sum(usage_freq_data)

    # Time spent score (0-5 scale)
    time_weights = [0, 1, 2, 3, 4, 5]
    time_score = sum(count * weight for count, weight in zip(time_spent_data, time_weights)) / sum(time_spent_data)

    print(f"Average Usage Frequency Score: {freq_score:.2f}/5.0")
    print(f"Average Time Investment Score: {time_score:.2f}/5.0")

    # Overall engagement
    overall_engagement = (freq_score + time_score) / 2
    print(f"Overall Engagement Score: {overall_engagement:.2f}/5.0")

    if overall_engagement >= 4.0:
        engagement_level = "Very High Engagement"
    elif overall_engagement >= 3.0:
        engagement_level = "High Engagement"
    elif overall_engagement >= 2.0:
        engagement_level = "Moderate Engagement"
    elif overall_engagement >= 1.0:
        engagement_level = "Low Engagement"
    else:
        engagement_level = "Very Low Engagement"

    print(f"Engagement Level: {engagement_level}")

    return {
        'frequency_analysis': freq_result,
        'time_analysis': time_result,
        'engagement_score': overall_engagement,
        'engagement_level': engagement_level
    }


# Actual survey data analysis
if __name__ == "__main__":
    print("USER BEHAVIOR PIE CHART ANALYSIS - ACTUAL SURVEY DATA")
    print("=" * 60)

    # Actual Usage Frequency Data
    print("\n1. USAGE FREQUENCY ANALYSIS")
    usage_freq_result = create_usage_frequency_chart(
        never=9,
        monthly_or_less=27,
        few_times_month=19,
        weekly=43,
        several_times_week=137,
        daily=281
    )

    print("\n" + "=" * 60)

    # Actual Time Spent Analysis
    print("\n2. TIME SPENT ANALYSIS")
    time_spent_result = create_time_spent_chart(
        zero_hours=10,
        less_than_1=35,
        one_to_5=152,
        six_to_10=140,
        eleven_to_20=77,
        more_than_20=102
    )

    print("\n" + "=" * 60)

    # Combined Analysis with actual data
    print("\n3. COMBINED ENGAGEMENT ANALYSIS")
    combined_result = analyze_user_engagement(
        usage_freq_data=[9, 27, 19, 43, 137, 281],
        time_spent_data=[10, 35, 152, 140, 77, 102]
    )

    print(f"\nActual Survey Analysis Complete!")
    print(f"Total respondents for usage frequency: {sum([9, 27, 19, 43, 137, 281])}")
    print(f"Total respondents for time spent: {sum([10, 35, 152, 140, 77, 102])}")
    print(f"All pie charts generated with real survey data!")