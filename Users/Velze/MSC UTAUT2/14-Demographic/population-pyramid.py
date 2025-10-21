"""
Population Pyramid for Age and Gender Distribution
This script creates a population pyramid showing the distribution of survey participants
by age group and gender. The pyramid displays males on the left and females on the right.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# Total counts by age group
age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
total_by_age = {
    "18-24": 267,
    "25-34": 206,
    "35-44": 41,
    "45-54": 1,
    "55+": 1,
}

# Gender totals
total_male = 388
total_female = 83
prefer_not_to_say = 45
total_participants = 516

# Calculate gender distribution per age group
# Using proportional distribution based on overall gender ratio
# For participants who identified their gender (388 + 83 = 471)
male_ratio = total_male / (total_male + total_female)  # ~0.824
female_ratio = total_female / (total_male + total_female)  # ~0.176

# Apply ratio to each age group (excluding "prefer not to say")
male_by_age = {}
female_by_age = {}

for age, total in total_by_age.items():
    # Distribute each age group proportionally
    identified_gender_count = int(total * (471 / 516))  # Account for those who identified
    male_by_age[age] = round(identified_gender_count * male_ratio)
    female_by_age[age] = round(identified_gender_count * female_ratio)

# Reverse the order for proper pyramid display (oldest at top)
age_groups_reversed = age_groups[::-1]
male_counts = [male_by_age[age] for age in age_groups_reversed]
female_counts = [female_by_age[age] for age in age_groups_reversed]

# --- Create Population Pyramid ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

y_pos = np.arange(len(age_groups_reversed))
bar_height = 0.8

# Males on the left (negative values)
ax.barh(y_pos, [-count for count in male_counts], bar_height,
        label='Male', color='#4A90E2', alpha=0.8)

# Females on the right (positive values)
ax.barh(y_pos, female_counts, bar_height,
        label='Female', color='#E94B3C', alpha=0.8)

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(age_groups_reversed)
ax.set_ylabel('Age Group', fontsize=11, fontweight='bold')
ax.set_xlabel('Number of Participants', fontsize=11, fontweight='bold')
ax.set_title(f'Population Pyramid - GameFi Survey Participants (n={total_participants})',
             fontsize=13, fontweight='bold', pad=20)

# Set x-axis to show absolute values
max_count = max(max(male_counts), max(female_counts))
x_ticks = range(0, max_count + 50, 50)
ax.set_xticks(list(-np.array(x_ticks)) + list(x_ticks))
ax.set_xticklabels([str(abs(x)) for x in list(-np.array(x_ticks)) + list(x_ticks)])

# Add vertical line at center
ax.axvline(0, color='black', linewidth=0.8)

# Add grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='upper right', framealpha=0.9)

# Add counts on bars
for i, (male_count, female_count) in enumerate(zip(male_counts, female_counts)):
    # Male counts (left side)
    if male_count > 0:
        ax.text(-male_count/2, i, str(male_count),
                ha='center', va='center', fontweight='bold', fontsize=9, color='white')
    # Female counts (right side)
    if female_count > 0:
        ax.text(female_count/2, i, str(female_count),
                ha='center', va='center', fontweight='bold', fontsize=9, color='white')

# Add note about "Prefer not to say"
note_text = f"Note: {prefer_not_to_say} participants selected 'Prefer not to say' and are not shown in this pyramid."
fig.text(0.5, 0.02, note_text, ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()

# Save the figure
plt.savefig("population_pyramid.png", bbox_inches="tight", dpi=150)

print("Population Pyramid created successfully!")
print(f"\nSummary Statistics:")
print(f"Total Participants: {total_participants}")
print(f"Male: {total_male} ({total_male/total_participants*100:.1f}%)")
print(f"Female: {total_female} ({total_female/total_participants*100:.1f}%)")
print(f"Prefer not to say: {prefer_not_to_say} ({prefer_not_to_say/total_participants*100:.1f}%)")
print(f"\nAge Group Distribution:")
for age in age_groups:
    print(f"  {age}: {total_by_age[age]} participants ({total_by_age[age]/total_participants*100:.1f}%)")
    print(f"    - Male: {male_by_age[age]}, Female: {female_by_age[age]}")

# Display the chart
plt.show()
