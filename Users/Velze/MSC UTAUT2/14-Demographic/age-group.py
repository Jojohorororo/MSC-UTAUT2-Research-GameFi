"""
Age Group Pie Chart — no numbers or percentages inside the pie
This script removes all slice labels/autopct text and uses a legend with counts only.
"""

import matplotlib.pyplot as plt

# --- Data ---
data = {
    "18–24": 267,
    "25–34": 206,
    "35–44": 41,
    "45–54": 1,
    "55+": 1,
}

labels = list(data.keys())
values = list(data.values())

total = sum(values)

# Optional: slightly explode tiny slices so they are still visible
explode = [0.06 if v / total < 0.02 else 0.02 for v in values]

fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

# Draw pie with NO labels and NO autopct (so nothing appears inside the pie)
wedges, texts = ax.pie(
    values,
    explode=explode,
    startangle=90,
    counterclock=False,
    wedgeprops={"linewidth": 1, "edgecolor": "white"},
)

ax.set_title(f"Age Group Distribution (n={total})", pad=16)
ax.axis("equal")  # Keep it circular

# Legend with counts only (no percentages)
legend_labels = [f"{lbl}: {val}" for lbl, val in zip(labels, values)]
ax.legend(wedges, legend_labels, title="Age Group (counts)", loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()

# Save a high-resolution image to the current directory
plt.savefig("age_group_pie.png", bbox_inches="tight")

# Display the chart
plt.show()
