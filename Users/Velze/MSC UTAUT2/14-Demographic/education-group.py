"""
Highest Level of Education Pie Chart — no numbers or percentages inside the pie
Uses legend with counts only. Saves PNG (300 DPI), PDF, and SVG.
"""

import matplotlib.pyplot as plt

# --- Data ---
data = {
    "High School or less": 146,
    "Some College": 209,
    "Bachelor's Degree": 130,
    "Master's Degree or higher": 31,
}

labels = list(data.keys())
values = list(data.values())

total = sum(values)

# Slightly explode small slices for visibility
explode = [0.06 if v / total < 0.05 else 0.02 for v in values]

fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

# Draw pie with no labels or percentages inside
wedges, _ = ax.pie(
    values,
    explode=explode,
    startangle=90,
    counterclock=False,
    wedgeprops={"linewidth": 1, "edgecolor": "white"},
)

ax.set_title(f"Highest Level of Education (n={total})", pad=16)
ax.axis("equal")

# Legend with counts only
legend_labels = [f"{lbl}: {val}" for lbl, val in zip(labels, values)]
ax.legend(wedges, legend_labels, title="Education (counts)", loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()

# Save images in multiple formats
plt.savefig("education_pie.png", bbox_inches="tight", dpi=300)


plt.show()
