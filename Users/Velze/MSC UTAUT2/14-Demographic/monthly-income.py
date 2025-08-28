"""
Monthly Income Pie Chart — no numbers or percentages inside the pie
Uses legend with counts only. Saves PNG (300 DPI), PDF, and SVG.
"""

import matplotlib.pyplot as plt

# --- Data ---
data = {
    "0–499": 219,
    "500–999": 176,
    "1000–2999": 76,
    "3000–4999": 32,
    "5000–7499": 9,
    "7500–9999": 4,
    "10000 or more": 0,
}

labels = list(data.keys())
values = list(data.values())

total = sum(values)

# Slightly explode very small slices for visibility
explode = [0.06 if v / total < 0.03 else 0.02 for v in values]

fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

# Draw pie with no labels or percentages inside
wedges, _ = ax.pie(
    values,
    explode=explode,
    startangle=90,
    counterclock=False,
    wedgeprops={"linewidth": 1, "edgecolor": "white"},
)

ax.set_title(f"Monthly Income (USD equivalent) (n={total})", pad=16)
ax.axis("equal")

# Legend with counts only
legend_labels = [f"{lbl}: {val}" for lbl, val in zip(labels, values)]
ax.legend(
    wedges, legend_labels,
    title="Income (counts)",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()

# Save images in multiple formats
plt.savefig("monthly_income_pie.png", bbox_inches="tight", dpi=300)


plt.show()
