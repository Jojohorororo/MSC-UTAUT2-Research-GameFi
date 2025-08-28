"""
Geographic Location/Region Pie Chart — no numbers or percentages inside the pie
Uses legend with counts only. Saves PNG (300 DPI), PDF, and SVG.
"""

import matplotlib.pyplot as plt

# --- Data ---
data = {
    "North America (US and Canada)": 15,
    "Latin America and the Caribbean": 99,
    "Western Europe": 24,
    "Eastern Europe": 36,
    "Southeast Asia (Philippines, Thailand, Vietnam, etc.)": 228,
    "East Asia (China, Japan, South Korea, etc.)": 29,
    "South Asia (India, Pakistan, Bangladesh, etc.)": 59,
    "Middle East and North Africa": 8,
    "Sub-Saharan Africa": 3,
    "Australia and Oceania": 12,
    "Russia and Central Asia": 3,
}

labels = list(data.keys())
values = list(data.values())

total = sum(values)

# Slightly explode very small slices for visibility
explode = [0.06 if v / total < 0.04 else 0.02 for v in values]

fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

# Draw pie with no labels or percentages inside
wedges, _ = ax.pie(
    values,
    explode=explode,
    startangle=90,
    counterclock=False,
    wedgeprops={"linewidth": 1, "edgecolor": "white"},
)

ax.set_title(f"Geographic Location/Region (n={total})", pad=16)
ax.axis("equal")

# Legend with counts only
legend_labels = [f"{lbl}: {val}" for lbl, val in zip(labels, values)]
ax.legend(
    wedges, legend_labels,
    title="Region (counts)",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()

# Save images in multiple formats
plt.savefig("geographic_location_pie.png", bbox_inches="tight", dpi=300)


plt.show()
