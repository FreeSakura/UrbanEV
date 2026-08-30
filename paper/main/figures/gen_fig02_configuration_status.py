import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, DATA_DIR, save_figure


overall = pd.read_csv(DATA_DIR / "status_overall.csv").sort_values("code")
by_table = pd.read_csv(DATA_DIR / "status_by_table.csv")
palette = [COLORS["blue"], COLORS["green"], COLORS["yellow"], COLORS["orange"], COLORS["vermillion"], COLORS["gray"]]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), gridspec_kw={"width_ratios": [1.0, 1.8]})

left = 0.0
for (_, row), color in zip(overall.iterrows(), palette):
    axes[0].barh([0], row["percent"], left=left, color=color, edgecolor="white", linewidth=0.8, height=0.48)
    if row["percent"] >= 7:
        axes[0].text(left + row["percent"] / 2, 0, f"{row['code']}\n{int(row['count'])}", ha="center", va="center", fontsize=8)
    if row["code"] == "S5":
        axes[0].annotate("S5 19", xy=(left + row["percent"] / 2, 0.24), xytext=(left + row["percent"] / 2, 0.58), ha="center", va="bottom", fontsize=7, arrowprops={"arrowstyle": "-", "color": COLORS["vermillion"], "lw": 0.7})
    left += row["percent"]
axes[0].set_xlim(0, 100)
axes[0].set_yticks([0], ["All 1,224"])
axes[0].set_xlabel("Share of frozen configurations (%)")

tables = sorted(by_table["table"].unique())
for y, table in enumerate(tables):
    subset = by_table[by_table["table"] == table].sort_values("code")
    left = 0.0
    for (_, row), color in zip(subset.iterrows(), palette):
        axes[1].barh([y], row["percent"], left=left, color=color, edgecolor="white", linewidth=0.8, height=0.56)
        if row["count"] >= 12:
            axes[1].text(left + row["percent"] / 2, y, f"{row['code']}\n{int(row['count'])}", ha="center", va="center", fontsize=7)
        left += row["percent"]
axes[1].set_xlim(0, 100)
axes[1].set_yticks(range(len(tables)), [f"Source Table {int(t)}" for t in tables])
axes[1].set_xlabel("Within-table share (%)")
axes[1].invert_yaxis()

handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in palette]
fig.legend(handles, overall["code"].tolist(), loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.92))
save_figure(fig, "fig02_configuration_status")
