import textwrap

import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, DATA_DIR, save_figure


data = pd.read_csv(DATA_DIR / "audit_timeline.csv")
lanes = list(data["lane"].drop_duplicates())
color_map = {"locked": COLORS["gray"], "warning": COLORS["orange"], "fail": COLORS["vermillion"], "repair": COLORS["sky"], "pass": COLORS["green"], "stop": COLORS["purple"]}

fig, ax = plt.subplots(figsize=(8.6, 4.5))
for lane_index, lane in enumerate(lanes):
    subset = data[data["lane"] == lane].sort_values("order")
    y = 1.0 - lane_index
    xs = list(range(len(subset)))
    ax.plot(xs, [y] * len(xs), color="#BBBBBB", linewidth=1.2, zorder=1)
    for x, (_, row) in zip(xs, subset.iterrows()):
        color = color_map[row["state"]]
        ax.scatter(x, y, s=100, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        offset = 0.23 if x % 2 == 0 else -0.23
        va = "bottom" if offset > 0 else "top"
        ax.text(x, y + offset, "\n".join(textwrap.wrap(row["label"], width=19)), ha="center", va=va, fontsize=8.0, color=COLORS["dark"])
    ax.text(-0.55, y, lane, ha="right", va="center", fontweight="bold", fontsize=9)

ax.set_xlim(-0.8, max(data.groupby("lane").size()) - 0.45)
ax.set_ylim(-0.75, 1.75)
ax.axis("off")
handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[key], markeredgecolor="white", markersize=8) for key in ["locked", "fail", "repair", "pass", "stop"]]
ax.legend(handles, ["metric-locked", "blocked/invalid", "repair", "audit pass", "stopped by gate"], loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False)
fig.tight_layout()
save_figure(fig, "fig05_audit_timeline")
