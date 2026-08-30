import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from paper_plot_style import COLORS, DATA_DIR, save_figure


status = pd.read_csv(DATA_DIR / "status_overall.csv")
ladder = pd.read_csv(DATA_DIR / "evidence_ladder.csv").sort_values("order")
total = int(status["count"].sum())
completed = int(status.loc[status["is_completed_artifact"], "count"].sum())
exact = int(status.loc[status["code"] == "S1", "count"].iloc[0])

fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

top_boxes = [(0.35, f"{total:,}\nregistered configurations"), (4.45, f"{completed:,}\ncompleted artifacts"), (8.55, f"{exact:,}\nS1 exact reproductions")]
for idx, (x, label) in enumerate(top_boxes):
    box = FancyBboxPatch((x, 4.55), 2.75, 0.85, boxstyle="round,pad=0.04", facecolor="#EEF3F7", edgecolor=COLORS["gray"], linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + 1.375, 4.98, label, ha="center", va="center", fontsize=11)
    if idx < 2:
        ax.add_patch(FancyArrowPatch((x + 2.75, 4.98), (top_boxes[idx + 1][0], 4.98), arrowstyle="-|>", mutation_scale=13, color=COLORS["gray"], linewidth=1.2))

palette = [COLORS["blue"], COLORS["green"], COLORS["yellow"], COLORS["orange"], COLORS["vermillion"], COLORS["gray"]]
bar_x, bar_y, bar_w, bar_h = 0.35, 3.85, 11.0, 0.34
cursor = bar_x
for (_, row), color in zip(status.sort_values("code").iterrows(), palette):
    width = bar_w * row["count"] / total
    ax.add_patch(Rectangle((cursor, bar_y), width, bar_h, facecolor=color, edgecolor="white", linewidth=0.7))
    ax.text(cursor + width / 2, bar_y - 0.12, f"{row['code']} {int(row['count'])}", ha="center", va="top", fontsize=7)
    cursor += width

stage_colors = {"diagnostic": COLORS["gray"], "pass": COLORS["blue"], "supported": COLORS["green"], "cost": COLORS["orange"], "stop": COLORS["vermillion"]}
xs = [0.35 + i * 1.43 for i in range(len(ladder))]
for idx, ((_, row), x) in enumerate(zip(ladder.iterrows(), xs)):
    color = stage_colors[row["status"]]
    if row["unit"] == "AUC":
        value = f"AUC {row['value']:.3f}"
    elif row["stage"] == "Latency cost":
        value = f"+{row['value']:.2f}%"
    else:
        value = f"{row['value']:+.3f}%"
    display_stage = "Proxy screen" if row["stage"] == "State-feature gate" else row["stage"]
    display_status = "ROUTER TEST" if row["stage"] == "State-feature gate" else row["status"].upper()
    label = "\n".join(textwrap.wrap(display_stage, width=16))
    box = FancyBboxPatch((x, 1.35), 1.16, 1.32, boxstyle="round,pad=0.03", facecolor="white", edgecolor=color, linewidth=1.6)
    ax.add_patch(box)
    ax.text(x + 0.58, 2.28, label, ha="center", va="center", fontsize=8)
    ax.text(x + 0.58, 1.78, value, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    ax.text(x + 0.58, 1.48, display_status, ha="center", va="center", fontsize=6.5, color=color)
    if idx < len(xs) - 1:
        ax.add_patch(FancyArrowPatch((x + 1.16, 2.01), (xs[idx + 1], 2.01), arrowstyle="-|>", mutation_scale=10, color="#999999", linewidth=0.8))

ax.text(0.35, 5.72, "Configuration evidence", fontsize=10, fontweight="bold", color=COLORS["dark"])
ax.text(0.35, 3.06, "Sequential claim qualification", fontsize=10, fontweight="bold", color=COLORS["dark"])
ax.text(11.3, 0.72, "Paris formal/protected: procedural vault; analytical access = 0", ha="right", va="center", fontsize=8.5, color=COLORS["purple"], bbox={"boxstyle": "round,pad=0.25", "facecolor": "#FAF0F6", "edgecolor": COLORS["purple"]})

save_figure(fig, "fig01_evidence_ladder")
