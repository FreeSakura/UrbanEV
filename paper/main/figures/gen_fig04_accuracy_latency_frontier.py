import matplotlib.pyplot as plt
import pandas as pd

from paper_plot_style import COLORS, DATA_DIR, save_figure


data = pd.read_csv(DATA_DIR / "latency_batch1.csv")
data = data[data["method"] != "horizon_fixed"].copy()
fig, ax = plt.subplots(figsize=(6.2, 4.2))

for _, row in data.iterrows():
    emphasis = row["method"] in {"timexer", "global_fixed"}
    color = COLORS["green"] if row["method"] == "global_fixed" else COLORS["blue"] if row["method"] == "timexer" else COLORS["gray"]
    ax.scatter(row["end_to_end_median_ms"], row["RMSE"], s=85 if emphasis else 42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
    offsets = {
        "nlinear": (5, 5),
        "dlinear": (5, 9),
        "gbdt_forecaster": (5, -13),
        "caper": (5, 5),
        "timexer": (5, 6),
        "global_fixed": (6, 6),
    }
    ax.annotate(row["method_label"], (row["end_to_end_median_ms"], row["RMSE"]), xytext=offsets[row["method"]], textcoords="offset points", fontsize=8, color=color)

tx = data.set_index("method").loc["timexer"]
gf = data.set_index("method").loc["global_fixed"]
ax.annotate("3.178% lower RMSE\n48.86% higher P50 latency", xy=(gf["end_to_end_median_ms"], gf["RMSE"]), xytext=(12.0, 0.07115), arrowprops={"arrowstyle": "->", "color": COLORS["orange"], "lw": 1.2}, fontsize=8, color=COLORS["orange"], ha="center")
ax.set_xlabel("Batch-1 end-to-end median latency (ms)")
ax.set_ylabel("Macro RMSE (lower is better)")
ax.grid(color="#E6E6E6", linewidth=0.7)
ax.set_xlim(left=-0.2)
ax.set_ylim(0.0708, 0.0785)
fig.tight_layout()
save_figure(fig, "fig04_accuracy_latency_frontier")
