import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import COLORS, DATA_DIR, save_figure


data = pd.read_csv(DATA_DIR / "gate_results.csv")
urban = data[data["dataset"].str.startswith("UrbanEV")].copy()
paris = data[data["dataset"] == "Paris development"].copy()

fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), gridspec_kw={"width_ratios": [2.6, 1.0]}, sharex=True)

def draw(ax, frame):
    y = np.arange(len(frame))
    colors = [COLORS["green"] if d in {"SUPPORTED", "PASS SUBCHECK"} else COLORS["vermillion"] for d in frame["decision"]]
    bars = ax.barh(y, frame["gain_percent"], color=colors, alpha=0.86, height=0.58)
    for i, (_, row) in enumerate(frame.iterrows()):
        if pd.notna(row["ci_lower"]):
            lower = row["gain_percent"] - row["ci_lower"]
            upper = row["ci_upper"] - row["gain_percent"]
            ax.errorbar(row["gain_percent"], i, xerr=[[lower], [upper]], fmt="none", ecolor=COLORS["dark"], capsize=3, linewidth=1)
        hatch = "//" if row["role"] == "negative-control subcheck" else None
        if hatch:
            bars[i].set_hatch(hatch)
        value_label = f"{row['gain_percent']:+.3f}%"
        if row["gain_percent"] < 0:
            value_label += " (worse)"
            ax.text(row["gain_percent"] / 2, i, value_label, va="center", ha="center", fontsize=8, fontweight="bold", color="white")
        else:
            ax.text(row["gain_percent"] + 0.08, i, value_label, va="center", ha="left", fontsize=8)
        if row["decision"] != "SUPPORTED":
            ax.text(3.92, i, row["decision"], va="center", ha="right", fontsize=7, color=colors[i])
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(1.0, color=COLORS["orange"], linestyle="--", linewidth=1.2)
    ax.set_yticks(y, frame["stage"])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.7)

draw(axes[0], urban)
draw(axes[1], paris)
axes[0].set_xlim(-3.7, 4.05)
axes[0].set_xlabel("Relative RMSE gain against each stage-specific reference (%)")
axes[1].set_xlabel("Relative RMSE gain (%)")
axes[0].text(0.0, 1.10, "(a) UrbanEV internal stages; stage-specific references", transform=axes[0].transAxes, fontsize=8, fontweight="bold")
axes[1].text(0.0, 1.10, "(b) Paris development only", transform=axes[1].transAxes, fontsize=8, fontweight="bold")
for ax in axes:
    ax.text(1.04, 1.01, "1% gate", transform=ax.get_xaxis_transform(), color=COLORS["orange"], fontsize=7, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save_figure(fig, "fig03_stage_gain_gates")
