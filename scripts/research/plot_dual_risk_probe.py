"""Render aggregate derivative evidence; requires optional matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
source = ROOT/'artifacts/summaries/dual_risk_probe_v1/directional_risk_report.json'
data = json.loads(source.read_text(encoding='utf-8'))
out = ROOT/'docs/research/sota/figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.hashsalt': 'dual-risk-probe-v1'})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, cut in zip(axes, (1056, 1392)):
    rows = [r for r in data['windows'] if r['evaluation_cut'] == cut]
    values = [r['mae_derivative']*1e5 for r in rows]
    colors = ['#16867A' if r['representation'] == 'duration' else '#8D9CA8' for r in rows]
    ax.bar(range(4), values, color=colors, width=.62)
    ax.axhline(0, color='#354656', linewidth=1)
    ax.set_xticks(range(4), ['Base', '+Duration', '+Duplicate', '+Permuted'])
    ax.set_title(f'Forward evaluation [{cut}, {cut+168})', fontsize=12)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    for index, value in enumerate(values):
        ax.annotate(f'{value:+.2f}', (index, value), xytext=(0, 5 if value >= 0 else -5),
                    textcoords='offset points', ha='center', va='bottom' if value >= 0 else 'top', fontsize=10)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
axes[0].set_ylabel('MAE directional derivative (x 10^-5)\nNegative = local improvement')
axes[0].set_ylim(-5, 15)
fig.suptitle('Duration representation preserves local MAE descent in both windows', fontsize=15, x=.52)
fig.text(.5, .025, 'All four representations have negative RMSE derivatives in both windows.\n'
         'Post-hoc development probe only: no finite-step gain, blind-test result or SOTA claim.',
         ha='center', fontsize=10, color='#536575')
fig.tight_layout(rect=(0, .1, 1, .92))
fig.savefig(out/'dual_risk_probe_20260913.svg', metadata={'Date': None})
svg = out/'dual_risk_probe_20260913.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
               encoding='utf-8', newline='\n')
fig.savefig(out/'dual_risk_probe_20260913.png', dpi=180)
plt.close(fig)
