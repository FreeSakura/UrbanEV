"""Plot only matched-core aggregate scores, keeping information/support explicit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=ROOT/'artifacts/summaries/comprehensive_development_comparison_v1'
    data=json.loads((source/'model_summary.json').read_text())['per_instance']
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    colors={'LOCAL_O_168':'#61717D','LOCAL_OD_168':'#167E8F','GLOBAL_O_168':'#BD642B'}
    for ax,support in zip(axes,['FOUR_H','H3_H12']):
        rows=[r for r in data if r['support']==support and r['postprocess']=='raw'];names=list(dict.fromkeys(r['model'] for r in rows))
        for i,name in enumerate(names):
            rr=[r for r in rows if r['model']==name];values=np.array([r['macro_rmse'] for r in rr]);color=colors[rr[0]['information_track']]
            ax.barh(i,values.mean(),color=color,alpha=.75)
            ax.scatter(values,np.full(len(values),i),color='#222222',s=17,zorder=3)
        ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Raw endpoint macro RMSE (lower is better)',title=support+' — points are individual runs')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Matched development core; gray=local O, blue=local OD, orange=global O\nDifferent information tracks; H6/H9 native unavailable; no six-fold/SOTA claim',fontsize=10)
    p=ROOT/'docs/research/sota/figures/comprehensive_development_comparison.png';p.parent.mkdir(exist_ok=True);fig.savefig(p,dpi=220);plt.close(fig);print(p)

if __name__=='__main__':main()
