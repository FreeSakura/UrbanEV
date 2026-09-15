"""Public aggregate plot: selected DEV performance versus fixed40 mechanism."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=ROOT/'artifacts/summaries/fixed_reference_correction_v1'
    m=json.loads((source/'model_summary.json').read_text())
    rows=[r for r in m['per_run'] if r['support']=='FOUR_H' and r['postprocess']=='raw' and r['target_scope']=='terminal_H']
    with (source/'mechanism_fixed40_scores.csv').open() as stream:mech=list(csv.DictReader(stream))
    kinds=['FIXED_PRODUCT','FIXED_SEPARABLE','FIXED_CONCAT'];colors=['#B52640','#037F91','#74578C']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');coordinates=[]
    for kind,color in zip(kinds,colors):
        rr=[r for r in rows if r['model']==kind]
        axes[0].scatter([r['macro_rmse'] for r in rr],[r['macro_mae'] for r in rr],color=color,s=35,alpha=.8)
        axes[0].scatter(sum(r['macro_rmse'] for r in rr)/3,sum(r['macro_mae'] for r in rr)/3,color=color,marker='x',s=95,linewidth=2,label=kind)
        for seed in ('20260915','20260916','20260917'):
            mm=[r for r in mech if r['model']==kind and r['seed']==seed]
            x=sum(2*float(r['alignment_A']) for r in mm)/4;y=sum(float(r['correction_energy_B']) for r in mm)/4;coordinates.append((x,y));axes[1].scatter(x,y,color=color,s=45)
    for kind,color,marker in [('BASE_FIXED','black','*'),('LINEAR_REPAIR','#CE8A13','D'),('RIDGE_FULL','#61717D','s')]:
        r=next(r for r in rows if r['model']==kind);axes[0].scatter(r['macro_rmse'],r['macro_mae'],color=color,marker=marker,s=100,label=kind)
        mm=[r for r in mech if r['model']==kind];x=sum(2*float(r['alignment_A']) for r in mm)/4;y=sum(float(r['correction_energy_B']) for r in mm)/4
        coordinates.append((x,y));axes[1].scatter(x,y,color=color,marker=marker,s=100)
    lo=min(0.,min(x for x,y in coordinates));hi=max(max(x,y) for x,y in coordinates);pad=max(hi-lo,1e-5)*.06
    axes[1].plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#777777',lw=1,label='G = 0; below line = MSE gain')
    axes[0].set(xlabel='Four-H endpoint macro RMSE',ylabel='Four-H endpoint macro MAE',title='DEV: selected checkpoints; dots = runs, x = metric mean')
    axes[1].set(xlabel='Mean over four H: 2 x alignment A',ylabel='Mean over four H: correction energy B',title='MECH: fixed epoch40, no checkpoint selection')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False);ax.legend(fontsize=7,loc='best')
    fig.suptitle('Fixed-reference correction: different windows and purposes; no pooled performance',fontsize=11)
    path=ROOT/'docs/reports/figures/fixed_reference_correction.png';path.parent.mkdir(parents=True, exist_ok=True);fig.savefig(path,dpi=220);plt.close(fig);print(path)

if __name__=='__main__':main()
