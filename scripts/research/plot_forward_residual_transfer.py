"""Plot public metrics only; no private predictions or input series are read."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
    s=ROOT/'artifacts/summaries/forward_residual_transfer_v1'
    m=json.loads((s/'model_summary.json').read_text())
    rows=[r for r in m['per_run'] if r['support']=='FOUR_H' and r['postprocess']=='raw' and r['target_scope']=='terminal_H']
    with (s/'base_transfer_scores.csv').open() as stream:diag=list(csv.DictReader(stream))
    kinds=['P_IS_X','P_FWD_X','P_IS_BC','P_FWD_BC','C_IS_BC','C_FWD_BC']
    colors=['#61717D','#037F91','#CE8A13','#B52640','#6276BD','#74578C']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for kind,color in zip(kinds,colors):
        rr=[r for r in rows if r['model']==kind]
        axes[0].scatter([r['macro_rmse'] for r in rr],[r['macro_mae'] for r in rr],color=color,s=35,alpha=.8)
        axes[0].scatter(sum(r['macro_rmse'] for r in rr)/3,sum(r['macro_mae'] for r in rr)/3,color=color,marker='x',s=95,linewidth=2,label=kind)
        if '_FWD_' in kind:
            dd=[r for r in diag if r['model']==kind]
            axes[1].plot([0,1],[sum(r['macro_rmse'] for r in rr)/3,sum(float(r['rmse']) for r in dd)/len(dd)],'-o',color=color,label=kind)
    bf=next(r for r in rows if r['model']=='BASE_OD');dd=[r for r in diag if r['model']=='B3']
    axes[0].scatter(bf['macro_rmse'],bf['macro_mae'],color='black',marker='*',s=140,label='BF base')
    axes[1].plot([0,1],[bf['macro_rmse'],sum(float(r['rmse']) for r in dd)/4],'--*',color='black',label='Base only',markersize=9)
    axes[0].set(xlabel='Four-H endpoint macro RMSE',ylabel='Four-H endpoint macro MAE',title='BF main track: dots = runs, crosses = metric means')
    axes[1].set(xticks=[0,1],xticklabels=['BF: main deployment','B3: diagnostic only'],ylabel='Mean four-H endpoint RMSE',title='Same selected checkpoints; no base selection')
    selected=json.loads((s/'frozen_selection.json').read_text())['selected']
    zero=sum('_FWD_' in r['model'] and r['selected']['epoch']==0 for r in selected)
    axes[1].text(.03,.97,f'{zero}/9 FWD runs selected epoch 0;\nzero correction = base alias',transform=axes[1].transAxes,va='top',fontsize=9,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    for ax in axes:ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False);ax.legend(fontsize=8,loc='best')
    axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('Forward residual transfer: three registered seeds, previously exposed development window',fontsize=11)
    destination=ROOT/'docs/research/sota/figures/forward_residual_transfer.png';destination.parent.mkdir(exist_ok=True);fig.savefig(destination,dpi=220);plt.close(fig);print(destination)

if __name__=='__main__':main()
