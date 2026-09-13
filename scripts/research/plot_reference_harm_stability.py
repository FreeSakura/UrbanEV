"""Render only published aggregate metrics; no raw series or prediction access."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/summaries/reference_harm_stability_v1'

def main():
    model=json.loads((SOURCE/'model_summary.json').read_text())
    stability=json.loads((SOURCE/'stability_summary.json').read_text())
    rows=[r for r in model['per_run'] if r['support']=='FOUR_H' and r['postprocess']=='raw' and r['target_scope']=='terminal_H']
    kinds=['PRODUCT_MSE','PRODUCT_POSREG','PRODUCT_MIX','PRODUCT_L1','CONCAT_MSE','CONCAT_POSREG']
    colors=['#61717D','#B52640','#037F91','#CE8A13','#6276BD','#74578C']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for kind,color in zip(kinds,colors):
        rr=[r for r in rows if r['model']==kind]
        axes[0].scatter([r['macro_rmse'] for r in rr],[r['macro_mae'] for r in rr],color=color,s=35,alpha=.75)
        axes[0].scatter(sum(r['macro_rmse'] for r in rr)/3,sum(r['macro_mae'] for r in rr)/3,color=color,marker='x',s=95,linewidth=2,label=kind)
        ss=[r for r in stability['relative_disagreement'] if r['model']==kind and r['postprocess']=='raw']
        axes[1].plot([r['horizon'] for r in ss],[float('nan') if r['eta'] is None else r['eta'] for r in ss],'-o',color=color,lw=1.5,markersize=4)
    base=next(r for r in rows if r['model']=='BASE_OD')
    axes[0].scatter(base['macro_rmse'],base['macro_mae'],color='black',marker='*',s=140,label='BASE')
    axes[0].set(xlabel='Four-H endpoint macro RMSE',ylabel='Four-H endpoint macro MAE',title='Raw scores: dots = runs, crosses = metric means')
    axes[1].set(xlabel='Forecast horizon (hours)',ylabel='Relative seed disagreement V / T',title='Scale-relative disagreement (not accuracy)',xticks=[3,6,9,12])
    for ax in axes:ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
    axes[0].legend(fontsize=7,loc='best')
    fig.suptitle('UrbanEV development comparison: three new seeds, one previously exposed window',fontsize=11)
    destination=ROOT/'docs/research/sota/figures/reference_harm_stability.png';destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination,dpi=220);plt.close(fig)
    print(destination)

if __name__=='__main__':main()
