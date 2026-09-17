"""Post-hoc geometry of frozen AP1/AP2 observations; no model fitting/scoring."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.event_cover_geometry import EventCoverGeometry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/research/run_shared_missing_p0.py')
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
    c=json.loads((ROOT/'configs/research/SHARED_MISSING_EVENTS_AP2_20260917.json').read_text())
    data,_,_=pilot.read_data(a.data,c);rows=[];coverrows=[];joined=[];clock=time.perf_counter()
    for phase in ('ap1','ap2'):
        frame=pd.read_csv(ROOT/f'artifacts/summaries/shared_missing_events_{phase}/new_year_hierarchy.csv')
        frame=frame[frame['mask']=='natural']
        old=pd.read_csv(ROOT/f'artifacts/summaries/shared_missing_events_ap2/{phase}_panel_structure.csv')
        keys=['dataset','series','K','L','scope','origin_start','N'];old=old.set_index(keys)
        for ident,g in frame.groupby(keys,sort=True):
            dataset,station,k,ell,sco,start,n=ident;k,ell,start,n=map(int,(k,ell,start,n))
            observed=pilot.observed_bits(data[dataset,station][0][start:start+n+k-1],c['datasets'][dataset]['threshold'])
            geo=EventCoverGeometry(observed,k,ell);result=geo.diagnose();prior=old.loc[ident].structure_status
            assert result['long_window_regime']
            if prior=='STRUCTURALLY_EXACT':assert result['status']=='PAIRWISE_EXACT'
            if prior=='HIGHER_STRUCTURE':assert result['status']=='PAIRWISE_INSUFFICIENT'
            meta=dict(zip(keys,ident),phase=phase.upper())
            rows.append({**meta,'ambiguous_labels':result['ambiguous_labels'],'previous_status':prior,
                'geometry_status':result['status'],'essential_cover_count':result['essential_cover_count']})
            for label in result['label_records']:
                if label['private_run_start'] is None:coverrows.append({**meta,**label})
            for _,comparison in g.iterrows():
                joined.append({**meta,'model_A':comparison.model_A,'model_B':comparison.model_B,
                    'ambiguous_labels':result['ambiguous_labels'],'geometry_status':result['status'],
                    'pairwise_direction':comparison.pairwise_direction,'joint_direction':comparison.joint_direction,
                    'higher_order_new':bool(comparison.higher_order_new),
                    'objective_tighter':(comparison.pairwise_upper-comparison.pairwise_lower)-(comparison.joint_upper-comparison.joint_lower)>1e-10})
        print(json.dumps({'phase':phase,'panels_done':len(rows)}),flush=True)
    f=pd.DataFrame(rows);f.to_csv(a.output/'frozen_panel_geometry.csv',index=False)
    pd.DataFrame(coverrows).to_csv(a.output/'essential_covers.csv',index=False)
    j=pd.DataFrame(joined);j.to_csv(a.output/'geometry_and_cached_decisions.csv',index=False)
    counts=f[f.ambiguous_labels>0].groupby(['phase','geometry_status']).size().reset_index(name='panels')
    counts.to_csv(a.output/'ambiguous_geometry_summary.csv',index=False)
    cross=f.groupby(['phase','previous_status','geometry_status']).size().reset_index(name='panels');cross.to_csv(a.output/'previous_certificate_crosscheck.csv',index=False)
    cj=j[j.ambiguous_labels>0].groupby(['phase','geometry_status']).agg(comparisons=('model_A','size'),objective_tighter=('objective_tighter','sum'),higher_order_new=('higher_order_new','sum')).reset_index()
    cj.to_csv(a.output/'structure_objective_decision_summary.csv',index=False)
    pilot.dump(a.output/'frozen_geometry_execution.json',{'status':'COMPLETE','wall_seconds_excluding_data_load':time.perf_counter()-clock,
        'panels':len(rows),'new_training':0,'new_prediction':0,'new_risk_scoring':0,'old_conclusive_structural_checks_agree':True,
        'role':'Post-hoc structural interpretation of already exposed AP1/AP2 observations; not an independent confirmation study'})
    print(counts.to_string(index=False),flush=True);print(cj.to_string(index=False),flush=True)


if __name__=='__main__':main()
