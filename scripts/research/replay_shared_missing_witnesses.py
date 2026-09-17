"""Verify exported missing-bit assignments using a separate cumulative-sum labeler."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from urbanev_audit.event_witnesses import exact_brier_difference


def labels_by_sums(bits,k,ell):
    sums=np.r_[0,np.cumsum(bits,dtype=np.int64)]
    hits=(sums[ell:]-sums[:-ell])==ell
    count=np.r_[0,np.cumsum(hits,dtype=np.int64)];m=k-ell+1
    return (count[m:]-count[:-m]>0).astype(np.int8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','ap1','ap2','results'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/research/run_shared_missing_p0.py')
    pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot);receipts=[]
    for phase in ('ap1','ap2'):
        cfg=json.loads((ROOT/f'configs/research/SHARED_MISSING_EVENTS_{phase.upper()}_202609{16 if phase=="ap1" else 17}.json').read_text())
        data,_,_=pilot.read_data(a.data,cfg);cache={};private=getattr(a,phase)
        frame=pd.read_csv(a.results/f'{phase}_explanations.csv').set_index('comparison_id')
        count=0
        with (a.results/f'{phase}_completion_witnesses.jsonl').open() as f:
            for line in f:
                w=json.loads(line);row=frame.loc[w['comparison_id']];s,n,k,ell=map(int,(row.origin_start,row.N,row.K,row.L))
                dc=cfg['datasets'][row.dataset];key=(row.dataset,row.series,k)
                if key not in cache:
                    with np.load(private/f'{row.dataset}_{row.series}_K{k}_predictions.npz') as z:cache[key]={m:z[m] for m in z.files}
                offset=s-dc['calibration_end'];pa=cache[key][row.model_A][offset:offset+n];pb=cache[key][row.model_B][offset:offset+n]
                o=pilot.observed_bits(data[row.dataset,row.series][0][s:s+n+k-1],dc['threshold']);bits=o.copy()
                assert hashlib.sha256(o.tobytes()).hexdigest()==w['observed_bits_sha256']
                assert hashlib.sha256(pa.tobytes()+pb.tobytes()).hexdigest()==w['predictions_sha256']
                missing=np.flatnonzero(o==-1);assert missing.tolist()==w['missing_offsets']
                bits[missing]=np.array(list(w['assigned_bits']),np.int8)
                assert np.isin(bits,[0,1]).all()
                labels=labels_by_sums(bits,k,ell)
                assert hashlib.sha256(labels.tobytes()).hexdigest()==w['labels_sha256']
                exact=exact_brier_difference(pa,pb,labels)
                for field in ('sign','sum_numerator','sum_denominator_power2'):assert exact[field]==w[field]
                count+=1
        receipts.append({'phase':phase,'witnesses_replayed':count,'all_observations_preserved':True,
            'independent_cumulative_sum_labels_match':True,'exact_rational_scores_match':True})
    pilot.dump(a.results/'witness_replay.json',{'status':'PASS','records':receipts})
    print(json.dumps(receipts),flush=True)


if __name__=='__main__':main()
