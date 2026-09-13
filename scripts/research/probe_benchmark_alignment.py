"""Code-derived calendar/metric probes. No dataset files, training, or forecasts read."""
import argparse
import calendar
import json
from pathlib import Path

import numpy as np
from urbanev_forecast.benchmark_metrics import scoped_scores


def index_span(first, stop):
    count=max(0,stop-first)
    return {'first_terminal_index':first if count else None,'last_terminal_index':stop-1 if count else None,'count':count}


def source_contracts():
    rows=[]; hours=0
    for fold,(year,month) in enumerate([(2022,9),(2022,10),(2022,11),(2022,12),(2023,1),(2023,2)],1):
        hours+=24*calendar.monthrange(year,month)[1]
        train=int(hours*.8)
        classical_val=int(train+hours*.1)
        transformer_val=hours-int(hours*.1)
        for h in [3,6,9,12]:
            rows.append({'fold':fold,'calendar_hours':hours,'horizon':h,'source_launch_history':12,'classical_validation_stop':classical_val,'transformer_validation_stop':transformer_val,'classical_validation_targets':index_span(train+12+h-1,classical_val-1),'transformer_validation_targets':index_span(train+h-1,transformer_val),'classical_test_targets':index_span(classical_val+12+h-1,hours-1),'transformer_test_targets':index_span(transformer_val+h-1,hours)})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a new output path; do not overwrite prior results')
    truth=np.zeros((1,3,1)); candidates={'A':np.array([.6,.6,0.]).reshape(1,3,1),'B':np.array([.3,.3,.3]).reshape(1,3,1)}
    results={scope:{name:scoped_scores(p,truth,target_scope=scope,postprocess='raw') for name,p in candidates.items()} for scope in ['terminal_H','path_1_to_H']}
    output={'status':'PUBLIC_SOURCE_CONTRACT_PROBE_COMPLETE','upstream_commit':'44f2aa0c8d89f192bce00bafb0def74a21b39c68','real_data_reads':0,'model_fits':0,'foundation_inference':0,'actual_paper_run_reproduced':False,'calendar_assumption':'Complete hourly calendar, not verified by opening dataset rows','rank_reversal_example':results,'calendar_origin_contracts':source_contracts(),'interpretation':'Different prediction targets and origin populations can change ranking; this probe scores only two fixed illustrative arrays, never project predictions. No SOTA claim follows.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':output['status'],'calendar_cells':len(output['calendar_origin_contracts']),'example':results},indent=2))


if __name__=='__main__': main()
