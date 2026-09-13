"""Exact finite-sum synthetic mechanism check; never loads UrbanEV data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
import time

import numpy as np


def coefficients(p, steps):
    """Coefficient matrices of (P diag(1,z,1)) ** steps."""
    f = np.zeros((steps + 1, 3, 3))
    f[0] = np.eye(3)
    for _ in range(steps):
        move = f @ p
        new = move.copy()
        new[:, :, 1] = 0
        new[1:, :, 1] = move[:-1, :, 1]
        f = new
    return f


def forward_filter(p, pi, history, steps):
    """Independent normalized HMM count filter, no coefficient matrices."""
    obs = np.array([0, 1, 1])
    v = pi * (obs == history[0])
    likelihood = float(v.sum())
    if likelihood == 0:
        return 0.0, v
    v = v / likelihood
    for interval in range(1, 4):
        if interval < 3:
            joint = np.zeros((steps + 1, 3))
            joint[0] = v
            for _ in range(steps):
                arrivals = joint @ p
                joint = np.zeros_like(joint)
                joint[:, 0] = arrivals[:, 0]
                joint[:, 2] = arrivals[:, 2]
                joint[1:, 1] = arrivals[:-1, 1]
            v = joint[history[3 + interval]]
        else:
            for _ in range(steps):
                v = v @ p
        v = v * (obs == history[interval])
        prob = float(v.sum())
        likelihood *= prob
        if prob == 0:
            return 0.0, v
        v = v / prob
    return likelihood, v


def calculate_system(p, pi, config):
    steps = config['microsteps_per_hour']
    f = coefficients(p, steps)
    b = np.linalg.matrix_power(p, steps)
    busy = np.array([0., 1., 1.])
    masks = [np.diag(1-busy), np.diag(busy)]
    future = np.stack([np.linalg.matrix_power(b, j) @ busy for j in range(1, 13)], axis=1)
    histories, weights, predictions = [], [], []
    max_filter, max_likelihood = 0., 0.
    for os in itertools.product(range(2), repeat=4):
        for ks in itertools.product(range(steps + 1), repeat=2):
            history = os + ks
            w = pi @ masks[os[0]] @ f[ks[0]] @ masks[os[1]] @ f[ks[1]] @ masks[os[2]] @ b @ masks[os[3]]
            mass = float(w.sum())
            if mass == 0:
                continue
            pred = w @ future / mass
            likelihood, filtered = forward_filter(p, pi, history, steps)
            max_likelihood = max(max_likelihood, abs(mass-likelihood))
            max_filter = max(max_filter, float(np.max(np.abs(pred-filtered @ future))))
            histories.append(history)
            weights.append(mass)
            predictions.append(pred)
    weights = np.array(weights)
    predictions = np.array(predictions)
    risk_rows, metrics, gamma, identity_errors = [], {}, {}, []
    full_mse = np.sum(weights[:, None]*predictions*(1-predictions), axis=0)
    for name, columns in config['information'].items():
        sums = {}
        keys = [tuple(h[i] for i in columns) for h in histories]
        for key, weight, pred in zip(keys, weights, predictions):
            if key not in sums:
                sums[key] = [0., np.zeros(12)]
            sums[key][0] += weight
            sums[key][1] += weight*pred
        grouped = np.stack([sums[key][1]/sums[key][0] for key in keys])
        # Score against full conditional truth; do not assume the identity being tested.
        mse = np.sum(weights[:, None]*(predictions*(1-grouped)**2+(1-predictions)*grouped**2), axis=0)
        mae = np.sum(weights[:, None]*(predictions*(1-grouped)+(1-predictions)*grouped), axis=0)
        med = (grouped > .5).astype(float)
        median_mae = np.sum(weights[:, None]*(predictions*(1-med)+(1-predictions)*med), axis=0)
        g = np.sum(weights[:, None]*(predictions-grouped)**2, axis=0)
        identity_errors.extend([float(np.max(np.abs(mse-full_mse-g))), float(np.max(np.abs(mae-2*mse)))])
        gamma[name] = g.tolist()
        per_h = {str(h): {'rmse': float(np.sqrt(np.mean(mse[:h]))), 'mae_same_mean': float(np.mean(mae[:h])), 'mae_median_separate': float(np.mean(median_mae[:h]))} for h in config['horizons']}
        metrics[name] = {'per_horizon': per_h, 'macro_rmse': float(np.mean([x['rmse'] for x in per_h.values()])), 'macro_mae_same_mean': float(np.mean([x['mae_same_mean'] for x in per_h.values()]))}
        for j in range(12):
            risk_rows.append([name, j+1, mse[j], np.sqrt(mse[j]), mae[j], median_mae[j], g[j]])
    checks = {'positive_support_histories':len(histories), 'enumerated_histories':2**4*(steps+1)**2, 'mass_error':abs(float(weights.sum())-1), 'row_sum_error':float(np.max(abs(p.sum(axis=1)-1))), 'stationary_error':float(np.max(abs(pi @ p-pi))), 'coefficient_sum_error':float(np.max(abs(f.sum(axis=0)-b))), 'hmm_probability_error':max_filter, 'hmm_history_mass_error':max_likelihood, 'risk_identity_error':max(identity_errors), 'minimum_joint_probability':float(weights.min())}
    gains = {name:100*(1-metrics['I4']['macro_rmse']/metrics[name]['macro_rmse']) for name in metrics}
    return risk_rows, metrics, gamma, checks, gains


def counterexamples():
    price=np.array([1.,2.]); u1=-price; u2=-2*price+np.array([1.,2.])
    def logit(u):
        e=np.exp(np.r_[0.,u]); return e/e.sum()
    return {'snapshot_minus_integral': {'active_interval':'[0,1)', 'departure_time':1, 'D1':1, 'O1':0, 'O1_minus_D1':-1, 'conclusion':'Snapshot minus accumulated activity is not an instantaneous non-active count; this is a continuous-time illustration distinct from the discrete protocol endpoint convention.'}, 'service_aliasing': {'arrival_rate':4, 'service_A_hours':[.5], 'service_B_hours':[.25,.75], 'service_B_probabilities':[.5,.5], 'both_mean_hours':.5, 'integer_sample_distribution':'iid Poisson(2)', 'reason':'Stationary independent Poisson arrivals with iid marks; service <1 hour makes the marked-arrival sets serving different integer snapshots disjoint. Counts have the same mean lambda E[S].', 'scope':'Only integer occupancy snapshots, not an equivalence claim after duration observations.'}, 'price_equivalence': {'price':price.tolist(), 'utility_1':u1.tolist(), 'utility_2':u2.tolist(), 'logit_with_outside_1':logit(u1).tolist(), 'logit_with_outside_2':logit(u2).tolist(), 'max_difference':float(np.max(abs(logit(u1)-logit(u2)))), 'scope':'Unrestricted latent utility shocks; no causal elasticity identification.'}, 'rejected_arrivals': {'busy_path_A':[1,1,1], 'busy_path_B':[1,1,1], 'rejected_A':2, 'rejected_B':20, 'admitted_both':0, 'assumption':'All resources busy throughout interval; arrivals leave without influencing incumbent service; no rejection log observed.'}}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--claim',type=Path,required=True)
    args=parser.parse_args()
    config_bytes=args.config.read_bytes(); config=json.loads(config_bytes)
    if args.output.exists():
        raise FileExistsError('Output must be new; do not overwrite a completed or failed experiment')
    args.claim.parent.mkdir(parents=True,exist_ok=True)
    with args.claim.open('x',encoding='utf-8') as file:
        json.dump({'protocol_id':config['protocol_id'],'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'status':'CONSUMED_ON_START'},file)
    args.output.mkdir(parents=True)
    start=time.perf_counter()
    write_json(args.output/'mechanism_preregistration.json',config)
    summary, checks, all_rows={}, {}, []
    for system, rows in config['systems'].items():
        base=np.array([[float(Fraction(v)) for v in row] for row in rows])
        pi=np.array([float(Fraction(v)) for v in config['stationary'][system]])
        for scale in config['scales']:
            name=system+'_r'+scale.replace('/','_')
            p=np.eye(3)+float(Fraction(scale))*(base-np.eye(3))
            rows, metrics, gamma, check, gains=calculate_system(p,pi,config)
            if system=='L': check['lumpable_memory_max_gamma']=max(max(x) for x in gamma.values())
            all_rows.extend([[name]+row for row in rows])
            summary[name]={'metrics':metrics,'gamma_by_future_hour':gamma,'I4_relative_macro_rmse_improvement_percent':gains}
            checks[name]=check
    issues=[]
    for name,c in checks.items():
        for key in ['mass_error','row_sum_error','stationary_error','coefficient_sum_error']:
            if c[key]>config['mass_tolerance']: issues.append(name+':'+key)
        for key in ['hmm_probability_error','hmm_history_mass_error','risk_identity_error','lumpable_memory_max_gamma']:
            if c.get(key,0)>config['identity_tolerance']: issues.append(name+':'+key)
    with (args.output/'observed_history_risks.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file,lineterminator='\n');writer.writerow(['system','information','future_hour','mse','rmse','mae_same_mean','mae_median_separate','gamma_to_I4']);writer.writerows(all_rows)
    write_json(args.output/'memory_value_by_horizon.json',summary)
    write_json(args.output/'same_information_baseline_checks.json',checks)
    write_json(args.output/'identifiability_counterexamples.json',counterexamples())
    receipt={'protocol_id':config['protocol_id'],'status':'SYNTHETIC_IMPLEMENTATION_BLOCKED' if issues else 'SYNTHETIC_MECHANISM_CHECK_COMPLETE','issues':issues,'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__,'systems':4,'enumerated_histories':10816,'risk_rows':len(all_rows),'real_data_reads':0,'model_fits':0,'foundation_inference':0,'alpha_searches':0,'parameter_searches':0,'executions':1,'real_method_gate':'NOT_RUN','real_data_stage_authorized':False,'automation_status':'PAUSED','elapsed_seconds':time.perf_counter()-start,'peak_memory':'NOT_MEASURED','numerical_method':'float64 finite sums and matrix products, not symbolic exact arithmetic or Monte Carlo'}
    write_json(args.output/'execution_receipt.json',receipt)
    print(json.dumps({'receipt':receipt,'gains':{s:r['I4_relative_macro_rmse_improvement_percent'] for s,r in summary.items()}},indent=2))
    if issues: raise SystemExit(1)


if __name__=='__main__': main()
