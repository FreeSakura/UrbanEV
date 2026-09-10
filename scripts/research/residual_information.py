"""Rolling residual-information screen and conditionally admitted validation."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from urbanev_forecast.data import load_rate_prefix, scores
from urbanev_forecast.foundation import FoundationBackend, file_hash
from replicate_signals import derangement

CFG = ROOT/'configs/research/RESIDUAL_INFORMATION_V1.json'
HS = (3, 12)
SYSTEMS = ('native', 'bias', 'occupancy', 'duplicate', 'raw_duration', 'orthogonal_duration', 'permuted_orthogonal')
ALPHAS = (0, .25, .5, 1)
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)


def duration_prefix(path, info, csv, hours):
    with path.open('rb') as stream:
        raw = b''.join(stream.readline() for _ in range(hours+1))
    frame = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    cols = pd.read_csv(csv, nrows=0).columns[1:].astype(str).tolist()
    cap = pd.read_csv(info, dtype={'TAZID':str}).groupby('TAZID')['charge_count'].sum().reindex(cols).to_numpy(float)
    if frame.shape != (hours,275) or list(frame.columns.astype(str)) != cols or not frame.index.equals(pd.date_range('2022-09-01', periods=hours, freq='h')):
        raise ValueError('Duration alignment mismatch')
    if not np.isfinite(cap).all() or np.any(cap <= 0):
        raise ValueError('Invalid capacity')
    d = frame.to_numpy(float)/cap
    if not np.isfinite(d).all() or np.any((d<0)|(d>1)):
        raise ValueError('Invalid duration values')
    return d, {'prefix_sha256':hashlib.sha256(raw).hexdigest(), 'rows':hours, 'info_sha256':file_hash(info), 'endpoint_and_release_verified':False}


def features(o, d, origins, base):
    xx, dd = [], []
    for i,s in enumerate(origins):
        x = [o[s-l] for l in LAGS]
        x += [o[s-k:s].mean(0) for k in (24,168)]
        x += [o[s-k:s].std(0) for k in (24,168)]
        x += [base[i,j] for j in range(base.shape[1])]
        x += [np.full(275, v) for v in (np.sin(s*2*np.pi/24),np.cos(s*2*np.pi/24),np.sin(s*2*np.pi/168),np.cos(s*2*np.pi/168))]
        z = [d[s-l-1] for l in LAGS]
        z += [d[s-k-1:s-1].mean(0) for k in (24,168)]
        xx.append(np.stack(x, axis=1)); dd.append(np.stack(z, axis=1))
    return np.concatenate(xx).astype(float), np.stack(dd).astype(float)


def standardize(x, vx, zero_constant=False):
    mean, std = x.mean(0), x.std(0)
    small = std < 1e-6; std[small] = 1
    a, b = (x-mean)/std, (vx-mean)/std
    if zero_constant:
        a[:,small] = 0; b[:,small] = 0
    return a,b


def ridge(x,y):
    reg = np.eye(x.shape[1])*.01; reg[-1,-1] = 0
    return np.linalg.solve(x.T@x/len(x)+reg, x.T@y/len(x))


def residual_features(x, vx, d, vd):
    ds,vds = standardize(d,vd)
    a = np.linalg.lstsq(x,ds,rcond=1e-10)[0]
    z, vz = ds-x@a, vds-vx@a
    diagnostic = float(np.max(np.abs(x.T@z))/len(x))
    z,vz = standardize(z,vz,True)
    return z,vz,diagnostic


def fit_corrections(x, vx, d, vd, error):
    x,vx = standardize(x,vx)
    x,vx = np.column_stack([x,np.ones(len(x))]), np.column_stack([vx,np.ones(len(vx))])
    occ = vx@ridge(x,error)
    result = {'native':np.zeros_like(occ), 'bias':np.broadcast_to(error.mean(0),occ.shape).copy(), 'occupancy':occ}
    diagnostics = {}
    # Intercept remains last and unpenalized in all joint designs.
    for system, td, tv in [('duplicate',x[:,:11],vx[:,:11]),('raw_duration',d.reshape(-1,11),vd.reshape(-1,11))]:
        td,tv = standardize(td,tv)
        design = np.column_stack([x[:,:-1],td,np.ones(len(x))])
        valid = np.column_stack([vx[:,:-1],tv,np.ones(len(vx))])
        result[system] = valid@ridge(design,error)
    for system, perm in [('orthogonal_duration',np.arange(275)),('permuted_orthogonal',derangement(275))]:
        z,vz,diagnostic = residual_features(x,vx,d[:,perm].reshape(-1,11),vd[:,perm].reshape(-1,11))
        # No added intercept: occupancy head already supplies it.
        gamma = np.linalg.solve(z.T@z/len(z)+.01*np.eye(11), z.T@(error-x@ridge(x,error))/len(z))
        result[system] = occ+vz@gamma
        diagnostics[system] = {'training_projection_max_cross_moment':diagnostic, 'gamma_norm':float(np.linalg.norm(gamma)), 'active_residual_features':int(np.sum(z.std(0)>1e-6)), 'projection_coefficients':int(x.shape[1]*11), 'extra_target_coefficients':int(11*error.shape[1])}
    return result,diagnostics


def metric(pred,y):
    return scores(np.clip(pred,0,1),y)


def macro(rows):
    return {m:float(np.mean([r[m] for r in rows])) for m in ('rmse','mae')}


def compare(cells, chosen, candidate, reference):
    cc,rr = [],[]; windows = {}; origin_loss = []
    for cell in cells:
        y = cell['truth'].astype(float)
        cp = np.clip(cell['predictions'][candidate][chosen[candidate]],0,1).astype(float)
        rp = np.clip(cell['predictions'][reference][chosen[reference]],0,1).astype(float)
        a,b = scores(cp,y),scores(rp,y); cc.append(a);rr.append(b)
        windows.setdefault(cell['cut'],[]).append((a['rmse'],b['rmse']))
        origin_loss.append(((cp-y)**2,(rp-y)**2))
    c,r = macro(cc),macro(rr)
    gain = 100*(1-c['rmse']/r['rmse'])
    degrade = [100*(a['rmse']/b['rmse']-1) for a,b in zip(cc,rr)]
    positive = sum(np.mean([v[0] for v in pairs])<np.mean([v[1] for v in pairs]) for pairs in windows.values())
    # Paired within each cell; shared uniforms couple the two horizons per cut.
    rng=np.random.default_rng(20260910); draws=[]
    losses=[(a.mean((1,2)),b.mean((1,2))) for a,b in origin_loss]
    for _ in range(2000):
        vals=[]; u_by_cut={}
        for cell,(a,b) in zip(cells,losses):
            block=2 if len(a)<30 else 24
            u=u_by_cut.setdefault(cell['cut'],rng.random(int(np.ceil(len(a)/block))))
            starts=np.floor(u*len(a)).astype(int)
            ix=((starts[:,None]+np.arange(block))%len(a)).ravel()[:len(a)]
            vals.append((np.sqrt(a[ix].mean()),np.sqrt(b[ix].mean())))
        draws.append(100*(1-np.mean([v[0] for v in vals])/np.mean([v[1] for v in vals])))
    return {'candidate':candidate,'reference':reference,'rmse_gain_percent':gain,'mae_change':c['mae']-r['mae'],
            'cell_rmse_degradation_percent':degrade,'positive_windows':int(positive),'windows':len(windows),
            'rmse_gain_descriptive_interval':np.quantile(draws,[.025,.975]).tolist(),
            'point_pass':bool(gain>=1 and c['mae']<=r['mae'] and max(degrade)<=1)}


def main():
    p=argparse.ArgumentParser()
    for name in ('csv','duration','info','model-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['screen','confirm'],default='screen');p.add_argument('--screen',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('New output required')
    config=json.loads(CFG.read_text()); hours=1747 if a.stage=='screen' else 1966
    if (config['train_end'],config['validation_end'],tuple(config['horizons']),tuple(config['rolling_cuts']),tuple(config['alphas']),config['ridge'])!=(1747,1966,HS,(720,1056,1392),ALPHAS,.01):
        raise ValueError('Frozen config and implementation differ')
    prior=None
    if a.stage=='confirm':
        prior=json.loads((a.screen/'result.json').read_text())
        if not prior['information_gate'] or prior['config_sha256']!=file_hash(CFG):raise ValueError('Confirmation not admitted')
    o,source=load_rate_prefix(a.csv,hours);d,ds=duration_prefix(a.duration,a.info,a.csv,hours)
    if prior:
        _,prefix=load_rate_prefix(a.csv,1747)
        if prefix!=prior['source']:raise ValueError('Training prefix changed')
    a.output.mkdir(parents=True)
    backend=FoundationBackend('chronos2',a.model_dir,'cuda');backend.model.eval()
    if prior:
        if backend.metadata!=prior['backend']:raise ValueError('Backend changed')
    cells=[]; cache_manifest=[]
    with threadpool_limits(limits=2):
        for h in HS:
            if a.stage=='screen':
                origins=np.arange(192,1747-h+1,12)
                base=np.stack([backend.predict(o[s-168:s],h)[1] for s in origins])
                np.save(a.output/f'private_origins_h{h}.npy',origins);np.save(a.output/f'private_base_h{h}.npy',base)
            else:
                origins=np.load(a.screen/f'private_origins_h{h}.npy');base=np.load(a.screen/f'private_base_h{h}.npy')
                rec=next(r for r in prior['cache'] if r['horizon']==h)
                if file_hash(a.screen/f'private_base_h{h}.npy')!=rec['prediction_sha256'] or file_hash(a.screen/f'private_origins_h{h}.npy')!=rec['origins_sha256']:raise ValueError('Cache changed')
            if a.stage=='screen':
                cache_manifest.append({'horizon':h,'origins_sha256':file_hash(a.output/f'private_origins_h{h}.npy'),'prediction_sha256':file_hash(a.output/f'private_base_h{h}.npy'),'origin_count':len(origins)})
            cuts=(720,1056,1392) if a.stage=='screen' else (1747,)
            for cut in cuts:
                tr=origins+h<=cut; train_origins=origins[tr];tb=base[tr]
                if a.stage=='screen':
                    val=(origins>=cut)&(origins+h<=cut+168);vo=origins[val];vb=base[val]
                else:
                    vo=np.arange(1747,1966-h+1);vb=np.stack([backend.predict(o[s-168:s],h)[1] for s in vo])
                x,td=features(o,d,train_origins,tb);vx,vd=features(o,d,vo,vb)
                target=o[train_origins[:,None]+np.arange(h)].transpose(0,2,1).reshape(-1,h)
                error=target-tb.transpose(0,2,1).reshape(-1,h)
                corrections,diagnostics=fit_corrections(x,vx,td,vd,error)
                truth=o[vo[:,None]+np.arange(h)];predictions={};records=[]
                for system,delta in corrections.items():
                    delta=delta.reshape(len(vo),275,h).transpose(0,2,1)
                    predictions[system]={}
                    alphas=ALPHAS if prior is None else [prior['selected_alphas'][system]]
                    for alpha in alphas:
                        pred=vb+alpha*delta;predictions[system][alpha]=pred
                        name=f'private_{cut}_h{h}_{system}_a{alpha}.npy';np.save(a.output/name,pred)
                        records.append({'system':system,'alpha':alpha,'raw':scores(pred,truth),'clipped':metric(pred,truth),'prediction_file':name,'prediction_sha256':file_hash(a.output/name)})
                cell={'cut':cut,'horizon':h,'truth':truth,'predictions':predictions,'records':records,'diagnostics':diagnostics,
                      'train_origins':len(train_origins),'train_last_target':int(train_origins[-1]+h-1),'validation_origins':len(vo),
                      'validation_first_origin':int(vo[0]),'validation_last_origin':int(vo[-1])}
                cells.append(cell);print(json.dumps({'cut':cut,'horizon':h,'fit_complete':True,'diagnostics':diagnostics}),flush=True)
    def system_metrics(system,alpha):return [metric(c['predictions'][system][alpha],c['truth']) for c in cells]
    native=system_metrics('native',0);nm=macro(native);chosen={};selection=[]
    for system in SYSTEMS:
        if prior:
            chosen[system]=prior['selected_alphas'][system];continue
        admissible=[]
        for alpha in ALPHAS:
            rows=system_metrics(system,alpha);m=macro(rows)
            valid=m['mae']<=nm['mae'] and max(100*(r['rmse']/n['rmse']-1) for r,n in zip(rows,native))<=1
            selection.append({'system':system,'alpha':alpha,**m,'admissible':valid})
            if valid:admissible.append((m['rmse'],alpha))
        chosen[system]=min(admissible)[1]
    macros={s:macro(system_metrics(s,chosen[s])) for s in SYSTEMS}
    ref=min(('native','bias','occupancy','duplicate'),key=lambda s:macros[s]['rmse'])
    info=compare(cells,chosen,'orthogonal_duration',ref)
    neg=compare(cells,chosen,'orthogonal_duration','permuted_orthogonal')
    mechanism=compare(cells,chosen,'orthogonal_duration','raw_duration')
    gate=info['point_pass'] and chosen['orthogonal_duration']>0 and neg['rmse_gain_percent']>0 and neg['mae_change']<=0
    if a.stage=='screen':gate=gate and info['positive_windows']>=2
    report={'protocol':config['id'],'stage':a.stage,'source':source,'duration_source':ds,'backend':backend.metadata,
            'cache':cache_manifest,'cells':[{k:v for k,v in c.items() if k not in ('truth','predictions')} for c in cells],
            'selection':selection,'selected_alphas':chosen,'macro':macros,'comparisons':[info,neg,mechanism],
            'information_gate':bool(gate),'mechanism_gate':mechanism['point_pass'],
            'config_sha256':file_hash(CFG),'script_sha256':file_hash(Path(__file__)),'fold_three_test_loaded':False,
            'earlier_fold_test_enters_training':True,'independent_confirmation':False,'sota_claim':False}
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'alphas':chosen,'macro':macros,'information_gate':bool(gate),'mechanism_gate':mechanism['point_pass'],'comparisons':report['comparisons']}),flush=True)


if __name__=='__main__':main()
