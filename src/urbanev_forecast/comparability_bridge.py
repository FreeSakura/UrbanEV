"""Deterministic bridge inputs; no training or legacy candidate selection."""
import numpy as np


def zone_capacities(info, columns):
    values=info.groupby('TAZID',sort=False)['charge_count'].sum().reindex(columns).to_numpy(np.float64)
    if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid capacity mapping')
    return values


def build_inputs(y,d,origins):
    origins=np.asarray(origins)
    if origins.min()<169 or origins.max()>len(y) or origins.max()-1>len(d):raise ValueError('Insufficient history prefix')
    return {'origins':origins,'short':np.stack([y[origins-2],y[origins-1],d[origins-2]],axis=-1),
            'long_occupancy':np.stack([y[o-168:o] for o in origins]),
            'long_duration':np.stack([d[o-169:o-1] for o in origins])}


def baseline_predictions(y,origins,h):
    index=np.asarray(origins)[:,None]+np.arange(h)[None,:]
    if h>12 or np.any(index-24>=np.asarray(origins)[:,None]):raise ValueError('Historical visibility violated')
    return {'last':np.repeat(y[np.asarray(origins)-1,None,:],h,axis=1),'day':y[index-24],'week':y[index-168]}
