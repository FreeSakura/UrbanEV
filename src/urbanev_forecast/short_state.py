"""Shared inputs and ordinary regressors for the frozen short-state experiment."""
import numpy as np

NEURAL=('RELAX_SHORT_O','RELAX_SHORT_OD','DIRECT_SHORT_OD','DIRECT_LONG_OD')
SEEDS=(20260913,20260914)
HORIZONS=(3,6,9,12)
LIMITS={'FIT':{'occupancy.csv':1056,'duration.csv':1043},'SELECT':{'occupancy.csv':1224,'duration.csv':1211},'PREDICT':{'occupancy.csv':1548,'duration.csv':1547},'SCORE':{'occupancy.csv':1560}}


def origins(stage):
    if stage=='FIT':return np.arange(169,1045)
    if stage=='SELECT':return np.arange(1056,1213,12)
    if stage in ('PREDICT','SCORE'):return np.arange(1392,1549,12)
    raise ValueError('Unknown stage')


def phase(o,period):
    mod=np.asarray(o)%period;angle=2*np.pi*mod/period
    sn,cs=np.sin(angle),np.cos(angle)
    for k,s,c in [(0,0.,1.),(period//4,1.,0.),(period//2,0.,-1.),(3*period//4,-1.,0.)]:
        sn=np.where(mod==k,s,sn);cs=np.where(mod==k,c,cs)
    return sn,cs


def features(y,d,cap,oo,kind):
    oo=np.asarray(oo,dtype=int);n=len(cap)
    if min(oo)<169 or max(oo)>len(y) or max(oo)-1>len(d):raise ValueError('History outside prefix')
    s24,c24=phase(oo,24);s168,c168=phase(oo,168)
    calendar=np.stack([s24,c24,s168,c168],axis=1)
    context=np.concatenate([np.broadcast_to(calendar[:,None,:],(len(oo),n,4)),np.broadcast_to(np.log1p(cap)[None,:,None],(len(oo),n,1))],axis=2)
    if kind in ('SHORT_OD','SHORT_O'):
        duration=d[oo-2] if kind=='SHORT_OD' else np.zeros((len(oo),n))
        x=np.concatenate([np.stack([y[oo-2],y[oo-1],duration],axis=2),context],axis=2)
    elif kind=='LONG_OD':
        past_y=np.stack([y[o-168:o].T for o in oo]);past_d=np.stack([d[o-169:o-1].T for o in oo])
        x=np.concatenate([past_y,past_d,context],axis=2)
    else:raise ValueError('Unknown representation')
    return x.reshape(len(oo)*n,-1)


def targets(y,oo):
    oo=np.asarray(oo)
    if min(oo)<0 or max(oo)+12>len(y):raise ValueError('Target outside numeric prefix')
    return y[oo[:,None]+np.arange(12)].transpose(0,2,1).reshape(-1,12)


def fit_transform(x):
    mean=x.mean(0);std=x.std(0);inactive=std<=1e-10*np.maximum(1.,np.abs(mean))
    return {'mean':mean,'std':std,'inactive':inactive}


def transform(x,stats):
    z=(x-stats['mean'])/np.where(stats['inactive'],1.,stats['std'])
    z[:,stats['inactive']]=0.
    return z


def fit_ridge(x,residual):
    # Centered standardized features have an unpenalized intercept, solved explicitly.
    width=x.shape[1];gram=np.empty((width+1,width+1));gram[0,0]=1.
    gram[0,1:]=gram[1:,0]=x.mean(0);gram[1:,1:]=x.T@x/len(x)+np.eye(width)*.01
    rhs=np.vstack([residual.mean(0),x.T@residual/len(x)])
    return np.linalg.solve(gram,rhs)


def ridge_predict(x,anchor,beta):return anchor[:,None]+beta[0]+x@beta[1:]


def checkpoint_choice(rows):
    # Only primary endpoint RMSE; neither MAE nor an improvement threshold is consulted.
    return min(rows,key=lambda row:(row['selection_rmse'],row['epoch']))


class FitBudget:
    def __init__(self):self.neural=0;self.ridge=0;self.closed=False
    def start(self,kind):
        if self.closed:raise RuntimeError('Fitting closed')
        limit=8 if kind=='neural' else 2 if kind=='ridge' else 0
        if not limit or getattr(self,kind)>=limit:raise RuntimeError('Fit budget exceeded')
        setattr(self,kind,getattr(self,kind)+1)
    def close(self):
        if (self.neural,self.ridge)!=(8,2):raise RuntimeError('Required models incomplete')
        self.closed=True
