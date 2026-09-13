"""One held-fixed reference, linear repair control and prespecified fixed40 probe."""
import numpy as np
from . import short_state as base
from .conditional_od import InteractionNet as OriginalNet, learning_rate, choose_checkpoint, state_groups, grouped_losses
from .reference_harm import relative_stability
from .forward_residual_transfer import meta_origins, fit_model as original_fit_model, predict_model

MODELS=('FIXED_PRODUCT','FIXED_SEPARABLE','FIXED_CONCAT')
SEEDS=(20260915,20260916,20260917)
CHECKPOINTS=(0,5,10,20,40)
DETERMINISTIC=('BASE_FIXED','LINEAR_REPAIR','RIDGE_FULL')

class InteractionNet(OriginalNet):
    def __init__(self,scheme,seed):
        mapping={'FIXED_PRODUCT':'OD_PRODUCT','FIXED_SEPARABLE':'OD_SEPARABLE','FIXED_CONCAT':'OD_CONCAT_MLP'}
        if scheme not in mapping:raise ValueError('Unknown scheme')
        super().__init__(mapping[scheme],seed)

def parameter_count(scheme):
    if scheme not in MODELS:raise ValueError('Unknown neural scheme')
    return 9108

def mechanism_origins():return np.arange(1224,1381,12)

def mechanism_checkpoint(run):
    rows=[r for r in run['checkpoints'] if r['epoch']==40]
    if len(rows)!=1:raise ValueError('Exactly one registered epoch40 required')
    return rows[0]

def fit_scale(e64):
    if e64.dtype!=np.float64 or not np.isfinite(e64).all():raise ValueError('Finite raw float64 fixed-reference residual required')
    rms=float(np.sqrt(np.mean(e64**2)))
    return {'rms_float64':rms,'scale_float64':max(rms,1e-6),'floor_applied':rms<1e-6,'source':'BASE_FIXED residual on common meta origins; reference did not fit these labels'}

def normal_equation_error(z,e64,beta):
    width=z.shape[1];g=np.empty((width+1,width+1));g[0,0]=1.;g[0,1:]=g[1:,0]=z.mean(0)
    g[1:,1:]=z.T@z/len(z)+np.eye(width)*.01;rhs=np.vstack([e64.mean(0),z.T@e64/len(z)])
    denom=np.linalg.norm(g,np.inf)*np.linalg.norm(beta,np.inf)+np.linalg.norm(rhs,np.inf)
    error=float(np.linalg.norm(g@beta-rhs,np.inf)/max(denom,np.finfo(float).tiny))
    if not np.isfinite(beta).all() or error>1e-10:raise ValueError('Ridge normal equation failure')
    return error

def fit_model(y,d,cap,origins):
    model=original_fit_model(y,d,cap,origins)
    z=base.transform(base.features(y,d,cap,origins,'LONG_OD'),model['stats'])
    residual=base.targets(y,origins)-y[np.asarray(origins)-1].reshape(-1,1)
    model['normal_equation_relative_error']=normal_equation_error(z,residual,model['beta'])
    return model

def linear_repair(z,e64):
    beta=base.fit_ridge(z,e64);error=normal_equation_error(z,e64,beta)
    return beta,{'normal_equation_relative_error':error,'coefficients':int(beta.size),'lambda':.01,'target_units':'raw occupancy residual'}

def alignment(target,reference,candidate):
    e=target-reference;delta=candidate-reference
    a=float(np.mean(e*delta));b=float(np.mean(delta*delta));gain=float(np.mean(e*e-(target-candidate)**2));error=abs(gain-(2*a-b))
    if error>1e-10:raise ValueError('Alignment identity failed')
    return {'alignment_A':a,'correction_energy_B':b,'mse_gain_G':gain,'identity_error':error,'zero_correction':bool(np.count_nonzero(delta)==0)}

class Budget:
    def __init__(self):self.deterministic=[];self.neural=[];self.closed=False
    def start_deterministic(self,name):
        if self.closed or name not in DETERMINISTIC or name in self.deterministic or name!=DETERMINISTIC[len(self.deterministic)]:raise RuntimeError('Deterministic fit order/budget violation')
        if name=='RIDGE_FULL' and len(self.neural)!=9:raise RuntimeError('Full reference only after neural training')
        self.deterministic.append(name)
    def start_neural(self,name,seed):
        if self.closed or len(self.deterministic)!=2 or name not in MODELS or seed not in SEEDS or (name,seed) in self.neural:raise RuntimeError('Neural budget violation')
        self.neural.append((name,seed))
    def close(self):
        if len(self.deterministic)!=3 or len(self.neural)!=9:raise RuntimeError('Required fits missing')
        self.closed=True
