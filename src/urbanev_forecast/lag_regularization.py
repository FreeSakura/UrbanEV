"""Time-ordered generalized ridge controls in standardized D-lag coordinates."""
import numpy as np
FAMILIES=('TIME_SMOOTH_D','SCRAMBLED_SMOOTH_D','D_NORM','GLOBAL_RIDGE')
GAMMAS=(0.,.01,.1,1.)
SYSTEMS=('BASE_OD','TIME_SELECTED','SCRAMBLED_SELECTED','D_NORM_SELECTED','GLOBAL_RIDGE_SELECTED','SCRAMBLED_MATCHED')

def fit_id(family,gamma):return 'BASE_OD' if gamma==0 else family+'_g'+format(gamma,'g')

def penalties(inactive):
    active=np.flatnonzero(~np.asarray(inactive,bool));k=len(active)
    dcols=active[(active>=168)&(active<336)];m=len(dcols)
    dmap={int(v):i for i,v in enumerate(dcols)};rows=[]
    for origin in range(168,334):
        if all(origin+j in dmap for j in range(3)):
            row=np.zeros(m);row[[dmap[origin+j] for j in range(3)]]=[1,-2,1];rows.append(row)
    if not rows:raise ValueError('No fully active D triple; no fallback')
    l=np.stack(rows);local=l.T@l;trace=float(np.trace(local));qt=local*m/trace
    perm=np.random.default_rng(20260913).permutation(m);p=np.eye(m)[perm];qs=p.T@qt@p
    if np.array_equal(qt,qs):raise ValueError('Scrambling did not change penalty; no redraw')
    reduced_d=np.array([int(np.flatnonzero(active==v)[0])+1 for v in dcols]);size=k+1
    qtime=np.zeros((size,size));qtime[np.ix_(reduced_d,reduced_d)]=qt
    qscr=np.zeros_like(qtime);qscr[np.ix_(reduced_d,reduced_d)]=qs
    qnorm=np.zeros_like(qtime);qnorm[reduced_d,reduced_d]=1
    qglobal=np.diag(np.r_[0.,np.full(k,m/k)])
    matrices=dict(zip(FAMILIES,(qtime,qscr,qnorm,qglobal)))
    checks={}
    for name,q in matrices.items():
        eigen=np.linalg.eigvalsh(q);sym=float(np.max(abs(q-q.T)))
        if sym>1e-10 or eigen.min() < -1e-10 or abs(np.trace(q)-m)>1e-10:raise ValueError('Penalty PSD/symmetry/trace mismatch')
        checks[name]={'trace':float(np.trace(q)),'rank_tolerance_1e_10':int(np.sum(eigen>1e-10)),'minimum_eigenvalue':float(eigen.min()),'symmetry_error':sym}
    spectrum_error=float(np.max(abs(np.linalg.eigvalsh(qt)-np.linalg.eigvalsh(qs))))
    if spectrum_error>1e-10:raise ValueError('Scrambled spectrum changed')
    manifest={'active_features':active.tolist(),'inactive_features':np.flatnonzero(inactive).tolist(),'active_D_features':dcols.tolist(),'k':k,'m':m,'difference_rows':len(rows),'unnormalized_trace':trace,'permutation':perm.tolist(),'penalty_checks':checks,'time_scrambled_spectrum_max_error':spectrum_error,'coordinate_system':'FIT-standardized coefficients, not physical response kernels','same_spectrum_is_not_same_edf':True}
    return active,reduced_d,l,matrices,manifest

def solve(g,b,active,q,gamma,residual_second_moment,l,dpos):
    ix=np.r_[0,active+1];gg=g[np.ix_(ix,ix)];bb=b[ix]
    base=np.diag(np.r_[0.,np.full(len(active),.01)]);a=gg+base+gamma*q
    sol=np.linalg.solve(a,np.column_stack([bb,gg]));beta=sol[:,:bb.shape[1]]
    denominator=np.linalg.norm(a,np.inf)*np.linalg.norm(beta,np.inf)+np.linalg.norm(bb,np.inf)
    residual=float(np.linalg.norm(a@beta-bb,np.inf)/max(denominator,np.finfo(float).tiny))
    if not np.isfinite(sol).all() or residual>1e-10:raise ValueError('Normal equations failed')
    full=np.zeros((g.shape[0],b.shape[1]));full[ix]=beta
    data_loss=residual_second_moment-2*np.sum(beta*bb,axis=0)+np.sum(beta*(gg@beta),axis=0)
    base_pen=.01*np.sum(beta[1:]**2,axis=0);penalty=np.sum(beta*(q@beta),axis=0);extra=gamma*penalty
    roughness=np.linalg.norm(l@beta[dpos],axis=0);dnorm=np.linalg.norm(beta[dpos],axis=0)
    details={'normal_equation_relative_residual':residual,'edf':float(np.trace(sol[:,bb.shape[1]:])),'coefficient_count':int(beta.size),'full_coefficient_shape':list(full.shape),'training_objective_total':float(np.sum(data_loss+base_pen+extra)),'per_output':[{'future_hour':j+1,'data_mse':float(data_loss[j]),'base_penalty':float(base_pen[j]),'actual_Q_penalty':float(penalty[j]),'extra_penalty':float(extra[j]),'training_objective':float(data_loss[j]+base_pen[j]+extra[j]),'D_second_difference_norm':float(roughness[j]),'D_coefficient_norm':float(dnorm[j])} for j in range(b.shape[1])]}
    return full,details

def select(scores):
    chosen={family:min(GAMMAS,key=lambda gamma:(scores[fit_id(family,gamma)],gamma)) for family in FAMILIES}
    mapping={'BASE_OD':'BASE_OD','TIME_SELECTED':fit_id(FAMILIES[0],chosen[FAMILIES[0]]),'SCRAMBLED_SELECTED':fit_id(FAMILIES[1],chosen[FAMILIES[1]]),'D_NORM_SELECTED':fit_id(FAMILIES[2],chosen[FAMILIES[2]]),'GLOBAL_RIDGE_SELECTED':fit_id(FAMILIES[3],chosen[FAMILIES[3]]),'SCRAMBLED_MATCHED':fit_id(FAMILIES[1],chosen[FAMILIES[0]])}
    return chosen,mapping

class Budget:
    def __init__(self):self.ids=[];self.closed=False
    def start(self,id):
        allowed={'BASE_OD'}|{fit_id(f,g) for f in FAMILIES for g in GAMMAS[1:]}
        if self.closed or id not in allowed or id in self.ids:raise RuntimeError('Fit budget or identity violated')
        self.ids.append(id)
    def close(self):
        if len(self.ids)!=13:raise RuntimeError('Thirteen distinct fits required')
        self.closed=True

def error_groups(y,base_raw):return np.searchsorted([.01,.05,.10],abs(y-base_raw),side='right')

def grouped_loss(y,base_raw,reference,candidate):
    eb=y-reference;et=y-candidate;change=candidate-reference;g1=abs(eb)-abs(et);g2=eb**2-et**2
    identity=float(np.max(abs(g2-(2*eb*change-change**2))));group=error_groups(y,base_raw);bins=[]
    for index,label in enumerate(('[0,0.01)','[0.01,0.05)','[0.05,0.10)','[0.10,infinity)')):
        mask=group==index;row={'group':label,'sample_count':int(mask.sum())}
        for name,v in [('mae',g1[mask]),('mse',g2[mask])]:row[name]={'improved_count':int((v>0).sum()),'worsened_count':int((v<0).sum()),'equal_count':int((v==0).sum()),'positive_gain_sum':float(v[v>0].sum()),'negative_gain_sum':float(v[v<0].sum()),'net_sum':float(v.sum()),'mean_gain':float(v.mean()) if v.size else None}
        bins.append(row)
    restore=max(abs(sum(x['mae']['net_sum'] for x in bins)-g1.sum()),abs(sum(x['mse']['net_sum'] for x in bins)-g2.sum()))
    if identity>1e-10 or restore>1e-10:raise ValueError('Grouped loss identity failed')
    return {'sample_count':y.size,'base_group_definition':'absolute BASE_OD raw error, shared across postprocessing and references','mae_mean_gain':float(g1.mean()),'mse_mean_gain':float(g2.mean()),'identity_error':identity,'group_sum_error':float(restore),'groups':bins},g1,g2
