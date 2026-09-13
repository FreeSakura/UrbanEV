"""Fixed history masks and ridge sufficient statistics; no information-theory claim."""
import numpy as np

CONTEXT=list(range(336,341))
MASKS={
    'S':[166,167,335]+CONTEXT,
    'O_LONG':list(range(168))+[335]+CONTEXT,
    'D_LONG':[166,167]+list(range(168,336))+CONTEXT,
    'OD_LONG':list(range(341)),
    'O_LONG_D24':list(range(168))+list(range(312,336))+CONTEXT,
    'O24_D_LONG':list(range(144,336))+CONTEXT,
    'O_LONG_D0':list(range(168))+CONTEXT,
    'O_LONG_DUP':list(range(168))+[335]+list(range(167))+CONTEXT,
}
CONTRASTS=[('O_LONG','S'),('D_LONG','S'),('OD_LONG','O_LONG'),('OD_LONG','D_LONG'),
           ('O_LONG_D24','O_LONG'),('OD_LONG','O_LONG_D24'),('O24_D_LONG','D_LONG'),
           ('OD_LONG','O24_D_LONG'),('O_LONG','O_LONG_D0'),('O_LONG_DUP','O_LONG'),
           ('OD_LONG','O_LONG_DUP'),('OD_LONG','S')]


def sufficient_statistics(z,residual):
    n=len(z);k=z.shape[1]
    g=np.empty((k+1,k+1));g[0,0]=1;g[0,1:]=g[1:,0]=z.mean(0);g[1:,1:]=z.T@z/n
    b=np.vstack([residual.mean(0),z.T@residual/n])
    return g,b


def solve_mask(g,b,mask):
    ix=np.r_[0,np.asarray(mask)+1];gg=g[np.ix_(ix,ix)];bb=b[ix]
    a=gg.copy();a[1:,1:]+=np.eye(len(mask))*.01
    solution=np.linalg.solve(a,np.column_stack([bb,gg]));beta=solution[:,:bb.shape[1]]
    denominator=np.linalg.norm(a,np.inf)*np.linalg.norm(beta,np.inf)+np.linalg.norm(bb,np.inf)
    residual=float(np.linalg.norm(a@beta-bb,np.inf)/max(denominator,np.finfo(float).tiny))
    edf=float(np.trace(solution[:,bb.shape[1]:]))
    if not np.isfinite(beta).all() or not np.isfinite(edf) or residual>1e-10:raise ValueError('Ridge normal equation check failed')
    return beta,{'normal_equation_relative_residual':residual,'effective_df':edf,'nominal_columns':len(mask),'unique_column_identities':len(set(mask)),'coefficient_count':int(beta.size)}


def predict(z,anchor,mask,beta):return anchor[:,None]+beta[0]+z[:,mask]@beta[1:]


def factorial(mse):
    s,o,d,od=(mse[k] for k in ('S','O_LONG','D_LONG','OD_LONG'))
    out={'O_given_short':s-o,'D_given_short':s-d,'D_given_longO':o-od,'O_given_longD':d-od,'total':s-od,'interaction_J':o+d-s-od,
         'D_1_to_24_given_longO':o-mse['O_LONG_D24'],'D_24_to_168_given_longO':mse['O_LONG_D24']-od,
         'O_2_to_24_given_longD':d-mse['O24_D_LONG'],'O_24_to_168_given_longD':mse['O24_D_LONG']-od,
         'single_D_given_longO':mse['O_LONG_D0']-o,'duplicate_O_effect':o-mse['O_LONG_DUP'],'true_D_vs_duplicate_O':mse['O_LONG_DUP']-od}
    out['path_identity_max_error']=max(abs(out['total']-out['O_given_short']-out['D_given_longO']),abs(out['total']-out['D_given_short']-out['O_given_longD']))
    return out


def loss_attribution(target,reference,candidate):
    eb=target-reference;ec=target-candidate;delta=candidate-reference
    g1=abs(eb)-abs(ec);g2=eb**2-ec**2
    error=float(np.max(abs(g2-(2*eb*delta-delta**2))))
    out={'sample_count':target.size,'loss_identity_error':error}
    for key,g in [('mae',g1),('mse',g2)]:
        out[key]={'improved_count':int((g>0).sum()),'worsened_count':int((g<0).sum()),'equal_count':int((g==0).sum()),'positive_gain_sum':float(g[g>0].sum()),'negative_gain_sum':float(g[g<0].sum()),'net_mean_gain':float(g.mean())}
    if error>1e-10:raise ValueError('Loss decomposition identity failed')
    return out,g1,g2
