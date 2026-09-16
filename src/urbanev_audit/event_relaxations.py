"""All pairwise implications and their exact order-polytope LP relaxation."""
import time
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from .persistent_events import label_bounds, njit


@njit(cache=True)
def _implication_intervals(o, window, run_length, ambiguous):
    t=len(o);streak=np.zeros(t+1,np.int64);suffix=np.zeros(t+1,np.int64)
    zeros=np.zeros(t+1,np.int64)
    for i in range(t):
        streak[i+1]=streak[i]+1 if o[i]==1 else 0
        zeros[i+1]=zeros[i]+(o[i]==0)
    for i in range(t-1,-1,-1):suffix[i]=suffix[i+1]+1 if o[i]==1 else 0
    first=np.empty(len(ambiguous),np.int64);last=np.empty(len(ambiguous),np.int64)
    for q in range(len(ambiguous)):
        s=ambiguous[q];lower=-t;upper=2*t
        for r in range(s,s+window-run_length+1):
            if zeros[r+run_length]==zeros[r]:
                # Minimal completion for this run: extend only through observed ones.
                left=r-streak[r];right=r+run_length-1+suffix[r+run_length]
                lower=max(lower,left-window+run_length)
                upper=min(upper,right-run_length+1)
        first[q]=lower;last[q]=upper
    return first,last


class PairwiseLP:
    """0<=e<=1 and every valid e_i<=e_j, built without prediction scores.

    An interval segment graph represents all implication targets compactly.
    Eliminating its auxiliary variables gives exactly the explicit pairwise LP.
    """
    def __init__(self,observed,window,run_length):
        start=time.perf_counter()
        self.lo,self.hi=label_bounds(observed,window,run_length)
        self.ambiguous=np.flatnonzero(self.lo!=self.hi)
        n=len(self.ambiguous);self.edge_count=0;self.variable_count=n;self.matrix=None
        self.first=np.empty(0,dtype=int);self.last=np.empty(0,dtype=int)
        self.lp_indices=np.empty(0,dtype=int)
        if n:
            self.first,self.last=_implication_intervals(np.asarray(observed,np.int8),window,run_length,self.ambiguous)
            left=np.searchsorted(self.ambiguous,self.first)
            right=np.searchsorted(self.ambiguous,self.last,side='right')
            self.edge_count=int(np.sum(right-left-1))
            if self.edge_count:
                delta=np.zeros(n+1,dtype=int)
                valid=right-left>1
                np.add.at(delta,left[valid],1);np.add.at(delta,right[valid],-1)
                self.lp_indices=np.flatnonzero(np.cumsum(delta[:-1])>0)
                left=np.searchsorted(self.lp_indices,left[self.lp_indices])
                right=np.searchsorted(self.lp_indices,right[self.lp_indices])
                n=len(self.lp_indices)
                edges=[];nodes=[]
                def tree(lo,hi):
                    if hi-lo==1:return lo
                    node=n+len(nodes);nodes.append(None);mid=(lo+hi)//2
                    a,b=tree(lo,mid),tree(mid,hi);nodes[node-n]=(lo,hi,a,b)
                    edges.extend([(node,a),(node,b)]);return node
                root=tree(0,n)
                def cover(source,node,lo,hi):
                    if node<n:
                        if lo<=node<hi and node!=source:edges.append((source,node))
                        return
                    x,y,a,b=nodes[node-n]
                    if hi<=x or lo>=y:return
                    if lo<=x and y<=hi:edges.append((source,node));return
                    cover(source,a,lo,hi);cover(source,b,lo,hi)
                for i,(a,b) in enumerate(zip(left,right)):
                    # Excluding self avoids unnecessary cycles in singleton ranges.
                    if a<i:cover(i,root,int(a),i)
                    if i+1<b:cover(i,root,i+1,int(b))
                self.variable_count=n+len(nodes)
                e=np.asarray(edges,dtype=np.int64)
                rr=np.repeat(np.arange(len(e)),2);cc=e.reshape(-1);vv=np.tile([1.,-1.],len(e))
                self.matrix=coo_matrix((vv,(rr,cc)),shape=(len(e),self.variable_count)).tocsr()
        self.preprocess_seconds=time.perf_counter()-start

    def bounds(self,a,b,timeout=120):
        start=time.perf_counter();a=np.asarray(a);b=np.asarray(b);c=2*(b-a);n=len(c)
        base=float(np.sum(a*a-b*b+c*self.lo));w=c[self.ambiguous]
        if self.matrix is None or np.all(w>=0) or np.all(w<=0):
            low=float(np.minimum(w,0).sum());high=float(np.maximum(w,0).sum());status='ANALYTIC'
        else:
            active=w[self.lp_indices];isolated=np.ones(len(w),bool);isolated[self.lp_indices]=False
            isolated_low=float(np.minimum(w[isolated],0).sum());isolated_high=float(np.maximum(w[isolated],0).sum())
            objective=np.zeros(self.variable_count);objective[:len(active)]=active
            low_result=linprog(objective,A_ub=self.matrix,b_ub=np.zeros(self.matrix.shape[0]),bounds=(0,1),method='highs',options={'time_limit':timeout})
            high_result=linprog(-objective,A_ub=self.matrix,b_ub=np.zeros(self.matrix.shape[0]),bounds=(0,1),method='highs',options={'time_limit':timeout})
            if not low_result.success or not high_result.success:
                return {'lower':None,'upper':None,'status':'CUTOFF_OR_FAILURE','solver_messages':[low_result.message,high_result.message],
                        'seconds':time.perf_counter()-start}
            # Any nonnegative Lagrange multiplier gives a valid box-relaxation
            # lower bound. Round the fsum residuals outwards, not inward.
            matrix=self.matrix.tocsc()
            def dual_lower(cost,result):
                lam=np.maximum(0.,-result.ineqlin.marginals)
                residual=[]
                for col in range(matrix.shape[1]):
                    begin,end=matrix.indptr[col],matrix.indptr[col+1]
                    value=math.fsum([float(cost[col])]+[float(v*lam[row]) for row,v in zip(matrix.indices[begin:end],matrix.data[begin:end])])
                    residual.append(min(0.,np.nextafter(value,-np.inf)))
                return float(np.nextafter(math.fsum(residual),-np.inf))
            low=dual_lower(objective,low_result)+isolated_low
            high=-dual_lower(-objective,high_result)+isolated_high;status='OPTIMAL_DUAL_OUTWARD'
        return {'lower':(base+low)/n,'upper':(base+high)/n,'status':status,'seconds':time.perf_counter()-start}

    def preference_conflict(self,coefficients,endpoint):
        weights=np.asarray(coefficients)[self.ambiguous]
        want_one=weights<0 if endpoint=='lower' else weights>0
        want_zero=weights>0 if endpoint=='lower' else weights<0
        zero_starts=self.ambiguous[want_zero]
        for q in np.flatnonzero(want_one):
            j=int(np.searchsorted(zero_starts,self.first[q]))
            if j<len(zero_starts) and zero_starts[j]<=self.last[q]:
                return [int(self.ambiguous[q]),int(zero_starts[j])]
        return []
