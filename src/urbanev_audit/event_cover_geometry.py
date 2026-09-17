"""Observed-run interval geometry for fixed-K persistent-event labels.

Coverage tests apply for all 1<=L<=K. Completeness and the polynomial
pairwise-sufficiency criterion require K>=2L-1.
"""
from itertools import combinations
import numpy as np
from .persistent_events import label_bounds


class EventCoverGeometry:
    def __init__(self, observed, window, run_length):
        self.o=np.asarray(observed,np.int8);self.k=int(window);self.ell=int(run_length)
        if not 1<=self.ell<=self.k<=len(self.o):raise ValueError('Require 1 <= L <= K <= T')
        if not np.isin(self.o,[-1,0,1]).all():raise ValueError('Expected -1/0/1 observations')
        self.lo,self.hi=label_bounds(self.o,self.k,self.ell)
        self.n=len(self.lo);self.ambiguous=np.flatnonzero(self.lo!=self.hi)
        t=len(self.o);prefix=np.zeros(t+1,dtype=int);suffix=np.zeros(t+1,dtype=int)
        for i in range(t):prefix[i+1]=prefix[i]+1 if self.o[i]==1 else 0
        for i in range(t-1,-1,-1):suffix[i]=suffix[i+1]+1 if self.o[i]==1 else 0
        zeros=np.r_[0,np.cumsum(self.o==0)]
        self.runs=np.flatnonzero(zeros[self.ell:]==zeros[:-self.ell])
        left=self.runs-prefix[self.runs]
        right=self.runs+self.ell-1+suffix[self.runs+self.ell]
        self.left=left-self.k+self.ell
        self.right=right-self.ell+1

    def witnesses(self,j):
        a=np.searchsorted(self.runs,j);b=np.searchsorted(self.runs,j+self.k-self.ell,side='right')
        return self.runs[a:b],self.left[a:b],self.right[a:b]

    def cover_valid(self,j,cover):
        """Exact e_j <= sum_{i in cover} e_i test, including known labels."""
        cover=np.asarray(cover,dtype=int)
        if self.hi[j]==0:return True
        if j in cover or np.any(self.lo[cover]==1):return True
        if self.lo[j]==1:return False # all missing bits 0 refutes the cover
        _,left,right=self.witnesses(j)
        return all(np.any((cover>=a)&(cover<=b)) for a,b in zip(left,right))

    def diagnose(self):
        """A private witness for every ambiguous label, or an essential 2-cover.

        No trajectory optimization and no prediction weights are used.
        Each failed private-witness test returns i<j<k with j -> i OR k,
        but neither j -> i nor j -> k. This proves structural insufficiency
        for any K,L. Passing is sufficient only in the stated long-window regime.
        """
        records=[]
        for j in self.ambiguous:
            runs,left,right=self.witnesses(j);a=int(left.max());b=int(right.min())
            pi=np.searchsorted(self.ambiguous,a)-1;si=np.searchsorted(self.ambiguous,b,side='right')
            prev=int(self.ambiguous[pi]) if pi>=0 else None
            nxt=int(self.ambiguous[si]) if si<len(self.ambiguous) else None
            private=(np.ones(len(runs),bool) if prev is None else left>prev)&(np.ones(len(runs),bool) if nxt is None else right<nxt)
            record={'origin':int(j),'implied_origin_left':a,'implied_origin_right':b,
                    'private_run_start':int(runs[np.flatnonzero(private)[0]]) if private.any() else None,
                    'left_cover_origin':None,'right_cover_origin':None}
            if not private.any():
                assert prev is not None and nxt is not None
                record.update(left_cover_origin=prev,right_cover_origin=nxt)
            records.append(record)
        failures=[r for r in records if r['private_run_start'] is None]
        status='PAIRWISE_INSUFFICIENT' if failures else ('PAIRWISE_EXACT' if self.k>=2*self.ell-1 else 'COVERS_COMPLETE_BUT_BRIDGES_UNCHECKED')
        return {'status':status,'long_window_regime':self.k>=2*self.ell-1,'ambiguous_labels':len(records),
                'essential_cover_count':len(failures),'label_records':records}

    def binary_cover_feasible(self, labels):
        """Test all zero-window covers by only their nearest left/right points."""
        y=np.asarray(labels,np.int8)
        if np.any(y<self.lo) or np.any(y>self.hi):return False
        zero=np.flatnonzero(y==0)
        for j in np.flatnonzero(y==1):
            if self.lo[j]:continue
            loc=np.searchsorted(zero,j);cover=[]
            if loc:cover.append(int(zero[loc-1]))
            if loc<len(zero):cover.append(int(zero[loc]))
            if self.cover_valid(j,cover):return False
        return True

    def realize(self,labels):
        """Construct a compatible trace in the long-window regime, or return None."""
        if self.k<2*self.ell-1:raise ValueError('Construction requires K >= 2L-1')
        y=np.asarray(labels,np.int8)
        if y.shape!=(self.n,) or not np.isin(y,[0,1]).all():raise ValueError('Expected full binary label vector')
        if np.any(y<self.lo) or np.any(y>self.hi):return None
        zeros=np.flatnonzero(y==0);delta=np.zeros(len(self.o)+1,dtype=int)
        for j in np.flatnonzero((y==1)&(self.lo==0)):
            runs,left,right=self.witnesses(j);loc=np.searchsorted(zeros,j)
            safe=np.ones(len(runs),bool)
            if loc:safe &= left>zeros[loc-1]
            if loc<len(zeros):safe &= right<zeros[loc]
            if not safe.any():return None
            r=int(runs[np.flatnonzero(safe)[0]]);delta[r]+=1;delta[r+self.ell]-=1
        bits=((self.o==1)|(np.cumsum(delta[:-1])>0)).astype(np.int8)
        return bits

    def cover_lp(self, coefficients):
        """Diagnostic LP of every valid unary/binary cover; not a hull claim."""
        from scipy.optimize import linprog
        rows=[]
        for j in self.ambiguous:
            others=[int(i) for i in self.ambiguous if i!=j]
            for size in (1,2):
                for cover in combinations(others,size):
                    if self.cover_valid(j,cover):
                        row=np.zeros(self.n);row[j]=1
                        for i in cover:row[i]-=1
                        rows.append(row)
        result=linprog(coefficients,A_ub=np.asarray(rows) if rows else None,
                       b_ub=np.zeros(len(rows)) if rows else None,
                       bounds=list(zip(self.lo,self.hi)),method='highs')
        if not result.success:raise RuntimeError(result.message)
        return result
