"""Independent small-case oracle: exact rational segment construction and Decimal80 optimization.

No imports from the production solver and no event-update reuse. Deliberately
rescans samples on every segment, so it is suitable only for small test cases.
"""
from decimal import Decimal,localcontext
from fractions import Fraction
import math
from functools import reduce


class RefBoundary:
    """Reference-only exact endpoint, independently implemented from production."""
    def __init__(self,value=None,poly=None,which=0):
        self.value=Fraction(value) if value is not None else None;self.poly=poly;self.which=which
    def identity(self):return ('fraction',self.value) if self.value is not None else ('root',self.poly,self.which)
    def bracket(self,k):
        if self.value is not None:return self.value,self.value
        a,b,c=self.poly;dis=b*b-4*a*c;q=1<<k;n=math.isqrt(dis*q*q)
        if self.which<0:return Fraction(-b*q-n-1,2*a*q),Fraction(-b*q-n,2*a*q)
        return Fraction(-b*q+n,2*a*q),Fraction(-b*q+n+1,2*a*q)
    def decimal(self):
        a,b=self.bracket(320);v=(a+b)/2
        return Decimal(v.numerator)/Decimal(v.denominator)
    def encode(self):
        if self.value is not None:return {'kind':'rational','numerator':str(self.value.numerator),'denominator':str(self.value.denominator)}
        return {'kind':'quadratic','polynomial':[str(x) for x in self.poly],'root':'lower' if self.which<0 else 'upper'}


def ref_compare(a,b):
    if a.identity()==b.identity():return 0
    for k in (80,160,320,640):
        al,au=a.bracket(k);bl,bu=b.bracket(k)
        if au<=bl:return -1
        if bu<=al:return 1
    raise ArithmeticError('Reference root ordering unresolved after640 bits')


def ref_intersect(interval,abc):
    low,high=interval;a,b,c=abc
    if a==0:
        if b==0:return interval if c<=0 else None
        root=RefBoundary(-c/b)
        if b>0:
            if ref_compare(root,high)<0:high=root
        elif ref_compare(root,low)>0:low=root
    else:
        assert a>0
        den=math.lcm(a.denominator,b.denominator,c.denominator)
        aa,bb,cc=[int(x*den) for x in (a,b,c)]
        g=reduce(math.gcd,(abs(aa),abs(bb),abs(cc)));aa,bb,cc=aa//g,bb//g,cc//g
        disc=bb*bb-4*aa*cc
        if disc<0:return None
        sq=math.isqrt(disc)
        if sq*sq==disc:
            left,right=RefBoundary(Fraction(-bb-sq,2*aa)),RefBoundary(Fraction(-bb+sq,2*aa))
        else:left,right=RefBoundary(poly=(aa,bb,cc),which=-1),RefBoundary(poly=(aa,bb,cc),which=1)
        if ref_compare(left,low)>0:low=left
        if ref_compare(right,high)<0:high=right
    return None if ref_compare(low,high)>0 else (low,high)


def reference_solve(cells):
    with localcontext() as ctx:
        ctx.prec=80
        F=lambda x:Fraction.from_float(float(x))
        D=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
        data=[[(F(p),F(d),F(y)) for p,d,y in zip(c['p'].ravel(),c['delta'].ravel(),c['y'].ravel())] for c in cells]
        clip=lambda x:max(Fraction(0),min(Fraction(1),x))
        native_mse=[sum((clip(p)-y)**2 for p,d,y in cell)/len(cell) for cell in data]
        native_mae=sum(sum(abs(clip(p)-y) for p,d,y in cell)/len(cell) for cell in data)/len(data)
        breaks={Fraction(0),Fraction(1)}
        for cell in data:
            for p,d,y in cell:
                if d:
                    for target in (Fraction(0),y,Fraction(1)):
                        t=(target-p)/d
                        if 0<t<1:breaks.add(t)
        breaks=sorted(breaks);components=[];candidates=[]
        def actual(alpha):
            metrics=[]
            for cell in data:
                e=[max(Decimal(0),min(Decimal(1),D(p)+alpha*D(d)))-D(y) for p,d,y in cell]
                metrics.append(((sum(v*v for v in e)/len(e)).sqrt(),sum(abs(v) for v in e)/len(e)))
            return sum(v[0] for v in metrics)/len(metrics),sum(v[1] for v in metrics)/len(metrics)
        exact_components=[]
        for left,right in zip(breaks[:-1],breaks[1:]):
            midpoint=(left+right)/2
            interval=(RefBoundary(left),RefBoundary(right))
            coefficients=[];vectors=[]
            for cell in data:
                rr=[];vv=[];sl=[]
                for p,d,y in cell:
                    middle_raw=p+midpoint*d
                    if middle_raw<=0:u,v=Fraction(0),Fraction(0)
                    elif middle_raw>=1:u,v=Fraction(1),Fraction(0)
                    else:u,v=p,d
                    middle=u+midpoint*v-y
                    rr.append(u-y);vv.append(v);sl.append(v if middle>0 else (-v if middle<0 else Fraction(0)))
                n=len(cell)
                coefficients.append((sum(v*v for v in rr)/n,sum(r*v for r,v in zip(rr,vv))/n,sum(v*v for v in vv)/n,
                                     sum((1 if r+midpoint*v>0 else (-1 if r+midpoint*v<0 else 0))*r for r,v in zip(rr,vv))/n,sum(sl)/n))
                vectors.append((list(map(D,rr)),list(map(D,vv))))
            B=sum(v[3] for v in coefficients)/len(coefficients)-native_mae
            m=sum(v[4] for v in coefficients)/len(coefficients)
            interval=ref_intersect(interval,(Fraction(0),m,B))
            if interval is None:continue
            for (w,g,u,b,m),native in zip(coefficients,native_mse):
                interval=ref_intersect(interval,(u,2*g,w-Fraction(10201,10000)*native))
                if interval is None:break
            if interval is None:continue
            low,high=interval;lo,hi=low.decimal(),high.decimal()
            if exact_components and ref_compare(low,exact_components[-1][1])<=0:
                if ref_compare(high,exact_components[-1][1])>0:exact_components[-1][1]=high
            else:exact_components.append([low,high])
            def derivative(t,side):
                terms=[]
                for rr,vv in vectors:
                    error=[r+t*v for r,v in zip(rr,vv)];n=len(error)
                    norm=(sum(r*r for r in error)/n).sqrt()
                    if norm:terms.append(sum(r*v for r,v in zip(error,vv))/n/norm)
                    else:terms.append(Decimal(side)*(sum(v*v for v in vv)/n).sqrt())
                return sum(terms)/len(terms)
            if ref_compare(low,high)==0 or derivative(lo,1)>=0:minimum=lo
            elif derivative(hi,-1)<0:minimum=hi
            else:
                l,r=lo,hi
                for _ in range(250):
                    if r-l<Decimal('1e-60'):break
                    mid=(l+r)/2
                    if derivative(mid,1)>=0:r=mid
                    else:l=mid
                minimum=r
            for point in (lo,hi,minimum):
                candidates.append((point,actual(point)[0]))
        candidates.append((Decimal(0),actual(Decimal(0))[0]))
        smallest=min(r for a,r in candidates)
        alpha=min(a for a,r in candidates if r<=smallest+Decimal('1e-12'))
        objective,mae=actual(alpha)
        return {'alpha':float(alpha),'objective':float(objective),'mae':float(mae),
                'components':[[float(a.decimal()),float(b.decimal())] for a,b in exact_components],
                'exact_components':[[a.encode(),b.encode()] for a,b in exact_components],
                'precision':80,'boundary_semantics':'independent exact rational and enclosing root comparison'}
