"""Independent small-case oracle: exact rational segment construction and Decimal80 optimization.

No imports from the production solver and no event-update reuse. Deliberately
rescans samples on every segment, so it is suitable only for small test cases.
"""
from decimal import Decimal,localcontext
from fractions import Fraction


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
        for left,right in zip(breaks[:-1],breaks[1:]):
            midpoint=(left+right)/2;length=D(right-left);lo,hi=Decimal(0),length
            coefficients=[];vectors=[]
            for cell in data:
                rr=[];vv=[];sl=[]
                for p,d,y in cell:
                    q=clip(p+left*d);v=d if 0<p+midpoint*d<1 else Fraction(0)
                    middle=clip(p+midpoint*d)-y
                    rr.append(q-y);vv.append(v);sl.append(v if middle>0 else (-v if middle<0 else Fraction(0)))
                n=len(cell)
                coefficients.append((sum(v*v for v in rr)/n,sum(r*v for r,v in zip(rr,vv))/n,sum(v*v for v in vv)/n,sum(abs(r) for r in rr)/n,sum(sl)/n))
                vectors.append((list(map(D,rr)),list(map(D,vv))))
            B=sum(v[3] for v in coefficients)/len(coefficients)-native_mae
            m=sum(v[4] for v in coefficients)/len(coefficients)
            if m==0:
                if B>0:continue
            elif m>0:hi=min(hi,D(-B/m))
            else:lo=max(lo,D(-B/m))
            for (w,g,u,b,m),native in zip(coefficients,native_mse):
                a=u;bb=2*g;cc=w-Fraction(10201,10000)*native
                if a==0:
                    if bb==0:
                        if cc>0:hi=Decimal(-1);break
                    elif bb>0:hi=min(hi,D(-cc/bb))
                    else:lo=max(lo,D(-cc/bb))
                else:
                    discriminant=bb*bb-4*a*cc
                    if discriminant<0:hi=Decimal(-1);break
                    root=D(discriminant).sqrt()
                    low=(-D(bb)-root)/(2*D(a));high=(-D(bb)+root)/(2*D(a))
                    lo,hi=max(lo,low),min(hi,high)
            if lo>hi:continue
            aa,bb=D(left)+lo,D(left)+hi
            if components and aa<=components[-1][1]:components[-1][1]=max(components[-1][1],bb)
            else:components.append([aa,bb])
            def derivative(t,side):
                terms=[]
                for rr,vv in vectors:
                    error=[r+t*v for r,v in zip(rr,vv)];n=len(error)
                    norm=(sum(r*r for r in error)/n).sqrt()
                    if norm:terms.append(sum(r*v for r,v in zip(error,vv))/n/norm)
                    else:terms.append(Decimal(side)*(sum(v*v for v in vv)/n).sqrt())
                return sum(terms)/len(terms)
            if lo==hi or derivative(lo,1)>=0:minimum=lo
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
                alpha=D(left)+point;candidates.append((alpha,actual(alpha)[0]))
        candidates.append((Decimal(0),actual(Decimal(0))[0]))
        smallest=min(r for a,r in candidates)
        alpha=min(a for a,r in candidates if r<=smallest+Decimal('1e-12'))
        objective,mae=actual(alpha)
        return {'alpha':float(alpha),'objective':float(objective),'mae':float(mae),
                'components':[[float(a),float(b)] for a,b in components],'precision':80}
