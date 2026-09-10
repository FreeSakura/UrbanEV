"""Exact rational/quadratic boundaries and containing rational intervals.

No rounded scan coefficients enter this module. Refinement is finite and an
unresolved relation is never silently interpreted as equality or emptiness.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import reduce
import math

BITS=(64,128,256,512)


class BoundaryUnresolved(ArithmeticError):
    pass


def note_bits(stats,*values):
    if stats is None:return
    for value in values:
        if isinstance(value,F):size=max(abs(value.numerator).bit_length(),value.denominator.bit_length())
        else:size=abs(int(value)).bit_length()
        stats['max_exact_integer_bits']=max(stats.get('max_exact_integer_bits',0),size)


def sqrt_bounds(value,bits):
    """Containing interval for sqrt(nonnegative rational), no decimal rounding."""
    value=F(value)
    if value<0:raise BoundaryUnresolved('NEGATIVE_EXACT_RADICAND')
    nr,dr=math.isqrt(value.numerator),math.isqrt(value.denominator)
    if nr*nr==value.numerator and dr*dr==value.denominator:
        exact=F(nr,dr);return exact,exact
    scale=1<<bits
    integer=math.isqrt((value.numerator<<(2*bits))//value.denominator)
    return F(integer,scale),F(integer+1,scale)


@dataclass(frozen=True)
class Boundary:
    rational:F|None=None
    polynomial:tuple|None=None
    branch:int=0

    @staticmethod
    def of(value):return Boundary(rational=F(value))

    def bounds(self,bits=64):
        if self.rational is not None:return self.rational,self.rational
        a,b,c=self.polynomial;disc=b*b-4*a*c
        integer=math.isqrt(disc<<(2*bits));scale=1<<bits
        low,high=F(integer,scale),F(integer+1,scale)
        if self.branch<0:return (-b-high)/(2*a),(-b-low)/(2*a)
        return (-b+low)/(2*a),(-b+high)/(2*a)

    def identity(self):
        if self.rational is not None:return ('r',self.rational.numerator,self.rational.denominator)
        return ('q',*self.polynomial,self.branch)

    def encode(self):
        if self.rational is not None:
            return {'kind':'rational','numerator':str(self.rational.numerator),'denominator':str(self.rational.denominator)}
        return {'kind':'quadratic','polynomial':[str(v) for v in self.polynomial],
                'root':'lower' if self.branch<0 else 'upper'}

    def approximate(self):
        a,b=self.bounds(128)
        return float((a+b)/2)


def quadratic_roots(a,b,c,stats=None):
    a,b,c=map(F,(a,b,c))
    if a<=0:raise BoundaryUnresolved('QUADRATIC_REQUIRES_POSITIVE_LEADING_TERM')
    common=math.lcm(a.denominator,b.denominator,c.denominator)
    aa,bb,cc=[int(v*common) for v in (a,b,c)]
    divisor=reduce(math.gcd,(abs(aa),abs(bb),abs(cc)))
    aa,bb,cc=aa//divisor,bb//divisor,cc//divisor
    if aa<0:aa,bb,cc=-aa,-bb,-cc
    disc=bb*bb-4*aa*cc;note_bits(stats,aa,bb,cc,disc)
    if disc<0:return []
    integer=math.isqrt(disc)
    if integer*integer==disc:
        return [Boundary.of(F(-bb-integer,2*aa)),Boundary.of(F(-bb+integer,2*aa))]
    return [Boundary(polynomial=(aa,bb,cc),branch=-1),Boundary(polynomial=(aa,bb,cc),branch=1)]


def compare_boundaries(a,b,stats=None):
    if a.identity()==b.identity():return 'EQUAL'
    for bits in BITS:
        if stats is not None:stats['max_root_refinement_bits']=max(stats.get('max_root_refinement_bits',0),bits)
        al,au=a.bounds(bits);bl,bu=b.bounds(bits)
        if au<=bl:return 'LESS'
        if bu<=al:return 'GREATER'
    if stats is not None:stats['boundary_comparisons_unresolved']=stats.get('boundary_comparisons_unresolved',0)+1
    return 'UNRESOLVED'


def compare(a,b,stats=None):
    result=compare_boundaries(a,b,stats)
    if result=='UNRESOLVED':raise BoundaryUnresolved('BOUNDARY_ORDER_UNRESOLVED')
    return {'LESS':-1,'EQUAL':0,'GREATER':1}[result]


def sign_at(polynomial,boundary,stats=None):
    a,b,c=map(F,polynomial)
    if boundary.rational is not None:
        x=boundary.rational;value=(a*x+b)*x+c
        return 'NEGATIVE' if value<0 else ('POSITIVE' if value>0 else 'ZERO')
    aa,bb,cc=boundary.polynomial
    linear=b-a*F(bb,aa);constant=c-a*F(cc,aa)
    if linear==0:return 'NEGATIVE' if constant<0 else ('POSITIVE' if constant>0 else 'ZERO')
    relation=compare_boundaries(boundary,Boundary.of(-constant/linear),stats)
    if relation=='UNRESOLVED':return relation
    sign={'LESS':-1,'EQUAL':0,'GREATER':1}[relation]*(1 if linear>0 else -1)
    return 'NEGATIVE' if sign<0 else ('POSITIVE' if sign>0 else 'ZERO')


def sign(polynomial,boundary,stats=None):
    result=sign_at(polynomial,boundary,stats)
    if result=='UNRESOLVED':raise BoundaryUnresolved('CONSTRAINT_SIGN_UNRESOLVED')
    return {'NEGATIVE':-1,'ZERO':0,'POSITIVE':1}[result]


def intersect_polynomial(interval,polynomial,stats=None):
    """Closed convex polynomial sublevel; None only after an exact empty proof."""
    lo,hi=interval;a,b,c=map(F,polynomial)
    note_bits(stats,a,b,c)
    if a<0:raise BoundaryUnresolved('NONCONVEX_EXACT_CONSTRAINT')
    if a==0:
        if b==0:return (lo,hi) if c<=0 else None
        root=Boundary.of(-c/b)
        if b>0:
            if compare(root,hi,stats)<0:hi=root
        else:
            if compare(root,lo,stats)>0:lo=root
    else:
        roots=quadratic_roots(a,b,c,stats)
        if not roots:return None
        lower,upper=roots
        if compare(lower,lo,stats)>0:lo=lower
        if compare(upper,hi,stats)<0:hi=upper
    return None if compare(lo,hi,stats)>0 else (lo,hi)


def add_interval(a,b):return a[0]+b[0],a[1]+b[1]


def mul_interval(a,b):
    products=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]
    return min(products),max(products)


def divide_interval(a,b):
    if b[0]<=0<=b[1]:raise BoundaryUnresolved('INTERVAL_DIVISION_CONTAINS_ZERO')
    return mul_interval(a,(1/b[1],1/b[0]))


def polynomial_bounds(polynomial,point,bits=64):
    a,b,c=map(F,polynomial)
    if point.rational is not None:
        x=point.rational;value=(a*x+b)*x+c;return value,value
    # Reduce before enclosing: fewer dependencies and exact zeros stay exact.
    aa,bb,cc=point.polynomial
    linear=b-a*F(bb,aa);constant=c-a*F(cc,aa)
    low,high=point.bounds(bits)
    if linear>=0:return constant+linear*low,constant+linear*high
    return constant+linear*high,constant+linear*low


def midpoint_inside(lo,hi,stats=None):
    if compare(lo,hi,stats)>=0:raise BoundaryUnresolved('NO_POSITIVE_WIDTH_INTERVAL')
    for bits in BITS:
        _,a=lo.bounds(bits);b,_=hi.bounds(bits)
        if a<b:return Boundary.of((a+b)/2)
    raise BoundaryUnresolved('INTERIOR_POINT_UNRESOLVED')
