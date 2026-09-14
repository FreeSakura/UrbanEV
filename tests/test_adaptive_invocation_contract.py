"""CPU-only interface witnesses, not a trained gate or a latency benchmark."""
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
import math

import pytest


@dataclass(frozen=True)
class Request:
    origin: int
    horizon: int
    cheap_state: int
    regions: int = 275


def forecast(value, horizon):
    return tuple(tuple(float(value) for _ in range(275)) for _ in range(horizon))


def check_forecast(q,request):
    if request.regions!=275 or len(q)!=request.horizon or any(len(row)!=275 for row in q):
        raise ValueError('A request must return its complete H by 275 region output')
    if not all(math.isfinite(v) for row in q for v in row):raise ValueError('Nonfinite synthetic output')


def dispatch(requests,cheap_predictions,gate,build_expensive,forward_expensive,*,batch_size=1):
    """A toy harness with causal gate inputs and lazy full-request execution.

    Callback counters witness only this interface; they are not a security sandbox
    and say nothing about real model latency or inaccessible global variables.
    """
    if len(requests)!=len(cheap_predictions) or batch_size<1:raise ValueError('Invalid synthetic batch')
    output=list(cheap_predictions);decisions=[];built=0;batches=0;accepted=0
    for start in range(0,len(requests),batch_size):
        selected=[];inputs=[]
        for index in range(start,min(start+batch_size,len(requests))):
            request=requests[index];cheap=cheap_predictions[index];check_forecast(cheap,request)
            view=MappingProxyType({'origin':request.origin,'horizon':request.horizon,
                                   'cheap_state':request.cheap_state,'cheap_prediction':cheap})
            decision=gate(view)
            if type(decision) is not bool:raise TypeError('Gate must decide one complete request, not a region mask')
            decisions.append(decision)
            if decision:
                inputs.append(build_expensive(request));built+=1;accepted+=1;selected.append(index)
        if selected:
            predictions=forward_expensive(inputs);batches+=1
            if len(predictions)!=len(selected):raise ValueError('Wrong number of full request outputs')
            for index,pred in zip(selected,predictions):check_forecast(pred,requests[index]);output[index]=pred
    return {'outputs':output,'decisions':decisions,'gate_requests':len(requests),
            'accepted_origin_horizon_requests':accepted,'expensive_input_builds':built,'forward_batches':batches}


def test_rejected_branch_does_not_build_inputs_or_call_expensive_stub():
    req=Request(169,3,0);cheap=forecast(.2,3)
    def forbidden(*args):raise AssertionError('Expensive work executed before acceptance')
    result=dispatch([req],[cheap],lambda view:False,forbidden,forbidden)
    assert result['outputs'][0] is cheap
    assert result['expensive_input_builds']==result['forward_batches']==0


def test_unavailable_expensive_output_and_target_do_not_enter_gate_view():
    reqs=[Request(169,3,0),Request(170,3,1)];cheap=[forecast(.2,3)]*2
    views=[]
    def gate(view):
        assert set(view)=={'origin','horizon','cheap_state','cheap_prediction'}
        assert 'target' not in view and 'expensive_prediction' not in view and 'disagreement' not in view
        views.append(view['cheap_state']);return view['cheap_state']==1
    decisions=[]
    for hidden_target,hidden_expensive_value in [(0,.1),(1,.9)]:
        # Target is deliberately unavailable to dispatch and gate.
        result=dispatch(reqs,cheap,gate,lambda r:r,lambda xs:[forecast(hidden_expensive_value,r.horizon) for r in xs])
        decisions.append(result['decisions'])
    assert decisions==[[False,True],[False,True]] and views==[0,1,0,1]


def test_full_region_granularity_rejects_masks_and_counts_one_full_call():
    req=Request(169,6,1);cheap=forecast(.2,6)
    with pytest.raises(TypeError,match='region mask'):
        dispatch([req],[cheap],lambda _: [True]+[False]*274,lambda r:r,lambda xs:[])
    result=dispatch([req],[cheap],lambda _:True,lambda r:r,lambda xs:[forecast(.3,r.horizon) for r in xs])
    assert result['accepted_origin_horizon_requests']==result['forward_batches']==1
    assert len(result['outputs'][0][-1])==275
    with pytest.raises(ValueError,match='complete'):
        dispatch([Request(169,6,1,regions=1)],[cheap],lambda _:True,lambda r:r,lambda xs:[])


def test_prespecified_budgets_gate_overhead_and_batch_tail_counters():
    for budget in (0,10,25,50,75,100):
        requests=[Request(169+i,3,int(i<budget)) for i in range(100)]
        result=dispatch(requests,[forecast(.2,3)]*100,lambda v:v['cheap_state']==1,
                        lambda r:r,lambda xs:[forecast(.3,r.horizon) for r in xs])
        assert result['accepted_origin_horizon_requests']==result['forward_batches']==budget
        # Abstract additive SERIAL units: cheap=2, gate=1, incremental branch=10.
        cost=result['gate_requests']*3+result['accepted_origin_horizon_requests']*10
        assert cost==300+10*budget
        assert 100*2 < 300  # zero activation still pays the gate overhead
    req=[Request(169+i,3,1) for i in range(19)]
    grouped=dispatch(req,[forecast(.2,3)]*19,lambda _:True,lambda r:r,
                     lambda xs:[forecast(.3,r.horizon) for r in xs],batch_size=16)
    assert grouped['gate_requests']==grouped['accepted_origin_horizon_requests']==19
    assert grouped['expensive_input_builds']==19 and grouped['forward_batches']==2
    # These batch counters are not converted into milliseconds or speedup claims.


def test_complementary_experts_can_have_no_predictable_pre_call_gain():
    # Cheap information z is constant; Y is a fair Bernoulli. q0=0, q1=1.
    r0=r1=Fraction(1,2);oracle=Fraction(0)
    assert oracle<r0 and r0-r1==0
    for p in (Fraction(0),Fraction(1,10),Fraction(1,4),Fraction(1,2),Fraction(3,4),Fraction(1)):
        risk=(1-p)*r0+p*r1
        assert risk==Fraction(1,2)
    # The expression is constant for every p, not evidence about UrbanEV features.


def test_always_expensive_is_not_an_accuracy_upper_endpoint():
    # Here z is observed before prediction and Y=z; each state has probability 1/2.
    # q0=0,q1=1; a legitimate g(z)=z is exact with only half the calls.
    cheap=expensive=routed=Fraction(0);calls=Fraction(0)
    for z in (0,1):
        p=Fraction(1,2);y=z;g=z
        cheap+=p*(y-0)**2;expensive+=p*(y-1)**2
        routed+=p*(y-g)**2;calls+=p*g
    assert cheap==expensive==Fraction(1,2)
    assert routed==0 and calls==Fraction(1,2)
