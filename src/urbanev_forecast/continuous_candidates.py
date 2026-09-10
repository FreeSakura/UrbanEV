"""Linear, dependency-complete reduction of a fixed finite candidate set."""
from dataclasses import dataclass
from fractions import Fraction as F
from .continuous_boundaries import BITS,Boundary,BoundaryUnresolved,compare


def encode_fraction(value):
    value=F(value)
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}


def encode_interval(interval):
    return {'lower':encode_fraction(interval[0]),'upper':encode_fraction(interval[1])}


@dataclass(frozen=True)
class Candidate:
    point:Boundary
    objective:tuple
    segment:int


class CandidateReductionBlocked(BoundaryUnresolved):
    def __init__(self,reason,diagnostics):
        super().__init__(reason);self.diagnostics=diagnostics


def reduce_candidates(candidates,refine_bounds,tolerance=F(1,10**12),stats=None):
    """Refine A union P without reoptimizing, adding or removing candidate points.

    refine_bounds(indices,bits) returns {bounds: {index: (L,U)},
    refined_segment_count: int}. Inputs are frozen candidate records; only local
    objective enclosures change, through intersection with newly supplied bounds.
    """
    candidates=tuple(candidates);bounds=[tuple(map(F,c.objective)) for c in candidates]
    diagnostics={'rounds':[],'classification_visits':0,'minimum_bound_visits':0,
                 'candidate_count':len(candidates),'unresolved_indices':[],
                 'candidate_positions_changed':False}
    if stats is not None:stats.setdefault('max_candidate_objective_refinement_bits',64)
    def block(reason):raise CandidateReductionBlocked(reason,diagnostics)
    if not candidates:block('EMPTY_FINITE_CANDIDATE_SET')
    if any(low>high for low,high in bounds):block('INVALID_CANDIDATE_OBJECTIVE_ENCLOSURE')
    for round_index,bits in enumerate(BITS):
        min_lower,min_upper=bounds[0]
        for low,high in bounds:
            diagnostics['minimum_bound_visits']+=1
            if low<min_lower:min_lower=low
            if high<min_upper:min_upper=high
        included=[];excluded=0;ambiguous=[];refine=[];possible_count=0
        for i,(low,high) in enumerate(bounds):
            diagnostics['classification_visits']+=1
            if high<=min_lower+tolerance:
                included.append(i);unresolved=False
            elif low>min_upper+tolerance:
                excluded+=1;unresolved=False
            else:
                ambiguous.append(i);unresolved=True
            possible=low<=min_upper
            if possible:possible_count+=1
            if unresolved or possible:refine.append(i)
        entry={'precision_bits':bits,'candidate_count':len(candidates),
               'min_lower':encode_fraction(min_lower),'min_upper':encode_fraction(min_upper),
               'in_count':len(included),'out_count':excluded,'unresolved_count':len(ambiguous),
               'possible_minimizer_count':possible_count,'refined_candidate_count':0,
               'refined_segment_count':0}
        diagnostics['rounds'].append(entry)
        diagnostics['unresolved_indices']=list(ambiguous)
        diagnostics['finite_candidate_minimum_bounds']=encode_interval((min_lower,min_upper))
        if not ambiguous:
            if not included:block('EMPTY_DEFINITE_TIE_SET')
            chosen=included[0]
            for i in included[1:]:
                try:
                    if compare(candidates[i].point,candidates[chosen].point,stats)<0:chosen=i
                except BoundaryUnresolved as error:
                    diagnostics['alpha_order_unresolved_pair']=[chosen,i]
                    block(str(error))
            diagnostics['selected_index']=chosen
            diagnostics['selected_objective_bounds']=encode_interval(bounds[chosen])
            diagnostics['selected_exact_point']=candidates[chosen].point.encode()
            return {'selected_index':chosen,'bounds':bounds,'diagnostics':diagnostics}
        if bits==512:block('FINITE_CANDIDATE_TIE_COMPARISON_UNRESOLVED')
        next_bits=BITS[round_index+1]
        entry['next_precision_bits']=next_bits
        entry['requested_refined_candidate_count']=len(refine)
        try:update=refine_bounds(tuple(refine),next_bits)
        except BoundaryUnresolved as error:block(str(error))
        if not isinstance(update,dict) or not isinstance(update.get('bounds'),dict):
            block('INVALID_CANDIDATE_REFINEMENT_INTERFACE')
        supplied=update['bounds']
        if len(supplied)!=len(refine) or any(i not in supplied for i in refine):
            block('CANDIDATE_REFINEMENT_INCOMPLETE')
        entry['refined_candidate_count']=len(refine)
        entry['refined_segment_count']=int(update['refined_segment_count'])
        if stats is not None:
            stats['max_candidate_objective_refinement_bits']=max(stats.get('max_candidate_objective_refinement_bits',64),next_bits)
            stats['max_objective_refinement_bits']=max(stats.get('max_objective_refinement_bits',0),next_bits)
        for i in refine:
            new_lower,new_upper=map(F,supplied[i])
            if new_lower>new_upper:block('INVALID_NEW_CANDIDATE_ENCLOSURE')
            low=max(bounds[i][0],new_lower);high=min(bounds[i][1],new_upper)
            if low>high:
                diagnostics['contradiction_candidate_index']=i
                diagnostics['old_enclosure']=encode_interval(bounds[i])
                diagnostics['new_enclosure']=encode_interval((new_lower,new_upper))
                block('CANDIDATE_ENCLOSURE_CONTRADICTION')
            bounds[i]=(low,high)
    block('FINITE_CANDIDATE_TIE_COMPARISON_UNRESOLVED')
