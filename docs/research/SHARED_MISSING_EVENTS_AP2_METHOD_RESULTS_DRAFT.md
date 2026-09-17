# Working draft: comparison certificates under shared missing observations

Status: method and results draft, 2026-09-17. This is not a submission-ready manuscript. Numerical evidence is indexed in the [AP2 report](../reports/audit/SHARED_MISSING_EVENTS_AP2_REPORT.md).

## Problem and scope

Overlapping forecasting windows can inherit uncertain event labels from the same missing sensor observations. Treating those labels as independently assignable may produce risk comparisons that cannot arise from any common completion of the underlying sequence. We study retrospective comparison of fixed probabilistic forecasts for the event that a length-K window contains L consecutive threshold exceedances. Our estimand is the range of the finite-panel mean Brier-loss difference over all binary completions consistent with the observed threshold indicators. This range is not a confidence interval and requires no random-missingness assumption; it remains conditional on the declared completion set and on error-free observed indicators.

## Comparison certificates

We distinguish independent labels, all valid pairwise implications, and complete trajectory consistency. For monotone events with freely completable missing bits, the ambiguous-label set contains the all-zero and all-one patterns. Pairwise relations therefore reduce to implications. Their continuous relaxation is an order polytope with integral optima. A gap from the trajectory optimum can consequently reflect an unrealizable binary pattern rather than fractional labels.

The attainable label set equals its pairwise representation exactly when it is closed under coordinatewise conjunction and disjunction. We give a self-contained proof and use it to check small components connected by shared unknown observations. Components beyond the stated enumeration budget remain unresolved unless an explicit pairwise-valid but trajectory-infeasible pattern is found. These arguments apply classical order-polytope and Boolean-relation results; we do not claim a new general optimization algorithm.

For a currently undirected comparison, we recover two legal underlying completions using weighted-automaton traceback. The sign of each resulting Brier-loss difference is computed exactly for the stored binary floating-point forecasts. Opposite signs prove that the available observations cannot support a strict uniform ordering. This conclusion does not require claiming that the recovered trajectories identify the true missing values. All exported assignments are replayed using a separate cumulative-sum event-label implementation.

## Fixed temporal replication

AP2 evaluates the twelve Beijing stations over March 2015–February 2016 and household power over July 2008–June 2009, with exclusive right endpoints. The selected AP1 models are loaded without fitting or retuning. Thresholds, event settings, features, and artificial-masking rules remain unchanged. We report seven-day origin blocks, station-year panels, and separately optimized pooled annual comparisons. Neighboring blocks may share label support and are not independent replications. Beijing K24 remains the primary setting and K72 the supplementary setting.

Across 4,098 natural weekly comparisons, joint consistency adds 17 strict orderings beyond independent bounds: five for Beijing K24 and twelve for Beijing K72. Pairwise consistency recovers sixteen of these. Thirteen weekly undominated candidate sets shrink, with ten newly certified unique best forecasts. One additional station-year ordering does not change its candidate set. Neither household-power setting nor pooled annual comparisons shows a new natural ordering. Thus the evidence supports a limited retrospective use case, not a general model-selection or deployment advantage.

The remaining 141 undirected AP2 comparisons all admit legal completions with strictly opposite risk-difference signs; the same holds for all 154 remaining AP1 comparisons. Among ambiguous panels, structural sufficiency is certified for 91 of 200 AP1 panels and 175 of 270 AP2 panels. Higher-order structural differences are demonstrated for 41 and 48 panels, respectively; the remaining 68 and 47 panels are unresolved under the diagnostic budget. Numerical endpoint equality is reported separately from these structural certificates.

## A real higher-order exception

In the Changping block beginning 4 October 2015, the K72/L3 comparison between the constant forecast and gradient boosting has pairwise bounds approximately [−0.0449372, +0.00173654], while trajectory bounds are [−0.0381158, −0.000590087]. The latter certify the constant forecast as the unique best candidate. Independent binary integer programming reproduces the endpoints. The support contains 153 consecutive missing observations among 239 records.

A minimal three-label conflict requires event labels 0, 1, and 0 at relative origins 80, 145, and 150. Every pair is feasible, but the two zero-event windows jointly cover every potential run start in the middle window. This yields a transparent higher-order incompatibility. Adding only that single covering inequality to the pairwise program still leaves a positive upper endpoint, approximately 0.00148277; the local explanation does not account for the entire optimization gain. Both the conflict extraction and this single-cut diagnostic are post-hoc analyses of a fixed replication result.

## Reliability and limitations

For 234 artificial-masking comparisons, independent bounds produce 165 strict orderings and pairwise/joint bounds produce 177, with no wrong directions. LOCF label imputation and identified-label-only point estimates each direct all 234 comparisons, with three and seventeen wrong directions, respectively. All three interval methods contain the known complete differences. Natural data have no complete reference truth, so no natural-data wrong-direction rate is asserted.

The scientific contribution must rest on the shared-label evaluation problem, inspectable certificates, and demonstrated applicability limits. Missing-label evaluation, order-polytope integrality, and cost-regular path optimization are established ideas. The present evidence does not establish a general runtime advantage, widespread necessity of high-order optimization, causal missingness mechanisms, or sufficient novelty for a specific journal tier.
