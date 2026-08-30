"""Parameter-free validation probe for state-selective phase relaxation.

All beta and retention coefficients are fitted on each formal training split.
The probe then evaluates real UrbanEV validation labels for the pre-registered
screen cells only.  No neural parameters and no formal test arrays are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from innovation.canonical import canonical_boundaries, canonical_target_indices
from innovation.multires_state import load_five_minute_states
from innovation.phase_mean_reversion_experiment import fit_phase_beta
from innovation.physical_constraints import load_physical_data
from repro.metrics import audited_metrics


def fit_retention(
    values: np.ndarray,
    targets: np.ndarray,
    horizon: int,
    lags: tuple[int, int],
) -> tuple[float, float]:
    beta, _ = fit_phase_beta(values, targets, lags)
    phase = (1.0 - beta) * values[targets - lags[0]] + beta * values[targets - lags[1]]
    origin = values[targets - horizon]
    response = (values[targets] - phase).reshape(-1)
    departure = (origin - phase).reshape(-1)
    denominator = float(departure @ departure)
    raw = float(departure @ response / denominator) if denominator > 0 else 0.0
    return beta, float(np.clip(raw, 0.0, 1.0))


def component_prediction(
    values: np.ndarray,
    targets: np.ndarray,
    horizon: int,
    lags: tuple[int, int],
    beta: float,
    retention: float,
) -> np.ndarray:
    phase = (1.0 - beta) * values[targets - lags[0]] + beta * values[targets - lags[1]]
    origin = values[targets - horizon]
    return retention * origin + (1.0 - retention) * phase


def run(root: Path, source_root: Path) -> Path:
    data = load_physical_data(root / "audited" / "data")
    fine = load_five_minute_states(root, source_root, data)
    state = fine.state_rate[::12].astype(np.float64)
    active, unavailable = state[..., 1], state[..., 2]
    total = active + unavailable
    rows: list[dict] = []
    coefficients: list[dict] = []
    for fold in (1, 3, 6):
        bounds = canonical_boundaries(data.time, fold)
        for horizon in (3, 12):
            train = canonical_target_indices(bounds, horizon, "train", 168)
            valid = canonical_target_indices(bounds, horizon, "valid", 168)
            fitted: dict[str, tuple[float, float]] = {}
            for name, values in (
                ("active", active), ("unavailable", unavailable), ("total", total)
            ):
                fitted[name] = fit_retention(values, train, horizon, (24, 168))
            beta_a, retain_a = fitted["active"]
            beta_u, retain_u = fitted["unavailable"]
            beta_o, retain_o = fitted["total"]
            beta_a_off, retain_a_off = fit_retention(
                active, train, horizon, (23, 167)
            )
            beta_u_off, retain_u_off = fit_retention(
                unavailable, train, horizon, (23, 167)
            )
            coefficients.append(
                {
                    "fold": fold,
                    "horizon": horizon,
                    "beta_active": beta_a,
                    "beta_unavailable": beta_u,
                    "beta_total": beta_o,
                    "retention_active": retain_a,
                    "retention_unavailable": retain_u,
                    "retention_total": retain_o,
                    "ordered_active_below_unavailable": retain_a < retain_u,
                }
            )
            correct = component_prediction(
                active, valid, horizon, (24, 168), beta_a, retain_a
            ) + component_prediction(
                unavailable, valid, horizon, (24, 168), beta_u, retain_u
            )
            reversed_order = component_prediction(
                active, valid, horizon, (24, 168), beta_a, retain_u
            ) + component_prediction(
                unavailable, valid, horizon, (24, 168), beta_u, retain_a
            )
            tied = 0.5 * (retain_a + retain_u)
            tied_state = component_prediction(
                active, valid, horizon, (24, 168), beta_a, tied
            ) + component_prediction(
                unavailable, valid, horizon, (24, 168), beta_u, tied
            )
            phase_only = component_prediction(
                active, valid, horizon, (24, 168), beta_a, 0.0
            ) + component_prediction(
                unavailable, valid, horizon, (24, 168), beta_u, 0.0
            )
            total_relaxation = component_prediction(
                total, valid, horizon, (24, 168), beta_o, retain_o
            )
            offphase = component_prediction(
                active, valid, horizon, (23, 167), beta_a_off, retain_a_off
            ) + component_prediction(
                unavailable, valid, horizon, (23, 167), beta_u_off, retain_u_off
            )
            origin = total[valid - horizon]
            target = total[valid]
            variants = {
                "ordered_state_relaxation": correct,
                "reversed_retention_order": reversed_order,
                "tied_state_retention": tied_state,
                "state_phase_only": phase_only,
                "total_phase_relaxation": total_relaxation,
                "offphase_state_relaxation": offphase,
                "origin_persistence": origin,
            }
            for variant, prediction in variants.items():
                prediction = np.clip(prediction, 0.0, 1.0)
                rows.append(
                    {
                        "fold": fold,
                        "horizon": horizon,
                        "variant": variant,
                        "samples": len(valid),
                        **audited_metrics(prediction, target),
                    }
                )
    frame = pd.DataFrame(rows)
    output = root / "innovation" / "state_phase_relaxation_results"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "STATE_PHASE_RELAXATION_RUNS.csv", index=False)
    pd.DataFrame(coefficients).to_csv(
        output / "STATE_PHASE_RELAXATION_COEFFICIENTS.csv", index=False
    )
    summary = (
        frame.groupby("variant", as_index=False)
        .agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), wins=("RMSE", "size"))
        .sort_values("RMSE")
    )
    summary.to_csv(output / "STATE_PHASE_RELAXATION_SUMMARY.csv", index=False)
    score = dict(zip(summary.variant, summary.RMSE))
    decision = {
        "data_source": "real_UrbanEV_release_from_user_workspace",
        "selection_split": "validation_only",
        "test_arrays_loaded": False,
        "cells": 6,
        "ordered_RMSE": float(score["ordered_state_relaxation"]),
        "total_only_RMSE": float(score["total_phase_relaxation"]),
        "reversed_RMSE": float(score["reversed_retention_order"]),
        "offphase_RMSE": float(score["offphase_state_relaxation"]),
        "mechanism_controls_won": bool(
            score["ordered_state_relaxation"] < score["reversed_retention_order"]
            and score["ordered_state_relaxation"] < score["tied_state_retention"]
            and score["ordered_state_relaxation"] < score["offphase_state_relaxation"]
        ),
        "recommended_neural_followup": bool(
            score["ordered_state_relaxation"] < score["total_phase_relaxation"]
            and score["ordered_state_relaxation"] < score["reversed_retention_order"]
        ),
        "provenance": fine.provenance,
    }
    (output / "STATE_PHASE_RELAXATION_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps({k: v for k, v in decision.items() if k != "provenance"}, ensure_ascii=False, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-root", type=Path, default=Path("data/UrbanEV")
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.source_root.resolve())


if __name__ == "__main__":
    main()
