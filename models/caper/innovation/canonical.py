from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CanonicalBoundaries:
    """Time boundaries matching the Transformer loader's integer rounding."""

    fold_end: int
    train_end: int
    valid_start: int
    valid_end: int
    test_start: int
    test_end: int


def canonical_boundaries(time: pd.DatetimeIndex, fold: int) -> CanonicalBoundaries:
    """Return expanding-month boundaries without reusing model-specific loaders.

    The official Transformer family sets ``num_train=floor(0.8*N)`` and
    ``num_test=floor(0.1*N)``, then starts validation at ``num_train`` and test
    at ``N-num_test``.  This differs by up to two hours from the conventional
    scripts' ``floor(0.8*N)+floor(0.1*N)`` test boundary.
    """

    periods = time.to_period("M")
    months = list(pd.Index(periods).unique())
    if fold < 1 or fold > len(months):
        raise ValueError(f"fold must be within 1..{len(months)}")
    fold_end = int(pd.Index(periods).isin(months[:fold]).sum())
    train_end = int(0.8 * fold_end)
    test_start = fold_end - int(0.1 * fold_end)
    if not 0 < train_end < test_start < fold_end:
        raise ValueError("canonical split is empty or unordered")
    return CanonicalBoundaries(
        fold_end=fold_end,
        train_end=train_end,
        valid_start=train_end,
        valid_end=test_start,
        test_start=test_start,
        test_end=fold_end,
    )


def canonical_target_indices(
    bounds: CanonicalBoundaries,
    horizon: int,
    split: str,
    common_history_budget: int = 168,
) -> np.ndarray:
    """Return endpoint targets for a leakage-free direct-horizon forecast.

    For target ``t`` and horizon ``h``, the forecast origin is ``t-h``.  The
    earliest target that admits a history budget ``L`` is therefore
    ``L+h-1``.  Validation and test indices reproduce the endpoint evaluated
    by the official Transformer loader while ensuring that each forecast
    origin precedes its split boundary.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if common_history_budget <= 0:
        raise ValueError("common_history_budget must be positive")
    if split == "train":
        start, stop = common_history_budget + horizon - 1, bounds.train_end
    elif split == "valid":
        start, stop = bounds.valid_start + horizon - 1, bounds.valid_end
    elif split == "test":
        start, stop = bounds.test_start + horizon - 1, bounds.test_end
    else:
        raise ValueError(f"unknown split: {split}")
    indices = np.arange(start, stop, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(
            f"empty canonical split: split={split}, horizon={horizon}, "
            f"history_budget={common_history_budget}"
        )
    return indices


def legacy_common_target_indices(
    bounds: CanonicalBoundaries,
    horizon: int,
    split: str,
    common_history_budget: int = 168,
    sequence_guard: int = 12,
) -> np.ndarray:
    """Return the historical statistical-script window for sensitivity only.

    This deliberately preserves the extra ``12+h`` guard used by the
    conventional scripts.  It must not be presented as Transformer-native.
    """

    if split == "train":
        start, stop = common_history_budget + horizon, bounds.train_end
    elif split == "valid":
        start, stop = bounds.valid_start + sequence_guard + horizon, bounds.valid_end
    elif split == "test":
        conventional_test_start = int(0.8 * bounds.fold_end) + int(0.1 * bounds.fold_end)
        start, stop = conventional_test_start + sequence_guard + horizon, bounds.test_end
    else:
        raise ValueError(f"unknown split: {split}")
    indices = np.arange(start, stop, dtype=np.int64)
    if indices.size == 0:
        raise ValueError(f"empty legacy split: split={split}, horizon={horizon}")
    return indices


def history_indices(target_index: int, horizon: int, history_length: int) -> np.ndarray:
    """Return the exact observed indices used to forecast one target."""

    if horizon <= 0 or history_length <= 0:
        raise ValueError("horizon and history_length must be positive")
    forecast_origin = int(target_index) - horizon
    first = forecast_origin - history_length + 1
    if first < 0:
        raise ValueError("target does not have enough observable history")
    result = np.arange(first, forecast_origin + 1, dtype=np.int64)
    if len(result) != history_length or int(result.max()) > forecast_origin:
        raise AssertionError("history construction leaked past the forecast origin")
    return result


def calendar_features(time: pd.DatetimeIndex) -> np.ndarray:
    """Encode known-in-advance hour-of-day and day-of-week covariates."""

    hour = time.hour.to_numpy(dtype=np.float32)
    weekday = time.dayofweek.to_numpy(dtype=np.float32)
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
        ],
        axis=-1,
    ).astype(np.float32)


def assert_target_identity(*target_sets: np.ndarray) -> None:
    """Fail loudly if models would be evaluated on different target IDs."""

    if len(target_sets) < 2:
        return
    reference = np.asarray(target_sets[0])
    for candidate in target_sets[1:]:
        if not np.array_equal(reference, np.asarray(candidate)):
            raise AssertionError("candidate models do not share identical target indices")
