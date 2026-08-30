"""Deterministic metric primitives shared by public replay commands."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def sha256_array(values: np.ndarray) -> str:
    """Hash a canonical little-endian, C-contiguous array representation."""
    array = np.asarray(values)
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def rmse(prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} != {reference.shape}")
    valid = np.isfinite(prediction) & np.isfinite(reference)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != prediction.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {prediction.shape}")
        valid &= mask
    if not np.any(valid):
        raise ValueError("no finite, selected observations")
    return float(np.sqrt(np.mean(np.square(prediction[valid] - reference[valid]))))


def relative_gain(reference_rmse: float, candidate_rmse: float) -> float:
    if reference_rmse <= 0:
        raise ValueError("reference RMSE must be positive")
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse


def equal_cell_macro(values: Iterable[float]) -> float:
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("macro input must be non-empty and finite")
    return float(values.mean())
