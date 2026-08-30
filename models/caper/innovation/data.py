from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OccupancyData:
    counts: np.ndarray
    capacity: np.ndarray
    rate: np.ndarray
    time: pd.DatetimeIndex
    zone_ids: tuple[str, ...]


@dataclass(frozen=True)
class FoldBoundaries:
    fold_end: int
    train_end: int
    valid_start: int
    valid_end: int
    test_start: int
    test_end: int


def load_occupancy(data_dir: Path) -> OccupancyData:
    """Load occupied-pile counts and recover the zone-level exposure denominator."""
    occupancy = pd.read_csv(data_dir / "occupancy.csv", index_col=0)
    occupancy.columns = occupancy.columns.astype(str)
    info = pd.read_csv(data_dir / "inf.csv")
    info["TAZID"] = info["TAZID"].astype(str)
    capacity_by_zone = info.groupby("TAZID", sort=False)["charge_count"].sum()
    missing = [zone for zone in occupancy.columns if zone not in capacity_by_zone.index]
    if missing:
        raise ValueError(f"Missing charge_count for zones: {missing[:5]}")
    capacity = capacity_by_zone.loc[occupancy.columns].to_numpy(dtype=np.float64)
    counts = occupancy.to_numpy(dtype=np.float64)
    if np.any(capacity <= 0):
        raise ValueError("All zone capacities must be positive")
    rate = counts / capacity[np.newaxis, :]
    if not np.isfinite(rate).all():
        raise ValueError("Occupancy rate contains non-finite values")
    if rate.min() < 0 or rate.max() > 1 + 1e-8:
        raise ValueError(f"Occupancy rate outside [0, 1]: [{rate.min()}, {rate.max()}]")
    return OccupancyData(
        counts=counts,
        capacity=capacity,
        rate=rate,
        time=pd.DatetimeIndex(pd.to_datetime(occupancy.index)),
        zone_ids=tuple(occupancy.columns),
    )


def fold_boundaries(time: pd.DatetimeIndex, fold: int) -> FoldBoundaries:
    months = list(pd.Index(time.month).unique())
    if fold < 1 or fold > len(months):
        raise ValueError(f"fold must be within 1..{len(months)}")
    fold_end = int(pd.Index(time.month).isin(months[:fold]).sum())
    train_end = int(fold_end * 0.8)
    valid_end = int(train_end + fold_end * 0.1)
    return FoldBoundaries(
        fold_end=fold_end,
        train_end=train_end,
        valid_start=train_end,
        valid_end=valid_end,
        test_start=valid_end,
        test_end=fold_end,
    )

