"""Reconstruct licensed targets locally from upstream-format data files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def _text_scalar(value: np.ndarray) -> str:
    scalar = np.asarray(value).reshape(-1)[0]
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def _requested_dtype(package: np.lib.npyio.NpzFile) -> np.dtype:
    if "target_dtype" not in package:
        raise ValueError("public package lacks target_dtype metadata")
    return np.dtype(_text_scalar(package["target_dtype"]))


@lru_cache(maxsize=4)
def _load_urbanev_panel(occupancy_name: str, info_name: str) -> tuple[pd.DataFrame, pd.Series]:
    occupancy_path, info_path = Path(occupancy_name), Path(info_name)
    occupancy = pd.read_csv(occupancy_path, index_col=0)
    occupancy.columns = occupancy.columns.astype(str)
    info = pd.read_csv(info_path)
    info["TAZID"] = info["TAZID"].astype(str)
    capacity = info.groupby("TAZID", sort=False)["charge_count"].sum()
    return occupancy, capacity


def reconstruct_urbanev(package: np.lib.npyio.NpzFile, data_root: Path) -> np.ndarray:
    occupancy_path = (data_root / "occupancy.csv").resolve()
    info_path = (data_root / "inf.csv").resolve()
    if not occupancy_path.is_file() or not info_path.is_file():
        raise FileNotFoundError("UrbanEV reconstruction requires occupancy.csv and inf.csv")
    occupancy, capacity = _load_urbanev_panel(str(occupancy_path), str(info_path))
    if "zone_ids" in package:
        zone_ids = [str(item) for item in np.asarray(package["zone_ids"]).reshape(-1)]
    else:
        zone_ids = list(occupancy.columns)
    missing = [zone for zone in zone_ids if zone not in occupancy.columns or zone not in capacity.index]
    if missing:
        raise ValueError(f"UrbanEV zone identity mismatch: {missing[:5]}")
    if "target_construction" not in package:
        raise ValueError("public package lacks target_construction metadata")
    construction = _text_scalar(package["target_construction"])
    if construction == "occupancy_float64":
        values = occupancy.loc[:, zone_ids].to_numpy(np.float64)
        rates = values / capacity.loc[zone_ids].to_numpy(np.float64)[None, :]
    elif construction in {"occupancy_float32", "occupancy_float32_then_storage_dtype"}:
        values = occupancy.loc[:, zone_ids].to_numpy(np.float32)
        rates = values / capacity.loc[zone_ids].to_numpy(np.float32)[None, :]
    else:
        raise ValueError(f"unknown UrbanEV target construction: {construction}")
    indices = np.asarray(package["target_index"], dtype=np.int64)
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= rates.shape[0]:
        raise ValueError("UrbanEV target index outside the licensed panel")
    return np.asarray(rates[indices], dtype=_requested_dtype(package))


@lru_cache(maxsize=4)
def _find_paris_source(data_root_name: str) -> Path:
    data_root = Path(data_root_name)
    candidates = [data_root / "train.csv", data_root / "development_state_shard.csv"]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(data_root.glob("*.csv"))
    for path in matches:
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"date", "Station", "Available", "Charging", "Passive", "Other"} <= columns:
            return path
    raise FileNotFoundError("Paris reconstruction requires an upstream CSV with date, Station, Available, Charging, Passive, and Other")


@lru_cache(maxsize=4)
def _load_paris_matrix(source_name: str) -> pd.DataFrame:
    source = Path(source_name)
    columns = ["date", "Station", "Available", "Charging", "Passive", "Other"]
    frame = pd.read_csv(source, usecols=columns)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["date"] <= pd.Timestamp("2020-11-30 23:59:59")].copy()
    frame = frame.loc[frame["date"].dt.minute.eq(0)]
    if frame.duplicated(["date", "Station"]).any():
        raise ValueError("duplicate Paris station/timestamp observations")
    state_columns = ["Available", "Charging", "Passive", "Other"]
    total = frame[state_columns].sum(axis=1)
    if (total <= 0).any() or (frame[state_columns] < 0).any().any():
        raise ValueError("invalid Paris state counts")
    frame["target"] = (frame["Charging"] + frame["Passive"] + frame["Other"]) / total
    return frame.pivot(index="date", columns="Station", values="target")


def reconstruct_paris(package: np.lib.npyio.NpzFile, data_root: Path) -> np.ndarray:
    source = _find_paris_source(str(data_root.resolve()))
    matrix = _load_paris_matrix(str(source.resolve()))
    station_ids = [str(item) for item in np.asarray(package["station_ids"]).reshape(-1)]
    target_time = pd.to_datetime(np.asarray(package["target_time_ns"], dtype=np.int64))
    target = matrix.reindex(index=target_time, columns=station_ids).to_numpy(_requested_dtype(package))
    expected_shape = tuple(np.asarray(package["target_shape"], dtype=np.int64))
    if target.shape != expected_shape:
        raise ValueError(f"Paris reconstructed shape mismatch: {target.shape} != {expected_shape}")
    return target


def reconstruct_target(package: np.lib.npyio.NpzFile, data_root: Path) -> np.ndarray:
    if "dataset" not in package or "target_index" not in package:
        raise ValueError("public package lacks dataset/target_index metadata")
    dataset = _text_scalar(package["dataset"])
    if dataset == "urbanev":
        target = reconstruct_urbanev(package, data_root)
    elif dataset == "paris-development":
        target = reconstruct_paris(package, data_root)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    expected_shape = tuple(np.asarray(package["target_shape"], dtype=np.int64))
    if target.shape != expected_shape:
        raise ValueError(f"reconstructed target shape mismatch: {target.shape} != {expected_shape}")
    return target
