from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbanev_audit.targets import reconstruct_target


def test_urbanev_target_reconstruction(tmp_path: Path):
    pd.DataFrame({"1": [1.0, 2.0], "2": [3.0, 1.0]}, index=["2020-01-01", "2020-01-02"]).to_csv(tmp_path / "occupancy.csv")
    pd.DataFrame({"TAZID": ["1", "2"], "charge_count": [2.0, 4.0]}).to_csv(tmp_path / "inf.csv", index=False)
    package_path = tmp_path / "package.npz"
    np.savez(
        package_path,
        dataset=np.asarray("urbanev"),
        target_index=np.asarray([1]),
        zone_ids=np.asarray(["1", "2"]),
        target_dtype=np.asarray("float64"),
        target_shape=np.asarray([1, 2]),
        target_construction=np.asarray("occupancy_float32_then_storage_dtype"),
    )
    with np.load(package_path, allow_pickle=False) as package:
        target = reconstruct_target(package, tmp_path)
    np.testing.assert_allclose(target, [[1.0, 0.25]])


def test_paris_target_reconstruction(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "date": ["2020-08-01 00:00:00", "2020-08-01 00:00:00"],
            "Station": ["A", "B"],
            "Available": [2, 1],
            "Charging": [1, 1],
            "Passive": [1, 0],
            "Other": [0, 0],
        }
    )
    frame.to_csv(tmp_path / "development_state_shard.csv", index=False)
    package_path = tmp_path / "package.npz"
    timestamp = pd.Timestamp("2020-08-01 00:00:00").value
    np.savez(
        package_path,
        dataset=np.asarray("paris-development"),
        target_index=np.asarray([0]),
        target_time_ns=np.asarray([timestamp]),
        station_ids=np.asarray(["A", "B"]),
        target_dtype=np.asarray("float32"),
        target_shape=np.asarray([1, 2]),
    )
    with np.load(package_path, allow_pickle=False) as package:
        target = reconstruct_target(package, tmp_path)
    np.testing.assert_allclose(target, [[0.5, 0.5]])


def test_paris_full_source_requires_explicit_authorization(tmp_path: Path):
    pd.DataFrame(
        {
            "date": ["2020-08-01 00:00:00"],
            "Station": ["A"],
            "Available": [1],
            "Charging": [1],
            "Passive": [0],
            "Other": [0],
        }
    ).to_csv(tmp_path / "train.csv", index=False)
    package_path = tmp_path / "package.npz"
    np.savez(
        package_path,
        dataset=np.asarray("paris-development"),
        target_index=np.asarray([0]),
        target_time_ns=np.asarray([pd.Timestamp("2020-08-01 00:00:00").value]),
        station_ids=np.asarray(["A"]),
        target_dtype=np.asarray("float32"),
        target_shape=np.asarray([1, 1]),
    )
    with np.load(package_path, allow_pickle=False) as package:
        with pytest.raises(PermissionError, match="allow-full-source"):
            reconstruct_target(package, tmp_path)
    with np.load(package_path, allow_pickle=False) as package:
        target = reconstruct_target(package, tmp_path, allow_full_source=True)
    np.testing.assert_allclose(target, [[0.5]])


def test_paris_development_shard_rejects_late_timestamp(tmp_path: Path):
    pd.DataFrame(
        {
            "date": ["2020-12-01 00:00:00"],
            "Station": ["A"],
            "Available": [1],
            "Charging": [1],
            "Passive": [0],
            "Other": [0],
        }
    ).to_csv(tmp_path / "development_state_shard.csv", index=False)
    package_path = tmp_path / "package.npz"
    np.savez(
        package_path,
        dataset=np.asarray("paris-development"),
        target_index=np.asarray([0]),
        target_time_ns=np.asarray([pd.Timestamp("2020-12-01 00:00:00").value]),
        station_ids=np.asarray(["A"]),
        target_dtype=np.asarray("float32"),
        target_shape=np.asarray([1, 1]),
    )
    with np.load(package_path, allow_pickle=False) as package:
        with pytest.raises(ValueError, match="formal/protected"):
            reconstruct_target(package, tmp_path)
