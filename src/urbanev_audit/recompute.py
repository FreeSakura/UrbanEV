"""Recompute stored-prediction metrics after locally reconstructing licensed targets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from .artifacts import validate_npz
from .metrics import relative_gain, rmse, sha256_array
from .targets import reconstruct_target


def _decode_scalar(value: np.ndarray) -> str:
    scalar = np.asarray(value).reshape(-1)[0]
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def recompute_package(
    package_path: Path,
    target_root: Path,
    development_shard: Path | None = None,
    allow_full_source: bool = False,
) -> dict[str, object]:
    validate_npz(package_path)
    with np.load(package_path, allow_pickle=False) as package:
        if "target_id" not in package or "target_sha256" not in package:
            raise ValueError(f"missing target identity metadata: {package_path}")
        target_id = _decode_scalar(package["target_id"])
        expected_hash = _decode_scalar(package["target_sha256"])
        target = reconstruct_target(package, target_root, development_shard, allow_full_source)
        actual_hash = sha256_array(target)
        if actual_hash != expected_hash:
            raise ValueError(f"target hash mismatch for {target_id}")
        mask = package["mask"] if "mask" in package else None
        scores: dict[str, float] = {}
        for key in ("prediction", "predictions", "raw_prediction", "clipped_prediction", "caper", "timexer", "global_fixed", "horizon_fixed", "hard", "soft", "mlp"):
            if key in package and np.asarray(package[key]).shape == target.shape:
                scores[key] = rmse(package[key], target, mask)
        gains: dict[str, float] = {}
        if "timexer" in scores:
            for key, score in scores.items():
                if key != "timexer":
                    gains[f"{key}_vs_timexer_pct"] = relative_gain(scores["timexer"], score)
        source_label = _decode_scalar(package["source_label"]) if "source_label" in package else package_path.name
        return {"package": package_path.name, "source_label": source_label, "target_id": target_id, "rmse": scores, "gains_pct": gains}


def _macro(results: list[dict[str, object]], needle: str, key: str) -> float | None:
    values = [item["rmse"][key] for item in results if needle in item["source_label"] and key in item["rmse"]]
    return float(np.mean(values)) if values else None


def _paris_teacher(
    paths: list[Path],
    data_root: Path,
    development_shard: Path | None,
    allow_full_source: bool,
) -> dict[str, float] | None:
    groups: dict[tuple[str, int], dict[str, Path]] = {}
    pattern = re.compile(r"_(2020-(?:08|09|10|11))_h(3|6|9|12)_")
    for path in paths:
        with np.load(path, allow_pickle=False) as package:
            if _decode_scalar(package["dataset"]) != "paris-development":
                continue
            label = _decode_scalar(package["source_label"])
            match = pattern.search(label)
            if not match:
                continue
            model = "caper" if "Paris_CAPER_phase_only" in label else "timexer" if "TimeXer_local" in label else ""
            if model:
                groups.setdefault((match.group(1), int(match.group(2))), {})[model] = path
    if len(groups) != 16 or any(set(pair) != {"caper", "timexer"} for pair in groups.values()):
        return None
    repo_root = Path(__file__).resolve().parents[2]
    weights_path = repo_root / "artifacts/summaries/paris_development/PARIS_TEACHER_WEIGHTS.csv"
    with weights_path.open("r", encoding="utf-8-sig", newline="") as handle:
        weights = {row["fold"]: float(row["alpha_TimeXer"]) for row in csv.DictReader(handle)}
    scores = {"caper": [], "timexer": [], "teacher": []}
    for (fold, _horizon), pair in sorted(groups.items()):
        with np.load(pair["caper"], allow_pickle=False) as caper, np.load(pair["timexer"], allow_pickle=False) as timexer:
            target = reconstruct_target(caper, data_root, development_shard, allow_full_source)
            if _decode_scalar(caper["target_sha256"]) != _decode_scalar(timexer["target_sha256"]):
                raise ValueError(f"Paris pair target identity mismatch: {fold}/{_horizon}")
            if not np.array_equal(caper["mask"], timexer["mask"]):
                raise ValueError(f"Paris pair mask mismatch: {fold}/{_horizon}")
            mask = caper["mask"]
            caper_prediction = caper["clipped_prediction"].astype(np.float64)
            timexer_prediction = timexer["clipped_prediction"].astype(np.float64)
            teacher = np.clip(caper_prediction + weights[fold] * (timexer_prediction - caper_prediction), 0.0, 1.0)
            scores["caper"].append(rmse(caper_prediction, target, mask))
            scores["timexer"].append(rmse(timexer_prediction, target, mask))
            scores["teacher"].append(rmse(teacher, target, mask))
    macro = {key: float(np.mean(value)) for key, value in scores.items()}
    best_single = min(macro["caper"], macro["timexer"])
    macro["teacher_gain_vs_best_single_pct"] = relative_gain(best_single, macro["teacher"])
    return macro


def headline_summary(
    paths: list[Path],
    results: list[dict[str, object]],
    data_root: Path,
    development_shard: Path | None = None,
    allow_full_source: bool = False,
) -> dict[str, object]:
    router = {key: _macro(results, "router_v1_", key) for key in ("caper", "timexer", "global_fixed", "horizon_fixed", "hard", "soft", "mlp")}
    router = {key: value for key, value in router.items() if value is not None}
    if "timexer" in router and "global_fixed" in router:
        router["global_fixed_gain_vs_timexer_pct"] = relative_gain(router["timexer"], router["global_fixed"])
    chronos = _macro(results, "full__f", "clipped_prediction")
    kd = {
        branch: _macro(results, needle, "prediction")
        for branch, needle in (("GT_only", "m7c_v1_GT_only"), ("aligned_KD", "m7c_v1_aligned_KD"), ("shuffled_teacher", "m7c_v1_shuffled_teacher"))
    }
    kd = {key: value for key, value in kd.items() if value is not None}
    if "GT_only" in kd and "aligned_KD" in kd:
        kd["aligned_gain_vs_GT_only_pct"] = relative_gain(kd["GT_only"], kd["aligned_KD"])
    if "shuffled_teacher" in kd and "aligned_KD" in kd:
        kd["aligned_gain_vs_shuffled_pct"] = relative_gain(kd["shuffled_teacher"], kd["aligned_KD"])
    headline: dict[str, object] = {}
    if router:
        headline["urbanev_router_and_fixed_fusion"] = router
    if chronos is not None:
        headline["urbanev_chronos2"] = {"macro_rmse": chronos}
    if kd:
        headline["urbanev_residual_distillation"] = kd
    paris = _paris_teacher(paths, data_root, development_shard, allow_full_source)
    if paris is not None:
        headline["paris_development_teacher"] = paris
    return headline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["headline"], default="headline")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path("release-assets"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/recomputed_headline.json"))
    parser.add_argument("--development-shard", type=Path)
    parser.add_argument("--allow-full-source", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.asset_root.rglob("*.npz"))
    results = [
        recompute_package(path, args.data_root, args.development_shard, args.allow_full_source)
        for path in paths
    ]
    headline = headline_summary(
        paths, results, args.data_root, args.development_shard, args.allow_full_source
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": args.scope, "headline": headline, "packages": results}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} from {len(results)} package(s)")


if __name__ == "__main__":
    main()
