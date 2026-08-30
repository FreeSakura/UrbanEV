from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from innovation.canonical import (
    calendar_features,
    canonical_boundaries,
    canonical_target_indices,
    history_indices,
)
from innovation.deep_baselines import target_index_hash
from innovation.losses import masked_beta_binomial_nll
from innovation.physical_constraints import (
    PhysicalData,
    decode_duration_empirical_power_bounds,
    decode_duration_positive_power,
    effective_power_kw,
    load_physical_data,
)
from repro.metrics import audited_metrics, official_metrics


Variant = Literal[
    "occupancy_only",
    "multivariate_occupancy",
    "physical_state_occupancy",
    "activity_augmented_occupancy",
    "global_power_augmented_occupancy",
    "stock_flow_multivariate",
    "stock_flow_physics",
    "matched_two_head_absolute",
    "free_convex_origin_gate",
    "capacity_relaxation",
    "birth_death_capacity",
    "dual_view_physics_gate",
    "dual_view_stock_flow_gate",
    "adaptive_dual_stock_flow_gate",
    "relative_power_augmented_occupancy",
    "physics_augmented_occupancy",
    "bb_only",
    "multitask_independent",
    "pace_hard",
    "pace_hard_bb",
    "pace_zone_bounded",
    "pace_zone_bounded_bb",
    "pace_detached_bridge",
    "pace_augmented_zone_bounded",
]
GradientStrategy = Literal["sum", "pcgrad_primary"]
PhysicsTransform = Literal["aligned", "permuted_history"]


@dataclass(frozen=True)
class PhysicalScales:
    volume_per_capacity_scale: float
    power_reference_kw: float
    occupancy_rate_mean: float
    duration_rate_mean: float
    volume_scaled_mean: float
    fitted_stop_index: int
    zone_power_low_kw: tuple[float, ...]
    zone_power_high_kw: tuple[float, ...]
    power_log_mean: float
    power_log_std: float


@dataclass(frozen=True)
class PhysicalRunConfig:
    fold: int
    horizon: int
    variant: Variant
    history_length: int
    seed: int
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden1: int
    hidden2: int
    node_embedding_dim: int
    lambda_duration: float
    lambda_volume: float
    lambda_bb: float
    min_delta: float
    gradient_clip: float
    device: str
    amp: bool
    train_limit: int
    gradient_strategy: GradientStrategy = "sum"
    lambda_view: float = 0.25
    physics_transform: PhysicsTransform = "aligned"
    transform_seed: int = 17
    run_id: str | None = None
    attempt_id: int = 1


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def fit_physical_scales(data: PhysicalData, train_end: int) -> PhysicalScales:
    """Fit every data-dependent scale on the current fold's training segment."""

    if train_end <= 0 or train_end > len(data.time):
        raise ValueError("train_end is outside the data range")
    capacity = data.capacity[np.newaxis, :]
    occupancy_rate = data.occupancy_count[:train_end] / capacity
    duration_rate = data.duration_hours[:train_end] / capacity
    volume_per_capacity = data.volume_kwh[:train_end] / capacity
    positive_volume = volume_per_capacity[volume_per_capacity > 1e-12]
    if positive_volume.size == 0:
        raise ValueError("training segment contains no positive volume")
    volume_scale = float(np.quantile(positive_volume, 0.95))
    if not np.isfinite(volume_scale) or volume_scale <= 0:
        raise ValueError("invalid training-only volume scale")
    power = effective_power_kw(
        data.volume_kwh[:train_end], data.duration_hours[:train_end]
    )
    finite_power = power[np.isfinite(power) & (power > 0)]
    if finite_power.size == 0:
        raise ValueError("training segment contains no positive effective power")
    power_reference = float(np.median(finite_power))
    log_power = np.log1p(finite_power)
    zone_lows: list[float] = []
    zone_highs: list[float] = []
    for node in range(power.shape[1]):
        values = power[:, node]
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            zone_lows.append(float(np.min(finite_power)))
            zone_highs.append(float(np.max(finite_power)))
        else:
            zone_lows.append(float(np.min(values)))
            zone_highs.append(float(np.max(values)))
    return PhysicalScales(
        volume_per_capacity_scale=volume_scale,
        power_reference_kw=power_reference,
        occupancy_rate_mean=float(np.mean(occupancy_rate)),
        duration_rate_mean=float(np.mean(duration_rate)),
        volume_scaled_mean=float(np.mean(volume_per_capacity / volume_scale)),
        fitted_stop_index=int(train_end),
        zone_power_low_kw=tuple(zone_lows),
        zone_power_high_kw=tuple(zone_highs),
        power_log_mean=float(np.mean(log_power)),
        power_log_std=max(float(np.std(log_power)), 1e-8),
    )


def _uses_auxiliary_targets(variant: Variant) -> bool:
    return variant in {
        "multitask_independent",
        "pace_hard",
        "pace_hard_bb",
        "pace_zone_bounded",
        "pace_zone_bounded_bb",
        "pace_detached_bridge",
        "pace_augmented_zone_bounded",
    }


def _uses_auxiliary_history(variant: Variant) -> bool:
    return variant in {
        "multivariate_occupancy",
        "physical_state_occupancy",
        "activity_augmented_occupancy",
        "global_power_augmented_occupancy",
        "stock_flow_multivariate",
        "matched_two_head_absolute",
        "free_convex_origin_gate",
        "capacity_relaxation",
        "birth_death_capacity",
        "relative_power_augmented_occupancy",
        "physics_augmented_occupancy",
    } or _uses_auxiliary_targets(variant)


def _uses_physical_state_history(variant: Variant) -> bool:
    return variant == "physical_state_occupancy"


def _uses_physics_augmented_history(variant: Variant) -> bool:
    return variant in {
        "physics_augmented_occupancy",
        "pace_augmented_zone_bounded",
    }


def _uses_activity_augmented_history(variant: Variant) -> bool:
    return variant == "activity_augmented_occupancy"


def _uses_relative_power_augmented_history(variant: Variant) -> bool:
    return variant == "relative_power_augmented_occupancy"


def _uses_global_power_augmented_history(variant: Variant) -> bool:
    return variant in {
        "global_power_augmented_occupancy",
        "stock_flow_physics",
        "dual_view_physics_gate",
        "dual_view_stock_flow_gate",
        "adaptive_dual_stock_flow_gate",
    }


def _uses_stock_flow_decoder(variant: Variant) -> bool:
    return variant in {
        "stock_flow_multivariate",
        "stock_flow_physics",
        "dual_view_stock_flow_gate",
    }


def _uses_dual_view(variant: Variant) -> bool:
    return variant in {
        "dual_view_physics_gate",
        "dual_view_stock_flow_gate",
        "adaptive_dual_stock_flow_gate",
    }


def _uses_adaptive_stock_flow(variant: Variant) -> bool:
    return variant == "adaptive_dual_stock_flow_gate"


def _uses_beta_binomial(variant: Variant) -> bool:
    return variant in {"bb_only", "pace_hard_bb", "pace_zone_bounded_bb"}


class PhysicalSequenceDataset(Dataset):
    """Leakage-free 12-hour histories for all 275 UrbanEV regions."""

    def __init__(
        self,
        data: PhysicalData,
        scales: PhysicalScales,
        targets: np.ndarray,
        horizon: int,
        history_length: int,
        include_auxiliary_history: bool,
        causal_auxiliary_history: bool = False,
        use_physical_state_history: bool = False,
        use_physics_augmented_history: bool = False,
        use_activity_augmented_history: bool = False,
        use_relative_power_augmented_history: bool = False,
        use_global_power_augmented_history: bool = False,
        physics_transform: PhysicsTransform = "aligned",
        transform_seed: int = 17,
    ) -> None:
        capacity = data.capacity[np.newaxis, :]
        occupancy_rate = data.occupancy_count / capacity
        duration_rate = data.duration_hours / capacity
        volume_scaled = data.volume_kwh / (
            capacity * scales.volume_per_capacity_scale
        )
        channels = [occupancy_rate]
        if (
            use_physics_augmented_history
            or use_activity_augmented_history
            or use_relative_power_augmented_history
            or use_global_power_augmented_history
        ):
            power = effective_power_kw(data.volume_kwh, data.duration_hours)
            active = np.isfinite(power)
            low = np.asarray(scales.zone_power_low_kw, dtype=np.float64)[np.newaxis, :]
            high = np.asarray(scales.zone_power_high_kw, dtype=np.float64)[np.newaxis, :]
            width = high - low
            relative_power = np.zeros_like(power)
            variable_support = active & (width > 1e-8)
            relative_power[variable_support] = np.clip(
                (power[variable_support] - np.broadcast_to(low, power.shape)[variable_support])
                / np.broadcast_to(width, power.shape)[variable_support],
                0.0,
                1.0,
            )
            fixed_support = active & (width <= 1e-8)
            relative_power[fixed_support] = 0.5
            channels.extend([duration_rate, volume_scaled])
            if use_physics_augmented_history or use_activity_augmented_history:
                channels.append(active.astype(np.float64))
            if use_physics_augmented_history or use_relative_power_augmented_history:
                channels.append(relative_power)
            if use_global_power_augmented_history:
                standardized_power = np.zeros_like(power)
                standardized_power[active] = (
                    np.log1p(power[active]) - scales.power_log_mean
                ) / scales.power_log_std
                channels.append(standardized_power)
        elif use_physical_state_history:
            power = effective_power_kw(data.volume_kwh, data.duration_hours)
            active = np.isfinite(power)
            standardized_power = np.zeros_like(power)
            standardized_power[active] = (
                np.log1p(power[active]) - scales.power_log_mean
            ) / scales.power_log_std
            channels.extend([duration_rate, active.astype(np.float64), standardized_power])
        elif include_auxiliary_history:
            channels.extend([duration_rate, volume_scaled])
        if causal_auxiliary_history and len(channels) > 1:
            # Hourly duration/volume at timestamp t aggregate the twelve
            # released 5-minute intervals t..t+55.  A forecast issued at the
            # t snapshot may therefore use occupancy[t], but auxiliary
            # interval features are only complete through t-1.
            shifted = [channels[0]]
            for values in channels[1:]:
                previous = np.zeros_like(values)
                previous[1:] = values[:-1]
                shifted.append(previous)
            channels = shifted
        self.features = np.stack(channels, axis=-1).astype(np.float32)
        self.occupancy_count = data.occupancy_count.astype(np.float32, copy=False)
        self.occupancy_rate = occupancy_rate.astype(np.float32, copy=False)
        self.duration = data.duration_hours.astype(np.float32, copy=False)
        self.duration_rate = duration_rate.astype(np.float32, copy=False)
        self.volume = data.volume_kwh.astype(np.float32, copy=False)
        self.volume_scaled = volume_scaled.astype(np.float32, copy=False)
        self.calendar = calendar_features(data.time)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.horizon = int(horizon)
        self.history_length = int(history_length)
        self.physics_transform = physics_transform
        self.transform_seed = int(transform_seed)
        self.causal_auxiliary_history = bool(causal_auxiliary_history)
        if self.targets.ndim != 1 or self.targets.size == 0:
            raise ValueError("targets must be a non-empty one-dimensional array")
        history_indices(int(self.targets.min()), self.horizon, self.history_length)

    def __len__(self) -> int:
        return int(self.targets.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_index = int(self.targets[item])
        observed = history_indices(target_index, self.horizon, self.history_length)
        if int(observed[-1]) != target_index - self.horizon:
            raise AssertionError("history does not end at the forecast origin")
        history = np.ascontiguousarray(self.features[observed].transpose(1, 0, 2))
        if self.physics_transform == "permuted_history":
            generator = np.random.default_rng(
                np.random.SeedSequence([self.transform_seed, target_index])
            )
            permutation = generator.permutation(self.history_length)
            history = history.copy()
            history[:, :, -1] = history[:, permutation, -1]
        elif self.physics_transform != "aligned":
            raise ValueError(f"unsupported physics transform: {self.physics_transform}")
        return {
            "history": torch.from_numpy(history),
            "target_calendar": torch.from_numpy(
                np.ascontiguousarray(self.calendar[target_index])
            ),
            "occupancy_rate": torch.from_numpy(
                np.ascontiguousarray(self.occupancy_rate[target_index])
            ),
            "occupancy_count": torch.from_numpy(
                np.ascontiguousarray(self.occupancy_count[target_index])
            ),
            "duration_rate": torch.from_numpy(
                np.ascontiguousarray(self.duration_rate[target_index])
            ),
            "duration": torch.from_numpy(
                np.ascontiguousarray(self.duration[target_index])
            ),
            "volume_scaled": torch.from_numpy(
                np.ascontiguousarray(self.volume_scaled[target_index])
            ),
            "volume": torch.from_numpy(
                np.ascontiguousarray(self.volume[target_index])
            ),
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-4, 1 - 1e-4))
    return math.log(value / (1 - value))


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    if value > 20:
        return value
    return math.log(math.expm1(value))


def _standardized_log_capacity(capacity: np.ndarray) -> np.ndarray:
    values = np.log1p(np.asarray(capacity, dtype=np.float32))
    return ((values - values.mean()) / max(float(values.std()), 1e-8)).astype(np.float32)


class PhysicalMultiTaskMLP(nn.Module):
    """Shared per-region encoder with optional hard capacity-energy decoding."""

    def __init__(
        self,
        variant: Variant,
        history_length: int,
        input_channels: int,
        capacity: np.ndarray,
        scales: PhysicalScales,
        hidden1: int,
        hidden2: int,
        node_embedding_dim: int,
        forecast_horizon: int = 3,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.scales = scales
        self.forecast_horizon = int(forecast_horizon)
        n_nodes = int(len(capacity))
        self.node_embedding = nn.Embedding(n_nodes, node_embedding_dim)
        self.register_buffer("capacity", torch.from_numpy(np.asarray(capacity, dtype=np.float32)))
        self.register_buffer(
            "capacity_context", torch.from_numpy(_standardized_log_capacity(capacity))
        )
        self.register_buffer(
            "zone_power_low",
            torch.tensor(scales.zone_power_low_kw, dtype=torch.float32),
        )
        self.register_buffer(
            "zone_power_high",
            torch.tensor(scales.zone_power_high_kw, dtype=torch.float32),
        )
        input_dim = history_length * input_channels + node_embedding_dim + 5
        if _uses_dual_view(variant):
            context_dim = node_embedding_dim + 5
            branch_hidden = 48
            branch_latent = 24
            self.base_view_encoder = nn.Sequential(
                nn.Linear(history_length * 3 + context_dim, branch_hidden),
                nn.GELU(),
                nn.Linear(branch_hidden, branch_latent),
                nn.GELU(),
            )
            self.physics_view_encoder = nn.Sequential(
                nn.Linear(history_length * 4 + context_dim, branch_hidden),
                nn.GELU(),
                nn.Linear(branch_hidden, branch_latent),
                nn.GELU(),
            )
            self.base_view_head = nn.Linear(branch_latent, 1)
            self.physics_view_head = nn.Linear(branch_latent, 1)
            if _uses_adaptive_stock_flow(variant):
                self.base_stock_head = nn.Linear(branch_latent, 1)
                self.physics_stock_head = nn.Linear(branch_latent, 1)
                self.raw_stock_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.view_gate = nn.Sequential(
                nn.Linear(branch_latent * 2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            if _uses_adaptive_stock_flow(variant):
                nn.init.constant_(
                    self.base_view_head.bias, _logit(scales.occupancy_rate_mean)
                )
                nn.init.constant_(
                    self.physics_view_head.bias, _logit(scales.occupancy_rate_mean)
                )
                for head in (self.base_stock_head, self.physics_stock_head):
                    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                    nn.init.zeros_(head.bias)
            elif _uses_stock_flow_decoder(variant):
                for head in (self.base_view_head, self.physics_view_head):
                    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
                    nn.init.zeros_(head.bias)
            else:
                nn.init.constant_(
                    self.base_view_head.bias, _logit(scales.occupancy_rate_mean)
                )
                nn.init.constant_(
                    self.physics_view_head.bias, _logit(scales.occupancy_rate_mean)
                )
            return
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
        )
        self.occupancy_head = nn.Linear(hidden2, 1)
        if _uses_stock_flow_decoder(variant):
            # A zero flow means persistence at the forecast origin.  Keeping the
            # initial decoder close to zero makes that physical reference state
            # explicit without freezing any learnable parameter.
            nn.init.normal_(self.occupancy_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.occupancy_head.bias)
        else:
            nn.init.constant_(self.occupancy_head.bias, _logit(scales.occupancy_rate_mean))
        if variant in {
            "matched_two_head_absolute",
            "free_convex_origin_gate",
            "capacity_relaxation",
            "birth_death_capacity",
        }:
            self.decay_head = nn.Linear(hidden2, 1)
            if variant == "birth_death_capacity":
                initial_turnover = 0.1
                initial_occupancy = float(scales.occupancy_rate_mean)
                nn.init.zeros_(self.occupancy_head.weight)
                nn.init.constant_(
                    self.occupancy_head.bias,
                    _inverse_softplus(initial_turnover * initial_occupancy),
                )
                nn.init.zeros_(self.decay_head.weight)
                nn.init.constant_(
                    self.decay_head.bias,
                    _inverse_softplus(initial_turnover * (1.0 - initial_occupancy)),
                )
            elif variant == "free_convex_origin_gate":
                initial_retention = math.exp(-0.1 * float(self.forecast_horizon))
                nn.init.zeros_(self.decay_head.weight)
                nn.init.constant_(self.decay_head.bias, _logit(initial_retention))
            else:
                nn.init.zeros_(self.decay_head.weight)
                nn.init.constant_(
                    self.decay_head.bias,
                    0.0
                    if variant == "matched_two_head_absolute"
                    else _inverse_softplus(0.1),
                )
        if variant == "pace_detached_bridge":
            self.physics_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.GELU(),
                nn.Linear(hidden1, hidden2),
                nn.GELU(),
            )
            self.bridge_head = nn.Linear(hidden2, 1)
            self.bridge_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        if _uses_auxiliary_targets(variant):
            self.duration_head = nn.Linear(hidden2, 1)
            nn.init.constant_(self.duration_head.bias, _logit(scales.duration_rate_mean))
            if variant == "multitask_independent":
                self.volume_head = nn.Linear(hidden2, 1)
                nn.init.constant_(
                    self.volume_head.bias, _inverse_softplus(scales.volume_scaled_mean)
                )
            else:
                self.power_head = nn.Linear(hidden2, 1)
                power_bias = (
                    0.0
                    if variant
                    in {
                        "pace_zone_bounded",
                        "pace_zone_bounded_bb",
                        "pace_detached_bridge",
                        "pace_augmented_zone_bounded",
                    }
                    else _inverse_softplus(1.0)
                )
                nn.init.constant_(self.power_head.bias, power_bias)
        if _uses_beta_binomial(variant):
            self.raw_concentration = nn.Parameter(torch.tensor(48.0, dtype=torch.float32))

    def forward(
        self, history: torch.Tensor, target_calendar: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, n_nodes, _, _ = history.shape
        flattened = history.flatten(start_dim=2)
        node = self.node_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        capacity_context = self.capacity_context.view(1, n_nodes, 1).expand(batch, -1, -1)
        calendar = target_calendar.unsqueeze(1).expand(-1, n_nodes, -1)
        features = torch.cat([flattened, node, capacity_context, calendar], dim=-1)
        if _uses_dual_view(self.variant):
            context = torch.cat([node, capacity_context, calendar], dim=-1)
            base_history = history[..., :3].flatten(start_dim=2)
            base_latent = self.base_view_encoder(
                torch.cat([base_history, context], dim=-1)
            )
            physics_latent = self.physics_view_encoder(
                torch.cat([flattened, context], dim=-1)
            )
            base_logits = self.base_view_head(base_latent).squeeze(-1)
            physics_logits = self.physics_view_head(physics_latent).squeeze(-1)
            gate = torch.sigmoid(
                self.view_gate(torch.cat([base_latent, physics_latent], dim=-1))
            ).squeeze(-1)
            if _uses_adaptive_stock_flow(self.variant):
                origin_rate = history[:, :, -1, 0].clamp(0.0, 1.0)

                def decode_flow(logits: torch.Tensor) -> torch.Tensor:
                    signed = torch.tanh(logits)
                    return (
                        origin_rate
                        + (1.0 - origin_rate) * torch.relu(signed)
                        - origin_rate * torch.relu(-signed)
                    )

                base_absolute = torch.sigmoid(base_logits)
                physics_absolute = torch.sigmoid(physics_logits)
                base_stock = decode_flow(self.base_stock_head(base_latent).squeeze(-1))
                physics_stock = decode_flow(
                    self.physics_stock_head(physics_latent).squeeze(-1)
                )
                stock_gate = torch.sigmoid(self.raw_stock_gate)
                base_rate = (1.0 - stock_gate) * base_absolute + stock_gate * base_stock
                physics_rate = (
                    (1.0 - stock_gate) * physics_absolute + stock_gate * physics_stock
                )
            elif _uses_stock_flow_decoder(self.variant):
                origin_rate = history[:, :, -1, 0].clamp(0.0, 1.0)

                def decode_flow(logits: torch.Tensor) -> torch.Tensor:
                    signed = torch.tanh(logits)
                    return (
                        origin_rate
                        + (1.0 - origin_rate) * torch.relu(signed)
                        - origin_rate * torch.relu(-signed)
                    )

                base_rate = decode_flow(base_logits)
                physics_rate = decode_flow(physics_logits)
            else:
                base_rate = torch.sigmoid(base_logits)
                physics_rate = torch.sigmoid(physics_logits)
            occupancy_rate = (1 - gate) * base_rate + gate * physics_rate
            result = {
                "occupancy_logits": torch.logit(occupancy_rate.clamp(1e-6, 1 - 1e-6)),
                "occupancy_rate": occupancy_rate,
                "occupancy_count": occupancy_rate * self.capacity,
                "base_view_rate": base_rate,
                "physics_view_rate": physics_rate,
                "physics_view_gate": gate,
            }
            if _uses_adaptive_stock_flow(self.variant):
                absolute_rate = (1.0 - gate) * base_absolute + gate * physics_absolute
                stock_rate = (1.0 - gate) * base_stock + gate * physics_stock
                result.update(
                    {
                        "absolute_regime_rate": absolute_rate,
                        "stock_regime_rate": stock_rate,
                        "stock_regime_gate": stock_gate.expand_as(occupancy_rate),
                    }
                )
            return result
        latent = self.encoder(features)
        physics_latent = (
            self.physics_encoder(features)
            if self.variant == "pace_detached_bridge"
            else latent
        )
        occupancy_logits = self.occupancy_head(latent).squeeze(-1)
        if self.variant == "pace_detached_bridge":
            correction = self.bridge_head(physics_latent.detach()).squeeze(-1)
            occupancy_logits = occupancy_logits + self.bridge_gate * correction
        signed_turnover = None
        if self.variant == "matched_two_head_absolute":
            # Parameter-count control for capacity_relaxation.  The extra
            # linear head is algebraically redundant with occupancy_head, so
            # it adds exactly the same 49 parameters without introducing the
            # origin-state retention law.
            occupancy_logits = occupancy_logits + self.decay_head(latent).squeeze(-1)
            occupancy_rate = torch.sigmoid(occupancy_logits)
        elif self.variant == "free_convex_origin_gate":
            origin_rate = history[:, :, -1, 0].clamp(0.0, 1.0)
            equilibrium_rate = torch.sigmoid(occupancy_logits)
            retention = torch.sigmoid(self.decay_head(latent).squeeze(-1))
            decay_rate = -torch.log(retention.clamp_min(1e-8)) / float(self.forecast_horizon)
            arrival_rate = decay_rate * equilibrium_rate
            departure_rate = decay_rate * (1.0 - equilibrium_rate)
            occupancy_rate = retention * origin_rate + (1.0 - retention) * equilibrium_rate
        elif self.variant == "capacity_relaxation":
            origin_rate = history[:, :, -1, 0].clamp(0.0, 1.0)
            equilibrium_rate = torch.sigmoid(occupancy_logits)
            decay_rate = F.softplus(self.decay_head(latent).squeeze(-1)) + 1e-6
            arrival_rate = decay_rate * equilibrium_rate
            departure_rate = decay_rate * (1.0 - equilibrium_rate)
            retention = torch.exp(-decay_rate * float(self.forecast_horizon))
            occupancy_rate = retention * origin_rate + (1.0 - retention) * equilibrium_rate
        elif self.variant == "birth_death_capacity":
            # Mean-field solution of a finite-capacity two-state process:
            # idle --arrival--> occupied --departure--> idle.
            origin_rate = history[:, :, -1, 0].clamp(0.0, 1.0)
            arrival_rate = F.softplus(occupancy_logits) + 1e-6
            departure_rate = F.softplus(self.decay_head(latent).squeeze(-1)) + 1e-6
            decay_rate = arrival_rate + departure_rate
            equilibrium_rate = arrival_rate / decay_rate
            retention = torch.exp(-decay_rate * float(self.forecast_horizon))
            occupancy_rate = retention * origin_rate + (1.0 - retention) * equilibrium_rate
        elif _uses_stock_flow_decoder(self.variant):
            origin_rate = history[:, :, -1, 0].clamp(0.0, 1.0)
            signed_turnover = torch.tanh(occupancy_logits)
            occupancy_rate = (
                origin_rate
                + (1.0 - origin_rate) * torch.relu(signed_turnover)
                - origin_rate * torch.relu(-signed_turnover)
            )
        else:
            signed_turnover = None
            occupancy_rate = torch.sigmoid(occupancy_logits)
        result = {
            "occupancy_logits": occupancy_logits,
            "occupancy_rate": occupancy_rate,
            "occupancy_count": occupancy_rate * self.capacity,
        }
        if signed_turnover is not None:
            result["signed_turnover"] = signed_turnover
        if self.variant in {
            "free_convex_origin_gate",
            "capacity_relaxation",
            "birth_death_capacity",
        }:
            result.update(
                {
                    "equilibrium_rate": equilibrium_rate,
                    "decay_rate": decay_rate,
                    "retention": retention,
                }
            )
            result.update(
                {
                    "arrival_rate": arrival_rate,
                    "departure_rate": departure_rate,
                }
            )
        if not _uses_auxiliary_targets(self.variant):
            return result
        duration_logits = self.duration_head(physics_latent).squeeze(-1)
        if self.variant == "multitask_independent":
            duration = self.capacity * torch.sigmoid(duration_logits)
            volume_scaled = F.softplus(self.volume_head(physics_latent).squeeze(-1))
            volume = (
                self.capacity
                * float(self.scales.volume_per_capacity_scale)
                * volume_scaled
            )
            result.update(
                {
                    "duration": duration,
                    "duration_rate": duration / self.capacity,
                    "volume": volume,
                    "volume_scaled": volume_scaled,
                }
            )
            return result
        raw_power = self.power_head(physics_latent).squeeze(-1)
        if self.variant in {
            "pace_zone_bounded",
            "pace_zone_bounded_bb",
            "pace_detached_bridge",
            "pace_augmented_zone_bounded",
        }:
            duration, power, volume = decode_duration_empirical_power_bounds(
                duration_logits,
                raw_power,
                self.capacity,
                self.zone_power_low,
                self.zone_power_high,
            )
        else:
            duration, power, volume = decode_duration_positive_power(
                duration_logits,
                raw_power,
                self.capacity,
                self.scales.power_reference_kw,
            )
        result.update(
            {
                "duration": duration,
                "duration_rate": duration / self.capacity,
                "power": power,
                "volume": volume,
                "volume_scaled": volume
                / (self.capacity * float(self.scales.volume_per_capacity_scale)),
            }
        )
        return result


def physical_objective(
    model: PhysicalMultiTaskMLP,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: PhysicalRunConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    occupancy_mse = torch.mean((output["occupancy_rate"] - batch["occupancy_rate"]) ** 2)
    terms: dict[str, torch.Tensor] = {"occupancy_mse": occupancy_mse}
    total = occupancy_mse
    if _uses_dual_view(config.variant):
        base_view_mse = torch.mean(
            (output["base_view_rate"] - batch["occupancy_rate"]) ** 2
        )
        physics_view_mse = torch.mean(
            (output["physics_view_rate"] - batch["occupancy_rate"]) ** 2
        )
        terms.update(
            {
                "base_view_mse": base_view_mse,
                "physics_view_mse": physics_view_mse,
            }
        )
        total = total + config.lambda_view * (base_view_mse + physics_view_mse)
    if _uses_auxiliary_targets(config.variant):
        duration_mse = torch.mean((output["duration_rate"] - batch["duration_rate"]) ** 2)
        volume_mse = torch.mean((output["volume_scaled"] - batch["volume_scaled"]) ** 2)
        terms.update({"duration_mse": duration_mse, "volume_mse": volume_mse})
        total = total + config.lambda_duration * duration_mse + config.lambda_volume * volume_mse
    if _uses_beta_binomial(config.variant):
        capacity = model.capacity.view(1, -1).expand_as(batch["occupancy_count"])
        bb_nll, integer_mask = masked_beta_binomial_nll(
            output["occupancy_logits"],
            model.raw_concentration,
            batch["occupancy_count"],
            capacity,
        )
        terms["bb_nll"] = bb_nll
        terms["bb_integer_fraction"] = integer_mask.float().mean()
        total = total + config.lambda_bb * bb_nll
    terms["total"] = total
    return total, terms


def pcgrad_primary_backward(
    model: PhysicalMultiTaskMLP,
    terms: dict[str, torch.Tensor],
    config: PhysicalRunConfig,
) -> float:
    """Protect the occupancy gradient from conflicting physical-task gradients.

    The auxiliary gradient is projected onto the normal plane of the primary
    gradient only when their global dot product is negative.  Head-specific
    parameters retain their own task gradient.  The returned cosine is logged
    before projection to make conflict frequency auditable.
    """

    primary = terms["occupancy_mse"]
    if "bb_nll" in terms:
        primary = primary + config.lambda_bb * terms["bb_nll"]
    if "duration_mse" not in terms or "volume_mse" not in terms:
        raise ValueError("PCGrad requires physical auxiliary losses")
    auxiliary = (
        config.lambda_duration * terms["duration_mse"]
        + config.lambda_volume * terms["volume_mse"]
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    primary_grad = torch.autograd.grad(
        primary, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_grad = torch.autograd.grad(
        auxiliary, parameters, allow_unused=True
    )
    shared = [
        (first, second)
        for first, second in zip(primary_grad, auxiliary_grad)
        if first is not None and second is not None
    ]
    if not shared:
        raise RuntimeError("primary and physical tasks share no trainable parameters")
    dot = sum(torch.sum(first * second) for first, second in shared)
    primary_norm = sum(torch.sum(first * first) for first, _ in shared)
    auxiliary_norm = sum(torch.sum(second * second) for _, second in shared)
    denominator = torch.sqrt(primary_norm * auxiliary_norm).clamp_min(1e-20)
    cosine = float((dot / denominator).detach())
    coefficient = dot / primary_norm.clamp_min(1e-20) if float(dot.detach()) < 0 else None
    for parameter, first, second in zip(parameters, primary_grad, auxiliary_grad):
        if first is None and second is None:
            parameter.grad = None
        elif first is None:
            parameter.grad = second.detach().clone()
        elif second is None:
            parameter.grad = first.detach().clone()
        else:
            corrected = second - coefficient * first if coefficient is not None else second
            parameter.grad = (first + corrected).detach().clone()
    return cosine


def _target_sets(data: PhysicalData, config: PhysicalRunConfig) -> dict[str, np.ndarray]:
    bounds = canonical_boundaries(data.time, config.fold)
    targets = {
        split: canonical_target_indices(
            bounds,
            config.horizon,
            split,
            common_history_budget=config.history_length,
        )
        for split in ("train", "valid", "test")
    }
    if config.train_limit > 0:
        targets["train"] = targets["train"][: config.train_limit]
    if np.intersect1d(targets["train"], targets["valid"]).size:
        raise AssertionError("training and validation targets overlap")
    if np.intersect1d(targets["valid"], targets["test"]).size:
        raise AssertionError("validation and test targets overlap")
    return targets


def build_datasets(
    data: PhysicalData, config: PhysicalRunConfig
) -> tuple[dict[str, PhysicalSequenceDataset], dict[str, np.ndarray], PhysicalScales]:
    bounds = canonical_boundaries(data.time, config.fold)
    scales = fit_physical_scales(data, bounds.train_end)
    targets = _target_sets(data, config)
    datasets = {
        split: PhysicalSequenceDataset(
            data,
            scales,
            indices,
            config.horizon,
            config.history_length,
            # Multivariate occupancy isolates feature information from
            # auxiliary-task supervision.
            include_auxiliary_history=_uses_auxiliary_history(config.variant),
            use_physical_state_history=_uses_physical_state_history(config.variant),
            use_physics_augmented_history=_uses_physics_augmented_history(config.variant),
            use_activity_augmented_history=_uses_activity_augmented_history(config.variant),
            use_relative_power_augmented_history=_uses_relative_power_augmented_history(
                config.variant
            ),
            use_global_power_augmented_history=_uses_global_power_augmented_history(
                config.variant
            ),
            physics_transform=config.physics_transform,
            transform_seed=config.transform_seed,
        )
        for split, indices in targets.items()
    }
    return datasets, targets, scales


def _loader(
    dataset: PhysicalSequenceDataset,
    config: PhysicalRunConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: PhysicalMultiTaskMLP,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp and device.type == "cuda",
        ):
            output = model(batch["history"], batch["target_calendar"])
        arrays = {
            "occupancy_prediction": output["occupancy_rate"],
            "occupancy_target": batch["occupancy_rate"],
            "occupancy_count_prediction": output["occupancy_count"],
            "occupancy_count_target": batch["occupancy_count"],
            "target_index": batch["target_index"],
        }
        if _uses_auxiliary_targets(model.variant):
            arrays.update(
                {
                    "duration_prediction": output["duration"],
                    "duration_target": batch["duration"],
                    "volume_prediction": output["volume"],
                    "volume_target": batch["volume"],
                }
            )
            if "power" in output:
                arrays["power_prediction"] = output["power"]
        if _uses_dual_view(model.variant):
            arrays.update(
                {
                    "base_view_prediction": output["base_view_rate"],
                    "physics_view_prediction": output["physics_view_rate"],
                    "physics_view_gate": output["physics_view_gate"],
                }
            )
            if _uses_adaptive_stock_flow(model.variant):
                arrays.update(
                    {
                        "absolute_regime_prediction": output["absolute_regime_rate"],
                        "stock_regime_prediction": output["stock_regime_rate"],
                        "stock_regime_gate": output["stock_regime_gate"],
                    }
                )
        if model.variant in {
            "free_convex_origin_gate",
            "capacity_relaxation",
            "birth_death_capacity",
        }:
            arrays.update(
                {
                    "equilibrium_prediction": output["equilibrium_rate"],
                    "decay_rate": output["decay_rate"],
                    "retention": output["retention"],
                }
            )
            arrays.update(
                {
                    "arrival_rate": output["arrival_rate"],
                    "departure_rate": output["departure_rate"],
                }
            )
        for key, value in arrays.items():
            converted = value.cpu().numpy() if key == "target_index" else value.float().cpu().numpy()
            parts.setdefault(key, []).append(converted)
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def train_model(
    model: PhysicalMultiTaskMLP,
    datasets: dict[str, PhysicalSequenceDataset],
    config: PhysicalRunConfig,
) -> tuple[PhysicalMultiTaskMLP, pd.DataFrame, int, float]:
    device = torch.device(config.device)
    model.to(device)
    train_loader = _loader(datasets["train"], config, shuffle=True)
    valid_loader = _loader(datasets["valid"], config, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batch_count = 0
        for cpu_batch in train_loader:
            batch = _move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            if config.gradient_strategy == "pcgrad_primary":
                output = model(batch["history"], batch["target_calendar"])
                objective, terms = physical_objective(model, output, batch, config)
                gradient_cosine = pcgrad_primary_backward(model, terms, config)
                sums["gradient_cosine_before_projection"] = (
                    sums.get("gradient_cosine_before_projection", 0.0) + gradient_cosine
                )
            else:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=config.amp and device.type == "cuda",
                ):
                    output = model(batch["history"], batch["target_calendar"])
                    objective, terms = physical_objective(model, output, batch, config)
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if config.gradient_strategy == "pcgrad_primary":
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            batch_count += 1
        valid = evaluate(model, valid_loader, device, config.amp)
        valid_metrics = audited_metrics(
            valid["occupancy_prediction"], valid["occupancy_target"]
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{key}": value / batch_count for key, value in sums.items()},
            "valid_occupancy_RMSE": valid_metrics["RMSE"],
            "valid_occupancy_MAE": valid_metrics["MAE"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if valid_metrics["RMSE"] < best_rmse - config.min_delta:
            best_rmse = valid_metrics["RMSE"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    runtime = time.perf_counter() - started
    if best_epoch <= 0 or not np.isfinite(best_rmse):
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch, runtime


def _metric_rows(
    artifact: dict[str, np.ndarray],
    split: str,
    config: PhysicalRunConfig,
    run_id: str,
    parameters: int,
    best_epoch: int,
    runtime: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    targets = [("occupancy_rate", "occupancy_prediction", "occupancy_target")]
    if _uses_auxiliary_targets(config.variant):
        targets.extend(
            [
                ("duration_hours", "duration_prediction", "duration_target"),
                ("volume_kwh", "volume_prediction", "volume_target"),
            ]
        )
    for target_name, pred_key, true_key in targets:
        for semantics, function in (("audited", audited_metrics), ("official", official_metrics)):
            rows.append(
                {
                    "run_id": run_id,
                    "variant": config.variant,
                    "fold": config.fold,
                    "horizon": config.horizon,
                    "seed": config.seed,
                    "history_length": config.history_length,
                    "split": split,
                    "target": target_name,
                    "metric_semantics": semantics,
                    "samples": len(artifact["target_index"]),
                    "parameters": parameters,
                    "best_epoch": best_epoch,
                    "runtime_seconds": runtime,
                    **function(artifact[pred_key], artifact[true_key]),
                }
            )
    return rows


def _physical_violations(
    artifact: dict[str, np.ndarray], capacity: np.ndarray
) -> dict[str, float | int]:
    cap = capacity[np.newaxis, :]
    result: dict[str, float | int] = {
        "occupancy_below_zero": int(np.count_nonzero(artifact["occupancy_count_prediction"] < 0)),
        "occupancy_above_capacity": int(
            np.count_nonzero(artifact["occupancy_count_prediction"] > cap + 1e-6)
        ),
    }
    if "duration_prediction" in artifact:
        result.update(
            {
                "duration_below_zero": int(np.count_nonzero(artifact["duration_prediction"] < 0)),
                "duration_above_capacity": int(
                    np.count_nonzero(artifact["duration_prediction"] > cap + 1e-6)
                ),
                "volume_below_zero": int(np.count_nonzero(artifact["volume_prediction"] < 0)),
            }
        )
    if "power_prediction" in artifact:
        identity_error = np.max(
            np.abs(
                artifact["volume_prediction"]
                - artifact["duration_prediction"] * artifact["power_prediction"]
            )
        )
        result["max_abs_volume_duration_power_identity_error"] = float(identity_error)
    return result


def _validate_config(config: PhysicalRunConfig) -> None:
    if config.history_length != 12:
        raise ValueError("the PACE-EV main protocol is frozen to a 12-hour history")
    if config.epochs <= 0 or config.patience <= 0 or config.batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("optimizer configuration is invalid")
    if min(
        config.lambda_duration,
        config.lambda_volume,
        config.lambda_bb,
        config.lambda_view,
    ) < 0:
        raise ValueError("loss weights must be non-negative")
    if config.hidden1 <= 0 or config.hidden2 <= 0 or config.node_embedding_dim < 0:
        raise ValueError("model dimensions are invalid")
    if config.train_limit < 0 or config.attempt_id <= 0:
        raise ValueError("run controls are invalid")
    if config.physics_transform != "aligned" and not _uses_global_power_augmented_history(
        config.variant
    ):
        raise ValueError("physics transforms require an effective-power history variant")
    if config.gradient_strategy == "pcgrad_primary":
        if not _uses_auxiliary_targets(config.variant):
            raise ValueError("pcgrad_primary requires an auxiliary-target variant")
        if config.amp:
            raise ValueError("pcgrad_primary currently requires --no-amp for exact gradients")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(root: Path, config: PhysicalRunConfig) -> Path:
    _validate_config(config)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_deterministic(config.seed)
    torch.set_num_threads(1)
    total_started = time.perf_counter()
    data_dir = root / "audited" / "data"
    data = load_physical_data(data_dir)
    datasets, targets, scales = build_datasets(data, config)
    input_channels = (
        5
        if _uses_physics_augmented_history(config.variant)
        else 4
        if (
            _uses_activity_augmented_history(config.variant)
            or _uses_relative_power_augmented_history(config.variant)
            or _uses_global_power_augmented_history(config.variant)
        )
        else 4
        if _uses_physical_state_history(config.variant)
        else 3
        if _uses_auxiliary_history(config.variant)
        else 1
    )
    model = PhysicalMultiTaskMLP(
        config.variant,
        config.history_length,
        input_channels,
        data.capacity,
        scales,
        config.hidden1,
        config.hidden2,
        config.node_embedding_dim,
        config.horizon,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = config.run_id or (
        f"pace_{config.variant}_L12_f{config.fold}_h{config.horizon}_s{config.seed}_{stamp}"
    )
    run_dir = root / "innovation" / "physical_runs" / run_id / f"attempt_{config.attempt_id:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_payload = {
        **asdict(config),
        "run_id": run_id,
        "parameters": parameters,
        "input_channels": input_channels,
        "physical_scales": asdict(scales),
        "target_counts": {key: int(len(value)) for key, value in targets.items()},
        "target_index_sha256": {key: target_index_hash(value) for key, value in targets.items()},
        "target_ranges": {key: [int(value[0]), int(value[-1])] for key, value in targets.items()},
        "input_sha256": {
            name: _sha256(data_dir / name)
            for name in ("occupancy.csv", "duration.csv", "volume.csv", "inf.csv")
        },
        "source_sha256": {
            name: _sha256(Path(__file__).resolve().parent / name)
            for name in (
                "canonical.py",
                "losses.py",
                "physical_constraints.py",
                "physical_multitask.py",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "semantics": {
            "protocol": "canonical_transformer_native_direct_endpoint",
            "history_budget": 12,
            "checkpoint_selection": "validation_occupancy_RMSE",
            "primary_target": "occupancy_rate",
            "power_constraint": "nonnegative_softplus_no_fake_upper_bound",
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "running", "started_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )
    try:
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model, history, best_epoch, runtime = train_model(model, datasets, config)
        device = torch.device(config.device)
        validation = evaluate(model, _loader(datasets["valid"], config, shuffle=False), device, config.amp)
        test = evaluate(model, _loader(datasets["test"], config, shuffle=False), device, config.amp)
        rows = _metric_rows(validation, "valid", config, run_id, parameters, best_epoch, runtime)
        rows.extend(_metric_rows(test, "test", config, run_id, parameters, best_epoch, runtime))
        pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
        history.to_csv(run_dir / "history.csv", index=False)
        arrays = {
            **{f"test_{key}": value for key, value in test.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
            "zone_ids": np.asarray(data.zone_ids),
            "test_target_time": data.time[test["target_index"]].astype(str).to_numpy(),
            "test_origin_index": test["target_index"] - config.horizon,
            "test_input_start_index": test["target_index"]
            - config.horizon
            - config.history_length
            + 1,
        }
        np.savez_compressed(run_dir / "predictions.npz", **arrays)
        violations = {
            "validation": _physical_violations(validation, data.capacity),
            "test": _physical_violations(test, data.capacity),
        }
        (run_dir / "physical_violations.json").write_text(
            json.dumps(violations, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        peak_vram = int(torch.cuda.max_memory_allocated()) if config.device.startswith("cuda") else 0
        status = {
            "status": "success",
            "finished_at": datetime.now().isoformat(),
            "best_epoch": best_epoch,
            "runtime_seconds": runtime,
            "total_runtime_seconds": time.perf_counter() - total_started,
            "peak_vram_bytes": peak_vram,
        }
        (run_dir / "status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
        print(json.dumps({"run_dir": str(run_dir), **status}), flush=True)
        return run_dir
    except Exception as error:
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "finished_at": datetime.now().isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="PACE-EV 12-hour physical multitask runner")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--horizon", type=int, required=True, choices=(3, 6, 9, 12))
    parser.add_argument(
        "--variant",
        choices=(
            "occupancy_only",
            "multivariate_occupancy",
            "physical_state_occupancy",
            "activity_augmented_occupancy",
            "global_power_augmented_occupancy",
            "stock_flow_multivariate",
            "stock_flow_physics",
            "matched_two_head_absolute",
            "free_convex_origin_gate",
            "capacity_relaxation",
            "birth_death_capacity",
            "dual_view_physics_gate",
            "dual_view_stock_flow_gate",
            "adaptive_dual_stock_flow_gate",
            "relative_power_augmented_occupancy",
            "physics_augmented_occupancy",
            "bb_only",
            "multitask_independent",
            "pace_hard",
            "pace_hard_bb",
            "pace_zone_bounded",
            "pace_zone_bounded_bb",
            "pace_detached_bridge",
            "pace_augmented_zone_bounded",
        ),
        required=True,
    )
    parser.add_argument("--history-length", type=int, default=12, choices=(12,))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden1", type=int, default=96)
    parser.add_argument("--hidden2", type=int, default=48)
    parser.add_argument("--node-embedding-dim", type=int, default=8)
    parser.add_argument("--lambda-duration", type=float, default=1.0)
    parser.add_argument("--lambda-volume", type=float, default=1.0)
    parser.add_argument("--lambda-bb", type=float, default=1e-3)
    parser.add_argument("--lambda-view", type=float, default=0.25)
    parser.add_argument(
        "--physics-transform",
        choices=("aligned", "permuted_history"),
        default="aligned",
    )
    parser.add_argument("--transform-seed", type=int, default=17)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument(
        "--gradient-strategy", choices=("sum", "pcgrad_primary"), default="sum"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id", type=int, default=1)
    args = parser.parse_args()
    config = PhysicalRunConfig(
        fold=args.fold,
        horizon=args.horizon,
        variant=args.variant,
        history_length=args.history_length,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        node_embedding_dim=args.node_embedding_dim,
        lambda_duration=args.lambda_duration,
        lambda_volume=args.lambda_volume,
        lambda_bb=args.lambda_bb,
        lambda_view=args.lambda_view,
        physics_transform=args.physics_transform,
        transform_seed=args.transform_seed,
        min_delta=args.min_delta,
        gradient_clip=args.gradient_clip,
        device=args.device,
        amp=args.amp,
        train_limit=args.train_limit,
        gradient_strategy=args.gradient_strategy,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
    )
    run(args.root.resolve(), config)


if __name__ == "__main__":
    main()
