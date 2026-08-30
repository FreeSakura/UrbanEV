"""Three-state neural CTMC decoder for the UrbanEV observation semantics.

State order is I/A/U:
  I: available/idle piles;
  A: actively providing electricity;
  U: unavailable or busy but not actively providing electricity.

UrbanEV occupancy observes A + U at an hourly snapshot.  Duration observes
the path integral of A during the target hour, and volume is duration times an
effective charging power.  The functions below encode those distinctions
without imposing the invalid hourly constraint duration <= occupancy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ThreeStateVariant = Literal[
    "free_three_head",
    "capacity_relaxation_multitask",
    "birth_death_multitask",
    "matched_free_simplex",
    "three_state_ctmc",
]


@dataclass(frozen=True)
class LatentStateReconstruction:
    state_rate: np.ndarray
    consistent_mask: np.ndarray
    active_count_raw: np.ndarray
    active_count_projected: np.ndarray
    projection_magnitude: np.ndarray


def reconstruct_iau_state(
    occupancy_count: np.ndarray,
    duration_5min_hours: np.ndarray,
    capacity: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> LatentStateReconstruction:
    """Reconstruct a synchronized I/A/U state from released 5-minute data.

    A five-minute duration value is measured in pile-hours, hence active pile
    count is 12 * duration.  The 0.1204% inconsistent released cells are
    projected onto A <= O for decoding and remain explicitly marked for the
    required mask/projection sensitivity analysis.
    """

    occupancy = np.asarray(occupancy_count, dtype=np.float64)
    duration = np.asarray(duration_5min_hours, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    if occupancy.shape != duration.shape:
        raise ValueError("occupancy and duration shapes differ")
    if cap.ndim != 1 or occupancy.shape[-1] != cap.size:
        raise ValueError("capacity does not match the node dimension")
    if np.any(cap <= 0) or np.any(occupancy < 0) or np.any(duration < 0):
        raise ValueError("counts, durations, and capacities must be non-negative")
    broadcast_capacity = np.broadcast_to(cap, occupancy.shape)
    if np.any(occupancy > broadcast_capacity + tolerance):
        raise ValueError("occupancy exceeds aggregated capacity")

    active_raw = 12.0 * duration
    consistent = active_raw <= occupancy + tolerance
    active = np.minimum(active_raw, occupancy)
    unavailable_non_active = occupancy - active
    idle = broadcast_capacity - occupancy
    state_count = np.stack([idle, active, unavailable_non_active], axis=-1)
    state_rate = state_count / broadcast_capacity[..., np.newaxis]
    if np.max(np.abs(state_rate.sum(axis=-1) - 1.0)) > 1e-10:
        raise AssertionError("reconstructed I/A/U state is not a simplex")
    return LatentStateReconstruction(
        state_rate=state_rate,
        consistent_mask=consistent,
        active_count_raw=active_raw,
        active_count_projected=active,
        projection_magnitude=np.maximum(active_raw - occupancy, 0.0),
    )


def generator_from_parameters(parameters: torch.Tensor) -> torch.Tensor:
    """Build the preregistered five-transition I/A/U generator.

    Parameter order is lambda, mu, rho, kappa, r, where non-negative rates are
    I->A=lambda, I->U=rho, A->I=(1-r)mu, A->U=r*mu, U->I=kappa.  There is no
    unsupported direct U->A transition.
    """

    if parameters.shape[-1] != 5:
        raise ValueError("lambda, mu, rho, kappa, and r are required")
    rate_values = parameters[..., :4]
    split = parameters[..., 4]
    if torch.any(rate_values < 0):
        raise ValueError("CTMC transition rates must be non-negative")
    if torch.any((split < 0) | (split > 1)):
        raise ValueError("completion split r must lie in [0,1]")
    arrival, completion, unavailable, recovery = rate_values.unbind(dim=-1)
    q = parameters.new_zeros(*parameters.shape[:-1], 3, 3)
    q[..., 0, 1] = arrival
    q[..., 0, 2] = unavailable
    q[..., 1, 0] = (1.0 - split) * completion
    q[..., 1, 2] = split * completion
    q[..., 2, 0] = recovery
    diagonal = -q.sum(dim=-1)
    q[..., 0, 0] = diagonal[..., 0]
    q[..., 1, 1] = diagonal[..., 1]
    q[..., 2, 2] = diagonal[..., 2]
    return q


def propagate_state(
    initial_state: torch.Tensor, generator: torch.Tensor, elapsed_hours: float
) -> torch.Tensor:
    """Propagate a row-vector state through exp(Q * elapsed_hours)."""

    if initial_state.shape[-1] != 3 or generator.shape[-2:] != (3, 3):
        raise ValueError("expected [...,3] state and [...,3,3] generator")
    if elapsed_hours < 0:
        raise ValueError("elapsed time must be non-negative")
    transition = torch.matrix_exp(generator * float(elapsed_hours))
    return torch.matmul(initial_state.unsqueeze(-2), transition).squeeze(-2)


def active_path_integral(
    initial_state: torch.Tensor, generator: torch.Tensor, interval_hours: float
) -> torch.Tensor:
    """Return the integral of active-state probability over an interval.

    A block matrix exponential is used because Q is singular and Q^{-1} is
    therefore not a valid general implementation of the integral.
    """

    if initial_state.shape[-1] != 3 or generator.shape[-2:] != (3, 3):
        raise ValueError("expected [...,3] state and [...,3,3] generator")
    if interval_hours < 0:
        raise ValueError("interval length must be non-negative")
    augmented = generator.new_zeros(*generator.shape[:-2], 4, 4)
    augmented[..., :3, :3] = generator
    augmented[..., 1, 3] = 1.0
    start = torch.cat([initial_state, initial_state.new_zeros(*initial_state.shape[:-1], 1)], dim=-1)
    evolved = torch.matmul(
        start.unsqueeze(-2),
        torch.matrix_exp(augmented * float(interval_hours)),
    ).squeeze(-2)
    return evolved[..., 3]


def decode_three_state_ctmc(
    raw_steps: torch.Tensor,
    initial_state: torch.Tensor,
    capacity: torch.Tensor,
    horizon_hours: float,
    power_reference_kw: float,
) -> dict[str, torch.Tensor]:
    """Decode snapshot occupancy and target-hour duration/volume jointly.

    The target occupancy is x(h)[A]+x(h)[U].  Target-hour duration is
    C * integral_h^{h+1} x_A(s) ds, implemented by first propagating to h and
    then integrating one hour.  This is intentionally not integral_0^h.
    """

    if raw_steps.shape[-1] != 6:
        raise ValueError("each step requires lambda, mu, rho, kappa, r, and power")
    if horizon_hours <= 0 or power_reference_kw <= 0:
        raise ValueError("horizon and reference power must be positive")
    horizon_steps = int(horizon_hours)
    if float(horizon_steps) != float(horizon_hours):
        raise ValueError("the hourly piecewise decoder requires an integer horizon")
    if raw_steps.shape[-2] != horizon_steps + 1:
        raise ValueError("raw step sequence must cover h transitions plus the target hour")
    if initial_state.shape != raw_steps.shape[:-2] + (3,):
        raise ValueError("initial state shape does not match raw outputs")
    if torch.any(capacity <= 0):
        raise ValueError("capacity must be positive")

    state = initial_state
    generators: list[torch.Tensor] = []
    transition_parameters: list[torch.Tensor] = []
    for step in range(horizon_steps + 1):
        raw = raw_steps[..., step, :]
        parameters = torch.cat(
            [F.softplus(raw[..., :4]) + 1e-6, torch.sigmoid(raw[..., 4:5])],
            dim=-1,
        )
        generator = generator_from_parameters(parameters)
        generators.append(generator)
        transition_parameters.append(parameters)
        if step < horizon_steps:
            state = propagate_state(state, generator, 1.0)
    target_state = state
    target_hour_generator = generators[-1]
    active_integral = active_path_integral(target_state, target_hour_generator, 1.0)
    duration = capacity * active_integral
    power = float(power_reference_kw) * (F.softplus(raw_steps[..., -1, 5]) + 1e-6)
    occupancy_rate = target_state[..., 1] + target_state[..., 2]
    return {
        "state_rate": target_state,
        "occupancy_rate": occupancy_rate,
        "occupancy_count": occupancy_rate * capacity,
        "duration_rate": active_integral,
        "duration": duration,
        "power": power,
        "volume": duration * power,
        "generator": torch.stack(generators, dim=-3),
        "transition_parameters": torch.stack(transition_parameters, dim=-2),
    }


def decode_matched_free_simplex(
    raw_steps: torch.Tensor,
    capacity: torch.Tensor,
    horizon_hours: int,
    power_reference_kw: float,
) -> dict[str, torch.Tensor]:
    """Equal-parameter free-simplex state/duration/volume control.

    The target state is a direct simplex prediction and target-hour duration is
    a separate bounded head.  There is no generator or Markov reward integral.
    The same shared six-output step head is used as in the CTMC variant.
    """

    if raw_steps.shape[-1] != 6 or raw_steps.shape[-2] != int(horizon_hours) + 1:
        raise ValueError("matched raw sequence must have shape [...,h+1,6]")
    target_raw = raw_steps[..., int(horizon_hours) - 1, :]
    target_hour_raw = raw_steps[..., int(horizon_hours), :]
    state_rate = torch.softmax(target_raw[..., :3], dim=-1)
    duration_rate = torch.sigmoid(target_hour_raw[..., 3:5].mean(dim=-1))
    power = float(power_reference_kw) * (F.softplus(target_hour_raw[..., 5]) + 1e-6)
    duration = capacity * duration_rate
    occupancy_rate = state_rate[..., 1] + state_rate[..., 2]
    return {
        "state_rate": state_rate,
        "occupancy_rate": occupancy_rate,
        "occupancy_count": occupancy_rate * capacity,
        "duration_rate": duration_rate,
        "duration": duration,
        "power": power,
        "volume": duration * power,
    }


def decode_free_three_head(
    raw_steps: torch.Tensor,
    capacity: torch.Tensor,
    horizon_hours: int,
    volume_per_capacity_scale: float,
) -> dict[str, torch.Tensor]:
    """M0: bounded point heads with independent volume and no state law."""

    target_raw = raw_steps[..., int(horizon_hours) - 1, :]
    target_hour_raw = raw_steps[..., int(horizon_hours), :]
    occupancy_rate = torch.sigmoid(target_raw[..., :2].mean(dim=-1))
    duration_rate = torch.sigmoid(target_hour_raw[..., 2:4].mean(dim=-1))
    volume_scaled = F.softplus(target_hour_raw[..., 4:6].mean(dim=-1)) + 1e-6
    duration = capacity * duration_rate
    volume = capacity * float(volume_per_capacity_scale) * volume_scaled
    return {
        "occupancy_rate": occupancy_rate,
        "occupancy_count": capacity * occupancy_rate,
        "duration_rate": duration_rate,
        "duration": duration,
        "volume": volume,
        "volume_scaled": volume_scaled,
    }


def decode_two_state_multitask(
    raw_steps: torch.Tensor,
    initial_state: torch.Tensor,
    capacity: torch.Tensor,
    horizon_hours: int,
    power_reference_kw: float,
    *,
    parameterization: Literal["equilibrium", "birth_death"],
) -> dict[str, torch.Tensor]:
    """M1/M2 controls with identical auxiliary duration-power heads."""

    target_raw = raw_steps[..., int(horizon_hours) - 1, :]
    target_hour_raw = raw_steps[..., int(horizon_hours), :]
    origin_occupancy = initial_state[..., 1] + initial_state[..., 2]
    if parameterization == "equilibrium":
        equilibrium = torch.sigmoid(target_raw[..., 0])
        decay = F.softplus(target_raw[..., 1]) + 1e-6
        arrival = decay * equilibrium
        departure = decay * (1.0 - equilibrium)
    elif parameterization == "birth_death":
        arrival = F.softplus(target_raw[..., 0]) + 1e-6
        departure = F.softplus(target_raw[..., 1]) + 1e-6
        decay = arrival + departure
        equilibrium = arrival / decay
    else:
        raise ValueError(f"unsupported two-state parameterization: {parameterization}")
    retention = torch.exp(-decay * float(horizon_hours))
    occupancy_rate = retention * origin_occupancy + (1.0 - retention) * equilibrium
    duration_rate = torch.sigmoid(target_hour_raw[..., 2:4].mean(dim=-1))
    duration = capacity * duration_rate
    power = float(power_reference_kw) * (
        F.softplus(target_hour_raw[..., 4:6].mean(dim=-1)) + 1e-6
    )
    return {
        "occupancy_rate": occupancy_rate,
        "occupancy_count": capacity * occupancy_rate,
        "duration_rate": duration_rate,
        "duration": duration,
        "power": power,
        "volume": duration * power,
        "equilibrium_rate": equilibrium,
        "decay_rate": decay,
        "retention": retention,
        "arrival_rate": arrival,
        "departure_rate": departure,
    }


class ThreeStateForecaster(nn.Module):
    """Shared hourly O/D/V encoder with equal-budget physical/direct heads."""

    def __init__(
        self,
        variant: ThreeStateVariant,
        history_length: int,
        capacity: np.ndarray,
        power_reference_kw: float,
        volume_per_capacity_scale: float,
        horizon_hours: int,
        *,
        hidden1: int = 96,
        hidden2: int = 48,
        node_embedding_dim: int = 8,
        occupancy_rate_mean: float = 0.25,
        duration_rate_mean: float = 0.18,
        volume_scaled_mean: float = 0.25,
    ) -> None:
        super().__init__()
        if variant not in {
            "free_three_head",
            "capacity_relaxation_multitask",
            "birth_death_multitask",
            "matched_free_simplex",
            "three_state_ctmc",
        }:
            raise ValueError(f"unsupported variant: {variant}")
        if history_length <= 0 or horizon_hours <= 0:
            raise ValueError("history length and horizon must be positive")
        self.variant = variant
        self.horizon_hours = int(horizon_hours)
        self.power_reference_kw = float(power_reference_kw)
        self.volume_per_capacity_scale = float(volume_per_capacity_scale)
        cap = np.asarray(capacity, dtype=np.float32)
        self.register_buffer("capacity", torch.from_numpy(cap))
        context = np.log1p(cap)
        context = (context - context.mean()) / max(float(context.std()), 1e-8)
        self.register_buffer("capacity_context", torch.from_numpy(context.astype(np.float32)))
        self.node_embedding = nn.Embedding(len(cap), node_embedding_dim)
        input_dim = history_length * 3 + node_embedding_dim + 1 + 4 + 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
        )
        self.step_head = nn.Linear(hidden2 + 5, 6)
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)
        if variant == "three_state_ctmc":
            initial_rate = math.log(math.expm1(0.1))
            self.step_head.bias.data[:4] = initial_rate
            self.step_head.bias.data[4] = 0.0
            self.step_head.bias.data[5] = math.log(math.expm1(1.0))
        elif variant == "matched_free_simplex":
            active = max(min(float(duration_rate_mean), 1.0 - 2e-4), 1e-4)
            unavailable = max(float(occupancy_rate_mean) - active, 1e-4)
            idle = max(1.0 - float(occupancy_rate_mean), 1e-4)
            simplex = np.asarray([idle, active, unavailable], dtype=np.float64)
            simplex /= simplex.sum()
            self.step_head.bias.data[:3] = torch.log(
                torch.tensor(simplex, dtype=self.step_head.bias.dtype)
            )
            duration_logit = math.log(active / (1.0 - active))
            self.step_head.bias.data[3:5] = duration_logit
            self.step_head.bias.data[5] = math.log(math.expm1(1.0))
        elif variant == "free_three_head":
            occupancy = min(max(float(occupancy_rate_mean), 1e-4), 1.0 - 1e-4)
            duration = min(max(float(duration_rate_mean), 1e-4), 1.0 - 1e-4)
            self.step_head.bias.data[:2] = math.log(occupancy / (1.0 - occupancy))
            self.step_head.bias.data[2:4] = math.log(duration / (1.0 - duration))
            volume_mean = max(float(volume_scaled_mean), 1e-5)
            self.step_head.bias.data[4:6] = math.log(math.expm1(volume_mean))
        elif variant == "capacity_relaxation_multitask":
            occupancy = min(max(float(occupancy_rate_mean), 1e-4), 1.0 - 1e-4)
            duration = min(max(float(duration_rate_mean), 1e-4), 1.0 - 1e-4)
            self.step_head.bias.data[0] = math.log(occupancy / (1.0 - occupancy))
            self.step_head.bias.data[1] = math.log(math.expm1(0.1))
            self.step_head.bias.data[2:4] = math.log(duration / (1.0 - duration))
            self.step_head.bias.data[4:6] = math.log(math.expm1(1.0))
        elif variant == "birth_death_multitask":
            occupancy = min(max(float(occupancy_rate_mean), 1e-4), 1.0 - 1e-4)
            duration = min(max(float(duration_rate_mean), 1e-4), 1.0 - 1e-4)
            turnover = 0.1
            self.step_head.bias.data[0] = math.log(math.expm1(turnover * occupancy))
            self.step_head.bias.data[1] = math.log(
                math.expm1(turnover * (1.0 - occupancy))
            )
            self.step_head.bias.data[2:4] = math.log(duration / (1.0 - duration))
            self.step_head.bias.data[4:6] = math.log(math.expm1(1.0))

    def forward(
        self,
        history: torch.Tensor,
        target_calendar: torch.Tensor,
        initial_state: torch.Tensor,
        path_calendar: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, nodes, _, channels = history.shape
        if channels != 3:
            raise ValueError("three-state experiments require hourly O/D/V history")
        if initial_state.shape != (batch, nodes, 3):
            raise ValueError("initial state must have shape [batch,nodes,3]")
        node = self.node_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        capacity_context = self.capacity_context.view(1, nodes, 1).expand(batch, -1, -1)
        calendar = target_calendar.unsqueeze(1).expand(-1, nodes, -1)
        features = torch.cat(
            [history.flatten(start_dim=2), node, capacity_context, calendar, initial_state],
            dim=-1,
        )
        latent = self.encoder(features)
        steps = self.horizon_hours + 1
        if path_calendar is None:
            path_calendar = target_calendar.unsqueeze(1).expand(-1, steps, -1)
        if path_calendar.shape != (batch, steps, 4):
            raise ValueError("path_calendar must have shape [batch,h+1,4]")
        latent_steps = latent.unsqueeze(-2).expand(-1, -1, steps, -1)
        calendar_steps = path_calendar.unsqueeze(1).expand(-1, nodes, -1, -1)
        lead = torch.linspace(
            1.0 / steps,
            1.0,
            steps,
            dtype=history.dtype,
            device=history.device,
        ).view(1, 1, steps, 1).expand(batch, nodes, -1, -1)
        raw = self.step_head(torch.cat([latent_steps, calendar_steps, lead], dim=-1))
        # matrix_exp is deliberately kept in FP32 even when the surrounding
        # runner later enables mixed precision.
        with torch.autocast(device_type=history.device.type, enabled=False):
            raw32 = raw.float()
            initial32 = initial_state.float()
            capacity32 = self.capacity.view(1, -1).expand(batch, -1).float()
            if self.variant == "three_state_ctmc":
                result = decode_three_state_ctmc(
                    raw32,
                    initial32,
                    capacity32,
                    self.horizon_hours,
                    self.power_reference_kw,
                )
            elif self.variant == "matched_free_simplex":
                result = decode_matched_free_simplex(
                    raw32,
                    capacity32,
                    self.horizon_hours,
                    self.power_reference_kw,
                )
            elif self.variant == "free_three_head":
                result = decode_free_three_head(
                    raw32,
                    capacity32,
                    self.horizon_hours,
                    self.volume_per_capacity_scale,
                )
            elif self.variant == "capacity_relaxation_multitask":
                result = decode_two_state_multitask(
                    raw32,
                    initial32,
                    capacity32,
                    self.horizon_hours,
                    self.power_reference_kw,
                    parameterization="equilibrium",
                )
            else:
                result = decode_two_state_multitask(
                    raw32,
                    initial32,
                    capacity32,
                    self.horizon_hours,
                    self.power_reference_kw,
                    parameterization="birth_death",
                )
            if "volume_scaled" not in result:
                result["volume_scaled"] = result["volume"] / (
                    capacity32 * self.volume_per_capacity_scale
                )
            return result
