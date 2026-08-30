from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_count_inputs(count: torch.Tensor, capacity: torch.Tensor) -> None:
    if count.shape != capacity.shape:
        raise ValueError(f"count and capacity shapes differ: {count.shape} vs {capacity.shape}")
    if torch.any(capacity <= 0):
        raise ValueError("capacity must be positive")
    if torch.any(count < 0) or torch.any(count > capacity):
        raise ValueError("count must satisfy 0 <= count <= capacity")


def _validate_discrete_counts(count: torch.Tensor, capacity: torch.Tensor) -> None:
    """Reject fractional observations for genuinely discrete likelihoods."""
    _validate_count_inputs(count, capacity)
    tolerance = 1e-6
    if torch.any(torch.abs(count - torch.round(count)) > tolerance):
        raise ValueError(
            "a discrete count likelihood requires integer count values; "
            "use quasi_binomial_nll for UrbanEV's fractional observations"
        )
    if torch.any(torch.abs(capacity - torch.round(capacity)) > tolerance):
        raise ValueError("a discrete count likelihood requires integer capacities")


def binomial_nll(logits: torch.Tensor, count: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood for occupied-pile counts."""
    _validate_discrete_counts(count, capacity)
    distribution = torch.distributions.Binomial(total_count=capacity, logits=logits)
    return -distribution.log_prob(count).mean()


def quasi_binomial_nll(
    logits: torch.Tensor,
    count: torch.Tensor,
    capacity: torch.Tensor,
    *,
    reduction: str = "exposure_mean",
) -> torch.Tensor:
    """Binomial log-kernel extended to fractional counts.

    UrbanEV contains 691 half-count observations, so this objective is a
    quasi-likelihood rather than a discrete probability mass function.  The
    default reduction weights each charging pile equally.  ``zone_mean`` first
    normalizes by capacity and then weights every zone-time observation equally.
    """
    _validate_count_inputs(count, capacity)
    log_kernel = count * F.logsigmoid(logits) + (capacity - count) * F.logsigmoid(-logits)
    if reduction == "exposure_mean":
        return -log_kernel.sum() / capacity.sum()
    if reduction == "zone_mean":
        return -(log_kernel / capacity).mean()
    raise ValueError(f"unknown reduction: {reduction}")


def beta_binomial_nll(
    mean_logits: torch.Tensor,
    raw_concentration: torch.Tensor,
    count: torch.Tensor,
    capacity: torch.Tensor,
    minimum_concentration: float = 2.0,
) -> torch.Tensor:
    """Mean Beta-Binomial NLL with a stable mean/concentration parameterization."""
    _validate_discrete_counts(count, capacity)
    probability = torch.sigmoid(mean_logits).clamp(1e-6, 1 - 1e-6)
    concentration = F.softplus(raw_concentration) + minimum_concentration
    alpha = probability * concentration
    beta = (1 - probability) * concentration
    log_choose = (
        torch.lgamma(capacity + 1)
        - torch.lgamma(count + 1)
        - torch.lgamma(capacity - count + 1)
    )
    log_beta_posterior = (
        torch.lgamma(count + alpha)
        + torch.lgamma(capacity - count + beta)
        - torch.lgamma(capacity + alpha + beta)
    )
    log_beta_prior = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return -(log_choose + log_beta_posterior - log_beta_prior).mean()


def masked_beta_binomial_nll(
    mean_logits: torch.Tensor,
    raw_concentration: torch.Tensor,
    count: torch.Tensor,
    capacity: torch.Tensor,
    *,
    integer_tolerance: float = 1e-6,
    minimum_concentration: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict Beta-Binomial NLL on integer UrbanEV observations only.

    UrbanEV contains a small number of half-count observations.  Feeding them
    to the continuous gamma extension would produce a finite number, but that
    number is not a discrete Beta-Binomial probability mass.  This helper
    masks such cells and returns the mask for audit logging.
    """

    _validate_count_inputs(count, capacity)
    integer_mask = torch.abs(count - torch.round(count)) <= integer_tolerance
    if not torch.any(integer_mask):
        raise ValueError("batch contains no integer observations for Beta-Binomial NLL")
    masked_logits = mean_logits[integer_mask].float()
    masked_count = torch.round(count[integer_mask]).float()
    masked_capacity = capacity[integer_mask].float()
    probability = torch.sigmoid(masked_logits).clamp(1e-6, 1 - 1e-6)
    concentration = F.softplus(raw_concentration.float()) + minimum_concentration
    alpha = probability * concentration
    beta = (1 - probability) * concentration
    log_choose = (
        torch.lgamma(masked_capacity + 1)
        - torch.lgamma(masked_count + 1)
        - torch.lgamma(masked_capacity - masked_count + 1)
    )
    log_beta_posterior = (
        torch.lgamma(masked_count + alpha)
        + torch.lgamma(masked_capacity - masked_count + beta)
        - torch.lgamma(masked_capacity + alpha + beta)
    )
    log_beta_prior = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return -(log_choose + log_beta_posterior - log_beta_prior).mean(), integer_mask


def beta_binomial_mean(mean_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mean_logits)
