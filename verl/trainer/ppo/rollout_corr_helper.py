# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import torch

from verl.protocol import DataProto
import verl.utils.torch_functional as verl_F


SAFETY_BOUND = 20.0


def _prefix_rollout_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    prefixed: dict[str, float] = {}
    for key, value in metrics.items():
        prefixed[f"rollout_corr/{key}"] = value.item() if isinstance(value, torch.Tensor) else float(value)
    return prefixed


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor | None,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics: dict[str, float] = {}
    mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
    training_ppl = torch.exp(-mean_log_prob_training).mean()
    metrics["training_ppl"] = training_ppl.detach().item()
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    if rollout_log_prob is None:
        return metrics

    metrics["kl"] = verl_F.masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()

    log_ratio = old_log_prob - rollout_log_prob
    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    k3_kl_matrix = torch.exp(log_ratio_safe) - log_ratio_safe - 1
    metrics["k3_kl"] = verl_F.masked_mean(k3_kl_matrix, response_mask).detach().item()

    mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
    rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()
    metrics["rollout_ppl"] = rollout_ppl.detach().item()
    metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
    metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
    metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
    metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()
    metrics["ppl_ratio"] = torch.exp(log_ppl_diff).mean().detach().item()

    rho_token = torch.exp(log_ratio_safe)
    metrics["chi2_token"] = (verl_F.masked_mean(rho_token.square(), response_mask) - 1.0).detach().item()
    log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
    log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    metrics["chi2_seq"] = (torch.exp(2.0 * log_ratio_sum_safe).mean() - 1.0).detach().item()
    return metrics


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    return _prefix_rollout_metrics(metrics)


def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if rollout_is not in {"token", "sequence"}:
        raise ValueError(f"Invalid rollout_is={rollout_is!r}; expected 'token' or 'sequence'.")
    if rollout_is_threshold <= 0:
        raise ValueError(f"rollout_is_threshold must be positive, got {rollout_is_threshold}.")

    if rollout_is == "token":
        log_ratio_for_metrics = log_ratio
        weights = torch.exp(torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND))
    else:
        log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum
        weights = torch.exp(torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)).expand_as(log_ratio)

    weights = weights * response_mask
    metrics = compute_is_metrics(
        rollout_is_weights=weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
    )
    weights = weights.clamp(max=rollout_is_threshold).detach()

    if rollout_is_batch_normalize:
        mask_float = response_mask.to(dtype=weights.dtype)
        if rollout_is == "token":
            mean_weight = verl_F.masked_mean(weights, mask_float)
        else:
            seq_weights = verl_F.masked_mean(weights, mask_float, axis=-1)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=weights.dtype)
            mean_weight = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)
        if mean_weight > 1e-8:
            weights = weights / mean_weight
            metrics["rollout_is_batch_norm_factor"] = mean_weight.detach().item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return weights, metrics


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics: dict[str, float] = {}
    lower_threshold = 1.0 / rollout_is_threshold
    device = rollout_is_weights.device
    log_upper = torch.log(torch.tensor(rollout_is_threshold, device=device))
    log_lower = torch.log(torch.tensor(lower_threshold, device=device))

    if rollout_is == "sequence":
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_ratio_for_metrics.max(), max=SAFETY_BOUND)).item()
        metrics["rollout_is_min"] = torch.exp(torch.clamp(log_ratio_for_metrics.min(), min=-SAFETY_BOUND)).item()
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()
        metrics["rollout_is_ratio_fraction_high"] = (log_ratio_for_metrics > log_upper).float().mean().item()
        metrics["rollout_is_ratio_fraction_low"] = (log_ratio_for_metrics < log_lower).float().mean().item()
    else:
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(
            (rollout_is_weights > rollout_is_threshold).float(), response_mask
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(
            (rollout_is_weights < lower_threshold).float(), response_mask
        ).item()
        mask_bool = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    weights_for_stats = rollout_is_weights.clamp(min=lower_threshold, max=rollout_is_threshold)
    mean = verl_F.masked_mean(weights_for_stats, response_mask)
    variance = verl_F.masked_mean(weights_for_stats.square(), response_mask) - mean.square()
    metrics["rollout_is_std"] = torch.sqrt(torch.clamp(variance, min=0.0)).item()

    normalized = weights_for_stats / (mean + 1e-8)
    metrics["rollout_is_eff_sample_size"] = 1.0 / verl_F.masked_mean(normalized.square(), response_mask).item()

    seq_mean_weights = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)
    metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item()
    metrics["rollout_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
    metrics["rollout_is_seq_max"] = seq_mean_weights.max().item()
    metrics["rollout_is_seq_min"] = seq_mean_weights.min().item()
    metrics["rollout_is_seq_max_deviation"] = (seq_mean_weights - 1.0).abs().max().item()
    metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean().item()
    metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < lower_threshold).float().mean().item()
    return metrics


def compute_rollout_correction_and_add_to_batch(batch: DataProto, rollout_corr_config: Any) -> tuple[DataProto, dict]:
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    if rollout_rs is not None:
        raise ValueError("This NFPO overlay keeps rollout rejection sampling out of the release path.")

    old_log_prob = batch.batch["old_log_probs"]
    rollout_log_prob = batch.batch["rollout_log_probs"]
    response_mask = batch.batch["response_mask"]
    log_ratio = old_log_prob - rollout_log_prob

    metrics = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    rollout_is = rollout_corr_config.get("rollout_is", None)
    if rollout_is is not None:
        rollout_is_weights, is_metrics = compute_rollout_correction_weights(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_corr_config.get("rollout_is_threshold", 2.0),
            rollout_is_batch_normalize=rollout_corr_config.get("rollout_is_batch_normalize", False),
        )
        metrics.update(is_metrics)
        batch = batch.union(DataProto.from_dict(tensors={"rollout_is_weights": rollout_is_weights}))

    return batch, _prefix_rollout_metrics(metrics)
