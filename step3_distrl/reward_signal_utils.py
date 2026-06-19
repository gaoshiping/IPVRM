from __future__ import annotations

import torch


def zero_after_valid_length_(tensor: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
    for row_index, valid_length in enumerate(valid_lengths.tolist()):
        valid_length = int(valid_length)
        tensor[row_index, valid_length:] = 0
    return tensor


def build_step_reward_scores(
    td_rewards: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    step_reward_scores = torch.zeros_like(td_rewards)
    for row_index, valid_length in enumerate(valid_lengths.tolist()):
        last_valid_index = int(valid_length) - 1
        if last_valid_index < 0:
            continue
        step_reward_scores[row_index, :last_valid_index] = td_rewards[row_index, :last_valid_index]
    return step_reward_scores


def build_candidate_td_scores(candidate_td_rewards: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
    candidate_td_scores = torch.zeros_like(candidate_td_rewards)
    for row_index, valid_length in enumerate(valid_lengths.tolist()):
        last_valid_index = int(valid_length) - 1
        if last_valid_index <= 0:
            continue
        candidate_td_scores[row_index, :last_valid_index] = candidate_td_rewards[row_index, :last_valid_index]
    return candidate_td_scores


def masked_suffix_sum(score: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    assert score.shape == valid_mask.shape
    masked_score = score * valid_mask
    return torch.cumsum(masked_score.flip(dims=[1]), dim=1).flip(dims=[1])


def masked_suffix_mean(score: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    assert score.shape == valid_mask.shape
    masked_score = score.flip(dims=[1]) * valid_mask.flip(dims=[1])
    suffix_sum = torch.cumsum(masked_score, dim=1).flip(dims=[1])
    suffix_count = torch.cumsum(valid_mask.flip(dims=[1]), dim=1).flip(dims=[1]).clamp(min=1)
    return (suffix_sum / suffix_count) * valid_mask


def masked_suffix_geometric_mean(score: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    assert score.shape == valid_mask.shape
    assert (score > 0).all(), "Score must be positive for geometric mean"

    flipped_score = score.flip(dims=[1])
    flipped_mask = valid_mask.flip(dims=[1])
    masked_log_score = torch.log(flipped_score) * flipped_mask
    suffix_log_sum = torch.cumsum(masked_log_score, dim=1).flip(dims=[1])
    suffix_count = torch.cumsum(flipped_mask, dim=1).flip(dims=[1]).clamp(min=1)
    return torch.exp(suffix_log_sum / suffix_count) * valid_mask


def build_candidate_token_mask(
    old_log_prob_topk_values: torch.Tensor,
    response_mask: torch.Tensor,
    min_probability: float = 0.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    _, _, topk = old_log_prob_topk_values.shape
    response_mask = response_mask.clone().bool()
    response_lengths = response_mask.sum(-1)

    for row_index, response_length in enumerate(response_lengths.tolist()):
        if response_length > 0:
            response_mask[row_index, int(response_length) - 1] = False

    candidate_token_mask = response_mask.unsqueeze(-1).expand(-1, -1, topk)

    if min_probability > 1e-5:
        candidate_token_mask = candidate_token_mask & (old_log_prob_topk_values.exp() > min_probability)

    if 1 - top_p > 1e-5:
        old_probability = old_log_prob_topk_values.exp()
        sorted_probability, sorted_indices = torch.sort(old_probability, dim=-1, descending=True)
        cumulative_probability = torch.cumsum(sorted_probability, dim=-1)
        sorted_top_p_mask = cumulative_probability <= top_p
        sorted_top_p_mask[..., 0] = True
        reverse_indices = torch.argsort(sorted_indices, dim=-1)
        candidate_token_mask = candidate_token_mask & torch.gather(sorted_top_p_mask, dim=-1, index=reverse_indices)
    candidate_token_mask[:, :, 0] = response_mask
    return candidate_token_mask
