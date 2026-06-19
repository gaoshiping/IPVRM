import torch
import torch.nn.functional as F

import verl
import verl.utils.torch_functional as verl_F


def build_prompt_level_reward_tensors(
    scores,
    prompt_group_id: torch.Tensor,
    sample_rank_in_group: torch.Tensor,
    n_samples: int,
    base_margin: float,
    use_dlw: bool,
    use_adb: bool,
) -> dict[str, torch.Tensor]:
    prompt_group_id = prompt_group_id.view(-1)
    sample_rank_in_group = sample_rank_in_group.view(-1)
    acc = torch.as_tensor(scores, device=prompt_group_id.device, dtype=torch.float32).view(-1)

    if not (acc.shape == prompt_group_id.shape == sample_rank_in_group.shape):
        raise ValueError(
            "scores, prompt_group_id, and sample_rank_in_group must have the same flattened shape. "
            f"Got {tuple(acc.shape)}, {tuple(prompt_group_id.shape)}, and {tuple(sample_rank_in_group.shape)}."
        )

    if acc.numel() % n_samples != 0:
        raise ValueError(f"Batch size {acc.numel()} must be divisible by rollout.n={n_samples}.")

    dlw_weight = torch.ones_like(acc)
    margin = torch.full_like(acc, float(base_margin))
    expected_ranks = torch.arange(n_samples, device=sample_rank_in_group.device, dtype=sample_rank_in_group.dtype)

    unique_prompt_ids = torch.unique(prompt_group_id, sorted=True)
    for prompt_id in unique_prompt_ids:
        group_indices = torch.nonzero(prompt_group_id == prompt_id, as_tuple=False).flatten()
        if group_indices.numel() != n_samples:
            raise ValueError(
                f"Prompt group {prompt_id.item()} has {group_indices.numel()} samples, expected rollout.n={n_samples}."
            )

        group_indices = group_indices[torch.argsort(sample_rank_in_group[group_indices])]
        group_ranks = sample_rank_in_group[group_indices]
        if not torch.equal(group_ranks, expected_ranks):
            raise ValueError(
                f"Prompt group {prompt_id.item()} has sample ranks {group_ranks.tolist()}, "
                f"expected {expected_ranks.tolist()}."
            )

        group_acc = acc[group_indices]
        group_acc_mean = group_acc.mean()

        if use_dlw:
            group_weight = torch.where(group_acc > 0.5, 1.0 - group_acc_mean, group_acc_mean)
            group_weight_sum = group_weight.sum()
            if group_weight_sum.abs().item() < 1e-8:
                group_weight = torch.ones_like(group_weight)
            else:
                group_weight = group_weight * (float(n_samples) / group_weight_sum)
            dlw_weight[group_indices] = group_weight

        if use_adb:
            probability = torch.clamp(group_acc_mean, 1e-6, 1 - 1e-6)
            difficulty_boundary = torch.log(probability / (1 - probability))
            margin[group_indices] = torch.where(
                group_acc > 0.5,
                float(base_margin) - difficulty_boundary,
                float(base_margin) + difficulty_boundary,
            )

    return {"acc": acc, "dlw_weight": dlw_weight, "margin": margin}


def compute_implicitprm_loss(token_level_scores, acc, response_mask, beta, loss_weight, margin=None):
    sequence_scores = (token_level_scores * response_mask).sum(dim=1)
    logits = sequence_scores * beta
    acc = torch.as_tensor(acc, device=logits.device, dtype=logits.dtype).view_as(logits)
    loss_weight = torch.as_tensor(loss_weight, device=logits.device, dtype=logits.dtype)
    if loss_weight.ndim == 0:
        loss_weight = loss_weight.expand_as(logits)
    else:
        loss_weight = loss_weight.view_as(logits)
    if margin is None:
        cur_dpo_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, acc, reduction="none")
    else:
        margin = torch.as_tensor(margin, device=logits.device, dtype=logits.dtype)
        if margin.ndim == 0:
            margin = margin.expand_as(logits)
        else:
            margin = margin.view_as(logits)
        signed_logits = logits * ((acc > 0).float() * 2 - 1) - margin
        cur_dpo_loss = -F.logsigmoid(signed_logits)
    loss = (cur_dpo_loss * loss_weight).mean()
    return loss

def compute_ipvrm_loss(
    log_ratio_per_token: torch.Tensor,   # [B, T], values = log pi_phi - log pi_ref
    acc: torch.Tensor,                   # [B], values in {0,1}
    response_mask: torch.Tensor,         # [B, T], bool
    beta: float,
    dlw_weight: torch.Tensor,            # [B] or scalar -> will be viewed as [B]
    margin: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Implicit IPVRM RM update with a double-sided margin:
      - Convert label to s in {-1,+1}, multiply into per-token log-ratio first.
      - Logit = beta * mean_prefix(log-ratio * s) - margin
      - Loss = -logsigmoid(logit) = softplus(margin - beta * s * mean_prefix)
    This sets neutral points at mean_prefix = +/-(margin / beta) for positive/negative classes.
    """
    if margin is None:
        margin = 0.5 * beta

    # ----- dtype and device normalization -----
    device = log_ratio_per_token.device
    log_ratio_per_token = log_ratio_per_token.to(device)
    acc = acc.to(device=device).float().view(-1)                 # [B]
    response_mask = response_mask.to(device=device).float()      # [B, T] in {0,1}
    dlw_weight = torch.as_tensor(dlw_weight, device=device, dtype=log_ratio_per_token.dtype).view(-1)

    # ----- shape checks -----
    B, T = log_ratio_per_token.shape
    assert acc.shape == (B,), f"acc must be [B], got {tuple(acc.shape)}"
    assert response_mask.shape == (B, T), f"response_mask must be [B,T], got {tuple(response_mask.shape)}"
    assert dlw_weight.shape == (B,), f"dlw_weight must be [B], got {tuple(dlw_weight.shape)}"

    margin = torch.as_tensor(margin, device=device, dtype=log_ratio_per_token.dtype)
    if margin.ndim == 0:
        margin = margin.expand(B)
    else:
        margin = margin.view(-1)
    assert margin.shape == (B,), f"margin must be scalar or [B], got {tuple(margin.shape)}"

    # ----- map {0,1} labels to {-1,+1} and fold the sign into features -----
    sign = (acc * 2 - 1).unsqueeze(-1)                           # [B, 1]
    signed_lr = log_ratio_per_token * sign                       # [B, T]

    # ----- prefix sums and counts over valid response tokens only -----
    masked_lr   = signed_lr * response_mask                      # [B, T]
    prefix_sum  = torch.cumsum(masked_lr, dim=-1)                # [B, T]
    t_count     = torch.cumsum(response_mask, dim=-1).clamp_min(1.0)  # [B, T]
    prefix_mean = prefix_sum / t_count                           # [B, T] = mean_prefix(s * log-ratio)

    # ----- double-sided margin logits and loss -----
    # The margin is not multiplied by sign, which creates symmetric thresholds.
    logits = beta * prefix_mean - margin.unsqueeze(-1)           # [B, T]
    per_tok_loss = -F.logsigmoid(logits)                         # [B, T]  = softplus(margin - beta * s * mean)

    # ----- average over valid response tokens only -----
    denom_tok = response_mask.sum(dim=-1).clamp_min(1.0)         # [B]
    loss_per_seq = (per_tok_loss * response_mask).sum(dim=-1) / denom_tok  # [B]

    # ----- prompt-level normalized DLW is applied as a direct per-sample multiplier -----
    loss = (loss_per_seq * dlw_weight).mean()                    # scalar

    return loss


def build_prompt_local_dpo_pairs(
    acc: torch.Tensor,
    prompt_group_id: torch.Tensor,
    sample_rank_in_group: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    acc = acc.float().view(-1)
    prompt_group_id = prompt_group_id.view(-1)
    sample_rank_in_group = sample_rank_in_group.view(-1)

    if not (acc.shape == prompt_group_id.shape == sample_rank_in_group.shape):
        raise ValueError(
            "acc, prompt_group_id, and sample_rank_in_group must have the same flattened shape. "
            f"Got {tuple(acc.shape)}, {tuple(prompt_group_id.shape)}, and {tuple(sample_rank_in_group.shape)}."
        )

    unique_prompt_ids = torch.unique(prompt_group_id, sorted=True)
    chosen_indices = []
    rejected_indices = []
    paired_prompt_ids = []
    valid_prompt_count = 0
    skipped_prompt_count = 0

    for prompt_id in unique_prompt_ids:
        group_indices = torch.nonzero(prompt_group_id == prompt_id, as_tuple=False).flatten()
        group_indices = group_indices[torch.argsort(sample_rank_in_group[group_indices])]

        positive_indices = group_indices[acc[group_indices] > 0.5]
        negative_indices = group_indices[acc[group_indices] < 0.5]
        pair_count = min(positive_indices.numel(), negative_indices.numel())
        if pair_count == 0:
            skipped_prompt_count += 1
            continue

        valid_prompt_count += 1
        chosen_indices.append(positive_indices[:pair_count])
        rejected_indices.append(negative_indices[:pair_count])
        paired_prompt_ids.append(prompt_id.repeat(pair_count))

    if chosen_indices:
        chosen_indices = torch.cat(chosen_indices, dim=0)
        rejected_indices = torch.cat(rejected_indices, dim=0)
        paired_prompt_ids = torch.cat(paired_prompt_ids, dim=0)
    else:
        chosen_indices = torch.empty(0, dtype=torch.long, device=acc.device)
        rejected_indices = torch.empty(0, dtype=torch.long, device=acc.device)
        paired_prompt_ids = torch.empty(0, dtype=prompt_group_id.dtype, device=prompt_group_id.device)

    return {
        "chosen_indices": chosen_indices,
        "rejected_indices": rejected_indices,
        "paired_prompt_ids": paired_prompt_ids,
        "pair_count": int(chosen_indices.numel()),
        "valid_prompt_count": valid_prompt_count,
        "skipped_prompt_count": skipped_prompt_count,
        "total_prompt_count": int(unique_prompt_ids.numel()),
    }


def compute_dporm_loss(
    token_level_scores: torch.Tensor,
    acc: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_group_id: torch.Tensor,
    sample_rank_in_group: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | float]]:
    pair_info = build_prompt_local_dpo_pairs(
        acc=acc,
        prompt_group_id=prompt_group_id,
        sample_rank_in_group=sample_rank_in_group,
    )

    zero_loss = token_level_scores.sum() * 0.0
    if pair_info["pair_count"] == 0:
        pair_info["pair_logits"] = token_level_scores.new_empty((0,))
        pair_info["pair_loss_mean"] = 0.0
        return zero_loss, pair_info

    sequence_scores = (token_level_scores * response_mask).sum(dim=1)
    chosen_scores = sequence_scores[pair_info["chosen_indices"]]
    rejected_scores = sequence_scores[pair_info["rejected_indices"]]
    pair_logits = beta * (chosen_scores - rejected_scores)
    pair_loss = -F.logsigmoid(pair_logits)
    loss_sum = pair_loss.sum()

    pair_info["pair_logits"] = pair_logits.detach()
    pair_info["pair_loss_mean"] = pair_loss.detach().mean().item()
    return loss_sum, pair_info


def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (
            token_level_scores[start_id : start_id + n_samples] * response_mask[start_id : start_id + n_samples]
        ).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id : start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        if cur_acc_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (
                ((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()
            ).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()


def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
