import torch
import numpy as np
import verl.utils.torch_functional as verl_F
from verl import DataProto
from sklearn.metrics import roc_curve
from .reward_signal_utils import (
    build_candidate_token_mask,
)


def build_candidate_token_mask_from_batch(data: DataProto, config):
    response_length = data.batch["old_log_prob_topk_values"].shape[1]
    response_mask = data.batch["attention_mask"][:, -response_length:].bool()
    candidate_token_mask = build_candidate_token_mask(
        old_log_prob_topk_values=data.batch["old_log_prob_topk_values"],
        response_mask=response_mask,
        min_probability=config.candidate_min_p,
        top_p=config.candidate_top_p,
    )
    return DataProto.from_dict(tensors={"candidate_token_mask": candidate_token_mask})


def compute_logprob_advantage_correlation(logp, adv, candidate_token_mask):
    # mask
    logp = logp[candidate_token_mask]
    adv = adv[candidate_token_mask]
    
    # mean
    u = logp.mean()
    v = adv.mean()
    
    # covariance
    cov = ((logp - u) * (adv - v)).mean()
    
    # standard deviations
    std_logp = logp.std()
    std_adv = adv.std()
    
    # avoid division by zero
    if std_logp == 0 or std_adv == 0:
        return 0.0
    
    # correlation
    corr = cov / (std_logp * std_adv)
    return corr

def find_best_threshold(pos_scores: list[float], neg_scores: list[float]) -> float:
    # Build binary labels and score values for threshold selection.
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores

    # Compute the ROC curve and choose the best Youden-J point.
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Youden's J statistic: TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    return float(best_threshold)


def masked_corrcoef(x, y, mask):
    x_masked = x[mask]
    y_masked = y[mask]
    if x_masked.numel() < 2:
        return torch.tensor(float('nan'))  # torch.corrcoef needs at least two values.
    return torch.corrcoef(torch.stack((x_masked, y_masked)))[0, 1].detach()


def calculate_tok_metric(token_level_reward, response_mask, accs, metric):
    mask = response_mask
    float_mask = response_mask.float()

    cnt = torch.cumsum(float_mask, dim=-1)  # Valid token count through each position.
    cnt[~response_mask] = 1
    tok_cumsum = torch.cumsum(token_level_reward * float_mask, dim=-1)

    # Divide only on valid response positions.
    tok_cumsummean = torch.zeros_like(tok_cumsum)
    tok_cumsummean[mask] = (tok_cumsum / cnt)[mask]
    tok_sigmoidcumsummean = torch.sigmoid(tok_cumsummean)

    # Track how token-level reward summaries correlate with correctness.
    metric["tok_logpratio_cor"] = masked_corrcoef(token_level_reward, accs, mask)
    metric["tok_cumsum_cor"] = masked_corrcoef(tok_cumsum, accs, mask)
    metric["tok_cumsummean_cor"] = masked_corrcoef(tok_cumsummean, accs, mask)
    metric["tok_sigmoidcumsummean_cor"] = masked_corrcoef(tok_sigmoidcumsummean, accs, mask)

    return cnt, tok_cumsum, tok_cumsummean, tok_sigmoidcumsummean, metric


def calculate_seq_metric(token_level_reward, response_mask, acc, metric):
    seq_reward_sum = verl_F.masked_sum(token_level_reward, response_mask, axis=-1)
    seq_reward_mean = verl_F.masked_mean(token_level_reward, response_mask, axis=-1)
    seq_mask = response_mask.sum(-1) > 0 
    metric["seq_sum_cor"] = masked_corrcoef(seq_reward_sum, acc, seq_mask)
    metric["seq_mean_cor"] = masked_corrcoef(seq_reward_mean, acc, seq_mask)

    return seq_reward_sum, seq_reward_mean, metric

def _compute_grpo_advantages(
    outcome_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    n_samples: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    grpo_advantages = torch.zeros_like(response_mask, dtype=torch.float32)

    for start_pos in range(0, outcome_rewards.shape[0], n_samples):
        end_pos = min(start_pos + n_samples, outcome_rewards.shape[0])
        group_rewards = outcome_rewards[start_pos:end_pos].float()
        group_mean = group_rewards.mean()
        group_std = group_rewards.std(unbiased=False).clamp_min(eps)
        normalized_group_rewards = (group_rewards - group_mean) / group_std
        grpo_advantages[start_pos:end_pos] = normalized_group_rewards.unsqueeze(-1).expand(
            -1, response_mask.shape[1]
        )

    return grpo_advantages * response_mask.float()


def _compute_masked_gae(
    token_td_advantages: torch.Tensor,
    token_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    gae_advantages = torch.zeros_like(token_td_advantages)
    lastgaelam = torch.zeros(token_td_advantages.shape[0], device=token_td_advantages.device, dtype=token_td_advantages.dtype)

    for timestep in reversed(range(token_td_advantages.shape[1])):
        lastgaelam = token_td_advantages[:, timestep] + gamma * lam * lastgaelam
        lastgaelam = torch.where(
            token_mask[:, timestep],
            lastgaelam,
            torch.zeros_like(lastgaelam),
        )
        gae_advantages[:, timestep] = lastgaelam

    return gae_advantages


def compute_distrl_advantages_and_returns(data: DataProto, response_mask: torch.Tensor, n_samples, config):
    candidate_td_advantages = data.batch["candidate_td_advantages"].float()
    response_mask = response_mask.bool()

    # `top1` is forced to be the actually sampled token, so it provides the sampled-token TD signal.
    sampled_token_td_advantages = candidate_td_advantages[:, :, 0]

    # Appendix C: reconstruct prefix values along the sampled path and use their minibatch std
    # to normalize all candidate TD advantages.
    prefix_values = torch.cumsum(sampled_token_td_advantages, dim=-1)
    prefix_values = torch.where(response_mask, prefix_values, torch.zeros_like(prefix_values))
    valid_prefix_values = prefix_values[response_mask]
    if valid_prefix_values.numel() > 1:
        prefix_value_std = valid_prefix_values.std(unbiased=False)
    else:
        prefix_value_std = prefix_values.new_tensor(1.0)
    prefix_value_std = prefix_value_std.clamp_min(1e-6)

    # Token-level GAE uses centered sampled-token TD advantages.
    sampled_td_mean = sampled_token_td_advantages[response_mask].mean()
    sampled_td_advantages_for_gae = (sampled_token_td_advantages - sampled_td_mean) / prefix_value_std

    # Distribution-level/top-k policy loss uses uncentered, std-scaled candidate TD advantages.
    scaled_candidate_td_advantages = candidate_td_advantages / prefix_value_std

    # Token-level adv. by GAE.
    gae_advantages = _compute_masked_gae(
        token_td_advantages=sampled_td_advantages_for_gae,
        token_mask=response_mask,
        gamma=float(getattr(config.algorithm, "gamma", 1.0)),
        lam=float(getattr(config.algorithm, "lam", 1.0)),
    )
    gae_advantages = verl_F.masked_whiten(gae_advantages, response_mask)

    # Seq-level adv by Verifiable Rewards.
    grpo_advantages = _compute_grpo_advantages(
        outcome_rewards=data.batch["acc"].float(),
        response_mask=response_mask,
        n_samples=n_samples,
    )
    
    # Main paper Eq. (13): combine GRPO-style group-normalized outcome signal with token-level GAE.
    advantages = (grpo_advantages * config.algorithm.reward_gt_coef + gae_advantages * config.algorithm.reward_gae_coef) * response_mask.float()

    # Correlation between advantages and accs for debugging.
    acc_expanded = data.batch["acc"].float().unsqueeze(-1).expand_as(grpo_advantages)
    cor_td_acc = masked_corrcoef(scaled_candidate_td_advantages[:, :, 0], acc_expanded, response_mask)
    cor_gae_acc = masked_corrcoef(gae_advantages, acc_expanded, response_mask)
    cor_grpo_acc = masked_corrcoef(grpo_advantages, acc_expanded, response_mask)
    cor_tok_acc = masked_corrcoef(advantages, acc_expanded, response_mask)
    advantage_metrics = {
        "advantage/cor_td_acc": cor_td_acc.detach().item(),
        "advantage/cor_gae_acc": cor_gae_acc.detach().item(),
        "advantage/cor_grpo_acc": cor_grpo_acc.detach().item(),
        "advantage/cor_tok_acc": cor_tok_acc.detach().item(),
    }

    return advantages, scaled_candidate_td_advantages, advantage_metrics

def compute_distrl_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    candidate_token_mask,
    cliprange,
    clip_ratio_low,
    clip_ratio_high,
    policy_log_prob_topk_values,
    candidate_td_advantages,
    old_log_prob_topk_values,
    clip_ratio_c=3.0):
    
    # breakpoint()
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    pg_losses = -advantages * ratio

    if clip_ratio_low is None:
        clip_ratio_low = cliprange
    if clip_ratio_high is None:
        clip_ratio_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    clip_pg_losses1 = torch.max(pg_losses, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses2, pg_losses3) * (advantages < 0).float(), eos_mask)
    # We only apply the dual-clip when the advantage is negative.
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    # step1: DistRL KLD
    topk = policy_log_prob_topk_values.shape[2]
    mask_adv = candidate_token_mask
    topk_ppo_kl = verl_F.masked_mean(torch.exp(old_log_prob_topk_values) * (old_log_prob_topk_values - policy_log_prob_topk_values),mask_adv)
    
    # step2: DistRL policy loss
    topk_prob_ratio = torch.exp(policy_log_prob_topk_values - old_log_prob_topk_values)
    topk_pg_losses = - candidate_td_advantages * topk_prob_ratio
    topk_pg_losses2 = - candidate_td_advantages * torch.clamp(topk_prob_ratio, 1.0 - cliprange, 1.0 + cliprange)
    topk_clip_pg_losses1 = torch.max(topk_pg_losses, topk_pg_losses2)
    topk_pg_clipfrac = verl_F.masked_mean(torch.gt(topk_pg_losses2, topk_pg_losses).float(), mask_adv)
    topk_pg_losses3 = -candidate_td_advantages * clip_ratio_c
    topk_clip_pg_losses2 = torch.min(topk_pg_losses3, topk_clip_pg_losses1)
    topk_pg_clipfrac_lower = verl_F.masked_mean(torch.gt(topk_clip_pg_losses2, topk_pg_losses3) * (candidate_td_advantages < 0).float(), mask_adv) # TODO
    topk_pg_losses = torch.where(candidate_td_advantages < 0, topk_clip_pg_losses2, topk_clip_pg_losses1)

    # Compute the expectation over the full vocabulary.
    old_prob = torch.exp(old_log_prob_topk_values[mask_adv])
    topk_pg_loss = (topk_pg_losses[mask_adv] * old_prob).sum() / old_prob.sum()
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, topk_pg_loss, topk_pg_clipfrac, topk_ppo_kl, topk_pg_clipfrac_lower
