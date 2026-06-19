"""
Single Process Actor
"""

import itertools
import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from .utils import chunked_logprobs_from_logits
from .dl_core_algos import compute_distrl_policy_loss


class BasePPOActor:
    def __init__(self, config):
        self.config = config

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = [
    "DataParallelDistRLActor",
]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelDistRLActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # SUPPORT DISTRIBUTION-LEVEL ADV
        self.topk = config.distrl_topk

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)


                # SUPPORT DISTRIBUTION-LEVEL ADV.
                if "old_log_prob_topk_indices" in micro_batch:
                    log_prob_topk_indices = micro_batch["old_log_prob_topk_indices"]
                    log_prob_topk_indices_rmpad, *_ = unpad_input(log_prob_topk_indices, attention_mask)
                    log_prob_topk_indices_rmpad = log_prob_topk_indices_rmpad.unsqueeze(0)  # log_prob_topk_indices_rmpad (1, total_nnz, topk)
                    log_prob_topk_indices_rmpad_rolled = torch.roll(log_prob_topk_indices_rmpad, shifts=-1, dims=1)  # (1, total_nnz, topk)

                    if self.use_ulysses_sp:
                        log_prob_topk_indices_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(log_prob_topk_indices_rmpad_rolled, None, self.ulysses_sequence_parallel_size)
                    log_prob_topk_indices_rmpad_rolled = log_prob_topk_indices_rmpad_rolled.squeeze(0)

                else:
                    # log_prob_topk_indices_rmpad_rolled = torch.topk(logits_rmpad, k=self.topk, dim=-1, largest=True, sorted=True).indices
                    log_prob_topk_indices_rmpad_rolled = topk_with_forced_top1(
                        logits=logits_rmpad,
                        input_ids_rolled=input_ids_rmpad_rolled, 
                        k=self.topk)

                log_prob_topk_values_rmpad = chunked_logprobs_from_logits(logits_rmpad, log_prob_topk_indices_rmpad_rolled)
                if self.use_ulysses_sp:
                    log_prob_topk_values_rmpad = gather_outpus_and_unpad(log_prob_topk_values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    log_prob_topk_indices_rmpad_rolled = gather_outpus_and_unpad(log_prob_topk_indices_rmpad_rolled, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                
                if log_prob_topk_values_rmpad.dim() < 2:
                    log_prob_topk_values_rmpad = log_prob_topk_values_rmpad.unsqueeze(-1)

                log_prob_topk_values = pad_input(hidden_states=log_prob_topk_values_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)[:, -response_length - 1:-1]
                log_prob_topk_indices = pad_input(hidden_states=log_prob_topk_indices_rmpad_rolled, indices=indices, batch=batch_size, seqlen=seqlen)
                log_prob_topk_indices = torch.roll(log_prob_topk_indices, shifts=1, dims=1) # unrolled

            else:  # not using rmpad and no ulysses sp
                raise NotImplementedError
            return log_probs, log_prob_topk_values, log_prob_topk_indices, entropy

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> tuple[list, list | None]:
            # selected keys
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            if 'old_log_prob_topk_indices' in data.batch.keys():
                select_keys.extend(['old_log_prob_topk_indices'])
            batch = data.select(batch_keys=select_keys).batch
            
            # Delete some multi-modal support
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        log_prob_topk_values_lst = []
        log_prob_topk_indices_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                log_probs, log_prob_topk_values, log_prob_topk_indices, entropy = \
                self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            log_prob_topk_values_lst.append(log_prob_topk_values)
            log_prob_topk_indices_lst.append(log_prob_topk_indices)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        log_prob_topk_values = torch.concat(log_prob_topk_values_lst, dim=0)
        log_prob_topk_indices = torch.concat(log_prob_topk_indices_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            log_prob_topk_values = log_prob_topk_values[revert_indices]
            log_prob_topk_indices = log_prob_topk_indices[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, log_prob_topk_values, log_prob_topk_indices, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ['acc', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_prob', 'advantages', 
               'normalized_candidate_td_advantages',
               'old_log_prob_topk_indices', 'old_log_prob_topk_values', 'candidate_token_mask']
        select_keys = [k for k in select_keys if k in data.batch.keys()]
        
        if self.config.use_kl_loss:
            select_keys.extend(["ref_log_prob", "ref_log_prob_topk_values"])
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    
                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    micro_batch_metrics = {}

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload
                    old_log_prob = data["old_log_prob"]
                    advantages = data["advantages"]

                    # SUPPORT DISTRIBUTION-LEVEL ADV.
                    attention_mask = data['attention_mask']
                    responses = data['responses']
                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]
                    normalized_candidate_td_advantages = data["normalized_candidate_td_advantages"]
                    old_log_prob_topk_values = data['old_log_prob_topk_values']
                    candidate_token_mask = data["candidate_token_mask"]
                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    log_prob, log_prob_topk_values, log_prob_topk_indices, entropy = \
                        self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, \
                    topk_pg_loss, topk_pg_clipfrac, topk_ppo_kl, topk_pg_clipfrac_lower = \
                    compute_distrl_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        eos_mask=response_mask,
                        candidate_token_mask=candidate_token_mask,
                        cliprange=clip_ratio,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        policy_log_prob_topk_values=log_prob_topk_values,
                        candidate_td_advantages=normalized_candidate_td_advantages,
                        old_log_prob_topk_values=old_log_prob_topk_values,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics.update({"actor/entropy_loss": entropy_loss.detach().item()})

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # Add NLL loss
                    nll_loss_coef = self.config.get('nll_loss_coef', 0)
                    if nll_loss_coef > 0 and data["acc"].sum().item() > 0:
                        nll_loss = - log_prob[(response_mask * data["acc"].unsqueeze(-1)).bool()].mean() # debug nll loss
                    else:
                        nll_loss = 0
                        
                    # compute policy loss
                    pg_loss_coef = self.config.get('pg_loss_coef', 1)
                    distrl_loss_coef = self.config.distrl_loss_coef
                    policy_loss = pg_loss * pg_loss_coef + distrl_loss_coef * topk_pg_loss + nll_loss_coef * nll_loss
                    # breakpoint()
                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        ref_log_prob_topk_values = data["ref_log_prob_topk_values"]

                        # compute kl loss
                        tok_level_kl, dis_level_kl, metric = topk_kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            tok_mask=response_mask,
                            policy_log_prob_topk_values=log_prob_topk_values,
                            ref_log_prob_topk_values=ref_log_prob_topk_values,
                            old_log_prob_topk_values=old_log_prob_topk_values,
                            threshold=self.config.kl_threshold,
                        )
                        tok_level_kl_loss = agg_loss(loss_mat=tok_level_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        dis_level_kl_loss = agg_loss(loss_mat=dis_level_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + dis_level_kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = dis_level_kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        micro_batch_metrics.update(metric)

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),

                            # SUPPORT DISTRIBUTION-LEVEL ADV.
                            "actor/topk_pg_loss": topk_pg_loss.detach().item(),
                            "actor/distrl_loss": topk_pg_loss.detach().item(),
                            "actor/topk_pg_clipfrac": topk_pg_clipfrac.detach().item(),
                            "actor/topk_ppo_kl": topk_ppo_kl.detach().item(),
                            "actor/candidate_ppo_kl": topk_ppo_kl.detach().item(),
                            "actor/topk_pg_clipfrac_lower": topk_pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics


def topk_with_forced_top1(logits: torch.Tensor, input_ids_rolled: torch.Tensor, k: int):
    """
    Performs top-k on logits, forcing input_ids_rolled to be the top-1 in each position.
    If input_ids_rolled is not in the top-k, it is inserted as top-1 and the rest shifted.

    Args:
        logits: Tensor of shape [seq_len, vocab_size] or [batch_size, seq_len, vocab_size]
        input_ids_rolled: Tensor of shape [seq_len] or [batch_size, seq_len]
        k: int, number of top-k values to return

    Returns:
        Tensor of shape [seq_len, k] or [batch_size, seq_len, k]
    """
    original_shape = logits.shape
    is_batched = logits.dim() == 3

    if not is_batched:
        logits = logits.unsqueeze(0)  # [1, seq_len, vocab_size]
        input_ids_rolled = input_ids_rolled.unsqueeze(0)  # [1, seq_len]

    batch_size, seq_len, vocab_size = logits.shape

    # Step 1: Get top-k indices
    topk = torch.topk(logits, k=k, dim=-1, largest=True, sorted=True)
    topk_indices = topk.indices  # [B, S, k]

    # Step 2: Check presence of input_ids_rolled in top-k
    input_ids_exp = input_ids_rolled.unsqueeze(-1)  # [B, S, 1]
    match_mask = (topk_indices == input_ids_exp)  # [B, S, k]
    found_mask = match_mask.any(dim=-1)  # [B, S]
    where_found = match_mask.float().argmax(dim=-1)  # [B, S]

    # Clone topk indices for result
    result = topk_indices.clone()

    # --- Case 1: Already in topk -> swap to top-1 ---
    swap_mask = found_mask
    swap_pos = where_found[swap_mask]    
    result[swap_mask, swap_pos], result[swap_mask, torch.zeros(swap_mask.sum(), dtype=torch.int)] = result[swap_mask, torch.zeros(swap_mask.sum(), dtype=torch.int)], result[swap_mask, swap_pos]
    # --- Case 2: Not in topk -> insert at position 0, shift others ---
    not_found_mask = ~found_mask

    # Only process non-found ones
    if not_found_mask.any().item():
        b_idx, s_idx = torch.where(not_found_mask)  # Indices of [B, S] where input_ids not in top-k

        # Shift old top-k right by one (discard last)
        result[b_idx, s_idx, 1:] = result[b_idx, s_idx, :-1]
        # Insert input_ids_rolled as new top-1
        result[b_idx, s_idx, 0] = input_ids_rolled[b_idx, s_idx]

    if not is_batched:
        result = result.squeeze(0)  # [seq_len, k]

    return result


def topk_kl_penalty(
        logprob: torch.Tensor,
        ref_logprob: torch.Tensor,
        tok_mask: torch.Tensor,
        policy_log_prob_topk_values: torch.Tensor,
        ref_log_prob_topk_values: torch.Tensor,
        old_log_prob_topk_values: torch.Tensor,
        dis_mask: torch.Tensor = None,
        threshold: float = 0.0
    ):
    # logprob, ref_logprob [batch-size, seq-len]
    # policy_log_prob_topk_values, ref_log_prob_topk_values [batch-size, seq-len, topk]
    # threshold: float
    

    tok_mask = tok_mask.bool()
    if dis_mask is None:
        dis_mask = tok_mask.unsqueeze(-1).repeat(1, 1, old_log_prob_topk_values.shape[-1])

    # caculate the logp-ratio
    tok_logpratio = logprob - ref_logprob
    dis_logpratio = policy_log_prob_topk_values - ref_log_prob_topk_values
    pol_dis_rest = torch.clamp(1 - torch.exp(policy_log_prob_topk_values).sum(-1), min=1e-5)
    ref_dis_rest = torch.clamp(1 - torch.exp(ref_log_prob_topk_values).sum(-1), min=1e-5)
    res_logpratio = pol_dis_rest.log() - ref_dis_rest.log()

    # log the logpratio
    metric = {}
    metric["kl/tok_logpratio_max"]    = tok_logpratio[tok_mask].max().item()
    metric["kl/tok_logpratio_min"]    = tok_logpratio[tok_mask].min().item()
    metric["kl/tok_logpratio_mean"]   = tok_logpratio[tok_mask].mean().item()
    metric["kl/tok_logpratio_median"] = tok_logpratio[tok_mask].median().item()
    metric["kl/tok_logpratio_099"]    = torch.quantile(tok_logpratio[tok_mask], 0.99).item()
    metric["kl/tok_logpratio_001"]    = torch.quantile(tok_logpratio[tok_mask], 0.01).item()
    metric["kl/dis_logpratio_max"]    = dis_logpratio[dis_mask].max().item()
    metric["kl/dis_logpratio_min"]    = dis_logpratio[dis_mask].min().item()
    metric["kl/dis_logpratio_mean"]   = dis_logpratio[dis_mask].mean().item()
    metric["kl/dis_logpratio_median"] = dis_logpratio[dis_mask].median().item()
    metric["kl/dis_logpratio_099"]    = torch.quantile(dis_logpratio[dis_mask], 0.99).item()
    metric["kl/dis_logpratio_001"]    = torch.quantile(dis_logpratio[dis_mask], 0.01).item()
    metric["kl/res_logpratio_max"]    = res_logpratio[tok_mask].max().item()
    metric["kl/res_logpratio_min"]    = res_logpratio[tok_mask].min().item()
    metric["kl/res_logpratio_mean"]   = res_logpratio[tok_mask].mean().item()
    metric["kl/res_logpratio_median"] = res_logpratio[tok_mask].median().item()
    metric["kl/res_logpratio_099"]    = torch.quantile(res_logpratio[tok_mask], 0.99).item()
    metric["kl/res_logpratio_001"]    = torch.quantile(res_logpratio[tok_mask], 0.01).item()

    # Do not constrain the update of the policy when `logprario < threshhold`
    tok_logpratio[tok_logpratio.abs() < threshold] = 0
    dis_logpratio[dis_logpratio.abs() < threshold] = 0
    res_logpratio[res_logpratio.abs() < threshold] = 0
    # calculate the token-level kl [batch-size, seq-len]
    kl = torch.clamp(- tok_logpratio, min=-20, max=20)
    ratio = torch.exp(kl)
    token_level_kl = (ratio - kl - 1).contiguous()
    token_level_kl = torch.clamp(token_level_kl, min=-10, max=10)
    # calculate the distribution-level kl [batch-size, seq-len]
    dist_topk_level_kl = (torch.exp(policy_log_prob_topk_values) * dis_logpratio)
    dist_topk_level_kl = dist_topk_level_kl.sum(-1) + (pol_dis_rest * res_logpratio)
    return token_level_kl, dist_topk_level_kl, metric
