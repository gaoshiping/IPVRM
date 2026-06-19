import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.distributed as dist
from torch import Tensor
from verl.utils.ulysses import slice_input_tensor


def _pad_tensor_nd(x: torch.Tensor, dim: int, pad_len: int, value: int = 0) -> torch.Tensor:
    """
    Pad `pad_len` copies of `value` on the right side of dimension `dim`.

    Works for tensors with any rank.
    """
    if pad_len == 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad_tensor), dim=dim)


def chunked_logprobs_from_logits(logits: torch.FloatTensor, labels: torch.LongTensor, batch_size: int = 64):
    """
    Memory-efficient logprob computation from logits.
    
    Supports:
    - logits: [..., V]
    - labels: [...], or [..., K]
    """
    logits = logits.contiguous()
    labels = labels.contiguous()
    
    # Ensure labels have at least 1 trailing dim (e.g., [B] -> [B, 1])
    if labels.dim() == logits.dim() - 1:
        labels = labels.unsqueeze(-1)  # [..., 1]

    orig_shape = labels.shape  # [..., K]
    *prefix_dims, vocab_size = logits.shape
    flat_batch = int(torch.tensor(prefix_dims).prod().item())  # B
    label_suffix = labels.shape[-1]  # K

    logits_flat = logits.view(flat_batch, vocab_size)          # [B, V]
    labels_flat = labels.view(flat_batch, label_suffix)        # [B, K]

    if logits.dtype in [torch.float32, torch.float64]:
        logsumexp = torch.logsumexp(logits_flat, dim=-1, keepdim=True)  # [B, 1]
        logits_for_labels = torch.gather(logits_flat, dim=1, index=labels_flat)  # [B, K]
        logprobs = logits_for_labels - logsumexp  # [B, K]
    else:
        logprobs_chunks = []
        for i in range(0, flat_batch, batch_size):
            logits_batch = logits_flat[i:i + batch_size]  # [b, V]
            labels_batch = labels_flat[i:i + batch_size]  # [b, K]
            logprobs_batch = F.log_softmax(logits_batch, dim=-1)  # [b, V]
            selected_logprobs = torch.gather(logprobs_batch, dim=1, index=labels_batch)  # [b, K]
            logprobs_chunks.append(selected_logprobs)
        logprobs = torch.cat(logprobs_chunks, dim=0)  # [B, K]

    logprobs = logprobs.view(orig_shape)  # match input label shape
    logprobs = logprobs.squeeze(-1)
    return logprobs


def ulysses_pad_and_slice_inputs(
        input_ids_rmpad: Tensor,                       # 2-D or 3-D: [B, S]  or  [B, S, K]
        position_ids_rmpad: Optional[Tensor] = None,   # 2-D:        [1, S]  (unchanged)
        sp_size: int = 1
) -> Tuple[Tensor, Optional[Tensor], int]:
    """
    Supports both standard token ids and DistRL top-k candidate ids:
      * [bsz, seqlen]
      * [bsz, seqlen, topk]

    The same sequence-parallel padding rule is applied in both cases.
    """
    if position_ids_rmpad is not None:
        # Position ids stay 2-D, while candidate token ids may be 3-D.
        assert position_ids_rmpad.size(0) == 1
        assert input_ids_rmpad.size(1) == position_ids_rmpad.size(1)

    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0

    seqlen = input_ids_rmpad.size(1)
    pad_size = (sp_size - seqlen % sp_size) % sp_size
    if pad_size > 0:
        # Pad the sequence dimension while preserving all remaining dimensions.
        input_ids_rmpad = _pad_tensor_nd(input_ids_rmpad, dim=1, pad_len=pad_size, value=0)

        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(
                pad_size, device=position_ids_rmpad.device).unsqueeze(0)  # [1, pad]
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)

    # Slice along the sequence dimension; 3-D inputs remain [B, S/sp, K].
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)

    if position_ids_rmpad is not None:
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)

    return input_ids_rmpad, position_ids_rmpad, pad_size
