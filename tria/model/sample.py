import math
import torch
import torch.nn.functional as F

from typing import Iterable, Union, Optional
import numpy as np
from numpy.random import RandomState

from .mask import cosine_schedule, format_seed

################################################################################
# Utilities for sampling from trained TRIA model
################################################################################


def top_p_top_k(
    logits: torch.Tensor, 
    top_p: float = None, 
    top_k: int = None,
):
    """
    Adapted from `vampnet.modules.transformer.sample_from_logits` by Hugo Flores
    Garcia. See: https://github.com/hugofloresgarcia/vampnet/
    
    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., n_classes)
    """
    logits = logits.clone()
    n_classes = logits.shape[-1]

    # Mask logits outside top-k by setting to -inf
    if top_k is not None and 0 < top_k < n_classes:
        thresh = logits.topk(top_k, dim=-1).values[..., -1:]  # (..., 1)
        logits[logits < thresh] = float("-inf")

    # Mask logits outside top-p by setting to -inf
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort descending
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)   # (..., n_classes)
        sorted_probs = F.softmax(sorted_logits, dim=-1)                    # (..., n_classes)
        cumsum = sorted_probs.cumsum(dim=-1)                               # (..., n_classes)

        # Keep at least one logit
        to_remove = cumsum > top_p
        to_remove[..., 0] = False
        remove_idx = torch.zeros_like(to_remove).scatter(-1, sorted_idx, to_remove)
        logits[remove_idx] = float("-inf")
        
    return logits


def sample(
    logits: torch.Tensor,
    temp: float,
    argmax: bool = False,
):
    """
    Adapted from `vampnet.modules.transformer.sample_from_logits` by Hugo Flores
    Garcia. See: https://github.com/hugofloresgarcia/vampnet/
    
    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., n_classes)

    Returns
    -------
    torch.Tensor
        Sampled tokens, shape of `logits` with trailing `n_classes` dimension
        removed
    torch.Tensor
        Probabilities of sampled tokens, shape of `logits` with trailing 
        `n_classes` dimension removed
    """
    if temp <= 0:
        argmax = True
        temp = 1.0

    if argmax:
        sampled = logits.argmax(dim=-1)
        probs = F.softmax(
            logits, dim=-1
        ).take_along_dim(sampled.unsqueeze(-1), dim=-1).squeeze(-1)
        return sampled, probs

    probs = F.softmax(logits / temp, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    draws = torch.multinomial(flat, 1).squeeze(-1)
    sampled = draws.view(*probs.shape[:-1])
    chosen = probs.take_along_dim(sampled.unsqueeze(-1), dim=-1).squeeze(-1)
    return sampled, chosen


def mask_by_confidence(
    probs: torch.Tensor,
    n: torch.Tensor,
    temp: float,
    causal_bias: float,
    state: Iterable[RandomState],
    eligible: Optional[torch.Tensor] = None,
):
    """
    Re-mask predicted tokens in a single codebook such that `n` previously-
    masked tokens are left unmasked, using confidence (probability assigned to 
    tokens during sampling) to select which tokens remain. This confidence can 
    be mediated by random noise and a bias to unmask early (leftward) positions 
    first.

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities assigned to sampled tokens, shape (n_batch, n_frames)
    n : torch.Tensor
        Target number of unmasked tokens, shape (n_batch,)
    temp : float
        Mask temperature, corresponding to randomness in unmasking process
    causal_bias : float
        Bias towards unmasking early (leftward) token positions first; typically 
        in (0, 1]. Note that large values of `temp` can effectively "wash out"
        this causal bias
    state : Iterable[RandomState]
        Random seeds for reproducibility
    eligible : torch.Tensor
        Optional indicator for positions eligible for unmasking, shape (n_batch, n_frames)
    
    """
    
    n_batch, n_frames = probs.shape
    device = probs.device

    if eligible is None:
        eligible = torch.isfinite(probs) & (probs > 0)
    else:
        eligible = eligible.to(torch.bool)

    # Masked token count and target
    n_masked = eligible.long().sum(dim=-1)
    n_unmask = (n_masked - n).clamp_min(0)

    # Gumbel noise to introduce randomness into unmasking
    u = torch.stack([
        torch.from_numpy(s.uniform(1e-6, 1 - 1e-6, n_frames)) for s in state
    ], dim=0).to(probs)
    gumbel = -torch.log(-torch.log(u))

    # Log-confidences + noise
    s = probs.clamp_min(1e-12)
    confs = torch.log(s) + temp * gumbel

    # Optional causal bias in log-domain
    if causal_bias > 0:
        frame_relpos = (1 - (torch.arange(n_frames, device=device, dtype=confs.dtype) + 1) / n_frames).view(1, -1)
        confs = confs + causal_bias * frame_relpos

    # Only eligible positions can be chosen
    confs_masked = confs.masked_fill(~eligible, float("-inf"))
    sorted_vals, sorted_idx = confs_masked.sort(dim=-1, descending=True)
    rank = torch.arange(n_frames, device=device).view(1, n_frames).expand_as(confs_masked)
    k = n_unmask.view(n_batch, 1)
    pick_sorted = rank < k
    pick = torch.zeros_like(pick_sorted, dtype=torch.bool).scatter(-1, sorted_idx, pick_sorted)

    # Return tokens_mask semantics (True = unmasked/keep)
    mask = ~(eligible & (~pick))
    return mask

