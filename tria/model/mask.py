from typing import Iterable
from typing import Union

import torch
from audiotools.core.util import random_state

################################################################################
# Utilities for masked language modeling
################################################################################


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Map timestep in [0, 1] to masking ratio in (0, 1] via cosine schedule
    proposed by Chang et al. in "MaskGIT: Masked generative image
    transformer" (2022).

    Parameters
    ----------
    t : torch.Tensor
        Timestep in [0, 1]

    Returns
    -------
    torch.Tensor
        Mask proportion in (0, 1]
    """
    return (t * torch.pi / 2).cos().clamp(1e-10, 1.0)


def format_seed(seed):
    if isinstance(seed, (int, float)):
        seed = [seed]
    elif isinstance(seed, torch.Tensor):
        seed = seed.tolist()
    elif isinstance(seed, Iterable):
        pass
    else:
        raise ValueError(f"Invalid random seed of type {type(seed)}")

    return [random_state(s) for s in seed]


def get_span_mask(
    tokens: torch.Tensor,
    min_prop: float,
    max_prop: float,
    seed: Union[int, Iterable[int]],
) -> torch.Tensor:
    """
    Mask a random span of consecutive frames across all codebooks, varying
    across batch.

    Parameters
    ----------
    tokens : torch.Tensor
        Tokens to be masked, shape (n_batch, n_codebooks, n_frames)
    min_prop : float
        Minimum proportion of frames to mask
    max_prop : float
        Maximum proportion of frames to mask
    seed : Iterable[int]
        One or more random seeds to determine masks

    Returns
    -------
    torch.Tensor
        Mask of shape (n_batch, n_frames)
    """
    assert min_prop >= 0.0
    assert max_prop <= 1.0

    n_batch, n_codebooks, n_frames = tokens.shape

    states = format_seed(seed)
    assert len(states) == n_batch

    mask = torch.ones(
        n_batch,
        n_frames,
        device=tokens.device,
        dtype=torch.bool,
    )  # (n_batch, n_frames)

    for i, s in enumerate(states):
        prop = s.uniform(min_prop, max_prop) if min_prop < max_prop else min_prop

        if prop >= 1.0:
            mask[i] = False
        else:
            span = int(prop * n_frames)
            st = s.randint(0, max(n_frames - span, 1))
            mask[i, st : st + span] = False

    return mask


def get_current_codebook_mask(
    tokens: torch.Tensor, codebooks: torch.Tensor
) -> torch.Tensor:
    """
    Given tokens and batch of selected codebooks, mask all codebooks "above" and
    "below" selected codebooks.

    Parameters
    ----------
    tokens : torch.Tensor
        Tokens to be masked, shape (n_batch, n_codebooks, n_frames)
    codebooks : torch.Tensor
        Selected codebooks "above" which tokens should be masked, shape
        (n_batch,)

    Returns
    -------
    torch.Tensor
        Mask of shape (n_batch, n_codebooks)
    """

    n_batch, n_codebooks, n_frames = tokens.shape

    assert codebooks.ndim == 1
    assert codebooks.shape[0] in [1, n_batch]
    codebooks = codebooks.repeat(n_batch // codebooks.shape[0])

    mask = (
        torch.arange(
            n_codebooks,
            dtype=codebooks.dtype,
            device=codebooks.device,
        )[None, :]
        == codebooks[:, None]
    )  # (n_batch, n_codebooks)

    return mask


def get_next_codebooks_mask(
    tokens: torch.Tensor, codebooks: torch.Tensor
) -> torch.Tensor:
    """
    Given tokens and batch of selected codebooks, mask all codebooks "above"
    selected codebooks.

    Parameters
    ----------
    tokens : torch.Tensor
        Tokens to be masked, shape (n_batch, n_codebooks, n_frames)
    codebooks : torch.Tensor
        Selected codebooks "above" which tokens should be masked, shape
        (n_batch,)

    Returns
    -------
    torch.Tensor
        Mask of shape (n_batch, n_codebooks)
    """

    n_batch, n_codebooks, n_frames = tokens.shape

    assert codebooks.ndim == 1
    assert codebooks.shape[0] in [1, n_batch]
    codebooks = codebooks.repeat(n_batch // codebooks.shape[0])

    mask = (
        torch.arange(
            n_codebooks,
            dtype=codebooks.dtype,
            device=codebooks.device,
        )[None, :]
        <= codebooks[:, None]
    )  # (n_batch, n_codebooks)

    return mask


def get_random_mask(
    tokens: torch.Tensor,
    prop: Union[float, Iterable[float]],
    seed: Union[int, Iterable[int]],
) -> torch.Tensor:
    """
    Parameters
    ----------
    tokens : torch.Tensor
        Tokens to be masked, shape (n_batch, n_codebooks, n_frames)
    prop : torch.Tensor
        Proportion of tokens to be masked, shape (n_batch,)
    seed : Iterable[int]
        One or more random seeds to determine masks

    Returns
    -------
    torch.Tensor
        Random mask of shape (n_batch, n_codebooks, n_frames)
    """
    n_batch, n_codebooks, n_frames = tokens.shape

    if isinstance(prop, torch.Tensor):
        prop = prop.tolist()
    assert len(prop) == n_batch

    states = format_seed(seed)
    assert len(states) == n_batch

    mask = torch.ones(
        n_batch,
        n_codebooks,
        n_frames,
        device=tokens.device,
        dtype=torch.bool,
    )  # (n_batch, n_codebooks, n_frames)

    for i, (s, p) in enumerate(zip(states, prop)):
        mask[i] = torch.from_numpy(s.rand(n_codebooks, n_frames)).to(mask.device) > p

    return mask


def combine_masks(
    mask_span: torch.Tensor,
    mask_current_codebook: torch.Tensor,
    mask_next_codebooks: torch.Tensor,
    mask_random: torch.Tensor,
    leak: bool = False,
) -> torch.Tensor:
    """
    Combine sampled masks to allow for application to token buffer.

    Parameters
    ----------
    mask_span : torch.Tensor
        Shape (n_batch, n_frames)
    mask_current_codebook : torch.Tensor
        Shape (n_batch, n_codebooks)
    mask_next_codebooks : torch.Tensor
        Shape (n_batch, n_codebooks)
    mask_random : torch.Tensor
        Shape (n_batch, n_codebooks, n_frames)

    Returns
    -------
    torch.Tensor
        Combined mask, shape (n_batch, n_codebooks, n_frames)
    torch.Tensor
    """

    mask_current_level = mask_current_codebook[:, :, None] & (~mask_random)

    if leak:
        # Allow leakage from "higher" codebooks inside masked span
        higher = (~mask_next_codebooks[:, :, None]) & (~mask_random)
    else:
        # Strictly mask "higher" codebooks inside masked span
        higher = ~mask_next_codebooks[:, :, None]

    # Inside span, unmask everything except "higher" codebooks and masked
    # positions in current codebook
    mask = ~(higher | mask_current_level)

    # Outside span, fully unmask
    mask = mask | mask_span[:, None, :]

    return mask
