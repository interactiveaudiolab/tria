import torch

################################################################################
# Utilities for positional encoding
################################################################################


def build_sinusoidal_cache(seq_len: int, n_channels: int, device, dtype):
    """
    Returns
    -------
    torch.Tensor
        Cache, shape (seq_len, n_channels)
    """
    assert n_channels % 2 == 0
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (seq_len, 1)
    i = torch.arange(n_channels // 2, device=device, dtype=dtype).unsqueeze(
        0
    )  # (1, n_channels/2)
    inv_freq = 1.0 / (10000 ** (i / (n_channels // 2)))
    ang = pos * inv_freq  # (seq_len, n_channels/2)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (seq_len, n_channels)
    return emb


def apply_sinusoidal(x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Shape (n_batch, n_heads, seq_len, head_channels) or (n_batch, seq_len, n_channels)
    cache: torch.Tensor
        Shape (seq_len, n_channels)

    Returns
    -------
    torch.Tensor
         Shape (n_batch, n_heads, seq_len, head_channels) or (n_batch, seq_len, n_channels)
    """
    if x.ndim == 4:
        n_batch, n_heads, seq_len, head_channels = x.shape
        return x + cache.to(x.dtype)[None, None, :seq_len, :head_channels]
    elif x.ndim == 3:
        n_batch, seq_len, n_channels = x.shape
        return x + cache.to(x.dtype)[None, :seq_len, :n_channels]
    else:
        raise ValueError(
            f"Invalid input shape {tuple(x.shape)}; "
            f"expected (n_batch, [n_heads], seq_len, n_channels)"
        )


def build_rope_cache(
    seq_len: int, n_channels: int, device, dtype, base: float = 10000.0
):
    """
    Returns
    ----------
    torch.Tensor, torch.Tensor
        Caches, shape (seq_len, n_channels/2)
    """
    assert n_channels % 2 == 0
    theta = 1.0 / (
        base
        ** (torch.arange(0, n_channels, 2, device=device, dtype=dtype) / n_channels)
    )
    seq = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->td", seq, theta)  # (seq_len, n_channels/2)
    return torch.cos(freqs), torch.sin(
        freqs
    )  # (seq_len, n_channels/2), (seq_len, n_channels/2)


def apply_rope(
    q_or_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Parameters
    ----------
    q_or_k : torch.Tensor
        Shape (n_batch, n_heads, seq_len, head_channels) where head_channels even
    cos : torch.Tensor
        Shape (seq_len, head_channels/2)
    sin : torch.Tensor
        Shape (seq_len, head_channels/2)

    Returns
    -------
    torch.Tensor
        Shape (n_batch, n_heads, seq_len, head_channels)
    """
    n_batch, n_heads, seq_len, head_channels = q_or_k.shape
    q = q_or_k.reshape(n_batch, n_heads, seq_len, head_channels // 2, 2)
    q1, q2 = q[..., 0], q[..., 1]  # (n_batch, n_heads, seq_len, n_channels / 2)
    c = cos[:seq_len].to(q_or_k.dtype)[None, None, :, :]
    s = sin[:seq_len].to(q_or_k.dtype)[None, None, :, :]
    out1 = q1 * c - q2 * s
    out2 = q1 * s + q2 * c
    return torch.stack([out1, out2], dim=-1).reshape(
        n_batch, n_heads, seq_len, head_channels
    )
