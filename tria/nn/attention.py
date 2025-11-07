import math
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import QKNorm
from .pos_enc import apply_rope
from .pos_enc import apply_sinusoidal
from .pos_enc import build_rope_cache
from .pos_enc import build_sinusoidal_cache

################################################################################
# Multihead attention operation
################################################################################


def ensure_masks(
    n_batch: int,
    seq_len_q: int,
    seq_len_k: int,
    device,
    mask_q: Optional[torch.Tensor],
    mask_k: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    n_batch : int
    seq_len_q : int
    seq_len_k : int
    mask_q : torch.Tensor
        Shape (n_batch, seq_len_q)
    mask_k : torch.Tensor
        Shape (n_batch, seq_len_k)
    """
    if mask_q is None:
        mask_q = torch.ones(n_batch, seq_len_q, dtype=torch.bool, device=device)
    if mask_k is None:
        mask_k = torch.ones(n_batch, seq_len_k, dtype=torch.bool, device=device)
    return mask_q, mask_k


def make_attn_mask(
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
    dtype,
) -> torch.Tensor:
    """
    Use "key padding mask" convention to prevent empty rows in attention score
    matrix (and thus softmax issues).

    Parameters
    ----------
    mask_q : torch.Tensor
        Query sequence mask, shape (n_batch, seq_len_q)
    mask_k : torch.Tensor
        Key sequence mask, shape (n_batch, seq_len_k)

    Returns
    -------
    torch.Tensor
        Additive attention mask for scaled_dot_product_attention, shape
        (n_batch, 1, seq_len_q, seq_len_k)
    """
    n_batch, seq_len_q = mask_q.shape
    seq_len_k = mask_k.shape[1]

    exclude = (
        (~mask_k)[:, None, :].expand(n_batch, seq_len_q, seq_len_k).unsqueeze(1)
    )  # (n_batch, 1, seq_len_q, seq_len_k)
    mask = exclude.to(dtype=dtype).masked_fill(exclude, float("-inf"))

    return mask  # (n_batch, 1, seq_len_q, seq_len_k)


def sdpa_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    p_dropout: float,
    training: bool,
    use_sdpa: bool = True,
) -> torch.Tensor:
    """
    Optionally use PyTorch scaled_dot_product_attention (SDPA), which picks
    efficient attention implementations (e.g. flash attention) if available

    Parameters
    ----------
    q : torch.Tensor
        Query, shape (n_batch, n_heads, seq_len_q, head_channels)
    k : torch.Tensor
        Key, shape (n_batch, n_heads, seq_len_k, head_channels)
    v : torch.Tensor
        Value, shape (n_batch, n_heads, seq_len_k, head_channels)
    attn_mask : torch.Tensor
        Additive attention mask (0 or -inf), shape (n_batch, 1, seq_len_q, seq_len_k)

    Returns
    -------
    torch.Tensor
        Shape (n_batch, n_heads, seq_len_q, head_channels)
    """

    n_batch, n_heads, seq_len_q, head_channels = q.shape
    seq_len_k = k.shape[2]

    if use_sdpa and q.is_cuda:
        if attn_mask is not None and (
            (attn_mask.dtype == torch.bool and attn_mask.all())
            or (attn_mask.dtype != torch.bool and not attn_mask.ne(0).any())
        ):
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=p_dropout if training else 0.0,
            is_causal=False,
        )
        return out

    # Fallback
    scale = 1.0 / math.sqrt(head_channels)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    if attn_mask is not None:
        scores = scores + attn_mask  # Additive mask
    attn = scores.softmax(dim=-1)
    if training and p_dropout > 0.0:
        attn = F.dropout(attn, p=p_dropout)
    out = torch.einsum("bhts,bhsd->bhtd", attn, v)
    return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        bias: bool = True,
        max_len: int = 8192,
        pos_enc: Optional[str] = "rope",
        qk_norm: bool = True,
        use_sdpa: bool = True,
    ):
        super().__init__()
        assert n_channels % n_heads == 0, "`n_channels` must be divisible by `n_heads`"
        assert pos_enc in ("rope", "absolute", "none", None)

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.head_channels = n_channels // n_heads
        self.p_dropout = p_dropout
        self.pos_enc = pos_enc
        self.max_len = max_len
        self.use_sdpa = use_sdpa

        self.q_proj = nn.Linear(n_channels, n_channels, bias=bias)
        self.k_proj = nn.Linear(n_channels, n_channels, bias=bias)
        self.v_proj = nn.Linear(n_channels, n_channels, bias=bias)
        self.o_proj = nn.Linear(n_channels, n_channels, bias=bias)

        self.o_dropout = nn.Dropout(p_dropout)

        self.qk_norm = QKNorm(self.head_channels) if qk_norm else None
        self.pos_cache = None

    def _maybe_build_pos_cache(self, device, dtype):
        if self.pos_enc in [None, "none"] or self.pos_cache is not None:
            return
        if self.pos_enc == "absolute":
            self.pos_cache = build_sinusoidal_cache(
                self.max_len, self.head_channels, device, dtype=torch.float32
            )
        elif self.pos_enc == "rope":
            cos, sin = build_rope_cache(
                self.max_len, self.head_channels, device, dtype=torch.float32
            )
            self.pos_cache = (cos, sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            Query, shape (n_batch, seq_len_q, n_channels)
        k : torch.Tensor
            Key, shape (n_batch, seq_len_k, n_channels)
        v : torch.Tensor
            Value, shape (n_batch, seq_len_k, n_channels)
        mask_q : torch.Tensor
            Boolean mask, `True` for valid positions; shape (n_batch, seq_len_q)
        mask_k : torch.Tensor
            Boolean mask, `True` for valid positions; shape (n_batch, seq_len_k)
        attn_mask : torch.tensor
            Additive (0, -inf) mask; shape (n_batch, 1, seq_len_q, seq_len_k)
        """

        n_batch, seq_len_q, _ = q.shape
        seq_len_k = k.shape[1]
        device, dtype = q.device, q.dtype

        # Projections (n_batch, seq_len, n_channels) -> (n_batch, n_heads, seq_len, head_channels)
        q = (
            self.q_proj(q)
            .view(n_batch, seq_len_q, self.n_heads, self.head_channels)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(k)
            .view(n_batch, seq_len_k, self.n_heads, self.head_channels)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(v)
            .view(n_batch, seq_len_k, self.n_heads, self.head_channels)
            .transpose(1, 2)
        )

        # Positional encoding
        self._maybe_build_pos_cache(device=device, dtype=dtype)
        if self.pos_enc == "absolute":
            cache = self.pos_cache  # (max_seq_len, head_channels)
            q = apply_sinusoidal(q, cache)
            k = apply_sinusoidal(k, cache)
        elif self.pos_enc == "rope":
            cos, sin = self.pos_cache  # (max_seq_len, head_channels/2)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # QK-Norm
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Masks
        mask_q, mask_k = ensure_masks(
            n_batch, seq_len_q, seq_len_k, device, mask_q, mask_k
        )
        pad_mask = make_attn_mask(
            mask_q, mask_k, dtype
        )  # (n_batch, 1, seq_len_q, seq_len_k)

        if attn_mask is not None:
            pad_mask = pad_mask + attn_mask

        # Attention
        y = sdpa_with_fallback(
            q,
            k,
            v,
            attn_mask=pad_mask,
            p_dropout=self.p_dropout,
            training=self.training,
            use_sdpa=self.use_sdpa,
        )  # (n_batch, n_heads, seq_len_q, head_channels)

        y = y.transpose(1, 2).contiguous().view(n_batch, seq_len_q, self.n_channels)
        y = self.o_proj(y)  # (n_batch, seq_len_q, n_channels)
        y = self.o_dropout(y)

        # Mask outputs
        if mask_q is not None:
            with torch.no_grad():
                y.masked_fill_(~mask_q[:, :, None], 0.0)
        return y
