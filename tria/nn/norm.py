from typing import Tuple

import torch
import torch.nn as nn

################################################################################
# Normalization layers
################################################################################


class RMSNorm(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize over final dimension
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * rms  # Broadcast targets final dimension


class QKNorm(nn.Module):
    """
    RMS-normalize query and key across channel dimension with a learnable gain.
    Applied per-head, per-position.
    """

    def __init__(self, head_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.g_q = nn.Parameter(torch.ones(head_channels))
        self.g_k = nn.Parameter(torch.ones(head_channels))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        q : torch.Tensor
            Query, shape (n_batch, n_heads, seq_len_q, head_channels)
        k : torch.Tensor
            Key, shape (n_batch, n_heads, seq_len_k, head_channels)
        """

        def _rmsnorm(x, g):
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x * rms * g  # Broadcast targets final dimension

        return _rmsnorm(q, self.g_q), _rmsnorm(k, self.g_k)
