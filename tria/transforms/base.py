from typing import Optional

import torch
from audiotools import AudioSignal
from audiotools.data.transforms import BaseTransform

from ..constants import EPS

################################################################################
# Base transform class with added audio normalization
################################################################################


def _energy(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Compute per-item signal energy

    Parameters
    ----------
    x : torch.Tensor
        Audio signal, shape (n_batch, n_channels, n_samples)

    Returns
    -------
    torch.Tensor
        Shape (n_batch, 1, 1) metric values.
    """
    assert x.ndim == 3

    # Average energy over channels and time
    return torch.mean(x * x, dim=(-2, -1), keepdim=True) + eps


class NormalizedBaseTransform(BaseTransform):
    """
    Extension of audiotools.data.transforms.BaseTransform with optional
    post-transform audio normalization.
    """

    def __init__(
        self,
        *args,
        match_energy: bool = False,
        clamp_gain: Optional[float] = None,
        ensure_max_of_audio: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.match_energy = match_energy
        self.clamp_gain = clamp_gain
        self.ensure_max_of_audio = ensure_max_of_audio

        if self.clamp_gain is not None and self.clamp_gain <= 0:
            raise ValueError("`clamp_gain` must be > 0 when provided.")

    def transform(self, signal: AudioSignal, **kwargs):
        """
        Optionally match output energy to input energy, clamp output gain, and
        ensure output signals do not clip (for transformed batch items only)
        """
        tfm_kwargs = self._prepare(kwargs)
        mask = tfm_kwargs["mask"]

        if torch.any(mask):
            # Precompute energy on masked subset if requested
            if self.match_energy:
                x_in = signal[mask].audio_data  # (n_transformed, n_channels, n_samples)
                m_in = _energy(x_in)

            # Apply transform
            tfm_kwargs_masked = self.apply_mask(tfm_kwargs, mask)
            tfm_kwargs_masked = {
                k: v for k, v in tfm_kwargs_masked.items() if k != "mask"
            }
            signal[mask] = self._transform(signal[mask], **tfm_kwargs_masked)

            # Match energy
            if self.match_energy:
                x_out = signal[mask].audio_data
                m_out = _energy(x_out)

                gain = torch.sqrt(m_in / m_out.clamp(min=EPS))

                if self.clamp_gain is not None:
                    cg = self.clamp_gain
                    gain = torch.clamp(gain, 1.0 / cg, cg)

                signal.audio_data[mask] = signal.audio_data[mask] * gain

            # Ensure valid
            if self.ensure_max_of_audio:
                signal[mask].ensure_max_of_audio()

        return signal
