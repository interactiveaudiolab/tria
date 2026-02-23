from typing import Optional

import torch
from numpy.random import RandomState

from audiotools import AudioSignal
from audiotools.core.util import ensure_tensor
from audiotools.core.util import sample_from_dist

from ..dsp import low_pass, high_pass
from .base import NormalizedBaseTransform

################################################################################
# Simple delay with filtering and attenuation
################################################################################


class Delay(NormalizedBaseTransform):
    """
    Multi-tap delay by summing shifted copies of the signal, with optional
    filtering applied to wet signal.

    Parameters
    ----------
    delay_ms:
        Delay time in milliseconds
    feedback:
        Per-tap gain decay. Tap k has gain feedback**k (k=1..n_taps)
    n_taps:
        Number of delayed copies (taps).
    mix:
        Wet/dry mix in [0,1]
    low_cutoff:
        Low-pass cutoff (Hz) for wet signal
    high_cutoff:
        High-pass cutoff (Hz) for wet signal
    zeros:
        Pass filter zeros
    """

    def __init__(
        self,
        delay_ms: tuple = ("uniform", 30.0, 300.0),
        feedback: tuple = ("uniform", 0.2, 0.7),
        n_taps: int = 16,
        mix: tuple = ("uniform", 0.1, 0.5),
        low_cutoff: Optional[tuple] = ("choice", (100, 250, 500, 1000)),
        high_cutoff: Optional[tuple] = ("choice", (8000, 4000, 2000)),
        zeros: int = 51,
        name: str = None,
        prob: float = 1.0,
        eps: float = 1e-8,
        # Normalization
        match_energy: bool = True,
        clamp_gain: Optional[float] = None,
        ensure_max_of_audio: bool = True,
    ):
        super().__init__(
            name=name, 
            prob=prob,
            match_energy=match_energy, 
            clamp_gain=clamp_gain, 
            ensure_max_of_audio=ensure_max_of_audio,
        )
        self.delay_ms = delay_ms
        self.feedback = feedback
        self.n_taps = int(n_taps)
        self.mix = mix
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.zeros = int(zeros)
        self.eps = float(eps)

        if self.n_taps < 1:
            raise ValueError("`n_taps` must be >= 1")

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        delay_ms = sample_from_dist(self.delay_ms, state)
        feedback = sample_from_dist(self.feedback, state)
        mix = sample_from_dist(self.mix, state)

        out = {
            "delay_ms": delay_ms,
            "feedback": feedback,
            "mix": mix,
        }

        if self.low_cutoff is not None:
            out["low_cutoff"] = sample_from_dist(self.low_cutoff, state)
        if self.high_cutoff is not None:
            out["high_cutoff"] = sample_from_dist(self.high_cutoff, state)

        return out

    def _transform(
        self,
        signal,
        delay_ms,
        feedback,
        mix,
        low_cutoff=None,
        high_cutoff=None,
    ):
        x = signal.audio_data
        n_batch = signal.batch_size
        sr = float(signal.sample_rate)
        n_samples = int(signal.signal_length)

        delay_ms = ensure_tensor(delay_ms, 2, n_batch).to(x.device).view(-1).float().clamp(min=0.0)
        feedback = ensure_tensor(feedback, 2, n_batch).to(x.device).view(-1).float().clamp(0.0, 1.0)
        mix = ensure_tensor(mix, 2, n_batch).to(x.device).view(-1, 1, 1).float().clamp(0.0, 1.0)

        # Delay in samples
        base_delay = torch.round(delay_ms * (sr / 1000.0)).to(torch.long).clamp(min=1)

        # Per-tap delays/gains
        k = torch.arange(1, self.n_taps + 1, device=x.device, dtype=torch.long)[None, :]  # (1, n_taps + 1)
        delays = base_delay[:, None] * k  # (n_batch, n_taps + 1)

        kk = torch.arange(1, self.n_taps + 1, device=x.device, dtype=x.dtype)[None, :]  # (1, n_taps + 1)
        gains = feedback[:, None].to(x.dtype).clamp(0.0, 1.0) ** kk  # (n_batch, n_taps + 1)

        dry = signal.clone()

        # Wet source: copy and filter
        wet_src = signal.clone()
        if low_cutoff is not None:
            wet_src = low_pass(wet_src, high_cutoff, zeros=self.zeros)
        if high_cutoff is not None:
            wet_src = high_pass(wet_src, low_cutoff, zeros=self.zeros)

        xs = wet_src.audio_data

        # Sum shifted copies
        wet = torch.zeros_like(xs)
        t = torch.arange(n_samples, device=x.device, dtype=torch.long)[None, None, :]  # (1, 1, n_samples)

        for i in range(self.n_taps):
            d = delays[:, i].view(-1, 1, 1)            # (n_batch, 1, 1)
            g = gains[:, i].view(-1, 1, 1)             # (n_batch, 1, 1)

            idx = t - d                                # (n_batch, 1, n_samples)
            mask = idx >= 0                            # (n_batch, 1, n_samples)
            idx = idx.clamp(min=0).expand_as(xs)       # (n_batch, n_channels, n_samples)

            x_delayed = xs.gather(dim=-1, index=idx)   # (n_batch, 1, n_samples)
            wet = wet + g * x_delayed * mask.to(xs.dtype)

        # Mix
        signal.audio_data = (1.0 - mix) * dry.audio_data + mix * wet
        signal.stft_data = None
        return signal
