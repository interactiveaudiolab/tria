import copy
import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import ensure_tensor
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from audiotools.data.datasets import AudioLoader
from audiotools.data.transforms import BaseTransform
from numpy.random import RandomState

################################################################################
# Noise transform for encouraging robust rhythm feature extraction
################################################################################


def _mix(
    signal: AudioSignal,
    other: AudioSignal,
    snr: Union[torch.Tensor, np.ndarray, float] = None,
    other_eq: Union[torch.Tensor, np.ndarray] = None,
):
    signal = signal.clone()
    other = other.clone()

    if other.num_channels < signal.num_channels:
        assert other.num_channels == 1
        other.audio_data = other.audio_data.repeat(1, signal.num_channels, 1)

    if other.num_channels > signal.num_channels:
        assert signal.num_channels == 1
        other.audio_data = other.audio_data.mean(dim=1, keepdim=True)

    snr = ensure_tensor(snr).to(signal.device).view(-1, 1, 1)
    pad_len = max(0, signal.signal_length - other.signal_length)

    other.zero_pad(0, pad_len)
    other.truncate_samples(signal.signal_length)

    if other_eq is not None:
        other = other.equalizer(other_eq)

    signal_energy = torch.mean(
        torch.square(signal.audio_data.mean(dim=1, keepdim=True)),
        dim=(1, 2),
        keepdim=True,
    )
    other_energy = torch.mean(
        torch.square(other.audio_data.mean(dim=1, keepdim=True)),
        dim=(1, 2),
        keepdim=True,
    )

    signal_db = 10 * torch.log10(signal_energy + 1e-8)
    other_db = 10 * torch.log10(other_energy + 1e-8)

    scale = torch.sqrt(torch.pow(10, (signal_db - other_db - snr) / 10))

    signal.audio_data = signal.audio_data + scale * other.audio_data
    signal.ensure_max_of_audio()

    return signal


class FilteredNoise(BaseTransform):
    """
    Filtered Gaussian noise.
    """

    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = sample_from_dist(self.snr, state)

        bg_signal = signal[0].clone()
        bg_signal.audio_data = torch.randn_like(bg_signal.audio_data)
        bg_signal.ensure_max_of_audio()

        return {"eq": eq, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, signal, bg_signal, snr, eq):
        if isinstance(snr, torch.Tensor):
            snr = snr.to(signal.device)
        if isinstance(eq, torch.Tensor):
            eq = eq.to(signal.device)
        # return signal.mix(bg_signal.clone().to(signal.device), snr, eq)
        return _mix(signal, bg_signal.to(signal.device), snr, eq)


class BackgroundNoise(BaseTransform):
    """
    Recorded background noise.
    """

    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        sources: List[str] = None,
        weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 3,
        name: str = None,
        prob: float = 1.0,
        loudness_cutoff: float = None,
    ):
        super().__init__(name=name, prob=prob)

        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.loader = AudioLoader(sources, weights)
        self.loudness_cutoff = loudness_cutoff

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = sample_from_dist(self.snr, state)

        bg_signal = self.loader(
            state,
            signal.sample_rate,
            duration=signal.signal_duration,
            loudness_cutoff=self.loudness_cutoff,
            num_channels=signal.num_channels,
        )["signal"]

        return {"eq": eq, "bg_signal": bg_signal, "snr": snr}

    def _transform(self, signal, bg_signal, snr, eq):
        if isinstance(snr, torch.Tensor):
            snr = snr.to(signal.device)
        if isinstance(eq, torch.Tensor):
            eq = eq.to(signal.device)
        # return signal.mix(bg_signal.clone().to(signal.device), snr, eq)
        return _mix(signal, bg_signal.to(signal.device), snr, eq)
