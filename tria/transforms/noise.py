import copy
import math
from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core.util import ensure_tensor
from audiotools.core.util import sample_from_dist
from numpy.random import RandomState

from ..dsp import resample
from ..util import read_manifest, normalize_source_weights, load_excerpt

from .base import NormalizedBaseTransform

################################################################################
# Additive noise
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


class FilteredNoise(NormalizedBaseTransform):
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


class BackgroundNoise(NormalizedBaseTransform):
    """
    Recorded background noise.
    """

    def __init__(
        self,
        snr: tuple = ("uniform", 10.0, 30.0),
        sources: List[str] = None,
        source_weights: Optional[List[float]] = None,
        relative_path: str = "",
        path_col: str = "path",
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 3,
        name: str = None,
        prob: float = 1.0,
        loudness_cutoff: Optional[float] = None,
        num_tries: int = None,
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

        assert sources is not None
        self.snr = snr
        self.eq_amount = eq_amount
        self.n_bands = n_bands

        self.loudness_cutoff = loudness_cutoff
        self.num_tries = int(num_tries or 0)
        self.path_col = path_col

        per_source = read_manifest(
            sources=sources,
            columns=[path_col],
            relative_path=relative_path,
            strict=True,
        )
        kept_mask = [len(lst) > 0 for lst in per_source]
        self.source_rows = [lst for lst in per_source if len(lst) > 0]
        if len(self.source_rows) == 0:
            raise RuntimeError("BackgroundNoise: no valid noise rows after filtering.")

        self._weights = normalize_source_weights(
            source_weights=source_weights,
            n_sources=len(sources),
            kept_mask=kept_mask,
        )

    def _pick_path(self, state: np.random.RandomState) -> str:
        sidx = int(state.choice(len(self.source_rows), p=self._weights))
        rows = self.source_rows[sidx]
        ridx = int(state.randint(len(rows)))
        p = rows[ridx]["paths"].get(self.path_col, "")
        return p

    def _instantiate(self, state: RandomState, signal: AudioSignal):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        snr = sample_from_dist(self.snr, state)

        noise_path = self._pick_path(state)
        if not noise_path or not Path(noise_path).is_file():
            raise RuntimeError(f"BackgroundNoise: sampled invalid noise path: {noise_path}")

        bg_sig, _ = load_excerpt(
            noise_path,
            duration=float(signal.signal_duration),
            sample_rate=int(signal.sample_rate),
            state=state,
            from_start=False,
            loudness_cutoff=self.loudness_cutoff,
            num_tries=self.num_tries,
            num_channels=int(signal.num_channels),
            resample=False,
        )
        bg_sig = resample(bg_sig.to(signal.device), signal.sample_rate)

        return {"eq": eq, "bg_signal": bg_sig, "snr": snr}

    def _transform(self, signal, bg_signal, snr, eq):
        if isinstance(snr, torch.Tensor):
            snr = snr.to(signal.device)
        if isinstance(eq, torch.Tensor):
            eq = eq.to(signal.device)
        return _mix(signal, bg_signal.to(signal.device), snr, eq)
