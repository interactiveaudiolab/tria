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
from .base import NormalizedBaseTransform

################################################################################
# Waveform distortion
################################################################################


class ClippingDistortion(NormalizedBaseTransform):
    """
    Clip waveform symmetrically so that proportion `p` of samples falls outside
    clipping threshold. For example:
      * `p` = 0.0: no clipping
      * `p` = 0.1: clamp to [q=0.05, q=0.95] quantiles
    """

    def __init__(
        self,
        perc: tuple = ("uniform", 0.0, 0.1),
        mix: tuple = ("const", 1.0),
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
        self.perc = perc
        self.mix = mix

    def _instantiate(self, state: RandomState):
        return {
            "perc": sample_from_dist(self.perc, state),
            "mix": sample_from_dist(self.mix, state),
        }

    def _transform(self, signal, perc, mix):
        x = signal.audio_data
        n_batch = signal.batch_size

        p = ensure_tensor(perc, 1).to(x.device).float()
        p = p.clamp(0.0, 1.0)

        m = ensure_tensor(mix, 1).to(x.device).view(-1, 1, 1).float()
        m = m.clamp(0.0, 1.0)

        min_thr, max_thr = [], []
        for i, _p in enumerate(p):
            qi = float(_p.item())
            flat = x[i].reshape(-1)
            min_thr += [torch.quantile(flat, qi / 2.0)]
            max_thr += [torch.quantile(flat, 1.0 - (qi / 2.0))]

        min_thr = torch.stack(min_thr, dim=0).to(x.device, dtype=x.dtype).view(-1, 1, 1)
        max_thr = torch.stack(max_thr, dim=0).to(x.device, dtype=x.dtype).view(-1, 1, 1)
        
        y = x.clamp(min_thr, max_thr)
    
        signal.audio_data = m * y + (1.0 - m) * x
        signal.stft_data = None
        return signal


class WaveshaperDistortion(NormalizedBaseTransform):
    """
    Apply waveshaping distortion via TanH nonlinearity.
    """

    def __init__(
        self,
        drive: tuple = ("uniform", 1.0, 10.0),
        mix: tuple = ("const", 1.0),
        normalize: bool = False,
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
        self.drive = drive
        self.mix = mix
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def _instantiate(self, state: RandomState):
        d = sample_from_dist(self.drive, state)
        m = sample_from_dist(self.mix, state)
        return {"drive": d, "mix": m}

    def _transform(self, signal, drive, mix):
        x = signal.audio_data
        n_batch = signal.batch_size

        d = ensure_tensor(drive, 2, n_batch).to(x.device).view(-1, 1, 1).float().clamp(min=0.0)
        m = ensure_tensor(mix, 2, n_batch).to(x.device).view(-1, 1, 1).float().clamp(0.0, 1.0)

        y = torch.tanh(d * x)
        if self.normalize:
            y = y / torch.tanh(d).clamp(min=self.eps)

        signal.audio_data = m * y + (1.0 - m) * x
        signal.stft_data = None
        return signal
