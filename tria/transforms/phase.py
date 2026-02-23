import copy
import math
from typing import List, Optional

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import ensure_tensor
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from numpy.random import RandomState

from .base import NormalizedBaseTransform

################################################################################
# Phase shift transform for encouraging robust rhythm feature extraction
################################################################################


class ShiftPhase(NormalizedBaseTransform):
    """
    Patch `audiotools.data.transforms.ShiftPhase` to allow processing on GPU
    """

    def __init__(
        self,
        shift: tuple = ("uniform", -np.pi, np.pi),
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
        self.shift = shift

    def _instantiate(self, state: RandomState):
        return {"shift": sample_from_dist(self.shift, state)}

    def _transform(self, signal, shift):
        signal.stft()
        shift = ensure_tensor(shift, ndim=signal.phase.ndim).to(signal.device)
        sig = signal.shift_phase(shift)
        signal.istft()
        return sig
    
