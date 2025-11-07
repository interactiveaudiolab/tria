import copy
import math
from typing import List

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import ensure_tensor
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from audiotools.data.transforms import SpectralTransform
from numpy.random import RandomState

################################################################################
# Phase shift transform for encouraging robust rhythm feature extraction
################################################################################


class ShiftPhase(SpectralTransform):
    """
    Patch `audiotools.data.transforms.ShiftPhase` to allow processing on GPU
    """

    def __init__(
        self,
        shift: tuple = ("uniform", -np.pi, np.pi),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.shift = shift

    def _instantiate(self, state: RandomState):
        return {"shift": sample_from_dist(self.shift, state)}

    def _transform(self, signal, shift):
        shift = ensure_tensor(shift, ndim=signal.phase.ndim).to(signal.device)
        sig = signal.shift_phase(shift)
        sig.ensure_max_of_audio()
        return sig
