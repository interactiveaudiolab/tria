import copy
import math
from typing import List

import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from audiotools.data.datasets import AudioLoader
from audiotools.data.transforms import BaseTransform
from numpy.random import RandomState


################################################################################
# Reverb transform for encouraging robust rhythm feature extraction
################################################################################


class Reverb(BaseTransform):
    """
    Patch device error in `audiotools.data.transforms.RoomImpulseResponse` to
    allow processing on GPU.
    """

    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        sources: List[str] = None,
        weights: List[float] = None,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        offset: float = 0.0,
        duration: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.drr = drr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.use_original_phase = use_original_phase

        self.loader = AudioLoader(sources, weights)
        self.offset = offset
        self.duration = duration

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        drr = sample_from_dist(self.drr, state)

        ir_signal = self.loader(
            state,
            signal.sample_rate,
            offset=self.offset,
            duration=self.duration,
            loudness_cutoff=None,
            num_channels=signal.num_channels,
        )["signal"]
        ir_signal.zero_pad_to(signal.sample_rate)

        return {"eq": eq, "ir_signal": ir_signal, "drr": drr}

    def _transform(self, signal, ir_signal, drr, eq):
        if isinstance(drr, torch.Tensor):
            drr = drr.to(signal.device)
        if isinstance(eq, torch.Tensor):
            eq = eq.to(signal.device)
        ir_signal = ir_signal.clone().to(signal.device)

        return signal.apply_ir(
            ir_signal, drr, eq, use_original_phase=self.use_original_phase
        )
