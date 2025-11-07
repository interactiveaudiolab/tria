import copy
import math
from typing import List

import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from audiotools.data.transforms import BaseTransform
from numpy.random import RandomState

################################################################################
# Bandpass transform for encouraging robust rhythm feature extraction
################################################################################


class BandPass(BaseTransform):
    """
    Band pass filter.
    """

    def __init__(
        self,
        low_cutoff: tuple = ("choice", [50, 100, 250, 500]),
        high_cutoff: tuple = ("choice", [1000, 2000, 4000, 8000]),
        zeros: int = 51,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.zeros = zeros

        # Validate cutoffs, fail if overlap
        lo_min, lo_max = self._dist_min_max(self.low_cutoff)
        hi_min, hi_max = self._dist_min_max(self.high_cutoff)

        if not (lo_max < hi_min):
            raise ValueError(
                "BandPass cutoffs may overlap or invert based on the provided distributions: "
                f"low_cutoff spec={self.low_cutoff} → range≈[{lo_min}, {lo_max}], "
                f"high_cutoff spec={self.high_cutoff} → range≈[{hi_min}, {hi_max}]. "
                "Ensure all possible low values are strictly less than all possible high values."
            )

    @staticmethod
    def _flatten_payload(spec: tuple):
        if not isinstance(spec, tuple) or len(spec) < 2:
            return []
        payload = spec[1:]
        if len(payload) == 1 and isinstance(payload[0], (list, tuple)):
            vals = list(payload[0])
        else:
            vals = list(payload)
        return [float(v) for v in vals if isinstance(v, (int, float))]

    @classmethod
    def _dist_min_max(cls, spec: tuple):
        if not isinstance(spec, tuple) or len(spec) < 2:
            return (float("inf"), float("-inf"))

        kind = str(spec[0]).lower()
        vals = cls._flatten_payload(spec)

        if kind in ("choice", "categorical", "const"):
            if not vals:
                return (float("inf"), float("-inf"))
            return (min(vals), max(vals))

        if kind in ("uniform", "loguniform", "between", "randint", "triangular"):
            if len(vals) >= 2:
                a, b = float(vals[0]), float(vals[1])
                lo, hi = (a, b) if a <= b else (b, a)
                return (lo, hi)

        return (float("-inf"), float("inf"))

    def _instantiate(self, state: RandomState):
        # Sample concrete cutoffs for this instantiation
        lo = sample_from_dist(self.low_cutoff, state)
        hi = sample_from_dist(self.high_cutoff, state)

        # Defensive check (should always pass if ranges are disjoint)
        if not (float(lo) < float(hi)):
            raise ValueError(
                f"BandPass sampled invalid band: low_cutoff={lo}, high_cutoff={hi}. "
                "Adjust your distributions so low < high."
            )
        return {"low_cutoff": float(lo), "high_cutoff": float(hi)}

    def _transform(self, signal, low_cutoff, high_cutoff):
        sig = signal.high_pass(low_cutoff, zeros=self.zeros)
        sig = sig.low_pass(high_cutoff, zeros=self.zeros)
        sig = sig.ensure_max_of_audio()
        return sig


class Equalizer(BaseTransform):
    """
    Mel-band equalization.
    """

    def __init__(
        self,
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.eq_amount = eq_amount
        self.n_bands = n_bands

    def _instantiate(self, state: RandomState):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        return {"eq": eq}

    def _transform(self, signal, eq):
        sig = signal.equalizer(eq)
        sig = sig.ensure_max_of_audio()
        return sig
