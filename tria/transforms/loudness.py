import copy
import math
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import torch
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import ensure_tensor
from audiotools.core.util import random_state
from audiotools.core.util import sample_from_dist
from audiotools.data.transforms import BaseTransform
from numpy.random import RandomState

################################################################################
# Normalization transform for encouraging robust rhythm feature extraction
################################################################################


class VolumeNorm(BaseTransform):
    """
    Normalize to a target RMS level (in dBFS).
    """

    def __init__(
        self,
        db: tuple = ("const", -24.0),
        name: Optional[str] = None,
        prob: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__(name=name, prob=prob)
        self.db = db
        self.eps = float(eps)

    def _instantiate(self, state):
        return {"db": sample_from_dist(self.db, state)}

    def _transform(
        self, signal: AudioSignal, db: Union[float, Sequence[float], torch.Tensor]
    ):
        x = signal.audio_data  # (n_batch, n_channels, n_samples)
        n_batch, n_channels, n_samples = x.shape

        db = ensure_tensor(db, ndim=1, batch_size=n_batch).to(x.device)

        rms = torch.sqrt((x**2).mean(dim=(1, 2)).clamp_min(self.eps))
        cur_db = 20.0 * torch.log10(rms)  # (n_batch,)

        # Gain in dB and linear scale
        gain_db = db - cur_db  # (n_batch,)
        ln10 = torch.log(torch.tensor([10.0], device=x.device, dtype=x.dtype))
        linear_gain = torch.exp((gain_db / 20.0) * ln10).view(
            n_batch, 1, 1
        )  # (n_batch, 1, 1)

        x = x * linear_gain

        signal.audio_data = x
        signal.ensure_max_of_audio()

        return signal
