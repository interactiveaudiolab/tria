"""
Adapted from `torch_pitch_shift` by Kento Nishi (MIT license); see:
https://github.com/KentoNishi/torch-pitch-shift/
"""
from collections import Counter
from fractions import Fraction
from itertools import chain
from itertools import count
from itertools import islice
from itertools import repeat
from math import log2
from typing import Optional

import numpy as np
import torch
import torchaudio
from audiotools import AudioSignal
from audiotools.core.util import sample_from_dist
from audiotools.data.transforms import BaseTransform
from packaging import version
from primePy import primes
from torchaudio.transforms import TimeStretch

################################################################################
# Pitch shift transform to ensure robust rhythm feature extraction
################################################################################


def semitones_to_ratio(semitones: float) -> Fraction:
    return Fraction(2.0 ** (semitones / 12.0))


def ratio_to_semitones(ratio: Fraction) -> float:
    return float(12.0 * log2(ratio))


def _combinations_without_repetition(r, iterable=None, values=None, counts=None):
    if iterable:
        values, counts = zip(*Counter(iterable).items())

    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), f(count(j), counts[j:])):
            indices[i] = j


def get_fast_shifts(
    sample_rate: int, condition=lambda x: x >= 0.5 and x <= 2 and x != 1
):
    """
    Return a list of Fractions i/j such that (sample_rate / (i/j)) is integer
    """

    fast = set()
    factors = primes.factors(sample_rate)
    products = []
    for i in range(1, len(factors) + 1):
        products.extend(
            [
                torch.prod(torch.tensor(x)).item()
                for x in _combinations_without_repetition(i, iterable=factors)
            ]
        )
    for i in products:
        for j in products:
            f = Fraction(int(i), int(j))
            if condition(f):
                fast.add(f)
    return list(fast)


class PitchShift(BaseTransform):
    """
    Pitch-shift via phase vocoder time-stretch and resample.
    """

    def __init__(
        self,
        shift_semitones: tuple = ("uniform", -3.0, 3.0),
        force_fast: bool = True,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)
        self.shift_semitones = shift_semitones
        self.force_fast = bool(force_fast)

    def _instantiate(self, state, signal: Optional[AudioSignal] = None):
        semis = float(sample_from_dist(self.shift_semitones, state))
        return {"shift_semitones": semis}

    def _transform(self, signal: AudioSignal, shift_semitones):
        sample_rate = int(signal.sample_rate)
        n_fft = max(2, sample_rate // 64)
        hop_length = max(1, n_fft // 32)
        n_freq = n_fft // 2 + 1

        x = signal.audio_data
        n_batch, n_channels, n_samples = x.shape

        def _to_list(v, B):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, (list, tuple, np.ndarray)):
                v = list(map(float, v))
                if len(v) not in (1, B):
                    raise ValueError(
                        f"shift_semitones length {len(v)} must be 1 or {B}"
                    )
                return v if len(v) == B else v * B
            return [float(v)] * B

        semis_list = _to_list(shift_semitones, n_batch)

        # Optional fast-ratio snapping (once)
        fast_ratios = get_fast_shifts(sample_rate) if self.force_fast else None

        # TimeStretch (per-item overriding_rate)
        ts = TimeStretch(n_freq=n_freq, hop_length=hop_length).to(x.device)

        outs = []
        for b in range(n_batch):
            # Ratio for this item
            semis_b = semis_list[b]
            r = float(semitones_to_ratio(semis_b))
            if fast_ratios:
                best = min(
                    fast_ratios, key=lambda fr: abs(ratio_to_semitones(fr) - semis_b)
                )
                r = float(best)

            # STFT
            tmp = signal.clone()
            tmp.audio_data = x[b : b + 1].contiguous()  # (1, n_channels, n_samples)
            spec = tmp.stft(
                window_length=n_fft, hop_length=hop_length
            )  # (1, n_channels, n_freq, n_frames)
            _, _, f_chk, t_frames = spec.shape
            assert f_chk == n_freq

            # Collapse channels to batch for TimeStretch
            spec_bc = spec.reshape(
                n_channels, n_freq, t_frames
            )  # (n_channels, n_freq, n_frames)

            # Phase vocoder
            spec_stretched_bc = ts(
                spec_bc, overriding_rate=float(1.0 / r)
            )  # (n_channels, n_freq, n_frames')
            spec_stretched = spec_stretched_bc.reshape(
                1, n_channels, n_freq, spec_stretched_bc.shape[-1]
            )

            # iSTFT
            tmp.stft_data = spec_stretched
            tmp.istft(
                window_length=n_fft,
                hop_length=hop_length,
                length=int(tmp.signal_length * r),
            )
            y = tmp.audio_data  # (1, n_channels, n_samples')

            # Resample
            out_b = signal.clone()
            out_b.audio_data = y

            target_sr = max(1, int(round(sample_rate / r)))
            out_b = out_b.resample(target_sr)
            out_b.sample_rate = signal.sample_rate

            cur_len = out_b.audio_data.shape[-1]
            if cur_len > n_samples:
                out_b.audio_data = out_b.audio_data[..., :n_samples]
            elif cur_len < n_samples:
                out_b = out_b.zero_pad_to(n_samples)

            outs.append(out_b.audio_data)

        yb = torch.cat(outs, dim=0)
        out = signal.clone()
        out.audio_data = yb  # (n_batch, n_channels, n_samples)
        out.ensure_max_of_audio()
        return out
