"""
Adapted from `torch_pitch_shift` by Kento Nishi (MIT license); see:
https://github.com/KentoNishi/torch-pitch-shift/
"""
from collections import Counter, OrderedDict
from fractions import Fraction
from functools import lru_cache
from itertools import chain, count, islice, repeat
from math import log2
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core.util import sample_from_dist
from primePy import primes
from torchaudio.transforms import TimeStretch

from ..dsp import resample
from .base import NormalizedBaseTransform


################################################################################
# Pitch shift
################################################################################


def semitones_to_ratio(semitones: float) -> Fraction:
    return Fraction(2.0 ** (semitones / 12.0))


def ratio_to_semitones(ratio: Fraction) -> float:
    return float(12.0 * log2(ratio))


def _combinations_without_repetition(r: int, iterable=None, values=None, counts=None):
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
    sample_rate: int, condition: Callable[[Fraction], bool] = lambda x: x >= 0.5 and x <= 2 and x != 1
) -> List[Fraction]:
    """
    Return a list of Fractions i/j such that (sample_rate / (i/j)) is integer.
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


@lru_cache(maxsize=16)
def _cached_fast_ratios_and_semis(sample_rate: int) -> Tuple[Tuple[Fraction, ...], Tuple[float, ...]]:
    """
    Cache efficient pitch shifts in ratio and semitone formats
    """
    ratios = tuple(get_fast_shifts(sample_rate))
    semis = tuple(ratio_to_semitones(r) for r in ratios)
    return ratios, semis


def _to_list(v, B: int) -> List[float]:
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, (list, tuple, np.ndarray)):
        v = list(map(float, v))
        if len(v) not in (1, B):
            raise ValueError(f"shift_semitones length {len(v)} must be 1 or {B}")
        return v if len(v) == B else v * B
    return [float(v)] * B


class PitchShift(NormalizedBaseTransform):
    """
    Pitch-shift via phase vocoder time-stretch and resample
    """

    def __init__(
        self,
        shift_semitones: tuple = ("uniform", -3.0, 3.0),
        force_fast: bool = True,
        name: str = None,
        prob: float = 1.0,
        _ts_cache_max: int = 4,
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
        self.shift_semitones = shift_semitones
        self.force_fast = bool(force_fast)

        self._ts_cache_max = int(_ts_cache_max)
        self._ts_cache: "OrderedDict[Tuple[str, int, int], TimeStretch]" = OrderedDict()

    def _instantiate(self, state, signal: Optional[AudioSignal] = None):
        semis = float(sample_from_dist(self.shift_semitones, state))
        return {"shift_semitones": semis}

    def _get_ts(self, device: torch.device, n_freq: int, hop_length: int) -> TimeStretch:
        """
        Create cached torchaudio `TimeStretch` objects to facilitate pitch 
        shifting, keyed by (device_str, n_freq, hop_length)
        """
        key = (str(device), int(n_freq), int(hop_length))
        if key in self._ts_cache:
            ts = self._ts_cache.pop(key)
            self._ts_cache[key] = ts
            return ts

        ts = TimeStretch(n_freq=n_freq, hop_length=hop_length).to(device)
        self._ts_cache[key] = ts
        while len(self._ts_cache) > self._ts_cache_max:
            self._ts_cache.popitem(last=False)
        return ts

    @torch.no_grad()
    def _transform(self, signal: AudioSignal, shift_semitones):

        sample_rate = int(signal.sample_rate)
        n_fft = max(2, sample_rate // 64)
        hop_length = max(1, n_fft // 32)
        n_freq = n_fft // 2 + 1

        x = signal.audio_data

        n_batch, n_channels, n_samples = x.shape
        device = x.device
        dtype = x.dtype

        semis_list = _to_list(shift_semitones, n_batch)

        # Fast-ratio snapping
        if self.force_fast:
            fast_ratios, fast_semis = _cached_fast_ratios_and_semis(sample_rate)
        else:
            fast_ratios, fast_semis = (), ()

        # Compute per-item ratio r; keep exact Fraction where possible (for 
        # stable grouping by ratio), and keep float for math.
        r_frac: List[Fraction] = []
        r_float = torch.empty((n_batch,), device=device, dtype=torch.float32)

        if fast_ratios:
            # Snap by nearest semitone distance among fast ratios
            for b in range(n_batch):
                s = float(semis_list[b])
                # Find best index in cached semitone list
                best_i = min(range(len(fast_ratios)), key=lambda i: abs(fast_semis[i] - s))
                fr = fast_ratios[best_i]
                r_frac.append(fr)
                r_float[b] = float(fr)
        else:
            for b in range(n_batch):
                fr = semitones_to_ratio(float(semis_list[b]))
                r_frac.append(fr)
                r_float[b] = float(fr)

        # Fold channels into batch for STFT
        xb = x.reshape(n_batch * n_channels, n_samples)

        spec = torch.stft(
            xb,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft, device=device, dtype=dtype),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # (n_batch*n_channels, n_freq, n_frames)
        n_frames = spec.shape[-1]
        spec_bcf = spec.reshape(n_batch, n_channels, n_freq, n_frames)

        # Group pitch shifts by unique ratio
        groups: Dict[Fraction, List[int]] = {}
        for b, fr in enumerate(r_frac):
            groups.setdefault(fr, []).append(b)

        # Get torchaudio `TimeStretch` object
        ts = self._get_ts(device=device, n_freq=n_freq, hop_length=hop_length)

        y_out = torch.empty((n_batch, n_channels, n_samples), device=device, dtype=dtype)
        tmp_sig = signal.clone()

        # Iterate over unique pitch shifts
        for fr, idxs in groups.items():
            idx = torch.tensor(idxs, device=device, dtype=torch.long)
            r = float(fr)
            rate = float(1.0 / r)

            # Select relevant spectrograms
            spec_g = spec_bcf.index_select(0, idx)  # (group_size, n_channels, n_freq, n_frames)
            group_size = spec_g.shape[0]
            spec_gc = spec_g.reshape(group_size * n_channels, n_freq, n_frames)

            # Phase vocoder time-stretch
            spec_stretched_gc = ts(spec_gc, overriding_rate=rate)  # (group_size*n_channels, n_freq, n_frames')

            # iSTFT on time-stretched spectrogram
            length_stretch = int(n_samples * r)
            y_gc = torch.istft(
                spec_stretched_gc,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=torch.hann_window(n_fft, device=device, dtype=dtype),
                center=True,
                normalized=False,
                onesided=True,
                length=length_stretch,
                return_complex=False,
            )  # (group_size*n_channels, length_stretch)

            y_gct = y_gc.reshape(group_size, n_channels, length_stretch)

            # Resample to target_sr = round(sample_rate / r), then overwrite sample rate
            target_sr = max(1, int(round(sample_rate / r)))

            tmp_sig.audio_data = y_gct
            tmp_sig.sample_rate = sample_rate

            tmp_sig = resample(tmp_sig, target_sr, inplace=True)
            tmp_sig.sample_rate = sample_rate

            yg = tmp_sig.audio_data  # (group_size, n_channels, n_samples')

            # Pad/trim to original signal length
            n_samples_out = yg.shape[-1]
            if n_samples_out > n_samples:
                yg = yg[..., :n_samples]
            elif n_samples_out < n_samples:
                pad = (0, n_samples - n_samples_out)
                yg = torch.nn.functional.pad(yg, pad, mode="constant", value=0.0)

            # Scatter back into output
            y_out.index_copy_(0, idx, yg)

        out = signal.clone()
        out.audio_data = y_out
        out.ensure_max_of_audio()
        return out