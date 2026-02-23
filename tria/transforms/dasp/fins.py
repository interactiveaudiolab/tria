import copy
import math
from functools import lru_cache

from typing import List, Optional, Union
from pathlib import Path

import torch
import numpy as np
import scipy.signal

from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import sample_from_dist, ensure_tensor
from audiotools.data.transforms import BaseTransform
from numpy.random import RandomState

from ...dsp import apply_ir
from ..base import NormalizedBaseTransform

################################################################################
# Reverberation via filtered noise shaping of impulse responses
################################################################################


_OCTAVE_BANDS = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


@lru_cache(maxsize=32)
def _octave_band_filterbank(num_taps: int, sample_rate: int) -> torch.Tensor:

    assert num_taps % 2 == 1
    sr = float(sample_rate)

    filts = []

    # Lowpass filter
    filt = scipy.signal.firwin(int(num_taps), 12.0, fs=sr)
    filt = torch.from_numpy(filt.astype("float32"))
    filts.append(torch.flip(filt, dims=[0]))

    # Bandpass filters
    for fc in _OCTAVE_BANDS:
        f_min = float(fc) / math.sqrt(2.0)
        f_max = float(fc) * math.sqrt(2.0)
        f_max = float(np.clip(f_max, a_min=0.0, a_max=(sr / 2.0) * 0.999))
        filt = scipy.signal.firwin(
            int(num_taps),
            [f_min, f_max],
            fs=sr,
            pass_zero=False,
        )
        filt = torch.from_numpy(filt.astype("float32"))
        filts.append(torch.flip(filt, dims=[0]))

    # Highpass filter
    filt = scipy.signal.firwin(int(num_taps), 18000.0, fs=sr, pass_zero=False)
    filt = torch.from_numpy(filt.astype("float32"))
    filts.append(torch.flip(filt, dims=[0]))

    filts = torch.stack(filts, dim=0)[:, None, :]  # (num_bands, 1, num_taps)
    return filts


class FiNSReverb(NormalizedBaseTransform):
    """
    Construct impulse responses by summing independently shaped and filtered 
    noise across octave-spaced frequency bands; adapted from 
    https://github.com/csteinmetz1/dasp-pytorch/blob/main/dasp_pytorch/functional.py

    Parameters
    ----------
    ir_duration:
        Impulse response duration in seconds
    gain_amount:
        Gain per band in linear amplitude
    decay:
        Decay time per band in seconds
    attack:
        Attack time per band in seconds
    mix:
        Wet/dry mix ratio in linear amplitude
    """
    
    def __init__(
        self,
        ir_duration: tuple = ("uniform", 0.05, 1.5),
        gain_amount: tuple = ("uniform", 0.0, 1.0),
        decay: tuple = ("uniform", 0.03, 0.8),
        attack: tuple = ("const", 0.0),
        mix: tuple = ("uniform", 0.5, 1.0),
        num_bandpass_taps: int = 1023,
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        start_at_peak: bool = False,
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
        self.ir_duration = ir_duration
        self.gain_amount = gain_amount
        self.decay = decay
        self.attack = attack
        self.mix = mix
        self.num_bandpass_taps = int(num_bandpass_taps)
        self.use_original_phase = bool(use_original_phase)
        self.start_at_peak = bool(start_at_peak)
        self.eps = float(eps)

        if self.num_bandpass_taps % 2 != 1:
            raise ValueError("num_bandpass_taps must be odd")

        self.num_bands = len(_OCTAVE_BANDS) + 2  # lowpass + (bands) + highpass

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        ir_duration = float(sample_from_dist(self.ir_duration, state))
        mix = float(sample_from_dist(self.mix, state))

        # Sample per-band gains and decays
        gain_amount = float(sample_from_dist(self.gain_amount, state))
        gains = gain_amount * state.rand(self.num_bands)  # (num_bands,)

        tau_lo, tau_hi = None, None
        taus = np.asarray(
            [float(sample_from_dist(self.decay, state)) for _ in range(self.num_bands)],
            dtype=np.float32,
        )

        # Sample per-band attacks
        attacks = np.asarray(
            [float(sample_from_dist(self.attack, state)) for _ in range(self.num_bands)],
            dtype=np.float32,
        )

        return {
            "ir_duration": ir_duration,
            "band_gains": gains,
            "band_taus": taus,
            "band_attacks": attacks,
            "mix": mix,
        }

    def _transform(self, signal, ir_duration, band_gains, band_taus, band_attacks, mix):
        x = signal.audio_data
        n_batch = signal.batch_size
        sr = int(signal.sample_rate)
        n_channels = int(signal.num_channels)

        mix = (
            ensure_tensor(mix, 2, n_batch)
            .to(x.device)
            .view(-1, 1, 1)
            .float()
            .clamp(0.0, 1.0)
        )

        ir_duration = (
            ensure_tensor(ir_duration, 2, n_batch)
            .to(x.device)
            .view(-1)
            .float()
            .clamp(min=0.0)
        )

        band_gains = ensure_tensor(band_gains, 2).to(x.device).float()
        if band_gains.ndim == 1:
            band_gains = band_gains[None, :].expand(n_batch, -1)
        band_gains = band_gains.clamp(min=0.0).view(n_batch, 1, self.num_bands, 1)

        band_taus = ensure_tensor(band_taus, 2).to(x.device).float()
        if band_taus.ndim == 1:
            band_taus = band_taus[None, :].expand(n_batch, -1)
        band_taus = band_taus.clamp(min=self.eps).view(n_batch, 1, self.num_bands, 1)

        band_attacks = ensure_tensor(band_attacks, 2).to(x.device).float()
        if band_attacks.ndim == 1:
            band_attacks = band_attacks[None, :].expand(n_batch, -1)
        band_attacks = band_attacks.clamp(min=0.0).view(n_batch, 1, self.num_bands, 1)

        # Determine IR length in samples
        lengths = torch.clamp((ir_duration * float(sr)).ceil().to(torch.long), min=1)
        max_len = int(lengths.max().item())

        # Construct filters
        filters = _octave_band_filterbank(self.num_bandpass_taps, sr).to(
            device=x.device, dtype=x.dtype
        )
        num_bands = int(filters.shape[0])
        pad_size = self.num_bandpass_taps - 1

        # Generate white noise for IR generation
        wn = torch.randn(
            (n_batch * n_channels, num_bands, max_len + pad_size),
            device=x.device,
            dtype=x.dtype,
        )

        # Filter white noise with each bandpass filter via grouped convolution
        wn_filt = torch.nn.functional.conv1d(
            wn, filters, groups=num_bands
        )  # (n_batch*n_channels, num_bands, n_samples)
        wn_filt = wn_filt.view(n_batch, n_channels, num_bands, max_len)

        # Apply per-band exponential envelopes: exp(-t / tau)
        t = (torch.arange(max_len, device=x.device, dtype=x.dtype) / float(sr)).view(
            1, 1, 1, -1
        )
        env = torch.exp(-t / band_taus.to(x.dtype))  # (n_batch, 1, num_bands, n_samples)

        # Apply per-band attack envelopes
        attack_env = (t / band_attacks.clamp(min=self.eps).to(x.dtype)).clamp(0.0, 1.0)
        attack_env = torch.where(
            band_attacks.to(x.dtype) <= self.eps,
            torch.ones_like(attack_env),
            attack_env,
        )

        wn_filt = wn_filt * env * attack_env * band_gains  # (n_batch, n_channels, num_bands, n_samples)

        # Average across bands to create IR
        ir = wn_filt.mean(dim=2)  # (n_batch, n_channels, n_samples)

        # Zero tail past per-item lengths
        mask = (torch.arange(max_len, device=x.device)[None, :] < lengths[:, None])  # (n_batch, n_samples)
        ir = ir * mask[:, None, :].to(ir.dtype)

        ir_signal = AudioSignal(ir, sample_rate=sr)

        dry = signal.clone()
        wet = apply_ir(
            signal,
            ir_signal,
            drr=None,
            ir_eq=None,
            use_original_phase=self.use_original_phase,
            start_at_peak=self.start_at_peak,
        )

        wet.audio_data = mix * wet.audio_data + (1.0 - mix) * dry.audio_data
        wet.stft_data = None
        return wet



