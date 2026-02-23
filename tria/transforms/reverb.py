import copy
import math
from functools import lru_cache

from typing import List, Optional, Union
from pathlib import Path

import torch
import numpy as np

from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools.core.util import sample_from_dist, ensure_tensor
from numpy.random import RandomState

from ..dsp import resample, apply_ir
from ..util import read_manifest, normalize_source_weights, load_excerpt
from .base import NormalizedBaseTransform

################################################################################
# Reverberation
################################################################################


class NoiseReverb(NormalizedBaseTransform):
    """
    Synthetic reverb via exponentially-decaying white-noise impulse response.

    IR(t) = N(0,1) * attack(t) * exp(-t / decay)

    Parameters
    ----------
    decay : tuple
        Exponential decay time constant in seconds (tau). Larger => longer tail.
    ir_duration : tuple
        Impulse length in seconds.
    attack : tuple
        Attack time in seconds. Typical values are very small (e.g. < 20 ms).
    eq_amount : tuple
        Same semantics as your IR reverb: eq = -eq_amount * rand(n_bands)
    mix : tuple
        Wet/dry mix in [0,1]
    start_at_peak : bool
        If True, rely on _apply_ir/_convolve path to roll IR so peak is at t=0.
    """

    def __init__(
        self,
        decay: tuple = ("uniform", 0.05, 0.6),
        ir_duration: tuple = ("uniform", 0.05, 0.5),
        attack: tuple = ("uniform", 0.0, 0.02),
        eq_amount: tuple = ("const", 0.0),
        n_bands: int = 6,
        mix: tuple = ("uniform", 0.1, 0.7),
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        start_at_peak: bool = True,
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
        self.decay = decay
        self.ir_duration = ir_duration
        self.attack = attack
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.mix = mix
        self.use_original_phase = use_original_phase
        self.start_at_peak = start_at_peak
        self.eps = eps

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        decay = sample_from_dist(self.decay, state)
        ir_duration = sample_from_dist(self.ir_duration, state)
        attack = sample_from_dist(self.attack, state)
        mix = sample_from_dist(self.mix, state)

        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)

        return {
            "decay": decay,
            "ir_duration": ir_duration,
            "attack": attack,
            "eq": eq,
            "mix": mix,
        }

    def _transform(self, signal, decay, ir_duration, attack, eq, mix):
        x = signal.audio_data
        n_batch = signal.batch_size
        sr = signal.sample_rate
        n_channels = signal.num_channels

        decay = ensure_tensor(decay, 2, n_batch).to(x.device).view(-1).float().clamp(min=self.eps)
        ir_duration = (
            ensure_tensor(ir_duration, 2, n_batch)
            .to(x.device)
            .view(-1)
            .float()
            .clamp(min=0.0)
        )
        attack = (
            ensure_tensor(attack, 2, n_batch)
            .to(x.device)
            .view(-1)
            .float()
            .clamp(min=0.0)
        )
        mix = ensure_tensor(mix, 2, n_batch).to(x.device).view(-1, 1, 1).float().clamp(0.0, 1.0)

        eq = ensure_tensor(eq, 2).to(x.device).float()
        if eq.ndim == 1:
            eq = eq[None, :].expand(n_batch, -1)

        # Build exponential-envelope noise IR with attack
        lengths = torch.clamp((ir_duration * sr).ceil().to(torch.long), min=1)
        max_len = int(lengths.max().item())

        t = (
            torch.arange(max_len, device=x.device, dtype=x.dtype)[None, None, :]
            / float(sr)
        )  # (1, 1, n_samples)

        decay_env = torch.exp(-t / decay.view(-1, 1, 1).to(x.dtype))

        attack_env = (t / attack.view(-1, 1, 1).clamp(min=self.eps)).clamp(0.0, 1.0)

        env = attack_env * decay_env

        ir = torch.randn(
            (n_batch, n_channels, max_len),
            device=x.device,
            dtype=x.dtype,
        ) * env

        # Zero tail past specified lengths
        mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
        ir = ir * mask[:, None, :].to(ir.dtype)

        ir_signal = AudioSignal(ir, sample_rate=sr)

        dry = signal.clone()
        wet = apply_ir(
            signal,
            ir_signal,
            drr=None,
            ir_eq=eq,
            use_original_phase=self.use_original_phase,
            start_at_peak=self.start_at_peak,
        )

        wet.audio_data = mix * wet.audio_data + (1.0 - mix) * dry.audio_data
        wet.stft_data = None
        return wet
        

class RoomImpulseResponse(NormalizedBaseTransform):
    """
    Patches device error in `audiotools.data.transforms.RoomImpulseResponse` to
    allow processing on GPU.
    """

    def __init__(
        self,
        drr: tuple = ("uniform", 0.0, 30.0),
        sources: List[str] = None,
        source_weights: Optional[List[float]] = None,
        relative_path: str = "",
        path_col: str = "path",
        eq_amount: tuple = ("const", 1.0),
        n_bands: int = 6,
        name: str = None,
        prob: float = 1.0,
        use_original_phase: bool = False,
        offset: float = 0.0,
        duration: float = 1.0,
        start_at_peak: bool = True,
        resample_ir: bool = True,
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
        self.drr = drr
        self.eq_amount = eq_amount
        self.n_bands = n_bands
        self.use_original_phase = use_original_phase

        self.offset = float(offset)
        self.duration = float(duration)
        self.path_col = path_col
        self.start_at_peak = start_at_peak
        self.resample_ir = resample_ir

        per_source = read_manifest(
            sources=sources,
            columns=[path_col],
            relative_path=relative_path,
            strict=True,
        )
        kept_mask = [len(lst) > 0 for lst in per_source]
        self.source_rows = [lst for lst in per_source if len(lst) > 0]
        if len(self.source_rows) == 0:
            raise RuntimeError("Reverb: no valid IR rows after filtering.")

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

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        eq_amount = sample_from_dist(self.eq_amount, state)
        eq = -eq_amount * state.rand(self.n_bands)
        drr = sample_from_dist(self.drr, state)

        ir_path = self._pick_path(state)
        if not ir_path or not Path(ir_path).is_file():
            raise RuntimeError(f"Reverb: sampled invalid IR path: {ir_path}")
        
        ir_sig, _ = load_excerpt(
            ir_path,
            duration=self.duration,
            sample_rate=int(signal.sample_rate),
            state=state,
            from_start=True,
            loudness_cutoff=None,
            num_tries=None,
            offset=0.0,
            num_channels=int(signal.num_channels),
            resample=False,
        )
        if self.resample_ir:    
            ir_sig = resample(ir_sig.to(signal.device), signal.sample_rate)
        else:
            # Re-interpret / overwrite RIR sample rate, effectively 
            # "speeding up/down" to match signal sample rate
            ir_sig = ir_sig.to(signal.device)
            ir_sig.sample_rate = signal.sample_rate
        
        ir_sig.zero_pad_to(int(signal.sample_rate))

        return {"eq": eq, "ir_signal": ir_sig, "drr": drr}

    def _transform(self, signal, ir_signal, drr, eq):
        if isinstance(drr, torch.Tensor):
            drr = drr.to(signal.device)
        if isinstance(eq, torch.Tensor):
            eq = eq.to(signal.device)
        ir_signal = ir_signal.clone().to(signal.device)
        return apply_ir(
            signal, ir_signal, drr, eq, 
            use_original_phase=self.use_original_phase,
            start_at_peak=self.start_at_peak,
        )
