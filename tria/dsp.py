import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import julius
from audiotools import AudioSignal
from audiotools.core.util import ensure_tensor, sample_from_dist

################################################################################
# Signal processing utilities for filters, etc.
################################################################################


@dataclass(frozen=True)
class FilterKey:
    """
    Cache low/high-pass kernels to avoid re-initialization and device movement
    """
    kind: str              # "lp" or "hp"
    cutoff_norm_q: float   # normalized cutoff (Hz / sr), quantized
    zeros: int
    device: str
    dtype: torch.dtype


@dataclass(frozen=True)
class ResampleKey:
    """
    Cache resampling kernels to avoid re-initialization and device movement
    """
    old_sr: int
    new_sr: int
    zeros: int
    rolloff: float
    device: str
    dtype: torch.dtype


class FilterCache:
    """
    Cache julius.{LowPassFilter,HighPassFilter} modules with memory-based eviction.

    - Per-process (DDP-safe by construction)
    - Per-device budgets (e.g. "cuda:0", "cuda:1", "cpu")
    - LRU eviction within each device when over budget
    """

    def __init__(
        self,
        max_bytes_per_device: Optional[Dict[str, int]] = None,
        default_max_bytes: int = 256 * 1024 * 1024,  # 256MB
    ):
        self.default_max_bytes = int(default_max_bytes)
        self.max_bytes_per_device = dict(max_bytes_per_device or {})

        self._cache: Dict[str, "OrderedDict[FilterKey, torch.nn.Module]"] = {}
        self._bytes: Dict[str, int] = {}

    def set_max_bytes(self, device: str, max_bytes: int):
        self.max_bytes_per_device[str(device)] = int(max_bytes)

    def get_max_bytes(self, device: str) -> int:
        return int(self.max_bytes_per_device.get(str(device), self.default_max_bytes))

    def clear(self, device: Optional[str] = None):
        if device is None:
            self._cache.clear()
            self._bytes.clear()
            return
        d = str(device)
        self._cache.pop(d, None)
        self._bytes.pop(d, None)

    @staticmethod
    def _module_bytes(m: torch.nn.Module) -> int:
        kb = 0
        for _, buf in m.named_buffers(recurse=True):
            kb += buf.numel() * buf.element_size()
        for _, p in m.named_parameters(recurse=True):
            kb += p.numel() * p.element_size()
        return int(kb)

    def _evict_if_needed(self, device: str):
        d = str(device)
        budget = self.get_max_bytes(d)
        if budget <= 0:
            self.clear(d)
            return

        cache_d = self._cache.get(d)
        if not cache_d:
            return

        cur = int(self._bytes.get(d, 0))
        while cache_d and cur > budget:
            _, m = cache_d.popitem(last=False)
            cur -= self._module_bytes(m)
        self._bytes[d] = max(0, cur)

    def get(
        self,
        *,
        kind: str,
        cutoff_norm_q: float,
        zeros: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        kind = str(kind)
        if kind not in ("lp", "hp"):
            raise ValueError(f"kind must be 'lp' or 'hp', got {kind}")

        d = str(device)
        if self.get_max_bytes(d) <= 0:
            # caching disabled
            if kind == "lp":
                return julius.LowPassFilter(float(cutoff_norm_q), zeros=int(zeros)).to(device=device, dtype=dtype)
            else:
                return julius.HighPassFilter(float(cutoff_norm_q), zeros=int(zeros)).to(device=device, dtype=dtype)

        key = FilterKey(
            kind=kind,
            cutoff_norm_q=float(cutoff_norm_q),
            zeros=int(zeros),
            device=d,
            dtype=dtype,
        )

        cache_d = self._cache.setdefault(d, OrderedDict())
        m = cache_d.get(key)
        if m is not None:
            cache_d.move_to_end(key)
            return m

        if kind == "lp":
            m = julius.LowPassFilter(float(cutoff_norm_q), zeros=int(zeros)).to(device=device, dtype=dtype)
        else:
            m = julius.HighPassFilter(float(cutoff_norm_q), zeros=int(zeros)).to(device=device, dtype=dtype)

        cache_d[key] = m
        cache_d.move_to_end(key)

        self._bytes[d] = int(self._bytes.get(d, 0)) + self._module_bytes(m)
        self._evict_if_needed(d)

        return m


class ResampleCache:
    """
    Cache julius.ResampleFrac modules with memory-based eviction.

    - Per-process (DDP-safe by construction: each rank is its own process)
    - Separate LRU per device string (e.g. "cuda:0", "cpu")
    - Memory-based eviction within each device
    """

    def __init__(
        self,
        max_bytes_per_device: Optional[Dict[str, int]] = None,
        default_max_bytes: int = 512 * 1024 * 1024,  # 512MB
    ):
        self.default_max_bytes = int(default_max_bytes)
        self.max_bytes_per_device = dict(max_bytes_per_device or {})

        self._cache: Dict[str, "OrderedDict[ResampleKey, torch.nn.Module]"] = {}
        self._bytes: Dict[str, int] = {}

    def set_max_bytes(self, device: str, max_bytes: int):
        self.max_bytes_per_device[str(device)] = int(max_bytes)

    def get_max_bytes(self, device: str) -> int:
        return int(self.max_bytes_per_device.get(str(device), self.default_max_bytes))

    def clear(self, device: Optional[str] = None):
        if device is None:
            self._cache.clear()
            self._bytes.clear()
            return
        d = str(device)
        self._cache.pop(d, None)
        self._bytes.pop(d, None)

    @staticmethod
    def _module_bytes(m: torch.nn.Module) -> int:
        kb = 0
        for _, buf in m.named_buffers(recurse=True):
            kb += buf.numel() * buf.element_size()
        for _, p in m.named_parameters(recurse=True):
            kb += p.numel() * p.element_size()
        return int(kb)

    def _evict_if_needed(self, device: str):
        d = str(device)
        budget = self.get_max_bytes(d)
        if budget <= 0:
            # budget 0 means "don't cache"
            self.clear(d)
            return

        cache_d = self._cache.get(d)
        if not cache_d:
            return

        cur = int(self._bytes.get(d, 0))
        while cache_d and cur > budget:
            _, m = cache_d.popitem(last=False)
            cur -= self._module_bytes(m)
        self._bytes[d] = max(0, cur)

    def get(
        self,
        old_sr: int,
        new_sr: int,
        *,
        zeros: int,
        rolloff: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        old_sr = int(old_sr)
        new_sr = int(new_sr)
        d = str(device)

        if self.get_max_bytes(d) <= 0:
            return julius.ResampleFrac(old_sr, new_sr, int(zeros), float(rolloff)).to(
                device=device, dtype=dtype
            )

        key = ResampleKey(
            old_sr=old_sr,
            new_sr=new_sr,
            zeros=int(zeros),
            rolloff=float(rolloff),
            device=d,
            dtype=dtype,
        )

        cache_d = self._cache.setdefault(d, OrderedDict())
        m = cache_d.get(key)
        if m is not None:
            cache_d.move_to_end(key)
            return m

        m = julius.ResampleFrac(old_sr, new_sr, int(zeros), float(rolloff)).to(
            device=device, dtype=dtype
        )

        cache_d[key] = m
        cache_d.move_to_end(key)

        self._bytes[d] = int(self._bytes.get(d, 0)) + self._module_bytes(m)
        self._evict_if_needed(d)

        return m


def set_filter_cache_max_bytes(device: str, max_bytes: int):
    _FILTER_CACHE.set_max_bytes(device, max_bytes)


def clear_filter_cache(device: Optional[str] = None):
    _FILTER_CACHE.clear(device=device)

def set_resample_cache_max_bytes(device: str, max_bytes: int):
    _RESAMPLE_CACHE.set_max_bytes(device, max_bytes)


def clear_resample_cache(device: Optional[str] = None):
    _RESAMPLE_CACHE.clear(device=device)


# Initialize caches per process
_FILTER_CACHE = FilterCache()
_RESAMPLE_CACHE = ResampleCache()


def _quantize_hz(x_hz: torch.Tensor, q_hz: float) -> torch.Tensor:
    # q_hz <= 0 => no quantization (but that can explode cache)
    q = float(q_hz)
    if q <= 0:
        return x_hz
    return torch.round(x_hz / q) * q


@torch.no_grad()
def low_pass(
    sig: AudioSignal,
    cutoffs_hz: Union[torch.Tensor, np.ndarray, float],
    *,
    zeros: int = 51,
    quantize_hz: float = 1.0,
    inplace: bool = True,
) -> AudioSignal:
    """
    Cached Julius low-pass on an AudioSignal. Efficient on GPU.

    cutoffs_hz can be float or (B,) tensor/array. Per-item cutoffs supported.
    """
    out = sig if inplace else sig.clone()
    B = int(out.batch_size)
    sr = float(out.sample_rate)

    # Ensure (B,) tensor on device
    cut = ensure_tensor(cutoffs_hz, 2, B).to(out.device).view(-1).float()
    cut = cut.clamp(min=0.0, max=0.5 * sr)

    # Quantize in Hz to keep cache bounded
    cut_q = _quantize_hz(cut, quantize_hz)

    # Normalize (0..0.5)
    cut_norm_q = (cut_q / sr).clamp(min=0.0, max=0.5)

    x = out.audio_data  # (B,C,T)
    y = torch.empty_like(x)

    # Group identical quantized cutoffs to reduce module lookups/calls
    # Use CPU unique for tiny vectors (B usually small); avoids GPU sync drama.
    uniq = torch.unique(cut_norm_q.detach().cpu())
    for v in uniq.tolist():
        m = _FILTER_CACHE.get(
            kind="lp",
            cutoff_norm_q=float(v),
            zeros=int(zeros),
            device=x.device,
            dtype=x.dtype,
        )
        mask = (cut_norm_q == float(v))
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        y[idx] = m(x[idx])

    out.audio_data = y
    out.stft_data = None
    return out


@torch.no_grad()
def high_pass(
    sig: AudioSignal,
    cutoffs_hz: Union[torch.Tensor, np.ndarray, float],
    *,
    zeros: int = 51,
    quantize_hz: float = 1.0,
    inplace: bool = True,
) -> AudioSignal:
    """
    Cached Julius high-pass on an AudioSignal. Efficient on GPU.

    cutoffs_hz can be float or (B,) tensor/array. Per-item cutoffs supported.
    """
    out = sig if inplace else sig.clone()
    B = int(out.batch_size)
    sr = float(out.sample_rate)

    cut = ensure_tensor(cutoffs_hz, 2, B).to(out.device).view(-1).float()
    cut = cut.clamp(min=0.0, max=0.5 * sr)

    cut_q = _quantize_hz(cut, quantize_hz)
    cut_norm_q = (cut_q / sr).clamp(min=0.0, max=0.5)

    x = out.audio_data
    y = torch.empty_like(x)

    uniq = torch.unique(cut_norm_q.detach().cpu())
    for v in uniq.tolist():
        m = _FILTER_CACHE.get(
            kind="hp",
            cutoff_norm_q=float(v),
            zeros=int(zeros),
            device=x.device,
            dtype=x.dtype,
        )
        mask = (cut_norm_q == float(v))
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        y[idx] = m(x[idx])

    out.audio_data = y
    out.stft_data = None
    return out


@torch.no_grad()
def resample_tensor(
    x: torch.Tensor,
    old_sr: int,
    new_sr: int,
    *,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
    kernel_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Cached julius resampler. Works on CPU/CUDA depending on x.device.

    x: (..., T) with time last. AudioSignal.audio_data is (B,C,T) so it works.
    """
    old_sr = int(old_sr)
    new_sr = int(new_sr)
    if old_sr == new_sr:
        return x

    kd = kernel_dtype if kernel_dtype is not None else x.dtype
    m = _RESAMPLE_CACHE.get(
        old_sr,
        new_sr,
        zeros=zeros,
        rolloff=rolloff,
        device=x.device,
        dtype=kd,
    )
    # julius accepts (.., T) time-last; output_length/full forwarded
    y = m(x, output_length=output_length, full=full)

    # Ensure output dtype matches input dtype unless caller explicitly wanted kernel dtype
    if y.dtype != x.dtype:
        y = y.to(dtype=x.dtype)
    return y


@torch.no_grad()
def resample(
    sig: AudioSignal,
    new_sr: int,
    *,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
    inplace: bool = True,
    kernel_dtype: Optional[torch.dtype] = None,
) -> AudioSignal:
    """
    Resample an AudioSignal using cached julius kernels.

    By default, modifies `sig` in-place (mirrors AudioSignal.resample behavior).
    """
    new_sr = int(new_sr)
    if int(sig.sample_rate) == new_sr:
        return sig

    out = sig if inplace else sig.clone()
    out.audio_data = resample_tensor(
        out.audio_data,
        int(out.sample_rate),
        new_sr,
        zeros=zeros,
        rolloff=rolloff,
        output_length=output_length,
        full=full,
        kernel_dtype=kernel_dtype,
    )
    out.sample_rate = new_sr
    return out


def _roll_to_peak(x: torch.Tensor) -> torch.Tensor:
    """
    Roll each IR so the maximum magnitude lands at t=0, avoiding delay.
    """
    n_batch, n_channels, n_samples = x.shape
    
    flat = x.abs().reshape(n_batch, n_channels * n_samples)
    idx_lin = flat.argmax(dim=-1)                 # (n_batch,)
    idx_t = (idx_lin % n_samples).to(torch.long)  # (n_batch,)

    t = torch.arange(n_samples, device=x.device)  # (n_samples,)
    gather_idx = (t[None, None, :] + idx_t[:, None, None]) % n_samples  # (n_batch, 1, n_samples)
    gather_idx = gather_idx.expand(n_batch, n_channels, n_samples)
    return x.gather(dim=-1, index=gather_idx)


def _convolve(
    signal: AudioSignal,
    ir: AudioSignal,
    *,
    start_at_peak: bool = True,
) -> AudioSignal:
    """
    Perform convolution via FFT multiplication, padding/truncating IR to signal 
    length and optionally rolling IR to peak to avoid delay,
    """
    ir = ir.clone()

    pad_len = signal.signal_length - ir.signal_length
    if pad_len > 0:
        ir.zero_pad(0, pad_len)
    else:
        ir.truncate_samples(signal.signal_length)

    if ir.num_channels not in (1, signal.num_channels):
        raise ValueError(
            f"IR channels ({ir.num_channels}) must be 1 or match signal channels ({signal.num_channels})."
        )
    if ir.num_channels == 1 and signal.num_channels > 1:
        ir.audio_data = ir.audio_data.repeat(1, signal.num_channels, 1)

    if start_at_peak:
        ir = AudioSignal(_roll_to_peak(ir.audio_data), ir.sample_rate)

    length = signal.signal_length
    delta = torch.zeros_like(ir.audio_data)
    delta[..., 0] = 1

    delta_fft = torch.fft.rfft(delta, length)
    ir_fft = torch.fft.rfft(ir.audio_data, length)
    sig_fft = torch.fft.rfft(signal.audio_data, length)

    y = torch.fft.irfft(ir_fft * sig_fft, length)
    delta_y = torch.fft.irfft(ir_fft * delta_fft, length)

    delta_max = delta_y.abs().max(dim=-1, keepdim=True).values  # (B,C,1)
    y = y * (1.0 / delta_max.clamp(1e-5))

    signal.audio_data = y
    return signal


def apply_ir(
    signal: AudioSignal,
    ir: AudioSignal,
    drr=None,
    ir_eq=None,
    use_original_phase: bool = False,
    start_at_peak: bool = True,
) -> AudioSignal:
    
    ir = ir.clone()

    if ir_eq is not None:
        ir = ir.equalizer(ir_eq)
    if drr is not None:
        ir = ir.alter_drr(drr)

    max_spk = signal.audio_data.abs().max(dim=-1, keepdim=True).values

    phase = signal.phase if use_original_phase else None
    _convolve(signal, ir, start_at_peak=start_at_peak)

    if use_original_phase:
        signal.stft()
        signal.stft_data = signal.magnitude * torch.exp(1j * phase)
        signal.istft()

    max_transformed = signal.audio_data.abs().max(dim=-1, keepdim=True).values
    scale_factor = max_spk.clamp(1e-8) / max_transformed.clamp(1e-8)
    signal.audio_data = signal.audio_data * scale_factor

    return signal

