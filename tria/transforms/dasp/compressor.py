import math
from typing import Optional

import torch
from numpy.random import RandomState

from audiotools import AudioSignal
from audiotools.core.util import ensure_tensor
from audiotools.core.util import sample_from_dist

from ..base import NormalizedBaseTransform

################################################################################
# Dynamic range compression
################################################################################


def freqdomain_fir(x, H, n_fft):
    X = torch.fft.rfft(x, n_fft)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n_fft)
    return y


def fft_freqz(b, a, n_fft: int = 512):
    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)
    H = B / A
    return H


def lfilter_via_fsm(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor = None):
    """
    Approximate IIR filter via frequency sampling method.

    Parameters
    ----------
    x : torch.Tensor
        Time domain signal, shape (n_batch, 1, n_samples)
    b : torch.Tensor
        Numerator coefficients, shape (n_batch, N)
    a : torch.Tensor
        Denominator coefficients, shape (n_batch, N)

    Returns
    -------
    y : torch.Tensor
        Filtered time domain signal, shape (n_batch, 1, n_samples)
    """
    n_batch, n_channels, n_samples = x.shape
    assert n_channels == 1

    # Round up to nearest power of 2 for FFT
    n_fft = 2 ** torch.ceil(
        torch.log2(torch.tensor(n_samples + n_samples - 1, device=x.device))
    )
    n_fft = int(n_fft.item())

    b = b.to(device=x.device, dtype=x.dtype)

    if a is None:
        # Directly compute FFT of numerator coefficients
        H = torch.fft.rfft(b, n_fft)
    else:
        a = a.to(device=x.device, dtype=x.dtype)
        # Compute complex response as ratio of polynomials
        H = fft_freqz(b, a, n_fft=n_fft)

    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    # Apply as FIR filter in frequency domain
    y = freqdomain_fir(x, H, n_fft)
    y = y[..., :n_samples]

    return y


def compressor(
    x: torch.Tensor,
    sample_rate: float,
    threshold_db: torch.Tensor,
    ratio: torch.Tensor,
    attack_ms: torch.Tensor,
    release_ms: torch.Tensor,
    knee_db: torch.Tensor,
    makeup_gain_db: torch.Tensor,
    eps: float = 1e-8,
    lookahead_samples: int = 0,
    sidechain: str = "sum",
):
    n_batch, n_channels, n_samples = x.shape

    # Side-chain detector
    if sidechain == "sum":
        x_side = x.sum(dim=1, keepdim=True)  # (n_batch, 1, n_samples)
    elif sidechain == "peak":
        x_side = x.abs().amax(dim=1, keepdim=True)  # (n_batch, 1, n_samples)
    else:
        raise ValueError(f"Invalid sidechain: {sidechain}. Expected 'sum' or 'peak'.")

    threshold_db = threshold_db.view(-1, 1, 1)
    ratio = ratio.view(-1, 1, 1)
    attack_ms = attack_ms.view(-1, 1, 1)
    release_ms = release_ms.view(-1, 1, 1)
    knee_db = knee_db.view(-1, 1, 1)
    makeup_gain_db = makeup_gain_db.view(-1, 1, 1)

    # Compute time constants
    normalized_attack_time = float(sample_rate) * (attack_ms / 1e3)
    normalized_release_time = float(sample_rate) * (release_ms / 1e3)
    constant = torch.tensor([9.0], device=x.device, dtype=x.dtype)
    alpha_A = torch.exp(-torch.log(constant) / normalized_attack_time)
    alpha_R = torch.exp(-torch.log(constant) / normalized_release_time)

    # Compute energy in db
    x_db = 20 * torch.log10(torch.abs(x_side).clamp(eps))

    # Static characteristic with soft knee
    x_sc = x_db.clone()

    # When signal is at the threshold, engage knee (only when knee_db > 0)
    knee_active = knee_db > 0
    if bool(knee_active.any().item()):
        idx1 = x_db >= (threshold_db - (knee_db / 2))
        idx2 = x_db <= (threshold_db + (knee_db / 2))
        idx = torch.logical_and(idx1, idx2)
        idx = torch.logical_and(idx, knee_active)
        x_sc_below = x_db + ((1 / ratio) - 1) * (
            (x_db - threshold_db + (knee_db / 2)) ** 2
        ) / (2 * knee_db.clamp(min=eps))
        x_sc[idx] = x_sc_below[idx]

    # When signal is above threshold, linear response
    idx = x_db > (threshold_db + (knee_db / 2))
    x_sc_above = threshold_db + ((x_db - threshold_db) / ratio)
    x_sc[idx] = x_sc_above[idx]

    # Output of gain computer
    g_c = x_sc - x_db

    # Design attack smoothing filter
    b = torch.cat(
        [(1 - alpha_A), torch.zeros_like(alpha_A)],
        dim=-1,
    ).squeeze(1)
    a = torch.cat(
        [torch.ones_like(alpha_A), -alpha_A],
        dim=-1,
    ).squeeze(1)
    g_c_attack = lfilter_via_fsm(g_c, b, a)

    # Add makeup gain in db
    g_s = g_c_attack + makeup_gain_db

    # Convert db gains back to linear
    g_lin = 10 ** (g_s / 20.0)

    # Design release smoothing filter in linear domain, then enforce slow release:
    # fast attack via g_lin, slow recovery via a lowpassed follower, using min().
    b = torch.cat(
        [(1 - alpha_R), torch.zeros_like(alpha_R)],
        dim=-1,
    ).squeeze(1)
    a = torch.cat(
        [torch.ones_like(alpha_R), -alpha_R],
        dim=-1,
    ).squeeze(1)
    g_lin_release = lfilter_via_fsm(g_lin, b, a)
    g_lin = torch.minimum(g_lin, g_lin_release)

    # Look-ahead by advancing the gain curve, no output delay
    if int(lookahead_samples) > 0:
        la = int(lookahead_samples)
        g_lin = torch.roll(g_lin, -la, dims=-1)
        g_lin[:, :, -la:] = 1.0

    # Apply time-varying gain and makeup gain
    y = x * g_lin

    return y


class Compressor(NormalizedBaseTransform):
    """
    Dynamic range compressor (feedforward) with differentiable ballistics via
    FSM approximation of a 1-pole smoother.

    Parameters
    ----------
    threshold_db:
        Threshold in dBFS at which to begin gain reduction (more negative = lower).
    ratio:
        Compression ratio (> 1).
    attack_ms:
        Attack time in milliseconds.
    release_ms:
        Release time in milliseconds.
    knee_db:
        Knee width in dB (>= 0). Higher = softer knee.
    makeup_gain_db:
        Makeup gain in dB applied after compression.
    lookahead_ms:
        Lookahead in milliseconds (implemented as integer sample delay).
    sidechain:
        Side-chain detector: "sum" or "peak".
    """

    def __init__(
        self,
        threshold_db: tuple = ("uniform", -40.0, -6.0),
        ratio: tuple = ("uniform", 2.0, 16.0),
        attack_ms: tuple = ("uniform", 1.0, 50.0),
        release_ms: tuple = ("uniform", 20.0, 200.0),
        knee_db: tuple = ("uniform", 0.0, 12.0),
        makeup_gain_db: tuple = ("uniform", 0.0, 6.0),
        lookahead_ms: tuple = ("const", 0.0),
        sidechain: str = "peak",
        name: str = None,
        prob: float = 1.0,
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

        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.knee_db = knee_db
        self.makeup_gain_db = makeup_gain_db
        self.lookahead_ms = lookahead_ms
        self.sidechain = str(sidechain)
        self.eps = float(eps)

        if self.sidechain not in ("sum", "peak"):
            raise ValueError("sidechain must be 'sum' or 'peak'")

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {
            "threshold_db": sample_from_dist(self.threshold_db, state),
            "ratio": sample_from_dist(self.ratio, state),
            "attack_ms": sample_from_dist(self.attack_ms, state),
            "release_ms": sample_from_dist(self.release_ms, state),
            "knee_db": sample_from_dist(self.knee_db, state),
            "makeup_gain_db": sample_from_dist(self.makeup_gain_db, state),
            "lookahead_ms": sample_from_dist(self.lookahead_ms, state),
        }

    def _transform(
        self,
        signal,
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        knee_db,
        makeup_gain_db,
        lookahead_ms,
    ):
        x = signal.audio_data
        n_batch = signal.batch_size
        sample_rate = float(signal.sample_rate)

        threshold_db = ensure_tensor(threshold_db, 2, n_batch).to(x.device).view(-1).float()
        ratio = (
            ensure_tensor(ratio, 2, n_batch)
            .to(x.device)
            .view(-1)
            .float()
            .clamp(min=1.0 + self.eps)
        )
        attack_ms = (
            ensure_tensor(attack_ms, 2, n_batch)
            .to(x.device)
            .view(-1)
            .float()
            .clamp(min=self.eps)
        )
        release_ms = (
            ensure_tensor(release_ms, 2, n_batch)
            .to(x.device)
            .view(-1)
            .float()
            .clamp(min=self.eps)
        )
        knee_db = ensure_tensor(knee_db, 2, n_batch).to(x.device).view(-1).float().clamp(min=0.0)
        makeup_gain_db = ensure_tensor(makeup_gain_db, 2, n_batch).to(x.device).view(-1).float()

        lookahead_ms = ensure_tensor(lookahead_ms, 2, n_batch).to(x.device).view(-1).float().clamp(min=0.0)
        lookahead_samples = torch.round(lookahead_ms * (sample_rate / 1000.0)).to(torch.long)
        lookahead_samples = int(lookahead_samples.max().item()) if lookahead_samples.numel() else 0

        y = compressor(
            x,
            sample_rate=sample_rate,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            knee_db=knee_db,
            makeup_gain_db=makeup_gain_db,
            eps=self.eps,
            lookahead_samples=lookahead_samples,
            sidechain=self.sidechain,
        )

        signal.audio_data = y
        signal.stft_data = None
        return signal