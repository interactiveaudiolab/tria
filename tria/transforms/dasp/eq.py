import math
from typing import Optional, Tuple

import torch
from numpy.random import RandomState

from audiotools import AudioSignal
from audiotools.core.util import ensure_tensor
from audiotools.core.util import sample_from_dist

from ..base import NormalizedBaseTransform

################################################################################
# Parametric EQ via frequency sampling method (FSM) on a biquad cascade
################################################################################


def _next_pow_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _biquad(
    gain_db: torch.Tensor,
    cutoff_freq: torch.Tensor,
    q_factor: torch.Tensor,
    sample_rate: float,
    filter_type: str = "peaking",
):
    assert filter_type in [
        "high_shelf", "low_shelf", "peaking", "low_pass", "high_pass"
    ]
    n_batch = gain_db.shape[0]

    gain_db = gain_db.view(n_batch, -1)
    cutoff_freq = cutoff_freq.view(n_batch, -1)
    q_factor = q_factor.view(n_batch, -1)

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / float(sample_rate))
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = torch.stack([b0, b1, b2], dim=1).view(n_batch, -1)
    a = torch.stack([a0, a1, a2], dim=1).view(n_batch, -1)

    # Normalize by a0
    b = b.type_as(gain_db) / a0
    a = a.type_as(gain_db) / a0

    return b, a


def _sosfreqz_fsm(sos: torch.Tensor, n_fft: int):
    """
    Second-order section (SOS) cascade via FFT.

    Parameters
    ----------
    sos : torch.Tensor
        Shape (n_batch, n_sections, 6) where final dimension holds
        [b0 b1 b2 a0 a1 a2] already normalized by a0
    n_fft : int
        Number of FFT bins

    Returns
    -------
    torch.Tensor
        Frequency-domain filter, shape (n_batch, n_bins)
    """
    n_batch, n_sections, n_coeffs = sos.shape
    assert n_coeffs == 6

    b = sos[:, :, :3]  # (n_batch, S, 3)
    a = sos[:, :, 3:]  # (n_batch, S, 3)

    # Zero-pad numerator/denominator polynomials to n_fft and FFT along last dim.
    # rfft output bins: n_fft//2 + 1
    b_pad = torch.nn.functional.pad(b, (0, int(n_fft) - 3))
    a_pad = torch.nn.functional.pad(a, (0, int(n_fft) - 3))

    B = torch.fft.rfft(b_pad, n=int(n_fft), dim=-1)
    A = torch.fft.rfft(a_pad, n=int(n_fft), dim=-1)

    H_sec = B / A
    H = H_sec.prod(dim=1)  # Cascade multiply across sections

    return H  # (n_batch, n_bins)


def _sosfilt_via_fsm(sos: torch.Tensor, x: torch.Tensor):
    """
    Apply second-order section (SOS) cascade via frequency sampling method.

    Parameters
    ----------
    sos : torch.Tensor
        Shape (n_batch, n_sections, 6) where final dimension holds
        [b0 b1 b2 a0 a1 a2] already normalized by a0
    x : torch.Tensor
        Input signal, shape (n_batch, n_channels, n_samples)

    Returns
    -------
    torch.Tensor
        Filtered signal, shape (n_batch, n_channels, n_samples)
    """
    assert x.ndim == 3
    n_batch, _, n_samples = x.shape

    n_fft = _next_pow_two((2 * n_samples) - 1)
    H = _sosfreqz_fsm(sos, n_fft=n_fft)  # (n_batch, n_bins)

    # Broadcast H over all non-time dims except batch
    for _ in range(x.ndim - 2):
        H = H.unsqueeze(1)

    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    Y = X * H.type_as(X)
    y = torch.fft.irfft(Y, n=n_fft, dim=-1)

    return y[..., :n_samples]


class ParametricEqualizer(NormalizedBaseTransform):
    """
    Six-band parametric EQ (low-shelf + 4 peaking + high-shelf) applied via
    frequency sampling method of a biquad cascade.

    Parameters
    ----------
    low_shelf_gain_db, band*_gain_db, high_shelf_gain_db:
        Gains in dB
    low_shelf_cutoff_hz, band*_cutoff_hz, high_shelf_cutoff_hz:
        Cutoffs/center frequencies in Hz
    low_shelf_q, band*_q, high_shelf_q:
        Q-factors
    """

    def __init__(
        self,
        low_shelf_gain_db: tuple = ("uniform", -6.0, 6.0),
        low_shelf_cutoff_hz: tuple = ("choice", (60.0, 80.0, 120.0, 160.0)),
        low_shelf_q: tuple = ("uniform", 0.5, 1.2),
        band0_gain_db: tuple = ("uniform", -8.0, 8.0),
        band0_cutoff_hz: tuple = ("choice", (250.0, 350.0, 500.0, 700.0)),
        band0_q: tuple = ("uniform", 0.5, 3.0),
        band1_gain_db: tuple = ("uniform", -8.0, 8.0),
        band1_cutoff_hz: tuple = ("choice", (800.0, 1000.0, 1400.0, 2000.0)),
        band1_q: tuple = ("uniform", 0.5, 3.0),
        band2_gain_db: tuple = ("uniform", -8.0, 8.0),
        band2_cutoff_hz: tuple = ("choice", (2500.0, 3500.0, 5000.0, 7000.0)),
        band2_q: tuple = ("uniform", 0.5, 3.0),
        band3_gain_db: tuple = ("uniform", -8.0, 8.0),
        band3_cutoff_hz: tuple = ("choice", (8000.0, 10000.0, 12000.0)),
        band3_q: tuple = ("uniform", 0.5, 3.0),
        high_shelf_gain_db: tuple = ("uniform", -6.0, 6.0),
        high_shelf_cutoff_hz: tuple = ("choice", (8000.0, 10000.0, 12000.0, 14000.0)),
        high_shelf_q: tuple = ("uniform", 0.5, 1.2),
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

        self.low_shelf_gain_db = low_shelf_gain_db
        self.low_shelf_cutoff_hz = low_shelf_cutoff_hz
        self.low_shelf_q = low_shelf_q

        self.band0_gain_db = band0_gain_db
        self.band0_cutoff_hz = band0_cutoff_hz
        self.band0_q = band0_q

        self.band1_gain_db = band1_gain_db
        self.band1_cutoff_hz = band1_cutoff_hz
        self.band1_q = band1_q

        self.band2_gain_db = band2_gain_db
        self.band2_cutoff_hz = band2_cutoff_hz
        self.band2_q = band2_q

        self.band3_gain_db = band3_gain_db
        self.band3_cutoff_hz = band3_cutoff_hz
        self.band3_q = band3_q

        self.high_shelf_gain_db = high_shelf_gain_db
        self.high_shelf_cutoff_hz = high_shelf_cutoff_hz
        self.high_shelf_q = high_shelf_q

        self.eps = float(eps)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {
            "low_shelf_gain_db": sample_from_dist(self.low_shelf_gain_db, state),
            "low_shelf_cutoff_hz": sample_from_dist(self.low_shelf_cutoff_hz, state),
            "low_shelf_q": sample_from_dist(self.low_shelf_q, state),
            "band0_gain_db": sample_from_dist(self.band0_gain_db, state),
            "band0_cutoff_hz": sample_from_dist(self.band0_cutoff_hz, state),
            "band0_q": sample_from_dist(self.band0_q, state),
            "band1_gain_db": sample_from_dist(self.band1_gain_db, state),
            "band1_cutoff_hz": sample_from_dist(self.band1_cutoff_hz, state),
            "band1_q": sample_from_dist(self.band1_q, state),
            "band2_gain_db": sample_from_dist(self.band2_gain_db, state),
            "band2_cutoff_hz": sample_from_dist(self.band2_cutoff_hz, state),
            "band2_q": sample_from_dist(self.band2_q, state),
            "band3_gain_db": sample_from_dist(self.band3_gain_db, state),
            "band3_cutoff_hz": sample_from_dist(self.band3_cutoff_hz, state),
            "band3_q": sample_from_dist(self.band3_q, state),
            "high_shelf_gain_db": sample_from_dist(self.high_shelf_gain_db, state),
            "high_shelf_cutoff_hz": sample_from_dist(self.high_shelf_cutoff_hz, state),
            "high_shelf_q": sample_from_dist(self.high_shelf_q, state),
        }

    def _transform(
        self,
        signal,
        low_shelf_gain_db,
        low_shelf_cutoff_hz,
        low_shelf_q,
        band0_gain_db,
        band0_cutoff_hz,
        band0_q,
        band1_gain_db,
        band1_cutoff_hz,
        band1_q,
        band2_gain_db,
        band2_cutoff_hz,
        band2_q,
        band3_gain_db,
        band3_cutoff_hz,
        band3_q,
        high_shelf_gain_db,
        high_shelf_cutoff_hz,
        high_shelf_q,
    ):
        x = signal.audio_data
        n_batch = signal.batch_size
        sr = float(signal.sample_rate)

        nyq = 0.5 * sr
        hz_min = 1.0
        hz_max = float(nyq) * 0.999

        def _hz(v):
            return (
                ensure_tensor(v, 2, n_batch)
                .to(x.device)
                .view(-1)
                .float()
                .clamp(min=hz_min, max=hz_max)
            )

        def _q(v):
            return (
                ensure_tensor(v, 2, n_batch)
                .to(x.device)
                .view(-1)
                .float()
                .clamp(min=self.eps)
            )

        def _db(v):
            return ensure_tensor(v, 2, n_batch).to(x.device).view(-1).float()

        low_shelf_gain_db = _db(low_shelf_gain_db)
        low_shelf_cutoff_hz = _hz(low_shelf_cutoff_hz)
        low_shelf_q = _q(low_shelf_q)

        band0_gain_db = _db(band0_gain_db)
        band0_cutoff_hz = _hz(band0_cutoff_hz)
        band0_q = _q(band0_q)

        band1_gain_db = _db(band1_gain_db)
        band1_cutoff_hz = _hz(band1_cutoff_hz)
        band1_q = _q(band1_q)

        band2_gain_db = _db(band2_gain_db)
        band2_cutoff_hz = _hz(band2_cutoff_hz)
        band2_q = _q(band2_q)

        band3_gain_db = _db(band3_gain_db)
        band3_cutoff_hz = _hz(band3_cutoff_hz)
        band3_q = _q(band3_q)

        high_shelf_gain_db = _db(high_shelf_gain_db)
        high_shelf_cutoff_hz = _hz(high_shelf_cutoff_hz)
        high_shelf_q = _q(high_shelf_q)

        # Six second-order sections
        sos = torch.zeros(n_batch, 6, 6, device=x.device, dtype=x.dtype)

        # Low shelf
        b, a = _biquad(
            low_shelf_gain_db,
            low_shelf_cutoff_hz,
            low_shelf_q,
            sr,
            "low_shelf",
        )
        sos[:, 0, :] = torch.cat((b, a), dim=-1)

        # Band 0
        b, a = _biquad(
            band0_gain_db,
            band0_cutoff_hz,
            band0_q,
            sr,
            "peaking",
        )
        sos[:, 1, :] = torch.cat((b, a), dim=-1)

        # Band 1
        b, a = _biquad(
            band1_gain_db,
            band1_cutoff_hz,
            band1_q,
            sr,
            "peaking",
        )
        sos[:, 2, :] = torch.cat((b, a), dim=-1)

        # Band 2
        b, a = _biquad(
            band2_gain_db,
            band2_cutoff_hz,
            band2_q,
            sr,
            "peaking",
        )
        sos[:, 3, :] = torch.cat((b, a), dim=-1)

        # Band 3
        b, a = _biquad(
            band3_gain_db,
            band3_cutoff_hz,
            band3_q,
            sr,
            "peaking",
        )
        sos[:, 4, :] = torch.cat((b, a), dim=-1)

        # High shelf
        b, a = _biquad(
            high_shelf_gain_db,
            high_shelf_cutoff_hz,
            high_shelf_q,
            sr,
            "high_shelf",
        )
        sos[:, 5, :] = torch.cat((b, a), dim=-1)

        signal.audio_data = _sosfilt_via_fsm(sos, x)
        signal.stft_data = None
        return signal