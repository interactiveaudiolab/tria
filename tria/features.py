import torch
from audiotools import AudioSignal

from .dsp import resample

################################################################################
# Utilities for extracting rhythm feature representations
################################################################################


def _moving_average(x: torch.Tensor, window_length: int):
    """
    Smooth features with moving average over frames.

    Parameters
    ----------
    x : torch.Tensor
        Shape (n_batch, n_feats, n_frames)
    window_length : int
        Smoothing window length
    """
    if window_length <= 1:
        return x
    n_feats = x.shape[1]
    kernel = torch.ones(
        (n_feats, 1, window_length), 
        device=x.device, dtype=x.dtype
    ) / window_length

    pad_left = (window_length - 1) // 2
    pad_right = window_length // 2
    x_pad = torch.nn.functional.pad(x, (pad_left, pad_right), mode="reflect")

    # Smooth separately over feature channels
    return torch.nn.functional.conv1d(x_pad, kernel, groups=n_feats)


# The 'original' TRIA features can be recovered using:
#  * `slow_ma_ms` = None
#  * `post_smooth_ms` = None
#  * `legacy_normalize` = True
def rhythm_features(
    signal: AudioSignal,
    sample_rate: int = 44_100,
    n_bands: int = 2,
    n_mels: int = 80,
    window_length: int = 1024,
    hop_length: int = 512,
    normalize_quantile: float = 0.98,
    quantization_levels: int = 33,
    clamp_max: float = 50.0,
    eps: float = 1e-8,
    slow_ma_ms: float = 100.0,
    post_smooth_ms: float = 10.0,
    legacy_normalize: bool = False,
):
    """
    Extract multi-band 'rhythm' features from audio by adaptively splitting 
    spectrogram along frequency axis and applying normalization, quantization,
    and smoothing / sparsity filtering.
    
    Parameters
    ----------
    signal : AudioSignal
        Audio from which to extract features
    sample_rate : int
        Sample rate at which to extract features
    n_bands : int
        Number of frequency bands into which to adaptively divide spectrogram
    n_mels : int
        Number of base mel frequency bins in spectrogram
    window_length : int
        Spectrogram window length
    hop_length : int
        Spectrogram hop length
    normalize_quantile : float
        Optionally normalize each band relative to top-p largest magnitude 
        rather than absolute max
    quantization_levels : int
        Number of bins into which feature magnitudes are quantized
    clamp_max : float
        Maximum allowed spectrogram magnitude
    eps : float
        For numerical stability
    slow_ma_ms : float
        Smoothing filter length in milliseconds for transient emphasis (smoothed
        features are subtracted)
    post_smooth_ms : float
        Smoothing filter length in milliseconds for transient smoothing
    legacy_normalize : bool
        If `True`, use mean/std and sigmoid normalization as described in 
        original TRIA paper
    """

    assert n_bands >= 1
    assert quantization_levels >= 2

    # Loudness normalization
    signal = resample(signal.clone().to_mono(), sample_rate).normalize(-16.)
    signal.ensure_max_of_audio()

    # Clamped mel spectrogram
    mel = signal.mel_spectrogram(
        n_mels=n_mels,
        hop_length=hop_length,
        window_length=window_length,
    ).mean(1)  # (n_batch, n_mels, n_frames)
    mel = torch.clamp(mel, 0.0, clamp_max)

    n_batch, _, n_frames = mel.shape

    if legacy_normalize:
        # Original normalization: divide by number of mels
        mel = mel / n_mels
    else:
        # Compress logarithmically
        mel = torch.log1p(mel) / torch.log1p(torch.tensor(clamp_max, device=mel.device, dtype=mel.dtype))

    # Split spectrogram into bands adaptively
    energy_per_bin = mel.mean(dim=-1)          # (n_batch, n_mels)
    cum = energy_per_bin.cumsum(dim=1)         # (n_batch, n_mels)
    total = cum[:, -1:]                        # (n_batch, 1)

    if n_bands == 1:
        bands = mel.sum(dim=1, keepdim=True)                   # (n_batch, 1, n_frames)
    else:
        targets = torch.linspace(
            1.0 / n_bands, (n_bands - 1) / n_bands, n_bands - 1,
            device=mel.device, dtype=mel.dtype
        )[None, :] * total                                     # (n_batch, n_bands-1)

        edges = torch.searchsorted(cum, targets, right=False)  # (n_batch, n_bands-1)

        cuts = torch.cat(
            [
                torch.zeros(n_batch, 1, dtype=torch.long, device=mel.device),
                edges + 1,
                torch.full((n_batch, 1), mel.size(1), dtype=torch.long, device=mel.device),
            ],
            dim=1
        )  # (n_batch, n_bands+1)

        prefix = mel.cumsum(dim=1)  # (n_batch, n_mels, n_frames)
        prefix_pad = torch.cat(
            [torch.zeros(n_batch, 1, n_frames, device=mel.device, dtype=mel.dtype), prefix],
            dim=1
        )

        a_idx = cuts[:, :-1].unsqueeze(-1).expand(n_batch, n_bands, n_frames)
        b_idx = cuts[:, 1: ].unsqueeze(-1).expand(n_batch, n_bands, n_frames)
        bands = prefix_pad.gather(1, b_idx) - prefix_pad.gather(1, a_idx)  # (n_batch, n_bands, n_frames)

    # Emphasize transients by subtracting smoothed features
    transient = bands.clone()
    to_frames = lambda ms: max(1, int(round((ms / 1000.0) * sample_rate / hop_length)))

    if slow_ma_ms is not None:
        slow_win = to_frames(slow_ma_ms)
        bands_slow = _moving_average(bands, slow_win)  # (n_batch, n_bands, n_frames)
        transient = torch.relu(bands - bands_slow)

    # Apply additional smoothing to transients
    if post_smooth_ms is not None:
        ps_win = to_frames(post_smooth_ms)
        if ps_win > 1:
            transient = _moving_average(transient, ps_win)

    # Normalize features across time per band
    if legacy_normalize:
        # Original normalization (mean/std with sigmoid compression)
        mean = transient.mean(dim=-1, keepdim=True)
        std = transient.std(dim=-1, keepdim=True).clamp_min(eps)
        transient = torch.sigmoid((transient - mean) / std)
    
    else:
        # Quantile-based normalization
        q = torch.quantile(
            transient.clamp_min(0.0),
            q=normalize_quantile,
            dim=-1,
            keepdim=True
        ).clamp_min(eps)
        transient = (transient / q).clamp(0.0, 1.0)

    # Quantize feature intensities into bins to ensure a tight information
    # bottleneck
    steps = quantization_levels - 1
    return torch.round(transient * steps) / steps
