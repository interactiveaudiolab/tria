import math
import os
from functools import partial
from pathlib import Path
from typing import Callable

import gradio as gr
import librosa
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util

from tria.constants import ASSETS_DIR
from tria.constants import PRETRAINED_DIR
from tria.dsp import resample
from tria.features import rhythm_features
from tria.model.tria import TRIA
from tria.pipelines.tokenizer import Tokenizer
from tria.pipelines.tokenizer import TokenSequence

################################################################################
# Run inference with trained TRIA models
################################################################################


# Spectrogram plots
SPEC_MELS = 128
SMALL_WIDTH = 900  # Timbre/rhythm prompts
LARGE_WIDTH = 600  # Generated outputs

# Device
DEVICE = None

# Batched outputs
N_OUTPUTS = 3

# Loaded configuration
LOADED = dict(
    name=None,
    model=None,
    tokenizer=None,
    feature_fn=None,
    sample_rate=None,
    max_duration=None,
)

# Pretrained models
MODEL_ZOO = {
    "small_musdb_moises_fsl_2b": {
        "checkpoint": "pretrained/tria/small_musdb_moises_fsl_2b/latest/model.pt",
        "model_cfg": {
            "codebook_size": 1024,
            "n_codebooks": 9,
            "n_channels": 512,
            "n_feats": 2,
            "n_heads": 8,
            "n_layers": 12,
            "mult": 4,
            "p_dropout": 0.0,
            "bias": True,
            "max_len": 1000,
            "pos_enc": "rope",
            "qk_norm": True,
            "use_sdpa": True,
            "interp": "nearest",
            "share_emb": True,
        },
        "tokenizer_cfg": {"name": "dac"},
        "feature_cfg": {
            "sample_rate": 16_000,
            "n_bands": 2,
            "n_mels": 40,
            "window_length": 384,
            "hop_length": 192,
            "quantization_levels": 5,
            "slow_ma_ms": 200,
            "post_smooth_ms": 100,
            "legacy_normalize": False,
            "clamp_max": 50.0,
            "normalize_quantile": 0.98,
        },
        "max_duration": 6.0,
    },
    "small_musdb_moises_2b": {
        "checkpoint": "pretrained/tria/small_musdb_moises_2b/latest/model.pt",
        "model_cfg": {
            "codebook_size": 1024,
            "n_codebooks": 9,
            "n_channels": 512,
            "n_feats": 2,
            "n_heads": 8,
            "n_layers": 12,
            "mult": 4,
            "p_dropout": 0.0,
            "bias": True,
            "max_len": 1000,
            "pos_enc": "rope",
            "qk_norm": True,
            "use_sdpa": True,
            "interp": "nearest",
            "share_emb": True,
        },
        "tokenizer_cfg": {"name": "dac"},
        "feature_cfg": {
            "sample_rate": 16_000,
            "n_bands": 2,
            "n_mels": 40,
            "window_length": 384,
            "hop_length": 192,
            "quantization_levels": 5,
            "slow_ma_ms": 200,
            "post_smooth_ms": 100,
            "legacy_normalize": False,
            "clamp_max": 50.0,
            "normalize_quantile": 0.98,
        },
        "max_duration": 6.0,
    },
    "tiny_musdb_moises_fsl_2b": {
        "checkpoint": "pretrained/tria/distilled_tiny_musdb_moises_fsl_2b/latest/model.pt",
        "model_cfg": {
            "codebook_size": 1024,
            "n_codebooks": 9,
            "n_channels": 384,
            "n_feats": 2,
            "n_heads": 6,
            "n_layers": 6,
            "mult": 4,
            "p_dropout": 0.0,
            "bias": True,
            "max_len": 1000,
            "pos_enc": "rope",
            "qk_norm": True,
            "use_sdpa": True,
            "interp": "nearest",
            "share_emb": True,
        },
        "tokenizer_cfg": {"name": "dac"},
        "feature_cfg": {
            "sample_rate": 16_000,
            "n_bands": 2,
            "n_mels": 40,
            "window_length": 384,
            "hop_length": 192,
            "quantization_levels": 5,
            "slow_ma_ms": 200,
            "post_smooth_ms": 100,
            "legacy_normalize": False,
            "clamp_max": 50.0,
            "normalize_quantile": 0.98,
        },
        "max_duration": 6.0,
    },
}


# Example audio
TIMBRE_EXAMPLE = ASSETS_DIR / "drums" / "drums_1.wav"
RHYTHM_EXAMPLE = ASSETS_DIR / "beatbox" / "beatbox_1.wav"

########################################
# Audio utilities
########################################


def db_to_linear(db: float):
    return 10.0 ** (db / 20.0)


def bandpass(signal: AudioSignal):
    signal = signal.clone().high_pass(250)
    signal = signal.low_pass(8000)
    signal.ensure_max_of_audio()

    return signal


def to_gradio_audio(signal: AudioSignal):
    audio = signal.clone().audio_data[0]
    if audio.shape[0] == 1:
        audio = audio.flatten()
    else:
        audio = audio.transpose(0, 1)
    return (
        signal.sample_rate,
        audio.numpy().astype(np.float32),
    )


def from_gradio_audio(sample_rate: int, x: np.ndarray):
    x = np.asarray(x)
    if x.ndim == 1:
        x = torch.from_numpy(x[None, None, :].astype(np.float32))
    elif x.ndim == 2:
        x = torch.from_numpy(x.T[None, :, :].astype(np.float32))

    return AudioSignal(x, sample_rate=int(sample_rate))


########################################
# Spectrogram plotting
########################################


def _strip_silent_borders_uint8(img2d: np.ndarray, thr: int = 2):
    """
    Trim spectrogram columns.
    """
    assert img2d.ndim == 2 and img2d.dtype == np.uint8
    colmax = img2d.max(axis=0)
    left = 0
    while left < colmax.size and colmax[left] <= thr:
        left += 1
    right = colmax.size
    while right > left and colmax[right - 1] <= thr:
        right -= 1
    if right - left < min(8, img2d.shape[1] // 10):
        return img2d
    return img2d[:, left:right]


def _resize_width_linear_uint8(img2d: np.ndarray, target_w: int):
    """
    Resize spectrogram along time dimension.
    """
    H, W = img2d.shape
    if W == target_w:
        return img2d
    x_old = np.linspace(0.0, 1.0, W, endpoint=True)
    x_new = np.linspace(0.0, 1.0, target_w, endpoint=True)
    idx = np.searchsorted(x_old, x_new, side="left")
    idx = np.clip(idx, 1, W - 1)
    x0 = x_old[idx - 1][None, :]
    x1 = x_old[idx][None, :]
    y0 = img2d[:, idx - 1].astype(np.float32)
    y1 = img2d[:, idx].astype(np.float32)
    w = (x_new[None, :] - x0) / (x1 - x0 + 1e-12)
    out = y0 * (1.0 - w) + y1 * w
    return out.clip(0, 255).astype(np.uint8)


def _to_rgb_uint8(img2d: np.ndarray):
    return np.repeat(img2d[:, :, None], 3, axis=2)


def spectrogram_fast(signal: AudioSignal, small: bool = False):
    """
    Fast log-mel spectrogram.
    """
    target_w = SMALL_WIDTH if small else LARGE_WIDTH

    signal = signal.clone().to_mono()
    y = signal.audio_data.flatten().numpy()

    if y.size == 0:
        return np.zeros((SPEC_MELS, target_w, 3), dtype=np.uint8)

    n_fft = 512 if small else 1024
    hop = 128 if small else 256
    n_mels = SPEC_MELS

    S = librosa.feature.melspectrogram(
        y=y,
        sr=signal.sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = np.clip(S_db, -80.0, 0.0)
    img = (255.0 * (S_db + 80.0) / 80.0).astype(np.uint8)  # (n_mels, n_frames)
    img = np.flipud(img)

    img = _strip_silent_borders_uint8(img, thr=2)
    img = _resize_width_linear_uint8(img, target_w=target_w)
    return _to_rgb_uint8(img)


########################################
# Inference utilities
########################################


def _pick_device():
    global DEVICE
    if DEVICE is None:
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cpu")
    return DEVICE


def load_model_by_name(name: str):
    global LOADED
    device = _pick_device()
    cfg = MODEL_ZOO[name]

    model = TRIA(**cfg["model_cfg"])
    sd = torch.load(cfg["checkpoint"], map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer(**cfg["tokenizer_cfg"])
    tokenizer = tokenizer.to(device)

    feat_fn = partial(rhythm_features, **cfg.get("feature_cfg", {}))

    LOADED.update(
        dict(
            name=name,
            model=model,
            tokenizer=tokenizer,
            feature_fn=feat_fn,
            sample_rate=tokenizer.sample_rate,
            max_duration=cfg["max_duration"],
        )
    )
    return f"Loaded model '{name}' on {device}"


def _to_channels(signal: AudioSignal, n: int):
    if n == 1:
        return signal.to_mono()
    else:
        if signal.num_channels == n:
            return signal
        else:
            assert signal.num_channels == 1
            signal.audio_data = signal.audio_data(1, n, 1)
        return signal


def _prepare_inputs(
    timbre_prompt: AudioSignal,
    rhythm_prompt: AudioSignal,
    prefix_dur: float,
    buffer_dur: float,
    sample_rate: int,
    feat_fn: Callable,
    tokenizer: Tokenizer,
    interp: str,
    device: str,
):
    assert timbre_prompt.batch_size == rhythm_prompt.batch_size
    n_channels = tokenizer.n_channels

    # Prepare audio
    timbre_prompt = resample(timbre_prompt.clone(), sample_rate)
    rhythm_prompt = resample(rhythm_prompt.clone(), sample_rate)

    timbre_prompt = _to_channels(timbre_prompt, n_channels)
    rhythm_prompt = _to_channels(rhythm_prompt, n_channels)

    timbre_prompt = timbre_prompt.truncate_samples(int(prefix_dur * sample_rate))
    rhythm_prompt = rhythm_prompt.truncate_samples(
        int(buffer_dur * sample_rate) - timbre_prompt.signal_length
    )

    # Tokenize
    timbre_tokens = tokenizer.encode(timbre_prompt)
    rhythm_tokens = tokenizer.encode(rhythm_prompt)

    tokens = torch.cat([timbre_tokens.tokens, rhythm_tokens.tokens], dim=-1)
    n_batch, n_codebooks, n_frames = tokens.shape
    prefix_frames = timbre_tokens.tokens.shape[-1]

    # Extract features
    _feats = feat_fn(rhythm_prompt)
    _feats = torch.nn.functional.interpolate(
        _feats, n_frames - prefix_frames, mode=interp
    )
    feats = torch.zeros(n_batch, _feats.shape[1], n_frames, device=device)
    feats[..., prefix_frames:] = _feats

    # Construct masks
    prefix_mask = (
        torch.arange(n_frames, device=device)[None, :].repeat(n_batch, 1)
        < prefix_frames
    )  # (n_batch, n_frames)
    tokens_mask = prefix_mask[:, None, :].repeat(
        1, n_codebooks, 1
    )  # (n_batch, n_codebooks, n_frames)
    feats_mask = ~prefix_mask  # (n_batch, n_frames)

    return (
        tokens,
        feats,
        tokens_mask,
        feats_mask,
        timbre_tokens,
        rhythm_tokens,
        prefix_frames,
    )


def _prepare_loopable_inputs(
    timbre_prompt: AudioSignal,
    rhythm_prompt: AudioSignal,
    prefix_dur: float,
    buffer_dur: float,
    sample_rate: int,
    feat_fn: Callable,
    tokenizer: Tokenizer,
    interp: str,
    device: str,
    loop_pad_frames: int,
):
    timbre_prompt = resample(timbre_prompt.clone(), sample_rate)
    rhythm_prompt = resample(rhythm_prompt.clone(), sample_rate)

    timbre_prompt = _to_channels(timbre_prompt, tokenizer.n_channels)
    rhythm_prompt = _to_channels(rhythm_prompt, tokenizer.n_channels)

    timbre_prompt = timbre_prompt.truncate_samples(int(prefix_dur * sample_rate))
    max_rhythm_samples = int(buffer_dur * sample_rate) - timbre_prompt.signal_length
    max_rhythm_samples -= 2 * int(loop_pad_frames) * tokenizer.hop_length
    rhythm_prompt = rhythm_prompt.truncate_samples(max(max_rhythm_samples, 0))

    timbre_tokens = tokenizer.encode(timbre_prompt)
    rhythm_tokens = tokenizer.encode(rhythm_prompt)
    n_batch, n_codebooks, n_rhythm_frames = rhythm_tokens.tokens.shape
    prefix_frames = timbre_tokens.tokens.shape[-1]
    loop_pad_frames = min(int(loop_pad_frames), n_rhythm_frames)

    left = rhythm_tokens.tokens[..., -loop_pad_frames:]
    right = rhythm_tokens.tokens[..., :loop_pad_frames]
    tokens = torch.cat(
        [timbre_tokens.tokens, left, rhythm_tokens.tokens, right], dim=-1
    )
    n_frames = tokens.shape[-1]

    _feats = feat_fn(rhythm_prompt)
    _feats = torch.nn.functional.interpolate(_feats, n_rhythm_frames, mode=interp)
    feats = torch.zeros(n_batch, _feats.shape[1], n_frames, device=device)
    feats[..., prefix_frames : prefix_frames + loop_pad_frames] = _feats[
        ...,
        -loop_pad_frames:,
    ]
    feats[
        ...,
        prefix_frames
        + loop_pad_frames : prefix_frames
        + loop_pad_frames
        + n_rhythm_frames,
    ] = _feats
    feats[..., -loop_pad_frames:] = _feats[..., :loop_pad_frames]

    tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
    tokens_mask[..., :prefix_frames] = True
    feats_mask = torch.zeros(n_batch, n_frames, dtype=torch.bool, device=device)
    feats_mask[:, prefix_frames:] = True

    return (
        tokens,
        feats,
        tokens_mask,
        feats_mask,
        timbre_tokens,
        rhythm_tokens,
        prefix_frames,
        loop_pad_frames,
    )


def _timbre_vocab_mask(tokens, prefix_frames, n_codebooks, codebook_size, n_restrict):
    n_restrict = min(int(n_restrict), n_codebooks)
    if not n_restrict:
        return None

    vocab_mask = torch.ones(
        n_codebooks,
        codebook_size,
        dtype=torch.bool,
        device=tokens.device,
    )
    vocab_mask[:n_restrict] = False
    prefix_tokens = tokens[:, :n_restrict, :prefix_frames]
    for codebook_idx in range(n_restrict):
        vocab_mask[codebook_idx].scatter_(
            0,
            prefix_tokens[:, codebook_idx, :].reshape(-1),
            True,
        )
    return vocab_mask


def _crossfade(a: torch.Tensor, b: torch.Tensor, n: int):
    n = min(int(n), a.shape[-1], b.shape[-1])
    if n <= 0:
        return torch.cat([a, b], dim=-1)

    fade = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype).view(1, 1, -1)
    x = a[..., -n:] * (1.0 - fade) + b[..., :n] * fade
    return torch.cat([a[..., :-n], x, b[..., n:]], dim=-1)


@torch.no_grad()
def _inference_windowed(
    timbre_prompt: AudioSignal,
    rhythm_prompt: AudioSignal,
    sampling_ctrls,
    inference_ctrls,
    audio_ctrls,
    schedule_ctrls,
    seed,
    prefix_dur: float,
    device: str,
):
    feat_fn = LOADED["feature_fn"]
    tokenizer = LOADED["tokenizer"]
    buffer_dur = LOADED["max_duration"]
    sample_rate = LOADED["sample_rate"]
    model = LOADED["model"]
    interp = model.interp

    timbre_prompt = resample(timbre_prompt.clone(), sample_rate)
    rhythm_prompt = resample(rhythm_prompt.clone(), sample_rate)

    timbre_prompt = _to_channels(timbre_prompt, tokenizer.n_channels)
    rhythm_prompt = _to_channels(rhythm_prompt, tokenizer.n_channels)

    timbre_prompt = timbre_prompt.truncate_samples(int(prefix_dur * sample_rate))
    timbre_tokens = tokenizer.encode(timbre_prompt)
    rhythm_tokens = tokenizer.encode(rhythm_prompt)

    n_batch, n_codebooks, n_rhythm_frames = rhythm_tokens.tokens.shape
    prefix_frames = timbre_tokens.tokens.shape[-1]

    max_rhythm_samples = int(buffer_dur * sample_rate) - timbre_prompt.signal_length
    max_rhythm_samples = max(max_rhythm_samples, tokenizer.hop_length)
    probe = rhythm_prompt.clone().truncate_samples(max_rhythm_samples)
    max_rhythm_frames = tokenizer.encode(probe).tokens.shape[-1]
    overlap_frames = min(
        int(inference_ctrls["window_overlap_frames"]),
        max(0, max_rhythm_frames - 1),
    )

    feats_full = feat_fn(rhythm_prompt)
    feats_full = torch.nn.functional.interpolate(
        feats_full,
        n_rhythm_frames,
        mode=interp,
    )

    stereo = inference_ctrls["pseudo_stereo_codebook"] is not None
    loop_pad_frames = (
        int(inference_ctrls["loop_pad_frames"]) if inference_ctrls["loopable"] else 0
    )
    first_head = None
    context = None
    out_audio = None
    pos = 0
    samples_written = 0
    n_windows = 0

    while pos < n_rhythm_frames:
        ctx_frames = 0 if context is None else min(overlap_frames, context.shape[-1])
        ctx_tokens = None if ctx_frames == 0 else context[..., -ctx_frames:]
        suffix_tokens = None
        suffix_frames = 0
        new_capacity = max_rhythm_frames - ctx_frames
        if loop_pad_frames and first_head is not None:
            suffix_frames = min(loop_pad_frames, first_head.shape[-1])
            if n_rhythm_frames - pos <= new_capacity - suffix_frames:
                suffix_tokens = first_head[..., :suffix_frames]
                new_capacity -= suffix_frames
            else:
                suffix_frames = 0
        new_frames = min(new_capacity, n_rhythm_frames - pos)
        if new_frames <= 0:
            break

        total_frames = prefix_frames + ctx_frames + new_capacity + suffix_frames
        valid_frames = prefix_frames + ctx_frames + new_frames + suffix_frames

        tokens = torch.zeros(
            n_batch,
            n_codebooks,
            total_frames,
            dtype=rhythm_tokens.tokens.dtype,
            device=device,
        )
        tokens[..., :prefix_frames] = timbre_tokens.tokens
        if ctx_tokens is not None:
            tokens[..., prefix_frames : prefix_frames + ctx_frames] = ctx_tokens
        tokens[
            ...,
            prefix_frames + ctx_frames : prefix_frames + ctx_frames + new_frames,
        ] = rhythm_tokens.tokens[..., pos : pos + new_frames]
        if suffix_tokens is not None:
            tokens[
                ...,
                prefix_frames
                + ctx_frames
                + new_frames : prefix_frames
                + ctx_frames
                + new_frames
                + suffix_frames,
            ] = suffix_tokens

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[..., : prefix_frames + ctx_frames] = True
        if suffix_tokens is not None:
            tokens_mask[
                ...,
                prefix_frames
                + ctx_frames
                + new_frames : prefix_frames
                + ctx_frames
                + new_frames
                + suffix_frames,
            ] = True
        tokens_mask[..., valid_frames:] = True

        feats = torch.zeros(n_batch, feats_full.shape[1], total_frames, device=device)
        feats[
            ...,
            prefix_frames + ctx_frames : prefix_frames + ctx_frames + new_frames,
        ] = feats_full[..., pos : pos + new_frames]

        feats_mask = torch.zeros(n_batch, total_frames, dtype=torch.bool, device=device)
        feats_mask[:, prefix_frames + ctx_frames : valid_frames] = True

        vocab_mask = _timbre_vocab_mask(
            tokens,
            prefix_frames,
            model.n_codebooks,
            model.codebook_size,
            inference_ctrls["timbre_vocab_codebooks"],
        )

        window_seed = [s + 10000019 * n_windows for s in seed]
        generated = model.inference(
            tokens,
            feats,
            tokens_mask,
            feats_mask,
            lengths=torch.full(
                (n_batch,), valid_frames, dtype=torch.long, device=device
            ),
            top_p=sampling_ctrls["top_p"],
            top_k=sampling_ctrls["top_k"],
            temp=sampling_ctrls["temperature"],
            mask_temp=sampling_ctrls["mask_temperature"],
            iterations=[int(v) for v in schedule_ctrls],
            guidance_scale=inference_ctrls["cfg_scale"],
            causal_bias=inference_ctrls["causal_bias"],
            seed=window_seed,
            pseudo_stereo_codebook=inference_ctrls["pseudo_stereo_codebook"],
            vocab_mask=vocab_mask,
        )

        main_generated = generated[::2] if stereo else generated
        new_context = main_generated[
            ...,
            prefix_frames + ctx_frames : prefix_frames + ctx_frames + new_frames,
        ]
        context = (
            new_context
            if context is None
            else torch.cat([context, new_context], dim=-1)
        )
        if first_head is None and loop_pad_frames:
            first_head = context[..., :loop_pad_frames]

        decode_tokens = generated[
            ...,
            prefix_frames : prefix_frames + ctx_frames + new_frames + suffix_frames,
        ]
        remaining_samples = rhythm_prompt.signal_length - samples_written
        new_samples = min(new_frames * tokenizer.hop_length, remaining_samples)
        overlap_samples = ctx_frames * tokenizer.hop_length
        suffix_samples = suffix_frames * tokenizer.hop_length
        token_seq = TokenSequence(
            tokens=decode_tokens,
            extras=dict(rhythm_tokens.extras),
        )
        token_seq.extras["signal_length"] = (
            overlap_samples + new_samples + suffix_samples
        )
        if stereo:
            token_seq.extras["n_channels"] = 2

        window_audio = tokenizer.decode(token_seq).audio_data
        out_audio = (
            window_audio
            if out_audio is None
            else _crossfade(out_audio, window_audio, overlap_samples)
        )

        pos += new_frames
        samples_written += new_samples
        n_windows += 1

    out = AudioSignal(out_audio, sample_rate=sample_rate)
    out.truncate_samples(rhythm_prompt.signal_length)
    out.zero_pad_to(rhythm_prompt.signal_length)

    if stereo:
        x = out.audio_data
        mid = x.mean(dim=1, keepdim=True)
        side = x - mid
        x = mid + inference_ctrls["pseudo_stereo_width"] * side
        rms = torch.sqrt((x * x).mean(dim=(1, 2), keepdim=True) + 1e-12)
        out.audio_data = x * (db_to_linear(audio_ctrls["loudness_db"]) / rms)
    else:
        out.normalize(audio_ctrls["loudness_db"])
    out.ensure_max_of_audio()

    msg = (
        f"Model='{LOADED['name']}' on {device} | sample_rate={sample_rate}, "
        f"prefix_dur={min(prefix_dur, timbre_prompt.duration):0.3f} | "
        f"windows={n_windows}, overlap_frames={overlap_frames}; "
        f"top_p={sampling_ctrls['top_p']}, "
        f"top_k={sampling_ctrls['top_k']}, "
        f"temp={sampling_ctrls['temperature']}, "
        f"mask_temp={sampling_ctrls['mask_temperature']}; "
        f"seed={int(seed[0])}, "
        f"causal_bias={inference_ctrls['causal_bias']}, "
        f"cfg={inference_ctrls['cfg_scale']}, "
        f"timbre_vocab_codebooks={inference_ctrls['timbre_vocab_codebooks']}, "
        f"loopable={inference_ctrls['loopable']}, "
        f"loop_pad_frames={inference_ctrls['loop_pad_frames']}, "
        f"pseudo_stereo_codebook={inference_ctrls['pseudo_stereo_codebook']}, "
        f"pseudo_stereo_width={inference_ctrls['pseudo_stereo_width']}"
    )
    return out, msg


@torch.no_grad()
def _inference(
    timbre_prompt: AudioSignal,
    rhythm_prompt: AudioSignal,
    sampling_ctrls,
    inference_ctrls,
    audio_ctrls,
    schedule_ctrls,
    seed,
):
    if (
        LOADED["model"] is None
        or LOADED["tokenizer"] is None
        or LOADED["feature_fn"] is None
    ):
        raise RuntimeError(f"No model loaded")

    device = _pick_device()

    feat_fn = LOADED["feature_fn"]
    tokenizer = LOADED["tokenizer"]
    buffer_dur = LOADED["max_duration"]
    sample_rate = LOADED["sample_rate"]
    tokenizer = LOADED["tokenizer"]
    model = LOADED["model"]
    interp = model.interp

    # Prefix mode: default to max length of 1/3 buffer
    prefix_dur = int(buffer_dur / 3)

    # Optionally filter rhythm audio
    if audio_ctrls["filter_inputs"]:
        rhythm_prompt = bandpass(rhythm_prompt)

    timbre_prompt = timbre_prompt.clone().to(device)
    rhythm_prompt = rhythm_prompt.clone().to(device)

    timbre_prompt.ensure_max_of_audio()
    rhythm_prompt.ensure_max_of_audio()

    # Random seed
    util.seed(seed[0])

    # Cleanup
    if sampling_ctrls["top_p"] in [0.0, 1.0]:
        sampling_ctrls["top_p"] = None
    if sampling_ctrls["top_k"] in [0]:
        sampling_ctrls["top_k"] = None
    if inference_ctrls["cfg_scale"] in [0.0]:
        inference_ctrls["cfg_scale"] = None

    max_rhythm_dur = buffer_dur - min(prefix_dur, timbre_prompt.duration)
    loop_pad_frames = (
        int(inference_ctrls["loop_pad_frames"]) if inference_ctrls["loopable"] else 0
    )
    max_rhythm_dur -= 2 * loop_pad_frames * tokenizer.hop_length / sample_rate
    if rhythm_prompt.duration > max_rhythm_dur:
        return _inference_windowed(
            timbre_prompt,
            rhythm_prompt,
            sampling_ctrls,
            inference_ctrls,
            audio_ctrls,
            schedule_ctrls,
            seed,
            prefix_dur,
            device,
        )

    if loop_pad_frames:
        (
            buffer,
            feats,
            buffer_mask,
            feats_mask,
            timbre_tokens,
            rhythm_tokens,
            prefix_frames,
            loop_pad_frames,
        ) = _prepare_loopable_inputs(
            timbre_prompt,
            rhythm_prompt,
            prefix_dur,
            buffer_dur,
            sample_rate,
            feat_fn,
            tokenizer,
            interp,
            device,
            loop_pad_frames,
        )
    else:
        (
            buffer,
            feats,
            buffer_mask,
            feats_mask,
            timbre_tokens,
            rhythm_tokens,
            prefix_frames,
        ) = _prepare_inputs(
            timbre_prompt,
            rhythm_prompt,
            prefix_dur,
            buffer_dur,
            sample_rate,
            feat_fn,
            tokenizer,
            interp,
            device,
        )

    vocab_mask = _timbre_vocab_mask(
        buffer,
        prefix_frames,
        model.n_codebooks,
        model.codebook_size,
        inference_ctrls["timbre_vocab_codebooks"],
    )

    # Inference
    generated = model.inference(
        buffer,
        feats,
        buffer_mask.clone(),
        feats_mask,
        top_p=sampling_ctrls["top_p"],
        top_k=sampling_ctrls["top_k"],
        temp=sampling_ctrls["temperature"],
        mask_temp=sampling_ctrls["mask_temperature"],
        iterations=[int(v) for v in schedule_ctrls],
        guidance_scale=inference_ctrls["cfg_scale"],
        causal_bias=inference_ctrls["causal_bias"],
        seed=seed,
        pseudo_stereo_codebook=inference_ctrls["pseudo_stereo_codebook"],
        vocab_mask=vocab_mask,
        loop_pad_frames=loop_pad_frames,
    )
    if loop_pad_frames:
        generated = generated[
            ...,
            prefix_frames : prefix_frames
            + loop_pad_frames
            + rhythm_tokens.tokens.shape[-1]
            + loop_pad_frames,
        ]
    else:
        generated = generated[..., prefix_frames:]

    # Write result
    stereo = inference_ctrls["pseudo_stereo_codebook"] is not None
    if stereo:
        expected_batch = 2 * rhythm_tokens.tokens.shape[0]
    else:
        expected_batch = rhythm_tokens.tokens.shape[0]
    if loop_pad_frames:
        assert generated.shape == (
            expected_batch,
            *rhythm_tokens.tokens.shape[1:-1],
            rhythm_tokens.tokens.shape[-1] + 2 * loop_pad_frames,
        )
    else:
        assert generated.shape == (expected_batch, *rhythm_tokens.tokens.shape[1:])
    generated_tokens = TokenSequence(
        tokens=generated,
        extras=dict(rhythm_tokens.extras),
    )
    if loop_pad_frames:
        generated_tokens.extras["signal_length"] = (
            rhythm_tokens.extras["signal_length"]
            + 2 * loop_pad_frames * tokenizer.hop_length
        )
    if stereo:
        generated_tokens.extras["n_channels"] = 2

    # Decode to audio
    out = tokenizer.decode(generated_tokens)
    if loop_pad_frames:
        start = loop_pad_frames * tokenizer.hop_length
        stop = start + rhythm_tokens.extras["signal_length"]
        out.audio_data = out.audio_data[..., start:stop]
    if stereo:
        x = out.audio_data
        mid = x.mean(dim=1, keepdim=True)
        side = x - mid
        x = mid + inference_ctrls["pseudo_stereo_width"] * side
        rms = torch.sqrt((x * x).mean(dim=(1, 2), keepdim=True) + 1e-12)
        out.audio_data = x * (db_to_linear(audio_ctrls["loudness_db"]) / rms)
    else:
        out.normalize(audio_ctrls["loudness_db"])
    out.ensure_max_of_audio()

    # Echo parameters
    msg = (
        f"Model='{LOADED['name']}' on {device} | sample_rate={sample_rate}, "
        f"prefix_dur={min(prefix_dur, timbre_prompt.duration):0.3f} | "
        f"top_p={sampling_ctrls['top_p']}, "
        f"top_k={sampling_ctrls['top_k']}, "
        f"temp={sampling_ctrls['temperature']}, "
        f"mask_temp={sampling_ctrls['mask_temperature']}; "
        f"seed={int(seed[0])}, "
        f"causal_bias={inference_ctrls['causal_bias']}, "
        f"cfg={inference_ctrls['cfg_scale']}, "
        f"timbre_vocab_codebooks={inference_ctrls['timbre_vocab_codebooks']}, "
        f"loopable={inference_ctrls['loopable']}, "
        f"loop_pad_frames={loop_pad_frames}, "
        f"pseudo_stereo_codebook={inference_ctrls['pseudo_stereo_codebook']}, "
        f"pseudo_stereo_width={inference_ctrls['pseudo_stereo_width']}"
    )
    return out, msg


########################################
# Gradio event handlers
########################################


def handle_audio_change(audio_tuple):
    if audio_tuple is None or audio_tuple[1] is None:
        return np.zeros((SPEC_MELS, SMALL_WIDTH, 3), dtype=np.uint8)
    signal = from_gradio_audio(*audio_tuple)
    return spectrogram_fast(signal, small=True)


def load_example_timbre():
    """
    Return a gr.Audio value and a spectrogram image.
    """
    signal = AudioSignal(TIMBRE_EXAMPLE)
    return to_gradio_audio(signal), spectrogram_fast(signal, small=True)


def load_example_rhythm():
    """
    Return a gr.Audio value and a spectrogram image.
    """
    signal = AudioSignal(RHYTHM_EXAMPLE)
    return to_gradio_audio(signal), spectrogram_fast(signal, small=True)


def on_select_model(name: str):
    msg = load_model_by_name(name)
    return msg


def on_generate(
    model_name,
    timbre_prompt,
    rhythm_prompt,
    top_p,
    top_k,
    temperature,
    mask_temperature,
    seed,
    causal_bias,
    cfg_scale,
    timbre_vocab_codebooks,
    pseudo_stereo_variation,
    pseudo_stereo_width,
    loudness_db,
    filter_inputs,
    window_overlap_frames,
    loopable,
    loop_pad_frames,
    *schedule_vals,
):
    # Ensure model up-to-date
    if LOADED["name"] != model_name:
        _ = load_model_by_name(model_name)
    sample_rate = LOADED["sample_rate"]

    sampling_ctrls = dict(
        top_p=float(top_p),
        top_k=int(top_k),
        temperature=float(temperature),
        mask_temperature=float(mask_temperature),
    )
    pseudo_stereo_variation = int(pseudo_stereo_variation)
    pseudo_stereo_width = float(pseudo_stereo_width)
    pseudo_stereo_codebook = (
        pseudo_stereo_variation
        if pseudo_stereo_variation < LOADED["tokenizer"].n_codebooks
        and pseudo_stereo_width > 0.0
        else None
    )
    inference_ctrls = dict(
        causal_bias=float(causal_bias),
        cfg_scale=float(cfg_scale),
        pseudo_stereo_codebook=pseudo_stereo_codebook,
        pseudo_stereo_width=pseudo_stereo_width,
        timbre_vocab_codebooks=int(timbre_vocab_codebooks),
        window_overlap_frames=int(window_overlap_frames),
        loopable=bool(loopable),
        loop_pad_frames=int(loop_pad_frames),
    )
    audio_ctrls = dict(
        loudness_db=float(loudness_db), filter_inputs=bool(filter_inputs)
    )
    schedule_ctrls = [int(v) for v in schedule_vals]

    seed = [seed + i for i in range(N_OUTPUTS)]

    # Convert to AudioSignal
    assert isinstance(timbre_prompt, tuple)
    assert isinstance(rhythm_prompt, tuple)
    timbre_prompt = from_gradio_audio(*timbre_prompt)
    rhythm_prompt = from_gradio_audio(*rhythm_prompt)

    # Batch inputs
    timbre_prompt = AudioSignal.batch([timbre_prompt] * N_OUTPUTS)
    rhythm_prompt = AudioSignal.batch([rhythm_prompt] * N_OUTPUTS)

    out, msg = _inference(
        timbre_prompt,
        rhythm_prompt,
        sampling_ctrls,
        inference_ctrls,
        audio_ctrls,
        schedule_ctrls,
        seed,
    )
    out = out.cpu()

    out_audio = [to_gradio_audio(out[i]) for i in range(N_OUTPUTS)]
    out_specs = [spectrogram_fast(out[i], small=False) for i in range(N_OUTPUTS)]

    return out_audio + out_specs + [msg]


########################################
# Interface
########################################


with gr.Blocks(title="TRIA: The Rhythm In Anything", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🥁 TRIA: The Rhythm In Anything 🥁")
    gr.Markdown(
        "Select a **Model**, load **Timbre** + **Rhythm** prompts (record or upload). "
        "Click **Generate**."
    )

    # Inputs row
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Timbre Prompt")
            timbre_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="Timbre Audio",
                show_download_button=True,
            )
            timbre_spec = gr.Image(
                label="Timbre Spectrogram", show_download_button=True
            )
            btn_timbre_ex = gr.Button("Load Example Timbre", variant="secondary")
        with gr.Column():
            gr.Markdown("### Rhythm Prompt")
            rhythm_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="Rhythm Audio",
                show_download_button=True,
            )
            rhythm_spec = gr.Image(
                label="Rhythm Spectrogram", show_download_button=True
            )
            btn_rhythm_ex = gr.Button("Load Example Rhythm", variant="secondary")

    # Outputs row
    gr.Markdown("### Generated Outputs (Batch of 3)")
    generate_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        with gr.Column():
            out_audio_1 = gr.Audio(
                type="numpy",
                label="Generated #1",
                interactive=False,
                show_download_button=True,
            )
            out_spec_1 = gr.Image(label="Spectrogram #1", show_download_button=True)
        with gr.Column():
            out_audio_2 = gr.Audio(
                type="numpy",
                label="Generated #2",
                interactive=False,
                show_download_button=True,
            )
            out_spec_2 = gr.Image(label="Spectrogram #2", show_download_button=True)
        with gr.Column():
            out_audio_3 = gr.Audio(
                type="numpy",
                label="Generated #3",
                interactive=False,
                show_download_button=True,
            )
            out_spec_3 = gr.Image(label="Spectrogram #3", show_download_button=True)
    params_text = gr.Markdown("")

    # Controls bottom: 2 columns (left: Model + Sampling + Audio; right: Schedule)
    with gr.Row():
        with gr.Column():
            # Model selector above everything on left
            model_names = list(MODEL_ZOO.keys())
            model_name = gr.Dropdown(
                choices=model_names, value=model_names[0], label="Model"
            )
            model_status = gr.Markdown("")

            with gr.Accordion("Sampling", open=False):
                with gr.Row():
                    top_p = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Top P")
                    top_k = gr.Slider(0, 1024, value=0, step=1, label="Top K")
                with gr.Row():
                    temperature = gr.Slider(
                        0.0, 2.0, value=1.0, step=0.01, label="Temperature"
                    )
                    mask_temperature = gr.Slider(
                        0.0, 50.0, value=10.5, step=0.1, label="Mask Temperature"
                    )
                with gr.Row():
                    seed = gr.Slider(0, 1000, value=0, step=1, label="Random Seed")
                    causal_bias = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.01, label="Causal Bias"
                    )
                    cfg_scale = gr.Slider(
                        0.0, 10.0, value=2.0, step=0.1, label="CFG Scale"
                    )
                timbre_vocab_codebooks = gr.Slider(
                    0, 9, value=9, step=1, label="Restrict Codebooks"
                )

            with gr.Accordion("Audio", open=False):
                with gr.Row():
                    loudness_db = gr.Slider(
                        -80.0, 0.0, value=-20.0, step=0.5, label="Loudness (dB)"
                    )
                    filter_inputs = gr.Checkbox(value=False, label="Filter Inputs")
                    window_overlap_frames = gr.Slider(
                        0, 64, value=10, step=1, label="Window Overlap Frames"
                    )

                with gr.Row():
                    loopable = gr.Checkbox(value=False, label="Loopable")
                    loop_pad_frames = gr.Slider(
                        0, 64, value=10, step=1, label="Loop Padding Frames"
                    )

                with gr.Row():
                    pseudo_stereo_variation = gr.Slider(
                        0, 9, value=5, step=1, label="Pseudo Stereo Variation"
                    )
                    pseudo_stereo_width = gr.Slider(
                        0.0, 2.0, value=1.0, step=0.05, label="Pseudo Stereo Width"
                    )

        with gr.Column():
            with gr.Accordion("Schedule", open=False):
                schedule_sliders = []
                for i in range(1, 10):
                    schedule_sliders.append(
                        gr.Slider(
                            0, 32, value=8, step=1, label=f"Codebook {i} Iterations"
                        )
                    )

    # Automatically update plots on audio upload
    timbre_audio.change(
        fn=handle_audio_change,
        inputs=timbre_audio,
        outputs=[timbre_spec],
        show_progress="hidden",
    )
    rhythm_audio.change(
        fn=handle_audio_change,
        inputs=rhythm_audio,
        outputs=[rhythm_spec],
        show_progress="hidden",
    )

    # Default examples
    btn_timbre_ex.click(
        fn=load_example_timbre,
        inputs=None,
        outputs=[timbre_audio, timbre_spec],
        show_progress="hidden",
    )
    btn_rhythm_ex.click(
        fn=load_example_rhythm,
        inputs=None,
        outputs=[rhythm_audio, rhythm_spec],
        show_progress="hidden",
    )

    # Load model and utilities
    model_name.change(
        fn=on_select_model,
        inputs=model_name,
        outputs=model_status,
        show_progress="hidden",
    )
    demo.load(fn=on_select_model, inputs=model_name, outputs=model_status)

    generate_btn.click(
        fn=on_generate,
        inputs=[
            model_name,
            timbre_audio,
            rhythm_audio,
            top_p,
            top_k,
            temperature,
            mask_temperature,
            seed,
            causal_bias,
            cfg_scale,
            timbre_vocab_codebooks,
            pseudo_stereo_variation,
            pseudo_stereo_width,
            loudness_db,
            filter_inputs,
            window_overlap_frames,
            loopable,
            loop_pad_frames,
            *schedule_sliders,
        ],
        outputs=[
            out_audio_1,
            out_audio_2,
            out_audio_3,
            out_spec_1,
            out_spec_2,
            out_spec_3,
            params_text,
        ],
        show_progress="full",
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)
