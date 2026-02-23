from dataclasses import dataclass
from typing import Dict

import torch
from audiotools import AudioSignal

from ...constants import PRETRAINED_DIR
from ...dsp import resample
from .dac import DAC

################################################################################
# Pipeline for discrete and continuous audio tokenizers
################################################################################


@dataclass
class TokenSequence:
    extras: Dict
    tokens: torch.Tensor


class Tokenizer(torch.nn.Module):
    valid = ["dac"]
    ckpt_dir = PRETRAINED_DIR / "tokenizer"

    def __init__(
        self,
        name: str = "dac",
        sample_rate: int = None,
        bitrate: float = None,
        **kwargs,
    ):
        super().__init__()

        assert name in self.valid, (
            f"Invalid tokenizer name {name}; " f"must be one of {self.valid}"
        )

        model_cls = None
        cfg = {}
        self.name = None
        self.model = None
        self.ckpt_pth = None

        # Tokenizer attributes
        self.sample_rate = None
        self.n_channels = None
        self.n_codebooks = None
        self.bitrate = None
        self.codebook_size = None
        self.hop_length = None
        self.normalize_db = None

        # DAC 7.7kbps 44.1kHz: 9 RVQ codebooks, each of size 1024, with
        # hop length 512 (86Hz frame rate)
        if name == "dac":
            # Only support single DAC configuration
            assert (sample_rate or 44_100) == 44_100
            assert (bitrate or 7.7) == 7.7

            self.name = "dac"
            self.sample_rate = 44_100
            self.n_channels = 1
            self.n_codebooks = 9
            self.bitrate = 7.7
            self.codebook_size = 1024
            self.hop_length = 512
            self.normalize_db = -16.0

            ckpt_pth = self.ckpt_dir / "dac" / "dac_44.1kHz_7.7kbps.pt"
            model_cls = DAC
            cfg = {
                "sample_rate": 44_100,
                "encoder_dim": 64,
                "encoder_rates": (2, 4, 8, 8),
                "decoder_dim": 1536,
                "decoder_rates": (8, 8, 4, 2),
                "n_codebooks": 9,
                "codebook_size": 1024,
                "codebook_dim": 8,
                "quantizer_dropout": 0.0,
            }

        else:
            raise NotImplementedError(f"Tokenizer {name} not yet implemented")

        # Initialize
        if ckpt_pth is not None:
            self.model = model_cls(**cfg)
            self.model.load_state_dict(torch.load(ckpt_pth, map_location="cpu"))

        self.model.eval()

        # Check for attributes
        for a in [
            self.name,
            self.model,
            self.sample_rate,
            self.n_channels,
            self.n_codebooks,
            self.bitrate,
            self.codebook_size,
            self.hop_length,
        ]:
            assert a is not None

    @torch.no_grad()
    def _rms_db(self, x: torch.Tensor, eps: float = 1e-12):
        assert x.dim() == 3  # (n_batch, n_channels, n_samples)
        rms = torch.sqrt((x * x).mean(dim=(1, 2)) + eps)
        return 20.0 * torch.log10(torch.clamp(rms, min=eps))

    @torch.no_grad()
    def _db_to_gain(self, db: torch.Tensor, device, dtype):
        return torch.pow(torch.tensor(10.0, device=device, dtype=dtype), db / 20.0)

    @torch.no_grad()
    def _preprocess(self, signal: AudioSignal):
        x = signal.audio_data
        assert x.ndim == 3
        n_batch, orig_n_channels, orig_signal_length = x.shape
        orig_sample_rate = signal.sample_rate
        orig_loudness = self._rms_db(x)

        if self.normalize_db is not None:
            target_db = float(self.normalize_db)
            gain_db = target_db - orig_loudness
            gain = self._db_to_gain(
                gain_db.unsqueeze(-1).unsqueeze(-1), x.device, x.dtype
            )
            x = x * gain

        # Handle channel mismatch by folding channels into batch
        if self.n_channels == 1 and orig_n_channels > 1:
            # Mono codec, multichannel audio
            x = x.reshape(n_batch * orig_n_channels, 1, orig_signal_length)
        elif self.n_channels > 1 and orig_n_channels == 1:
            # Multichannel codec, mono audio
            assert (
                self.n_channels // orig_n_channels
            ) * orig_n_channels == self.n_channels
            x = x.repeat(1, self.n_channels // orig_n_channels, 1)
        elif self.n_channels != orig_n_channels:
            raise ValueError(
                f"Channel mismatch not supported: model expects {self.n_channels}, "
                f"but got {orig_n_channels}"
            )

        # Wrap into AudioSignal and resample to tokenizer sample rate
        preprocessed = resample(
            AudioSignal(x, sample_rate=orig_sample_rate),
            self.sample_rate
        )

        return (
            preprocessed,
            orig_sample_rate,
            orig_n_channels,
            orig_loudness,
            orig_signal_length,
        )

    @torch.no_grad()
    def encode(self, signal: AudioSignal) -> TokenSequence:
        extras = {}

        # Store original attributes, normalize audio if required, and handle
        # channel count mismatch by folding channels into batch dimension and
        # encoding/decoding separately
        (
            preprocessed,
            orig_sample_rate,
            orig_n_channels,
            orig_loudness,
            orig_signal_length,
        ) = self._preprocess(signal)

        extras.update(
            {
                "sample_rate": orig_sample_rate,
                "n_channels": orig_n_channels,
                "loudness": orig_loudness,
                "signal_length": orig_signal_length,
            }
        )

        # Encode preprocessed audio to obtain tokens
        _tokens, *_ = self.model.encode(preprocessed.audio_data)
        assert _tokens.ndim == 3  # (n_batch, n_codebooks or latent_dim, n_frames)

        return TokenSequence(
            tokens=_tokens,
            extras=extras,
        )

    @torch.no_grad()
    def decode(self, tokens: TokenSequence) -> AudioSignal:
        # Decode tokens to obtain audio
        _tokens, extras = tokens.tokens, tokens.extras

        decoded = self.model.decode(_tokens)  # (n_batch', n_channels', n_samples')
        assert decoded.ndim == 3

        out = AudioSignal(decoded, sample_rate=self.sample_rate)

        # Restore original attributes
        (orig_sample_rate, orig_n_channels, orig_loudness, orig_signal_length) = (
            extras.get("sample_rate", self.sample_rate),
            extras.get("n_channels", self.n_channels),
            extras.get("loudness", None),
            extras.get("signal_length", None),
        )
        assert out.num_channels == self.n_channels

        out = resample(out, orig_sample_rate)

        if self.n_channels < orig_n_channels:
            # Mono codec, multichannel audio
            assert self.n_channels == 1
            n_batch_folded = out.shape[0]
            fold_factor = orig_n_channels // self.n_channels
            n_batch = n_batch_folded // fold_factor
            out.audio_data = out.audio_data.view(
                n_batch, fold_factor, out.num_channels, -1
            ).reshape(n_batch, orig_n_channels, -1)

        elif self.n_channels > orig_n_channels:
            # Multichannel codec, mono audio
            out = out.to_mono()

        n_batch = out.shape[0]

        x = out.audio_data
        cur_db = (
            self._rms_db(x).to(x.device).unsqueeze(-1).unsqueeze(-1)
        )  # (n_batch, 1, 1)
        target_db = orig_loudness.view(n_batch, 1, 1)
        gain_db = target_db - cur_db
        gain = self._db_to_gain(gain_db, x.device, x.dtype)
        x = x * gain
        out.audio_data = x

        # Restore original audio length (truncate / pad)
        cur_len = out.audio_data.shape[-1]
        if cur_len > orig_signal_length:
            out.audio_data = out.audio_data[..., :orig_signal_length]
        elif cur_len < orig_signal_length:
            out = out.zero_pad_to(orig_signal_length)

        return out
