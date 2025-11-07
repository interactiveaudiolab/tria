import math
from typing import List
from typing import Union

import numpy as np
import torch
from torch import nn

from .modules import Decoder
from .modules import Encoder
from .modules import init_weights
from .nn.quantize import ResidualVectorQuantize

################################################################################
# Descript Audio Codec (DAC)
################################################################################


class DAC(torch.nn.Module):
    """
    Descript Audio Codec (DAC) proposed by Kumar et al. in "High-Fidelity Audio
    Compression with Improved RVQGAN" (2023). Code adapted from:
    https://github.com/descriptinc/descript-audio-codec
    """

    def __init__(
        self,
        sample_rate: int = 44_100,
        encoder_dim: int = 64,
        encoder_rates: List[int] = (2, 4, 8, 8),
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = (8, 8, 4, 2),
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)

        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.apply(init_weights)

        self.delay = self.get_delay()

        # As long as we don't run chunked/segmented encoding and decoding,
        # we can keep padding on
        self.padding = True

    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value: bool):
        assert isinstance(value, bool)

        layers = [
            l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length: int):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        """
        Encode given audio data and return quantized latent codes.

        Parameters
        ----------
        audio_data : torch.Tensor
            Audio data to encode, shape (batch_size, 1, n_samples)

        Returns
        -------
        codes:
            Codebook indices across all quantizer levels, shape
            (n_batch, n_quantizers, n_frames)
        z_O: torch.Tensor
            Quantized output obtained by summing projected quantized residuals
            (z_o) over all quantizer levels, shape (n_batch, latent_dim, n_frames)
        z_i: torch.Tensor
            Continuous representation of inputs projected into codebook space,
            shape (n_batch, n_quantizers, codebook_dim, n_frames). Note that
            each quantizer level represents a predicted residual.
        z_q: torch.Tensor
            Quantized representation of input in codebook space, shape
            (n_batch, n_quantizers, codebook_dim, n_frames). Note that each
            quantizer level represents a quantized predicted residual.
        z_o: torch.Tensor
            Continuous representation of quantized input, projected back into
            latent space, shape (n_batch, n_quantizers, latent_dim, n_frames).
            Note that each quantizer level represents a projected quantized
            predicted residual.
        """
        # Predict continuous latents
        z = self.encoder(audio_data)  # (n_batch, latent_dim, n_frames)
        return *self.quantizer(z, n_quantizers=None), z

    def decode(
        self,
        codes: torch.Tensor,
    ):
        """
        Decode given quantized latent codes and return audio data

        Parameters
        ----------
        codes : torch.Tensor
            Quantized latent codes, shape (n_batch, n_quantizers, n_frames)

        Returns
        -------
        torch.Tensor
            Decoded audio data, shape (n_batch, 1, n_samples)
        """
        z_O = self.quantizer.from_codes(codes)  # (n_batch, latent_dim, n_frames)
        recons = self.decoder(z_O)  # (n_batch, 1, n_samples)
        return recons
