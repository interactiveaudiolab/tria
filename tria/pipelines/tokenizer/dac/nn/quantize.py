from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import WNConv1d

################################################################################
# Vector quantization modules for DAC bottleneck
################################################################################


class VectorQuantize(nn.Module):
    """
    Implementation of vector quantization similar to A. Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: perform nearest neighbor lookup in low-dimensional
           space for improved codebook usage
        2. L2-normalized codes: convert euclidean distance to cosine similarity
           to improve training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input tensor, i.e. latent dimension of VAE
        codebook_size : int
            Number of codebook vectors
        codebook_dim : int
            Dimension of codebook vectors
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)

        # Store `codebook_size` embedding vectors of dimension `codebook_dim`
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def embed_code(self, embed_id: torch.Tensor):
        """
        Map codebook indices to corresponding codebook vectors

        Parameters
        ----------
        embed_id : torch.Tensor
            Tensor of shape (batch_size, n_frames) containing codebook indices

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, n_frames, codebook_dim) containing
            corresponding codebook vectors. Note that the dimensions of the
            returned tensor are "out of order"
        """
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor):
        """
        Map codebook indices to corresponding codebook vectors

        Parameters
        ----------
        embed_id : torch.Tensor
            Tensor of shape (batch_size, n_frames) containing codebook indices

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, codebook_dim, n_frames) containing
            corresponding codebook vectors
        """
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        """
        Given continuous latents, quantize them using a fixed codebook with
        factorized, L2-normalized nearest-neighbor lookup

        Parameters
        ----------
        latents : torch.Tensor
            Input tensor of shape (n_batch, codebook_dim, n_frames)

        Returns
        -------
        torch.Tensor
            Quantized continuous representation of input of shape
            (n_batch, latent_dim, n_frames)
        torch.Tensor
            Codebook indices (quantized discrete representation of input),
            shape (n_batch, n_frames)
        """

        # Reshape to (batch_size * n_frames, codebook_dim) for nearest neighbor
        # lookup along codebook dimension
        encodings = latents.permute(0, 2, 1).reshape(-1, latents.shape[1])
        codebook = self.codebook.weight  # (codebook_size, codebook_dim)

        # L2-normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)  # (batch_size * n_frames, codebook_dim)
        codebook = F.normalize(codebook)  # (codebook_size, codebook_dim)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )  # (batch_size * n_frames, codebook_size)

        # Perform nearest neighbor lookup
        indices = (-dist).max(1)[1].view(latents.size(0), -1)  # (batch_size, n_frames)

        # Map indices to corresponding codebook vectors
        z_q = self.decode_code(indices)  # (batch_size, codebook_dim, n_frames)
        return z_q, indices

    def forward(self, z: torch.Tensor):
        """
        Quantize the input tensor using a fixed codebook and return the
        corresponding codebook vectors

        Parameters
        ----------
        z : torch.Tensor
            Input latents of shape (n_batch, latent_dim, n_frames)

        Returns
        -------
        z_i: torch.Tensor
            Continuous representation of inputs projected into codebook space,
            shape (n_batch, codebook_dim, n_frames)
        z_q: torch.Tensor
            Quantized representation of input in codebook space, shape
            (n_batch, codebook_dim, n_frames)
        z_o: torch.Tensor
            Continuous representation of quantized input, projected back into
            latent space, shape (n_batch, latent_dim, n_frames)
        codes:
            Codebook indices (quantized discrete representation of input),
            shape (n_batch, n_frames)
        """

        # Factorized codes (ViT-VQGAN): project input from latent space into
        # low-dimensional codebook space
        z_i = self.in_proj(z)  # (n_batch, codebook_dim, n_frames)

        # Quantize latents using nearest-neighbor lookup in codebook space
        z_q, codes = self.decode_latents(z_i)
        # z_q: (n_batch, codebook_dim, n_frames)
        # indices: (n_batch, n_frames)

        z_q = (
            z_i + (z_q - z_i).detach()
        )  # No-op in forward pass, straight-through gradient estimator in backward pass

        # Project quantized latents back into latent space
        z_o = self.out_proj(z_q)  # (n_batch, latent_dim, n_frames)

        return z_i, z_q, z_o, codes


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end-to-end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        input_dim : int, optional
            Latent dimension of input
        n_codebooks : int, optional
            Number of codebooks to use
        codebook_size : int, optional
            Number of vectors (quantized values) per codebook
        codebook_dim : int or list, optional
            Dimension of codebook vectors
        quantizer_dropout : float, optional
            Dropout probability for each quantizer
        """

        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z: torch.Tensor, n_quantizers: int = None):
        """
        Quantize the input tensor using a fixed set of `n` codebooks and return
        the corresponding codebook vectors.

        Parameters
        ----------
        z: torch.Tensor
            Input latents of shape (n_batch, latent_dim, n_frames)
        n_quantizers : int, optional
            Number of quantizers to use. If `self.quantizer_dropout` is True,
            this argument is ignored when in training mode, and a random number
            of quantizers is used.

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

        Informative metrics can be computed using the predicted representations,
        such as:
           * codebook error: Quantization error in codebook space at each level,
                             via `z_q - z_i`; this is numerically equivalent to
                             commitment error
           * latent error:   Quantization error in latent space across all
                             levels, via `z_O - z`. We can also compute the
                             latent error as quantizer levels are added via
                            `torch.cumsum(z_o, dim=1) - z.unsqueeze(1)`
        """

        # Quantized codebook vectors projected back into latent space, i.e.
        # quantization results
        z_O = 0.0

        # Residuals between input and quantized latents
        residual = z

        # Codebook indices
        codes = []

        # Quantization error in codebook space
        z_i, z_q, z_o = [], [], []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if self.training:
            # Determine quantizer dropout
            n_quantizers = (
                torch.ones((z.shape[0],)) * self.n_codebooks + 1
            )  # (n_batch,)
            dropout = torch.randint(
                1, self.n_codebooks + 1, (z.shape[0],)
            )  # (n_batch,)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)  # (n_batch,)

        for i, quantizer in enumerate(self.quantizers):
            # Allow limiting number of quantizers at inference
            if self.training is False and i >= n_quantizers:
                break

            # Apply current quantizer to residual to obtain:
            #
            #   * z_i_i: residual latents projected into codebook space
            #   * z_q_i: quantized codebook vectors in codebook space
            #   * z_o_i: quantized codebook vectors projected back into latent
            #            space
            #   * codes_i: codebook indices
            (
                z_i_i,  # (n_batch, codebook_dim, n_frames)
                z_q_i,  # (n_batch, codebook_dim, n_frames)
                z_o_i,  # (n_batch, latent_dim, n_frames)
                codes_i,  # (n_batch, n_frames)
            ) = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )

            # Add quantization result to output and optionally apply quantizer dropout
            z_O = z_O + z_o_i * mask[:, None, None]

            # Compute residual between input and output latent
            residual = residual - z_o_i

            codes.append(codes_i)
            z_i.append(z_i_i)
            z_q.append(z_q_i)
            z_o.append(z_o_i)

        codes = torch.stack(codes, dim=1)  # (n_batch, n_quantizers, n_frames)
        z_i = torch.stack(z_i, dim=1)  # (n_batch, n_quantizers, codebook_dim, n_frames)
        z_q = torch.stack(z_q, dim=1)  # (n_batch, n_quantizers, codebook_dim, n_frames)
        z_o = torch.stack(z_o, dim=1)  # (n_batch, n_quantizers, latent_dim, n_frames)

        return codes, z_O, z_i, z_q, z_o

    def from_codes(self, codes: torch.Tensor):
        """
        Given the quantized codes, reconstruct the continuous latent
        representation

        Parameters
        ----------
        codes : torch.Tensor
            Quantized discrete representation of input, shape
            (n_batch, n_quantizers, n_frames)

        Returns
        -------
        torch.Tensor
            Projected quantized representation of input in latent space,
            shape (n_batch, latent_dim, n_frames)
        """
        z_O = 0.0

        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_q_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_o_i = self.quantizers[i].out_proj(z_q_i)
            z_O = z_O + z_o_i

        return z_O

    def from_latents(self, z_i: torch.Tensor):
        """
        Given the projected un-quantized latents in codebook space,
        reconstruct the continuous representation after quantization.

        Parameters
        ----------
        z_i: torch.Tensor
            Projected un-quantized representation of input in codebook space,
            shape (n_batch, n_quantizers, codebook_dim, n_frames)

        Returns
        -------
        torch.Tensor
            Projected quantized representation of input in latent space,
            shape (n_batch, latent_dim, n_frames)
        """
        z_O = 0.0

        n_codebooks = z_i.shape[1]
        for i in range(n_codebooks):
            z_q_i = self.quantizers[i].decode_latents(z_i[:, i, :, :])
            z_o_i = self.quantizers[i].out_proj(z_q_i)
            z_O = z_O + z_o_i

        return z_O
