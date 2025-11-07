import torch

from typing import Optional, Union, Iterable

from ..nn.transformer import Transformer
from .mask import cosine_schedule, format_seed
from .sample import mask_by_confidence, top_p_top_k, sample

################################################################################
# TRIA masked language model
################################################################################


class TRIA(torch.nn.Module):

    def __init__(
        self,
        codebook_size: int = 1024,
        n_codebooks: int = 9,
        n_feats: int = 2,
        n_channels: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        mult: int = 4,
        p_dropout: float = 0.0,
        p_token_dropout: float = 0.0,
        bias: bool = False,
        max_len: int = 8192,
        pos_enc: Optional[str] = "rope",
        qk_norm: bool = True,
        use_sdpa: bool = True,
        interp: str = "nearest",
        share_emb: bool = True,
    ):
        super().__init__()

        assert interp in ["nearest", "linear"]

        self.adapter = torch.nn.Linear(n_feats, n_channels, bias=bias)
        self.in_proj = torch.nn.Linear(2 * n_channels, n_channels, bias=bias)
        
        self.backbone = Transformer(
            n_channels=n_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            mult=mult,
            p_dropout=p_dropout,
            bias=False,
            max_len=max_len,
            pos_enc_self_attn=pos_enc,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
        )

        self.tokens_emb = torch.nn.Embedding(codebook_size * n_codebooks, n_channels)
        self.head = torch.nn.Linear(n_channels, codebook_size * n_codebooks, bias=False)  # No bias on head, to allow weight-sharing
        if share_emb:
            self.tokens_emb.weight = self.head.weight
        
        # Masked token embedding
        self.tokens_mask_emb = torch.nn.Parameter(torch.zeros(n_channels))
        
        # Attributes
        self.p_token_dropout = p_token_dropout
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.interp = interp
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        feats: torch.Tensor, 
        codebook: torch.Tensor,
        tokens_mask: torch.Tensor,
        feats_mask: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : torch.Tensor
            Acoustic tokens, fully or partially masked; shape 
            (n_batch, n_codebooks, n_frames)
        feats : torch.Tensor
            Aligned features to guide generation; shape (n_batch, n_feats, n_frames)
        codebook : torch.Tensor
            Codebook in which to predict masked tokens; shape (n_batch,)
        tokens_mask : torch.Tensor
            Boolean tensor indicating umasked token positions (True where 
            unmasked, False where masked); shape (n_batch, n_codebooks, n_frames)
        feats_mask : torch.Tensor
        """
        
        assert tokens.ndim == 3       # (n_batch, n_codebooks, n_frames)
        assert feats.ndim == 3        # (n_batch, n_feats, n_frames')
        assert tokens_mask.ndim == 3  # (n_batch, n_codebooks, n_frames)
        assert feats_mask.ndim == 2   # (n_batch, n_frames')
        assert tokens.shape[1] == self.n_codebooks
        
        n_batch, n_codebooks, n_frames = tokens.shape
        
        # Interpolate features and mask to tokens resulution
        feats = torch.nn.functional.interpolate(feats, n_frames, mode=self.interp)
        feats_mask = torch.nn.functional.interpolate(
            feats_mask[:, None, :].float(), n_frames, mode="nearest")
        
        # Adapt features
        feats = self.adapter(feats.transpose(1, 2))  # (n_batch, n_frames, n_channels)

        # Embed tokens
        codebook_offsets = torch.arange(
            n_codebooks, dtype=tokens.dtype, device=tokens.device
        ).reshape(1, -1, 1) * self.codebook_size   # (1, n_codebooks, 1)
        tokens = tokens + codebook_offsets         # (n_batch, n_codebooks, n_frames)        
        tokens_emb = self.tokens_emb(tokens)       # (n_batch, n_codebooks, n_frames, n_channels)

        # Zero masked token embeddings
        tokens_emb = tokens_emb * tokens_mask.unsqueeze(-1).float()

        # Apply learned embedding to masked token positions in current codebook
        mask_pos = torch.arange(
            n_codebooks, dtype=tokens.dtype, device=tokens.device
        )[None, :] == codebook[:, None]  # (n_batch, n_codebooks)
        mask_pos = torch.logical_and(mask_pos.unsqueeze(-1), ~tokens_mask)  # (n_batch, n_codebooks, n_frames)
        
        tokens_emb = tokens_emb + (
            mask_pos.unsqueeze(-1).float()
        ) * self.tokens_mask_emb.reshape(1, 1, 1, -1)  # (n_batch, n_codebooks, n_frames, n_channels)
        
        # Token dropout (encourage attention to unmasked frames)
        if self.training and self.p_token_dropout > 0.0:

            # Apply dropout within masked frames and "below" current codebook
            below = torch.arange(
                n_codebooks, device=tokens.device
            )[None, :, None] < codebook[:, None, None]  # (n_batch, n_codebooks, 1)
            eligible = below & feats_mask.bool()       # (n_batch, n_codebooks, n_frames)            
            drop = (
                torch.rand(
                    n_batch, 1, n_frames, 1, device=tokens.device
                ) < self.p_token_dropout) & eligible[..., None]
            tokens_emb = tokens_emb.masked_fill(drop, 0.0)

        # Zero "ignored" features
        feats = feats * feats_mask.transpose(1, 2)

        # Sum embedded tokens across codebooks
        tokens_emb = tokens_emb.sum(dim=1)      # (n_batch, n_frames, n_channels)

        # Sum embedded tokens and adapted features
        x = torch.cat([feats, tokens_emb], dim=-1)  # (n_batch, n_frames, 2 * n_channels)
        x = self.in_proj(x)                         # (n_batch, n_frames, n_channels)
        
        # Process with transformer
        x = self.backbone(x=x)  # (n_batch, n_frames, n_channels)

        # Predict token logits
        logits = self.head(x)   # (n_batch, n_frames, n_codebooks * codebook_size)
        logits = logits.reshape(
            n_batch, n_frames, n_codebooks, self.codebook_size
        ).permute(0, 2, 1, 3)   # (n_batch, n_codebooks, n_frames, codebook_size)
        
        return logits

    @torch.inference_mode()
    def inference(
        self,
        tokens: torch.Tensor,
        feats: torch.Tensor,
        tokens_mask: torch.Tensor,
        feats_mask: torch.Tensor,
        top_p: Union[float, Iterable[float]] = 1.0,
        top_k: Union[int, Iterable[int]] = None,
        temp: Union[float, Iterable[float]] = 1.0,
        mask_temp: Union[float, Iterable[float]] = 10.5,
        iterations: Union[int, Iterable[int]] = 8,
        guidance_scale: Union[float, Iterable[float]] = None,
        causal_bias: Union[float, Iterable[float]] = None,
        seed: Union[int, Iterable[int]] = None,
    ):
        
        assert not self.training
        device = next(iter(self.parameters())).device
    
        # Avoid overwriting
        tokens = tokens.clone().to(device)
        tokens_mask = tokens_mask.clone().to(device)
        
        assert tokens.ndim == 3
        n_batch, n_codebooks, n_frames = tokens.shape
    
        assert feats.ndim == 3
        _, n_feats, _ = feats.shape
    
        assert n_codebooks == self.n_codebooks
        assert n_feats == self.n_feats
    
        # Interpolate features to token resolution
        feats = torch.nn.functional.interpolate(
            feats.to(device), n_frames, mode=self.interp,
        )
        feats_mask = torch.nn.functional.interpolate(
            feats_mask.unsqueeze(1).float().to(device), n_frames, mode="nearest",
        ).squeeze(1).to(feats_mask.dtype)
    
        # Account for per-codebook args
        def _to_codebooks(v):
            if isinstance(v, torch.Tensor):
                v = v.tolist()
            elif isinstance(v, Iterable):
                pass
            else:
                v = [v]
    
            if len(v) == n_codebooks:
                return v
            elif len(v) == 1:
                return v * n_codebooks
            else:
                raise ValueError(
                    f"Sampling parameters must be scalars, "
                    f"length-1 iterable, or length-n_codebooks ({n_codebooks})"
                )

        # Construct `n_codebooks` state lists of length `n_batch` each
        seed = seed or 0
        if not isinstance(seed, Iterable):
            seed = [seed]
        assert len(seed) in [1, n_batch]
        seed = seed * (n_batch // len(seed))
        state = [format_seed([s + 10007 * cb for s in seed]) for cb in range(n_codebooks)]
        
        top_p, top_k = _to_codebooks(top_p), _to_codebooks(top_k)
        temp, mask_temp = _to_codebooks(temp), _to_codebooks(mask_temp)
        iterations = _to_codebooks(iterations)
        guidance_scale = _to_codebooks(guidance_scale)
        causal_bias = _to_codebooks(causal_bias)
    
        # Track initial masked token counts
        n_masked_init = (~tokens_mask).long().sum(dim=-1)  # (n_batch, n_codebooks)

        # Generate one codebook at a time
        for codebook_idx, (
            _state, _top_p, _top_k, _temp, _mask_temp, 
            _iterations, _guidance_scale, _causal_bias,
        ) in enumerate(zip(
            state, top_p, top_k, temp, mask_temp, 
            iterations, guidance_scale, causal_bias,
        )):
            _causal_bias = _causal_bias or 0.
            assert 0. <= _causal_bias
    
            _temp = _temp or 1.0
            assert 0. < _temp
    
            _mask_temp = _mask_temp or 0.0
            assert 0. <= _mask_temp
    
            _iterations = max(_iterations or 1, 1)
    
            for _iter in range(_iterations):                  
                
                # CFG on features by masking
                if _guidance_scale:
                    tokens_cfg = torch.cat([tokens, tokens], dim=0)
                    tokens_mask_cfg = torch.cat([tokens_mask, tokens_mask], dim=0)
    
                    feats_cfg = torch.cat([feats, feats], dim=0)
                    feats_mask_cfg = torch.cat([feats_mask, torch.zeros_like(feats_mask)], dim=0)
    
                    logits_cond, logits_uncond = self.forward(
                        tokens_cfg, 
                        feats_cfg, 
                        torch.full(
                            (tokens_cfg.shape[0],), 
                            codebook_idx, 
                            dtype=torch.long, 
                            device=device,
                        ),
                        tokens_mask_cfg, 
                        feats_mask_cfg,
                    ).chunk(2, dim=0)  # (n_batch, n_codebooks, n_frames, codebook_size) x2
    
                    logits = logits_uncond + _guidance_scale * (logits_cond - logits_uncond)  # (n_batch, n_codebooks, n_frames, codebook_size)
                
                else:
                    logits = self.forward(
                        tokens, 
                        feats, 
                        torch.full(
                            (tokens.shape[0],), 
                            codebook_idx, 
                            dtype=torch.long, 
                            device=device,
                        ),
                        tokens_mask, 
                        feats_mask,
                    )  # (n_batch, n_codebooks, n_frames, codebook_size)

                # Truncate logits and sample tokens at masked positions
                logits = top_p_top_k(
                    logits[:, codebook_idx:codebook_idx+1, ...], _top_p, _top_k
                )  # (n_batch, 1, n_frames, codebook_size)
                sampled, probs = sample(
                    logits, _temp, argmax=(_iter==_iterations-1),
                )  # (n_batch, 1, n_frames) x2
                write_idx = ~(tokens_mask[:, codebook_idx, :])  # (n_batch, n_frames)
                tokens[:, codebook_idx, :][write_idx] = sampled[:, 0, :][write_idx]
                
                # Compute implied generation timestep and corresponding target mask
                # ratio
                t = (_iter + 1) / _iterations
                tgt_p_mask = cosine_schedule(torch.tensor([t]*n_batch, device=device))  # (n_batch,)
    
                # Compute target and actual number of masked positions in current
                # codebook
                tgt_n_masked = torch.floor(tgt_p_mask * n_masked_init[:, codebook_idx]).long()  # (n_batch,)
                n_masked = write_idx.long().sum(dim=-1)  # (n_batch,)
    
                # Do not complete unmasking until final iteration, i.e. always leave at 
                # least one token unmasked 
                if _iter < _iterations - 1:
                    tgt_n_masked = torch.minimum(n_masked - 1, tgt_n_masked).clamp_min(1)

                # Select which tokens to unmask via confidence (assigned probability), 
                # mediated by causal bias and random noise                
                _probs = torch.full_like(probs[:, 0, :], torch.inf)  # (n_batch, n_frames)
                _probs[write_idx] = probs[:, 0, :][write_idx]                
                tokens_mask[:, codebook_idx, :] = mask_by_confidence(
                    probs=_probs,
                    n=tgt_n_masked,
                    temp=_mask_temp * (1 - t),  # Mask temperature annealing
                    causal_bias=_causal_bias or 0.0,
                    state=_state,
                    eligible=write_idx,
                )
    
                # Re-apply span and codebook masks
                tokens_mask = ~torch.logical_and(~tokens_mask, feats_mask.unsqueeze(1))
                tokens_mask[:, :codebook_idx, :] = True

        return tokens
