"""
Continuous (non-tokenized) transformer autoencoder.

Compresses a whole continuous-valued sequence down into a single latent
bottleneck, then reconstructs the original sequence from it. Unlike a normal
token autoencoder, inputs/outputs here are real-valued vectors rather than
discrete ids (see `continuous.ContinuousTransformerWrapper`).

The motivation is to learn a compact "rl token" - a compressed state suitable
for downstream RL fine-tuning (physical-intelligence pi). See
https://www.pi.website/research/rlt

Flow: encode (transformer encoder + masked mean pool) -> bottleneck
(deterministic projection, or reparameterized gaussian with a KL penalty) ->
decode (prepend the latent as a token, transformer decoder reconstructs seq).
Losses: reconstruction (L1 or L2) plus optional bottleneck aux loss (KL).
"""

from __future__ import annotations
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

from x_transformers.x_transformers import Encoder, Decoder
from x_transformers.continuous import ContinuousTransformerWrapper

from einops.layers.torch import Rearrange

from torch_einops_utils import masked_mean, lens_to_mask

# rl token (continuous autoencoder)
# https://www.pi.website/research/rlt

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# bottleneck modules
# forward returns (latents, aux_loss | None)

class VariationalLatentBottleneck(Module):
    """Variational bottleneck - VAE-style reparameterized gaussian latent + KL.

    Projects the pooled encoding to a (mean, log_var) pair, samples a latent via
    the reparameterization trick, and returns a KL divergence loss to keep the
    posterior close to a standard normal prior.
    """

    def __init__(
        self,
        dim,
        dim_latent,
        kl_div_floor = 0.       # free-bits style floor; stored but currently unused - upstream v2.24.1 dropped the relu(kl - floor) step
    ):
        super().__init__()
        self.kl_div_floor = kl_div_floor

        # one linear producing both mean and log variance, split into two along the channel dim
        self.to_mean_log_var = nn.Sequential(
            nn.Linear(dim, dim_latent * 2),
            Rearrange('b (two d) -> two b d', two = 2)
        )

    def forward(self, x, return_unreduced_loss = False):
        mean, log_var = self.to_mean_log_var(x)
        std = (0.5 * log_var).exp()

        # reparameterization trick: sample = mean + std * eps, eps ~ N(0, 1)
        latents = mean + std * torch.randn_like(mean)

        # KL( N(mean, var) || N(0, 1) ) per dimension
        kl_loss = 0.5 * (
            log_var.exp() + mean.square() - log_var - 1.
        )

        # sum over the latent dim, leaving one KL value per batch sample
        kl_loss = kl_loss.sum(dim = -1)

        # reduce to a scalar unless the caller wants the per-sample values
        if not return_unreduced_loss:
            kl_loss = kl_loss.mean()

        return latents, kl_loss

class DeterministicLatentBottleneck(Module):
    """Deterministic bottleneck - plain linear projection to the latent dim.

    No sampling, no aux loss (returns None in its place).
    """

    def __init__(
        self,
        dim,
        dim_latent,
        activation = None
    ):
        super().__init__()
        self.proj = nn.Linear(dim, dim_latent)
        self.activation = default(activation, nn.Identity())

    def forward(self, x, return_unreduced_loss = False):
        latents = self.activation(self.proj(x))
        return latents, None     # None signals "no auxiliary loss" to the autoencoder

# main class

class ContinuousTransformerAutoencoder(Module):
    """Continuous sequence autoencoder built from two transformer wrappers.

    Encoder pools the sequence into one vector, a bottleneck compresses it into a
    latent, and the decoder reconstructs the sequence conditioned on that latent
    (prepended as a single token). Supports deterministic or variational
    bottlenecks and L1/L2 reconstruction loss.

    `dim` is the transformer's working width; `dim_input` is the width of the
    sequence being autoencoded and `dim_latent` the width of the code. Both
    default to `dim`, in which case the wrappers' in/out projections are dropped
    entirely via `use_identity_if_same_dim`.
    """

    def __init__(
        self,
        *,
        dim,
        enc_depth,
        dec_depth,
        max_seq_len,
        dim_input = None,       # width of the continuous vectors being autoencoded; defaults to the model dim, set it when the data width differs from the transformer width
        dim_latent = None,
        bottleneck: Module | None = None,
        bottleneck_type: Literal['deterministic', 'variational'] = 'deterministic',
        deterministic_bottleneck_kwargs: dict = dict(),
        variational_bottleneck_kwargs: dict = dict(),
        loss_type: Literal['l1', 'l2'] = 'l2',
        attn_dim_head = 64,
        heads = 8,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        latents_dropout_prob = 0.5,
        **kwargs
    ):
        super().__init__()
        # three widths are decoupled here: dim (transformer working width),
        # dim_input (the data being reconstructed), dim_latent (bottleneck code).
        # both default to dim, which reproduces the pre-`dim_input` behavior
        dim_input = default(dim_input, dim)
        dim_latent = default(dim_latent, dim)

        # encoder: continuous-vector transformer, used in `return_embeddings` mode for pooling
        self.encoder = ContinuousTransformerWrapper(
            dim_in = dim_input,
            max_seq_len = max_seq_len,
            use_identity_if_same_dim = True,    # dim_input == dim (the default) -> no project_in at all, rather than a square matmul
            attn_layers = Encoder(
                dim = dim,
                depth = enc_depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **enc_kwargs
            )
        )

        # bottleneck: use a provided module, else build by type
        if exists(bottleneck):
            self.bottleneck = bottleneck
        elif bottleneck_type == 'deterministic':
            self.bottleneck = DeterministicLatentBottleneck(dim, dim_latent, **deterministic_bottleneck_kwargs)
        elif bottleneck_type == 'variational':
            self.bottleneck = VariationalLatentBottleneck(dim, dim_latent, **variational_bottleneck_kwargs)
        else:
            raise ValueError(f'unknown bottleneck type {bottleneck_type}')

        # maps the latent back to model dim and shapes it as a single prepended token (b d -> b 1 d)
        self.from_latent_to_prepend_token = nn.Sequential(
            nn.Linear(dim_latent, dim),
            Rearrange('b d -> b 1 d')
        )

        # decoder: reconstructs the continuous sequence (dim -> dim) from the prepended latent token
        # since `dim_input` was introduced that round trip is dim_input -> dim_input, with the
        # transformer still working at `dim` in between; the two coincide by default
        self.decoder = ContinuousTransformerWrapper(
            dim_in = dim_input,
            dim_out = dim_input,    # reconstruction target is the raw input width, so recon loss compares against `seq` directly
            max_seq_len = max_seq_len,
            use_identity_if_same_dim = True,    # both projections collapse to Identity when dim_input == dim
            attn_layers = Decoder(
                dim = dim,
                depth = dec_depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **dec_kwargs
            )
        )

        # randomly drop the latent during training so the decoder cannot rely on it alone
        self.latents_dropout = nn.Dropout(latents_dropout_prob)

        assert loss_type in {'l1', 'l2'}
        self.loss_type = loss_type

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, seq, lens = None, return_aux_loss = False, return_unreduced_loss = False):
        """Encode a continuous sequence into a single latent (plus optional aux loss)."""
        mask = lens_to_mask(lens, max_len = seq.shape[1]) if exists(lens) else None

        # run encoder and keep raw embeddings, then masked-mean-pool seq -> single vector
        encoded = self.encoder(seq, mask = mask, return_embeddings = True)

        pooled = masked_mean(encoded, mask, dim = 1)

        latents, aux_loss = self.bottleneck(pooled, return_unreduced_loss = return_unreduced_loss)

        if not return_aux_loss:
            return latents

        return latents, aux_loss

    def forward(
        self,
        seq,
        lens = None,
        return_all_losses = False,
        return_unreduced_loss = False
    ):
        """Full encode -> bottleneck -> decode pass, returning the training loss(es)."""
        batch, seq_len, device = *seq.shape[:2], seq.device

        latents, aux_loss = self.encode(seq, lens = lens, return_aux_loss = True, return_unreduced_loss = return_unreduced_loss)

        # latent dropout
        # boolean mask of samples whose latent is dropped this step

        dropped_latents = ~self.latents_dropout(torch.ones((batch,), device = device)).bool()

        prepend_embeds = self.from_latent_to_prepend_token(latents)

        # decode: latent prepended as token 0; seq_start_pos shifts dropped samples so the
        # latent token sits "before" the sequence and is effectively ignored
        recon = self.decoder(
            seq,
            prepend_embeds = prepend_embeds,
            seq_start_pos = dropped_latents.long()
        )

        # slice out the prepended latent position

        recon = recon[:, 1:]

        # reconstruction loss

        if self.loss_type == 'l1':
            recon_loss = F.l1_loss(recon, seq, reduction = 'none')
        elif self.loss_type == 'l2':
            recon_loss = F.mse_loss(recon, seq, reduction = 'none')

        mask = lens_to_mask(lens, max_len = seq_len) if exists(lens) else None

        # average the per-element recon loss, ignoring padding positions
        # unreduced: average over (seq, feature) only, keeping the batch dim
        # otherwise: average over every dim down to a scalar

        dim = slice(1, None) if return_unreduced_loss else None
        recon_loss = masked_mean(recon_loss, mask, dim = dim)

        # total loss

        total_loss = recon_loss

        # add bottleneck aux loss (e.g. KL) when present (variational bottleneck)
        if exists(aux_loss):
            total_loss = total_loss + aux_loss

        if not return_all_losses:
            return total_loss

        return total_loss, (recon_loss, aux_loss)
