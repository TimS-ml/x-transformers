"""
Continuous-valued transformer wrappers.

These are the continuous analogues of the token-based wrappers: inputs are
real-valued vectors (e.g. image patches, sensor readings, world-model states)
instead of discrete token ids. `ContinuousTransformerWrapper` handles the
project-in / attention / project-out plumbing with optional probabilistic
(gaussian) outputs, while `ContinuousAutoregressiveWrapper` adds autoregressive
training, multi-step rollout, and sampling over continuous values.
"""

from __future__ import annotations

import torch
from torch import nn, cat, stack, arange
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Normal

import einx
from einops import rearrange, reduce, pack, repeat, unpack

from x_transformers.autoregressive_wrapper import align_right

from x_transformers.x_transformers import (
    Attention,
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    LayerNorm,
    masked_mean,
    always,
    pad_at_dim,
    slice_right_at_dim
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if not isinstance(d, Module) and callable(d) else d

def sample_from_mean_variance(
    mean,
    variance,
    eps = 1e-5,
    temperature = 1.
):
    # draw a sample from N(mean, variance); temperature scales the std for sharper/softer sampling
    std = variance.clamp(min = eps).sqrt()
    return torch.normal(mean, std * temperature)

def batch_masked_mean(t, mask):
    # one scalar per sample: sum over the feature dim, averaged over the unmasked positions
    # (den clamped to avoid div-by-zero). renamed from `masked_mean` upstream: the local def
    # shadowed the imported masked_mean, so average_pool_embed got (b,) back instead of (b, d)
    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)

    num = reduce(t, 'b n d -> b', 'sum')
    den = mask.sum(dim = -1)

    masked_average = num / den.clamp(min = 1.)
    return masked_average

# probabilistic loss fn

class GaussianNLL(Module):
    """Gaussian negative log-likelihood loss for probabilistic continuous heads.

    `pred` is a (mean, variance) pair; scores `target` under that gaussian.
    """

    def forward(self, pred, target):
        mean, var = pred
        return F.gaussian_nll_loss(mean, target, var, reduction = 'none')

# main classes

class ContinuousTransformerWrapper(Module):
    """Transformer wrapper for continuous vector sequences (no token embedding).

    Projects continuous inputs into model dim (`project_in`), adds positional
    embeddings, runs the attention layers, then projects out to either a plain
    continuous prediction or a gaussian (mean, variance) head when
    `probabilistic = True`. Supports masking, memory tokens, prepended embeds,
    average pooling, and kv caching.

    Either projection collapses to `nn.Identity` when its target width is not
    requested, or - with `use_identity_if_same_dim` - when the requested width
    already matches the model dim, so a wrapper can be asked for `dim_in`/`dim_out`
    unconditionally without paying for a redundant square matmul.
    """

    def __init__(
        self,
        *,
        max_seq_len = None,
        attn_layers: AttentionLayers,
        dim_in = None,
        dim_out = None,
        project_in: Module | None = None,
        project_out: Module | None = None,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        average_pool_embed = False,
        probabilistic = False,
        use_identity_if_same_dim = False,   # skip the in/out linear entirely (nn.Identity) when the requested width already equals the model dim, instead of learning a square projection
    ):
        super().__init__()
        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        # decide whether to use an absolute positional embedding at all (else identity / handled by attn layers)
        no_abs_pos_emb = not exists(max_seq_len) or max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # average pool

        self.average_pool_embed = average_pool_embed

        # project in and out

        assert not (exists(dim_in) and exists(project_in)), 'either `dim_in` or `project_in` can be passed in, but not both'
        assert not (exists(dim_out) and exists(project_out)), 'either `dim_out` or `project_out` can be passed in, but not both'

        # project continuous input -> model dim (identity if neither dim_in nor project_in given, or if use_identity_if_same_dim and dim_in already equals dim)
        has_project_in = exists(dim_in) and (not use_identity_if_same_dim or dim_in != dim)
        self.project_in = default(project_in, lambda: nn.Linear(dim_in, dim, bias = False) if has_project_in else nn.Identity())

        # output is multipled by 2 for outputting mean and log variance

        self.probabilistic = probabilistic

        # project out to dim_out (doubled when probabilistic, to carry mean + log variance)
        dim_out_with_prob = (dim_out * (2 if probabilistic else 1)) if exists(dim_out) else None

        # identity if neither dim_out nor project_out given, or if use_identity_if_same_dim and that (possibly doubled) width already equals dim
        has_project_out = exists(dim_out) and (not use_identity_if_same_dim or dim_out_with_prob != dim)

        self.project_out = default(project_out, lambda: nn.Linear(dim, dim_out_with_prob, bias = False) if has_project_out else nn.Identity())

        # can cache kv
        # only safe to cache if every attention module supports it

        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention)])

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        lens = None,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        sum_embeds = None,
        prepend_embeds = None,
        prepend_mask = None,
        cache: LayerIntermediates | None = None,
        input_not_include_cache = False,
        seq_start_pos = None,
        **kwargs
    ):
        """Project in, attend, project out; returns predictions or gaussian (mean, variance)."""
        batch, seq, orig_mask, device = *x.shape[:2], mask, x.device

        # maybe seq lengths passed in
        # build a boolean mask from per-sample lengths

        if exists(lens):
            assert not exists(mask), 'either `mask` or `lens` passed in, but not both'
            seq_arange = arange(seq, device = device)

            mask = einx.less('j, i -> i j', seq_arange, lens)

        # take care of position embedding offsets in the presence of cache and sequence is less than cache length (not full sequence)

        seq_pos_offset = 0

        if exists(cache) and input_not_include_cache:
            seq_pos_offset = cache.cache_length

        # project in + positional embedding

        x = self.project_in(x)
        x = x + self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos, offset = seq_pos_offset)

        if exists(sum_embeds):
            x = x + sum_embeds

        x = self.post_emb_norm(x)

        # memory tokens
        # prepend learned memory tokens to the sequence (packed so they can be split out later)

        if self.has_memory_tokens:
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((batch, seq), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((batch, prepend_seq), device = device, dtype = torch.bool))

                mask = cat((prepend_mask, mask), dim = -1)

        x = self.emb_dropout(x)

        # attention layers

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, input_not_include_cache = input_not_include_cache, seq_pos_offset = seq_pos_offset, return_hiddens = True, **kwargs)

        # splice out memory tokens
        # recover the memory token outputs and the real sequence

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        # maybe linear project out

        # skip projection when caller wants raw embeddings (e.g. for pooling in the autoencoder)
        out = self.project_out(x) if not return_embeddings else x

        if not return_embeddings and self.probabilistic:
            # split the doubled output into mean and log variance, then exponentiate to get variance
            mean, log_var = rearrange(out, '... (d mean_log_var) -> mean_log_var ... d', mean_log_var = 2)
            variance = log_var.exp()
            out = stack((mean, variance))

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = tuple(slice_right_at_dim(t, self.max_mem_len, dim = -2).detach() for t in hiddens)
            return out, new_mems

        if return_attn:
            attn_maps = tuple(t.post_softmax_attn for t in intermediates.attn_intermediates)
            return out, attn_maps

        return out

class ContinuousAutoregressiveWrapper(Module):
    """Autoregressive training / generation over continuous sequences.

    Wraps a `ContinuousTransformerWrapper` to: predict each next continuous value
    from the previous ones, generate sequences one step at a time, and optionally
    run multi-step rollouts. With a probabilistic net it uses a Gaussian NLL loss
    and samples from the predicted gaussian; otherwise it regresses with L1/L2.
    """

    def __init__(
        self,
        net: ContinuousTransformerWrapper,
        loss_fn: Module | None = None,
        use_l1_loss = False,
        equal_loss_weight_batch = False,  # setting this to True, if the mask is passed in and sequences are variable in length, each sequence will be weighted the same (as opposed to each token)
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len

        probabilistic = net.probabilistic
        self.probabilistic = probabilistic

        # default loss function
        # gaussian NLL for probabilistic heads, else L1 / L2 regression

        if not exists(loss_fn):
            if probabilistic:
                loss_fn = GaussianNLL()
            elif use_l1_loss:
                loss_fn = nn.L1Loss(reduction = 'none')
            else:
                loss_fn = nn.MSELoss(reduction = 'none')

        self.loss_fn = loss_fn
        self.equal_loss_weight_batch = equal_loss_weight_batch

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        temperature = 1.,
        cache_kv = True,
        **kwargs
    ):
        """Autoregressively generate `seq_len` continuous steps following `start_tokens`."""
        should_cache_kv = cache_kv and self.net.can_cache_kv
        device = start_tokens.device

        was_training = self.net.training
        num_dims = start_tokens.ndim

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'
        no_batch = num_dims == 2

        # allow unbatched (n d) input by adding a temporary batch dim
        if no_batch:
            start_tokens = rearrange(start_tokens, 'n d -> 1 n d')

        b, t, _, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens

        cache = None

        for _ in range(seq_len):
            x = out
            # crop context to the model's max sequence length
            if exists(self.max_seq_len):
                x = x[:, -self.max_seq_len:]

            net_out, new_cache = self.net(x, cache = cache, return_intermediates = True, **kwargs)

            # keep only the prediction for the final position
            last_output = net_out[..., -1:, :]

            # for a gaussian head, sample the next value from the predicted (mean, variance)
            if self.probabilistic:
                mean, var = last_output
                last_output = sample_from_mean_variance(mean, var, temperature = temperature)

            out = cat((out, last_output), dim = -2)

            if should_cache_kv:
                cache = new_cache

        # drop the original prompt, returning only the newly generated steps
        out = out[:, t:]

        if no_batch:
            out = rearrange(out, '1 n d -> n d')

        self.net.train(was_training)
        return out

    def forward_rollout(
        self,
        x,
        rollout_steps = 2,
        **kwargs
    ):
        """Multi-step rollout training: feed the model's own predictions back in for `rollout_steps`.

        Picks a random start offset per sample, autoregressively predicts that many
        steps (sampling when probabilistic), and scores all predictions against the
        true future. Encourages stability over multiple steps (world-model style).
        """
        assert rollout_steps > 1

        steps = rollout_steps

        device = x.device

        # assert inputs

        assert 'prepend_embeds' not in kwargs

        # lens
        # derive a mask from per-sample lengths if provided

        lens = kwargs.pop('lens', None)

        if exists(lens):
            assert 'mask' not in kwargs, 'either `mask` or `lens` passed in, but not both'
            seq_len, device = inp.shape[1], inp.device
            seq_arange = arange(seq_len, device = device)
            mask = einx.less('j, i -> i j', seq_arange, lens)
            kwargs['mask'] = mask

        # without lens, assume every sample uses its full sequence length
        if not exists(lens):
            batch, seq_len = x.shape[:2]
            lens = torch.full((batch,), seq_len, device = device)

        # handle mask manually

        mask = kwargs.pop('mask', None)

        # pick a random range for each batch sample and aligh the sequence to the right for rollout loss

        # only samples with enough tokens to roll out `steps` ahead are usable
        valid_tokens_for_rollout = (lens - steps).clamp(min = 0)
        valid_sample = valid_tokens_for_rollout > 0

        x = x[valid_sample] # remove invalid sequence (lens less than rollout steps)

        if exists(mask):
            mask = mask[valid_sample]

        batch = x.shape[0]
        # random start offset per sample (uniformly within its valid range)
        seq_start_pos = (torch.rand((batch,), device = device) * valid_tokens_for_rollout).floor().long()

        batch_arange = torch.arange(batch, device = device)
        batch_arange = rearrange(batch_arange, 'b -> b 1')

        # crop out sequence to use

        # right-align so the chosen windows end at the same position across the batch
        seq_end_pos = seq_start_pos + steps
        max_end_pos = seq_end_pos.amax().item()
        x = x[:, :max_end_pos]

        x = align_right(x, seq_end_pos)

        # get the input
        # last `steps` positions are the targets to predict

        inp, targets = x[:, :-steps], x[:, -steps:]

        # maybe rollout

        cache = None
        preds = []

        for _ in range(steps):

            out, cache = self.net(
                inp,
                seq_start_pos = seq_start_pos,
                return_intermediates = True,
                **kwargs
            )

            last_pred = out[..., -1:, :]

            # feed prediction back in as next input (sample when probabilistic)
            if self.probabilistic:
                mean, var = last_pred
                inp = sample_from_mean_variance(mean, var)
            else:
                inp = last_pred

            preds.append(last_pred)

        # stack for predictions

        preds = cat(preds, dim = 1)

        # loss
        # score the full rollout against the true future

        loss = self.loss_fn(preds, targets)

        return loss.mean()

    def forward(
        self,
        x,
        rollout_steps = 1, # they used 2 rollout steps in a successful world model paper https://ai.meta.com/vjepa/
        **kwargs
    ):
        """Single-step AR training: predict each next continuous value from the previous ones."""
        # delegate to multi-step rollout when requested
        if rollout_steps > 1:
            return self.forward_rollout(x, rollout_steps = rollout_steps, **kwargs)

        # standard next-step shift: inputs are all-but-last, targets are all-but-first
        inp, target = x[:, :-1], x[:, 1:]

        assert 'prepend_embeds' not in kwargs

        # lens
        # build a mask from per-sample lengths if given

        lens = kwargs.pop('lens', None)

        if exists(lens):
            assert 'mask' not in kwargs, 'either `mask` or `lens` passed in, but not both'
            seq_len, device = inp.shape[1], inp.device
            seq_arange = torch.arange(seq_len, device = device)
            mask = einx.less('j, i -> i j', seq_arange, lens)

            kwargs['mask'] = mask

        # mask

        # trim the mask to match the shifted input length
        mask = kwargs.get('mask', None)

        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(inp, **kwargs)

        loss = self.loss_fn(out, target)

        if exists(mask):
            assert loss.ndim > 1, 'loss should not be reduced if mask is passed in'

            # equal_loss_weight_batch: average per-sequence first so each sample counts equally;
            # otherwise just gather the unmasked positions
            if self.equal_loss_weight_batch:
                loss = batch_masked_mean(loss, mask)
            else:
                loss = loss[mask]

        return loss.mean()
