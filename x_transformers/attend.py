from __future__ import annotations

from functools import partial
from typing import Tuple, Callable

import torch
from torch.nn import Module, Parameter
from torch import cat, nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat, pack, unpack

# constants

@dataclass
class Intermediates:
    """
    Container for per-layer attention intermediates, returned alongside the
    attention output. Lets downstream code inspect / cache attention internals
    (for analysis, residual attention, kv caching, hybrid layers, etc).

    Fields:
        qk_similarities:   raw query-key similarity scores (post scale, pre mask/softmax).
        pre_softmax_attn:  attention logits right before the attn_fn (post mask, sink, cope...).
        post_softmax_attn: attention weights right after the attn_fn (the actual mixing weights).
        values:            the value tensor used in the weighted sum (when stashed).
        cached_kv:         (key, value) pair stashed for incremental kv-cache decoding.
                           NOTE: declared here, but actually produced and consumed in
                           Attention.forward (x_transformers.py) — Attend never sets it.
        layer_type:        single-char tag for the layer ('a' = attention) used by cache logic.
        hybrid_hidden:     hidden state from a hybrid (e.g. recurrent/conv) branch, if present.
        log_probs:         log probs for hard (gumbel) attention choices; populated by return_gumbel_log_probs.
    """
    qk_similarities:    Tensor | None = None
    pre_softmax_attn:   Tensor | None = None
    post_softmax_attn:  Tensor | None = None
    values:             Tensor | None = None
    cached_kv:          tuple[Tensor, Tensor] | None = None
    layer_type:         str | None = None
    hybrid_hidden:      Tensor | None = None
    log_probs:          Tensor | None = None  # log probs for hard (gumbel) attention choices; populated by return_gumbel_log_probs

    def to_tuple(self):
        # legacy 3-tuple view (qk sims, pre-softmax, post-softmax) for older callers
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def default_device(device = None):
    if exists(device):
        return torch.device(device)

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')

    if torch.cuda.is_available():
        return torch.device('cuda')

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1

def compact(arr):
    return [*filter(exists, arr)]

# l1 normalize - inverted attention softmaxes over the query axis, so the resulting
# rows no longer sum to 1 across keys and have to be renormalized before mixing values

def l1norm(t, dim = -1, eps = 1e-8):
    return F.normalize(t, p = 1, dim = dim, eps = eps)

@torch.jit.script
def softclamp(t: Tensor, value: float):
    return (t / value).tanh() * value

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# gumbel softmax attention related

def log_prob_from_hard_attend(intermeds: Intermediates) -> Tensor:
    """
    Compute per-position log probabilities for hard (argmax/gumbel) attention choices.

    Given an `Intermediates` object from a hard-attention forward pass, uses
    `pre_softmax_attn` (raw logits) and `post_softmax_attn` (hard one-hot weights)
    to extract the log probability of each chosen key position.

    Args:
        intermeds: Intermediates dataclass containing:
            - pre_softmax_attn:  (b h i j) raw attention logits
            - post_softmax_attn: (b h i j) hard one-hot attention weights (from argmax or gumbel straight-through)

    Returns:
        Tensor of shape (b h i) — log prob of the chosen key for each query position.
    """
    assert intermeds.pre_softmax_attn is not None and intermeds.post_softmax_attn is not None, \
        'pre_softmax_attn and post_softmax_attn must be present for log prob computation'
    log_probs = intermeds.pre_softmax_attn.log_softmax(dim = -1)

    one_hot = intermeds.post_softmax_attn.argmax(dim = -1, keepdim = True)
    log_prob = log_probs.gather(-1, one_hot)
    return rearrange(log_prob, 'b h i 1 -> b h i')

# selective attention
# https://arxiv.org/abs/2410.02703 - section 3.3
# it is a technique to allow each token to prevent itself from being attended to by future tokens
# if sim_head_gate not supplied, will use the first head of the attention logits (sim in this framework)

def selective_attn(
    sim,
    sim_head_gate = None,
    no_mask_sos = True
):
    i, j, device = *sim.shape[-2:], sim.device
    sim_head_gate = default(sim_head_gate, sim[:, 0])

    gate = F.relu(sim_head_gate) # only positive

    if no_mask_sos:
        gate = gate.clone()
        gate[..., -i] = 0.

    eye = torch.eye(i, device = device)

    if j > i:
        eye = F.pad(eye, (j - i, 0), value = 1.)

    gate = (1. - eye) * gate
    gate = F.pad(gate, (0, 0, 1, -1), value = 0.) # only allow for masking the future
    gate = gate.cumsum(dim = -2)

    return sim - rearrange(gate, 'b i j -> b 1 i j')

# alternative distance functions

def qk_l2_dist_squared(q, k):
    if k.ndim == 3:
        k = repeat(k, 'b j d -> b h j d', h = q.shape[1])

    q, packed_shape = pack_one(q, '* i d')
    k, _ = pack_one(k, '* j d')

    l2_dist_squared = torch.cdist(q, k) ** 2
    return unpack_one(l2_dist_squared, packed_shape, '* i j')

# one-hot straight through softmax
# `dim` picks the competing axis - -1 (keys) for ordinary attention, -2 (queries) when inverted

def one_hot_straight_through(logits, temperature = 1., dim = -1):
    one_hot_indices = logits.argmax(dim = dim, keepdim = True)
    one_hot = torch.zeros_like(logits).scatter(dim, one_hot_indices, 1.)

    soft_attn = (logits / temperature).softmax(dim = dim)
    return one_hot + soft_attn - soft_attn.detach()

# sparse topk attention - only keep topk attn logits for softmax
# optional straight through with masked out logits by setting `attn_sparse_topk_straight_through = True`

def sparse_topk_attn(
    logits,
    sparse_topk,
    temperature = 1.,
    straight_through = False,
    dim = -1                    # competing axis: -1 (keys) normally, -2 (queries) when inverted
):
    orig_logits = logits

    mask_value = -torch.finfo(logits.dtype).max
    top_values, _ = logits.topk(sparse_topk, dim = dim)

    # build a mask dynamically across the specific dim
    sparse_topk_mask = (logits >= top_values.select(dim, -1).unsqueeze(dim)) & (logits > mask_value)

    logits = logits.masked_fill(~sparse_topk_mask, mask_value)
    topk_attn = logits.softmax(dim = dim)

    if not straight_through:
        return topk_attn

    soft_attn = (orig_logits / temperature).softmax(dim = dim)
    return topk_attn.detach() + soft_attn - soft_attn.detach()

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# main class

class Attend(Module):
    """
    The core attention compute, decoupled from projections / heads (those live in
    Attention in x_transformers.py). Given q, k, v it returns the attended output
    plus an Intermediates bundle.

    Two code paths:
      - flash (`flash_attn`): delegates to F.scaled_dot_product_attention under an
        sdp backend context manager. Fast, but only supports a limited feature set
        (no sigmoid/hard/topk/cog/selective/talking-heads/softclamp/residual attn).
      - manual (`forward` fallback): explicit einsum + softmax pipeline that supports
        all the bells and whistles (talking heads, softclamp, cope, selective attn,
        learned sink, alternative attn_fns, residual/delta attention, cog signs, ...).

    The constructor wires up which features are active and asserts the
    incompatible-with-flash combinations up front.
    """

    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        sparse_topk = None,
        sparse_topk_straight_through = False, # https://arxiv.org/abs/2505.22074
        scale = None,
        qk_norm = False,
        l2_distance = False,
        sigmoid = False,
        gumbel_softmax = False,
        gumbel_softmax_temp = 1.,
        gumbel_softmax_hard = True,
        cog_signed = False,
        custom_attn_fn: Callable | None = None,
        flash = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        add_zero_kv = False,
        head_learned_sink = False,
        selective = False,
        hard = False,
        cope = None,
        onnxable = False,
        inverted_attention = False,      # normalize across queries instead of keys, slot-attention style - https://openreview.net/forum?id=WgQZNoQ5AB
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        ),
        flash_pack_seq = False,          # efficient flash attention packed sequence masking for variable length sequences.
    ):
        super().__init__()
        self.scale = scale

        # causal related

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        # attention type

        is_sparse_topk_attn = exists(sparse_topk)

        assert not (flash and sigmoid), 'sigmoid attention not available for flash'
        assert not (flash and hard), 'hard attention not available for flash'
        assert not (flash and is_sparse_topk_attn), 'topk attention not available for flash'
        assert not (flash and inverted_attention), 'flash attention not compatible with inverted attention'

        assert at_most_one_of(sigmoid, hard, l2_distance, gumbel_softmax, is_sparse_topk_attn)

        # inverted attention flips which axis the attention function normalizes over.
        # ordinary attention normalizes over j (keys), so each query spreads a unit of mass over keys.
        # inverted normalizes over i (queries), so queries compete for each key - the slot attention
        # arrangement. the axis-taking attn_fns below get bound to it (sigmoid is elementwise and a
        # custom_attn_fn is left alone); forward() renormalizes across keys afterwards.

        self.inverted_attention = inverted_attention
        attn_dim = -2 if inverted_attention else -1

        if exists(custom_attn_fn):
            self.attn_fn = custom_attn_fn
        elif sigmoid:
            self.attn_fn = F.sigmoid
        elif hard:
            self.attn_fn = partial(one_hot_straight_through, dim = attn_dim)
        elif is_sparse_topk_attn:
            self.attn_fn = partial(sparse_topk_attn, sparse_topk = sparse_topk, straight_through = sparse_topk_straight_through, dim = attn_dim)
        elif gumbel_softmax:
            self.attn_fn = partial(F.gumbel_softmax, dim = attn_dim, tau = gumbel_softmax_temp, hard = gumbel_softmax_hard)
        else:
            softmax_fn = partial(F.softmax, dim = attn_dim)
            self.attn_fn = partial(softmax_fn, dtype = torch.float32) if not qk_norm else softmax_fn

        # dropouts

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and (pre_talking_heads or post_talking_heads or pre_scale_post_talking_heads)), 'talking heads not compatible with flash attention'

        self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_talking_heads else None
        self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if post_talking_heads else None
        self.pre_scale_post_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_scale_post_talking_heads else None

        if exists(self.pre_softmax_talking_heads):
            nn.init.dirac_(self.pre_softmax_talking_heads.weight)

        if exists(self.post_softmax_talking_heads):
            nn.init.dirac_(self.post_softmax_talking_heads.weight)

        if exists(self.pre_scale_post_talking_heads):
            # an improvisation where heads are combined pre-softmax attention, then used to scale post-softmax attention
            nn.init.dirac_(self.pre_scale_post_talking_heads.weight)

        # selective attention

        assert not (flash and selective), 'selective attention cannot work on flash attention'
        assert not (selective and not causal), 'selective attention is designed for autoregressive'
        self.selective = selective

        # cog attention - negative weights for expressiveness
        # https://openreview.net/forum?id=ezRrwwbxd0

        assert not (flash and cog_signed), 'cog attention not available for flash'
        self.cog_signed = cog_signed

        # l2 distance attention

        self.l2_distance = l2_distance

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # learned sink concatted pre-softmax, working solution from gpt-oss

        assert not (head_learned_sink and flash), f'not supported for flash attention yet'

        self.head_learned_sink = head_learned_sink
        self.head_attn_sink = Parameter(torch.zeros(heads)) if head_learned_sink else None

        # soft clamp attention logit value

        if softclamp_logits:
            assert not flash, 'flash attention not compatible with logit softclamp value yet'
            assert logit_softclamp_value > 0.

        self.softclamp_logits = softclamp_logits
        self.logit_softclamp_value = logit_softclamp_value

        # contextual positional encoding

        self.cope = cope

        # flash attention

        self.flash = flash
        self.flash_pack_seq = flash_pack_seq

        torch_version = version.parse(torch.__version__)
        assert not (flash and torch_version < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        if self.flash:
            if self.flash_pack_seq:
                try:
                    from flash_attn import flash_attn_varlen_func
                    self.flash_attn_varlen_func = flash_attn_varlen_func
                except ImportError:
                    raise ImportError("block masking with Flash Attention requires the flash-attn package. Please install it with `pip install flash-attn`.")

            # torch 2.3 uses new backend and context manager
            if torch_version >= version.parse('2.3'):
                from torch.nn.attention import SDPBackend

                str_to_backend = dict(
                    enable_flash = SDPBackend.FLASH_ATTENTION,
                    enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
                    enable_math = SDPBackend.MATH,
                    enable_cudnn = SDPBackend.CUDNN_ATTENTION
                )

                sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in sdp_kwargs.items() if enable]

                self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
            else:
                self.sdp_context_manager = partial(torch.backends.cuda.sdp_kernel, **sdp_kwargs)

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        flash_pack_seq_kwargs = None, # https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py#L1370
        causal = None,
    ):
        """
        Flash attention path: dispatch to a fused SDPA kernel.

        Handles the bookkeeping that the manual path does inline: expanding
        multi-query k/v to full heads, folding the l2-distance trick into q/k,
        rescaling for cosine-sim attention, building/merging masks (key padding +
        causal), turning an alibi bias into an additive mask, and zeroing rows that
        end up fully masked. Either calls the variable-length packed flash kernel
        (`flash_attn_varlen_func`) or `F.scaled_dot_product_attention` under the sdp
        backend context manager. Returns (out, empty Intermediates).
        """
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # handle maybe l2 distance

        if self.l2_distance:
            k_norm_sq = k.norm(dim = -1, keepdim = True) ** 2
            k = F.pad(k, (0, 1), value = -1.)
            k = cat((k, k_norm_sq), dim = -1)

            q_norm_sq = q.norm(dim = -1, keepdim = True) ** 2
            q = cat((2 * q, q_norm_sq), dim = -1)
            q = F.pad(q, (0, 1), value = -1.)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        causal = default(causal, self.causal)

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask
        # Check if mask exists and expand to compatible shape
        # the mask arrives 4d already - forward() broadcasts a B L key padding mask up to B 1 1 L -
        # and gets materialized here to the full B H N L that scaled_dot_product_attention expects

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask
            causal = False

        # protect against an entire row being masked out

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = attn_bias.expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        if self.flash_pack_seq:
            # variable length path - the sequences are already concatenated into a single batch row,
            # and their boundaries are carried by the cu_seqlens_q / cu_seqlens_k (+ max_seqlen_*)
            # entries of flash_pack_seq_kwargs, which are forwarded untouched to the kernel below.
            # that is why a dense mask is rejected and batch must be 1. upstream also dropped the
            # older, stricter shape/monotonicity validation of the cu_seqlens tensors from here

            assert exists(flash_pack_seq_kwargs), 'flash_pack_seq_kwargs must be provided'
            assert not exists(mask), 'mask cannot be passed for block masking'
            assert q.shape[0] == k.shape[0] == v.shape[0] == 1, 'batch size must be 1 for sequence packing'
            assert device.type == 'cuda', 'flash-attn packed sequence masking is only available on CUDA devices'

            major, minor = torch.cuda.get_device_capability(device)
            assert major >= 8, f"block masking with Flash Attention requires SM80+ (Ampere or newer) GPUs, but your GPU has SM{major}{minor}."

            dropout_p = self.dropout if self.training else 0.

            # drop the singleton batch dim - the varlen kernel wants a flat (total_tokens, heads, dim)

            q, k, v = tuple(rearrange(t, '1 h n d -> n h d') for t in (q, k, v))

            # efficient packed sequences flash attention masking

            out = self.flash_attn_varlen_func(
                q, k, v,
                causal = causal,
                dropout_p = dropout_p,
                **flash_pack_seq_kwargs
            )

            out = rearrange(out, 'n h d -> 1 h n d')
        else:
            with self.sdp_context_manager():
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask = mask,
                    dropout_p = self.dropout if self.training else 0.,
                    is_causal = causal
                )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None,
        attn_delta = None,
        row_mask = None,             # query side padding mask (b i) - only consulted under inverted attention, where the pre-softmax masking has to happen on the query axis
        flash_pack_seq_kwargs = None,
        causal = None,
    ):
        """
        Main entry point. Routes to the flash path when possible, otherwise runs
        the manual attention pipeline below (the part that supports every feature).

        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = default(causal, self.causal)

        # handle key padding mask

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = tuple(rearrange(t, 'b 1 n d -> b n d') for t in (k, v))
        elif kv_heads < heads:
            k, v = tuple(repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads) for t in (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = tuple(F.pad(t, (0, 0, 1, 0), value = 0.) for t in (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        # take the flash path only when no per-element attn_delta is requested
        # (flash returns weights implicitly, so it can't have a delta added post-softmax)

        if self.flash and not exists(attn_delta):
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias, flash_pack_seq_kwargs = flash_pack_seq_kwargs)

        # einsum equation differs depending on whether k/v are shared across heads (3d) or per-head (4d)
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # similarity: standard dot-product, or negative squared l2 distance as an alternative metric
        if not self.l2_distance:
            sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
        else:
            sim = -qk_l2_dist_squared(q, k)

        # scale logits (1/sqrt(d) by default, or custom for cosine-sim attention)
        sim = sim * scale

        # residual attention - add previous layer's pre-softmax logits (RealFormer style)
        if exists(prev_attn):
            sim = sim + prev_attn

        qk_similarities = sim.clone()

        # talking-heads: mix information across heads via 1x1 convs over the head dim
        # pre_scale variant: compute a per-head scale now, apply it post-softmax
        if exists(self.pre_scale_post_talking_heads):
            pre_to_post_scale = self.pre_scale_post_talking_heads(sim)

        # pre-softmax talking heads: mix heads before the softmax
        if exists(self.pre_softmax_talking_heads):
            sim = sim + self.pre_softmax_talking_heads(sim)

        # additive positional bias (e.g. alibi / relative position)
        if exists(attn_bias):
            sim = sim + attn_bias

        # soft-clamp logits into (-value, value) via tanh, to bound outlier logits
        if self.softclamp_logits:
            sim = softclamp(sim, self.logit_softclamp_value)

        # pre-masking - handle cog by storing sign

        if self.cog_signed:
            sim_sign = sim.sign()
            sim = sim.abs()

        # masking - fill disallowed positions with a very negative value so they vanish after softmax

        i, j, dtype = *sim.shape[-2:], sim.dtype

        mask_value = -torch.finfo(sim.dtype).max

        # key padding / custom mask
        # inverted attention normalizes down the query axis, so masking keys here would be a no-op
        # for it - the query side row_mask is masked pre-softmax instead, and the key mask is applied
        # after the softmax further below

        attn_mask = mask
        if self.inverted_attention:
            attn_mask = rearrange(row_mask, 'b i -> b 1 i 1') if exists(row_mask) else None

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, mask_value)

        # causal mask - forbid attending to future positions
        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            sim = sim.masked_fill(causal_mask, mask_value)

        # track rows with nothing left to attend to, so their output can be zeroed later
        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # CoPE - contextual positional encoding, adds a content-conditioned position bias
        if exists(self.cope):
            sim = sim + self.cope(q, sim)

        # selective attention - let tokens suppress being attended to by future tokens
        if self.selective:
            sim = selective_attn(sim)

        if self.head_learned_sink:
            # learned attention sink: prepend a per-head learnable logit column, giving the
            # softmax an always-present "attend to nothing" option (absorbs leftover mass)
            attn_sink = repeat(self.head_attn_sink, 'h -> b h i 1', b = sim.shape[0], i = sim.shape[2])

            if self.cog_signed:
                attn_sink, attn_sink_sign = attn_sink.abs(), attn_sink.sign()
                sim_sign = cat((attn_sink_sign, sim_sign), dim = -1)

            sim = cat((attn_sink, sim), dim = -1)

        pre_softmax_attn = sim

        # turn logits into attention weights. attn_fn was selected in __init__ and is one of:
        # softmax (default) / sigmoid / hard one-hot straight-through / sparse-topk /
        # gumbel-softmax / custom — see the constructor.
        attn = self.attn_fn(sim)

        # inverted attention - the weights were normalized down the query axis, so zero out the
        # padded keys and then l1 normalize each query row back to summing to 1, making the value
        # aggregation a weighted mean again (slot attention's normalization step)

        if self.inverted_attention:
            if exists(mask):
                attn = attn.masked_fill(~mask, 0.)

            attn = l1norm(attn)

        attn = attn.type(dtype)

        # post-softmax additive correction to the attention weights
        # https://openreview.net/forum?id=uomCTwGflg

        if exists(attn_delta):
            attn = attn + attn_delta

        # cog attention: restore the signs stripped before softmax (allows negative weights)

        if self.cog_signed:
            attn = attn * sim_sign

        post_softmax_attn = attn

        if self.head_learned_sink:
            # drop the sink column so it doesn't contribute to the value-weighted sum
            attn = attn[..., 1:]

        # attention dropout
        attn = self.attn_dropout(attn)

        # post-softmax talking heads: mix the attention weights across heads
        if exists(self.post_softmax_talking_heads):
            attn = self.post_softmax_talking_heads(attn)

        # apply the per-head scale computed pre-softmax (pre_scale_post variant)
        if exists(self.pre_scale_post_talking_heads):
            attn = attn * pre_to_post_scale

        # weighted sum of values by attention weights -> attended output
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, intermediates
