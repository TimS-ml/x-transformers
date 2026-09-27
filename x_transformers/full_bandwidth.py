"""
Full-bandwidth transformer - latent feedback from each position into the next,
trained with temporal parallel passes.

Xi Wang et al. - https://arxiv.org/abs/2608.08888
(title, per the bibtex entry in this repo's README: "Full-bandwidth transformer")

In a plain decoder, attention at a given block only reads earlier positions'
states from that same block, so a state can climb at most one block per serial
step, and what the top of the stack computed at position t - 1 never reaches the
bottom of the stack at position t. The serial computation available for carrying
a running state is bounded by depth, which is why a task like running parity
(train_full_bandwidth_parity.py) fails to length generalize. The full-bandwidth
scheme feeds hidden states back down: the output of some block at position t - 1
is fused into the input of some (usually lower) block at position t, turning the
stack into a recurrence over positions.

Wiring (block depths are 1-indexed; a block is one repetition of the decoder's
layer pattern, attention + feedforward by default):

- `recirc_pairs` lists routes (out_d, in_d): the residual stream leaving block
  out_d at position t - 1 is fused into the stream entering block in_d at
  position t by a small learned `transition(latents, stream)` module (glu,
  residual, gru, linear, residual_glu, or a custom one). The default
  ((depth, 1),), shortcut 'paper', feeds the final (normed) hidden back in place
  of the token embedding. Position 0 has no predecessor and is never fused.
- Latents are read from `LayerIntermediates.block_hiddens` of the previous
  forward. The fused streams go back in through the `transform_block_inputs`
  hook: TransformerWrapper.forward applies entry 1 to the token embeddings
  (before the absolute position embedding is added and before `post_emb_norm`),
  AttentionLayers.forward applies entries > 1 at the start of the matching block.

Training / scoring - temporal parallel passes:

The exact recurrence is sequential over positions. `forward` instead runs the
whole sequence `temporal_parallel_passes` times: pass 1 is an ordinary forward,
and pass p fuses the latents of pass p - 1, shifted right by one position. This
is a Jacobi style fixed point iteration - pass p reproduces the sequential
recurrence exactly on the first p positions (position 0 is the same in every
pass), a length n sequence is at the fixed point after n passes, and with fewer
passes the later positions see the recurrence truncated to p - 1 feedback steps.
Nothing on the feedback path is detached, so the loss of a later pass
backpropagates through the transitions into every earlier pass; all passes share
the wrapped network's weights. In training the fed-back latents get small
uniform jitter.

Generation:

`generate` prefills the prompt with the same passes, then decodes with the kv
cache, running each new token through the network once with the previous step's
latents fused in. That is the sequential recurrence itself (greedy output matches
the n-pass fixed point when the prompt is no longer than the number of passes),
so at inference the feedback chain is unbounded, while training unrolls it only
`temporal_parallel_passes - 1` steps. Prompt positions beyond the first
`temporal_parallel_passes` are approximate.

Loss:

    loss = L_1 + feedback_pass_weight * F

L_1 is the next token cross entropy of pass 1 over all positions, L_p (p >= 2)
that of pass p over positions 1: only, and F is the plain mean of L_2 .. L_P or,
with `dynamic_rollout_loss_weight`, a weighted average whose weights decay with
the accumulated excess loss of the preceding passes (see
`dynamic_rollout_loss_weights` and the notes in `FullBandwidth.forward`).
"""

from __future__ import annotations

# Full-bandwidth Transformer - Xi Wang et al. https://arxiv.org/abs/2608.08888

from collections import namedtuple
from typing import Callable

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleDict
import torch.nn.functional as F

from einops import rearrange
from torch_einops_utils import (
    maybe_return,
    temp_eval,
    pack_with_inverse,
    cast_tensor,
    exclusive_cumsum
)

from x_transformers.x_transformers import (
    Decoder,
    TransformerWrapper,
    LinearNoBias,
    RMSNorm,
    Identity,
    LayerIntermediates,
    exists,
    first
)

from x_transformers.autoregressive_wrapper import top_k, FILTER_LOGITS_FN

# helper functions

def default(val, d):
    return val if exists(val) else d

def uniform_like(t, low = -1., high = 1.):
    # uniform noise in [low, high) with t's shape, dtype and device
    return torch.empty_like(t).uniform_(low, high)

# shift and fuse latent feedback into stream

def shift_and_fuse_latents(latents, stream, transition):
    """Fuse every position's stream with the previous position's latent (training and prefill path).

    latents: (b, n, d) - output of the source block from the previous pass
    stream:  (b, n, d) - token embeddings or residual stream entering the destination block, current pass

    Position t >= 1 becomes transition(latents[t - 1], stream[t]); position 0 has
    no predecessor and passes through unchanged. Returns (b, n, d). A sequence of
    length <= 1 has nothing to fuse and is returned as is.
    """
    if latents.shape[-2] <= 1:
        return stream

    # latents at 0 .. n-2 pair with stream at 1 .. n-1 -> (b, n - 1, d), then position 0 is put back in front

    fused = transition(latents[:, :-1], stream[:, 1:])
    return torch.cat((stream[:, :1], fused), dim = -2)

# fuse last latent during autoregressive generation

def fuse_last_latent(latents, stream, transition):
    """Generation path - fuse only the newest position.

    latents come from the previous decoding step, so latents[:, -1] belongs to the
    previous position. In a cached forward the stream is (b, 1, d) for in_d > 1
    (only the new token is run), but the full (b, n, d) token embeddings for
    in_d = 1 (the wrapper embeds the whole sequence before the attention layers
    slice it down to the new token). Either way only the last position is replaced.
    """
    fused = transition(latents[:, -1:], stream[:, -1:])
    return torch.cat((stream[:, :-1], fused), dim = -2)

# dynamic rollout loss weighting
# weight each rollout step by exp(-decay * cumulative loss of preceding steps), as errors compound through the dynamics model
# threshold accounts for the irreducible entropy floor of language (~1.0 nat), penalizing only excess modeling error

def dynamic_rollout_loss_weights(
    step_losses,
    decay = 1.,
    threshold = 1.
):
    """Per-pass weights from the per-pass losses:

        e_j = max(L_j - threshold, 0)
        w_k = exp(-decay * sum_{j < k} e_j)

    step_losses: (p,) losses in pass order, pass 1 first. FullBandwidth.forward
    passes them detached, so the weights are constants. Returns (p,) weights with
    w_1 = 1 (exclusive cumsum), every later pass discounted by how far the passes
    before it exceeded `threshold`; losses at or below the threshold add nothing.
    """
    excess_losses = (step_losses - threshold).clamp(min = 0.)

    # exclusive cumsum (cumsum - t): entry k sums entries 0 .. k-1, so entry 0 is 0

    cum_step_losses = exclusive_cumsum(excess_losses, dim = 0)
    return (-decay * cum_step_losses).exp()

# loss breakdown
# all entries unweighted, i.e. before feedback_pass_weight and any dynamic weighting
#   first_pass_loss      - scalar ce of pass 1 over all positions
#   feedback_pass_losses - list of temporal_parallel_passes - 1 scalar ces, one per feedback pass, positions 1: only
#   all_pass_losses      - [first_pass_loss, *feedback_pass_losses]

LossBreakdown = namedtuple('LossBreakdown', [
    'first_pass_loss',
    'feedback_pass_losses',
    'all_pass_losses'
])

# transitions
# every transition maps (latents, stream) -> fused stream, all (b, n, d):
#   latents - source block output from the previous pass (training) or previous decoding step (generation)
#   stream  - token embeddings (in_d = 1) or the residual stream entering block in_d
# and ends in an RMSNorm (F.normalize * sqrt(d) * learned gain), so fused positions come out at unit rms
# times the gain, whatever the scale of the stream going in
# note: for in_d > 1 this re-normalizes the residual stream at that block boundary on positions 1:, while
# position 0 (never fused) keeps its raw scale

# 1. GLU cross transition from Full-bandwidth Transformer (Eq. 4)

class GLUCrossTransition(Module):
    """out = RMSNorm(W_s RMSNorm(latents) * sigmoid(W_g RMSNorm(stream)))

    The content comes from the latent; the stream only decides, per feature, how
    much of it gets through. The stream's own values are not carried over, so at
    in_d = 1 a fused position's token embedding is replaced (token identity enters
    only through the gate), and at in_d > 1 the residual stream entering that
    block is replaced.
    """

    def __init__(self, dim):
        super().__init__()
        self.to_state = LinearNoBias(dim, dim)
        self.to_gate = LinearNoBias(dim, dim)
        self.latent_norm = RMSNorm(dim)
        self.stream_norm = RMSNorm(dim)
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        state = self.to_state(self.latent_norm(latents))
        gate = self.to_gate(self.stream_norm(stream)).sigmoid()
        return self.norm(state * gate)

# 2. Residual transition

class ResidualTransition(Module):
    """out = RMSNorm(stream + scale * W RMSNorm(latents)), `scale` a learned scalar (init 0.1)

    Keeps the stream additively, with the latent entering as a small learned-scale
    correction, so at init a fused position is close to a renormalized copy of the
    unfused one.
    """

    def __init__(self, dim, scale_init = 0.1):
        super().__init__()
        self.proj = LinearNoBias(dim, dim)
        self.latent_norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        residual = self.proj(self.latent_norm(latents)) * self.scale
        return self.norm(stream + residual)

# 3. GRU gating transition

class GRUGatingTransition(Module):
    """GRU style convex gate between the stream and a candidate computed from both.

        h   = [RMSNorm(latents), RMSNorm(stream)]        (b, n, 2d)
        z   = sigmoid(W_z h + b_z)
        c   = tanh(W_c h)
        out = RMSNorm((1 - z) * stream + z * c)

    Note the raw, not the normalized, stream is what gets interpolated.
    """

    def __init__(self, dim):
        super().__init__()
        self.latent_norm = RMSNorm(dim)
        self.stream_norm = RMSNorm(dim)
        self.gate = nn.Linear(dim * 2, dim, bias = True)
        self.to_candidate = LinearNoBias(dim * 2, dim)
        self.norm = RMSNorm(dim)

        # gate bias -1 -> z starts near sigmoid(-1) ~ 0.27, leaning towards keeping the incoming stream

        nn.init.constant_(self.gate.bias, -1.)

    def forward(self, latents, stream):
        packed = torch.cat((self.latent_norm(latents), self.stream_norm(stream)), dim = -1)
        z = self.gate(packed).sigmoid()
        candidate = self.to_candidate(packed).tanh()

        # lerp(stream, candidate, z) = stream + z * (candidate - stream)

        return self.norm(stream.lerp(candidate, z))

# 4. Normalized linear combination

class LinearTransition(Module):
    """out = RMSNorm(W_s RMSNorm(latents) + W_x RMSNorm(stream)) - ungated sum of both projections"""

    def __init__(self, dim):
        super().__init__()
        self.to_state = LinearNoBias(dim, dim)
        self.to_stream = LinearNoBias(dim, dim)
        self.latent_norm = RMSNorm(dim)
        self.stream_norm = RMSNorm(dim)
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        state = self.to_state(self.latent_norm(latents))
        stream_proj = self.to_stream(self.stream_norm(stream))
        return self.norm(state + stream_proj)

# 5. Residual GLU transition

class ResidualGLUTransition(Module):
    """out = RMSNorm(stream + scale * GLUCrossTransition(latents, stream)), `scale` learned (init 1.)

    The glu transition applied as a residual update, so the stream is kept rather
    than replaced.
    """

    def __init__(self, dim, scale_init = 1.0):
        super().__init__()
        self.glu = GLUCrossTransition(dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        return self.norm(stream + self.glu(latents, stream) * self.scale)

# registry of transition types
# the names accepted wherever a transition spec is a string (`transition`, `transitions`, third element of a pair)

TRANSITIONS = dict(
    glu = GLUCrossTransition,
    residual = ResidualTransition,
    gru = GRUGatingTransition,
    linear = LinearTransition,
    residual_glu = ResidualGLUTransition
)

def resolve_transition(spec, dim):
    """Turn a transition spec into a module of width `dim`.

    - Module instance - used as is (one instance given for several routes is shared between them)
    - str             - a TRANSITIONS key, instantiated as TRANSITIONS[spec](dim)
    - other callable  - a factory (e.g. a Module class), called as spec(dim)

    The instance check comes first because modules are callable too.
    """
    if isinstance(spec, Module):
        return spec

    if isinstance(spec, str):
        fn = TRANSITIONS.get(spec)
        assert exists(fn), f"unknown transition '{spec}', available: {list(TRANSITIONS.keys())}"
        return fn(dim)

    if callable(spec):
        return spec(dim)

    raise ValueError(f'invalid transition specification: {spec}')

# recirculation pattern shortcuts
# pairs are (out_d, in_d): block out_d's output at position t - 1 is fused into block in_d's input at position t
# depths 1-indexed, d = decoder depth; resolved for d = 3:
#   paper           ((3, 1),)                   top output replaces the token embedding, same as recirc_pairs = None
#   prev            ((2, 1), (3, 2))            each block feeds the block right below it
#   top_and_prev    ((3, 1), (2, 1), (3, 2))    paper + prev, de-duplicated in order (dict.fromkeys); for d = 2 both are (2, 1)
#   prev_with_self  ((1, 1), (2, 1), (3, 2))    prev, plus block 1 feeding its own input
#   self            ((1, 1), (2, 2), (3, 3))    each block feeds its own input
#   full            ((1, 1), (2, 1), (3, 1))    every block output fused into the token embedding
#   dense           all d * d pairs, source-major
# several sources routed into one in_d are fused one after another, in the order listed

RECIRC_SHORTCUTS = dict(
    paper = lambda d: ((d, 1),),
    prev = lambda d: tuple((i, i - 1) for i in range(2, d + 1)),
    top_and_prev = lambda d: tuple(dict.fromkeys(((d, 1), *[(i, i - 1) for i in range(2, d + 1)]))),
    prev_with_self = lambda d: tuple((i, max(i - 1, 1)) for i in range(1, d + 1)),
    self = lambda d: tuple((i, i) for i in range(1, d + 1)),
    full = lambda d: tuple((i, 1) for i in range(1, d + 1)),
    dense = lambda d: tuple((src, dst) for src in range(1, d + 1) for dst in range(1, d + 1)),
)

def resolve_recirc_pairs(depth: int, recirc_pairs: tuple | list | str | None):
    """Normalize `recirc_pairs` to a tuple of routes.

    None -> ((depth, 1),), the 'paper' route; a str -> that RECIRC_SHORTCUTS entry
    for this depth; anything else is taken as given, each item (out_d, in_d) or
    (out_d, in_d, transition_spec). Range checks happen in FullBandwidth.__init__.
    """
    if not exists(recirc_pairs):
        return ((depth, 1),)

    if isinstance(recirc_pairs, str):
        fn = RECIRC_SHORTCUTS.get(recirc_pairs)
        assert exists(fn), f"unknown recirc_pairs shortcut '{recirc_pairs}', available: {list(RECIRC_SHORTCUTS.keys())}"
        return fn(depth)

    return tuple(recirc_pairs)

# helper to extract block latent

def get_block_latent(intermediates: LayerIntermediates, depth: int) -> Tensor:
    """Output of block `depth` (1-indexed) from a forward's intermediates, (b, n, d).

    `block_hiddens[i]` is the residual stream right after the last layer of block
    i + 1. AttentionLayers overwrites the last entry with the final-normed output,
    so depth = decoder depth gives `last_hidden`, the tensor `to_logits` reads.
    Falls back to `last_hidden` when `block_hiddens` is missing or too short.

    note: with `layer_dropout` > 0 in training, a skipped block-final layer also
    skips that block's append in AttentionLayers.forward, so the list can come back
    shorter and misaligned with block depth, or empty - in which case every depth
    silently resolves to `last_hidden`.
    """
    block_hiddens = intermediates.block_hiddens

    if exists(block_hiddens) and 0 <= (depth - 1) < len(block_hiddens):
        return block_hiddens[depth - 1]

    assert exists(intermediates.last_hidden), f'cannot find latent for depth {depth}'
    return intermediates.last_hidden

# main class

class FullBandwidth(Module):
    """Wraps a TransformerWrapper decoder with full-bandwidth latent feedback.

    Requirements, asserted at init:
    - a Decoder - causality is what keeps position 0 identical across passes and
      makes the one-pass-per-token cached decoding in `generate` equal to the recurrence
    - depth scaled residuals (`depth_scale_residual`, or a residual scale set)
    - tied embeddings (`tie_embedding = True`, which makes `to_logits` a plain
      function rather than a module)
    - `post_emb_norm = True`, so fused token embeddings are layer normed (after the
      absolute position embedding is added) before entering the stack

    Holds one transition module per route in `self.transitions`, keyed 'out_d->in_d'.
    `forward` trains / scores with temporal parallel passes, `generate` samples with
    the per-token recurrence (see the module docstring).
    """

    def __init__(
        self,
        net: TransformerWrapper,                            # decoder to wrap; its weights are shared by every pass
        temporal_parallel_passes = 2,                       # default pass count for forward and generate; 1 = plain decoder, no feedback
        recirc_pairs: tuple | list | str | None = None,     # routes (out_d, in_d[, transition]) or a RECIRC_SHORTCUTS name; None = ((depth, 1),)
        transition: Module | str | Callable | None = None,  # default transition for every route; None = 'glu'
        transitions: dict | None = None,                    # per route overrides, keyed (out_d, in_d) or by in_d alone
        feedback_pass_weight = 1.,                          # multiplier on the combined feedback pass loss F
        ignore_index = -100,                                # target id excluded from every cross entropy
        jitter_noise_delta = 0.02,                          # paper used sigma = 0.02 - here the half-width of uniform noise on fed-back latents, training only; 0. disables
        dynamic_rollout_loss_weight = False,                # weight feedback passes by exp(-decay * cumulative excess loss of earlier passes) instead of a plain mean
        dynamic_loss_decay = 0.5,                           # decay in that exponent; larger discounts later passes faster
        dynamic_loss_threshold = 1.,                        # per pass loss (nats) up to which a pass adds no discount
    ):
        super().__init__()
        assert isinstance(net.attn_layers, Decoder), 'must be a decoder'
        assert net.attn_layers.depth_scale_residual or exists(net.attn_layers.residual_scale), 'depth scaling must be turned on'
        assert callable(net.to_logits) and not isinstance(net.to_logits, nn.Module), 'weight tying must be turned on'
        assert not isinstance(net.post_emb_norm, Identity), 'post embedding norm must be turned on'

        self.net = net
        self.temporal_parallel_passes = temporal_parallel_passes
        self.ignore_index = ignore_index
        self.jitter_noise_delta = jitter_noise_delta
        self.feedback_pass_weight = feedback_pass_weight
        self.dynamic_rollout_loss_weight = dynamic_rollout_loss_weight
        self.dynamic_loss_decay = dynamic_loss_decay
        self.dynamic_loss_threshold = dynamic_loss_threshold

        dim = net.attn_layers.dim
        depth = net.attn_layers.depth
        self.depth = depth

        raw_pairs = resolve_recirc_pairs(depth, recirc_pairs)

        parsed_pairs = []
        transitions_dict = dict()
        routes_by_in_depth = dict()

        # transition spec priority per route: third element of the pair, then transitions[(out_d, in_d)],
        # then transitions[in_d], then `transition`, then 'glu'

        for item in raw_pairs:
            out_d, in_d, *maybe_trans = item
            assert 1 <= out_d <= depth and 1 <= in_d <= depth, f'pair ({out_d}, {in_d}) out of range (1 to {depth})'

            trans_spec = first(maybe_trans)

            if not exists(trans_spec) and exists(transitions):
                trans_spec = transitions.get((out_d, in_d), transitions.get(in_d))

            trans_spec = default(trans_spec, default(transition, 'glu'))

            # note: a pair listed twice keeps only the last transition under its key, but is routed (so fused) twice

            parsed_pairs.append((out_d, in_d))
            transitions_dict[f'{out_d}->{in_d}'] = resolve_transition(trans_spec, dim)
            routes_by_in_depth.setdefault(in_d, []).append(out_d)

        # routes_by_in_depth: in_d -> [out_d, ...] in listed order, the order the fusions are chained in make_transforms

        self.transitions = ModuleDict(transitions_dict)
        self.recirc_pairs = tuple(parsed_pairs)
        self.routes_by_in_depth = routes_by_in_depth

        # device correct scalar stand-in for the feedback loss when there is only one pass

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    # transform block inputs helper

    def make_transforms(
        self,
        intermediates: LayerIntermediates,
        training = False,
        override_transition = None,
        generate = False
    ) -> dict[int, Callable]:
        """Build the `transform_block_inputs` dict for the next forward: {in_d: fn(stream) -> stream}.

        intermediates       - LayerIntermediates of the previous forward (previous pass, or previous
                              decoding step when `generate = True`), the source of the block latents
        training            - add uniform jitter in [-jitter_noise_delta, jitter_noise_delta] to the latents
        override_transition - one callable used for every route instead of the per route modules
        generate            - fuse only the last position (fuse_last_latent) instead of the whole
                              sequence shifted by one (shift_and_fuse_latents)

        Each fn fuses, one after another in the order the routes were given, every
        block latent routed into in_d. Latents are not detached, so the next pass's
        loss backpropagates into the pass that produced them.
        """
        fuse_fn = fuse_last_latent if generate else shift_and_fuse_latents
        transforms = dict()

        for in_d, out_depths in self.routes_by_in_depth.items():

            # in_d / out_depths bound as default args so each closure keeps its own loop values (python closures bind late)

            def transform(stream, in_d = in_d, out_depths = out_depths):
                for out_d in out_depths:
                    trans = default(override_transition, self.transitions[f'{out_d}->{in_d}'])
                    latent = get_block_latent(intermediates, out_d)

                    # note: uniform in [-delta, delta] has std delta / sqrt(3) (~0.0115 for the default 0.02),
                    # while the init comment cites the paper's sigma = 0.02

                    if training and self.jitter_noise_delta > 0.:
                        latent = latent + uniform_like(latent, -self.jitter_noise_delta, self.jitter_noise_delta)

                    stream = fuse_fn(latent, stream, trans)

                return stream

            transforms[in_d] = transform

        return transforms

    # generate

    @torch.no_grad()
    @temp_eval
    def generate(
        self,
        prompts: Tensor | list[Tensor],
        seq_len: int,
        temperature = 1.,
        filter_logits_fn: Callable | str = top_k,
        filter_kwargs: dict = dict(),
        should_fuse_latent: Callable | None = None,
        transition: Callable | None = None,
        temporal_parallel_passes: int | None = None,
        process_prompt = True,
        eos_token = None,
        **kwargs
    ):
        """Sample `seq_len` tokens with latent feedback. Runs in eval mode, without gradients.

        prompts                  - (b, n) or (n,) token ids; output is (b, seq_len) or (seq_len,) to match
        temperature              - 0. = greedy argmax (filter_logits_fn is then skipped)
        filter_logits_fn         - logits filter, or a FILTER_LOGITS_FN name, applied before the softmax
        should_fuse_latent       - step -> bool, whether the token sampled at decoding step `step`
                                   (0 = first new token) gets the previous position's latents fused in; default always
        transition               - one callable overriding every route's transition
        temporal_parallel_passes - passes for the prompt prefill; 1 also turns fusion off during decoding
        process_prompt           - False prefills the prompt with a single plain pass
        eos_token                - stop once every sequence in the batch samples it at the same step
                                   (an earlier eos in some rows is not masked)
        kwargs                   - forwarded to the wrapped network on every call

        The prefill runs the full prompt `temporal_parallel_passes` times without a
        cache, each pass fusing the previous one's latents, and keeps the kv cache of
        the last pass. Each decoding step then runs only the new token through the
        cached network with the previous step's (previous position's) latents fused
        in - one pass per token, which is the exact sequential recurrence. Returns
        only the generated tokens, fewer than `seq_len` if stopped at eos.

        note: the `list[Tensor]` prompt type goes through `cast_tensor`
        (torch.as_tensor), which rejects a list of multi-element tensors.
        """
        net = self.net
        assert net.can_cache_kv, 'FullBandwidth generation requires key/value caching'

        temporal_parallel_passes = default(temporal_parallel_passes, self.temporal_parallel_passes)
        should_fuse_latent = default(should_fuse_latent, lambda step: True)

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # (n,) -> (1, n), (b, n) unchanged; inverse_pack restores the leading shape at the end

        prompts = cast_tensor(prompts)
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')
        prompt_len = prompts.shape[-1]

        out = prompts

        # prefill, with optional temporal parallel passes

        logits, intermediates = net(out, return_intermediates = True, **kwargs)

        if process_prompt and temporal_parallel_passes > 1:
            for _ in range(temporal_parallel_passes - 1):
                transforms = self.make_transforms(intermediates, override_transition = transition)
                logits, intermediates = net(out, transform_block_inputs = transforms, return_intermediates = True, **kwargs)

        # intermediates of the last prefill pass double as the kv cache and the latent source for step 0

        cache = intermediates

        # autoregressive decoding, with latent feedback from preceding step

        for step in range(seq_len):

            # (b, n, l) -> (b, l), logits at the newest position

            logits = logits[:, -1]

            if temperature == 0.:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim = -1)

            if exists(eos_token) and (sample == eos_token).all():
                break

            # the last token is sampled - skip a forward whose logits would never be used

            if step == (seq_len - 1):
                break

            # fuse latents onto just sampled token
            # latents are the previous step's block outputs (position n - 2), fused into the new token (position n - 1)

            transforms = self.make_transforms(intermediates, override_transition = transition, generate = True) if (temporal_parallel_passes > 1 and should_fuse_latent(step)) else None

            logits, intermediates = net(
                out,
                transform_block_inputs = transforms,
                cache = cache,
                return_intermediates = True,
                **kwargs
            )

            # the new intermediates carry the extended kv cache and the new token's block latents for the next step

            cache = intermediates

        return inverse_pack(out[:, prompt_len:])

    # forward
    # maybe_return pops return_loss_breakdown / return_all_pass_logits and, when either is set, packs the
    # primary output ('loss' or 'logits', depending on return_loss) and the requested extras into a namedtuple

    @maybe_return(
        'loss_breakdown',
        'all_pass_logits',
        primary = lambda kwargs: 'loss' if kwargs.get('return_loss', True) else 'logits'
    )
    def forward(
        self,
        token_ids: Tensor,
        return_loss = True,
        temporal_parallel_passes: int | None = None,
        return_loss_breakdown = False,
        return_all_pass_logits = False,
        transition: Callable | None = None,
        dynamic_rollout_loss_weight: bool | None = None,
        dynamic_loss_decay: float | None = None,
        dynamic_loss_threshold: float | None = None,
        **kwargs
    ):
        """token_ids: (b, n).

        return_loss = True  - input token_ids[:, :-1], target token_ids[:, 1:]; returns the scalar loss
        return_loss = False - the whole sequence is the input; returns the last pass's logits (b, n, l)

        Extra outputs (via `maybe_return`, a namedtuple with the primary output first):
        `return_loss_breakdown` -> LossBreakdown (None without return_loss), and
        `return_all_pass_logits` -> list of temporal_parallel_passes logits, each
        (b, n - 1, l) with return_loss, (b, n, l) without.

        The remaining named args override the constructor defaults for this call;
        other kwargs go to the wrapped network on every pass.

        Pass 1 is a plain forward. Each of the temporal_parallel_passes - 1 feedback
        passes fuses the previous pass's block latents (jittered in training) into
        the routed block inputs one position to the right. Gradient flows through
        the whole chain of passes.
        """
        temporal_parallel_passes = default(temporal_parallel_passes, self.temporal_parallel_passes)
        assert temporal_parallel_passes >= 1, 'must have at least 1 pass'

        dynamic_rollout_loss_weight = default(dynamic_rollout_loss_weight, self.dynamic_rollout_loss_weight)
        dynamic_loss_decay = default(dynamic_loss_decay, self.dynamic_loss_decay)
        dynamic_loss_threshold = default(dynamic_loss_threshold, self.dynamic_loss_threshold)

        # cross entropy helper
        # (b n l) -> ((b n) l), mean over targets != ignore_index

        def ce_loss(logits, target):
            return F.cross_entropy(
                rearrange(logits, 'b n l -> (b n) l'),
                rearrange(target, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            )

        # next token split: (b, n) -> inp (b, n - 1), target (b, n - 1)

        inp, target = (token_ids[:, :-1], token_ids[:, 1:]) if return_loss else (token_ids, None)

        # first pass (standard)

        logits, intermediates = self.net(inp, return_intermediates = True, **kwargs)

        all_logits = [logits]
        first_pass_loss = ce_loss(logits, target) if return_loss else None
        feedback_pass_losses = []

        # feedback passes (temporal parallel)

        for _ in range(temporal_parallel_passes - 1):

            # fresh transforms each pass - they read the latents out of the previous pass's intermediates

            transforms = self.make_transforms(
                intermediates,
                training = self.training,
                override_transition = transition
            )

            logits, intermediates = self.net(
                inp,
                transform_block_inputs = transforms,
                return_intermediates = True,
                **kwargs
            )

            all_logits.append(logits)

            if not return_loss:
                continue

            # feedback loss only on positions that received latent states (1:)
            # position 0 is never fused and attention is causal, so its logits equal pass 1's and scoring it
            # again would only repeat part of L_1. with a single position the fallback repeats pass 1's loss

            pred, labels = (logits[:, 1:], target[:, 1:]) if logits.shape[1] > 1 else (logits, target)
            feedback_pass_losses.append(ce_loss(pred, labels))

        # no loss - last pass logits, with every pass's logits available through maybe_return

        if not return_loss:
            return all_logits[-1], dict(all_pass_logits = all_logits)

        # combine first pass loss with feedback pass losses
        # all_losses: (p,) = [L_1, L_2, ..., L_p]

        all_pass_losses = [first_pass_loss, *feedback_pass_losses]
        all_losses = torch.stack(all_pass_losses)

        feedback_loss = self.zero

        if len(feedback_pass_losses) > 0:
            feedback_losses = all_losses[1:]

            # weights come from detached losses, so they act as constants (no gradient through the weighting)
            # w_k = exp(-decay * sum_{j < k} max(L_j - threshold, 0)); pass 1's weight is dropped, the rest renormalized:
            #   F = sum_{k >= 2} w_k L_k / sum_{k >= 2} w_k
            # note: exp(-decay * max(L_1 - threshold, 0)) is a factor of every w_k (k >= 2) and cancels in the
            # normalization, so pass 1's loss never affects the weights; with 2 passes the single weight
            # normalizes to 1 and this equals the plain mean - the weighting only acts from 3 passes on

            if dynamic_rollout_loss_weight:
                weights = dynamic_rollout_loss_weights(
                    all_losses.detach(),
                    decay = dynamic_loss_decay,
                    threshold = dynamic_loss_threshold
                )[1:]
                feedback_loss = (feedback_losses * weights).sum() / weights.sum().clamp(min = 1e-8)
            else:
                feedback_loss = feedback_losses.mean()

        # L_1 always enters at weight 1, the combined feedback term F scaled by feedback_pass_weight

        loss = first_pass_loss + feedback_loss * self.feedback_pass_weight

        # breakdown holds the unweighted per-pass losses

        loss_breakdown = LossBreakdown(
            first_pass_loss = first_pass_loss,
            feedback_pass_losses = feedback_pass_losses,
            all_pass_losses = all_pass_losses
        )

        return loss, dict(
            loss_breakdown = loss_breakdown,
            all_pass_logits = all_logits
        )
