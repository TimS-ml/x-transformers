"""
Encoder-free latent variable decoder, trained with Explorative Modeling (XM).

Alexi Gladstone, Heng Ji, Yilun Du - "Explorative Modeling: Unlocking a Third
Pretraining Axis and End-to-End Generation", https://arxiv.org/abs/2607.27372
(title per the bibtex entry in README.md). The class docstring also credits
Lei Yang, https://arxiv.org/abs/2401.00036, added right after upstream commit
c18b9bc ("ddn/xm can be used to induce factorized latent variables"), which is
the multi-slot, per-slot-selection extension implemented here.

Problem. A plain autoregressive decoder has no global latent variable - any
sequence-level "plan" has to live implicitly in the token history. Latent
variable models usually infer the latent of a training sequence with an
encoder. XM has no encoder: the latent for a training sequence is found by
search over samples from the prior, with the model's own loss as the score.

Mechanism, one training step (`XMLatentDecoder.forward`):

1. Conditioning. There are `num_latents` latent slots, each a `latent_dim`
   vector drawn from N(0, I). They go through `latent_proj` and are prepended
   to the token embeddings (`prepend_embeds`), so under the causal mask every
   token attends to all of them. Their output positions are cut off again
   (`excise_prepend_embeds`), so logits line up with the tokens.
2. Exploration. One slot - the active slot, uniformly random unless
   `active_latent_index` is given - gets `candidates` (k) independent N(0, I)
   draws. The other slots are drawn once and shared by all k candidates. The
   batch is repeated k times and the net is run on every (sequence, candidate)
   pair.
3. Selection. Each candidate's fitness is its next token cross entropy,
   masked-mean over positions. A winner function (default
   `lowest_loss_winner_fn`, argmin) keeps one candidate per sequence, and the
   returned loss is the batch mean of the winners' losses only. The code labels
   this winner-takes-all selection "Forward XM". Selection is an index, so no
   gradient flows through it, and losing candidates get none from this term.
4. Diversity (optional, `repulsive_loss_weight > 0`). The generalized
   Jensen-Shannon divergence between the k candidates' next token
   distributions is subtracted from the loss, pushing different latents
   towards different predictions. This term reaches every candidate.

Upstream commit 5154be6 puts it as "at the core, the xm is an evolutionary
algorithm": a population of k random variants of one slot, a fitness function,
and one survivor per sequence, whose loss is what the optimizer minimizes. It
runs for a single generation per step, and nothing is carried across steps -
every call starts from fresh prior samples unless latents are passed in. The
minimum over k is below the average candidate loss only when the candidates'
predictions differ, so the objective rewards predictions that depend on the
latent. The optional repulsive term pushes the same way directly. `train_enwik8_xm.py`
tracks this as "latent KL": the KL between the predictive distributions under
two independent prior draws of all slots.

Factorized latents. Each slot has its own winner function (`winner_fns`), and
only one slot is explored per step, so different slots can be selected by
different criteria (the tests pair lowest loss for slot 0 with highest entropy
for slot 1).

Inference:

- `generate` draws every slot from the prior (or takes the given latents) and
  decodes with `AutoregressiveWrapper`, with the latents prepended.
- `generate_with_candidate_latents` runs the same exploration / selection on
  the prompt alone and then decodes with the winning latents. There are no
  labels at this point, so the winner function scores candidate logits with
  some label-free criterion (e.g. confidence). Given a sequence of winner
  functions, it sweeps the slots in order, one selection round per slot, with
  earlier winners held fixed.
- A slot can be pinned to a chosen latent by passing a per-slot tuple, with
  None for the slots to sample (`init_latents`).

Shape letters: b batch, k candidates, n token positions, c vocab, d latent
width. Latents are (b, num_latents, d), or (b, k, num_latents, d) per
candidate; the code's einops patterns for latents ('b k n d') reuse `n` for
the slot axis.
"""

from __future__ import annotations

import math
from random import random, randrange
from typing import Callable, Sequence

import torch
from torch import nn, Tensor, cat
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, reduce
from torch_einops_utils import temp_eval, batched_index_select, masked_mean, and_masks

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg

# a single value is broadcast to `length` copies; a tuple / list is returned as is,
# with no length check here

def cast_tuple(t, length = 1):
    return t if isinstance(t, (tuple, list)) else ((t,) * length)

# winner callback helpers

def lowest_loss_winner_fn(losses: Tensor, intermediates = None) -> Tensor:
    """Default training selection rule: the candidate with the lowest loss wins.

    losses: (b, k) per-candidate masked-mean cross entropy. `intermediates` is
    accepted and ignored, so the function fits the two-argument training call.
    Returns (b,) long indices into the candidate dim.

    Winner function contract in `XMLatentDecoder.forward`: return (b,) indices,
    or (b, k) scores, in which case the *highest* score wins (argmax). A
    score-returning rule that prefers low loss must return e.g. `-losses`.

    `forward` checks for this exact function by identity (`is`). Any other
    callable, even an equivalent wrapper, makes the net return intermediates and
    keeps every candidate's logits in memory.

    note: `generate_with_candidate_latents` calls winner functions with one
    argument, the candidate logits (b, k, n, c), and there is no loss. Applied
    there, this function takes argmin over the vocab and then argmax over
    positions. That gives position indices, not candidate indices (an IndexError
    in a quick check), so a label-free winner function has to be passed at
    generation time.
    """
    return losses.argmin(dim = -1)

# main class

class XMLatentDecoder(Module):
    """
    Latent Variable Decoder based on Explorative Modeling (Forward XM)
    by Alexi Gladstone et al. (https://arxiv.org/abs/2607.27372).
    and Lei Yang (https://arxiv.org/abs/2401.00036)

    Wraps a decoder-only `TransformerWrapper` (`net`). See the module docstring
    for the method end to end.

    - `forward(seq)`                     - training loss (scalar). With
                                           `return_loss = False`, the raw per-candidate
                                           logits and latents, with no selection
    - `generate`                         - decode with prior-sampled or given latents
    - `generate_with_candidate_latents`  - select latents on the prompt, then decode
    - `init_latents`                     - None / tensor / per-slot tuple -> latent tensor

    Trainable parameters are `net` and `latent_proj` (when it is a Linear). The
    latents themselves are fresh N(0, I) samples, not parameters.
    """

    def __init__(
        self,
        net: Module,                                                        # decoder-only TransformerWrapper; latents reach it as prepend_embeds
        num_latents = 4,                                                    # latent slots, i.e. prefix positions; each can be explored / selected on its own
        dim = None,                                                         # width latents are projected to - must equal net's token embedding width (see note below)
        latent_dim = None,                                                  # width of the sampled N(0, I) latents; defaults to dim
        candidates = 2,                                                     # k - candidates per sequence per step; 1 makes selection trivial
        max_batch_size = None,                                              # max (b k) rows per net call; None = all b * k rows in one call
        ignore_index = -100,                                                # target id masked out of the CE and of the repulsive term
        latent_drop_prob = 0.,                                              # training only - prob that a whole step runs as a plain LM with no latent prefix
        always_latent_proj = False,                                         # use a Linear latent_proj even when latent_dim == dim
        repulsive_loss_weight = 0.,                                         # weight of the JSD diversity bonus between candidates; 0. = off
        winner_fn: Callable | Sequence[Callable] = lowest_loss_winner_fn,   # training selection rule, broadcast to every slot (or one per slot if a sequence)
        winner_fns: Sequence[Callable] | None = None                        # one selection rule per slot, length num_latents; takes precedence over winner_fn
    ):
        super().__init__()
        self.net = net

        assert candidates >= 1, 'candidates must be at least 1'
        self.candidates = candidates
        self.max_batch_size = max_batch_size
        self.ignore_index = ignore_index

        # `dim` is the width latents are projected to, and must equal the token embedding
        # width of `net` - TransformerWrapper asserts this on prepend_embeds
        # note: when only `latent_dim` is given it takes precedence over net.attn_layers.dim
        # here, so dim = latent_dim, latent_proj becomes an Identity, and the prepend fails
        # that width assertion unless latent_dim equals the model width. passing `dim`
        # explicitly avoids it. net.attn_layers.dim is also not the right width when the
        # wrapper was built with a separate `emb_dim` - latents are concatenated before
        # project_emb, so they must be emb_dim wide

        dim = default(dim, latent_dim, net.attn_layers.dim)

        self.dim = dim
        self.num_latents = num_latents
        self.latent_drop_prob = latent_drop_prob

        self.latent_dim = default(latent_dim, dim)

        # a Linear only when the widths differ or it is forced; otherwise the raw N(0, I)
        # sample itself is prepended as the embedding

        has_latent_proj = self.latent_dim != self.dim or always_latent_proj

        self.latent_proj = nn.Linear(self.latent_dim, self.dim) if has_latent_proj else nn.Identity()

        # repulsive loss force between candidate distributions

        self.repulsive_loss_weight = repulsive_loss_weight

        # one selection rule per latent slot - a single callable is broadcast to all slots

        winner_fns = cast_tuple(default(winner_fns, winner_fn), num_latents)
        assert len(winner_fns) == num_latents, f'winner_fns length ({len(winner_fns)}) must match num_latents ({num_latents})'

        self.winner_fns = tuple(winner_fns)

    @property
    def max_seq_len(self):
        """Delegates to `net.max_seq_len`, so code expecting a TransformerWrapper-like
        `.max_seq_len` works. Counts tokens only - the num_latents prefix sits on top
        of this inside the attention sequence.
        """
        return self.net.max_seq_len

    def init_latents(
        self,
        latents: Tensor | Sequence[Tensor | None] | None,
        batch: int,
        device = None
    ) -> Tensor:
        """Turn the user-facing `latents` argument into a latent tensor.

        - None           -> fresh N(0, I) sample (b, num_latents, latent_dim), default
                            dtype, on `device`
        - a Tensor       -> returned untouched: (b, num_latents, latent_dim), or the
                            per-candidate (b, k, num_latents, latent_dim), which only
                            `forward` understands (and so the first round of
                            `generate_with_candidate_latents`); `generate` needs 3D
        - tuple / list   -> exactly num_latents entries, each (b, latent_dim) or None.
                            None slots are sampled from N(0, I) with the device / dtype
                            of the first given slot, then everything is stacked on dim 1
                            -> (b, num_latents, latent_dim)

        No gradient is involved unless the caller passes tensors that require grad.
        """
        if not exists(latents):
            return torch.randn(batch, self.num_latents, self.latent_dim, device = device)

        # a specific latent or None can be given for each slot, with None randomly initialized

        if not isinstance(latents, (tuple, list)):
            return latents

        assert len(latents) == self.num_latents, f'latents length ({len(latents)}) must match num_latents ({self.num_latents})'

        # first non-None slot supplies device / dtype for the sampled ones

        ref = default(*latents)
        device, dtype = (ref.device, ref.dtype) if exists(ref) else (device, None)

        latents = [default(latent, torch.randn(batch, self.latent_dim, device = device, dtype = dtype)) for latent in latents]

        return torch.stack(latents, dim = 1)

    @temp_eval
    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        seq_len: int,
        latents: Tensor | Sequence[Tensor | None] | None = None,
        **kwargs
    ) -> Tensor:
        """Sample `seq_len` tokens after `start_tokens` (b, t), conditioned on latents.

        latents: None (every slot drawn from the N(0, I) prior), a (b, num_latents,
        latent_dim) tensor, or a per-slot tuple (see `init_latents`). They go through
        `latent_proj` and are handed to a fresh `AutoregressiveWrapper` as
        `prepend_embeds`. With kv caching the latent prefix is encoded on the first
        step and then lives in the cache. Remaining kwargs (temperature,
        filter_logits_fn, cache_kv, ...) go to `AutoregressiveWrapper.generate`.

        Returns (b, seq_len) sampled token ids, prompt excluded. Runs in eval mode
        (restored afterwards) under no_grad.
        """
        batch, device = start_tokens.shape[0], start_tokens.device

        latents = self.init_latents(latents, batch, device = device)

        # (b, num_latents, latent_dim) -> (b, num_latents, dim)

        latent_cond = self.latent_proj(latents)

        auto_wrapper = AutoregressiveWrapper(self.net)
        return auto_wrapper.generate(start_tokens, seq_len, prepend_embeds = latent_cond, excise_prepend_embeds = True, **kwargs)

    @temp_eval
    @torch.no_grad()
    def generate_with_candidate_latents(
        self,
        start_tokens: Tensor,
        seq_len: int,
        winner_fn: Callable | Sequence[Callable] | None = None,
        winner_fns: Sequence[Callable] | None = None,
        candidates: int | None = None,
        latents: Tensor | Sequence[Tensor | None] | None = None,
        active_latent_index: int | None = None,
        return_best_latents = False,
        **kwargs
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Pick latents by exploration + selection on the prompt, then decode with them.

        Each selection round calls `self(start_tokens, return_loss = False,
        active_latent_index = i)`. Slot i gets k fresh N(0, I) draws while the other
        slots keep their current values, the net scores every candidate over the
        prompt, and `winner_fns[i](candidate_logits)` keeps one per batch element.
        The winning (b, num_latents, latent_dim) latents seed the next round, and
        after the last round they condition `generate`.

        Which slots get a round:

        - `active_latent_index` given              -> only that slot
        - a sequence of winner fns passed here     -> every slot, in order 0 .. num_latents - 1
                                                      (length is not checked here)
        - a single fn passed, or nothing passed    -> slot 0 only

        Winner functions here take ONE argument, the candidate logits (b, k, t, c)
        over the prompt (there are no labels), and return (b,) indices or (b, k)
        scores (argmax wins). Pass one explicitly: if nothing is passed, the
        constructor's `winner_fns` are used, and the default `lowest_loss_winner_fn`
        is a loss-based training rule that does not work on logits (see its
        docstring).

        Every round draws fresh candidates for its slot, candidate 0 included. The
        slot's previous value, including a latent the caller supplied for it, is
        not kept in the pool, so there is no elitism, and each slot is visited once
        per call. Selection is per sequence - one latent set per batch element,
        chosen before decoding starts - not per token.

        `candidates` overrides the constructor's k for these rounds. Runs in eval mode
        under no_grad, so the latent-drop branch of `forward` is never taken here.

        Returns (b, seq_len) tokens or, with `return_best_latents`,
        `(tokens, (candidate_latents, winners))`. candidate_latents is the
        (b, k, num_latents, latent_dim) pool of the LAST round only. winners is (b,)
        for a single round, else (b, rounds). The latents actually decoded with are
        candidate_latents[arange(b), winners[:, -1]].

        note: `**kwargs` go to the scoring forward pass(es) as well as to
        `generate`, so sampling-only kwargs reach the transformer's forward and
        raise a TypeError there (checked for `temperature` and `filter_logits_fn`).
        """

        # whether to sweep all slots is decided only by what this call was given - the
        # constructor's winner_fns never trigger a full sweep

        passed_winner_fns = default(winner_fns, winner_fn)
        is_seq_winners = isinstance(passed_winner_fns, (tuple, list))

        winner_fns = cast_tuple(default(passed_winner_fns, self.winner_fns), self.num_latents)

        batch, device = start_tokens.shape[0], start_tokens.device

        latents = self.init_latents(latents, batch, device = device)

        # a single winner fn selects winners for the first latent, while a sequence of winner fns sweeps all latents

        latent_indices = [active_latent_index] if exists(active_latent_index) else (range(self.num_latents) if is_seq_winners else [0])

        all_winners = []

        for latent_idx in latent_indices:
            # candidate_logits (b, k, t, c), candidate_latents (b, k, num_latents, latent_dim)
            # note: winner_fn is not passed through, so forward decides whether the net must
            # return intermediates from the constructor's winner_fns[latent_idx]; for a custom
            # one they are computed here and then discarded

            candidate_logits, candidate_latents = self(
                start_tokens,
                latents = latents,
                candidates = candidates,
                active_latent_index = latent_idx,
                return_loss = False,
                **kwargs
            )

            # label-free selection on the prompt - the winner fn only sees logits

            winner = winner_fns[latent_idx](candidate_logits)

            if winner.ndim > 1:
                winner = winner.argmax(dim = -1)

            all_winners.append(winner)

            # (b, k, num_latents, latent_dim) -> (b, num_latents, latent_dim) - this round's
            # winner is held fixed while the next slot is explored

            latents = batched_index_select(candidate_latents, winner, dim = 1)

        out = self.generate(
            start_tokens = start_tokens,
            seq_len = seq_len,
            latents = latents,
            **kwargs
        )

        if not return_best_latents:
            return out

        # (b,) for one round, else (b, rounds); pair with the last round's candidate_latents

        winner_result = all_winners[0] if len(all_winners) == 1 else torch.stack(all_winners, dim = -1)
        return out, (candidate_latents, winner_result)

    def forward(
        self,
        seq: Tensor,
        latents: Tensor | Sequence[Tensor | None] | None = None,
        candidates = None,
        max_batch_size = None,
        latent_drop_prob = None,
        return_loss = True,
        repulsive_loss_weight = None,
        mask: Tensor | None = None,
        winner_fn: Callable | Sequence[Callable] | None = None,
        winner_fns: Sequence[Callable] | None = None,
        active_latent_index: int | None = None,
        **kwargs
    ):
        """XM training step, or the raw candidate pass used for selection at inference.

        seq: (b, n + 1) token ids when `return_loss` (inputs seq[:, :-1], targets
        seq[:, 1:]), or (b, n) when not.
        mask: optional bool of the same length as seq, True = real token. Targets
        equal to `ignore_index` are left out of every loss as well.
        latents: None / tensor / per-slot tuple (see `init_latents`). A 4D
        (b, k, num_latents, latent_dim) tensor supplies every candidate explicitly
        and is used as is: no slot is resampled, and k must equal `candidates`.
        candidates, max_batch_size, latent_drop_prob, repulsive_loss_weight:
        per-call overrides of the constructor values.
        winner_fn / winner_fns: per-call override of the selection rule(s).
        active_latent_index: slot to explore, random when None. It also picks which
        winner fn applies.

        Returns:

        - return_loss = True: scalar loss, the batch mean of each sequence's winning
          per-sequence masked-mean CE, minus repulsive_loss_weight * JSD
        - return_loss = False: (candidate_logits (b, k, n, c), latents
          (b, k, num_latents, latent_dim)), with no selection applied
        - if the latent-drop branch fires (training mode only): the net runs with no
          latent prefix, returning a token-weighted masked-mean CE, or plain
          (b, n, c) logits when return_loss = False

        Gradient: selection is an argmin / argmax index, so the winner term trains
        `net` and `latent_proj` only through each sequence's winning candidate.
        Losers get gradient only from the repulsive term. Sampled latents are
        constants. Winner functions receive `candidate_losses` with grad attached,
        but only the indices they produce are used.

        Winner function contract (training): called as fn(losses (b, k),
        intermediates), falling back to fn(losses) on TypeError; returns (b,)
        indices or (b, k) scores (argmax wins).
        """
        candidates = default(candidates, self.candidates)
        max_batch_size = default(max_batch_size, self.max_batch_size)
        latent_drop_prob = default(latent_drop_prob, self.latent_drop_prob)

        repulsive_loss_weight = default(repulsive_loss_weight, self.repulsive_loss_weight)

        # precedence: per-call winner_fns, then per-call winner_fn, then the constructor's

        winner_fns = cast_tuple(default(winner_fns, winner_fn, self.winner_fns), self.num_latents)

        # one slot explored per call, uniformly at random by default
        # note: this and the latent-drop coin below use python's `random` module, so
        # torch.manual_seed alone does not make them reproducible

        active_latent_index = default(active_latent_index, randrange(self.num_latents))
        assert 0 <= active_latent_index < self.num_latents

        winner_fn = winner_fns[active_latent_index]

        batch, device = seq.shape[0], seq.device

        # autoregressive sequence targets

        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        # loss_mask (b, n) lines up with the targets: the user mask shifted by one
        # (mask[:, 1:]) and-ed with targets != ignore_index. without targets it is the mask
        # itself and-ed with seq != ignore_index. and_masks skips a None mask

        seq_mask = mask[:, 1:] if (exists(mask) and return_loss) else mask
        ignore_mask = (labels != self.ignore_index) if return_loss else (seq != self.ignore_index)

        loss_mask = and_masks((seq_mask, ignore_mask))

        # check if latents are dropped during training
        # one coin per call: with prob latent_drop_prob the whole batch trains as a plain
        # LM - no latent prefix at all and no candidates - so the net also learns to model
        # sequences without latents
        # note: `mask` is passed unshifted here while seq is already seq[:, :-1], so with
        # return_loss and a mask this call hits a length mismatch (n + 1 vs n). the main
        # path below passes the shifted loss_mask instead

        if self.training and latent_drop_prob > 0. and random() < latent_drop_prob:
            logits = self.net(seq, mask = mask, **kwargs)

            if not return_loss:
                return logits

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> (b n) c'),
                rearrange(labels, 'b n -> (b n)'),
                reduction = 'none',
                ignore_index = self.ignore_index
            )

            # token-weighted mean over all valid (b, n) positions, whereas the main path
            # averages per sequence first and then over the batch

            return masked_mean(rearrange(loss, '(b n) -> b n', b = batch), loss_mask)

        # the k candidates are folded into the batch, (b k) rows, and run through the net in
        # chunks of at most max_batch_size rows. under autograd every chunk's graph is kept
        # until backward, so this bounds the size of each net call rather than the total
        # activation memory; under no_grad (selection at inference) it does bound memory

        total = batch * candidates
        chunk_size = default(max_batch_size, total)

        # handle custom or random Gaussian noise latent candidates

        latents = self.init_latents(latents, batch, device = device)

        # only the active latent is explored, all other latents held constant across candidates
        # (b, num_latents, d) -> (b, k, num_latents, d): all candidates share the other slots,
        # and the active slot gets k independent N(0, I) draws - a value the caller supplied
        # for that slot is overwritten. clone() because repeat returns an expanded view
        # (stride 0 on k) that cannot be written in place. a 4D input skips this block

        if latents.ndim == 3:
            latents = repeat(latents, 'b n d -> b k n d', k = candidates).clone()
            latents[:, :, active_latent_index] = torch.randn(batch, candidates, self.latent_dim, device = device, dtype = latents.dtype)

        # (b, k, num_latents, latent_dim) -> (b k, num_latents, dim), b-major like the repeats below

        latent_cond = self.latent_proj(rearrange(latents, 'b k n d -> (b k) n d'))

        # repeat input sequence and targets K times across batch dimension
        # (b, n) -> (b k, n), b-major: row i * k + j is sequence i under candidate j,
        # matching the latent flattening above

        seq_candidates = repeat(seq, 'b ... -> (b k) ...', k = candidates)

        if return_loss:
            labels_candidates = repeat(labels, 'b ... -> (b k) ...', k = candidates)

        mask_candidates = repeat(loss_mask, 'b ... -> (b k) ...', k = candidates)

        losses = []
        all_logits = []
        all_intermediates = []

        # the JSD term needs every candidate's logits, and so does any custom winner fn
        # (through intermediates.logits). only the exact default rule (identity check)
        # skips both return_intermediates and keeping all (b k, n, c) logits

        calc_repulsion = repulsive_loss_weight > 0. and candidates > 1
        needs_intermediates = winner_fn is not lowest_loss_winner_fn
        collect_logits = calc_repulsion or needs_intermediates

        for start in range(0, total, chunk_size):
            chunk_batch_size = min(total - start, chunk_size)
            chunk = slice(start, start + chunk_batch_size)

            chunk_seq = seq_candidates[chunk]
            chunk_latents = latent_cond[chunk]

            # with return_loss, the net's key padding mask is the target-aligned loss_mask, not
            # the input mask mask[:, :-1], so it also hides the last real input token. under
            # causal attention only positions at or after that token can see it, and their
            # targets are masked, so right-padded batches get the same losses. the wrapper
            # pads the mask with True for the prepended latent positions

            chunk_kwargs = kwargs.copy()
            if exists(mask):
                chunk_kwargs['mask'] = mask_candidates[chunk]

            # latents in front of the tokens, cut back off the output -> logits (chunk, n, c)

            net_out = self.net(
                chunk_seq,
                prepend_embeds = chunk_latents,
                excise_prepend_embeds = True,
                return_intermediates = needs_intermediates,
                **chunk_kwargs
            )

            logits, intermediates = net_out if needs_intermediates else (net_out, None)

            if needs_intermediates:
                all_intermediates.append(intermediates)

            # inference path: no targets, only the logits are needed

            if not return_loss:
                all_logits.append(logits)
                continue

            if collect_logits:
                all_logits.append(logits)

            chunk_labels = labels_candidates[chunk]
            chunk_mask = mask_candidates[chunk]

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> (b n) c'),
                rearrange(chunk_labels, 'b n -> (b n)'),
                reduction = 'none',
                ignore_index = self.ignore_index
            )

            # ((chunk n),) -> (chunk, n) -> masked mean over positions -> (chunk,), one fitness
            # value per (sequence, candidate) row. a fully masked row gives 0, since
            # masked_mean clamps its denominator

            loss = rearrange(loss, '(b n) -> b n', b = chunk_batch_size)
            losses.append(masked_mean(loss, mask = chunk_mask, dim = -1))

        # (b k, n, c) -> (b, k, n, c). the whole (b, k, num_latents, d) latent pool is returned
        # so the caller can index out its winner

        if not return_loss:
            raw_logits = cat(all_logits, dim = 0)
            candidate_logits = rearrange(raw_logits, '(b k) ... -> b k ...', b = batch, k = candidates)
            return candidate_logits, latents

        if collect_logits:
            raw_logits = cat(all_logits, dim = 0)
            candidate_logits = rearrange(raw_logits, '(b k) ... -> b k ...', b = batch, k = candidates)

        # selection (winner-takes-all candidate selection - Forward XM)
        # (b k,) -> (b, k). 'mean' reduces no axis here - this is a pure reshape

        candidate_losses = reduce(cat(losses, dim = 0), '(b k) -> b k', 'mean', b = batch, k = candidates)

        intermediates = None

        # a custom winner fn sees intermediates.logits as (b, k, n, c); the other fields keep
        # the flattened (b k) batch layout of the net call
        # note: with max_batch_size < b * k there is one intermediates object per chunk and a
        # list is used, so the `.logits` assignment raises AttributeError ('list' object has
        # no attribute 'logits'). custom winner fns currently need a single chunk

        if needs_intermediates:
            intermediates = all_intermediates[0] if len(all_intermediates) == 1 else all_intermediates
            intermediates.logits = candidate_logits

        # two-argument (losses, intermediates) signature first, one-argument fallback
        # note: a TypeError raised *inside* a two-argument winner fn is caught as well, and
        # the fn is retried with one argument, which can hide the original error

        try:
            winner = winner_fn(candidate_losses, intermediates)
        except TypeError:
            winner = winner_fn(candidate_losses)

        # (b, k) scores -> index of the highest-scoring candidate, (b,)

        if winner.ndim > 1:
            winner = winner.argmax(dim = -1)

        # gather each sequence's winning loss (b,) and average over the batch. the index
        # carries no gradient, so net / latent_proj are trained only through the winners

        winner_loss = batched_index_select(candidate_losses, winner, dim = 1).mean()

        if not calc_repulsion:
            return winner_loss

        # diversity (repulsive loss via jensen-shannon divergence)
        # generalized JSD across the k candidate distributions at each position:
        # JSD = mean_k KL(p_k || m) with m = mean_k p_k (equivalently H(m) - mean_k H(p_k)),
        # bounded by log k. it is subtracted from the loss, so more divergence = lower loss.
        # nothing is detached - every candidate's logits, losers included, and the mixture
        # get gradient

        log_prob = candidate_logits.log_softmax(dim = -1)

        # log m = log(1/k sum_k p_k), stably via logsumexp over k: (b, k, n, c) -> (b, n, c),
        # then broadcast back to (b, k, n, c)

        log_mixture = log_prob.logsumexp(dim = 1) - math.log(candidates)
        log_mixture = repeat(log_mixture, 'b n c -> b k n c', k = candidates)

        # kl_div(input = log m, target = log p_k, log_target = True) is p_k * (log p_k - log m)
        # elementwise, i.e. KL(p_k || m) once summed over the vocab -> (b, k, n); then mean
        # over k -> (b, n)

        kl = F.kl_div(log_mixture, log_prob, log_target = True, reduction = 'none').sum(dim = -1)
        kl = reduce(kl, 'b k n -> b n', 'mean')

        # token-weighted mean over valid target positions -> scalar

        repulsive_kl = masked_mean(kl, mask = loss_mask)

        return winner_loss - repulsive_loss_weight * repulsive_kl
