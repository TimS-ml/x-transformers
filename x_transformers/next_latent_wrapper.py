"""
Next-latent prediction.

Trains a decoder transformer with the usual next-token cross entropy plus an
auxiliary objective: a small recurrent dynamics model must roll the final hidden
state forward in latent space and predict the hidden states the transformer
itself produces at the next `num_rollouts` positions. The embedding of the token
consumed at each step is the "action" driving the transition, so the dynamics
model is an action-conditioned world model over the representation space sitting
just below `to_logits`.

Scheme from "Next-Latent Prediction Transformers Learn Compact World Models"
(Teoh et al., 2026), https://arxiv.org/abs/2511.05963 - bibtex in README.md.

Rollout structure, for batch element b and position n:

    start state       curr_latent[b, n] = hidden_states[b, n]
    step r action     token_emb(x[b, n + 1 + r])
    step r target     hidden_states[b, n + 1 + r]

Step 0 is plain next-latent prediction; steps 1..num_rollouts-1 feed the model's
own predicted latent back in, so the whole rollout is produced without ever
re-running the transformer. Every position n rolls out in parallel.

Only the *targets* are detached. Gradient still reaches the transformer through
the starting latent, which is what makes this a regularizer on the
representation space rather than a probe fitted on top of a frozen encoder.

Relationship to `AutoregressiveWrapper`: this is a training-time wrapper only.
It reproduces the same cross entropy (same `inp, target = x, x[:, 1:]`
convention) but exposes no `generate`. After training, pull the plain
transformer back out of `.net` and hand it to `AutoregressiveWrapper` to sample
- see the `__main__` block at the bottom of this file, and `train_path_star.py`
in the repo root, which is the runnable driver (path-star graph task, toggled
with `--use_nextlat`).
"""

from __future__ import annotations
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn, stack, cat, tensor
from torch.nn import Module

import einx
from einops import rearrange, reduce
from torch_einops_utils import pad_right_at_dim, masked_mean, exclusive_cumsum, pack_with_inverse

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

# dynamic rollout loss weighting
# weight each rollout step by exp(-decay * cumulative loss of preceding steps), as errors compound through the dynamics model

def dynamic_rollout_loss_weights(
    step_losses, # (r b n)
    decay = 1.
):
    cum_step_losses = exclusive_cumsum(step_losses, dim = 0)
    return (-decay * cum_step_losses).exp()

# constants

# breakdown returned by `forward(..., return_loss_breakdown = True)`
# reported *before* the three global `*_loss_weight` scalars are applied (ce carries
# no weight); the per-rollout weights are already folded into `next_latent` and `kl`.
# disabled terms come back as a 0. tensor, not None - see the `zero` buffer

Losses = namedtuple('Losses', ['ce', 'next_latent', 'kl', 'sigreg'])

# classes

def MLP(
    dim_in,
    dim_out,
    dim_hidden,
    depth = 3
):
    """Plain GELU MLP factory, `depth` linears with no activation on the last one.

    The final Linear is left bare so the caller can zero-init it (see the
    `residual` branch of `NextLatentWrapper.__init__`).
    """

    layers = []

    for i in range(depth):
        is_last = i == (depth - 1)

        in_d = dim_in if i == 0 else dim_hidden
        out_d = dim_out if is_last else dim_hidden

        layers.append(nn.Linear(in_d, out_d))

        if not is_last:
            layers.append(nn.GELU())

    return nn.Sequential(*layers)

class ResidualDynamics(Module):
    """Latent transition model that predicts the *delta* between consecutive latents.

    latent <- latent + net(cat(latent, action_embed)). This is the paper's scheme
    (3-layer mlp encoding the delta). When `net` is the default one built by
    `NextLatentWrapper` its last Linear is zero-init'd, so the transition starts as
    the identity and the unroll cannot diverge at init; a user-supplied `net` gets
    no such treatment.

    `net` maps (dim * 2) -> dim. Every rollout step reuses the same weights, hence
    "recurrent" - the depth of the unroll is `r`, not the depth of `net`.
    """

    def __init__(
        self,
        net: Module
    ):
        super().__init__()
        self.net = net

    def forward(
        self,
        rollout_next_token_embeds, # (r b n d)
        curr_latent                # (b n d)
    ):
        """Unroll `r` steps from `curr_latent`, one action embedding per step."""

        # collapse (b n) into a single flat axis - every (batch, position) pair is an
        # independent rollout, so they are all just rows here
        # (r b n d) -> (r, b*n, d) and (b n d) -> (b*n, d)

        next_embeds, unpack_next_embeds = pack_with_inverse(rollout_next_token_embeds, 'r * d')
        curr_latent, _ = pack_with_inverse(curr_latent, '* d')

        next_latents = []

        # iterating a (r, b*n, d) tensor yields one (b*n, d) action slab per rollout step;
        # curr_latent is rebound each step, so step r consumes the prediction of step r-1

        for next_embed in next_embeds:
            dynamics_input = cat((curr_latent, next_embed), dim = -1)
            curr_latent = curr_latent + self.net(dynamics_input)

            next_latents.append(curr_latent)

        out = stack(next_latents)
        return unpack_next_embeds(out, 'r * d')  # (r, b*n, d) -> (r b n d)

class RolloutGRU(Module):
    """Drop-in alternative to `ResidualDynamics` using a single-layer GRU.

    The GRU's *time* axis is the rollout axis, the current latent is the initial
    hidden state, and the action embeddings are the inputs. Because hidden size ==
    dim, the GRU hidden states are themselves the predicted latents.

    Per the upstream commit that added it (f2b042e), this trains but performs
    worse than the residual-delta MLP; it is kept around as an ablation.
    """

    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.gru = nn.GRU(input_size = dim, hidden_size = dim, batch_first = False)

    def forward(
        self,
        rollout_next_token_embeds, # (r b n d)
        curr_latent                # (b n d)
    ):
        """Unroll `r` steps from `curr_latent` by running the GRU along the rollout axis."""

        # pack batch and sequence dimensions

        # (r b n d) -> (r, b*n, d), which is exactly nn.GRU's batch_first = False
        # layout with rollout step playing the role of sequence length

        next_embeds, unpack_next_embeds = pack_with_inverse(rollout_next_token_embeds, 'r * d')

        # (b n d) -> (b*n, d) -> (1, b*n, d), the leading 1 being num_layers * num_directions

        first_hidden, _ = pack_with_inverse(curr_latent, '* d')
        first_hidden = rearrange(first_hidden, '... -> 1 ...')

        # run gru over the rollout sequence length

        out, _ = self.gru(next_embeds, first_hidden)

        # unpack

        return unpack_next_embeds(out, 'r * d')  # (r, b*n, d) -> (r b n d)

# loss modules

class MSECosineSimLoss(Module):
    """Convex blend of per-element MSE and per-vector cosine distance.

    Mixing in a direction-only term stops the latent regression from being
    dominated by norm mismatch. Like the other loss options, stays unreduced so
    the caller can apply per-rollout weights and masking.
    """

    def __init__(
        self,
        weight = 0.9    # lerp toward the cosine term; 0. is pure mse, 1. is pure cosine
    ):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction = 'none')      # (... d)
        cos_loss = 1. - F.cosine_similarity(pred, target, dim = -1)  # (...), one scalar per vector
        cos_loss = rearrange(cos_loss, '... -> ... 1')               # trailing 1 to broadcast back over d

        # lerp(a, b, w) == (1 - w) * a + w * b, so the same scalar cosine distance is
        # blended into every feature of that vector

        return mse_loss.lerp(cos_loss, self.weight)

# main next latent wrapper

class NextLatentWrapper(Module):
    """Training wrapper adding next-latent prediction on top of a `TransformerWrapper`.

    Wraps a decoder and returns a single scalar loss combining, at most, four
    terms: next-token cross entropy, latent regression over a multi-step rollout,
    a KL between what the predicted and true latents decode to in logit space,
    and an optional isotropy regularizer on the hidden states.

    Reuses the wrapped net's `token_emb` (for the action inputs) and `to_logits`
    (frozen, for the KL), so the only new parameters are the dynamics model.

    No `generate` here - unwrap `.net` and use `AutoregressiveWrapper` for
    sampling.
    """

    @staticmethod
    def sigreg_loss(
        x,
        num_slices = 1024,   # number of random 1d directions the distribution is sliced along
        domain = (-5, 5),    # range of characteristic-function frequencies evaluated
        num_knots = 17       # quadrature points across `domain` for the trapezoid integral
    ):
        """SIGReg - push the embedding distribution toward an isotropic gaussian.

        Copied verbatim from `gpt_lejepa.LatentAutoregressive.sigreg_loss`, whose
        module header cites https://arxiv.org/abs/2511.08544. Sliced test: project
        `x` onto random unit directions, compare each slice's empirical
        characteristic function against that of a standard normal
        (exp(-t^2 / 2)), and integrate the gaussian-weighted squared error.

        Off by default here (`sigreg_loss_weight = 0.`).
        """

        dim, device = x.shape[-1], x.device

        # random unit directions to slice the high-dimensional distribution with

        rand_projs = torch.randn((num_slices, dim), device = device)
        rand_projs = l2norm(rand_projs)

        t = torch.linspace(*domain, num_knots, device = device)

        # characteristic function of N(0, 1), the reference to match against

        exp_f = (-0.5 * t.square()).exp()

        # project every sample onto every slice, then flatten all leading axes into
        # one sample axis: (... d) -> (n m)

        x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
        x_t = rearrange(x_t, '... m -> (...) m')

        # empirical characteristic function per slice: mean over samples of exp(i t x)
        # (n m 1) * (num_knots,) broadcasts to (n m num_knots), mean over n -> (m num_knots)

        x_t = rearrange(x_t, 'n m -> n m 1') * t
        ecf = (1j * x_t).exp().mean(dim = 0)

        # gaussian-weighted squared error, downweighting the noisy high-frequency tails

        err = ecf.sub(exp_f).abs().square().mul(exp_f)

        return torch.trapezoid(err, t, dim = -1).mean()

    def __init__(
        self,
        net,
        *,
        dim,
        num_rollouts = 1,                       # unroll depth; 1 is plain next-latent, >1 feeds predicted latents back in
        loss_type: str | Module = 'smooth_l1',  # 'smooth_l1' | 'mse' | 'mse_and_cosine_sim', or any Module - must be UNREDUCED, shape (r b n d)
        dynamics_type: str = 'residual',        # 'residual' is the paper's scheme (mlp on the delta); 'gru' trains but worse; 'custom' takes the module as-is
        dynamics_network: Module | None = None, # for 'residual' replaces the default mlp AND skips its zero-init; for 'custom' must accept (rollout_embeds, curr_latent)
        dynamics_hidden_dim = None,             # defaults to `dim`
        dynamics_num_layers = 3,                # 3-layer mlp is what the paper used
        next_latent_loss_weight = 1.0,          # 0. skips the latent regression entirely (loss is never computed)
        kl_loss_weight = 1.0,                   # 0. skips the logit-space kl entirely, including the frozen head deepcopy
        ignore_index = -100,                    # positions excluded from ce and from every rollout mask
        pad_value = 0,                          # ignore_index is not a valid token id, so inputs are rewritten to this before embedding
        rollout_weights: tuple[float, ...] | None = None,  # static per-step weights, renormalized to sum to 1; default uniform
        sigreg_loss_weight = 0.,                # off by default
        sigreg_loss_kwargs: dict = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        ),
        dynamic_rollout_loss_weight = True,     # additionally scale step r by how well steps < r are already doing (per token, recomputed every batch)
        dynamic_loss_decay = 1.0                # w_r = exp(-decay * sum of prior-step losses); larger cuts off deep steps more sharply
    ):
        """Build the dynamics model and cache which loss terms are active.

        `dim` must match the wrapped net's model dim, since the dynamics model
        operates directly on its final hidden states.
        """

        super().__init__()
        self.net = net
        self.ignore_index = ignore_index
        self.pad_value = pad_value

        assert num_rollouts > 0, 'num_rollouts must be greater than 0'
        self.num_rollouts = num_rollouts

        self.loss_type = loss_type

        # every branch keeps reduction = 'none' - the per-element loss is later scaled
        # by the rollout weights and masked-meaned, so reducing here would lose that

        if isinstance(loss_type, Module):
            self.loss_fn = loss_type
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction = 'none')
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction = 'none')
        elif loss_type == 'mse_and_cosine_sim':
            # https://arxiv.org/abs/2606.30534
            self.loss_fn = MSECosineSimLoss()
        else:
            raise ValueError(f'unknown loss type {loss_type}')

        self.next_latent_loss_weight = next_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

        # a zero weight is treated as "skip the term", not "compute then multiply by 0",
        # so the corresponding work is never done in forward

        self.has_next_latent_loss = next_latent_loss_weight > 0.
        self.has_kl_loss = kl_loss_weight > 0.
        self.has_sigreg = sigreg_loss_weight > 0.

        self.dynamic_rollout_loss_weight = dynamic_rollout_loss_weight
        self.dynamic_loss_decay = dynamic_loss_decay

        # rollout weights

        # normalized so they sum to 1, which is why the final `.sum()` over rollout
        # steps in forward is a weighted average and not a total that grows with
        # num_rollouts - changing num_rollouts leaves the latent loss on the same scale

        if not exists(rollout_weights):
            rollout_weights = (1.,) * num_rollouts

        assert len(rollout_weights) == num_rollouts
        rollout_weights = tensor(rollout_weights)
        self.register_buffer('rollout_loss_weights', rollout_weights / rollout_weights.sum(), persistent = False)

        # token embedding for next-token input to dynamics model

        # shared reference to the wrapped net's table, not a copy - the action embeds
        # are detached in forward so this loss never backprops into it through that path

        self.token_emb = net.token_emb

        # latent dynamics model

        assert dynamics_type in ('residual', 'gru', 'custom')

        if dynamics_type == 'residual':
            # default net: layernorm over the concatenated (latent, action) pair, then
            # an mlp producing the delta - input is 2 * dim, output is dim

            if not exists(dynamics_network):
                dynamics_hidden_dim = default(dynamics_hidden_dim, dim)
                dynamics_network = nn.Sequential(
                    nn.LayerNorm(dim * 2),
                    MLP(
                        dim_in = dim * 2,
                        dim_out = dim,
                        dim_hidden = dynamics_hidden_dim,
                        depth = dynamics_num_layers
                    )
                )

                # zero the mlp's final Linear ([-1] is the MLP, [-1][-1] its last layer) so
                # the residual transition starts as the identity and the rollout cannot
                # blow up at init; only applied to the default net, not a user-supplied one

                nn.init.zeros_(dynamics_network[-1][-1].weight)
                nn.init.zeros_(dynamics_network[-1][-1].bias)

            self.dynamics_model = ResidualDynamics(dynamics_network)

        elif dynamics_type == 'gru':
            self.dynamics_model = RolloutGRU(dim = dim)

        elif dynamics_type == 'custom':
            # used as-is, so it must already implement the (rollout_embeds, curr_latent)
            # -> (r b n d) contract, e.g. by wrapping the net in ResidualDynamics

            assert exists(dynamics_network)
            self.dynamics_model = dynamics_network

        # constant tensor stood in for disabled loss terms, so the Losses breakdown is
        # always tensors on the right device/dtype rather than a mix of tensor and float

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        **kwargs
    ):
        """One training step: cross entropy plus the multi-step latent rollout losses.

        Takes raw token ids (b, seq) and runs the transformer exactly once no matter how
        deep the unroll is - the whole rollout happens inside the small dynamics model on
        the already-computed hidden states.

        `kwargs` are forwarded untouched to the wrapped net. Returns the scalar total,
        or `(total, Losses)` when `return_loss_breakdown = True`.
        """

        seq, ignore_index, num_rollouts = x.shape[1], self.ignore_index, self.num_rollouts

        # prepare input and target

        # same convention as AutoregressiveWrapper: the full sequence goes in (needed
        # here, since positions past seq - 1 supply rollout targets) and the shift is
        # applied when slicing logits below
        # ignore_index (-100) is not a valid row of the embedding table, so it is
        # rewritten to pad_value before the lookup; the loss still skips those positions

        inp, target = x, x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # get hidden states from transformer

        # the post-final-norm representation feeding to_logits, i.e. the space this
        # wrapper regularizes; `intermediates` is unused

        hidden_states, intermediates = self.net(
            inp,
            return_embeddings_and_intermediates = True,
            **kwargs
        )

        # ce loss

        logits = self.net.to_logits(hidden_states[:, :-1])

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),  # cross_entropy wants classes on dim 1
            target,
            ignore_index = ignore_index
        )

        mask = target != ignore_index  # (b, seq - 1), True where the target is a real token

        # latent dynamics losses

        # the deepest rollout step reaches position n + num_rollouts, so the sequence
        # needs at least num_rollouts + 2 positions for any window to hold a real target

        assert seq > (num_rollouts + 1), f'sequence length ({seq}) must be greater than num_rollouts ({num_rollouts}) + 1'

        token_embeds = self.token_emb(inp)

        # the deepest step reads position n + num_rollouts, which runs off the end near the
        # tail; right-pad by num_rollouts - 1 so every sliding window is full length, and
        # let rollout_masks (padded with False) drop the fabricated entries from the means

        pad_amt = num_rollouts - 1

        padded_target_hiddens = pad_right_at_dim(hidden_states, pad_amt, dim = 1, value = 0.)
        padded_mask = pad_right_at_dim(mask, pad_amt, dim = 1, value = False)
        padded_token_embeds = pad_right_at_dim(token_embeds, pad_amt, dim = 1, value = 0.)

        # sliding windows of length seq - 1, stride 1, give exactly num_rollouts windows.
        # Tensor.unfold puts the window axis last, so the raw result is (b r d n) and the
        # rearrange moves it back to a leading rollout axis matching the dynamics model.
        # net effect: rollout_target_hiddens[r, b, n] == hidden_states[b, n + 1 + r]

        rollout_target_hiddens = padded_target_hiddens[:, 1:].unfold(1, seq - 1, 1)
        rollout_target_hiddens = rearrange(rollout_target_hiddens, 'b r d n -> r b n d')

        # mask is already shifted by one relative to x, so it is unfolded without the [:, 1:]
        # rollout_masks[r, b, n] gates whether position n + 1 + r is a real token

        rollout_masks = padded_mask.unfold(1, seq - 1, 1)
        rollout_masks = rearrange(rollout_masks, 'b r n -> r b n')

        # the "action" for step r at position n: the embedding of the token consumed to
        # move the state from n + r to n + 1 + r, i.e. token_embeds[b, n + 1 + r]

        rollout_next_token_embeds = padded_token_embeds[:, 1:].unfold(1, seq - 1, 1)
        rollout_next_token_embeds = rearrange(rollout_next_token_embeds, 'b r d n -> r b n d')

        # frozen copy of output head for kl loss

        # re-copied from the live head every forward pass, so it tracks training rather
        # than lagging like an EMA target network. requires_grad_(False) only stops the
        # *head* from being updated by the kl term - activations passed through it still
        # carry gradient back to the dynamics model and the encoder.
        # the .parameters() call requires to_logits to be a Module, i.e. the default
        # LinearNoBias, not the tied-embedding lambda variant of TransformerWrapper

        if self.has_kl_loss:
            frozen_to_logits = deepcopy(self.net.to_logits)
            for p in frozen_to_logits.parameters():
                p.requires_grad_(False)

        # start state for every rollout, deliberately NOT detached: with the targets and the
        # actions both stopped, this is the only path by which the rollout losses reach the
        # transformer, which is what makes them regularize its representations

        curr_latent = hidden_states[:, :-1]

        # actions detached, so nothing here trains the shared token embedding table
        # out: (r b n d), the predicted latents at positions n + 1 .. n + num_rollouts

        dynamics_out = self.dynamics_model(
            rollout_next_token_embeds.detach(),
            curr_latent
        )

        next_latent_loss = self.zero
        kl_loss = self.zero

        weights = self.rollout_loss_weights[:num_rollouts]
        step_masks_expanded = rearrange(rollout_masks, '... -> ... 1')  # (r b n) -> (r b n 1) to broadcast over d

        # compute latent loss

        # computed unconditionally because the dynamic weights below are derived from it
        # even when has_next_latent_loss is False. target is detached - a stop-gradient
        # that keeps the encoder from collapsing the latents to make itself predictable

        step_latent_loss_unreduced = self.loss_fn(
            dynamics_out,
            rollout_target_hiddens.detach()
        )

        # dynamic rollout loss weight

        # curriculum over rollout depth (added in ce4a20e): step r is discounted by how
        # badly steps 0..r-1 are currently doing, so deep predictions are only pushed once
        # the shallow ones are accurate. Fully detached - this shapes weights, not gradients.
        # exclusive cumsum along the rollout axis => cum[r] excludes step r's own loss, so
        # w_r = exp(-decay * cum[r]) = prod_{s<r} exp(-decay * L_s): step 0 always gets
        # exactly 1, and each later step keeps only what survives every earlier step.
        # (ebe0201 replaced the earlier sigmoid(-decay * (cum - threshold)) form, which gave
        # step 0 a weight below 1 and needed the extra `dynamic_loss_threshold` kwarg.)
        # granularity is per (rollout step, batch, position), i.e. an easy token can keep
        # training deep rollouts while a hard one in the same batch does not

        dynamic_weights = 1.

        if self.dynamic_rollout_loss_weight:
            step_latent_loss = reduce(step_latent_loss_unreduced.detach(), 'r b n d -> r b n', 'mean')
            dynamic_weights = dynamic_rollout_loss_weights(step_latent_loss, self.dynamic_loss_decay)

        # smooth l1 with stop-gradient on target

        if self.has_next_latent_loss:
            # broadcast the static per-step weights over (b n d)

            step_latent_loss_unreduced = einx.multiply('r b n d, r -> r b n d', step_latent_loss_unreduced, weights)

            if self.dynamic_rollout_loss_weight:
                step_latent_loss_unreduced = einx.multiply('r b n d, r b n -> r b n d', step_latent_loss_unreduced, dynamic_weights)

            # mean over (b n d) under the mask gives one scalar per rollout step; summing
            # those is a weighted average, since `weights` was normalized to sum to 1

            next_latent_loss = masked_mean(step_latent_loss_unreduced, step_masks_expanded, dim = (1, 2, 3)).sum()

        # kl divergence

        # a second, distribution-space view of the same rollout: decode both the predicted
        # and the true latent through the frozen head and match the resulting token
        # distributions. With a linear head this is blind to whatever the head projects
        # away, so it complements the raw regression above rather than duplicating it

        if self.has_kl_loss:
            with torch.no_grad():
                target_log_probs = F.log_softmax(frozen_to_logits(rollout_target_hiddens.detach()), dim = -1)

            pred_log_probs = F.log_softmax(frozen_to_logits(dynamics_out), dim = -1)

            # torch's kl_div takes the prediction as `input` and the reference as `target`,
            # so this is KL(target || pred); log_target = True since both are log-probs

            step_kl = F.kl_div(
                pred_log_probs,
                target_log_probs,
                log_target = True,
                reduction = 'none'
            )

            step_kl = einx.sum('... c -> ...', step_kl)  # sum over vocab: (r b n c) -> (r b n)
            step_kl = einx.multiply('r b n, r -> r b n', step_kl, weights)

            # reuse the same dynamic curriculum weights computed from the latent loss

            if self.dynamic_rollout_loss_weight:
                step_kl = step_kl * dynamic_weights

            kl_loss = masked_mean(step_kl, rollout_masks, dim = (1, 2)).sum()

        # sigreg regularization on embeddings

        # note: `mask` is (b, seq - 1) while `hidden_states` is (b, seq, d). This branch is
        # off by default (sigreg_loss_weight = 0.) and is not covered by tests/train_path_star.py

        sigreg_loss = self.zero

        if self.has_sigreg:
            sigreg_loss = self.sigreg_loss(hidden_states[mask], **self.sigreg_loss_kwargs)

        # total

        # ce is unweighted and always present; the auxiliaries are the tunable side terms

        total_loss = (
            ce_loss +
            self.next_latent_loss_weight * next_latent_loss +
            self.kl_loss_weight * kl_loss +
            self.sigreg_loss_weight * sigreg_loss
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, Losses(ce_loss, next_latent_loss, kl_loss, sigreg_loss)

# quick test

if __name__ == '__main__':
    from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8
        )
    )

    wrapper = NextLatentWrapper(
        model,
        dim = 512,
        num_rollouts = 3
    )

    token_ids = torch.randint(0, 256, (2, 1024))

    # forward and backwards on data as usual

    loss = wrapper(token_ids)
    loss.backward()

    # after much extract the original transformer
    # wrap it in the usual autoregressive wrapper and generate with the improved representation space

    # `model` is the very object the wrapper holds as `.net`, so nothing needs copying -
    # the dynamics model and the latent losses are simply dropped at inference time
    # (train_path_star.py does the same via `AutoregressiveWrapper(unwrapped_model.net)`)

    autoregressive_model = AutoregressiveWrapper(model)
    generated = autoregressive_model.generate(token_ids[:, :1], 10)

    print(generated.shape) # (2, 11)
