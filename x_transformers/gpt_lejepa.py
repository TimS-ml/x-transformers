"""
GPT trained LeJEPA-style - the trunk is trained to predict its own next hidden
state in latent space rather than the next token.

Randall Balestriero, Yann LeCun - https://arxiv.org/abs/2511.08544
Hai Huang - https://arxiv.org/abs/2509.14252v2

(titles, per the bibtex entries in this repo's README: "LeJEPA: Provable and
Scalable Self-Supervised Learning Without the Heuristics" and "LLM-JEPA: Large
Language Models Meet Joint Embedding Predictive Architectures")

A decoder transformer reads the token sequence as usual, but the loss lives on its
embeddings. A small MLP predictor maps the hidden state at position i - optionally
concatenated with an embedding of the token id consumed at that step, called the
"action" - to the network's own embedding at position i + 1. It is then unrolled
`num_rollouts` steps purely in latent space: the predictor is fed its own output
and the transformer is not re-run. Targets are the network's embeddings, detached
by default, which is the joint embedding predictive (JEPA) target branch.

Such an embedding prediction objective is trivially minimized by a constant
encoder, so the embeddings are additionally pushed towards an isotropic gaussian
by `sigreg_loss` (see the LeJEPA reference above): random unit directions
("slices") are drawn, embeddings are projected onto each one, and the empirical
characteristic function of those 1d projections is compared against exp(-t^2/2),
the characteristic function of N(0, 1), integrated over a grid of t.

Token-level cross entropy is still computed, but through an MLP probe on
embeddings whose gradient is scaled by `frac_gradient` (0. by default, i.e. fully
detached), so out of the box CE trains only the probe and the trunk is shaped by
the latent prediction + sigreg terms alone.
"""

from __future__ import annotations

import torch
from torch import stack
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

import einx
from einops import rearrange

from x_transformers import Decoder, TransformerWrapper, RMSNorm
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from torch_einops_utils import lens_to_mask, pad_right_at_dim, masked_mean
from torch_einops_utils.nn import Sequential

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, eps = 1e-6):
    return F.normalize(t, dim = -1, eps = eps)

def scale_grad(t, scale):
    """Straight-through gradient scaler - identity forward, gradient multiplied by `scale` backward.

    `t * scale + t.detach() * (1. - scale)` equals `t` in the forward pass, but only
    the first term carries gradient, so whatever is upstream of `t` receives `scale`
    of the incoming gradient. scale = 0. is a plain stop-gradient.
    """
    if scale == 1.:
        return t
    if scale == 0.:
        return t.detach()
    return t * scale + t.detach() * (1. - scale)

# classes

class LatentAutoregressive(Module):
    """Autoregression in embedding space - wraps a `TransformerWrapper` decoder.

    Loss terms, in the order they come back from `return_loss_breakdown`:

    - `ce`             - next token cross entropy through an MLP probe on the
                         embeddings, gradient-scaled by `frac_gradient` (0. = probe only)
    - `l2`             - mse between the rolled-out predictor latents and the network's
                         own (by default detached) embeddings one or more steps ahead,
                         with the next token id fed in as an "action"
    - `l2_no_action`   - same, from a separate predictor that is given no token id
    - `sigreg`         - isotropic gaussian regularization of the embeddings, which is
                         what stops the l2 terms being solved by a constant embedding

    Rollouts happen entirely in latent space - the predictor consumes its own output
    and the transformer is run once per forward.
    """

    @staticmethod
    def sigreg_loss(
        x,
        num_slices = 1024,      # number of random 1d directions ("slices") the embeddings are projected onto
        domain = (-5, 5),       # range of t (characteristic function argument) integrated over
        num_knots = 17          # grid points in that range, for the trapezoid rule
    ):
        """Isotropic gaussian regularizer by sliced characteristic function matching.

        For each random unit direction, compares the empirical characteristic
        function E[exp(i t z)] of the 1d projection z against exp(-t^2 / 2), the
        characteristic function of N(0, 1). The squared modulus of the difference is
        weighted by exp(-t^2 / 2) - which damps the tails of the grid where the
        empirical estimate is noisiest - integrated over t, then averaged over slices.

        Vanishes when every 1d projection looks standard normal, so it penalizes both
        anisotropy and collapse (a constant embedding gives a deterministic, hence
        maximally non-gaussian, projection).

        x: (..., d) - all leading dims are flattened into the sample dim.
        """
        dim, device = x.shape[-1], x.device

        # random directions, l2 normed to sit on the unit sphere

        rand_projs = torch.randn((num_slices, dim), device = device)
        rand_projs = l2norm(rand_projs)

        # integration grid, plus the standard normal characteristic function on it
        # exp_f serves twice: as the target and as the weighting function under the integral

        t = torch.linspace(*domain, num_knots, device = device)

        exp_f = (-0.5 * t.square()).exp()

        # project onto every slice, then fold batch/seq into a single sample dim
        # (... d) x (m d) -> (... m) -> (n m)

        x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
        x_t = rearrange(x_t, '... m -> (...) m')

        # outer product with the grid (n m) -> (n m t), then average over samples for
        # the empirical characteristic function per (slice, knot) -> complex (m t)

        x_t = rearrange(x_t, 'n m -> n m 1') * t
        ecf = (1j * x_t).exp().mean(dim = 0)

        # |ecf - exp(-t^2/2)|^2, windowed by the same gaussian

        err = ecf.sub(exp_f).abs().square().mul(exp_f)

        # trapezoid integrate over t per slice, then mean across slices -> scalar

        return torch.trapezoid(err, t, dim = -1).mean()

    def __init__(
        self,
        net,                                                    # a TransformerWrapper - its `to_logits` gets wrapped with the probe mlp below
        *,
        dim,
        sigreg_loss_weight = 0.05,                              # lam - convex mix between the latent prediction losses and sigreg
        l2_loss_weight = 1.,                                    # extra multiplier on the (1. - lam) weighted prediction losses
        num_rollouts = 1,                                       # steps the predictor is unrolled in latent space from each position
        rollout_loss_weights: tuple[float, ...] | None = None,  # per-step weights, normalized to sum to 1; defaults to uniform
        sigreg_loss_kwargs = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        ),
        frac_gradient = 0.,                                     # fraction of the ce probe gradient reaching the trunk; 0. = probe trains alone
        predict_next_cosine_sim = True,                         # l2norm preds and targets, turning the mse into a cosine distance (||a - b||^2 = 2 - 2cos)
        predictor_input_hiddens_index = -1,                     # which cached hidden feeds the predictor; -1 = residual stream entering the last attention block
        predict_next_embed_with_action = True,                  # predictor conditioned on the next token id
        predict_next_embed_no_action = False,                   # second, separate predictor that gets no token id
        detach_target = True,                                   # stop-gradient on the target embeddings (JEPA target branch)
        ce_probe_module = None, # i.e. extra transformer blocks
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.net = net

        self.ignore_index = ignore_index
        self.pad_value = pad_value

        self.sigreg_loss_weight = sigreg_loss_weight
        self.l2_loss_weight = l2_loss_weight

        self.num_rollouts = num_rollouts
        rollout_loss_weights = default(rollout_loss_weights, (1.,) * num_rollouts)

        assert len(rollout_loss_weights) == num_rollouts, f'rollout_loss_weights must be of length {num_rollouts}'

        # normalized so that summing the per-step losses in forward is a weighted average
        # note forward slices this to the usable number of rollouts without renormalizing

        rollout_weights = torch.tensor(rollout_loss_weights)
        self.register_buffer('rollout_loss_weights', rollout_weights / rollout_weights.sum(), persistent = False)

        self.sigreg_loss_kwargs = sigreg_loss_kwargs
        self.frac_gradient = frac_gradient
        self.predict_next_cosine_sim = predict_next_cosine_sim
        self.predictor_input_hiddens_index = predictor_input_hiddens_index

        assert predict_next_embed_with_action or predict_next_embed_no_action, 'at least one prediction head must be turned on'

        self.predict_next_embed_with_action = predict_next_embed_with_action
        self.predict_next_embed_no_action = predict_next_embed_no_action
        self.detach_target = detach_target

        # device/dtype correct scalar stand-in for whichever prediction losses are off

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        num_tokens = net.num_tokens

        # cross entropy linear probe - mlp on detached embeddings

        # note this wraps, rather than replaces, the wrapper's own to_logits, which stays
        # last in the chain as the unembedding; `Sequential` here drops None entries, so
        # the default ce_probe_module = None simply falls out

        net.to_logits = Sequential(
            ce_probe_module,
            RMSNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim),
            net.to_logits
        )

        # predict next embed given (embed, action)

        # the "action" is the ground truth token id at the transition being predicted;
        # its embedding is concatenated to the state, hence the dim * 2 input width

        if predict_next_embed_with_action:
            self.action_emb = nn.Embedding(num_tokens, dim)

            self.to_next_embed_pred = Sequential(
                RMSNorm(dim * 2),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim)
            )

        # predict next embed without sampled token id for that step

        if predict_next_embed_no_action:
            self.to_next_embed_no_action_pred = Sequential(
                RMSNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim)
            )

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        **kwargs
    ):
        """x: (b, n) token ids, with `ignore_index` marking positions excluded from every loss.

        Returns the scalar total loss, or `(total, (ce, l2, l2_no_action, sigreg))`
        when `return_loss_breakdown`. Disabled prediction heads report 0.
        """
        seq, ignore_index = x.shape[1], self.ignore_index

        # prepare input and target

        # the whole sequence goes in (the final position's embedding is needed as a
        # rollout target), and predictions are read off positions shifted by one

        inp, target = x, x[:, 1:]

        # ignore_index is not a valid embedding row - swap it for pad before the lookup;
        # `mask` below is what actually removes those positions from the losses

        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # get embeddings from network

        # return_embeddings stops before to_logits (the probe is applied explicitly below);
        # cache.hiddens is the residual stream captured just before each attention block

        embed, cache = self.net(
            inp,
            return_embeddings = True,
            return_intermediates = True,
            **kwargs
        )

        embed_prev = embed[:, :-1]

        # the predictor may be fed an earlier layer than the one being predicted
        # (default -1 = input to the final attention block, i.e. one block below `embed`)

        if exists(self.predictor_input_hiddens_index):
            embed_prev_lower = cache.hiddens[self.predictor_input_hiddens_index][:, :-1]
        else:
            embed_prev_lower = embed_prev

        mask = target != ignore_index      # (b, n - 1) valid target positions

        # cross entropy loss on detached embeddings (linear probe)

        # frac_gradient = 0. detaches entirely, so the probe reads the trunk without
        # steering it - the token objective becomes a readout, not a training signal

        embed_prev_ce = scale_grad(embed_prev, self.frac_gradient)
        logits = self.net.to_logits(embed_prev_ce)

        # (b n c) -> (b c n), cross_entropy wants the class dim second

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        # cannot roll out further than there are future positions to hit

        num_rollouts = min(self.num_rollouts, seq - 1)

        target_embeds = embed

        # stop-gradient on the target branch - gradient then reaches the trunk only
        # through the predictor's input hiddens, never through what it is chasing

        if self.detach_target:
            target_embeds = target_embeds.detach()

        # putting both sides on the unit sphere makes the mse below equal 2 - 2 * cosine sim

        if self.predict_next_cosine_sim:
            target_embeds = l2norm(target_embeds)

        # default l2 losses

        l2_loss = self.zero
        l2_no_action_loss = self.zero

        has_predictors = self.predict_next_embed_with_action or self.predict_next_embed_no_action

        if num_rollouts > 0 and has_predictors:

            # pad sequences on the right to handle future predictions smoothly
            # we pad exactly by num_rollouts - 1 to get exactly num_rollouts windows

            pad_amt = num_rollouts - 1

            # the padding is masked out by `padded_mask` (False), so its value is irrelevant

            padded_target_embeds = pad_right_at_dim(target_embeds, pad_amt, dim = 1, value = 0.)
            padded_mask = pad_right_at_dim(mask, pad_amt, dim = 1, value = False)

            # pre-compute targets and masks across all rollout steps using unfold

            # sliding windows of length n = seq - 1 over the one-shifted targets:
            # (b, seq + r - 2, d) -unfold-> (b, r, d, n) -> (r, b, n, d)
            # so step_targets[s, :, i] is the embedding at position i + 1 + s

            step_targets = rearrange(padded_target_embeds[:, 1:].unfold(1, seq - 1, 1), 'b r d n -> r b n d')

            # same windowing on the target mask: (b, n + r - 1) -unfold-> (b, r, n) -> (r, b, n)
            # (mask already lines up with the targets, so it is not sliced again)

            step_masks = rearrange(padded_mask.unfold(1, seq - 1, 1), 'b r n -> r b n')
            step_masks_expanded = rearrange(step_masks, '... -> ... 1')     # trailing 1 to broadcast over the feature dim

            # collect predictors

            predictors = []

            if self.predict_next_embed_with_action:
                padded_x = pad_right_at_dim(inp, pad_amt, dim = 1, value = self.pad_value)

                # token ids windowed exactly like step_targets, so actions[s, :, i] is the
                # teacher forced token id at position i + 1 + s

                actions = rearrange(padded_x[:, 1:].unfold(1, seq - 1, 1), 'b r n -> r b n')

                predictors.append((True, self.to_next_embed_pred))

            if self.predict_next_embed_no_action:
                predictors.append((False, self.to_next_embed_no_action_pred))

            # iterate over predictors and accumulate rollout losses

            for has_action, predictor in predictors:

                # the rollout starts from the trunk hidden and thereafter feeds on its own
                # output - the transformer is not re-run for steps beyond the first

                pred = embed_prev_lower
                pred_loss_inputs = []

                for step in range(num_rollouts):
                    pred_input = pred

                    # optionally condition on action

                    if has_action:
                        action_embed = self.action_emb(actions[step])
                        pred_input = torch.cat((pred, action_embed), dim = -1)

                    pred = predictor(pred_input)

                    # only the loss copy is normed; the raw `pred` is what carries into
                    # the next rollout step

                    pred_loss_input = l2norm(pred) if self.predict_next_cosine_sim else pred
                    pred_loss_inputs.append(pred_loss_input)

                # batched mse loss computation across all rollouts

                pred_loss_inputs = stack(pred_loss_inputs)      # r x (b, n, d) -> (r, b, n, d)

                step_losses = F.mse_loss(pred_loss_inputs, step_targets, reduction = 'none')

                # apply rollout weights

                # one weight per rollout step, broadcast over (b n d)

                weights = self.rollout_loss_weights[:num_rollouts]
                step_losses = einx.multiply('r b n d, r -> r b n d', step_losses, weights)

                # mean over valid tokens

                # average over valid (batch, position, feature) entries per rollout step -> (r,)
                # then sum, which is a weighted average since the weights sum to 1

                loss = masked_mean(step_losses, step_masks_expanded, dim = (1, 2, 3)).sum()

                if has_action:
                    l2_loss = loss
                else:
                    l2_no_action_loss = loss

        # sigreg regularization on embeddings

        # boolean indexing gathers only valid positions, (b, n, d) -> (num_valid, d)

        sreg_loss = self.sigreg_loss(embed_prev[mask], **self.sigreg_loss_kwargs)

        # when the predictor reads a lower layer, regularize that layer too and average,
        # so both ends of the prediction (input and target representation) stay isotropic

        if exists(self.predictor_input_hiddens_index):
            sreg_loss_lower = self.sigreg_loss(embed_prev_lower[mask], **self.sigreg_loss_kwargs)
            sreg_loss = (sreg_loss + sreg_loss_lower) / 2

        # total

        # lam is a convex mix between latent prediction and isotropy; ce enters unweighted
        # since by default it only trains the probe

        lam = self.sigreg_loss_weight

        total_loss = (
            ce_loss +
            (1. - lam) * l2_loss * self.l2_loss_weight +
            (1. - lam) * l2_no_action_loss * self.l2_loss_weight +
            lam * sreg_loss
        )

        if return_loss_breakdown:
            return total_loss, (ce_loss, l2_loss, l2_no_action_loss, sreg_loss)

        return total_loss

if __name__ == '__main__':
    from x_transformers import Decoder, TransformerWrapper

    transformer = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 1,
            heads = 8,
            rotary_pos_emb = True,
            pre_norm_has_final_norm = False
        )
    )

    model = LatentAutoregressive(
        transformer,
        dim = 512,
        num_rollouts = 4,
        rollout_loss_weights = [1., 0.5, 0.25, 0.125],
        predict_next_embed_no_action = True
    )

    x = torch.randint(0, 256, (2, 1024))

    loss = model(x)

    assert loss.ndim == 0, 'loss must be a scalar'
    loss.backward()
