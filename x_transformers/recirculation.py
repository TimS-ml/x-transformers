# recirculation - https://arxiv.org/abs/2608.17981
# full-bandwidth transformer - https://arxiv.org/abs/2608.08888v1

"""
Recirculation - give a feedforward decoder a recurrent state across tokens.

"Recirculation", Mozer, Siddiqui, Sawyer, Sanyal, Liu (bibtex `mozer2026recirculation`
in README.md), https://arxiv.org/abs/2608.17981. The second header link, "Full-bandwidth
transformer" (Wang et al., bibtex `wang2026fullbandwidthtransformer`), is related work
on feeding deep states back down; that scheme is implemented separately in
`full_bandwidth.py`.

A causal transformer carries nothing from one position to the next except its key /
values, and within a position information only moves upward through the stack, so the
serial computation available per token is bounded by depth. `train_parity_recirculation.py`
in the repo root frames it with running parity (cumsum mod 2), which needs recurrence a
plain transformer cannot supply beyond its depth. Recirculation adds that recurrence to
an unmodified `TransformerWrapper` decoder by feeding each token's deep residual state
back into a shallow layer before the next token is processed.

Per token t, against a kv cache:

1. first pass - token t runs through the whole stack. Its logits are read out here (the
   paper reads out after the first pass), and `cache.hiddens` records the residual stream
   entering every self attention layer.

2. mix (eq. 1) - for each route source -> destination (destination shallower), the deep
   state is rescaled to the destination's per-token L2 norm (eq. 2) and mixed in

       z_mixed = alpha * norm_matched(z_source) + beta * z_dest     (beta defaults to 1 - alpha)

   with alpha / beta fixed floats (alpha optionally ramped up over the first tokens), or
   produced per token and per feature by a small learned mlp (`RecirculationMixer`).

3. replay - token t's first pass key / values are stripped from every self attention
   cache, and token t is run again at the same position, with forward pre hooks swapping
   z_mixed in at each destination layer (see the note in `Recirculation.recirculate` on
   exactly what the hook replaces). The replayed key / values take the place of the first
   pass ones.

Later tokens therefore attend to the recirculated key / values of token t at and above
each destination, which is how state computed deep in the stack at step t reaches shallow
layers at step t + 1. The recurrence is strictly sequential, so there is no parallel
teacher forced pass: training and prompt prefill (`forward` without a cache) loop over the
sequence one token at a time, two network calls per token, and backprop runs through the
whole unrolled chain (nothing is detached). A decoding step (`forward` with a cache, which
is how `AutoregressiveWrapper.generate` calls its net) is one first pass plus one replay.
`Recirculation` forwards the attributes `AutoregressiveWrapper` reads, so it drops in as
that wrapper's `net`.

Layer indices count self attention layers only, 0-indexed in execution order, and name the
residual stream entering that attention layer (`destination_layer = 0` is the embedded
tokens entering the first attention layer).

`learn_parameters` fits the learned mixers alone by backprop through time on next token
loss, with the model's weights never stepped (section 4.6, appendix d.5).
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch.nn import LayerNorm, Linear, GELU, Sequential
import torch.nn.functional as F

from torch.optim import AdamW

from einops import rearrange

from x_transformers.x_transformers import LayerIntermediates

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def logit(p):
    """Inverse sigmoid, log(p / (1 - p)), as a python float - sets the mixer's output bias."""
    return torch.log(torch.tensor(p) / (1 - p)).item()

# leak residual from a deep layer (source) down to a shallow one (dest), alpha-convex mixture
# multiple routes as tuples, e.g. (7, 8) -> (2, 3); or learn per-token coefficients with a small mlp (adaptive, section 4.6)

def norm_matched(z_source: Tensor, z_dest: Tensor, eps = 1e-8):
    """Rescale each source vector to the L2 norm of the matching destination vector (eq. 2).

    z_source, z_dest: (..., d) -> (..., d). Direction comes from the source, magnitude
    from the destination, so the deep state arrives at the shallow layer's scale whatever
    the norm growth between the two layers. Gradient flows through both norms (nothing is
    detached).
    """

    # rescale source to match destination L2 norm (eq. 2)

    source_norm = z_source.norm(dim = -1, keepdim = True)   # (..., 1)
    dest_norm = z_dest.norm(dim = -1, keepdim = True)       # (..., 1)

    # a (near) zero source vector gets scale 0, i.e. contributes nothing, rather than
    # being blown up by a division by ~0

    scale = dest_norm / source_norm.clamp(min = eps)
    scale = scale.masked_fill(source_norm <= eps, 0.)

    return z_source * scale

# the small mlp that learns the mixture coefficients
# 2 hidden layer gelu mlp, layer norm at the input, sigmoid at the output, initialized at alpha = 0.1, beta = 0.9

class RecirculationMixer(Module):
    """Learned per token, per feature mixture coefficients for one route (section 4.6).

    Maps the pair (z_source, z_dest) to (alpha, beta) through a layer norm, a 2 hidden
    layer gelu mlp and a sigmoid, so each coefficient lies in (0, 1). alpha and beta are
    independent sigmoids - unlike the fixed default, nothing makes them sum to 1.

    Starts out as (near) the constant mixture init_alpha / init_beta for any input, see
    the output layer init below.
    """

    def __init__(
        self,
        dim,                # model width d; the mlp reads 2d (source ++ dest) and emits 2d (alpha ++ beta)
        hidden = None,      # width of both hidden layers, defaults to dim
        init_alpha = 0.1,   # starting weight on the norm matched source state
        init_beta = 0.9     # starting weight on the destination's own state
    ):
        super().__init__()
        hidden = default(hidden, dim)

        # layer norm over the concatenated pair, so the mlp input scale does not depend on
        # how large the two residual streams are

        self.input_norm = LayerNorm(dim * 2)

        self.mlp = Sequential(
            Linear(dim * 2, hidden),
            GELU(),
            Linear(hidden, hidden),
            GELU(),
            Linear(hidden, dim * 2)
        )

        out_layer = self.mlp[-1]

        # init outputs at the desired alpha / beta

        # sigmoid(logit(p)) = p, and the small weight keeps the input dependent part near
        # zero, so the first d outputs start at ~init_alpha and the last d at ~init_beta

        nn.init.normal_(out_layer.weight, std = 0.02)
        nn.init.zeros_(out_layer.bias)

        with torch.no_grad():
            out_layer.bias.copy_(torch.cat((
                torch.full((dim,), logit(init_alpha)),
                torch.full((dim,), logit(init_beta))
            )))

    def forward(self, z_source: Tensor, z_dest: Tensor):
        """z_source, z_dest: (b, n, d) -> alpha, beta: (b, n, d) each (n = 1 when called from `Recirculation.mix`)."""

        concat = torch.cat((z_source, z_dest), dim = -1)            # (b, n, 2d)

        coefficients = self.mlp(self.input_norm(concat)).sigmoid()   # (b, n, 2d), in (0, 1)
        alpha, beta = coefficients.chunk(2, dim = -1)                # first half alpha, second half beta

        return alpha, beta

# recirculation wrapper
# drop in net for the autoregressive wrapper

class Recirculation(Module):
    """Wraps a decoder `TransformerWrapper` so that every token is recirculated (see the module docstring).

    Drop-in `net` for `AutoregressiveWrapper`: it forwards the attributes that wrapper
    reads and follows its calling convention - `forward(x)` without a cache for training
    or the prompt, `forward(x, cache = cache, return_intermediates = True)` per decoding
    step.

    Asserted requirements: the net can cache key / values (e.g. no memory tokens, no
    recycling), is not looped, and has a `to_logits` head (checked on forward).

    Routes are given as parallel tuples, e.g. source_layer = (2, 3), destination_layer =
    (0, 1); each destination must be shallower than its source, and destinations must be
    unique since the mixtures are keyed by destination. A scalar alpha / beta applies to
    every route.

    The only new parameters are the optional learned mixers; with fixed coefficients the
    wrapper adds none.
    """

    def __init__(
        self,
        net,                        # TransformerWrapper around a causal Decoder, kv cacheable and not looped
        source_layer,               # attention layer ordinal (attention layers only, 0-indexed) whose input stream is read; int or tuple of routes
        destination_layer,          # shallower attention layer ordinal whose input receives the mixture; int or tuple, one per source
        alpha = 0.15,               # fixed weight on the norm matched source state; float or per-route tuple, unused when use_learned
        beta = None,                # fixed weight on the destination state; None means 1 - alpha (after ramping), keeping it convex
        ramp_steps = 10,            # ramp up alpha over the first 10 tokens (appendix b.3), 0 to disable
        use_learned = False,        # learn per-token alpha / beta with a small mlp (section 4.6)
        learned_hidden = None       # hidden width of each learned mixer mlp, defaults to the model dim
    ):
        super().__init__()

        self.net = net

        assert self.net.can_cache_kv, 'recirculation requires the wrapped transformer to be able to cache key / values during decoding'
        assert not self.net.looped, 'recirculation is not compatible with a looped transformer'

        source_layers = cast_tuple(source_layer)
        destination_layers = cast_tuple(destination_layer)

        assert len(source_layers) == len(destination_layers), 'number of source and destination layers must match'
        assert len(set(destination_layers)) == len(destination_layers), 'destination layers must be unique'

        num_routes = len(source_layers)

        assert all(dest < src for src, dest in zip(source_layers, destination_layers)), 'each destination layer must be shallower than its source layer'

        self.source_layers = source_layers
        self.destination_layers = destination_layers

        # one alpha / beta per route; a None beta is resolved to 1 - alpha in `mix_coefficients`

        self.alphas = cast_tuple(alpha, num_routes)
        self.betas = cast_tuple(beta, num_routes) if exists(beta) else (None,) * num_routes

        # the AttentionLayers stack, whether net is a TransformerWrapper (net.attn_layers) or the stack itself

        attn_layers = getattr(net, 'attn_layers', net)
        dim = attn_layers.dim

        # one mixer per route, (2d -> hidden -> hidden -> 2d)

        self.learned_mixers = ModuleList([RecirculationMixer(dim, learned_hidden) for _ in range(num_routes)]) if use_learned else None

        # ramping only for fixed coefficients; learned mixer sets the mixture per token
        self.ramp_steps = ramp_steps if not exists(self.learned_mixers) else 0

        self.attn_layers = attn_layers

        # attention layer ordinal -> index into attn_layers.layers, walking the execution order
        # and keeping self attention ('a') entries only. AttentionLayers appends to `hiddens`
        # in the same order, so hiddens[i] is the stream entering layers[attn_layer_indices[i]]
        # e.g. default Decoder layer_types (a, f, a, f, a, f) -> [0, 2, 4]

        # note: if layers_execute_order revisits a layer (weight sharing), that module sits at
        # several ordinals, and the hook registered in `recirculate` fires on every visit, so
        # z_mixed is injected at each of them, not only at the chosen destination

        self.attn_layer_indices = [ind for ind in attn_layers.layers_execute_order if attn_layers.layer_types[ind] == 'a']

        num_attn_layers = len(self.attn_layer_indices)

        for source, destination in zip(source_layers, destination_layers):
            assert source < num_attn_layers, f'source layer {source} is out of range for a model with {num_attn_layers} attention layers'
            assert destination < num_attn_layers, f'destination layer {destination} is out of range for a model with {num_attn_layers} attention layers'

    # attributes needed by the autoregressive wrapper

    # plain pass-throughs to the wrapped net: max_seq_len and add_continuous_pred_head are
    # read in AutoregressiveWrapper.__init__, can_cache_kv / can_cache_kv_outside_max_seq_len /
    # looped during generate, output_is_log_prob to pick the training loss

    @property
    def max_seq_len(self):
        return self.net.max_seq_len

    @property
    def add_continuous_pred_head(self):
        return self.net.add_continuous_pred_head

    @property
    def can_cache_kv(self):
        return self.net.can_cache_kv

    @property
    def can_cache_kv_outside_max_seq_len(self):
        return self.net.can_cache_kv_outside_max_seq_len

    @property
    def looped(self):
        return self.net.looped

    @property
    def output_is_log_prob(self):
        return self.net.output_is_log_prob

    # mixture coefficients for a route, fixed or learned

    def mix_coefficients(self, route, z_source: Tensor, z_dest: Tensor, step):
        """(alpha, beta) for one route at sequence position `step`.

        Learned: the route's mixer applied to (z_source, z_dest), two (b, 1, d) tensors.
        Fixed: two python floats, and `step` only matters for the ramp.
        """

        if exists(self.learned_mixers):
            return self.learned_mixers[route](z_source, z_dest)

        alpha = self.alphas[route]

        # linear warmup: alpha is 0 at step 0 and reaches its full value at step = ramp_steps

        if self.ramp_steps > 0:
            alpha = alpha * min(step / self.ramp_steps, 1.)

        # default beta follows the ramped alpha, keeping the mixture convex (so z_mixed = z_dest
        # at step 0); an explicit beta is not ramped, so at step 0 z_mixed = beta * z_dest

        beta = default(self.betas[route], 1. - alpha)

        return alpha, beta

    # the core recirculation mixture (eq. 1)
    # returns the mixed residual stream for each destination layer

    def mix(self, hiddens: list[Tensor], step = 0):
        """The recirculation mixture (eq. 1) for every route, from one first pass's `hiddens`.

        hiddens: one (b, n, d) tensor per attention layer - the residual stream entering it -
        of which only the latest position is used. Returns {destination: z_mixed (b, 1, d)}.

        Every route reads its source from the first pass, so one route's injection never
        feeds another route's source. Gradient flows into both the source and destination
        states, and into the mixers when learned.

        Assumes hiddens has one entry per attention layer in execution order, which holds
        unless training-time layer dropout skipped one (skipped layers append nothing).
        """

        z_mixed = {}

        for route, (source, destination) in enumerate(zip(self.source_layers, self.destination_layers)):

            # latest token only, (b, n, d) -> (b, 1, d)

            z_source = hiddens[source][..., -1:, :]
            z_dest = hiddens[destination][..., -1:, :]

            alpha, beta = self.mix_coefficients(route, z_source, z_dest, step)

            z_mixed[destination] = alpha * norm_matched(z_source, z_dest) + beta * z_dest

        return z_mixed

    # strip the latest token's kv from the cache

    def _strip_latest_kv(self, cache: LayerIntermediates):
        """Drop the newest position from every self attention kv cache and decrement cache_length.

        Mutates `cache` in place (its Intermediates and cache_length), no copy is made.
        Only key / values are rewound: other state carried in the cache, such as a hybrid
        module's `hybrid_hidden`, stays as the first pass left it.
        """

        for attn_intermediate in cache.attn_intermediates:
            if attn_intermediate.layer_type != 'a':
                continue

            # cut along the sequence dim, (b, h, n, d_head) -> (b, h, n - 1, d_head)

            attn_intermediate.cached_kv = tuple(kv[..., :-1, :] for kv in attn_intermediate.cached_kv)

        cache.cache_length -= 1

    # recirculate the latest token
    # mix source / destination residuals (eq. 1), replay through the upper stack, replacing its kv

    def recirculate(
        self,
        x: Tensor,
        cache: LayerIntermediates,
        seq_start_pos = None,
        kwargs: dict = dict(),
        step = 0
    ):
        """Recirculate the latest token: mix (eq. 1), strip its first pass kv, replay it with the mixture injected.

        x: (b, n) token ids; only x[:, -1:] is replayed. cache: the LayerIntermediates
        of the first pass on that token - it must carry `hiddens` (net called with
        return_intermediates = True) and is mutated in place by the strip. seq_start_pos
        and kwargs are forwarded to the replay call; step feeds the alpha ramp.

        Returns the replay's LayerIntermediates, whose kv caches hold the recirculated key /
        values for the latest token (cache_length back to what it was). The replay's own
        logits are discarded - read out happens on the first pass.

        The whole stack is re-run for the token, not only the upper part: layers below the
        shallowest destination see unchanged inputs and recompute the same key / values as
        the first pass (given no dropout), so the effective change starts at the destination.
        Everything stays in the autograd graph.
        """

        hiddens = cache.hiddens

        assert exists(hiddens), 'cache with hiddens required - the net must be called with `return_intermediates = True`'

        # absolute position of the latest token (cache_length still counts it at this point)

        position = cache.cache_length - 1

        z_mixed_by_destination = self.mix(hiddens, step = step)

        # strip the first pass kv so the recirculated one replaces it on replay

        self._strip_latest_kv(cache)

        # inject the mixed residual at each destination layer before replay

        hook_handles = []

        for destination, z_mixed in z_mixed_by_destination.items():

            # each entry of attn_layers.layers is (norms, block, residual_fn), with
            # norms = (pre_branch_norm, post_branch_norm, post_main_norm)

            block_index = self.attn_layer_indices[destination]
            norms, block, _ = self.attn_layers.layers[block_index]

            # hook the pre-branch norm when there is one (pre-norm), else the attention block
            # itself. a forward pre hook only sees positional args, here the (b, n, d) input

            # note: that input is the attention branch input, not the residual stream itself -
            # `residual_fn.prepare` has already saved the unmixed stream as the skip. so after
            # the destination attention the stream is z_dest + attn(norm(z_mixed)), rather than
            # z_mixed + attn(norm(z_mixed)); the mixture reaches the upper stack through that
            # attention output, and later tokens through the destination's key / values. the
            # replay's `hiddens` also still record the unmixed z_dest

            pre_branch_norm = norms[0]
            hook_target = pre_branch_norm if exists(pre_branch_norm) else block

            # `z_mixed = z_mixed` binds this iteration's tensor - a plain closure would see
            # only the last value of the loop variable

            def inject_z_mixed(module, input, z_mixed = z_mixed):
                (residual,) = input

                # swap the last position for z_mixed. with a cache the replay input is a single
                # token (AttentionLayers keeps only the last position), so this replaces all of
                # (b, 1, d); the cat form would leave any earlier positions untouched

                residual = torch.cat((residual[..., :-1, :], z_mixed), dim = -2)
                return (residual,)

            hook_handles.append(hook_target.register_forward_pre_hook(inject_z_mixed))

        # the finally removes the hooks even if the replay raises, so the net is never left hooked

        try:
            # replay only the latest token, at its original position

            device = x.device

            # `pos` pins the absolute positional embedding to the original position - with the
            # default input_not_include_cache = False the wrapper would otherwise place a lone
            # token at position 0

            # note: `pos` is also forwarded to every attention layer, where ALiBi
            # (`alibi_pos_bias`) builds its bias from it. a single position gives a 1 x 1
            # zero-distance bias that broadcasts over all keys, so under ALiBi the replay
            # attends without any distance penalty

            replay_pos = torch.tensor([position], device = device, dtype = torch.long)
            replay_kwargs = dict(pos = replay_pos)

            # given `pos`, AttentionLayers would also derive rotary from that single position,
            # so rotary is passed explicitly. keys are cached unrotated and rotated after being
            # concatenated with the new one, so the frequencies must span every key position,
            # 0 .. position (the cat equals arange(position + 1)); queries and keys each take
            # the trailing slice they need, which also holds after cache truncation

            rotary_pos_emb_module = self.attn_layers.rotary_pos_emb

            if exists(rotary_pos_emb_module):
                rotary_replay_pos = torch.cat((
                    torch.arange(position, device = device),
                    torch.tensor([position], device = device, dtype = torch.long)
                ))

                replay_kwargs.update(rotary_pos_emb = rotary_pos_emb_module(rotary_replay_pos))

            # note: input_not_include_cache is not passed, so with the wrapper default the
            # attention layers get seq_pos_offset = 0 and build the left padding mask from
            # seq_start_pos over this single token - (b, 1) instead of (b, position + 1). that
            # one entry broadcasts over every key, so rows with seq_start_pos > 0 (left padded
            # prompts) have all keys masked out during the replay; unpadded rows are unaffected

            _, replay_cache = self.net(
                x[:, -1:],
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs,
                **replay_kwargs
            )
        finally:
            for hook_handle in hook_handles:
                hook_handle.remove()

        return replay_cache

    # prefill runs token by token (cannot be parallelized per paper)
    # decoding is a single pass on the latest token, followed by recirculation

    def forward(
        self,
        x: Tensor,
        cache: LayerIntermediates | None = None,
        return_intermediates = False,
        seq_start_pos = None,
        **kwargs
    ):
        """Two modes, matching how `AutoregressiveWrapper` calls its net.

        - no cache (training, or the prompt at the start of generate): `_prefill`, token by
          token with a recirculation after each. x: (b, n) -> logits (b, n, c).
        - cache (one decoding step): x is the full (b, n) sequence so far, cached prefix
          included, of which the net keeps only the last position (default
          input_not_include_cache = False). One first pass on that token, then `recirculate`.
          -> logits (b, 1, c), from the first pass.

        Returns logits, or (logits, cache) with return_intermediates, where cache is the
        post-recirculation LayerIntermediates to hand back on the next step. Nothing is
        detached; extra kwargs go to every net call.
        """

        assert exists(self.net.to_logits), 'the wrapped network needs a to_logits head'

        if not exists(cache):
            return self._prefill(x, return_intermediates, seq_start_pos, kwargs)

        # decoding

        logits, new_cache = self.net(
            x,
            return_intermediates = True,
            cache = cache,
            seq_start_pos = seq_start_pos,
            **kwargs
        )

        # step = absolute position of the new token, the same indexing as t in `_prefill`

        new_cache = self.recirculate(x, new_cache, seq_start_pos, kwargs, step = new_cache.cache_length - 1)

        return (logits, new_cache) if return_intermediates else logits

    def _prefill(self, x: Tensor, return_intermediates, seq_start_pos, kwargs: dict):
        """Sequential pass over x: (b, n) -> logits (b, n, c), plus the final cache if asked.

        Each token gets a first pass against the running cache, its logits are read from
        that pass, and it is recirculated before the next token sees the cache - 2n
        sequential net calls. The autograd graph spans the whole chain, so training
        backpropagates through time across tokens and memory grows with n.
        """

        net = self.net

        logits = []
        cache = None

        for t in range(x.shape[1]):
            token = x[:, t:(t + 1)]     # (b, 1)

            # only the new token is fed, hence input_not_include_cache = True, which also
            # offsets its positions (and the left padding mask) by cache_length

            _, cache = net(
                token,
                return_intermediates = True,
                cache = cache,
                input_not_include_cache = True,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            # first pass readout (per paper), then recirculate

            # last_hidden is the final normed output, (b, 1, d) -> logits (b, 1, c)

            # note: only `net.to_logits` is applied, so post-processing TransformerWrapper does
            # around it (sigsoftmax, mixture of softmax, multiple output heads, to_logits_kwargs)
            # is absent from prefill logits, while the decoding branch of `forward` returns the
            # wrapper's own logits. for a default wrapper the two agree

            hidden = cache.last_hidden[:, -1:]
            logits.append(net.to_logits(hidden))

            cache = self.recirculate(token, cache, seq_start_pos, kwargs, step = t)

        logits = torch.cat(logits, dim = -2)

        return (logits, cache) if return_intermediates else logits

    # train learned mixer with bptt on next token prediction loss, model weights frozen (section 4.6, appendix d.5)

    def learn_parameters(
        self,
        x: Tensor,
        steps = 100,
        optimizer_cls = AdamW,
        optimizer_kwargs = dict(lr = 3e-4, weight_decay = 1e-4),
        batch_size = 32
    ):
        """Fit only the learned mixers by backprop through time on next token cross entropy.

        x: (b, n) token ids, a pool that `batch_size` rows are sampled from (with
        replacement) at every one of `steps` steps. Each minibatch goes through the
        sequential `forward`, so the gradient reaching the mixers spans the whole unrolled
        chain of recirculations. Token id 0 is treated as padding and ignored in the loss
        (hard-coded ignore_index = 0).

        The optimizer covers `learned_mixers` only and the net is put in eval mode for the
        duration (its mode is restored after), so the model's weights are never stepped.

        note: they are not frozen through requires_grad either - backward still accumulates
        .grad on the net's parameters, and optim.zero_grad only clears the mixers', so those
        grads build up over the steps and are left in place afterwards.

        Returns the last step's loss as a float (steps must be >= 1, or `loss` is unbound).
        """

        assert exists(self.learned_mixers), 'learned mixer must be turned on for `learn_parameters`'

        x = x.contiguous()

        batch, device = x.shape[0], x.device

        labels = x[:, 1:]       # (b, n - 1) next token targets for the whole pool

        optim = optimizer_cls(self.learned_mixers.parameters(), **optimizer_kwargs)

        was_training = self.net.training
        self.net.eval()

        for _ in range(steps):
            rand_indices = torch.randint(0, batch, (batch_size,), device = device)
            batch_x = x[rand_indices]

            # prefill path, (batch_size, n, c); the last position has no target, drop it

            logits = self.forward(batch_x)[:, :-1]

            # (b n c) -> (b c n), cross_entropy wants the class dim second

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                labels[rand_indices],
                ignore_index = 0
            )

            loss.backward()
            optim.step()
            optim.zero_grad()

        self.net.train(was_training)

        return loss.item()
