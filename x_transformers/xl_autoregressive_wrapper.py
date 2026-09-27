"""Transformer-XL segment memory + end-to-end test-time training (TTT).

Two mechanisms live in this file.

1. Transformer-XL segment memory. `XLAutoregressiveWrapper` chops a sequence
   longer than `net.max_seq_len` into segments, runs them left to right, and
   threads the per-layer hidden states of segment i into segment i+1 as extra
   keys / values (`mems`). The bookkeeping itself lives in the wrapped
   `TransformerWrapper`: under `return_mems = True` it concatenates the incoming
   mems with the new hiddens and keeps only the trailing `max_mem_len` positions
   (x_transformers.py, the `if return_mems:` block). `detach_mems` there decides
   whether gradient may cross a segment boundary.

2. End-to-end test-time training (e2e TTT). Selected submodules are swapped for
   `TTTModuleWrapper`s that carry a *per-batch-element* copy of their parameters
   ("fast weights"). After each segment the segment loss is differentiated w.r.t.
   those fast weights and an SGD step (optionally Muon-orthogonalized) is
   applied, so the model adapts to the sequence while reading it. In `forward`
   the inner step is built with `create_graph = True`, so the outer backward
   differentiates *through* the inner update - that is the "end-to-end" part,
   and it is what lets the outer optimizer learn weights that are good to
   test-time-train from. The citation added to README.md alongside this feature
   is "End-to-End Test-Time Training for Long Context" (Tandon et al., arXiv
   2512.23675).

Runnable example: `train_xl_enwik8.py` at the repo root. `--e2e_ttt` turns TTT on
and points `ttt_module_paths` at a single value projection near the top of the
decoder; `--ttt_custom_loss` additionally swaps the TTT objective for
`TTTMetaLearningTargetKLLoss`.
"""

from __future__ import annotations
from math import ceil
from functools import reduce
from contextlib import nullcontext

import torch
from torch import nn, cat
from torch.nn import Module
import torch.nn.functional as F
from torch.func import functional_call, vmap  # per-sample "fast weight" evaluation, see TTTModuleWrapper

from einops import rearrange, repeat, pack, unpack
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator, cast_tuple  # top_p is unused here, imported so callers can pass it as generate's filter_logits_fn

from torch_einops_utils import masked_mean, mask_after, maybe, tree_map_tensor
from torch_einops_utils.device import module_device

# helper functions

def exists(val):
    """
    Check if a value exists (is not None).

    Args:
        val: The value to check

    Returns:
        bool: True if val is not None, False otherwise
    """
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_empty(arr):
    return len(arr) == 0

def first(it, default = None):
    # first element of any iterable, without raising on an empty one
    return next(iter(it), default)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def get_module_by_path(model, path):
    # resolve a dotted attribute path ('attn_layers.layers.0.1.to_v') to a submodule.
    # nn.ModuleList / nn.Sequential expose integer indices as attributes, so plain
    # getattr chaining works for the numeric segments too
    return reduce(getattr, path.split('.'), model)

def set_module_by_path(model, path, new_mod):
    # in-place surgery: replace the submodule at a dotted path (used to slot a
    # TTTModuleWrapper in where the original module was)
    *parts, last = path.split('.')
    mod = reduce(getattr, parts, model)
    setattr(mod, last, new_mod)

# tensor functions

def repeat_batch(t, batch_size):
    # (...) -> (b ...). used to give every batch element its own copy of a parameter.
    # this is a differentiable broadcast, so gradient reaching the per-sample copies
    # still flows back (summed over b) into the original nn.Parameter
    return repeat(t, '... -> b ...', b = batch_size)

def pack_one_with_inverse(t, pattern):
    # flatten leading dims into one axis, returning the closure that restores them
    packed, ps = pack([t], pattern)
    def inverse(t_in):
        return unpack(t_in, ps, pattern)[0]
    return packed, inverse

def transpose(t):
    # swap the last two (matrix) axes, leaving any leading batch axes alone
    return rearrange(t, '... i j -> ... j i')

# Muon update rule (Newton-Schulz) - Keller Jordan et al.

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)  # quintic coefficients as used by the Muon implementation credited above
):
    """Approximately orthogonalize a batch of gradient matrices (Muon's core op).

    Runs the quintic Newton-Schulz iteration `t <- a*t + (b*A + c*A@A)@t` with
    `A = t @ t^T`, which drives the singular values of `t` toward 1 while keeping
    its singular vectors. The input is first normalized by its Frobenius norm so
    the iteration stays in the region where those fixed coefficients converge.

    Args:
        t: gradient tensor. Here it is a *batched* fast-weight gradient of shape
           (batch, ..., rows, cols) - the leading axes are just carried along.

    Returns:
        (tensor, applied) where `applied` says whether orthogonalization actually
        ran. `t.ndim < 3` bails out with the tensor untouched: with the per-sample
        batch axis prepended, ndim 2 means a per-sample *vector* (a bias), which
        has no matrix structure to orthogonalize. The caller uses this flag to
        pick `ttt_muon_lr` vs `ttt_lr` per parameter.
    """
    if t.ndim < 3:
        return t, False

    rows, cols = t.shape[-2:]
    should_transpose = rows > cols

    # each step forms A = t @ t^T, which is (rows x rows), so flip to the wide
    # orientation first to keep A the smaller of the two gram matrices, then flip back

    if should_transpose:
        t = transpose(t)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ transpose(t)
        B = b * A + c * A @ A
        t = a * t + B @ t

    t = inv_pack(t)

    if should_transpose:
        t = transpose(t)

    return t, True

# episodic memories

class EpisodicMemories(Module):
    """A learned bank of key / value vectors, one bank per self-attention layer.

    Shape is (2, depth, heads, seq_len, dim_head) - the leading 2 is the (key,
    value) split, and each layer indexes its own slice along `depth`. These are
    ordinary parameters of the model, but the point of them here is to be a
    *dedicated, long-term* TTT target: rather than perturbing a projection weight,
    the inner loop can write into memory slots that attention reads from directly,
    which is what `ttt_module_paths = ('episodic_mems',)` selects.

    A trivial `forward()` (no arguments) exists so the bank can be swapped for a
    `TTTModuleWrapper`, which only intercepts module calls.
    """

    def __init__(
        self,
        depth,      # one bank per self-attention layer
        heads,      # the net's key/value head count, which under GQA is fewer than its query heads
        seq_len,    # number of memory slots per layer
        dim_head
    ):
        super().__init__()
        self.mem_kv = nn.Parameter(torch.randn(2, depth, heads, seq_len, dim_head))
        nn.init.normal_(self.mem_kv, std = 0.02)  # overwrites the randn above with a small-std init

    def forward(self):
        return self.mem_kv

class EpisodicMemoryWrapper(Module):
    """Wraps a `TransformerWrapper` so every self-attention layer sees an extra
    learned kv bank prepended to its keys / values.

    The bank is delivered through the `self_attn_additional_kv` kwarg, which
    `AttentionLayers` routes to the topmost self-attention layers and `Attention`
    concatenates in front of the real keys / values. Exactly one entry per
    self-attention layer is produced here, so in practice every layer gets one.
    """

    def __init__(
        self,
        net,
        episodic_mem_len
    ):
        super().__init__()
        self.net = net

        # probe the net once with a single dummy token to discover its attention
        # geometry (number of self-attn layers, kv head count, head dim) instead of
        # asking the caller to restate it

        with torch.no_grad():
            mock_input = torch.zeros((1, 1), dtype = torch.long, device = module_device(net))
            _, intermediates = net(mock_input, return_intermediates = True)

        attn_intermediates = [interm for interm in intermediates.attn_intermediates if interm.layer_type == 'a']  # 'a' = self attention (vs 'c' cross, 'f' feedforward)

        assert not is_empty(attn_intermediates), 'no attention layers found for episodic memory wrapper'

        first_cached_kv = first(attn_intermediates).cached_kv[0]  # cached key, shape (b, kv_heads, n, dim_head)
        _, kv_heads, _, dim_head = first_cached_kv.shape
        depth = len(attn_intermediates)

        self.episodic_mems = EpisodicMemories(
            depth = depth,
            heads = kv_heads,
            seq_len = episodic_mem_len,
            dim_head = dim_head
        )

    def get_additional_kv(self, batch_size):
        mem_kv = self.episodic_mems()

        # ndim 5 == (2, depth, heads, n, dim_head), i.e. one bank shared by the whole
        # batch, so broadcast it. ndim 6 means the bank went through a TTTModuleWrapper
        # and vmap already produced a per-sample batch axis - do not repeat it again

        if mem_kv.ndim == 5:
            mem_kv = repeat_batch(mem_kv, batch_size)

        # (b, 2, depth, ...) -> split key/value, then unbind depth into a per-layer list
        # of (k, v), each (b, heads, n, dim_head) as `additional_key_values` expects

        mem_k, mem_v = mem_kv.unbind(dim = 1)
        return [(k, v) for k, v in zip(mem_k.unbind(dim = 1), mem_v.unbind(dim = 1))]

    def forward(self, x, **kwargs):
        batch = x.shape[0]

        kwargs.update(
            self_attn_additional_kv = self.get_additional_kv(batch)
        )

        return self.net(x, **kwargs)

# low rank linear module

class LowRankLinear(Module):
    """ replace a module's weight matrix with two low rank matrices, which hold the ttt memories instead of the weight itself """

    def __init__(
        self,
        module,
        low_rank_dim
    ):
        super().__init__()

        assert module.weight.ndim == 2, 'low rank only supported for 2 dimensional weight matrices'

        out_dim, in_dim = module.weight.shape

        assert low_rank_dim < min(out_dim, in_dim), f'low rank dim ({low_rank_dim}) must be less than the minimum of the input ({in_dim}) and output ({out_dim}) dimensions of the weight matrix'

        # two low rank matrices, no weight matrix, the memories will be stored in these
        # the memory is initially zero, but one of the matrices is randomly initialized so the gradients can flow into it

        self.weight_a = nn.Parameter(torch.randn(out_dim, low_rank_dim))
        self.weight_b = nn.Parameter(torch.zeros(low_rank_dim, in_dim))

        # carry over the bias, if it exists

        self.bias = None

        if exists(module.bias):
            self.bias = nn.Parameter(module.bias.detach().clone())

    def forward(self, x):
        weight = self.weight_a @ self.weight_b
        return F.linear(x, weight, self.bias)

# ttt module wrapper

class TTTModuleWrapper(Module):
    """Transparent stand-in for a submodule that can run on per-sample "fast weights".

    `self.module` is the original submodule and stays a normal child, so its
    `nn.Parameter`s remain in `state_dict()` / `parameters()` and keep receiving
    ordinary outer-loop gradients. Note this does shift the checkpoint key by one
    level (`....to_v.weight` becomes `....to_v.module.weight`).

    `self.batch_params` is a *plain attribute* holding a dict of tensors shaped
    (batch, *param_shape) - deliberately not registered as parameters or buffers,
    since they are transient inner-loop state, not something to save or hand to an
    optimizer. While it is set, forward runs `torch.func.functional_call` under
    `vmap`, so batch element b is evaluated with its own slice of the weights.
    Because each sample only touches its own slice, one `autograd.grad` call on a
    batch-aggregated loss yields correctly separated per-sample gradients.

    When `batch_params` is None the wrapper is a pure pass-through, which is what
    makes a TTT-enabled model numerically identical to a plain one before
    `init_ttt` runs.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.batch_params = None

    def reset(self):
        # drop the fast weights, reverting to the plain pass-through path
        self.batch_params = None

    @staticmethod
    def reset_all(module):
        # convenience for clearing every TTT wrapper anywhere under `module`,
        # e.g. between unrelated sequences at inference
        for mod in module.modules():
            if isinstance(mod, TTTModuleWrapper):
                mod.reset()

    def forward(self, *args, **kwargs):
        if not exists(self.batch_params):
            return self.module(*args, **kwargs)

        def call_single(params, *args_single):
            return functional_call(self.module, params, args_single, kwargs)

        # vmap maps over dim 0 of the params dict *and* of every positional arg, so
        # both must carry the batch on dim 0. kwargs are closed over rather than
        # mapped, so anything batch-dependent has to arrive positionally

        return vmap(call_single)(self.batch_params, *args)

# custom TTT loss modules

class TTTMetaLearningTargetKLLoss(Module):
    """A learned TTT objective: match a bidirectional re-read of the segment.

    Instead of test-time-training on the language-modelling loss, this builds a
    target from the decoder's own last-layer hiddens. Two heads read those hiddens:
    `to_prediction` reads them directly (causal view), `to_meta_prediction` reads
    them after a small bidirectional `Encoder` (so every position may look ahead).
    The TTT loss is KL(causal || encoded) per token.

    The encoder and both heads are ordinary parameters of the wrapper. They get
    gradient because the inner update they drive is differentiated through by the
    outer backward, so the outer optimizer *meta-learns* what target the inner
    loop should chase - hence "meta learning target". Neither side of the KL is
    detached, so gradient reaches both heads and the shared hiddens.

    Returns an unreduced (b, n) loss; `update_ttt` does the masking / reduction.
    """

    def __init__(
        self,
        *,
        dim,             # decoder hidden dim (must match `last_layer_hiddens`)
        num_classes,     # vocab size for both prediction heads
        depth = 1,       # depth of the bidirectional encoder that builds the meta target
        heads = 4,
        dim_head = 64,
        **kwargs         # forwarded to Encoder
    ):
        super().__init__()
        from x_transformers.x_transformers import Encoder  # imported lazily, only needed when this optional loss is constructed

        self.encoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **kwargs
        )

        self.to_prediction = nn.Linear(dim, num_classes)
        self.to_meta_prediction = nn.Linear(dim, num_classes)

    def forward(self, intermediates, mask = None):
        hiddens = intermediates.last_layer_hiddens  # (b, n, d) - decoder output before to_logits

        pred = self.to_prediction(hiddens)

        encoded = self.encoder(hiddens, mask = mask)  # non-causal pass: each position may see the whole segment
        meta_pred = self.to_meta_prediction(encoded)

        prob_earlier = pred.softmax(dim = -1)
        log_prob_encoded = meta_pred.log_softmax(dim = -1)

        # F.kl_div(input = log-probs, target = probs) computes KL(target || input),
        # so this is KL(causal || encoded); sum over the vocab axis leaves (b, n)

        loss = F.kl_div(log_prob_encoded, prob_earlier, reduction = 'none').sum(dim = -1)
        return loss

class TTTNextLatentPredictionLoss(Module):
    """ use a small MLP to generate the ttt inner loop gradients, by predicting the next latent, instead of cross entropy """

    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        num_layers = 3,
        loss_type = 'smooth_l1'
    ):
        super().__init__()

        hidden_dim = default(hidden_dim, dim)

        layers = [nn.LayerNorm(dim)]

        for i in range(num_layers):
            is_last = i == (num_layers - 1)

            in_dim = dim if i == 0 else hidden_dim
            out_dim = dim if is_last else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))

            if not is_last:
                layers.append(nn.GELU())

        self.predictor = nn.Sequential(*layers)

        if loss_type == 'smooth_l1':
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == 'mse':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'unknown loss type {loss_type}')

    def forward(self, intermediates, mask = None):
        hiddens = intermediates.last_layer_hiddens

        # the last token predicts the first token in the window

        pred = self.predictor(hiddens)
        target = cat((hiddens[:, 1:], hiddens[:, :1]), dim = 1).detach()

        loss = self.loss_fn(pred, target, reduction = 'none').mean(dim = -1) # (b n)

        if exists(mask):
            # target latent at position i + 1 is only valid if the token label at i + 1 is valid
            loss = loss.masked_fill(~cat((mask[:, 1:], mask[:, :1]), dim = 1), 0.)

        return loss

# xl autoregressive wrapper class

class XLAutoregressiveWrapper(Module):
    """Transformer-XL autoregressive training / sampling, with optional e2e TTT.

    Training (`forward`): splits the sequence into `net.max_seq_len` segments, runs
    them in order while threading `mems` forward, and returns a length-weighted sum
    of the per-segment losses. `tbptt_steps` controls how many segments the
    gradient is allowed to span before memories (and TTT fast weights) are
    detached.

    Sampling (`generate`): replays the prompt segment by segment to warm up `mems`,
    then decodes one token at a time, rolling over to a fresh segment (and a fresh
    intra-segment kv cache) at every `max_seq_len` boundary.

    Test-time training: if `ttt_module_paths` is non-empty, the named submodules
    are replaced in-place with `TTTModuleWrapper`s and an inner SGD loop runs over
    their per-sample fast weights after every segment - in both `forward` and
    `generate`. See `update_ttt` for the update rule and the module docstring for
    the overall picture.
    """

    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        tbptt_steps = 1,                                # truncate BPTT (and the TTT inner loop) every N segments; must be > 1 for TTT
        ttt_module_paths = tuple(),                     # dotted paths to test-time-train; str -> source == target, or (source_path, target_path) to apply one module's gradient to another
        ttt_lr = 1e-3,                                  # inner-loop step size for plain (non-Muon) gradients
        ttt_wd = 0.01,                                  # inner-loop decoupled weight decay, applied as -lr * wd * param
        ttt_use_muon = False,                           # orthogonalize inner-loop gradients with Newton-Schulz before stepping
        ttt_muon_steps = 5,                             # Newton-Schulz iterations per update
        ttt_muon_lr = 1e-2,                             # separate step size for parameters that Muon actually applied to (matrices only)
        ttt_low_rank_paths = tuple(),                   # (path, rank) pairs: swap that Linear for a zero-product LowRankLinear and test-time-train only its two factors
        ttt_custom_loss_module: Module | None = None,   # replaces the LM loss as the inner-loop objective; called as (intermediates, mask = ...)
        ttt_next_latent_loss = False,                   # shortcut for ttt_custom_loss_module = TTTNextLatentPredictionLoss(dim = net.attn_layers.dim)
        episodic_mem_len = 0                            # > 0 adds a learned per-layer kv bank; usable as a dedicated long-term TTT target
    ):
        """
        Initialize the XL Autoregressive Wrapper.

        Args:
            net: The Transformer-XL network to wrap. Must have a max_seq_len attribute.
            ignore_index (int, optional): Token index to ignore when calculating loss.
                Defaults to -100 (PyTorch's default for cross_entropy).
            pad_value (int, optional): Value to use for padding tokens in generated sequences.
                Defaults to 0.
        """
        super().__init__()
        # Store padding configuration
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.tbptt_steps = tbptt_steps

        # Store the network and extract its maximum sequence length
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.output_is_log_prob = net.output_is_log_prob  # set by mixture-of-softmax; decides cross_entropy vs nll_loss below

        self.loss_fn = F.cross_entropy if not self.output_is_log_prob else F.nll_loss

        self.ttt_lr = ttt_lr
        self.ttt_wd = ttt_wd
        self.ttt_use_muon = ttt_use_muon
        self.ttt_muon_steps = ttt_muon_steps
        self.ttt_muon_lr = ttt_muon_lr
        self.ttt_custom_loss_module = ttt_custom_loss_module

        if ttt_next_latent_loss:
            assert not exists(ttt_custom_loss_module), 'ttt_custom_loss_module and ttt_next_latent_loss cannot both be set'

            dim = getattr(getattr(net, 'attn_layers', None), 'dim', None)
            assert exists(dim), 'the model must have an attn_layers.dim attribute to derive the dimension for the next latent prediction loss'

            self.ttt_custom_loss_module = TTTNextLatentPredictionLoss(dim = dim)

        self.has_episodic_mem = episodic_mem_len > 0

        # maybe wrap network with episodic memory wrapper

        if self.has_episodic_mem:
            net = EpisodicMemoryWrapper(net, episodic_mem_len)

        self.net = net  # rebinds over the assignment above, now possibly the episodic wrapper

        # format ttt module paths to always be (source_path, target_path)

        # cast_tuple('a.b.c', 2) -> ('a.b.c', 'a.b.c'), so a bare string means
        # "differentiate this module and update this same module". An explicit pair
        # decouples the two, which is how a gradient computed at a later layer can be
        # applied to an earlier layer's weights

        self.ttt_paths_map = tuple(cast_tuple(item, 2) for item in ttt_module_paths)

        # low rank paths, specifying the path to a weight and the low rank dim, the weight will be replaced with two low rank matrices that hold the ttt memories

        self.ttt_low_rank_paths_map = dict()

        for item in ttt_low_rank_paths:
            path, low_rank_dim = item
            self.ttt_low_rank_paths_map[path] = low_rank_dim

        # modify paths to have 'net.' prepended if wrapped, with the exception of the episodic memories which live at the top level of the wrapper

        # `net` is now the EpisodicMemoryWrapper, so paths into the real transformer
        # gained a level. 'episodic_mems' is exempt because it hangs off the wrapper
        # itself, not off the inner network

        if self.has_episodic_mem:
            self.ttt_paths_map = tuple(
                (f'net.{src}' if src != 'episodic_mems' else src, f'net.{tgt}' if tgt != 'episodic_mems' else tgt)
                for src, tgt in self.ttt_paths_map
            )

            self.ttt_low_rank_paths_map = dict(
                (f'net.{path}', low_rank_dim)
                for path, low_rank_dim in self.ttt_low_rank_paths_map.items()
            )

        # the low rank matrices are both the source and target of the ttt update, as the memories are stored in them

        self.ttt_paths_map = self.ttt_paths_map + tuple(
            (path, path)
            for path in self.ttt_low_rank_paths_map.keys()
        )

        # gather all unique wrappers

        # in-place surgery: every path named as a source or a target gets its module
        # swapped for a TTTModuleWrapper, deduped so a module named twice is wrapped once

        self.ttt_wrappers = dict()

        for source_path, target_path in self.ttt_paths_map:
            for path in (source_path, target_path):
                if path in self.ttt_wrappers:
                    continue

                mod = get_module_by_path(net, path)

                # maybe replace the weight at this path with two low rank matrices to hold the ttt memories

                if path in self.ttt_low_rank_paths_map:
                    mod = LowRankLinear(mod, self.ttt_low_rank_paths_map[path])
                    set_module_by_path(net, path, mod)

                wrapper = TTTModuleWrapper(mod)
                set_module_by_path(net, path, wrapper)

                self.ttt_wrappers[path] = wrapper

        # with tbptt_steps == 1 every segment is a truncation boundary, so the fast
        # weights would be detached immediately after each update and the outer
        # backward could never reach through the inner loop - no "end-to-end" at all

        if self.has_ttt:
            assert tbptt_steps > 1, 'tbptt_steps must be greater than 1 if ttt is turned on'

    @property
    def has_ttt(self):
        return len(self.ttt_wrappers) > 0

    def init_ttt(self, batch):
        """Seed the fast weights: every batch element gets its own copy of the
        current (outer) weights, shape (batch, *param_shape).

        Called at the start of each `forward` / `generate`, so TTT state never
        survives across calls - each sequence starts from the slow weights again.

        `repeat_batch` is differentiable, so gradient arriving at the fast weights
        also flows back into the underlying `nn.Parameter` (summed over the batch).
        That is why a TTT model with `ttt_lr = 0` produces exactly the same outer
        gradients as the same model with TTT off.
        """
        if not self.has_ttt:
            return

        for wrapper in self.ttt_wrappers.values():
            wrapper.batch_params = tree_map_tensor(
                lambda t: repeat_batch(t, batch),
                dict(wrapper.module.named_parameters())  # recursive, dotted names - functional_call accepts them as-is
            )

    def update_ttt(
        self,
        logits,        # (b n c)
        chunk_labels,  # (b n)
        create_graph = False,
        intermediates = None
    ):
        """One inner-loop (test-time training) step, run after a segment.

        Mechanically:
          1. build a per-sample loss (b,) for the segment (LM cross-entropy / NLL,
             or the custom loss module if one was given),
          2. differentiate it w.r.t. the *source* wrappers' fast weights,
          3. optionally push each gradient through Newton-Schulz (Muon),
          4. apply `p <- p - lr * g - lr * wd * p` to the *target* wrappers' fast
             weights, so the update accumulates across segments.

        Gradient flow:
          - the differentiation target is `batch_params`, i.e. the per-sample copies
            inside `TTTModuleWrapper`, never the `nn.Parameter`s directly. Since
            `vmap` keeps each sample on its own slice, one grad call on the *sum* of
            the per-sample losses gives each sample exactly its own gradient, so the
            write is identical whether a sequence runs alone or in a batch.
          - `create_graph = True` (used in `forward` while training) keeps the inner
            step in the autograd graph, so the updated fast weight is a non-leaf whose
            history reaches back through `dL/dp` to the slow weights. The outer
            backward therefore takes second-order (double-backward) terms - this is
            what makes the scheme end-to-end.
          - `create_graph = False` (used in `generate`, and in eval) instead detaches
            the result below and re-leafs it, so the next call can differentiate again.
            Note the paired `retain_graph = create_graph`: with `create_graph = False`
            this frees the forward graph, so any `total_loss` built from the same
            forward can no longer be backwarded afterwards.
        """
        if not self.has_ttt:
            return

        # compute loss

        mask = chunk_labels != self.ignore_index

        # the custom loss reads `intermediates` (hiddens etc.) rather than the logits,
        # which is how a loss can be defined on top of the decoder rather than on the
        # next-token target. it may return (b n), reduced below per sample, or an
        # already-reduced tensor, hence the ndim check

        if exists(self.ttt_custom_loss_module):
            assert exists(intermediates), 'intermediates must be passed to update_ttt when using ttt_custom_loss_module'
            loss_t = self.ttt_custom_loss_module(intermediates, mask = mask)
            loss_t_per_batch = masked_mean(loss_t, mask, dim = -1) if loss_t.ndim > 1 else loss_t
        else:
            loss_t = self.loss_fn(
                rearrange(logits, 'b n c -> b c n'),  # F.cross_entropy wants classes on dim 1
                chunk_labels,
                ignore_index = self.ignore_index,
                reduction = 'none'
            )

            # reduce over the sequence only, giving one loss per sample (b,). e5b145e changed
            # this from a full reduction: a scalar mean over the whole batch scaled every
            # sample's fast-weight gradient by its share of the batch's tokens, so the
            # memory write depended on batch size (see test_ttt_write_batch_size_independent)

            loss_t_per_batch = masked_mean(loss_t, mask, dim = -1)

        # gather all batch parameters for source modules

        # flat list in ttt_paths_map order; `grad_idx` below walks it back in the same
        # order to re-associate gradients with their (source, target) pair

        all_batch_params = [
            param
            for source_path, _ in self.ttt_paths_map
            for param in self.ttt_wrappers[source_path].batch_params.values()
        ]

        # compute gradients

        # `.sum()` over the per-sample losses, not `.mean()`: each sample's fast weights
        # only see their own term, so a sum hands every sample its unscaled gradient.
        # allow_unused: a source module can be absent from the graph of a custom loss,
        # in which case its gradient comes back as None and is skipped below

        grads = torch.autograd.grad(
            loss_t_per_batch.sum(),
            all_batch_params,
            create_graph = create_graph,
            retain_graph = create_graph,
            allow_unused = True
        )

        # update target parameters with source gradients

        grad_idx = 0

        for source_path, target_path in self.ttt_paths_map:
            source_wrapper = self.ttt_wrappers[source_path]
            target_wrapper = self.ttt_wrappers[target_path]

            num_params = len(source_wrapper.batch_params)
            wrapper_grads = grads[grad_idx : grad_idx + num_params]
            grad_idx += num_params

            assert len(source_wrapper.batch_params) == len(target_wrapper.batch_params), f'source module {source_path} and target module {target_path} must have the same number of parameters'

            # maybe muon

            # newtonschulz5 reports back whether it actually orthogonalized: it declines
            # on anything with per-sample ndim < 2 (biases, scalars), which is why the
            # flag has to be tracked per parameter rather than per module

            if self.ttt_use_muon:
                muon_results = [newtonschulz5(g, steps = self.ttt_muon_steps) if exists(g) else (None, False) for g in wrapper_grads]
                wrapper_grads, is_muon_grads = tuple(zip(*muon_results))
            else:
                is_muon_grads = (False,) * len(wrapper_grads)

            # update test-time trained parameters

            # source and target params are paired positionally, in named_parameters()
            # order - the two modules must be structurally identical, which the shape
            # assert only partially enforces

            new_batch_params = dict()

            for (src_name, src_param), (tgt_name, tgt_param), grad, is_muon_grad in zip(source_wrapper.batch_params.items(), target_wrapper.batch_params.items(), wrapper_grads, is_muon_grads):
                assert src_param.shape == tgt_param.shape, f'source parameter {source_path}.{src_name} (shape {src_param.shape}) and target parameter {target_path}.{tgt_name} (shape {tgt_param.shape}) must have the exact same shape'

                # plain SGD step plus decoupled weight decay (adamw-style: the decay is
                # not folded into the gradient). Muon-orthogonalized directions are unit
                # scale regardless of the raw gradient norm, so they get their own lr

                if exists(grad):
                    lr = self.ttt_muon_lr if is_muon_grad else self.ttt_lr
                    tgt_param = tgt_param - lr * grad - lr * self.ttt_wd * tgt_param

                # non-differentiable path: cut the fast weight out of the graph and make
                # it a fresh leaf. `requires_grad_()` is mandatory, not cosmetic - the
                # next update_ttt differentiates w.r.t. these very tensors

                if not create_graph:
                    tgt_param = tgt_param.detach().requires_grad_()

                new_batch_params[tgt_name] = tgt_param

            target_wrapper.batch_params = new_batch_params

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_kwargs: dict = dict(),
        mems = None,
        **kwargs
    ):
        """
        Generate new tokens autoregressively using the Transformer-XL model.

        This method generates new tokens one at a time, maintaining memory across segments
        for extended context. It supports various sampling strategies through logit filtering
        (top-k, top-p), temperature scaling, and early stopping with end-of-sequence tokens.

        The generation process works in two phases:
        1. Pre-processing: Process all leading token segments to build up memory
        2. Generation: Autoregressively sample new tokens using the accumulated memory

        Args:
            start_tokens (torch.Tensor): Initial tokens to condition generation on.
                Shape can be (batch,) or (batch, seq_len).
            seq_len (int): Number of new tokens to generate.
            eos_token (int, optional): End-of-sequence token ID. If provided and generated,
                stops generation and pads remaining positions. Defaults to None.
            temperature (float, optional): Sampling temperature. Higher values (>1) increase
                randomness, lower values (<1) make sampling more deterministic. Defaults to 1.0.
            filter_logits_fn (callable, optional): Function to filter logits before sampling
                (e.g., top_k, top_p). Defaults to top_k.
            filter_kwargs (dict, optional): Keyword arguments to pass to filter_logits_fn.
                Defaults to empty dict.
            mems (optional): Initial memory state. If None, starts with no memory.
                Defaults to None.
            **kwargs: Additional keyword arguments to pass to the network's forward pass.

        Returns:
            torch.Tensor: Generated tokens (excluding the start_tokens), with the same
                batch dimensions as the input. Shape: (batch, seq_len) or original shape.

        When TTT is enabled the prompt replay in phase 1 doubles as the inner
        training loop: each prompt segment is scored against its own next-token
        labels and an inner step is taken, so the fast weights arrive at phase 2
        already adapted to the prompt. Those inner steps are built with
        `create_graph = False`, so nothing here is differentiable - only the
        segment-to-segment fast-weight recursion survives.
        """
        # Extract device and maximum sequence length for this generation
        device, max_seq_len = start_tokens.device, self.max_seq_len

        # Pack start_tokens to ensure consistent shape handling (batch, seq_len)
        # ps stores the packing specification for unpacking later
        start_tokens, ps = pack([start_tokens], '* n')

        batch, t = start_tokens.shape

        # `split` yields a trailing chunk of 1..max_seq_len tokens; `all_leading_tokens`
        # are the full segments before it, `remainder_tokens` the partial tail that
        # generation will continue from

        *all_leading_tokens, remainder_tokens = start_tokens.split(max_seq_len, dim = -1)

        # this method is under @torch.no_grad(), but the TTT inner loop needs a live
        # graph to differentiate; re-enable grad locally when TTT is on

        forward_context = torch.enable_grad if self.has_ttt else nullcontext

        # Phase 1: Process all leading segments to build up memory
        # This "catches up" the memory to the current position without generating

        with forward_context():
            self.init_ttt(batch)

            for idx, leading_tokens in enumerate(all_leading_tokens):
                logits, intermediates = self.net(
                    leading_tokens,
                    mems = mems,
                    return_mems = True,
                    return_intermediates = True,
                    **kwargs
                )
                mems = intermediates.mems  # roll segment i's hiddens forward into segment i+1

                if not self.has_ttt:
                    continue

                # next-token labels for this segment, read out of the full prompt so the
                # label for the last position can come from the following segment. safe
                # because `remainder_tokens` always holds at least one token

                curr_pos = idx * max_seq_len
                chunk_labels = start_tokens.narrow(1, curr_pos + 1, max_seq_len)
                self.update_ttt(logits, chunk_labels, create_graph = False, intermediates = intermediates)

        # test-time train on the remainder of the prompt

        # one more inner step on the partial tail (needs >= 2 tokens to form an
        # input/label pair). note this pass also advances `mems` past the remainder,
        # which phase 2 then re-reads from `curr_pos` - only reachable when TTT is on

        if self.has_ttt and remainder_tokens.shape[-1] > 1:
            with forward_context():
                remainder_x = remainder_tokens[:, :-1]
                remainder_labels = remainder_tokens[:, 1:]

                logits, intermediates = self.net(
                    remainder_x,
                    mems = mems,
                    return_mems = True,
                    return_intermediates = True,
                    **kwargs
                )
                mems = intermediates.mems

                self.update_ttt(logits, remainder_labels, create_graph = False, intermediates = intermediates)

        # Phase 2: Begin autoregressive generation from the current segment

        # Calculate starting position for generation (after all leading tokens)
        curr_pos = len(all_leading_tokens) * max_seq_len
        # Initialize current memories with the accumulated memory from leading segments
        curr_mems = mems

        # Cache will store intermediate activations for efficient generation
        cache = None
        # Output starts with all the start_tokens
        out = start_tokens

        is_greedy = temperature == 0.

        # unwrap episodic memory wrapper to avoid evaluating it per token

        # the learned kv bank does not change during decoding, so materialize it once
        # and pass it through kwargs, calling the inner net directly from here on

        net = self.net
        if self.has_episodic_mem:
            kwargs.update(
                self_attn_additional_kv = net.get_additional_kv(batch)
            )
            net = net.net

        for _ in range(seq_len):
            # Get the current total sequence length
            curr_segment_len = out.shape[-1]
            # Check if we've reached the end of a segment (divisible by max_seq_len)
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            # segment rollover: re-anchor `curr_pos` to the segment start, adopt the mems
            # produced by the completed segment and drop the intra-segment kv cache, so
            # the finished segment is reached only through `mems` from here on.
            # careful - because `curr_pos` is set to `out.shape[-1]`, the slice below is
            # empty on exactly the iterations where this branch fires, and `logits[:, -1]`
            # then indexes a zero-length axis

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems
                cache = None

            # everything in the current segment. when `cache` is not None the attention
            # layers slice this down to the newest `cache_age` (default 1) tokens
            # themselves, so handing them the whole suffix is intentional

            x = out[:, curr_pos:]

            if x.shape[-1] == 0:
                # the tokens end exactly on a segment boundary, recompute the last complete segment with a fresh cache
                x = out[:, curr_pos - max_seq_len:]

            logits, cache = net(
                x,
                mems = curr_mems,
                cache = cache,
                return_mems = True,
                return_intermediates = True,
                **kwargs
            )

            # Extract the updated memory from cache
            mems = cache.mems

            # Get logits for the last position only (next token prediction)
            logits = logits[:, -1]

            # temperature == 0 is treated as greedy decoding rather than a division by zero

            if is_greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            # same rollover as above, repeated after the forward so `curr_mems` picks up
            # the mems this segment just produced rather than the previous segment's

            # If we've completed a full segment, update position and memories
            if is_last_segment_tokens:
                # Move current position to start of next segment
                curr_pos = curr_segment_len
                # Update current memories to use for next segment
                curr_mems = mems
                cache = None

            out = torch.cat((out, sample), dim = -1)

            # Handle early stopping if end-of-sequence token is specified
            if exists(eos_token):
                # Check which positions in output contain the EOS token
                is_eos_tokens = (out == eos_token)

                # Check if all sequences in batch have generated at least one EOS token
                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    # verified behaviour of the helper: `mask_after(t, v, inclusive = False)`
                    # returns `(t == v).cumsum(-1) == 0`, i.e. True strictly *before* the
                    # first eos. the sibling AutoregressiveWrapper builds the complementary
                    # mask (shifted cumsum >= 1, True strictly after) for the same fill
                    mask = mask_after(out, eos_token, inclusive = False)
                    out = out.masked_fill(mask, self.pad_value)
                    # Stop generation early since all sequences have finished
                    break

        # Remove the initial start_tokens, keeping only newly generated tokens
        out = out[:, t:]

        # Unpack to restore original input shape
        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        mems = None,
        dropout_mems = False,       # withhold the XL segment memory, forcing the fast weights to carry cross-segment context
        ttt_recurrent_steps = 1,    # re-read each segment this many times, accumulating fast-weight updates; only the last pass scores
        **kwargs
    ):
        """Chunked Transformer-XL training step, optionally with an e2e TTT inner loop.

        Splits `x` into next-token (input, label) pairs, then walks segments of
        `max_seq_len` left to right. Each segment contributes its loss weighted by
        its share of the sequence, and hands its hidden states to the next segment
        as `mems`.

        Gradient truncation is on a `tbptt_steps` cadence: on segments where
        `(idx + 1) % tbptt_steps == 0`, `detach_mems = True` cuts the memory graph
        and the TTT fast weights are re-leafed, so backprop spans at most
        `tbptt_steps` segments. Between those boundaries both the memories and the
        fast-weight recursion stay differentiable.

        `dropout_mems` and `ttt_recurrent_steps` are the two knobs added for a
        "sleep"-style pass: drop the previous segment's kv memory, and/or replay the
        same segment several times so the fast weights absorb it, before scoring.
        """
        if ttt_recurrent_steps > 1:
            # treating the ttt gradients as the fast weights
            assert self.has_ttt, 'ttt must be turned on to use ttt_recurrent_steps'

        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        # Prepare input-label pairs for autoregressive training
        # Input: all tokens except the last one
        # Labels: all tokens except the first one (shifted by 1 position)
        x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape[:2]

        # Split sequences into chunks of max_seq_len
        # This is necessary because Transformer-XL has a maximum segment length

        # Split input tokens into chunks
        split_x = x.split(max_seq_len, dim = -1)
        # Split corresponding labels into chunks
        split_labels = labels.split(max_seq_len, dim = -1)
        # Calculate weight for each chunk based on its proportion of total length
        # This ensures chunks contribute to loss proportionally to their size
        loss_weights = tuple((t.shape[-1] / seq_len) for t in split_x)

        # Process each chunk and accumulate weighted losses

        total_loss = 0.
        num_chunks = len(split_x)

        # as in `generate`, TTT needs a graph even if the caller wrapped this in no_grad

        forward_context = torch.enable_grad if self.has_ttt else nullcontext

        with forward_context():
            # init the ttt batch parameters under grad context, to work even when called within an outer no_grad context
            # this also resets the fast weights to the current slow weights - TTT state does not persist across calls
            self.init_ttt(batch)

            for idx, (chunk, chunk_labels, loss_weight) in enumerate(zip(split_x, split_labels, loss_weights)):
                is_last_chunk = (idx == num_chunks - 1)
                should_detach = divisible_by(idx + 1, self.tbptt_steps)  # truncated-BPTT boundary: cut both mems and fast weights below

                # replaying the same chunk: every pass takes an inner TTT step, but only
                # the final pass contributes to the loss and advances `mems`

                for recurrent_step in range(ttt_recurrent_steps):
                    is_last_recurrent_step = (recurrent_step == ttt_recurrent_steps - 1)

                    # withholding mems only affects what this pass attends over; `mems`
                    # is still overwritten from this pass's own hiddens below

                    passed_mems = None if dropout_mems else mems

                    logits, intermediates = self.net(
                        chunk,
                        mems = passed_mems,
                        return_mems = True,
                        return_intermediates = True,
                        detach_mems = should_detach,  # False keeps the memory graph alive across the segment boundary
                        **kwargs
                    )

                    if is_last_recurrent_step:
                        mems = intermediates.mems

                    loss = self.loss_fn(
                        rearrange(logits, 'b n c -> b c n'),
                        chunk_labels,
                        ignore_index = ignore_index,
                        reduction = 'none'
                    )

                    mask = chunk_labels != ignore_index

                    # one loss per sample (b,), reduced over the sequence only - the TTT
                    # update needs it unreduced across the batch, and `.mean()` below turns
                    # it into this chunk's scalar contribution to the outer loss

                    loss_per_batch = masked_mean(loss, mask, dim = -1)

                    if is_last_recurrent_step:
                        total_loss = total_loss + loss_per_batch.mean() * loss_weight

                    # no chunk follows the last one, so its inner update would never be
                    # consumed - skip it and save a backward pass

                    if (is_last_chunk and is_last_recurrent_step) or not self.has_ttt:
                        continue

                    # create_graph = self.training: differentiable inner step while
                    # training (second-order terms reach the slow weights), plain
                    # non-differentiable adaptation in eval. in eval the paired
                    # retain_graph = False also frees this chunk's forward graph, so
                    # `total_loss` is no longer backward-able once TTT has run

                    self.update_ttt(logits, chunk_labels, create_graph = self.training, intermediates = intermediates)

                if not should_detach:
                    continue

                # detach

                # truncate the fast-weight recursion in step with the mems truncation
                # above. re-leafing preserves the original requires_grad flag so the next
                # update_ttt can still differentiate w.r.t. these tensors

                for wrapper in self.ttt_wrappers.values():
                    wrapper.batch_params = tree_map_tensor(
                        lambda t: t.detach().requires_grad_(t.requires_grad),
                        wrapper.batch_params
                    )

        return total_loss
