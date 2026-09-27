# DISCOVER: The Emergent Symbolic Structure of Artificial Neural Networks

# McCoy et al. https://arxiv.org/abs/2608.29530

"""
DISCOVER - probe whether a network's single-vector encoding of a symbol sequence is a
tensor product representation (TPR) of role-filler bindings, then edit it as one.

(paper, per the bibtex entry in this repo's README: "The Emergent Symbolic Structure of
Artificial Neural Networks" - McCoy, Soulos, Linzen, Smolensky)

The question: when a network squeezes a sequence of symbols into one vector, is that
vector compositional - a sum of (symbol, structural slot) terms - and if so, which notion
of "slot" does the network use? The module answers it in three steps.

1. A bottleneck model, `DiscoverDecoder`. One causal `TransformerWrapper` reads the input
   sequence, and its final layer hidden state at one position (the last valid token by
   default) is taken as the encoding E, shape (b, d). The same transformer then generates
   the output sequence with E prepended as a single input embedding in front of
   [bos, y_0, y_1, ...]. The decoding pass never sees the input tokens, so everything has
   to flow through E. It is trained with ordinary cross entropy (`train_discover_reversal.py`
   trains it to reverse lists of letters).

2. A TPR fitted to E, `TPREncoder`. Each input token is a "filler" (the symbol), and a
   hypothesized role scheme (`ROLE_SCHEMES` - position from the left, position from the
   right, both, the neighboring symbols, or no role at all) assigns it a "role". Fillers
   and roles get learned embeddings, and the approximation is

       E_hat = sum_n  B (filler_emb[f_n] outer role_emb[r_n]) + bias

   with B a learned (d, filler_dim, role_dim) binding tensor, i.e. a linear map from the
   flattened outer product to d. It is fitted by MSE against the frozen network encodings.
   Besides the fit MSE, the training script measures whether E_hat, decoded in place of E,
   still yields the correct output ("approximation accuracy"). A role scheme under which
   it does describes the structure the network actually uses, and comparing schemes
   contrasts the hypotheses. That comparison is only meaningful with the default linear
   readout (see the note on `TPREncoder`).

3. Constituent surgery, `DiscoverDecoder.surgery`. Substitute one symbol, compute the
   change in encoding that the TPR predicts for it, add that change to the real E and
   decode. If the output changes as if the input symbol had been replaced (for reversal:
   the new letter shows up in the mirrored slot), then the decoder really uses the TPR's
   filler and role directions, rather than E merely being correlated with them.

Gradient flow: the transformer is trained only by the cross entropy in
`DiscoverDecoder.forward`. There E is not detached, so the transformer receives gradient
both as decoder and as encoder. The TPR is fitted afterwards against detached encodings,
so fitting it never changes the network. `decode` and `surgery` run under no_grad.

Token convention: fillers are dense ids in [0, num_fillers). `bos_id` and `pad_id` are
extra ids of the transformer vocabulary outside that range (the training script and test
use bos = num_fillers, pad = num_fillers + 1, num_tokens = num_fillers + 2).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import einsum, Tensor
from torch.nn import Embedding, Linear, Parameter, SiLU, Module, Sequential

from torch_einops_utils import batched_index_select, lens_to_mask

from einops import rearrange

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.x_transformers import TransformerWrapper, exists, default

# hypotheses for which structural slot (role) a token occupies. each one gives a role id per
# position via `role_ids_for_tokens`, from a vocabulary sized by `num_roles_for_scheme`
#   left_to_right  - index from the start of the sequence
#   right_to_left  - index from the end of the sequence
#   bidirectional  - the (index from start, index from end) pair. the two always sum to
#                    length - 1, so this amounts to one role per (length, position)
#   wickel         - the (previous symbol, next symbol) pair, with a boundary symbol at the
#                    edges. the role depends on the neighboring fillers, not on position
#   bag_of_words   - one shared role, so order is ignored and E_hat is a sum over fillers

ROLE_SCHEMES = ('bidirectional', 'right_to_left', 'left_to_right', 'wickel', 'bag_of_words')

def num_roles_for_scheme(
    scheme: str,
    *,
    max_seq_len: int,
    num_fillers: int
) -> int:
    """size of the role vocabulary a scheme needs, i.e. `num_roles` for `TPREncoder`

    left_to_right, right_to_left - one role per position, max_seq_len
    bidirectional                - pairs (i, j) with i + j <= max_seq_len - 1, which is
                                   max_seq_len * (max_seq_len + 1) / 2
    wickel                       - (previous, next) pairs over num_fillers symbols plus one
                                   boundary symbol, (num_fillers + 1) ** 2
    bag_of_words                 - 1

    note: any unrecognized scheme name also falls through to 1, and `role_ids_for_tokens`
    maps it to all zeros, so a misspelled scheme silently behaves as bag_of_words
    """
    if scheme in ('left_to_right', 'right_to_left'):
        return max_seq_len
    if scheme == 'bidirectional':
        return max_seq_len * (max_seq_len + 1) // 2
    if scheme == 'wickel':
        return (num_fillers + 1) ** 2
    return 1

def role_ids_for_tokens(
    tokens: Tensor,
    lengths: Tensor,
    scheme: str,
    *,
    max_seq_len: int,
    num_fillers: int
) -> Tensor:
    """positional role schemes (roles shared across sequence lengths)

    tokens:  (b, n) filler ids, valid in the first `lengths` positions of each row (right padded)
    lengths: (b,) number of valid tokens per row
    returns: (b, n) long role ids in [0, num_roles_for_scheme(scheme, ...)), 0 at padded positions

    one role vocabulary (one embedding table in the TPR) serves every sequence length.
    under left_to_right / right_to_left the same index is the same role at every length,
    while bidirectional reserves a separate block of roles for each length (see below)

    note: `max_seq_len` is accepted but not used here, and nothing checks positions against
    it. out of range role ids are only caught later, by the range assert in `TPREncoder.forward`
    """
    device = tokens.device
    seq_len = tokens.shape[1]

    # positions (1, n) broadcast against per-row lengths (b, 1) -> (b, n)

    positions = rearrange(torch.arange(seq_len, device = device), 'n -> 1 n')
    valid = lens_to_mask(lengths, seq_len)          # (b, n) True within each row's length
    lengths = rearrange(lengths, 'b -> b 1')
    from_right = lengths - 1 - positions            # (b, n) index counted back from the last valid token, negative in the padding

    if scheme == 'left_to_right':
        roles = positions
    elif scheme == 'right_to_left':
        roles = from_right
    elif scheme == 'bidirectional':

        # cantor pairing of (i, j) = (index from left, index from right): T(i + j) + i, with
        # T(s) = s (s + 1) / 2. i + j = length - 1 for every valid token, so the role is
        # T(length - 1) + i - lengths 1, 2, 3 get the roles {0}, {1, 2}, {3, 4, 5}

        pair_sum = positions + from_right
        roles = pair_sum * (pair_sum + 1) // 2 + positions
    elif scheme == 'wickel':

        # role = (previous symbol, next symbol), each one of num_fillers + 1 values where the
        # id num_fillers marks the sequence boundary, flattened to prev * (num_fillers + 1) + next.
        # torch.roll wraps around, so the first position's "previous" and each row's last valid
        # position's "next" are overwritten with the boundary, which keeps valid positions from
        # reading wrapped or padded neighbors. e.g. with 6 fillers, [0, 1, 2] gets the pairs
        # (6, 1), (0, 2), (1, 6) -> roles [43, 2, 13]
        # since roles are built from the fillers, substituting one token also changes the
        # roles of its neighbors

        boundary = num_fillers
        prev_letter = torch.where(positions == 0, torch.full_like(positions, boundary), torch.roll(tokens, 1, dims = 1))
        next_letter = torch.where(positions == lengths - 1, torch.full_like(positions, boundary), torch.roll(tokens, -1, dims = 1))
        roles = prev_letter * (num_fillers + 1) + next_letter
    else:

        # bag_of_words (and any unrecognized scheme) - a single role for every position

        roles = torch.zeros_like(positions)

    # padded positions get role 0, a valid embedding row, and the `mask` given to the TPR
    # removes their contribution. the (1, n) schemes above are broadcast to (b, n) here

    return roles.masked_fill(~valid, 0)

class TPREncoder(Module):
    """TPR fitted to minimize MSE against some target encoding; optional MLP readout

    note: role-scheme contrasts (which scheme the network respects) are only diagnostic
    with a linear readout - an MLP readout tests expressibility, not structure

    For fillers f_n (token ids) in roles r_n (role ids), each sequence is encoded as

        state = sum_n  B (filler_emb[f_n] outer role_emb[r_n]) + bias      (b, dim)
        out   = readout(state)                                              (b, dim)

    B is the (dim, filler_dim, role_dim) `binding` tensor, a linear map from the flattened
    filler_dim x role_dim outer product to the target space. Without `mlp_dim` the readout
    is the identity, so the encoder is linear in the bound pairs and every (filler, role)
    pair contributes its own additive term. That is what makes a per-pair edit in `surgery`
    meaningful.

    All parameters (embeddings, binding, bias, mlp) are trained only by `fit`, against
    detached target encodings, so no gradient reaches the network being probed.
    """

    def __init__(
        self,
        *,
        dim: int,                   # dimension of the target encodings being approximated (the probed model's width)
        num_fillers: int,           # filler vocabulary - token ids must be dense in [0, num_fillers)
        num_roles: int,             # role vocabulary, see `num_roles_for_scheme`
        filler_dim: int = 24,       # filler embedding size
        role_dim: int = 24,         # role embedding size. the binding tensor holds dim * filler_dim * role_dim weights
        mlp_dim: int | None = None  # hidden width of an optional 2 layer SiLU MLP readout. None = identity readout (linear TPR)
    ):
        super().__init__()
        self.dim = dim
        self.filler_emb = Embedding(
            num_fillers,
            filler_dim
        )
        self.role_emb = Embedding(
            num_roles,
            role_dim
        )

        # binding tensor B and bias, both zero initialized, so the TPR starts out predicting 0
        # for every sequence. on the first step only B and bias receive gradient; the
        # embeddings start moving once B is non-zero

        self.binding = Parameter(torch.zeros(dim, filler_dim, role_dim))
        self.bias = Parameter(torch.zeros(dim))

        # small embeddings, std 0.05 instead of the N(0, 1) default of nn.Embedding

        torch.nn.init.normal_(self.filler_emb.weight, std = 0.05)
        torch.nn.init.normal_(self.role_emb.weight, std = 0.05)

        # optional nonlinear readout, dim -> mlp_dim -> dim. it replaces the identity rather
        # than adding a residual

        self.mlp = None
        if exists(mlp_dim):
            self.mlp = Sequential(
                Linear(dim, mlp_dim),
                SiLU(),
                Linear(mlp_dim, dim)
            )

    def state(
        self,
        tokens: Tensor,
        role_ids: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        """binding state per sequence - the sum over role-filler terms, before the readout

        tokens: (b, n) filler ids, role_ids: (b, n), mask: (b, n) bool with True = include
        returns: (b, dim)

        note: unlike `forward`, this does not sanitize padded positions. every token and
        role id must already be a valid embedding index; masked positions are only zeroed
        after the lookup
        """

        # contract B with every filler embedding once: (o f g) x (F f) -> (F o g), where
        # F = filler vocabulary, o = dim, f = filler_dim, g = role_dim.
        # indexing with the token ids then gives each position its filler-bound matrix

        bindings = einsum(
            'o f g, F f -> F o g',
            self.binding,
            self.filler_emb.weight
        )
        per_pair = bindings[tokens]         # (b, n, o, g)
        role = self.role_emb(role_ids)      # (b, n, g)

        # contracting with the role vector completes B (filler outer role) at each position -> (b, n, o)

        contributions = einsum(
            'b n o g, b n g -> b n o',
            per_pair,
            role
        )

        # drop padded positions from the sum

        if exists(mask):
            contributions = contributions.masked_fill(~rearrange(mask, 'b n -> b n 1'), 0.)

        # superpose the bound pairs over the sequence -> (b, o)

        return contributions.sum(dim = 1) + self.bias

    def readout(
        self,
        state: Tensor
    ) -> Tensor:
        """map the binding state to the target encoding space (identity unless an MLP is used)

        state: (b, dim) -> (b, dim)
        """
        return self.mlp(state) if exists(self.mlp) else state

    def fit(
        self,
        tokens: Tensor,
        role_ids: Tensor,
        target_encodings: Tensor,
        *,
        mask: Tensor | None = None,
        steps: int = 2000,
        batch_size: int = 1024,
        lr: float = 3e-3,
        ema_decay: float = 0.9,
        optim: Callable = torch.optim.Adam
    ) -> Tensor:
        """fit by MSE against (frozen) encodings; returns the running minibatch MSE as a diagnostic

        tokens, role_ids: (num_samples, n)
        target_encodings: (num_samples, dim)
        mask:             (num_samples, n) bool, True = valid, defaults to all valid

        runs `steps` optimizer steps (a fresh `optim(self.parameters(), lr = lr)` per call),
        each on `batch_size` rows drawn uniformly with replacement. only this module's
        parameters train.

        returns an exponential moving average (decay `ema_decay`) of the per-step loss.
        note: the average starts at 0. without bias correction, so it underestimates the loss
        on short fits - the zero start keeps weight ema_decay ** steps, e.g. about 0.12 after
        20 steps at the default 0.9. also note the value is a python float (built from
        `loss.item()`), although the return annotation says Tensor
        """
        num_samples, max_seq_len = tokens.shape
        assert role_ids.shape == tokens.shape, 'role ids must be one per position'
        assert target_encodings.shape == (num_samples, self.dim), f'target encodings must be of shape {(num_samples, self.dim)}'

        mask = default(
            mask,
            torch.ones(
                num_samples,
                max_seq_len,
                device = tokens.device,
                dtype = torch.bool
            )
        )

        optimizer = optim(self.parameters(), lr = lr)
        running_loss = 0.

        for _ in range(steps):

            # minibatch of row indices, sampled with replacement

            indices = torch.randint(
                0,
                num_samples,
                (batch_size,),
                device = tokens.device
            )

            optimizer.zero_grad()
            predictions = self(
                tokens[indices],
                role_ids[indices],
                mask[indices]
            )

            # the targets often come straight from `DiscoverDecoder.encode` with the autograd
            # graph still attached (as in the training script) - detaching keeps the MSE from
            # reaching the probed network

            loss = F.mse_loss(
                predictions,
                target_encodings[indices].detach()
            )
            loss.backward()
            optimizer.step()

            running_loss = ema_decay * running_loss + (1 - ema_decay) * loss.item()

        return running_loss

    def forward(
        self,
        tokens: Tensor,
        role_ids: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        """TPR approximation of the target encodings

        tokens: (b, n) filler ids, role_ids: (b, n), mask: (b, n) bool, True = valid, None = all valid
        returns: (b, dim)
        """

        # masked positions may hold out of range ids (e.g. the model's pad id). set them to 0 so
        # the range checks and lookups pass; `state` then drops them from the sum with the same mask

        if exists(mask):
            tokens = tokens.masked_fill(~mask, 0)
            role_ids = role_ids.masked_fill(~mask, 0)

        assert (
            (tokens.min() >= 0) and (tokens.max() < self.filler_emb.num_embeddings)
        ), 'tokens must be dense filler ids in [0, num_fillers); signal padding via `mask`'
        assert (
            (role_ids.min() >= 0) and (role_ids.max() < self.role_emb.num_embeddings)
        ), 'role ids out of range for this role scheme'

        return self.readout(self.state(tokens, role_ids, mask))

class DiscoverDecoder(Module):
    """token-level transformer as a bottleneck: input sequence -> one encoding vector -> output sequence

    one causal `TransformerWrapper` does both halves, with shared weights:

        encode:  tokens (b, n) -> transformer -> final layer hiddens (b, n, d) -> one position -> E (b, d)
        decode:  [E, bos, y_0, ..., y_{m-2}] -> same transformer -> logits for y_0, ..., y_{m-1}

    the decoding pass sees only E, never the input tokens, so E has to carry the whole
    sequence. E is the vector `TPREncoder` is fitted to and `surgery` edits
    """

    def __init__(
        self,
        *,
        decoder: TransformerWrapper,    # causal transformer used for both encoding and decoding (weights shared). its width is `dim`
        num_fillers: int,               # symbols are ids [0, num_fillers) - only used to validate the inputs to `surgery`
        bos_id: int,                    # start token of the decoding pass, placed right after the prepended encoding
        pad_id: int,                    # written past each row's end by `decode`. `surgery` also uses it to find each row's valid positions
        max_seq_len: int,               # longest output `decode` may produce (the list length, not the transformer's context length)
        enc_pos: int | str = 'last'     # which position's hidden becomes E - 'last' = last valid token of each row, or a fixed int index
    ):
        super().__init__()
        self.decoder = decoder
        self.num_fillers = num_fillers
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.enc_pos = enc_pos

    @property
    def dim(self) -> int:
        """width of the transformer, hence of the encodings E"""
        return self.decoder.attn_layers.dim

    def encode(
        self,
        tokens: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        """encoding of each input sequence - hidden state of the transformer at the chosen position

        tokens: (b, n), mask: (b, n) bool, True = valid, defaults to all valid
        returns E: (b, d)

        the hidden comes from `last_layer_hiddens`, the output of the attention stack after
        its final norm (what `to_logits` would read). nothing is detached - `forward` trains
        the transformer through this path
        """
        batch, seq_len = tokens.shape
        device = tokens.device

        mask = default(
            mask,
            torch.ones(
                batch,
                seq_len,
                device = device,
                dtype = torch.bool
            )
        )

        assert mask.dtype == torch.bool, 'mask must be a boolean tensor'
        mask = mask.bool()

        # the logits are computed by the wrapper but unused - only the hiddens matter here

        logits, intermediates = self.decoder(
            tokens,
            mask = mask,
            return_intermediates = True
        )
        hiddens = intermediates.last_layer_hiddens      # (b, n, d)

        # 'last': index of each row's final valid token, mask.sum - 1 (assumes right padding,
        # i.e. a contiguous valid prefix). with causal attention that is the position whose
        # hidden has seen the whole sequence. otherwise one fixed position for every row

        pos = mask.sum(dim = -1) - 1 if self.enc_pos == 'last' else torch.full(
            (batch,),
            int(self.enc_pos),
            device = device,
            dtype = torch.long
        )

        # gather one position per row: (b, n, d) with indices (b, 1) -> (b, 1, d)

        encodings = batched_index_select(
            hiddens,
            rearrange(pos, 'b -> b 1'),
            dim = 1
        )
        return rearrange(encodings, 'b 1 d -> b d')

    @torch.no_grad()
    def decode(
        self,
        encodings: Tensor,
        lengths: Tensor | None = None,
        *,
        eos_id: int | None = None,
        max_len: int | None = None
    ) -> Tensor:
        """
        greedily decode from the (possibly edited) encodings, prepended as embeddings to the
        autoregressive wrapper; terminate each row either at its exact `lengths` or at the first
        `eos_id` token (the rest of the row is padded)

        encodings: (b, d)
        lengths:   (b,) output length per row. exactly one of `lengths` / `eos_id` must be given
        returns:   (b, t) token ids with pad_id past each row's end, where t = lengths.max(),
                   or with `eos_id`, t = `max_len` (default `max_seq_len`). with `eos_id` the
                   eos token itself is kept

        note: with `eos_id`, `AutoregressiveWrapper.generate` stops as soon as every row has
        emitted eos and then returns fewer than t columns, but the final padding mask below is
        built at width t. if all rows finish after k steps with 1 < k < t this raises a shape
        mismatch (k = 1 only works because the single column broadcasts). also, an eos_id equal
        to bos_id is matched against the bos prompt itself, so generation stops after one step
        """
        assert exists(lengths) ^ exists(eos_id), 'provide either `lengths` or `eos_id`'

        batch = encodings.shape[0]
        device = encodings.device

        # every row is generated to the longest requested length; shorter rows are padded afterwards

        total_len = lengths.max().item() if exists(lengths) else default(max_len, self.max_seq_len)

        assert total_len <= self.max_seq_len

        # E goes in as a one-position embedding sequence ahead of the bos prompt, (b, d) -> (b, 1, d).
        # the wrapper concatenates it after the token embedding stage, so E gets no token or
        # absolute positional embedding of its own

        prompts = torch.full((batch, 1), self.bos_id, device = device)
        prepend_embeds = rearrange(encodings, 'b d -> b 1 d')

        # a throwaway wrapper around the same network, just for its generation loop.
        # temperature 0. = greedy argmax. with kv caching off, every step reruns the whole
        # [E, bos, ...] sequence. the returned ids exclude the bos prompt

        autoregressive = AutoregressiveWrapper(self.decoder)

        outputs = autoregressive.generate(
            prompts,
            total_len,
            eos_token = eos_id,
            temperature = 0.,
            cache_kv = False,
            prepend_embeds = prepend_embeds,
            prepend_mask = torch.ones(batch, 1, device = device, dtype = torch.bool)
        )

        # row length = index of the first eos + 1 (the eos is kept), or the full length if none

        if exists(eos_id):
            is_eos = outputs == eos_id
            lengths = torch.where(
                is_eos.any(dim = -1),
                is_eos.float().argmax(dim = -1) + 1,
                torch.full((batch,), total_len, device = device)
            )

        # positions (1, t) vs lengths (b, 1) -> (b, t) mask of everything past each row's end

        return outputs.masked_fill(
            rearrange(torch.arange(total_len, device = device), 't -> 1 t') >= rearrange(lengths, 'b -> b 1'),
            self.pad_id
        )

    @torch.no_grad()
    def surgery(
        self,
        tokens: Tensor,
        edited_tokens: Tensor,
        role_ids_old: Tensor,
        role_ids_new: Tensor,
        tpr: TPREncoder
    ) -> Tensor:
        """
        constituent surgery: retake the target encoding and add the TPR prediction for the edited
        sequence, minus the prediction for the original (exact for any readout; with a linear
        readout this is the paper's per-pair edit), then decode; role assignments are supplied by
        the caller (adjacent pair roles also change under `wickel`)

        tokens, edited_tokens:      (b, n) filler ids, right padded with `pad_id` at the same positions
        role_ids_old, role_ids_new: (b, n) role ids for each, under the scheme `tpr` was fitted with
        returns:                    (b, n) decoded ids, pad_id past each row's length

        with T the TPR, x the original and x' the edited sequence, the decoded vector is
        E(x) + T(x') - T(x) = T(x') + (E(x) - T(x)), i.e. the TPR's prediction for the edited
        sequence plus whatever part of the real encoding the TPR does not explain. with the
        identity readout the bias cancels, and for a single substitution whose role stays the
        same the edit is exactly B ((f_new - f_old) outer r)
        """
        device = tokens.device

        assert edited_tokens.shape == tokens.shape == role_ids_old.shape == role_ids_new.shape, 'one token and role id per position for both sequences'

        mask = tokens != self.pad_id        # (b, n) valid positions, inferred from the pad id
        assert (
            (mask == (edited_tokens != self.pad_id)).all()
        ), 'edits must fall inside each sequence (never on padded positions)'
        assert (
            (tokens.masked_select(mask).min() >= 0) and (tokens.masked_select(mask).max() < self.num_fillers)
        ), 'tokens must be dense filler ids within their length'
        assert (
            (edited_tokens.masked_select(mask).min() >= 0) and (edited_tokens.masked_select(mask).max() < self.num_fillers)
        ), 'edited tokens must be dense filler ids'

        # the network's real encoding of the unedited sequence - this is what gets edited

        encodings = self.encode(
            tokens,
            mask
        )

        # `state` does not sanitize, so pad ids are swapped for filler 0 before the lookups (the
        # mask then drops those terms). role ids are passed as given and must already be in range
        # at padded positions (`role_ids_for_tokens` puts 0 there)

        state_old = tpr.state(
            tokens.masked_fill(~mask, 0),
            role_ids_old,
            mask
        )
        state_new = tpr.state(
            edited_tokens.masked_fill(~mask, 0),
            role_ids_new,
            mask
        )

        # difference of full readouts rather than of states, so this is also the TPR-predicted
        # change under an MLP readout. (b, d)

        edits = tpr.readout(state_new) - tpr.readout(state_old)

        # decode to each row's own length - a substitution never changes lengths

        return self.decode(
            encodings + edits,
            lengths = mask.sum(dim = -1)
        )

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None
    ) -> Tensor:
        """encode the input sequences; if targets are given, teacher-force decode them from E and return logits

        tokens:  (b, n). every position is treated as valid (no mask is taken), so a batch
                 should hold sequences of a single length - the training script samples one
                 length per step
        targets: (b, m) output token ids
        returns: E (b, d) without targets, else logits (b, m, num_tokens) aligned with `targets`

        E is not detached, so cross entropy on these logits trains the transformer both as the
        decoder and, through the prepended encoding, as the encoder. in train mode the wrapper's
        embedding dropout, applied after the concat, also acts on E
        """
        batch = tokens.shape[0]
        device = tokens.device

        encodings = self.encode(tokens)

        if not exists(targets):
            return encodings

        # teacher forcing input: bos followed by the targets shifted right by one

        decode_ids = torch.cat(
            (torch.full((batch, 1), self.bos_id, device = device), targets[:, :-1]),
            dim = 1
        )

        # the transformer sees [E, bos, t_0, ..., t_{m-2}], length m + 1

        prepend_embeds = rearrange(encodings, 'b d -> b 1 d')

        logits = self.decoder(
            decode_ids,
            prepend_embeds = prepend_embeds,
            prepend_mask = torch.ones(batch, 1, device = device, dtype = torch.bool)
        )

        # drop the output at E's slot. the output at bos predicts t_0, ..., the one at t_{m-2}
        # predicts t_{m-1} -> (b, m, num_tokens)

        return logits[:, 1:]
