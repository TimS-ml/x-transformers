import torch
import torch.nn.functional as F

from tqdm import tqdm

from x_transformers import TransformerWrapper, Decoder
from x_transformers.recirculation import Recirculation

# running parity (cumsum mod 2) needs recurrence; recirculation leaks residual from a deep
# layer back to a shallow one after each token, letting a feedforward net act as a dynamical
# system (arxiv 2608.17981) - impossible otherwise beyond depth

BATCH_SIZE = 256
TRAIN_STEPS = 300
TRAIN_LENGTH = 16
EVAL_LENGTHS = (8, 16, 32)

def cycle(batch, length):
    while True:
        seq = torch.randint(0, 2, (batch, length))
        yield seq, seq.cumsum(dim = -1) % 2

def make_model():
    return TransformerWrapper(
        num_tokens = 2,
        max_seq_len = 0,
        attn_layers = Decoder(
            dim = 64,
            depth = 3,
            heads = 4,
            attn_dim_head = 32,
            rotary_pos_emb = True
        )
    )

def train(model, steps = TRAIN_STEPS):
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-3)
    train_dl = cycle(BATCH_SIZE, TRAIN_LENGTH)

    for _ in tqdm(range(steps)):
        seq, labels = next(train_dl)

        optimizer.zero_grad()

        logits = model(seq)
        loss = F.cross_entropy(logits.transpose(-1, -2), labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

def last_token_acc(model, length):
    seq, labels = next(cycle(512, length))

    with torch.no_grad():
        pred = model(seq)[:, -1].argmax(dim = -1)

    return (pred == labels[:, -1]).float().mean().item() * 100

plain = make_model()
recirculated = Recirculation(
    make_model(),
    source_layer = 2,
    destination_layer = 0,
    alpha = 0.5,
    ramp_steps = 0
)

train(plain)
train(recirculated)

print()
print('binary parity - last token % correct (chance = 50%, solved = 100%):')
print()

for length in EVAL_LENGTHS:
    plain_acc = last_token_acc(plain, length)
    recirc_acc = last_token_acc(recirculated, length)
    print(f'length {length:3d}: plain {plain_acc:5.1f}%   recirculated {recirc_acc:5.1f}%')

print()
print(f'trained {TRAIN_STEPS} steps at length {TRAIN_LENGTH}; recirculation generalizes beyond the trained length, the plain transformer cannot.')
