# /// script
# dependencies = [
#   "tqdm",
#   "x-transformers",
#   "wandb",
#   "fire",
#   "accelerate"
# ]
# ///

"""
Training script for character-level language modeling on the enwik8 dataset.

This script trains a GPT-like transformer model on the enwik8 benchmark dataset,
which consists of the first 100M bytes of Wikipedia XML. The model learns to predict
the next byte in a sequence, treating text as a sequence of raw bytes (0-255).

Features:
- Character-level (byte-level) language modeling with 256-token vocabulary
- Transformer decoder with rotary positional embeddings (RoPE)
- Orthogonal projected values for attention (experimental feature)
- Gradient accumulation for effective larger batch sizes
- Weights & Biases (wandb) integration for experiment tracking
- Periodic validation and text generation for monitoring training progress

The enwik8 dataset is a standard benchmark for evaluating character-level language models.
"""

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import fire
import wandb
from accelerate import Accelerator

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(loader):
    """
    Infinitely cycle through a DataLoader.

    This generator function wraps a DataLoader and yields batches indefinitely,
    restarting from the beginning when the dataset is exhausted. This is useful
    for training loops that don't want to manually handle epoch boundaries.

    Args:
        loader: PyTorch DataLoader to cycle through

    Yields:
        Batches of data from the loader, cycling infinitely
    """
    while True:
        for data in loader:
            yield data

def decode_token(token):
    """
    Decode a single byte token to its character representation.

    Converts a byte value (0-255) to its corresponding ASCII/Unicode character.
    Values below 32 (control characters) are clamped to 32 (space) to ensure
    readable output and avoid terminal control issues.

    Args:
        token: Integer token value (0-255) representing a byte

    Returns:
        String containing the single decoded character
    """
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    """
    Decode a sequence of byte tokens to a string.

    Converts a sequence of byte values to their string representation by
    applying decode_token to each element and concatenating the results.

    Args:
        tokens: Iterable of integer token values (0-255)

    Returns:
        Decoded string representation of the token sequence
    """
    return ''.join(list(map(decode_token, tokens)))

def train(
    num_batches = int(1e5),
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 500,
    generate_length = None,
    seq_len = 1024,
    track_experiment_online = False,
    run_name = 'baseline',
    cpu = False
):
    accelerator = Accelerator(cpu=cpu)
    device = accelerator.device

    generate_length = default(generate_length, seq_len)

    # instantiate GPT-like decoder model

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = seq_len,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8,
            rotary_pos_emb = False,
            polar_pos_emb = True,
            pre_and_post_norm = True
        )
    )

    model = AutoregressiveWrapper(model)

    # prepare enwik8 data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq.to(device)

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # experiment

    wandb.init(project = 'enwik8', mode = 'online' if track_experiment_online else 'disabled')
    wandb.run.name = run_name

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval=10., desc='training'):
        model.train()

        for _ in range(gradient_accumulate_every):
            loss = model(next(train_loader))
            accelerator.backward(loss / gradient_accumulate_every)

        print(f'training loss: {loss.item()}')
        if accelerator.is_main_process:
            wandb.log(dict(loss = loss.item()))

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))

                print(f'validation loss: {loss.item()}')
                if accelerator.is_main_process:
                    wandb.log(dict(valid_loss = loss.item()))

        if i % generate_every == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp.cpu().numpy())
            print(f'%s \n\n %s' % (prime, '*' * 100))

            sample = model.generate(
                prompts = inp,
                seq_len = generate_length,
                cache_kv = True
            )

            output_str = decode_tokens(sample.cpu().numpy())
            print(output_str)

if __name__ == '__main__':
    fire.Fire(train)
