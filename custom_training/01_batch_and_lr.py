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
import os

import wandb
import numpy as np

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024

# Model parameters (for wandb config)
MODEL_DIM = 512
MODEL_DEPTH = 6
MODEL_HEADS = 8
SEQ_LEN = 1024

# Initialize wandb
# Generate a run name based on learning rate and batch size
run_name = f"lr_{LEARNING_RATE}_bs_{BATCH_SIZE}"

wandb.init(
    project="x-transformers-tuning-practice", # Project name in wandb
    name=run_name, # Add the dynamic run name here
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "gradient_accumulate_every": GRADIENT_ACCUMULATE_EVERY,
        "num_batches": NUM_BATCHES,
        "model_dim": MODEL_DIM,
        "model_depth": MODEL_DEPTH,
        "model_heads": MODEL_HEADS,
        "rotary_pos_emb": True # As per model definition
    }
)

# helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model
model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(
        dim = MODEL_DIM, # Use constant for clarity
        depth = MODEL_DEPTH, # Use constant for clarity
        heads = MODEL_HEADS, # Use constant for clarity
        rotary_pos_emb = True
    )
)

model = AutoregressiveWrapper(model)
model.cuda()

# prepare enwik8 data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
data_file_path = os.path.join(project_root, 'data', 'enwik8.gz')

with gzip.open(data_file_path) as file:
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
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        # Accumulate gradients
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    # Calculate training metrics
    train_loss_val = loss.item()
    train_perplexity_val = np.exp(train_loss_val)
    train_bpc_val = train_loss_val / np.log(2) # Bits Per Character

    # Log training metrics to wandb
    wandb.log({
        "train_loss": train_loss_val,
        "train_perplexity": train_perplexity_val,
        "train_bpc": train_bpc_val,
        "step": i
    })
    
    print(f'training loss: {train_loss_val}') # Keep print for immediate feedback
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            val_loss_val = loss.item()
            val_perplexity_val = np.exp(val_loss_val)
            val_bpc_val = val_loss_val / np.log(2)

            # Log validation metrics to wandb
            wandb.log({
                "val_loss": val_loss_val,
                "val_perplexity": val_perplexity_val,
                "val_bpc": val_bpc_val,
                "step": i
            })
            print(f'validation loss: {val_loss_val}') # Keep print for immediate feedback

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1] # Get a sample from validation set
        prime = decode_tokens(inp)
        # Corrected f-string usage for print
        print(f'{prime} \n\n {"*" * 100}')

        sample = model.generate(
            prompts = inp.unsqueeze(0), # Add batch dimension for generate function
            seq_len = GENERATE_LENGTH,
            cache_kv = True # Enable KV caching for faster generation
        )

        output_str = decode_tokens(sample[0].tolist()) # sample is a tensor, convert to list for decode_tokens
        print(output_str)

        # Log generated text to wandb table
        # Create a new table for each generation step or append to a global one
        # For simplicity, creating a new table log each time.
        generated_table = wandb.Table(columns=["step", "prime_text", "generated_text"])
        generated_table.add_data(i, prime, output_str)
        wandb.log({"generated_samples": generated_table, "step": i})

# Finish wandb run at the end of the script
wandb.finish()