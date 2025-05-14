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
import re
import time

import wandb
import numpy as np
from datetime import datetime

from boring_utils.utils import cprint, tprint
from boring_utils.nn_utils import (
    cycle, resume_checkpoint, 
    log_optimizer_stats, log_throughput
)

from param_set import PARAM_SETS_BATCH_AND_LR


# constants
# 19M -> req 380M tokens data
MODEL_DIM = 512
MODEL_DEPTH = 6
MODEL_HEADS = 8

# 151.3M -> req 3B tokens data, enwik9 (~1B) is not enough
# MODEL_DIM = 1024
# MODEL_DEPTH = 12
# MODEL_HEADS = 16

# post_fix = ""
# post_fix = f"_dim_{MODEL_DIM}_depth_{MODEL_DEPTH}_heads_{MODEL_HEADS}"
post_fix = "_19M"

RUN_ID = 1

NUM_BATCHES = int(1e5)
BATCH_SIZE = PARAM_SETS_BATCH_AND_LR[RUN_ID]['batch_size']
GRADIENT_ACCUMULATE_EVERY = PARAM_SETS_BATCH_AND_LR[RUN_ID]['gradient_accumulate_every']
LEARNING_RATE = PARAM_SETS_BATCH_AND_LR[RUN_ID]['learning_rate']
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024
SAVE_EVERY = 1000  # Save checkpoint every 1000 steps

# New constants for monitoring
LOG_OPTIMIZER_STATS_EVERY = 100  # Log optimizer stats every 100 steps
LOG_THROUGHPUT_EVERY = 10  # Log throughput every 10 steps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

# Set to a specific checkpoint file path to resume, e.g., "checkpoints/20231027_1530_lr_0.0001_bs_4/model_step_1000.pt"
# RESUME_FROM_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', '250511_0948_lr_0.0001_bs_4')
RESUME_FROM_CHECKPOINT = None

resolved_checkpoint_file = resume_checkpoint(RESUME_FROM_CHECKPOINT)

if resolved_checkpoint_file:
    # Derive run_name from the checkpoint's parent directory
    run_name = os.path.basename(os.path.dirname(resolved_checkpoint_file))
    wandb_run_id = run_name  # NOTE: Assumes run_name was used as ID for the original run
    CHECKPOINT_DIR = os.path.dirname(resolved_checkpoint_file)
    print(f"Attempting to resume run '{run_name}' from checkpoint: {resolved_checkpoint_file}")
else:
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    base_run_name = f"lr_{LEARNING_RATE}_bs_{BATCH_SIZE}{post_fix}"
    run_name = f"{current_time}_{base_run_name}"
    wandb_run_id = run_name
    CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints", run_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Starting new training run: {run_name}")

wandb.init(
    project="x-transformers-tuning-practice", # Project name in wandb
    name=run_name, # Add the dynamic run name here
    id=wandb_run_id,
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
    },
    resume='allow' if resolved_checkpoint_file else 'never'
)

# helpers
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

# Count number of devices
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

# optimizer (defined before potential checkpoint loading)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Load Checkpoint if RESUME_FROM_CHECKPOINT is set ---
start_step = 0
if resolved_checkpoint_file:
    checkpoint = torch.load(resolved_checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint.get('step', 0) + 1
    
    print(f"Successfully loaded checkpoint. Resuming from step {start_step}.")
    wandb.config.update({"resumed_from_checkpoint": resolved_checkpoint_file, "resumed_step": start_step}, allow_val_change=True)
# --- End Load Checkpoint ---

# prepare enwik8 data
data_file_path = os.path.join(PROJECT_ROOT, 'data', 'enwik8.gz')

# train data: 90M -> 90M tokens (chars)
# test data: 5M
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

# training
for i in tqdm.tqdm(range(start_step, NUM_BATCHES), mininterval=10., desc='training', initial=start_step, total=NUM_BATCHES):
    model.train()
    
    # Start timing for throughput calculation
    batch_start_time = time.time()

    accumulated_loss = 0 # To store loss for checkpointing if needed before validation
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        # Accumulate gradients
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()
        accumulated_loss += loss.item()

    # Calculate training metrics
    train_loss_val = accumulated_loss / GRADIENT_ACCUMULATE_EVERY
    train_perplexity_val = np.exp(train_loss_val)
    train_bpc_val = train_loss_val / np.log(2) # Bits Per Character

    # Log training metrics to wandb
    wandb.log({
        "train/loss": train_loss_val,
        "train/perplexity": train_perplexity_val,
        "train/bpc": train_bpc_val,
        "step": i
    })
    
    print(f'training loss: {train_loss_val}') # Keep print for immediate feedback
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # Log optimizer statistics
    if i % LOG_OPTIMIZER_STATS_EVERY == 0:
        log_optimizer_stats(optim, wandb, i)
    
    # Update parameters
    optim.step()
    optim.zero_grad()
    
    # End timing and log throughput
    batch_end_time = time.time()
    if i % LOG_THROUGHPUT_EVERY == 0:
        log_throughput(
            BATCH_SIZE, 
            SEQ_LEN, 
            batch_start_time, 
            batch_end_time, 
            GRADIENT_ACCUMULATE_EVERY, 
            wandb, 
            i,
            num_devices
        )
    
    print(f'training loss: {train_loss_val}') # Keep print for immediate feedback

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            val_loss_val = loss.item()
            val_perplexity_val = np.exp(val_loss_val)
            val_bpc_val = val_loss_val / np.log(2)

            # Log validation metrics to wandb
            wandb.log({
                "val/loss": val_loss_val,
                "val/perplexity": val_perplexity_val,
                "val/bpc": val_bpc_val,
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

    # Save checkpoint
    if i % SAVE_EVERY == 0 and i > 0: # Also check i > 0 to avoid saving at step 0 if SAVE_EVERY is small
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_step_{i}.pt")
        torch.save({
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': train_loss_val, # Save the most recent training loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Optionally, save checkpoint as a wandb artifact
        # artifact = wandb.Artifact(name=f"{run_name}-step_{i}", type="model")
        # artifact.add_file(checkpoint_path)
        # wandb.log_artifact(artifact)

# Finish wandb run at the end of the script
wandb.finish()