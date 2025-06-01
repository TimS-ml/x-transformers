from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import os
import time
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

import wandb
import numpy as np
from datetime import datetime

from boring_utils.utils import cprint, tprint
from boring_utils.nn_utils import (
    cycle, resume_checkpoint, 
    calculate_optimizer_stats, calculate_throughput
)
from boring_utils.helpers import DEBUG, ContextVar
from param_set import *

RUN = ContextVar("RUN", None) 
RUN_NAME = ContextVar("RUN_NAME", None) 
SIZE = ContextVar("SIZE", 0)

post_fix = ""
if SIZE.value == 0:
    PARAM_SETS_BATCH_AND_LR = PARAM_SETS_BATCH_AND_LR_19M
    PROJECT_NAME = "x-transformers-tuning-practice_19M"
    post_fix += "_19M"
    MODEL_DIM = 512
    MODEL_DEPTH = 6
    MODEL_HEADS = 8
    TOTAL_TRAINING_TOKENS = 380e6
elif SIZE.value == 1:
    PARAM_SETS_BATCH_AND_LR = PARAM_SETS_BATCH_AND_LR_64M
    PROJECT_NAME = "x-transformers-tuning-practice"
    post_fix += "_64M"
    MODEL_DIM = 640
    MODEL_DEPTH = 12
    MODEL_HEADS = 10
    TOTAL_TRAINING_TOKENS = 1280e6

# ========================================
# Configuration Loading Logic
# ========================================

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
CHECKPOINTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

def load_run_config_from_checkpoint_or_wandb(checkpoint_file):
    """Try to load training config from checkpoint or wandb API"""
    checkpoint = torch.load(checkpoint_file)
    
    # Try checkpoint first
    if 'training_config' in checkpoint:
        config = checkpoint['training_config']
        print(f"Loaded config from checkpoint: batch_size={config.get('batch_size')}, lr={config.get('learning_rate')}, grad_accum={config.get('gradient_accumulate_every')}")
        return config
    
    # Fallback to wandb API
    try:
        api = wandb.Api()
        checkpoint_dir = os.path.dirname(checkpoint_file)
        wandb_run_name = os.path.basename(checkpoint_dir)
        
        # Try to find run by display name or ID
        wandb_run = None
        runs = api.runs(f"{PROJECT_NAME}", filters={"display_name": wandb_run_name})
        runs_list = list(runs)
        
        if len(runs_list) > 0:
            wandb_run = runs_list[0]
        else:
            try:
                wandb_run = api.run(f"{PROJECT_NAME}/{wandb_run_name}")
            except:
                pass
        
        if wandb_run and wandb_run.config:
            config = {
                'batch_size': wandb_run.config.get('batch_size'),
                'learning_rate': float(wandb_run.config.get('learning_rate', 0)),
                'gradient_accumulate_every': wandb_run.config.get('gradient_accumulate_every')
            }
            print(f"Loaded config from wandb API: batch_size={config['batch_size']}, lr={config['learning_rate']}, grad_accum={config['gradient_accumulate_every']}")
            return config
    except Exception as e:
        print(f"Failed to fetch config from wandb API: {e}")
    
    return None

# Validate input parameters
if RUN.value is None and RUN_NAME.value is None:
    raise ValueError("Must specify either RUN or RUN_NAME parameter!")

# Determine checkpoint file
resolved_checkpoint_file = None
if RUN_NAME.value:
    potential_dir_path = os.path.join(CHECKPOINTS_BASE_DIR, RUN_NAME.value)
    
    if os.path.isdir(potential_dir_path):
        RESUME_TARGET_DIR = potential_dir_path
        print(f"Found exact match for checkpoint directory: {RESUME_TARGET_DIR}")
    else:
        # Try as prefix
        all_items = os.listdir(CHECKPOINTS_BASE_DIR)
        matching_dirs = [d for d in all_items if d.startswith(RUN_NAME.value) and os.path.isdir(os.path.join(CHECKPOINTS_BASE_DIR, d))]
        
        if matching_dirs:
            matching_dirs.sort()
            RESUME_TARGET_DIR = os.path.join(CHECKPOINTS_BASE_DIR, matching_dirs[0])
            print(f"Found checkpoint directory by prefix '{RUN_NAME.value}': {RESUME_TARGET_DIR}")
        else:
            raise ValueError(f"Could not find checkpoint directory for RUN_NAME '{RUN_NAME.value}'")
    
    resolved_checkpoint_file = resume_checkpoint(RESUME_TARGET_DIR)
    if not resolved_checkpoint_file:
        raise ValueError(f"No valid checkpoint found in directory: {RESUME_TARGET_DIR}")

# Load training configuration based on the four scenarios
if RUN.value is not None and RUN_NAME.value is not None:
    # Scenario: RUN_NAME + RUN - load checkpoint + use RUN's config
    print(f"Scenario: RUN_NAME + RUN - Loading checkpoint from {RUN_NAME.value} with RUN {RUN.value} config")
    run_config = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]  # Convert to int
    BATCH_SIZE = run_config['batch_size']
    LEARNING_RATE = run_config['learning_rate']
    GRADIENT_ACCUMULATE_EVERY = run_config['gradient_accumulate_every']
    print(f"Using RUN {RUN.value} config: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, grad_accum={GRADIENT_ACCUMULATE_EVERY}")

elif RUN.value is not None:
    # Scenario: Only RUN - start new training
    print(f"Scenario: RUN only - Starting new training with RUN {RUN.value} config")
    run_config = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]  # Convert to int
    BATCH_SIZE = run_config['batch_size']
    LEARNING_RATE = run_config['learning_rate']
    GRADIENT_ACCUMULATE_EVERY = run_config['gradient_accumulate_every']
    print(f"Using RUN {RUN.value} config: batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, grad_accum={GRADIENT_ACCUMULATE_EVERY}")

elif RUN_NAME.value is not None:
    # Scenario: Only RUN_NAME - load checkpoint + checkpoint's config
    print(f"Scenario: RUN_NAME only - Loading checkpoint and config from {RUN_NAME.value}")
    saved_config = load_run_config_from_checkpoint_or_wandb(resolved_checkpoint_file)
    try:
        BATCH_SIZE = saved_config.get('batch_size')
        LEARNING_RATE = saved_config.get('learning_rate')
        GRADIENT_ACCUMULATE_EVERY = saved_config.get('gradient_accumulate_every')
    except:
        raise ValueError(f"No valid config found in directory: {RUN_NAME.vale}")


FORMATTED_LR = f"{LEARNING_RATE:.0e}"

# Setup run names and directories
if resolved_checkpoint_file:
    run_name = os.path.basename(os.path.dirname(resolved_checkpoint_file))
    wandb_run_id = run_name
    CHECKPOINT_DIR = os.path.dirname(resolved_checkpoint_file)
    print(f"Resuming run '{run_name}' from checkpoint: {resolved_checkpoint_file}")
else:
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    base_run_name = f"lr_{FORMATTED_LR}_bs_{BATCH_SIZE}{post_fix}"
    run_name = f"{current_time}_{base_run_name}"
    wandb_run_id = run_name
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", run_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Starting new training run: {run_name}")

# Constants
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024
SAVE_EVERY = 1000  # Save checkpoint every 1000 steps

PRECISION = "fp16"
post_fix += f"_{PRECISION}"

# New constants for monitoring
LOG_OPTIMIZER_STATS_EVERY = 100  # Log optimizer stats every 100 steps
LOG_THROUGHPUT_EVERY = 10  # Log throughput every 10 steps

# instantiate GPT-like decoder model
model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(
        dim = MODEL_DIM, 
        depth = MODEL_DEPTH, 
        heads = MODEL_HEADS, 
        rotary_pos_emb = True
    )
)

model = AutoregressiveWrapper(model)
model.cuda()

# Count number of devices
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

# ========================================
# Load model state and optimizer
# ========================================
start_step = 0
if resolved_checkpoint_file:
    checkpoint = torch.load(resolved_checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_step = checkpoint.get('step', 0) + 1
    print(f"Successfully loaded checkpoint. Resuming from step {start_step}.")

# Create optimizer after final learning rate is determined
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load optimizer state if resuming from checkpoint
if resolved_checkpoint_file:
    if 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state from checkpoint.")

# Calculate NUM_BATCHES after checkpoint loading with final parameters
# NOTE: we ignore the GRADIENT_ACCUMULATE_EVERY in calc here
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN
NUM_BATCHES = int(TOTAL_TRAINING_TOKENS / TOKENS_PER_BATCH)

cprint(BATCH_SIZE, LEARNING_RATE, GRADIENT_ACCUMULATE_EVERY, NUM_BATCHES)

# Initialize wandb after all parameters are finalized
wandb.init(
    project=PROJECT_NAME, # Project name in wandb
    name=run_name, # Add the dynamic run name here
    id=wandb_run_id,
    config={
        "learning_rate": FORMATTED_LR,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "gradient_accumulate_every": GRADIENT_ACCUMULATE_EVERY,
        "num_batches": NUM_BATCHES,
        "model_dim": MODEL_DIM,
        "model_depth": MODEL_DEPTH,
        "model_heads": MODEL_HEADS,
        "rotary_pos_emb": True,
        "precision": PRECISION
    },
    resume='allow' if resolved_checkpoint_file else 'never'
)

# Update wandb config if resumed from checkpoint
if resolved_checkpoint_file:
    wandb.config.update({"resumed_from_checkpoint": resolved_checkpoint_file, "resumed_step": start_step}, allow_val_change=True)


# ========================================
# helpers
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# prepare enwik data
# data_file_path = os.path.join(PROJECT_ROOT, 'data', 'enwik8.gz')
# # train data: 90M -> 90M tokens (chars)
# # test data: 5M
# with gzip.open(data_file_path) as file:
#     data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
#     train_x, valid_x = np.split(data, [int(90e6)])
#     data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

# train data: 320M tokens (chars)
# test data: 20M
data_file_path = os.path.join(PROJECT_ROOT, 'data', 'enwik9.gz')
with gzip.open(data_file_path) as file:
    data = np.frombuffer(file.read(int(340e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(data, [int(320e6)])
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


scaler = GradScaler()

# Training
for i in tqdm.tqdm(range(start_step, NUM_BATCHES), mininterval=10., desc='training', initial=start_step, total=NUM_BATCHES):
    model.train()
    
    batch_start_time = time.time()
    accumulated_loss = 0

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        with autocast(device_type='cuda'):
            loss = model(next(train_loader))
        
        # backward
        scaler.scale(loss / GRADIENT_ACCUMULATE_EVERY).backward()
        accumulated_loss += loss.item()

    # Calculate training metrics
    train_loss_val = accumulated_loss / GRADIENT_ACCUMULATE_EVERY
    train_perplexity_val = np.exp(train_loss_val)
    train_bpc_val = train_loss_val / np.log(2)  # Bits Per Character
    
    # Prepare all metrics to log
    metrics = {
        "train/loss": train_loss_val,
        "train/perplexity": train_perplexity_val,
        "train/bpc": train_bpc_val
    }
    
    # Unscale gradients
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # Log optimizer statistics
    if i % LOG_OPTIMIZER_STATS_EVERY == 0:
        optim_stats = calculate_optimizer_stats(optim)
        metrics.update(optim_stats)

    # Update parameters
    scaler.step(optim)
    scaler.update()
    
    # Update parameters
    # optim.step()
    optim.zero_grad()
    
    # End timing and log throughput
    batch_end_time = time.time()
    if i % LOG_THROUGHPUT_EVERY == 0:
        throughput_metrics = calculate_throughput(
            BATCH_SIZE, 
            SEQ_LEN, 
            batch_start_time, 
            batch_end_time, 
            GRADIENT_ACCUMULATE_EVERY,
            num_devices
        )
        metrics.update(throughput_metrics)

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            val_loss_val = loss.item()
            val_perplexity_val = np.exp(val_loss_val)
            val_bpc_val = val_loss_val / np.log(2)
            
            metrics.update({
                "val/loss": val_loss_val,
                "val/perplexity": val_perplexity_val,
                "val/bpc": val_bpc_val
            })
    
    # Log all metrics at once
    wandb.log(metrics, step=i)
    
    print(f'training loss: {train_loss_val}')

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
