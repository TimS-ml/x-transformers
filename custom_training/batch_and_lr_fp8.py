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

# pip install --no-build-isolation transformer_engine[pytorch]
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

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
        
        # Try to find the run by display name
        runs = api.runs("x-transformers-tuning-practice", 
                      filters={"display_name": wandb_run_name})
        runs_list = list(runs)
        
        wandb_run = None
        if len(runs_list) > 0:
            wandb_run = runs_list[0]
            print(f"Found wandb run by display name: {wandb_run_name}")
        else:
            # Try to use the run name as run ID directly
            try:
                wandb_run = api.run(f"x-transformers-tuning-practice/{wandb_run_name}")
                print(f"Found wandb run by ID: {wandb_run_name}")
            except:
                pass
        
        if wandb_run:
            wandb_config = wandb_run.config
            config = {
                'batch_size': wandb_config.get('batch_size', 4),
                'learning_rate': wandb_config.get('learning_rate', 1e-4),
                'gradient_accumulate_every': wandb_config.get('gradient_accumulate_every', 1)
            }
            print(f"Loaded config from wandb: batch_size={config['batch_size']}, lr={config['learning_rate']}, grad_accum={config['gradient_accumulate_every']}")
            return config
    except Exception as e:
        print(f"Failed to load config from wandb: {e}")
    
    # Return default config if all else fails
    print("Using default config")
    return {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1
    }

# Scenario 1: Only RUN - start new training
if RUN.value is not None and RUN_NAME.value is None:
    BATCH_SIZE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['batch_size']
    LEARNING_RATE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['learning_rate']
    GRADIENT_ACCUMULATE_EVERY = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['gradient_accumulate_every']
    resolved_checkpoint_file = None
    print(f"Scenario 1: Starting new training with RUN {RUN.value} config")

# Scenario 2: RUN_NAME + RUN - load checkpoint + override with RUN config
elif RUN_NAME.value is not None and RUN.value is not None:
    # Find checkpoint file
    potential_dir_path = os.path.join(CHECKPOINTS_BASE_DIR, RUN_NAME.value)
    if os.path.isdir(potential_dir_path):
        RESUME_TARGET_DIR = potential_dir_path
    else:
        all_items = os.listdir(CHECKPOINTS_BASE_DIR)
        matching_dirs = [d for d in all_items if d.startswith(RUN_NAME.value) and os.path.isdir(os.path.join(CHECKPOINTS_BASE_DIR, d))]
        RESUME_TARGET_DIR = os.path.join(CHECKPOINTS_BASE_DIR, matching_dirs[0]) if matching_dirs else None
    
    resolved_checkpoint_file = resume_checkpoint(RESUME_TARGET_DIR) if RESUME_TARGET_DIR else None
    
    # Use RUN config (override checkpoint config)
    BATCH_SIZE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['batch_size']
    LEARNING_RATE = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['learning_rate']
    GRADIENT_ACCUMULATE_EVERY = PARAM_SETS_BATCH_AND_LR[int(RUN.value)]['gradient_accumulate_every']
    print(f"Scenario 2: Loading checkpoint from {RUN_NAME.value} but using RUN {RUN.value} config")

# Scenario 3: RUN_NAME only - load checkpoint + use saved config
elif RUN_NAME.value is not None and RUN.value is None:
    # Find checkpoint file
    potential_dir_path = os.path.join(CHECKPOINTS_BASE_DIR, RUN_NAME.value)
    if os.path.isdir(potential_dir_path):
        RESUME_TARGET_DIR = potential_dir_path
    else:
        all_items = os.listdir(CHECKPOINTS_BASE_DIR)
        matching_dirs = [d for d in all_items if d.startswith(RUN_NAME.value) and os.path.isdir(os.path.join(CHECKPOINTS_BASE_DIR, d))]
        RESUME_TARGET_DIR = os.path.join(CHECKPOINTS_BASE_DIR, matching_dirs[0]) if matching_dirs else None
    
    resolved_checkpoint_file = resume_checkpoint(RESUME_TARGET_DIR) if RESUME_TARGET_DIR else None
    
    if resolved_checkpoint_file:
        # Load config from checkpoint or wandb
        saved_config = load_run_config_from_checkpoint_or_wandb(resolved_checkpoint_file)
        BATCH_SIZE = saved_config.get('batch_size', 4)
        LEARNING_RATE = saved_config.get('learning_rate', 1e-4)
        GRADIENT_ACCUMULATE_EVERY = saved_config.get('gradient_accumulate_every', 1)
        print(f"Scenario 3: Loading checkpoint and config from {RUN_NAME.value}")
    else:
        raise ValueError(f"Could not find checkpoint for RUN_NAME: {RUN_NAME.value}")

# Scenario 4: Neither RUN nor RUN_NAME - error
else:
    raise ValueError("Must provide either RUN or RUN_NAME parameter")

# Calculate derived values
FORMATTED_LR = f"{LEARNING_RATE:.0e}"
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024
SAVE_EVERY = 1000
PRECISION = "fp8"
post_fix += f"_{PRECISION}"

# Calculate NUM_BATCHES after all config is finalized
TOTAL_TRAINING_TOKENS = 1e9  # 1B tokens
TOKENS_PER_BATCH = BATCH_SIZE * SEQ_LEN
NUM_BATCHES = int(TOTAL_TRAINING_TOKENS / TOKENS_PER_BATCH)

print(f"Final config: BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE}, GRADIENT_ACCUMULATE_EVERY={GRADIENT_ACCUMULATE_EVERY}, NUM_BATCHES={NUM_BATCHES}")

# Setup run name and directories
if resolved_checkpoint_file:
    run_name = os.path.basename(os.path.dirname(resolved_checkpoint_file))
    wandb_run_id = run_name
    CHECKPOINT_DIR = os.path.dirname(resolved_checkpoint_file)
    tprint(f"Resuming run '{run_name}' from checkpoint: {resolved_checkpoint_file}")
else:
    current_time = datetime.now().strftime("%y%m%d_%H%M")
    base_run_name = f"lr_{FORMATTED_LR}_bs_{BATCH_SIZE}{post_fix}"
    run_name = f"{current_time}_{base_run_name}"
    wandb_run_id = run_name
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", run_name)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tprint(f"Starting new training run: {run_name}")

# New constants for monitoring
LOG_OPTIMIZER_STATS_EVERY = 100  # Log optimizer stats every 100 steps
LOG_THROUGHPUT_EVERY = 10  # Log throughput every 10 steps

PROJECT_NAME = "x-transformers-tuning-practice"

wandb.init(
    project=PROJECT_NAME,
    name=run_name,
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
    
    wandb.config.update({"resumed_from_checkpoint": resolved_checkpoint_file, "resumed_step": start_step}, allow_val_change=True)

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

# Create FP8 recipe configuration
fp8_recipe = recipe.DelayedScaling(
    fp8_format=recipe.Format.HYBRID,  # Use hybrid format: E4M3 for forward pass, E5M2 for backward pass
    amax_history_len=16,              # Keep amax history for 16 steps
    amax_compute_algo="max"           # Use max algorithm for amax computation
)

# Training
for i in tqdm.tqdm(range(start_step, NUM_BATCHES), mininterval=10., desc='training', initial=start_step, total=NUM_BATCHES):
    model.train()
    
    batch_start_time = time.time()
    accumulated_loss = 0

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        # Use FP8 autocast context
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            # Forward pass
            loss = model(next(train_loader))
        
        # Backward pass (outside fp8_autocast context)
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()
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
    
    # Clip gradient norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # Log optimizer statistics
    if i % LOG_OPTIMIZER_STATS_EVERY == 0:
        optim_stats = calculate_optimizer_stats(optim)
        metrics.update(optim_stats)

    # Update parameters
    optim.step()
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
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
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

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
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
            'training_config': {
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'gradient_accumulate_every': GRADIENT_ACCUMULATE_EVERY,
                'seq_len': SEQ_LEN,
                'model_dim': MODEL_DIM,
                'model_depth': MODEL_DEPTH,
                'model_heads': MODEL_HEADS
            }
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Optionally, save checkpoint as a wandb artifact
        # artifact = wandb.Artifact(name=f"{run_name}-step_{i}", type="model")
        # artifact.add_file(checkpoint_path)
        # wandb.log_artifact(artifact)

# Finish wandb run at the end of the script
wandb.finish()
