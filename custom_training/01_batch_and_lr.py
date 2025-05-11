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
from datetime import datetime

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 1024
SEQ_LEN = 1024
SAVE_EVERY = 1000  # Save checkpoint every 1000 steps

# --- Resume Configuration ---
# Set to a specific checkpoint file path to resume, e.g., "checkpoints/20231027_1530_lr_0.0001_bs_4/model_step_1000.pt"
# Set to None or empty string to train from scratch.
RESUME_FROM_CHECKPOINT = None # Or path to checkpoint file
# --- End Resume Configuration ---

# Model parameters (for wandb config)
MODEL_DIM = 512
MODEL_DEPTH = 6
MODEL_HEADS = 8
SEQ_LEN = 1024

# Initialize wandb
# Generate a run name based on learning rate, batch size, and current time
current_time = datetime.now().strftime("%Y%m%d_%H%M")
base_run_name = f"lr_{LEARNING_RATE}_bs_{BATCH_SIZE}"
run_name = f"{current_time}_{base_run_name}"

# Define checkpoint directory using the full run_name
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints", run_name)
# Create directory only if not resuming from a checkpoint that would have already created it
# or if RESUME_FROM_CHECKPOINT is None (training from scratch)
if not RESUME_FROM_CHECKPOINT:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
elif RESUME_FROM_CHECKPOINT and not os.path.exists(os.path.dirname(RESUME_FROM_CHECKPOINT)):
    # If resuming but the specific checkpoint's parent dir doesn't exist (e.g. typo in path), create current run's dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

# optimizer (defined before potential checkpoint loading)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Load Checkpoint if RESUME_FROM_CHECKPOINT is set ---
start_step = 0
if RESUME_FROM_CHECKPOINT and os.path.exists(RESUME_FROM_CHECKPOINT):
    print(f"Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}")
    checkpoint = torch.load(RESUME_FROM_CHECKPOINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint.get('step', 0) + 1 # Resume from the next step
    # Update CHECKPOINT_DIR to the resumed run's directory if different
    # This ensures new checkpoints are saved in the resumed run's folder
    resumed_run_checkpoint_dir = os.path.dirname(RESUME_FROM_CHECKPOINT)
    if CHECKPOINT_DIR != resumed_run_checkpoint_dir:
        print(f"Updating checkpoint directory to resumed run's directory: {resumed_run_checkpoint_dir}")
        CHECKPOINT_DIR = resumed_run_checkpoint_dir
    
    print(f"Successfully loaded checkpoint. Resuming from step {start_step}.")
    # Log to wandb that we are resuming
    wandb.config.update({"resumed_from_checkpoint": RESUME_FROM_CHECKPOINT, "resumed_step": start_step}, allow_val_change=True)
elif RESUME_FROM_CHECKPOINT:
    print(f"Warning: Checkpoint path {RESUME_FROM_CHECKPOINT} not found. Training from scratch.")
# --- End Load Checkpoint ---

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

# training

for i in tqdm.tqdm(range(start_step, NUM_BATCHES), mininterval=10., desc='training', initial=start_step, total=NUM_BATCHES):
    model.train()

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
        artifact = wandb.Artifact(name=f"{run_name}-step_{i}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

# Finish wandb run at the end of the script
wandb.finish()