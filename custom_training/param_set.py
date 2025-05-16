# Experiment: batch size and learning rate
PARAM_SETS_BATCH_AND_LR_19M = [
    # Default values
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    },
    # Keep batch size, adjust learning rate
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 3e-4
    },
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 7e-5
    },
    # Batch size 8 combinations
    {
        'batch_size': 8,
        'gradient_accumulate_every': 2,
        'learning_rate': 1e-4
    },
    {
        'batch_size': 8,
        'gradient_accumulate_every': 2,
        'learning_rate': 3e-4
    },
    {
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 2e-4
    },
    # Batch size 16 combinations
    {
        'batch_size': 16,
        'gradient_accumulate_every': 1,
        'learning_rate': 1e-4
    },
    {
        'batch_size': 16,
        'gradient_accumulate_every': 2,
        'learning_rate': 2e-4
    },
    {
        'batch_size': 16,
        'gradient_accumulate_every': 4,
        'learning_rate': 4e-4
    },
    # Batch size 32 combinations
    {
        'batch_size': 32,
        'gradient_accumulate_every': 1,
        'learning_rate': 2e-4
    },
    {
        'batch_size': 32,
        'gradient_accumulate_every': 2,
        'learning_rate': 4e-4
    }
]

PARAM_SETS_BATCH_AND_LR_64M = [
    # Default values
    {
        'batch_size': 2,
        'gradient_accumulate_every': 8,
        'learning_rate': 5e-5
    },
    # Keep batch size, adjust learning rate
    {
        'batch_size': 2,
        'gradient_accumulate_every': 8,
        'learning_rate': 1e-4
    },
    {
        'batch_size': 2,
        'gradient_accumulate_every': 8,
        'learning_rate': 3e-5
    },
    # Batch size 4 combinations
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 5e-5
    },
    {
        'batch_size': 4,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    },
    {
        'batch_size': 4,
        'gradient_accumulate_every': 8,
        'learning_rate': 7e-5
    },
    # Batch size 8 combinations
    {
        'batch_size': 8,
        'gradient_accumulate_every': 2,
        'learning_rate': 5e-5
    },
    {
        'batch_size': 8,
        'gradient_accumulate_every': 4,
        'learning_rate': 7e-5
    },
    {
        'batch_size': 8,
        'gradient_accumulate_every': 8,
        'learning_rate': 1e-4
    },
    # Batch size 16 combinations
    {
        'batch_size': 16,
        'gradient_accumulate_every': 2,
        'learning_rate': 7e-5
    },
    {
        'batch_size': 16,
        'gradient_accumulate_every': 4,
        'learning_rate': 1e-4
    }
]