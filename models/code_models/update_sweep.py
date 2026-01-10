"""
Script to update an existing WandB sweep with additional hyperparameter values.
W&B doesn't allow direct modification of sweeps, so we create a new sweep with updated config.
"""

import wandb
from pprint import pprint

# The ID of your existing sweep
EXISTING_SWEEP_ID = "wr0rkuhe"
PROJECT_NAME = "DenseNN-Sweep"

def get_existing_sweep_config(sweep_id):
    """Retrieve the configuration of an existing sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{PROJECT_NAME}/{sweep_id}")
    print(f"\nExisting sweep config for {sweep_id}:")
    pprint(sweep.config)
    return sweep.config


def create_updated_sweep():
    """
    Create a new sweep with additional hyperparameter values.
    Copy the existing sweep config and add new values.
    """
    # Get existing sweep config
    api = wandb.Api()
    existing_sweep = api.sweep(f"{PROJECT_NAME}/{EXISTING_SWEEP_ID}")
    existing_config = existing_sweep.config
    
    # Create updated config by adding new hyperparameter values
    updated_config = {
        'method': existing_config.get('method', 'bayes'),
        'metric': existing_config.get('metric', {'name': 'val_loss', 'goal': 'minimize'}),
        'parameters': {
            'batch_size': {
                'values': existing_config['parameters']['batch_size']['values'] + [256, 4096]  # Add new batch sizes
            },
            'dropout': existing_config['parameters']['dropout'],  # Keep existing distribution
            'epochs': existing_config['parameters']['epochs'],
            'hidden_dims': {
                'values': existing_config['parameters']['hidden_dims']['values'] + [
                    [1024, 512, 256, 128],  # Add new architectures
                    [256, 128, 64, 32]
                ]
            },
            'lr': existing_config['parameters']['lr'],
            'patience': existing_config['parameters']['patience'],
            'weight_decay': existing_config['parameters']['weight_decay']
        }
    }
    
    print("\nUpdated sweep config:")
    pprint(updated_config)
    
    # Create the new sweep
    new_sweep_id = wandb.sweep(updated_config, project=PROJECT_NAME)
    print(f"\nNew sweep created with ID: {new_sweep_id}")
    print(f"Original sweep ID: {EXISTING_SWEEP_ID}")
    print(f"New sweep ID: {new_sweep_id}")
    
    return new_sweep_id


def list_sweep_runs(sweep_id):
    """List all runs in a sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{PROJECT_NAME}/{sweep_id}")
    print(f"\nRuns in sweep {sweep_id}:")
    for run in sweep.runs:
        print(f"  - {run.name}: val_loss={run.summary.get('val_loss', 'N/A')}")


if __name__ == "__main__":
    print("="*60)
    print("W&B Sweep Update Helper")
    print("="*60)
    
    # Get existing sweep config
    existing_config = get_existing_sweep_config(EXISTING_SWEEP_ID)
    
    # List existing runs
    list_sweep_runs(EXISTING_SWEEP_ID)
    
    # Create updated sweep
    new_sweep_id = create_updated_sweep()
    
    print("\n" + "="*60)
    print("To run the new sweep:")
    print(f"python train_densenn_sweep.py --sweep-id {new_sweep_id}")
    print("="*60)
