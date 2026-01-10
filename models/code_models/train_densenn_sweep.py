"""
WandB Sweep training script for DenseNN.
Performs hyperparameter optimization using WandB sweeps.
"""

import wandb
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from torch_models import DenseNN
from training_utils import PyTorchTrainer

RootPath = Path(__file__).resolve().parents[2]
DataPath = RootPath

# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [512, 1028, 2048]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'epochs': {
            'value': 150
        },
        'hidden_dims': {
            'values': [
                [512, 256, 128],
                [256, 128, 64],
                [1024, 512, 256],
                [512, 256, 128, 64],
                [512, 256, 128, 64, 32]
            ]
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 0.000001,
            'max': 0.01
        },
        'patience': {
            'value': 80
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 0.000001,
            'max': 0.001
        }
    }
}


def load_data(batch_size):
    """Load training and validation datasets."""
    # Add safe globals for PyTorch 2.6+ weights_only security
    torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
    
    train_ds = torch.load(DataPath / "train_dataset.pt")
    val_ds = torch.load(DataPath / "val_dataset.pt")

    # Use num_workers=0 on Windows to avoid multiprocessing CUDA issues
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    )


def train_with_wandb():
    """Training function called by wandb sweep agent."""
    # Initialize wandb run
    with wandb.init() as run:
        # Get hyperparameters from wandb config
        config = wandb.config
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print("\n" + "="*60)
        print(f"Training DenseNN - Sweep Run: {run.name}")
        print("="*60)
        print(f"Batch Size: {config.batch_size}")
        print(f"Dropout: {config.dropout}")
        print(f"Epochs: {config.epochs}")
        print(f"Hidden Dims: {config.hidden_dims}")
        print(f"Learning Rate: {config.lr}")
        print(f"Patience: {config.patience}")
        print(f"Weight Decay: {config.weight_decay}")
        print("="*60 + "\n")
        
        # Load data
        train_loader, val_loader = load_data(batch_size=config.batch_size)
        
        # Build model
        input_dim = 75  # Adjust based on your data
        model = DenseNN(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=3,
            dropout=config.dropout
        )
        
        # Initialize trainer
        trainer = PyTorchTrainer(
            model=model,
            model_name=f"DenseNN-Sweep",
            project_name="DenseNN-Sweep",
            entity=None  # Set your W&B entity if needed
        )
        
        # Train with sweep hyperparameters
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            patience=config.patience,
            config=dict(config)
        )
        
        print("\n" + "="*60)
        print(f"Training completed for run: {run.name}")
        print("="*60 + "\n")
        
        # ===== TEST EVALUATION =====
        print("="*60)
        print("Testing on test dataset...")
        print("="*60)
        
        # Load test data
        torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
        test_ds = torch.load(DataPath / "test_dataset.pt")
        test_loader = DataLoader(
            test_ds, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=0  # Avoid Windows multiprocessing CUDA issues
        )
        
        # Evaluate on test set
        trainer.model.eval()
        test_loss_values = []
        test_mse_values = []
        test_rmse_values = []
        test_mae_values = []
        
        loss_fn = nn.MSELoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = trainer.model(data)
                
                # Calculate metrics per batch
                loss = loss_fn(output, target)
                mse = torch.mean((output - target) ** 2)
                rmse = torch.sqrt(mse)
                mae = torch.mean(torch.abs(output - target))
                
                test_loss_values.append(loss.item())
                test_mse_values.append(mse.item())
                test_rmse_values.append(rmse.item())
                test_mae_values.append(mae.item())
        
        # Calculate aggregate test metrics
        test_loss = np.mean(test_loss_values)
        test_mse = np.mean(test_mse_values)
        test_rmse = np.mean(test_rmse_values)
        test_mae = np.mean(test_mae_values)
        
        # Log test metrics to wandb
        wandb.log({
            "test_loss": test_loss,
            "test_mse": test_mse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "total_batches": len(test_loader)
        })
        wandb.summary.update({
            "final_test_loss": test_loss,
            "final_test_mse": test_mse,
            "final_test_rmse": test_rmse,
            "final_test_mae": test_mae
        })
        
        print(f"\nTest metrics:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Total batches: {len(test_loader)}")
        print("="*60 + "\n")


def run_test_with_best_config(sweep_id, project_name="DenseNN-Sweep"):
    """
    After sweep completes, run test evaluation with best configuration.
    Call this function manually after the sweep finishes.
    """
    # Get the best run from the sweep
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    best_run = sweep.best_run()
    
    print("\n" + "="*60)
    print("Best configuration from sweep:")
    print("="*60)
    for key, value in best_run.config.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    # Load test data
    torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
    test_ds = torch.load(DataPath / "test_dataset.pt")
    test_loader = DataLoader(
        test_ds, 
        batch_size=best_run.config['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        persistent_workers=True
    )
    
    # Rebuild model with best config
    input_dim = 75
    model = DenseNN(
        input_dim=input_dim,
        hidden_dims=best_run.config['hidden_dims'],
        output_dim=3,
        dropout=best_run.config['dropout']
    )
    
    # Load best model weights
    # You'll need to save the model path during training and load it here
    # For now, we'll retrain or you can modify trainer to save the model
    
    # Initialize wandb for test logging
    wandb.init(
        project=project_name,
        name=f"Test-{best_run.name}",
        config=best_run.config
    )
    
    model.eval()
    test_loss_values = []
    test_mse_values = []
    test_rmse_values = []
    test_mae_values = []
    
    loss_fn = nn.MSELoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            
            # Calculate metrics per batch
            loss = loss_fn(output, target)
            mse = torch.mean((output - target) ** 2)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(output - target))
            
            test_loss_values.append(loss.item())
            test_mse_values.append(mse.item())
            test_rmse_values.append(rmse.item())
            test_mae_values.append(mae.item())
    
    # Calculate aggregate test metrics
    test_loss = np.mean(test_loss_values)
    test_mse = np.mean(test_mse_values)
    test_rmse = np.mean(test_rmse_values)
    test_mae = np.mean(test_mae_values)
    
    # Log test metrics to wandb
    wandb.log({
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "total_batches": len(test_loader)
    })
    wandb.summary.update({
        "final_test_loss": test_loss,
        "final_test_mse": test_mse,
        "final_test_rmse": test_rmse,
        "final_test_mae": test_mae
    })
    
    print(f"\nTest metrics with best config:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    wandb.finish()


def main():
    """
    Main function to initialize and run the sweep.
    """
    # Create the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="DenseNN-Sweep"
    )
    
    print("\n" + "="*60)
    print(f"Sweep created with ID: {sweep_id}")
    print("="*60 + "\n")
    
    # Run the sweep agent
    # count parameter defines how many runs to execute
    wandb.agent(
        sweep_id, 
        function=train_with_wandb,
        count=10  # Number of runs to execute
    )
    
    print("\n" + "="*60)
    print("Sweep completed!")
    print(f"Sweep ID: {sweep_id}")
    print("="*60 + "\n")
    
    # Optionally run test with best config
    # Uncomment the line below to automatically test after sweep
    # run_test_with_best_config(sweep_id)


if __name__ == "__main__":
    main()
