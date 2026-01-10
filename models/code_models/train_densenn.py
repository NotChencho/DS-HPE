"""
Direct training script for DenseNN with specific hyperparameters.
No sweep - just training with your custom hyperparameters.
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

# Your hyperparameters
HYPERPARAMETERS = {
    "batch_size": 1028,
    "dropout": 0.15559998021833732,
    "epochs": 150,
    "hidden_dims": [512, 256, 128],
    "lr": 0.0009719478103336032,
    "patience": 80,
    "weight_decay": 0.00001,
}

def load_data(batch_size):
    """Load training and validation datasets."""
    # Add safe globals for PyTorch 2.6+ weights_only security
    torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
    
    train_ds = torch.load(DataPath / "train_dataset.pt")
    val_ds = torch.load(DataPath / "val_dataset.pt")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    )


def main():
    print("\n" + "="*60)
    print("Training DenseNN with custom hyperparameters")
    print("="*60)
    print(f"Batch Size: {HYPERPARAMETERS['batch_size']}")
    print(f"Dropout: {HYPERPARAMETERS['dropout']}")
    print(f"Epochs: {HYPERPARAMETERS['epochs']}")
    print(f"Hidden Dims: {HYPERPARAMETERS['hidden_dims']}")
    print(f"Learning Rate: {HYPERPARAMETERS['lr']}")
    print(f"Patience: {HYPERPARAMETERS['patience']}")
    print(f"Weight Decay: {HYPERPARAMETERS['weight_decay']}")
    print("="*60 + "\n")
    
    # Load data
    train_loader, val_loader = load_data(batch_size=HYPERPARAMETERS["batch_size"])
    
    # Build model
    input_dim = 75  # Adjust based on your data
    model = DenseNN(
        input_dim=input_dim,
        hidden_dims=HYPERPARAMETERS["hidden_dims"],
        output_dim=3,
        dropout=HYPERPARAMETERS["dropout"]
    )
    
    # Initialize trainer
    trainer = PyTorchTrainer(
        model=model,
        model_name="DenseNN-Custom",
        project_name="DenseNN-Tuning",
        entity=None  # Set your W&B entity if needed
    )
    
    # Train with your hyperparameters
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=HYPERPARAMETERS["epochs"],
        lr=HYPERPARAMETERS["lr"],
        weight_decay=HYPERPARAMETERS["weight_decay"],
        patience=HYPERPARAMETERS["patience"],
        config=HYPERPARAMETERS
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60 + "\n")
    
    # Test on test dataset
    print("="*60)
    print("Testing on test dataset...")
    print("="*60)
    torch.serialization.add_safe_globals([torch.utils.data.dataset.TensorDataset])
    test_ds = torch.load(DataPath / "test_dataset.pt")
    test_loader = DataLoader(test_ds, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False, num_workers=4, persistent_workers=True)
    
    # Re-initialize wandb for test logging (trainer finished the previous run)
    wandb.init(
        project="DenseNN-Tuning",
        name="DenseNN-Custom-Test",
        config=HYPERPARAMETERS
    )
    
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
            
            if batch_idx == 0:
                print(f"Test batch predictions shape: {output.shape}")
    
    # Calculate aggregate test metrics (average across all batches)
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
    
    print("\n" + "="*60)
    print("Testing completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
