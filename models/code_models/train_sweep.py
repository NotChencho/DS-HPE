import wandb
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_models import SimpleMLP, DeepMLP, AttentionMLP
from training_utils import PyTorchTrainer

import sys
from pathlib import Path


RootPath = Path(__file__).resolve().parents[2]

DataPath = RootPath / "data" / "interm"

def load_data(batch_size):

    train_ds = torch.load(DataPath / "train_dataset.pt")
    val_ds = torch.load(DataPath / "val_dataset.pt")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    )



def build_model(config, input_dim=75):
    if config.model_type == "SimpleMLP":
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dims=[config.hidden_dim, config.hidden_dim // 2],
            dropout=config.dropout
        )

    elif config.model_type == "DeepMLP":
        return DeepMLP(
            input_dim=input_dim,
            hidden_dims=[
                config.hidden_dim,
                config.hidden_dim // 2,
                config.hidden_dim // 4,
                max(16, config.hidden_dim // 8)
            ],
            dropout=config.dropout
        )

    elif config.model_type == "AttentionMLP":
        return AttentionMLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


def main():
    run = wandb.init()
    config = wandb.config

    train_loader, val_loader = load_data(batch_size=config.batch_size)
    model = build_model(config, input_dim=75)

    trainer = PyTorchTrainer(
        model=model,
        model_name=f"{config.model_type}_sweep",
        project_name="Test",
        entity="iqbalch-universidad-carlos-iii-de-madrid"
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        config=dict(config)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    main()
