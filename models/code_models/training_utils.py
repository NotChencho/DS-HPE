"""
Training utilities for HPC power consumption prediction.
Includes trainers for both sklearn and PyTorch models with W&B integration.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional, Tuple
import wandb
from tqdm import tqdm



class SklearnTrainer:
    """Trainer for sklearn models with W&B logging."""
    
    def __init__(self, model, model_name: str, project_name: str = "HPC-Power-Prediction", 
                 entity: Optional[str] = None):
        """
        Initialize sklearn trainer.
        
        Args:
            model: Sklearn model instance
            model_name: Name of the model for logging
            project_name: W&B project name
            entity: W&B entity (team name)
        """
        self.model = model
        self.model_name = model_name
        self.project_name = project_name
        self.entity = entity
    
    def train(self, X_train, y_train, X_val, y_val, config: Optional[Dict] = None):
        """
        Train sklearn model and log to W&B.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            config: Additional configuration to log
        
        Returns:
            Trained model and metrics
        """
        # Initialize W&B
        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=self.model_name,
            config=config or {}
        )
        
        # Train model
        print(f"Training {self.model_name}...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, train_pred, prefix="train")
        val_metrics = self._calculate_metrics(y_val, val_pred, prefix="val")
        
        # Log all metrics
        wandb.log({**train_metrics, **val_metrics})
        
        # Log summary
        wandb.summary.update({
            "best_val_rmse": val_metrics["val_rmse_mean"],
            "best_val_mae": val_metrics["val_mae_mean"],
            "best_val_r2": val_metrics["val_r2_mean"]
        })
        
        wandb.finish()
        
        return self.model, {**train_metrics, **val_metrics}
    
    def _calculate_metrics(self, y_true, y_pred, prefix: str = "") -> Dict[str, float]:
        """Calculate regression metrics for all outputs."""
        metrics = {}
        target_names = ["mean_power", "min_power", "max_power"]
        
        for i, name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            metrics[f"{prefix}_rmse_{name}"] = rmse
            metrics[f"{prefix}_mae_{name}"] = mae
            metrics[f"{prefix}_r2_{name}"] = r2
        
        # Overall metrics
        metrics[f"{prefix}_rmse_mean"] = np.mean([metrics[f"{prefix}_rmse_{name}"] for name in target_names])
        metrics[f"{prefix}_mae_mean"] = np.mean([metrics[f"{prefix}_mae_{name}"] for name in target_names])
        metrics[f"{prefix}_r2_mean"] = np.mean([metrics[f"{prefix}_r2_{name}"] for name in target_names])
        
        return metrics


class PyTorchTrainer:
    """Trainer for PyTorch models with W&B logging."""
    
    def __init__(self, model: nn.Module, model_name: str, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 project_name: str = "HPC-Power-Prediction",
                 entity: Optional[str] = None):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model instance
            model_name: Name of the model for logging
            device: Device to train on
            project_name: W&B project name
            entity: W&B entity (team name)
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.project_name = project_name
        self.entity = entity
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, lr: float = 0.001, weight_decay: float = 1e-5,
              patience: int = 10, config: Optional[Dict] = None):
        """
        Train PyTorch model with early stopping and W&B logging.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
            config: Additional configuration to log
        
        Returns:
            Trained model and best metrics
        """
        # Initialize W&B
        if self.project_name:
            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.model_name,
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "patience": patience,
                    "device": self.device,
                    **(config or {})
                }
            )
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Train
            train_metrics = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]['lr'],
                **train_metrics,
                **val_metrics
            }
            
            if wandb.run:
                wandb.log(epoch_metrics)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val RMSE: {val_metrics['val_rmse']:.4f}")
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                
                # Update best metrics in wandb
                if wandb.run:
                    wandb.summary.update({
                        "best_epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "best_val_rmse": np.sqrt(best_val_loss)
                    })
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if wandb.run:
            wandb.finish()
        
        return self.model, {"best_val_loss": best_val_loss}
    
    def _train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {
            "train_loss": avg_loss,
            "train_rmse": np.sqrt(avg_loss)
        }
    
    def _validate_epoch(self, val_loader: DataLoader, criterion) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate detailed metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        metrics = {"val_loss": avg_loss, "val_rmse": np.sqrt(avg_loss)}
        
        target_names = ["mean_power", "min_power", "max_power"]
        for i, name in enumerate(target_names):
            rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
            mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
            r2 = r2_score(all_targets[:, i], all_preds[:, i])
            
            metrics[f"val_rmse_{name}"] = rmse
            metrics[f"val_mae_{name}"] = mae
            metrics[f"val_r2_{name}"] = r2
        
        return metrics
    
    def evaluate_test_set(self, test_loader):
        """Evaluate on test set and return metrics"""
        self.model.eval()
        all_preds = []
        all_targets = []
    
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                predictions = self.model(X_batch)
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
    
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
    
        # Calculate metrics
        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)
        
        test_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Per-output metrics
        mse_per_output = []
        rmse_per_output = []
        mae_per_output = []
        r2_per_output = []
        
        for i in range(targets.shape[1]):
            mse_i = mean_squared_error(targets[:, i], preds[:, i])
            rmse_i = np.sqrt(mse_i)
            mae_i = mean_absolute_error(targets[:, i], preds[:, i])
            r2_i = r2_score(targets[:, i], preds[:, i])
            
            mse_per_output.append(mse_i)
            rmse_per_output.append(rmse_i)
            mae_per_output.append(mae_i)
            r2_per_output.append(r2_i)
        
        test_metrics['mse_per_output'] = mse_per_output
        test_metrics['rmse_per_output'] = rmse_per_output
        test_metrics['mae_per_output'] = mae_per_output
        test_metrics['r2_per_output'] = r2_per_output
        
        return test_metrics, preds


def evaluate_model(model, X_test, y_test, model_type: str = "sklearn") -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model (sklearn or PyTorch)
        X_test: Test features
        y_test: Test targets
        model_type: "sklearn" or "pytorch"
    
    Returns:
        Dictionary of test metrics and predictions
    """
    if model_type == "sklearn":
        y_pred = model.predict(X_test)
    else:  
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred = model(X_test_tensor).cpu().numpy()
    
    metrics = {}
    target_names = ["mean_power", "min_power", "max_power"]
    
    for i, name in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i])) * 100
        
        metrics[f"test_rmse_{name}"] = rmse
        metrics[f"test_mae_{name}"] = mae
        metrics[f"test_r2_{name}"] = r2
        metrics[f"test_mape_{name}"] = mape
    
    # Overall metrics
    metrics["test_rmse_mean"] = np.mean([metrics[f"test_rmse_{name}"] for name in target_names])
    metrics["test_mae_mean"] = np.mean([metrics[f"test_mae_{name}"] for name in target_names])
    metrics["test_r2_mean"] = np.mean([metrics[f"test_r2_{name}"] for name in target_names])
    metrics["test_mape_mean"] = np.mean([metrics[f"test_mape_{name}"] for name in target_names])
    
    return metrics, y_pred
