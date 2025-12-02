import wandb
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# Import W&B module for Sklearn
from models.training_utils import SklearnTrainer
from data import X_train, y_train, X_val, y_val

# W&B project name
project_name = "DS-HPE"

# Function to build the XGBoost model with hyperparameters from W&B config
def build_model(config):
    base = xgb.XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        random_state=42
    )
    return MultiOutputRegressor(base)

# Function to perform training within a W&B sweep
def sweep_train():
    wandb.init()
    config = wandb.config

    model = build_model(config)

    # Convert DataFrames to NumPy arrays
    X_train_np = X_train.to_numpy()
    X_val_np   = X_val.to_numpy()

    y_train_np = y_train.to_numpy()
    y_val_np   = y_val.to_numpy()


    trainer = SklearnTrainer(
        model=model,
        model_name="XGB",
        project_name=project_name
    )

    trainer.train(
        X_train_np, y_train_np,
        X_val_np, y_val_np,
        config=config
    )


if __name__ == "__main__":
    sweep_train()
