# train_catboost_sweep.py

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import wandb

from data import X_train, y_train, X_val, y_val  # Recover our data 
from models.training_utils import SklearnTrainer    


# We list categorical columns by name, as CatBoost can handle them directly
CAT_COLS = [
    "group",
"time_limit_cat",
"dow",
"dom",
"hour",
"is_weekend",
"month",
"is_night",
"is_peak",
"is_shared_job",
"has_req_nodes",
"num_tasks_missing_or_inconsistent",
"has_req_threads_per_core",

]

# Get their indices in the DataFrame
cat_indices = [X_train.columns.get_loc(c) for c in CAT_COLS]


def build_model(config):
    base = CatBoostRegressor(
        loss_function="RMSE",
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bylevel=config.colsample_bylevel,
        l2_leaf_reg=config.l2_leaf_reg,
        random_strength=config.random_strength,
        random_seed=42,
        verbose=False,
        # task_type="GPU",  
    )
    return MultiOutputRegressor(base)


def sweep_train():
    # W&B sets config via the sweep
    wandb.init(project="DS-HPE")
    config = wandb.config

    # Keep X as DataFrames for CatBoost (so cat_features works)
    X_train_df = X_train
    X_val_df   = X_val

    # y as numpy arrays
    y_train_np = y_train.to_numpy()
    y_val_np   = y_val.to_numpy()

    # Indices of categorical columns
    cat_indices = [X_train_df.columns.get_loc(c) for c in CAT_COLS]

    model = build_model(config)

    trainer = SklearnTrainer(
        model=model,
        model_name="CatBoost_sweep_run",
        project_name="DS-HPE",
        entity="100496657-universidad-carlos-iii-de-madrid",  # or None for personal
    )

    # config logged in W&B; cat_features used only for training
    trainer.train(
        X_train_df, y_train_np,
        X_val_df,   y_val_np,
        config=dict(config),
        cat_features=cat_indices,
    )


if __name__ == "__main__":
    sweep_train()