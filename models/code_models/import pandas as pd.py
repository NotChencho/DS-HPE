import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Importa tus modelos definidos
from torch_models import DenseNN


BASE_PATH = r"C:\Users\Jaime\Documents\GitHub\DS-HPE"


month_map = {1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10}




def preprocess_df(df):
    df["dow"] = pd.to_datetime(df["submit_time"]).dt.dayofweek
    df["dom"] = pd.to_datetime(df["submit_time"]).dt.day
    df["hour"] = pd.to_datetime(df["submit_time"]).dt.hour
    df["month"] = pd.to_datetime(df["submit_time"]).dt.month


    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] < 7) | (df["hour"] >= 22)).astype(int)
    df["is_peak"] = df["hour"].between(9, 18).astype(int)


    df.drop(columns=["submit_time"], inplace=True)


    cols_to_drop = (
        df.filter(like="dow_").columns.tolist() +
        df.filter(like="dom_").columns.tolist() +
        df.filter(like="hour_").columns.tolist()
    )
    df.drop(columns=cols_to_drop, errors="ignore", inplace=True)


    for col in ["group", "time_limit_cat"]:
        df[col] = df[col].astype("category").cat.codes


    return df




def evaluate_metrics(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    r2 = 1 - np.sum((y_true - y_pred) * 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) * 2, axis=0)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)), axis=0) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}




def automl_torch_dnn(approach,
                      batch_size=1028,
                      dropout=0.15559998021833732,
                      epochs=150,
                      hidden_dims=[512, 256, 128],
                      lr=0.0009719478103336032,
                      patience=80,
                      weight_decay=1e-5):


    feature_cols = [
        "group", "num_tasks_final", "num_tasks_missing_or_inconsistent",
        "time_limit_scaled", "time_limit_cat", "num_nodes_req",
        "has_req_nodes", "num_cores_req", "cores_per_task",
        "num_gpus_req", "mem_req", "has_req_threads_per_core",
        "is_shared_job", "dow", "dom", "hour", "is_weekend",
        "month", "is_night", "is_peak"
    ]
    target_cols = ["node_power_min", "node_power_mean", "node_power_max"]


    mape_history = {"mean_power": [], "min_power": [], "max_power": []}
    iterations = []


    scaler_X = StandardScaler()


    for i in [1, 2, 3, 4, 5, 6]:
        # ===============================
        # LOAD DATA
        # ===============================
        if approach == "accumulative":
            df = pd.read_csv(f"{BASE_PATH}/my_dataframe{i}.csv", low_memory=False)
            df = preprocess_df(df)
            X = df[feature_cols].values
            y = df[target_cols].values


            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.vstack([y_train, y_val])
            iterations.append(f"{i} months")


        elif approach == "pairs":
            if i < 6:
                df = pd.read_csv(f"{BASE_PATH}/my_dataframe2{i}.csv", low_memory=False)
                df = preprocess_df(df)
                X = df[feature_cols].values
                y = df[target_cols].values


                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


                X_train_full = np.vstack([X_train, X_val])
                y_train_full = np.vstack([y_train, y_val])
                iterations.append(f"pair {month_map[i]}-{month_map[i+1]}")
            else:
                continue


        elif approach == "future-testing":
            if i < 6:
                df_train = pd.read_csv(f"{BASE_PATH}/my_dataframe{i}.csv", low_memory=False)
                df_test = pd.read_csv(f"{BASE_PATH}/my_dataframe3{i+1}.csv", low_memory=False)
                iterations.append(f"acc.{month_map[i]}-0{month_map[i+1]}")
            elif i == 6:
                df_train = pd.read_csv(f"{BASE_PATH}/my_dataframe{i}.csv", low_memory=False)
                df_test = pd.read_csv(f"{BASE_PATH}/my_dataframe3{i}.csv", low_memory=False)
                iterations.append(f"train-test: all acc.-0{month_map[i]}")


            df_train = preprocess_df(df_train)
            df_test = preprocess_df(df_test)
            X_train_full = df_train[feature_cols].values
            y_train_full = df_train[target_cols].values
            X_test = df_test[feature_cols].values
            y_test = df_test[target_cols].values


        else:
            raise ValueError("Unknown approach")


        # ===============================
        # SCALE DATA
        # ===============================
        X_train_full = scaler_X.fit_transform(X_train_full)
        X_test = scaler_X.transform(X_test)


        # ===============================
        # TORCH DATA
        # ===============================
        X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


        # ===============================
        # MODEL
        # ===============================
        model = DenseNN(input_dim=X_train_full.shape[1], hidden_dims=hidden_dims, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()


        # ===============================
        # EARLY STOPPING SETUP
        # ===============================
        best_loss = float('inf')
        patience_counter = 0


        # ===============================
        # TRAIN
        # ===============================
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)


            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    model.load_state_dict(best_model_state)
                    break


        # ===============================
        # EVALUATE
        # ===============================
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor)
        metrics = evaluate_metrics(y_test_tensor, y_pred_test)
        mape_history["min_power"].append(metrics["mape"][0])
        mape_history["mean_power"].append(metrics["mape"][1])
        mape_history["max_power"].append(metrics["mape"][2])


    # ===============================
    # PLOT
    # ===============================
    plt.figure()
    plt.plot(iterations, mape_history["mean_power"], marker="o", label="Mean Power")
    plt.plot(iterations, mape_history["min_power"], marker="o", label="Min Power")
    plt.plot(iterations, mape_history["max_power"], marker="o", label="Max Power")
    plt.xlabel("Iteration")
    plt.ylabel("MAPE (%)")
    plt.title(f"MAPE Evolution - {approach}")
    plt.legend()
    plt.grid(True)
    plt.show()


automl_torch_dnn("accumulative")
automl_torch_dnn("pairs")
automl_torch_dnn("future-testing")
for a in ['accumulative', 'pairs', 'future-testing']:
    automl_torch_dnn(a)