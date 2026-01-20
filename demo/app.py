import streamlit as st
import pandas as pd
import numpy as np
import torch
from datetime import timedelta
import plotly.express as px
from sklearn.preprocessing import RobustScaler
import time
# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="HPC Energy Prediction Demo",
    layout="wide"
)

# =========================
# MODEL DEFINITION
# =========================
import torch.nn as nn
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError


class DeepMLP(BaseNeuralNetwork):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        output_dim: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__(input_dim, output_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


# =========================
# DATA & MODEL LOADING
# =========================
def load_data():
    hist = pd.read_csv(
        "historical_predictions_new.csv",
        parse_dates=["submit_time", "end_time"],
    )
    live = pd.read_csv(
        "final_live_jobs_new.csv",
        parse_dates=["submit_time", "end_time"],
    )
    return hist, live


@st.cache_resource(show_spinner=False)
def load_model(input_dim):
    model = DeepMLP(
        input_dim=75,
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3379305532460404,
    )
    model.load_state_dict(
        torch.load("new_model.pt", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


# =========================
# INITIAL LOAD
# =========================
hist_df, live_df = load_data()

MODEL_FEATURES = [
    c for c in live_df.columns
    if c not in ["submit_time", "end_time"]
]

model = load_model(len(MODEL_FEATURES))


# =========================
# TITLE & DESCRIPTION
# =========================
st.title("⚡ HPC Job Energy Consumption Prediction")

st.markdown("""
**Objective**  
Predict the **mean, minimum and maximum power consumption** of HPC jobs *at submission time*.

**What this demo shows**
- Historical model accuracy over 5 months
- Live simulation of job arrivals
- Expected hourly energy consumption
- Job-level inspection
""")


# =========================
# HISTORICAL PLOT
# =========================
st.subheader("📊 Historical Performance (Predicted vs Real)")

metric = st.selectbox("Select metric", ["mean", "min", "max"])

fig_hist = px.line(
    hist_df.sort_values("submit_time"),
    x="submit_time",
    y=[
        f"node_power_{metric}",
        f"predicted_job_{metric}_power_consumption",
    ],
)

st.plotly_chart(fig_hist, use_container_width=True)


# =========================
# SESSION STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.speed = 50
    st.session_state.virtual_time = live_df.submit_time.min()
    st.session_state.predictions = pd.DataFrame()
    st.session_state.processed_jobs = set()

SIM_START = live_df.submit_time.min()
SIM_END = live_df.submit_time.max()


# =========================
# CONTROLS
# =========================
st.subheader("⏱ Live Simulation Controls")

c1, c2, c3, c4 = st.columns(4)

if c1.button("▶ Play"):
    st.session_state.running = True

if c2.button("⏸ Pause"):
    st.session_state.running = False

st.session_state.speed = c3.slider(
    "Speed (×)", 1, 100, st.session_state.speed
)

if c4.button("🔄 Reset"):
    st.session_state.running = False
    st.session_state.virtual_time = SIM_START
    st.session_state.predictions = pd.DataFrame()
    st.session_state.processed_jobs = set()

st.markdown(f"**Simulated time:** `{st.session_state.virtual_time}`")


# =========================
# PLACEHOLDERS
# =========================
st.subheader("📈 Expected Energy Consumption")
live_plot = st.empty()

st.subheader("📋 Live Job Table")
live_table = st.empty()


# =========================
# SIMULATION STEP
# =========================
def simulation_step():
    vt = st.session_state.virtual_time

    new_jobs = live_df[
        (live_df.submit_time <= vt)
        & (~live_df.index.isin(st.session_state.processed_jobs))
    ]

    if not new_jobs.empty:
        X = new_jobs[MODEL_FEATURES].drop(
            columns=[
                "job_mean_power_consumption",
                "job_min_power_consumption",
                "job_max_power_consumption",
            ],
            errors="ignore",
        )

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            preds_tensor = model(X_tensor)
            preds = preds_tensor.detach().cpu().numpy()

        preds_df = new_jobs.copy()
        preds_df["pred_mean"] = preds[:, 0]
        preds_df["pred_min"] = preds[:, 1]
        preds_df["pred_max"] = preds[:, 2]

        st.session_state.predictions = pd.concat(
            [st.session_state.predictions, preds_df]
        )
        st.session_state.processed_jobs.update(new_jobs.index)

    st.session_state.virtual_time += timedelta(
        minutes=st.session_state.speed
    )


# =========================
# LIVE AGGREGATION
# =========================
if not st.session_state.predictions.empty:
    preds = st.session_state.predictions

    active = preds[
        (preds.submit_time <= st.session_state.virtual_time)
        & (preds.end_time >= st.session_state.virtual_time)
    ]

    hourly = (
        active.assign(interval=lambda x: x.submit_time.dt.floor("h"))
        .groupby("interval")["pred_mean"]
        .sum()
        .reset_index()
    )
    
    # Filter to show only last 2 hours
    two_hours_ago = st.session_state.virtual_time - timedelta(hours=8)
    hourly_filtered = hourly[hourly["interval"] >= two_hours_ago]

    fig_live = px.line(hourly_filtered, x="interval", y="pred_mean")
    live_plot.plotly_chart(fig_live, use_container_width=True)


# =========================
# LIVE JOB TABLE
# =========================
if not st.session_state.predictions.empty:
    table = st.session_state.predictions.copy()

    table["real_mean_visible"] = np.where(
        table.end_time <= st.session_state.virtual_time,
        table.get("job_mean_power_consumption", np.nan),
        np.nan,
    )

    display_cols = [
        "submit_time",
        "end_time",
        "pred_mean",
        "pred_min",
        "pred_max",
        "real_mean_visible",
    ]

    live_table.dataframe(
        table[display_cols].sort_values(
            "submit_time", ascending=False
        ),
        use_container_width=True,
    )
else:
    live_table.info("No jobs yet.")


# =========================
# TICK (NON-BLOCKING)
# =========================
if st.session_state.running and st.session_state.virtual_time < SIM_END:
    simulation_step()
    time.sleep(0.1)  # Gives the UI a moment to breathe
    st.rerun()
