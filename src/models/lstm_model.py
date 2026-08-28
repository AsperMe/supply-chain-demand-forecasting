
from __future__ import annotations

import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.duckdb"
MODEL_DIR = ROOT / "data" / "models"

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        logger.info("Using Apple Metal (MPS) acceleration — M2 Pro detected.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA GPU.")
        return torch.device("cuda")
    else:
        logger.info("Using CPU (no GPU/MPS available).")
        return torch.device("cpu")


DEVICE = get_device()

# Features used as LSTM inputs
LSTM_FEATURES = [
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
    "day_of_week", "is_weekend", "month",
    "snap_active", "has_event", "is_promotion",
    "sell_price", "rolling_nonzero_rate_28",
]

SEQ_LEN = 28      
HIDDEN  = 128
N_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3


# Dataset 
class SalesDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,     
        seq_len: int = SEQ_LEN,
        stride: int = 7,
    ):
        self.seq_len = seq_len
        self.X, self.y = [], []
        for start in range(0, len(data) - seq_len, stride):
            window = data[start : start + seq_len]
            target = data[start + seq_len, -1]   # sales at next step
            self.X.append(window[:, :-1])          # all features
            self.y.append(target)
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# Model 
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden: int = HIDDEN, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.ReLU(),   # Sales are non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last = out[:, -1, :]           # (batch, hidden)
        return self.head(last).squeeze(-1)  


def _prepare_series_data(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, StandardScaler]:
    """Prepare feature matrix for LSTM. Returns (data_array, scaler)."""
    available = [c for c in LSTM_FEATURES if c in df.columns]
    X = df[available].fillna(0).values.astype(np.float32)
    y = df["sales"].fillna(0).values.astype(np.float32).reshape(-1, 1)

    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return np.hstack([X, y]), scaler


def train_lstm(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[LSTMForecaster, StandardScaler, list[str]]:
    """
    Train a single global LSTM on all provided series.

    Returns:
        (trained_model, feature_scaler, feature_names)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading data for LSTM ({len(series_ids):,} series)…")

    ids_str = ", ".join(f"'{i}'" for i in series_ids)
    df = con.execute(
        f"SELECT * FROM features WHERE id IN ({ids_str}) ORDER BY id, date"
    ).df().dropna(subset=["sales_lag_56"])

    available_features = [c for c in LSTM_FEATURES if c in df.columns]
    logger.info(f"  Features: {available_features}")

    # Fit scaler on all data
    scaler = StandardScaler()
    X_all = df[available_features].fillna(0).values.astype(np.float32)
    scaler.fit(X_all)

    # Build datasets per series, concatenate
    all_data = []
    for sid in tqdm(series_ids[:500], desc="Preparing LSTM windows"):  # cap at 500 for speed
        sub = df[df["id"] == sid].sort_values("date")
        if len(sub) < SEQ_LEN + 10:
            continue
        data, _ = _prepare_series_data(sub, scaler, fit_scaler=False)
        all_data.append(data)

    if not all_data:
        raise ValueError("No series had enough data for LSTM training.")

    combined = np.vstack(all_data)
    dataset = SalesDataset(combined, seq_len=SEQ_LEN, stride=7)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    logger.info(f"  Total windows: {len(dataset):,}")

    # ── Model ──
    input_dim = len(available_features)
    model = LSTMForecaster(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss()

    logger.info(f"Training LSTM on {DEVICE} for {epochs} epochs…")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f} — LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save model + scaler
    torch.save(model.state_dict(), MODEL_DIR / "lstm_model.pt")
    with open(MODEL_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "lstm_features.pkl", "wb") as f:
        pickle.dump(available_features, f)

    logger.success("LSTM training complete. Model saved.")
    return model, scaler, available_features


def predict_lstm(
    con: duckdb.DuckDBPyConnection,
    series_ids: list[str],
    model: LSTMForecaster | None = None,
    scaler: StandardScaler | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate in-sample predictions from trained LSTM.

    Returns DataFrame: id, date, lstm_point
    """
    # Load if not provided
    if model is None:
        with open(MODEL_DIR / "lstm_features.pkl", "rb") as f:
            feature_names = pickle.load(f)
        with open(MODEL_DIR / "lstm_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        input_dim = len(feature_names)
        model = LSTMForecaster(input_dim=input_dim).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / "lstm_model.pt", map_location=DEVICE))

    model.eval()
    all_preds = []

    ids_str = ", ".join(f"'{i}'" for i in series_ids)
    df = con.execute(
        f"SELECT * FROM features WHERE id IN ({ids_str}) ORDER BY id, date"
    ).df().dropna(subset=["sales_lag_56"])

    for sid in tqdm(series_ids, desc="LSTM inference"):
        sub = df[df["id"] == sid].sort_values("date")
        if len(sub) < SEQ_LEN:
            continue
        data, _ = _prepare_series_data(sub, scaler, fit_scaler=False)

        preds_out = []
        with torch.no_grad():
            for t in range(SEQ_LEN, len(data)):
                window = torch.tensor(
                    data[t - SEQ_LEN : t, :-1], dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)
                pred = model(window).item()
                preds_out.append(max(0.0, pred))

        result = sub.iloc[SEQ_LEN:][["date"]].copy()
        result["id"] = sid
        result["lstm_point"] = preds_out
        all_preds.append(result)

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    logger.success(f"LSTM predictions: {len(combined):,} rows.")
    return combined


if __name__ == "__main__":
    con = duckdb.connect(str(DB_PATH))
    sample = con.execute(
        "SELECT DISTINCT id FROM demand_classes WHERE demand_class='Smooth' LIMIT 50"
    ).df()["id"].tolist()
    model, scaler, feats = train_lstm(con, sample, epochs=5)
    preds = predict_lstm(con, sample[:5], model, scaler, feats)
    print(preds.tail(10))
    con.close()
