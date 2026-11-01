# src/data_loader.py
"""Data loading and feature engineering utilities."""
import numpy as np
import pandas as pd
from src.config import DATA_PATH, TARGET_COL


def load_and_engineer(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV and add cyclical hour features."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.drop(columns=["hour"])
    return df


def temporal_split(df: pd.DataFrame, split_ratio: float = 0.8):
    """Temporal train/test split."""
    split_point = int(len(df) * split_ratio)
    return df.iloc[:split_point], df.iloc[split_point:]


def create_windowed_dataset(X_data: np.ndarray, y_data: np.ndarray, look_back: int):
    """Create (samples, timesteps, features) arrays for RNN training."""
    X, y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i : (i + look_back), :])
        y.append(y_data[i + look_back])
    return np.array(X), np.array(y)
