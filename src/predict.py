# src/predict.py
"""Load all saved models and run inference on the test set."""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

from src.config import (
    GRU_MODEL_PATH,
    LOOK_BACK,
    LSTM_MODEL_PATH,
    RNN_SCALER_X_PATH,
    RNN_SCALER_Y_PATH,
    STACK_MODEL_PATH,
    STACK_SCALER_PATH,
    TARGET_COL,
    TRAIN_TEST_SPLIT,
)
from src.data_loader import create_windowed_dataset, load_and_engineer, temporal_split

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_all_models():
    """Load all trained models and scalers."""
    return {
        "stacker": joblib.load(STACK_MODEL_PATH),
        "stacker_scaler": joblib.load(STACK_SCALER_PATH),
        "lstm": load_model(LSTM_MODEL_PATH, compile=False),
        "gru": load_model(GRU_MODEL_PATH, compile=False),
        "rnn_scaler_X": joblib.load(RNN_SCALER_X_PATH),
        "rnn_scaler_y": joblib.load(RNN_SCALER_Y_PATH),
    }


def get_prediction_data(models: dict | None = None) -> dict:
    """Run all models on the test split and return Chart.js-ready payload."""
    if models is None:
        models = load_all_models()

    df = load_and_engineer()
    _, test_df = temporal_split(df, TRAIN_TEST_SPLIT)

    stacker_features = [c for c in df.columns if c != TARGET_COL]
    rnn_features = df.columns.tolist()

    # RNN predictions
    X_rnn = models["rnn_scaler_X"].transform(test_df[rnn_features])
    y_rnn = models["rnn_scaler_y"].transform(test_df[[TARGET_COL]])
    X_3d, y_scaled = create_windowed_dataset(X_rnn, y_rnn.flatten(), LOOK_BACK)

    y_actual = models["rnn_scaler_y"].inverse_transform(y_scaled.reshape(-1, 1))
    y_lstm = models["rnn_scaler_y"].inverse_transform(models["lstm"].predict(X_3d, verbose=0))
    y_gru = models["rnn_scaler_y"].inverse_transform(models["gru"].predict(X_3d, verbose=0))

    # Stacker predictions (aligned)
    X_stack = models["stacker_scaler"].transform(test_df[stacker_features])
    y_stack = models["stacker"].predict(X_stack)[LOOK_BACK:]

    r2_lstm = r2_score(y_actual, y_lstm)
    r2_gru = r2_score(y_actual, y_gru)
    r2_stack = r2_score(y_actual, y_stack)
    print(f"R2 -> LSTM: {r2_lstm:.4f}  GRU: {r2_gru:.4f}  Stacker: {r2_stack:.4f}")

    labels = [f"Test Hour {i}" for i in range(len(y_actual))]
    return {
        "labels": labels,
        "actual_data": y_actual.flatten().tolist(),
        "predicted_data_lstm": y_lstm.flatten().tolist(),
        "predicted_data_gru": y_gru.flatten().tolist(),
        "predicted_data_stacker": y_stack.flatten().tolist(),
        "r2_lstm": f"{r2_lstm:.4f}",
        "r2_gru": f"{r2_gru:.4f}",
        "r2_stacker": f"{r2_stack:.4f}",
    }
