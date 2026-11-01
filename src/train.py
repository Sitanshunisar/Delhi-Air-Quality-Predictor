# src/train.py
"""Train all models and save artifacts to models/."""
import os
import warnings

import joblib
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.config import (
    GRU_MODEL_PATH,
    LOOK_BACK,
    LSTM_MODEL_PATH,
    MODEL_DIR,
    RNN_BATCH_SIZE,
    RNN_DROPOUT,
    RNN_EPOCHS,
    RNN_SCALER_X_PATH,
    RNN_SCALER_Y_PATH,
    RNN_UNITS,
    STACK_MODEL_PATH,
    STACK_SCALER_PATH,
    TARGET_COL,
    TRAIN_TEST_SPLIT,
    VALIDATION_SPLIT,
)
from src.data_loader import create_windowed_dataset, load_and_engineer, temporal_split

warnings.filterwarnings("ignore")


def train_stacking_regressor(train_df):
    """Train and save StackingRegressor."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    stacker_features = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[stacker_features]
    y_train = train_df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    estimators = [
        ("cat", CatBoostRegressor(verbose=0, allow_writing_files=False)),
        ("lgbm", LGBMRegressor(verbose=-1)),
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    model.fit(X_scaled, y_train)

    joblib.dump(model, STACK_MODEL_PATH)
    joblib.dump(scaler, STACK_SCALER_PATH)
    print("StackingRegressor saved.")
    return model, scaler


def train_rnn_models(train_df):
    """Train and save LSTM and GRU models."""
    rnn_features = train_df.columns.tolist()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(train_df[rnn_features])
    y_scaled = scaler_y.fit_transform(train_df[[TARGET_COL]])

    X_3d, y_rnn = create_windowed_dataset(X_scaled, y_scaled.flatten(), LOOK_BACK)
    n_timesteps, n_features = X_3d.shape[1], X_3d.shape[2]
    print(f"RNN input shape: {X_3d.shape}")

    joblib.dump(scaler_X, RNN_SCALER_X_PATH)
    joblib.dump(scaler_y, RNN_SCALER_Y_PATH)

    def _build(layer_cls):
        m = Sequential([
            layer_cls(RNN_UNITS, activation="relu", input_shape=(n_timesteps, n_features)),
            Dropout(RNN_DROPOUT),
            Dense(1),
        ])
        m.compile(optimizer="adam", loss="mean_squared_error")
        return m

    print("Training LSTM...")
    lstm = _build(LSTM)
    lstm.fit(X_3d, y_rnn, epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE,
             validation_split=VALIDATION_SPLIT, verbose=0)
    lstm.save(LSTM_MODEL_PATH)
    print("LSTM saved.")

    print("Training GRU...")
    gru = _build(GRU)
    gru.fit(X_3d, y_rnn, epochs=RNN_EPOCHS, batch_size=RNN_BATCH_SIZE,
             validation_split=VALIDATION_SPLIT, verbose=0)
    gru.save(GRU_MODEL_PATH)
    print("GRU saved.")

    return lstm, gru, scaler_X, scaler_y


def main():
    print("=== Delhi AQI — Training Pipeline ===")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_engineer()
    print(f"Data loaded: {df.shape}")

    train_df, _ = temporal_split(df, TRAIN_TEST_SPLIT)

    print("\n--- Training StackingRegressor ---")
    train_stacking_regressor(train_df)

    print("\n--- Training RNN Models ---")
    train_rnn_models(train_df)

    print("\n=== All training complete ===")


if __name__ == "__main__":
    main()
