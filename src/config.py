# src/config.py
"""Central configuration for paths and hyperparameters."""
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "delhiaqi.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model file paths
STACK_MODEL_PATH = os.path.join(MODEL_DIR, "stacking_model.pkl")
STACK_SCALER_PATH = os.path.join(MODEL_DIR, "stacking_scaler.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
GRU_MODEL_PATH = os.path.join(MODEL_DIR, "gru_model.h5")
RNN_SCALER_X_PATH = os.path.join(MODEL_DIR, "rnn_scaler_X.pkl")
RNN_SCALER_Y_PATH = os.path.join(MODEL_DIR, "rnn_scaler_y.pkl")

# Streamlit model paths (PCA-based static model)
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
STACKED_REGRESSOR_PATH = os.path.join(MODEL_DIR, "stacked_regressor_model.pkl")

# Hyperparameters
LOOK_BACK = 5
TRAIN_TEST_SPLIT = 0.8
RNN_EPOCHS = 50
RNN_BATCH_SIZE = 16
RNN_UNITS = 50
RNN_DROPOUT = 0.2
PCA_COMPONENTS = 10
VALIDATION_SPLIT = 0.1

# Target column
TARGET_COL = "pm2_5"
