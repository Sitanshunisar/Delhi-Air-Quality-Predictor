# notebooks/train_all_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import os
import warnings

warnings.filterwarnings('ignore')

print("--- Starting 3-Model Training (10-Feature 'Hour-Aware' dataset) ---")

# --- 0. Define File Paths ---
DATA_PATH = "data/delhiaqi.csv"
MODEL_DIR = "models"
LOOK_BACK = 5 

# Stacker Paths
STACK_MODEL_PATH = os.path.join(MODEL_DIR, "stacking_model.pkl")
STACK_SCALER_PATH = os.path.join(MODEL_DIR, "stacking_scaler.pkl")

# RNN Paths (Shared)
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
GRU_MODEL_PATH = os.path.join(MODEL_DIR, "gru_model.h5")
RNN_SCALER_X_PATH = os.path.join(MODEL_DIR, "rnn_scaler_X.pkl")
RNN_SCALER_Y_PATH = os.path.join(MODEL_DIR, "rnn_scaler_y.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Load Data & Engineer Time Features ---
try:
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    print("Data loaded.")
except Exception as e:
    print(f"Error loading {DATA_PATH}: {e}")
    exit()

# --- *** MODIFIED: STEP 2 (Add *ONLY* Hour Features) *** ---
print("Engineering cyclical HOUR features...")
df['hour'] = df.index.hour

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# We are *NOT* adding the day_of_week features, as they were noise.
df = df.drop(columns=['hour'])
# --- *** END OF MODIFIED FEATURES *** ---


# --- 2. Define Features and Target ---
target_col = 'pm2_5'
# stacker_features will now have 9 features (8 original + 2 hour - 1 target)
stacker_features = [col for col in df.columns if col != target_col]
# rnn_features will now have 10 features (8 original + 2 hour)
rnn_features = df.columns.tolist() 

print(f"Total features for RNNs: {len(rnn_features)}")
print(f"Total features for Stacker: {len(stacker_features)}")

# --- 3. Train Test Split (Temporal) ---
split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

# ===============================================
# 4. TRAIN STACKING REGRESSOR
# ===============================================
print("\n--- Training StackingRegressor (10-features) ---")
X_train_stacker = train_df[stacker_features]
y_train_stacker = train_df[target_col]
scaler_stacker = StandardScaler()
X_train_stacker_scaled = scaler_stacker.fit_transform(X_train_stacker)

estimators = [
    ('cat', CatBoostRegressor(verbose=0, allow_writing_files=False)),
    ('lgbm', LGBMRegressor(verbose=-1))
]
stacked_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stacked_model.fit(X_train_stacker_scaled, y_train_stacker)

joblib.dump(stacked_model, STACK_MODEL_PATH)
joblib.dump(scaler_stacker, STACK_SCALER_PATH)
print("StackingRegressor saved.")

# ===============================================
# 5. PREPARE DATA FOR RNNs (LSTM & GRU)
# ===============================================
print("\n--- Preparing data for RNN models (10-features) ---")
scaler_X_rnn = MinMaxScaler()
scaler_y_rnn = MinMaxScaler()

train_scaled_X = scaler_X_rnn.fit_transform(train_df[rnn_features])
train_scaled_y = scaler_y_rnn.fit_transform(train_df[[target_col]])

def create_windowed_dataset(X_data, y_data, look_back):
    X, y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i:(i + look_back), :])
        y.append(y_data[i + look_back])
    return np.array(X), np.array(y)

X_train_3D, y_train_rnn = create_windowed_dataset(
    train_scaled_X, train_scaled_y.flatten(), LOOK_BACK
)

n_timesteps = X_train_3D.shape[1]
n_features = X_train_3D.shape[2] # This should now be 10
print(f"RNN Input Shape (samples, timesteps, features): {X_train_3D.shape}")

# Save the RNN scalers
joblib.dump(scaler_X_rnn, RNN_SCALER_X_PATH)
joblib.dump(scaler_y_rnn, RNN_SCALER_Y_PATH)
print("RNN scalers saved.")

# ===============================================
# 6. TRAIN LSTM MODEL
# ===============================================
print("\n--- Training LSTM ---")
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(n_timesteps, n_features)),
    Dropout(0.2),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train_3D, y_train_rnn, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
model_lstm.save(LSTM_MODEL_PATH)
print("LSTM model saved.")

# ===============================================
# 7. TRAIN GRU MODEL
# ===============================================
print("\n--- Training GRU ---")
model_gru = Sequential([
    GRU(50, activation='relu', input_shape=(n_timesteps, n_features)),
    Dropout(0.2),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mean_squared_error')
model_gru.fit(X_train_3D, y_train_rnn, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
model_gru.save(GRU_MODEL_PATH)
print("GRU model saved.")

print("\n--- All Training Complete ---")