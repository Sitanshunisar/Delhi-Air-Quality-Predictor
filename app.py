# app.py

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score 
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 0. Initialize App and Define Paths ---
app = Flask(__name__)
# Tell Flask not to cache static files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 

DATA_PATH = "data/delhiaqi.csv"
MODEL_DIR = "models"
LOOK_BACK = 5 

# ... (rest of your model paths are correct) ...
STACK_MODEL_PATH = os.path.join(MODEL_DIR, "stacking_model.pkl")
STACK_SCALER_PATH = os.path.join(MODEL_DIR, "stacking_scaler.pkl")
RNN_SCALER_X_PATH = os.path.join(MODEL_DIR, "rnn_scaler_X.pkl")
RNN_SCALER_Y_PATH = os.path.join(MODEL_DIR, "rnn_scaler_y.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
GRU_MODEL_PATH = os.path.join(MODEL_DIR, "gru_model.h5")


# --- 1. Load All Models and Scalers ---
try:
    print("Loading all models and scalers...")
    model_stacker = joblib.load(STACK_MODEL_PATH)
    scaler_stacker = joblib.load(STACK_SCALER_PATH)
    model_lstm = load_model(LSTM_MODEL_PATH, compile=False)
    model_gru = load_model(GRU_MODEL_PATH, compile=False)
    scaler_X_rnn = joblib.load(RNN_SCALER_X_PATH)
    scaler_y_rnn = joblib.load(RNN_SCALER_Y_PATH)
    print("All files loaded successfully.")
except Exception as e:
    print(f"\n---!!! ERROR !!!---")
    print(f"Error loading files: {e}")
    print("Please run 'notebooks/train_all_models.py' first.")
    print("-------------------\n")
    exit()

# --- 2. Helper Function (get_prediction_data) ---
# (This function is correct, no changes needed from the last version)
def create_windowed_dataset(X_data, y_data, look_back):
    X, y = [], []
    for i in range(len(X_data) - look_back):
        X.append(X_data[i:(i + look_back), :])
        y.append(y_data[i + look_back])
    return np.array(X), np.array(y)

def get_prediction_data():
    print("Preparing test data and running all 3 models (10-feature version)...")
    
    # 1. Load data & Engineer Features
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    print("Engineering cyclical HOUR features for test data...")
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df = df.drop(columns=['hour'])

    # 2. Define Features and Target
    target_col = 'pm2_5'
    stacker_features = [col for col in df.columns if col != target_col]
    rnn_features = df.columns.tolist() 

    # 3. Temporal Split
    split_point = int(len(df) * 0.8)
    test_df = df.iloc[split_point:]

    # 4. GET RNN (LSTM & GRU) PREDICTIONS
    test_scaled_X_rnn = scaler_X_rnn.transform(test_df[rnn_features])
    test_scaled_y_col_rnn = scaler_y_rnn.transform(test_df[[target_col]])
    X_test_rnn, y_test_rnn_scaled = create_windowed_dataset(
        test_scaled_X_rnn, test_scaled_y_col_rnn.flatten(), LOOK_BACK
    )
    y_pred_lstm_scaled = model_lstm.predict(X_test_rnn, verbose=0)
    y_pred_gru_scaled = model_gru.predict(X_test_rnn, verbose=0)
    y_test_actual = scaler_y_rnn.inverse_transform(y_test_rnn_scaled.reshape(-1, 1))
    y_pred_lstm_actual = scaler_y_rnn.inverse_transform(y_pred_lstm_scaled)
    y_pred_gru_actual = scaler_y_rnn.inverse_transform(y_pred_gru_scaled)
    
    # 5. GET STACKER PREDICTIONS
    X_test_stacker = test_df[stacker_features]
    X_test_stacker_scaled = scaler_stacker.transform(X_test_stacker)
    y_pred_stacker_actual = model_stacker.predict(X_test_stacker_scaled)
    y_pred_stacker_aligned = y_pred_stacker_actual[LOOK_BACK:]

    # 6. CALCULATE R² SCORES
    r2_lstm = r2_score(y_test_actual, y_pred_lstm_actual)
    r2_gru = r2_score(y_test_actual, y_pred_gru_actual)
    r2_stacker = r2_score(y_test_actual, y_pred_stacker_aligned)
    print(f"R² Scores: GRU={r2_gru:.4f}, LSTM={r2_lstm:.4f}, Stacker={r2_stacker:.4f}")

    # 7. Format for Chart.js
    labels = [f"Test Hour {i}" for i in range(len(y_test_actual))]
    return {
        'labels': labels,
        'actual_data': y_test_actual.flatten().tolist(),
        'predicted_data_lstm': y_pred_lstm_actual.flatten().tolist(),
        'predicted_data_gru': y_pred_gru_actual.flatten().tolist(),
        'predicted_data_stacker': y_pred_stacker_aligned.flatten().tolist(),
        'r2_lstm': f"{r2_lstm:.4f}",
        'r2_gru': f"{r2_gru:.4f}",
        'r2_stacker': f"{r2_stacker:.4f}"
    }

# --- 3. Define Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_chart_data')
def get_chart_data():
    data = get_prediction_data()
    return jsonify(data)

# --- *** NEW CACHE-FIX FUNCTION *** ---
@app.after_request
def add_header(response):
    """
    Add headers to prevent caching.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
# --- *** END OF NEW FUNCTION *** ---

if __name__ == "__main__":
    app.run(debug=True)