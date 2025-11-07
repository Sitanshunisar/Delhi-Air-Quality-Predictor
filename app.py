
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
=======
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
st.set_page_config(page_title="Delhi AQI Predictor", layout="wide", initial_sidebar_state="collapsed")

# --- Load Pre-trained Models & Objects ---
@st.cache_resource
def load_model_assets():
    """Loads all the necessary model files for prediction."""
    try:
        scaler = joblib.load('models/scaler.pkl')
        pca = joblib.load('models/pca.pkl')
        model = joblib.load('models/stacked_regressor_model.pkl')
        return scaler, pca, model
    except FileNotFoundError:
        return None, None, None

scaler, pca, model = load_model_assets()

# --- Helper Functions ---
def get_aqi_category(pm25):
    """Provides an interpretation of the PM2.5 value with a fun emoji!"""
    if 0 <= pm25 <= 50:
        return "Good ✅", "Air quality is considered satisfactory, and air pollution poses little or no risk.", "#2ECC71"
    elif 51 <= pm25 <= 100:
        return "Moderate 🤔", "Air quality is acceptable; however, some pollutants may be a moderate health concern for a very small number of people.", "#F1C40F"
    elif 101 <= pm25 <= 150:
        return "Unhealthy for Sensitive Groups 😷", "Members of sensitive groups may experience health effects.", "#E67E22"
    elif 151 <= pm25 <= 200:
        return "Unhealthy 🤢", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.", "#E74C3C"
    elif 201 <= pm25 <= 300:
        return "Very Unhealthy 🥵", "Health alert: everyone may experience more serious health effects.", "#9B59B6"
    else:
        return "Hazardous ☠️", "Health warnings of emergency conditions. The entire population is more likely to be affected.", "#78281F"

# --- Initialize Session State for Inputs ---
input_features = {
    "co": 1000.0, "no": 30.0, "no2": 50.0, "o3": 30.0,
    "so2": 40.0, "pm10": 200.0, "nh3": 20.0
}
for key, value in input_features.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- App Layout ---
# --- NEW: Centered Header using st.markdown ---
st.markdown("<h1 style='text-align: center;'>🌬️ Delhi Air Quality Predictor 🌬️</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>Let's predict the <strong>PM2.5 concentration</strong>! You can use the default values or expand the section below to input your own.</p>", unsafe_allow_html=True)
st.markdown("---")

# Using columns to create a centered layout
left_margin, content_area, right_margin = st.columns([1, 2, 1])

with content_area:
    if model is None:
        st.error("🚨 Critical Error: Model files could not be loaded. Please check the `models` directory and restart the app.")
    else:
        # --- Expander with 2-Column Layout for Inputs ---
        with st.expander("👇 Adjust Pollutant Levels (Optional)"):
            col1, col2 = st.columns(2)

            with col1:
                st.session_state.co = st.number_input("Carbon Monoxide (CO)", 0.0, 12000.0, st.session_state.co, 10.0)
                st.session_state.co = st.slider("CO Slider", 0.0, 12000.0, st.session_state.co, 10.0, label_visibility="collapsed")
                
                st.session_state.no = st.number_input("Nitric Oxide (NO)", 0.0, 400.0, st.session_state.no, 1.0)
                st.session_state.no = st.slider("NO Slider", 0.0, 400.0, st.session_state.no, 1.0, label_visibility="collapsed")
                
                st.session_state.no2 = st.number_input("Nitrogen Dioxide (NO2)", 0.0, 250.0, st.session_state.no2, 1.0)
                st.session_state.no2 = st.slider("NO2 Slider", 0.0, 250.0, st.session_state.no2, 1.0, label_visibility="collapsed")

                st.session_state.o3 = st.number_input("Ozone (O3)", 0.0, 170.0, st.session_state.o3, 1.0)
                st.session_state.o3 = st.slider("O3 Slider", 0.0, 170.0, st.session_state.o3, 1.0, label_visibility="collapsed")
            
            with col2:
                st.session_state.so2 = st.number_input("Sulphur Dioxide (SO2)", 0.0, 500.0, st.session_state.so2, 1.0)
                st.session_state.so2 = st.slider("SO2 Slider", 0.0, 500.0, st.session_state.so2, 1.0, label_visibility="collapsed")

                st.session_state.pm10 = st.number_input("Particulate Matter (PM10)", 0.0, 1500.0, st.session_state.pm10, 10.0)
                st.session_state.pm10 = st.slider("PM10 Slider", 0.0, 1500.0, st.session_state.pm10, 10.0, label_visibility="collapsed")

                st.session_state.nh3 = st.number_input("Ammonia (NH3)", 0.0, 300.0, st.session_state.nh3, 1.0)
                st.session_state.nh3 = st.slider("NH3 Slider", 0.0, 300.0, st.session_state.nh3, 1.0, label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Prediction Button and Display Logic ---
        if st.button("✨ Predict PM2.5 Level!", type="primary", use_container_width=True):
            input_data = pd.DataFrame({key: [st.session_state[key]] for key in input_features})

            # Feature Engineering
            input_data['nox'] = input_data['no'] + input_data['no2']
            input_data['co_no_interaction'] = input_data['co'] * input_data['no']
            input_data['so2_o3_interaction'] = input_data['so2'] * input_data['o3']
            input_data['co_squared'] = input_data['co'] ** 2
            input_data['high_co'] = np.where(input_data['co'] > 2000, 1, 0)
            
            # Reorder columns to match model training
            final_feature_order = [
                'co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3', 
                'nox', 'co_no_interaction', 'so2_o3_interaction', 'co_squared', 'high_co'
            ]
            input_df_reordered = input_data[final_feature_order]

            # Scaling, PCA, and Prediction
            scaled_data = scaler.transform(input_df_reordered)
            pca_data = pca.transform(scaled_data)
            prediction = model.predict(pca_data)
            predicted_value = prediction[0]

            category, health_implications, color = get_aqi_category(predicted_value)

            # Display the Result
            st.markdown("---")
            st.subheader("🎉 Prediction Result 🎉")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Predicted PM2.5 Concentration", value=f"{predicted_value:.2f} µg/m³")
            with res_col2:
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
                    <h3 style="color: white;">{category}</h3>
                </div>
                """, unsafe_allow_html=True)

            st.info(f"**Health Implications:** {health_implications}")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)

