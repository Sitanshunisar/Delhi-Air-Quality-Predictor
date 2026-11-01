# app/streamlit_app.py
"""Streamlit Delhi AQI Predictor & 3-Model Comparison Dashboard."""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PCA_PATH, SCALER_PATH, STACKED_REGRESSOR_PATH
from src.predict import get_prediction_data, load_all_models

warnings.filterwarnings("ignore")

# --- Prometheus Metrics Setup ---
@st.cache_resource
def get_prometheus_metrics():
    """Create or retrieve cached Prometheus metrics and start HTTP server."""
    try:
        from prometheus_client import REGISTRY, Counter, Gauge, start_http_server

        # Start exporter if not already active
        try:
            start_http_server(8000, addr="0.0.0.0")
        except OSError:
            pass

        # Avoid DuplicateTimeseries on rerun
        if "delhi_aqi_predictions_total" in REGISTRY._names_to_collectors:
            counter = REGISTRY._names_to_collectors["delhi_aqi_predictions_total"]
        else:
            counter = Counter(
                "delhi_aqi_predictions_total", "Total AQI predictions requested"
            )

        if "delhi_aqi_last_predicted_pm25" in REGISTRY._names_to_collectors:
            gauge = REGISTRY._names_to_collectors["delhi_aqi_last_predicted_pm25"]
        else:
            gauge = Gauge(
                "delhi_aqi_last_predicted_pm25", "Latest predicted PM2.5 concentration"
            )

        return counter, gauge
    except Exception as exc:
        print(f"Prometheus metrics init note: {exc}")
        return None, None


PRED_COUNTER, LAST_PM25_GAUGE = get_prometheus_metrics()

st.set_page_config(
    page_title="Delhi AQI Intelligence & Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_all_assets():
    """Load both static inference assets and recurrent forecasting models."""
    try:
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        model = joblib.load(STACKED_REGRESSOR_PATH)
    except FileNotFoundError:
        scaler, pca, model = None, None, None

    try:
        time_series_models = load_all_models()
    except Exception:
        time_series_models = None

    return scaler, pca, model, time_series_models


scaler, pca, model, ts_models = load_all_assets()


def get_aqi_category(pm25: float):
    """Categorize AQI based on PM2.5 concentration (µg/m³)."""
    if pm25 <= 50:
        return "Good ✅", "Air quality is satisfactory; poses little or no health risk.", "#2ECC71"
    elif pm25 <= 100:
        return "Moderate 🤔", "Acceptable air quality; slight risk for sensitive individuals.", "#F1C40F"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive Groups 😷", "Sensitive groups (asthma, elderly) may experience irritation.", "#E67E22"
    elif pm25 <= 200:
        return "Unhealthy 🤢", "General public may begin to experience health effects.", "#E74C3C"
    elif pm25 <= 300:
        return "Very Unhealthy 🥵", "Health alert: serious effects for entire population.", "#9B59B6"
    else:
        return "Hazardous ☠️", "Emergency conditions. Entire population is likely affected.", "#78281F"


# --- Header ---
st.markdown("<h1 style='text-align:center;'>🌬️ Delhi AQI Predictor & Model Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>Comparative performance of 3 ML/DL models & Real-time PM2.5 Prediction</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab1, tab2 = st.tabs(["📊 3-Model Comparison Dashboard", "🔮 Live PM2.5 Predictor"])

# =========================================================================
# TAB 1: 3-Model Comparison Dashboard
# =========================================================================
with tab1:
    st.subheader("📈 Model Performance on Test Data (Temporal Split)")
    st.markdown(
        "Compare the **inference power** of the Stacking Regressor against the **forecasting sequence power** "
        "of GRU and LSTM deep learning recurrent neural networks."
    )

    if ts_models is None:
        st.warning("⚠️ Time-series model files not found. Run `python -m src.train` to generate all models.")
    else:
        with st.spinner("Generating comparative predictions across test hours..."):
            pred_data = get_prediction_data(ts_models)

        # Metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="🥇 Stacking Regressor (CatBoost + LightGBM + Ridge)",
                value=f"R² = {pred_data['r2_stacker']}",
                help="Predicts concurrent 10-feature values using static ensemble inference.",
            )
        with col2:
            st.metric(
                label="🥈 GRU (Gated Recurrent Unit)",
                value=f"R² = {pred_data['r2_gru']}",
                help="Forecasting model: predicts using only the past 5 hours of data.",
            )
        with col3:
            st.metric(
                label="🥉 LSTM (Long Short-Term Memory)",
                value=f"R² = {pred_data['r2_lstm']}",
                help="Sequence model: predicts using only the past 5 hours of historical lag data.",
            )

        st.markdown("---")

        # Range filter for chart clarity
        total_points = len(pred_data["labels"])
        view_range = st.slider(
            "Select range of test hours to display:",
            min_value=0,
            max_value=total_points,
            value=(0, min(150, total_points)),
            step=10,
        )

        start_idx, end_idx = view_range
        x_vals = pred_data["labels"][start_idx:end_idx]

        # Interactive Plotly Chart
        fig = go.Figure()

        # Actual data
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_data["actual_data"][start_idx:end_idx],
                mode="lines+markers",
                name="Actual PM2.5",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=4),
            )
        )

        # Stacking Regressor
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_data["predicted_data_stacker"][start_idx:end_idx],
                mode="lines",
                name=f"Stacking Regressor (R²: {pred_data['r2_stacker']})",
                line=dict(color="#2ca02c", width=2, dash="solid"),
            )
        )

        # GRU
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_data["predicted_data_gru"][start_idx:end_idx],
                mode="lines",
                name=f"GRU (R²: {pred_data['r2_gru']})",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
            )
        )

        # LSTM
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=pred_data["predicted_data_lstm"][start_idx:end_idx],
                mode="lines",
                name=f"LSTM (R²: {pred_data['r2_lstm']})",
                line=dict(color="#d62728", width=2, dash="dot"),
            )
        )

        fig.update_layout(
            title="Actual vs Predicted PM2.5 Concentrations (µg/m³)",
            xaxis_title="Timeline (Test Hours)",
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=520,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ How do these three models work?"):
            st.markdown(
                """
                - **Stacking Regressor ($R^2 \approx 0.96$):**
                  Ensemble consisting of CatBoost Regressor, LightGBM, and Ridge Regression meta-model.
                  It predicts the current PM2.5 using all other concurrent atmospheric parameters (CO, NO, NO2, O3, SO2, PM10, NH3, and cyclical hour embeddings).
                - **GRU ($R^2 \approx 0.87 - 0.89$):**
                  Recurrent deep neural network using Gated Recurrent Units with a 5-hour temporal look-back window.
                  Forecasts PM2.5 strictly from historical trajectory.
                - **LSTM ($R^2 \approx 0.80 - 0.84$):**
                  Long Short-Term Memory network utilizing memory cells. On this dataset, GRU's simpler gating architecture slightly outperforms LSTM.
                """
            )

# =========================================================================
# TAB 2: Live PM2.5 Predictor (User Input)
# =========================================================================
with tab2:
    st.subheader("🎯 Real-Time PM2.5 Prediction from Pollutant Inputs")
    st.markdown(
        "**Model in use:** `StackingRegressor` (CatBoost + LightGBM + Ridge with PCA dimensionality reduction) "
        "trained on Delhi atmospheric pollutants."
    )

    INPUT_DEFAULTS = {
        "co": 1000.0,
        "no": 30.0,
        "no2": 50.0,
        "o3": 30.0,
        "so2": 40.0,
        "pm10": 200.0,
        "nh3": 20.0,
    }

    for k, v in INPUT_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col_l, col_m, col_r = st.columns([1, 4, 1])
    with col_m:
        if model is None or scaler is None or pca is None:
            st.error("🚨 Static model files (`stacked_regressor_model.pkl`, `scaler.pkl`, `pca.pkl`) not found in `models/`.")
        else:
            with st.expander("👇 Adjust Atmospheric Pollutant Concentrations", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.co = st.slider("Carbon Monoxide (CO) µg/m³", 0.0, 12000.0, st.session_state.co, 10.0)
                    st.session_state.no = st.slider("Nitric Oxide (NO) µg/m³", 0.0, 400.0, st.session_state.no, 1.0)
                    st.session_state.no2 = st.slider("Nitrogen Dioxide (NO2) µg/m³", 0.0, 250.0, st.session_state.no2, 1.0)
                    st.session_state.o3 = st.slider("Ozone (O3) µg/m³", 0.0, 170.0, st.session_state.o3, 1.0)
                with c2:
                    st.session_state.so2 = st.slider("Sulphur Dioxide (SO2) µg/m³", 0.0, 500.0, st.session_state.so2, 1.0)
                    st.session_state.pm10 = st.slider("Particulate Matter 10 (PM10) µg/m³", 0.0, 1500.0, st.session_state.pm10, 10.0)
                    st.session_state.nh3 = st.slider("Ammonia (NH3) µg/m³", 0.0, 300.0, st.session_state.nh3, 1.0)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("✨ Predict PM2.5 Level Now", type="primary", use_container_width=True):
                # 1. Feature Engineering (matches PCA + Stacking pipeline)
                row = {k: [st.session_state[k]] for k in INPUT_DEFAULTS}
                input_df = pd.DataFrame(row)
                input_df["nox"] = input_df["no"] + input_df["no2"]
                input_df["co_no_interaction"] = input_df["co"] * input_df["no"]
                input_df["so2_o3_interaction"] = input_df["so2"] * input_df["o3"]
                input_df["co_squared"] = input_df["co"] ** 2
                input_df["high_co"] = np.where(input_df["co"] > 2000, 1, 0)

                feature_order = [
                    "co", "no", "no2", "o3", "so2", "pm10", "nh3",
                    "nox", "co_no_interaction", "so2_o3_interaction",
                    "co_squared", "high_co",
                ]
                ordered_data = input_df[feature_order]

                # 2. Transform & Predict
                scaled = scaler.transform(ordered_data)
                pca_features = pca.transform(scaled)
                pred_val = float(model.predict(pca_features)[0])
                category, health_msg, color_code = get_aqi_category(pred_val)

                # Record Prometheus metrics
                if PRED_COUNTER is not None:
                    PRED_COUNTER.inc()
                if LAST_PM25_GAUGE is not None:
                    LAST_PM25_GAUGE.set(pred_val)

                # 3. Display Result
                st.markdown("---")
                st.subheader("🎉 Prediction Result")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Predicted PM2.5 Concentration", value=f"{pred_val:.2f} µg/m³")
                with res_col2:
                    st.markdown(
                        f"""
                        <div style="padding: 15px; border-radius: 10px; background-color: {color_code}; color: white; text-align: center;">
                            <h3 style="margin: 0; color: white;">{category}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.info(f"**Health Implications:** {health_msg}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Delhi AQI Predictor • Powered by Streamlit & Scikit-learn</div>", unsafe_allow_html=True)
