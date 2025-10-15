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