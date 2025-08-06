import streamlit as st
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Crop Yield Predictor 🌾",
    page_icon="🌱",
    layout="centered",
)

# Logo or header image (optional - you can use a public image URL)
st.image("https://cdn-icons-png.flaticon.com/512/2909/2909767.png", width=100)
st.title("🌾 Crop Yield Predictor")
st.markdown("Predict the **most suitable crop** based on soil and weather conditions.")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "crop_recommender.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file not found. Please check the path: " + model_path)
    st.stop()

# --- Input Section ---
st.header("📥 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌿 Nitrogen (N)", 0, 140, 50)
    K = st.number_input("🌿 Potassium (K)", 5, 205, 50)
    humidity = st.number_input("💧 Humidity (%)", 10.0, 100.0, 50.0)
    ph = st.number_input("🌡️ Soil pH", 3.0, 10.0, 6.5)

with col2:
    P = st.number_input("🌿 Phosphorus (P)", 5, 145, 50)
    temperature = st.number_input("🌞 Temperature (°C)", 0.0, 50.0, 25.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 300.0, 100.0)

# --- Predict ---
if st.button("🚀 Predict Best Crop"):
    try:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]
        st.success(f"✅ **Recommended Crop**: `{prediction.upper()}`")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("Made by [Agents 404 🤖] | Sustainable Development Goals (Zero Hunger)")
