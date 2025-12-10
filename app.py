import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Suhu Harian", layout="wide")

# === LOAD MODEL PIPELINE (sudah termasuk scaler & encoder) ===
model = joblib.load("weather_temp_model.pkl")

st.title("Prediksi Suhu Harian – Weather Dataset Kebumen")

st.write("""
Aplikasi ini menggunakan model **Random Forest Regression** untuk memprediksi suhu harian
berdasarkan beberapa fitur cuaca. Nilai yang dimasukkan adalah **dalam satuan aslinya**.
""")

st.sidebar.header("Input Fitur (satuan asli)")

# Range slider sesuaikan dengan data: min 23.8 – max 30.1 (boleh dibulatkan)
tempmax = st.sidebar.slider("tempmax (°C)", 24.0, 32.0, 30.0, 0.1)
tempmin = st.sidebar.slider("tempmin (°C)", 22.0, 28.0, 25.0, 0.1)
humidity = st.sidebar.slider("humidity (%)", 40.0, 100.0, 80.0, 1.0)
windspeed = st.sidebar.slider("windspeed (km/h)", 0.0, 40.0, 10.0, 0.5)
sealevelpressure = st.sidebar.slider("sealevelpressure (hPa)", 1000.0, 1015.0, 1010.0, 0.1)

conditions = st.sidebar.selectbox(
    "conditions",
    ["Partially cloudy", "Rain, Overcast", "Rain, Partially cloudy"]
)

input_df = pd.DataFrame({
    "tempmax": [tempmax],
    "tempmin": [tempmin],
    "humidity": [humidity],
    "windspeed": [windspeed],
    "sealevelpressure": [sealevelpressure],
    "conditions": [conditions],
})

st.subheader("Hasil Prediksi")

if st.button("Prediksi Suhu"):
    # Prediksi langsung dalam °C
    y_pred_c = model.predict(input_df)[0]

    st.write(f"Perkiraan suhu harian: **{y_pred_c:.2f} °C**")
    st.caption("Model dilatih dengan target suhu dalam derajat Celsius.")
