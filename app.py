import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Suhu Harian", layout="wide")

# === LOAD MODEL & SCALER ===
model = joblib.load("weather_temp_model.pkl")
temp_scaler = joblib.load("temp_scaler.pkl")

st.title("Prediksi Suhu Harian – Weather Dataset Kebumen")

st.write("""
Aplikasi ini menggunakan model **Random Forest Regression** untuk memprediksi suhu harian 
berdasarkan beberapa fitur cuaca. Nilai numerik yang dimasukkan adalah **nilai yang sudah 
diskalakan (0–1)** sesuai proses *MinMaxScaler* di notebook.
""")

# --- Sidebar input (sama seperti sebelumnya) ---
st.sidebar.header("Input Fitur (skala 0–1)")

tempmax = st.sidebar.number_input("tempmax (scaled 0–1)", 0.0, 1.0, 0.60, 0.01)
tempmin = st.sidebar.number_input("tempmin (scaled 0–1)", 0.0, 1.0, 0.30, 0.01)
humidity = st.sidebar.number_input("humidity (scaled 0–1)", 0.0, 1.0, 0.50, 0.01)
windspeed = st.sidebar.number_input("windspeed (scaled 0–1)", 0.0, 1.0, 0.40, 0.01)
sealevelpressure = st.sidebar.number_input("sealevelpressure (scaled 0–1)", 0.0, 1.0, 0.50, 0.01)

conditions = st.sidebar.selectbox(
    "conditions",
    ["Partially cloudy", "Rain, Overcast", "Rain, Partially cloudy"]
)

# one-hot untuk conditions sesuai data kamu
conditions_map = {
    "Partially cloudy": "Partially cloudy",
    "Rain, Overcast": "Rain, Overcast",
    "Rain, Partially cloudy": "Rain, Partially cloudy",
}

input_df = pd.DataFrame({
    "tempmax": [tempmax],
    "tempmin": [tempmin],
    "humidity": [humidity],
    "windspeed": [windspeed],
    "sealevelpressure": [sealevelpressure],
    "conditions": [conditions_map[conditions]],
})

st.subheader("Hasil Prediksi")

if st.button("Prediksi Suhu"):
    # 1) prediksi dalam skala 0–1
    y_scaled = model.predict(input_df)[0]

    # 2) balikkan ke derajat Celsius
    y_celsius = temp_scaler.inverse_transform(
        np.array(y_scaled).reshape(-1, 1)
    )[0, 0]

    st.write(f"Perkiraan suhu (skala MinMax 0–1): **{y_scaled:.4f}**")
    st.success(f"Perkiraan suhu dalam °C: **{y_celsius:.2f} °C**")
    st.caption("Model dilatih pada target yang telah di-*scale* 0–1, "
               "kemudian hasil dikonversi kembali ke derajat Celsius.")
