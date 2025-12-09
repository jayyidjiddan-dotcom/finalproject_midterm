import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Load model (Pipeline Random Forest + Preprocess)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("weather_temp_model.pkl")
    return model

model = load_model()

# -------------------------------------------------
# UI Aplikasi
# -------------------------------------------------
st.title("Prediksi Suhu Harian – Weather Dataset Kebumen")

st.write(
    """
    Aplikasi ini menggunakan model **Random Forest Regression**  
    untuk memprediksi suhu harian berdasarkan beberapa fitur cuaca.
    Nilai numerik yang dimasukkan adalah **nilai yang sudah diskalakan (0–1)**  
    sesuai dengan proses *MinMaxScaler* di notebook.
    """
)

st.sidebar.header("Input Fitur Cuaca (Scaled 0–1)")

# Input fitur numerik (hasil scaling 0–1)
tempmax = st.sidebar.number_input("tempmax (scaled 0–1)", 0.0, 1.0, 0.6, step=0.01)
tempmin = st.sidebar.number_input("tempmin (scaled 0–1)", 0.0, 1.0, 0.3, step=0.01)
humidity = st.sidebar.number_input("humidity (scaled 0–1)", 0.0, 1.0, 0.5, step=0.01)
windspeed = st.sidebar.number_input("windspeed (scaled 0–1)", 0.0, 1.0, 0.4, step=0.01)
sealevelpressure = st.sidebar.number_input("sealevelpressure (scaled 0–1)", 0.0, 1.0, 0.5, step=0.01)

# Pilihan kondisi cuaca (disesuaikan dengan kategori di dataset)
conditions_options = [
    "Partially cloudy",
    "Rain, Overcast",
    "Rain, Partially cloudy",
]
conditions = st.sidebar.selectbox("conditions", conditions_options)

# -------------------------------------------------
# Prediksi
# -------------------------------------------------
if st.sidebar.button("Prediksi Suhu"):
    # Buat DataFrame 1 baris dengan nama kolom sama seperti saat training
    input_df = pd.DataFrame(
        {
            "tempmax": [tempmax],
            "tempmin": [tempmin],
            "humidity": [humidity],
            "windspeed": [windspeed],
            "sealevelpressure": [sealevelpressure],
            "conditions": [conditions],
        }
    )

    # Prediksi
    y_pred_scaled = model.predict(input_df)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"Perkiraan suhu (dalam **skala MinMax 0–1**): **{y_pred_scaled:.4f}**")

    st.info(
        "Catatan: Model dilatih menggunakan data yang sudah di-*scale* dengan MinMaxScaler, "
        "sehingga output masih dalam bentuk skala 0–1."
    )
