import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('house_price_model.pkl')

st.title("Prediksi Harga Rumah 🏠")

# Input pengguna
grlivarea = st.number_input('Luas bangunan di atas tanah (GrLivArea)', value=1500)
overallqual = st.slider('Kualitas bangunan (OverallQual)', 1, 10, 5)
yearbuilt = st.number_input('Tahun dibangun (YearBuilt)', value=2000)
totalbsmtsf = st.number_input('Luas basement (TotalBsmtSF)', value=1000)
garagecars = st.slider('Jumlah mobil muat di garasi (GarageCars)', 0, 4, 2)

# Prediksi
if st.button('Prediksi Harga'):
    features = np.array([[grlivarea, overallqual, yearbuilt, totalbsmtsf, garagecars]])
    prediction = model.predict(features)[0]
    st.success(f"Harga rumah diprediksi: ${prediction:,.2f}")
