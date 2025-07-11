import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Cambia esto si tu backend está en otro host/puerto

st.title("ClimaSeguro - Dashboard Básico")

# Última lectura
st.header("Última Lectura por Distrito")

distrito = st.text_input("Distrito (opcional)", value="Acobamba")

if st.button("Consultar última lectura"):
    try:
        params = {"distrito": distrito} if distrito else {}
        r = requests.get(f"{API_URL}/data/latest", params=params)
        if r.status_code == 200:
            st.json(r.json())
        else:
            st.error(f"Error: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Error de conexión: {e}")

# Comparación de modelos
st.header("Comparar un Modelo")

model_name = st.selectbox("Selecciona un modelo", [
    "Linear Regression", "Ridge", "Random Forest Regressor", "SVR", 
    "XGBoost", "LightGBM", "CatBoost", "ARIMA", "LSTM"
])

if st.button("Generar Comparación"):
    try:
        r = requests.get(f"{API_URL}/model/compare/{model_name}")
        if r.status_code == 200:
            result = r.json()
            st.write(f"Similitud: {result['similarity_percentage']}%")
            st.image(f"data:image/png;base64,{result['image_base64']}")
        else:
            st.error(f"Error: {r.status_code} - {r.text}")
    except Exception as e:
        st.error(f"Error de conexión: {e}")
