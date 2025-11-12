import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ---- CARGAR MODELO ----
@st.cache_resource
def load_model():
    model = joblib.load("pipeline_NNM.joblib")
    return model

model = load_model()

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.write("Aplicación desarrollada para predecir el riesgo de preeclampsia usando un modelo de Machine Learning entrenado en Python.")

# ---- ENTRADA DE DATOS ----
st.subheader("Ingrese los datos del paciente:")

edad = st.number_input("Edad (años):", min_value=15, max_value=50, value=25)
imc = st.number_input("Índice de Masa Corporal (IMC):", min_value=10.0, max_value=50.0, value=22.5)
p_a_sistolica = st.number_input("Presión Arterial Sistólica:", min_value=80, max_value=200, value=110)
p_a_diastolica = st.number_input("Presión Arterial Diastólica:", min_value=50, max_value=130, value=70)
proteinuria = st.selectbox("Proteinuria:", ["Sí", "No"])
historia_familiar = st.selectbox("Historia Familiar de Preeclampsia:", ["Sí", "No"])

# ---- PREPARAR DATOS ----
df = pd.DataFrame({
    "edad": [edad],
    "imc": [imc],
    "p_a_sistolica": [p_a_sistolica],
    "p_a_diastolica": [p_a_diastolica],
    "proteinuria": [1 if proteinuria == "Sí" else 0],
    "historia_familiar": [1 if historia_familiar == "Sí" else 0]
})

# ---- PREDICCIÓN ----
if st.button("🔍 Predecir riesgo"):
    try:
        prediction = model.predict(df)
        proba = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None

        st.success(f"Predicción: {'Riesgo ALTO' if prediction[0]==1 else 'Riesgo BAJO'}")

        if proba is not None:
            st.info(f"Probabilidad estimada de riesgo: {proba*100:.2f}%")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")