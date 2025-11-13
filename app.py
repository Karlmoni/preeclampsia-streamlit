import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ================================
st.set_page_config(
    page_title="Predicción de Preeclampsia",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.markdown("""
Esta aplicación utiliza un modelo de *Machine Learning* (NNM) para estimar el **riesgo de preeclampsia**
a partir de parámetros clínicos de la gestante.
""")

# ==========================================
# CARGA DE ARTEFACTOS
# ==========================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    art_dir = os.path.join(base_dir, "artefactos", "v1")

    # Rutas a los artefactos
    input_schema_path = os.path.join(art_dir, "input_schema.json")
    label_map_path = os.path.join(art_dir, "label_map.json")
    policy_path = os.path.join(art_dir, "decision_policy.json")
    pipeline_path = os.path.join(art_dir, "pipeline_NNM.joblib")

    # Cargar JSONs
    with open(input_schema_path, "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    # Modelo
    pipe = joblib.load(pipeline_path)

    # Mapa inverso
    rev_label = {v: k for k, v in label_map.items()}

    # Umbral
    thr = float(policy["threshold"])

    return pipe, input_schema, label_map, rev_label, thr, policy

PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, POLICY = load_artifacts()
FEATURES = list(INPUT_SCHEMA.keys())

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Alinea columnas y fuerza tipos según input_schema.json."""
    for col, tipo in INPUT_SCHEMA.items():
        if col not in df.columns:
            df[col] = np.nan
        if "int" in tipo or "float" in tipo:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string")
    return df[FEATURES]


def predict_batch(records, thr=None):
    """Devuelve probabilidad y clase textual."""
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]
    y_int = (proba >= thr).astype(int)

    resultados = []
    for p, y in zip(proba, y_int):
        resultados.append({
            "proba": float(p),
            "pred_int": int(y),
            "pred_label": REV_LABEL[int(y)],
            "threshold": thr
        })
    return resultados

# ==========================================
# INTERFAZ DE STREAMLIT
# ==========================================

st.subheader("1️⃣ Ingrese los datos de la paciente")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad (años)", min_value=10, max_value=60, value=30)
    imc = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
    p_a_sistolica = st.number_input("Presión Arterial Sistólica (mmHg)", min_value=80, max_value=200, value=120)
    p_a_diastolica = st.number_input("Presión Arterial Diastólica (mmHg)", min_value=40, max_value=130, value=80)

with col2:
    hipertension = st.selectbox("Hipertensión previa", ["SI", "NO"])
    diabetes = st.selectbox("Diabetes", ["SI", "NO"])
    creatinina = st.number_input("Creatinina (mg/dL)", min_value=0.1, max_value=10.0, value=1.0)
    ant_fam_hiper = st.selectbox("Antecedentes familiares de hipertensión", ["SI", "NO"])
    tec_repro_asistida = st.selectbox("Reproducción asistida", ["SI", "NO"])

# Diccionario EXACTO requerido por tu modelo
registro = {
    "edad": edad,
    "imc": imc,
    "p_a_sistolica": p_a_sistolica,
    "p_a_diastolica": p_a_diastolica,
    "hipertension": hipertension,
    "diabetes": diabetes,
    "creatinina": creatinina,
    "ant_fam_hiper": ant_fam_hiper,
    "tec_repro_asistida": tec_repro_asistida,
}

st.subheader("2️⃣ Resultado")

if st.button("Calcular riesgo"):
    resultado = predict_batch(registro)[0]
    proba_pct = resultado["proba"] * 100

    st.markdown("---")
    st.markdown(f"### Resultado: **{resultado['pred_label']}**")
    st.metric("Probabilidad de Riesgo", f"{proba_pct:.2f}%")
    st.caption(f"Umbral de decisión: {resultado['threshold']}")

    st.write("**Métricas del modelo en Test:**")
    st.json(POLICY["test_metrics"])

    st.info("⚠️ Este sistema es solo apoyo a la decisión clínica y no reemplaza el criterio médico.")
else:
    st.warning("Ingrese los datos y presione *Calcular riesgo*.")