import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_modelos():
    modelo = joblib.load("modelo_diabetes.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    return modelo, preprocessor

modelo, preprocessor = load_modelos()


st.set_page_config(page_title="Predicción de Diabetes", layout="centered")
st.title("🩺 Predicción de Diabetes")
st.markdown(
    """
    Carga un archivo **CSV** con los datos de pacientes para obtener las predicciones.
    El modelo ha sido entrenado para detectar probabilidad de diabetes a partir de variables clínicas.
    """
)

uploaded_file = st.file_uploader("📂 Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_nuevo = pd.read_csv(uploaded_file)
        st.success(f"Archivo cargado correctamente con {df_nuevo.shape[0]} filas y {df_nuevo.shape[1]} columnas.")
        st.dataframe(df_nuevo.head())

        # Botón para predecir
        if st.button("🔮 Realizar Predicciones"):
            X_new = preprocessor.transform(df_nuevo)
            preds = modelo.predict(X_new)
            probs = modelo.predict_proba(X_new)[:, 1]

            df_result = df_nuevo.copy()
            df_result["Predicción"] = np.where(preds == 1, "Diabético", "No diabético")
            df_result["Probabilidad (%)"] = np.round(probs * 100, 2)

            st.subheader("📊 Resultados de las predicciones")
            st.dataframe(df_result)

            # Descargar resultados
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Descargar resultados",
                csv_out,
                "predicciones_diabetes.csv",
                "text/csv"
            )
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV con los datos a evaluar.")
