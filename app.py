# app.py
# =============================================================================
# App Streamlit para predicción de Diabetes (CSV)
# Estructura esperada del repo:
#   / (raíz)
#   ├─ app.py
#   ├─ modelo_diabetes.joblib
#   ├─ requirements.txt
#   └─ Artefactos/
#      └─ preprocessor.joblib
# =============================================================================

import streamlit as st
st.set_page_config(page_title="Predicción de Diabetes", layout="centered")
# ^ Debe ser el PRIMER comando de Streamlit

# --- Imports estándar (sin usar st.* antes del set_page_config) ---
import os
import sys
import platform
import traceback
import numpy as np
import pandas as pd
import joblib

# --- Rutas robustas (relativas a este archivo, no al cwd) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_diabetes.joblib")                 # en raíz
PREPROC_PATH = os.path.join(BASE_DIR, "Artefactos", "preprocessor.joblib")    # en /Artefactos

# ===================== Panel de diagnóstico =====================
with st.expander("🔎 Diagnóstico del entorno (clic para abrir)", expanded=True):
    st.write({
        "Python": sys.version.split()[0],
        "SO": platform.platform(),
    })
    try:
        import sklearn
        st.write({
            "streamlit": st.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        })
    except Exception as e:
        st.write("No se pudieron leer versiones:", str(e))

    def ls_safe(path: str):
        try:
            return sorted(os.listdir(path))
        except Exception as e:
            return f"Error listando {path}: {e}"

    st.write("Directorio de la app (__file__):", BASE_DIR)
    st.write("Contenido de la raíz del repo:", ls_safe(BASE_DIR))
    st.write("Contenido de ./Artefactos:", ls_safe(os.path.join(BASE_DIR, "Artefactos")))

# ===================== Carga de artefactos =====================
@st.cache_resource(show_spinner=True)
def load_modelos(model_path: str, preproc_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"No se encontró el preprocesador en: {preproc_path}")

    try:
        modelo = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo ({model_path}): {e}")

    try:
        preprocessor = joblib.load(preproc_path)
    except Exception as e:
        raise RuntimeError(f"Error cargando preprocesador ({preproc_path}): {e}")

    return modelo, preprocessor

# Intento de carga con manejo de errores visible en UI
try:
    modelo, preprocessor = load_modelos(MODEL_PATH, PREPROC_PATH)
except Exception as e:
    st.error("❌ No se pudo cargar el modelo o el preprocesador.")
    st.caption("Revisa rutas, versiones de numpy/scikit-learn y que los artefactos estén en el repo.")
    st.exception(e)  # muestra traceback completo
    st.stop()

# ===================== UI Principal =====================
st.title("🩺 Predicción de Diabetes (CSV)")
st.markdown(
    "Sube un **CSV** con las columnas que espera el preprocesador. "
    "Si no estás seguro del esquema, descarga la **plantilla**."
)

# Intentar leer columnas esperadas del preprocesador
expected_cols = None
try:
    if hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
except Exception:
    expected_cols = None

# Botón para descargar plantilla (si el preproc las expone)
if expected_cols:
    plantilla_df = pd.DataFrame(columns=expected_cols)
    st.download_button(
        "📥 Descargar plantilla CSV (columnas esperadas)",
        data=plantilla_df.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_prediccion_diabetes.csv",
        mime="text/csv",
        help="Usa esta plantilla para asegurar que tus columnas coinciden con el preprocesador."
    )
else:
    st.warning(
        "No fue posible leer `feature_names_in_` del preprocesador. "
        "Si luego falla la transformación, revisa compatibilidad de versiones y re-genera los artefactos."
    )

# Función para alinear columnas al esquema esperado
def align_columns(df_in: pd.DataFrame, expected: list[str]):
    cols_in = df_in.columns.tolist()
    missing = [c for c in expected if c not in cols_in]
    extra   = [c for c in cols_in if c not in expected]

    # Para columnas faltantes, creamos con 0 (suele ser neutro para dummies y numéricas escaladas)
    for c in missing:
        df_in[c] = 0

    # Eliminamos columnas no esperadas
    if extra:
        df_in = df_in.drop(columns=extra, errors="ignore")

    # Reordenamos exactamente como el preprocesador espera
    return df_in[expected], missing, extra

# ===================== Cargador de CSV y Predicción =====================
uploaded_file = st.file_uploader("📂 Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Lectura con fallback de encoding
        try:
            df_nuevo = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df_nuevo = pd.read_csv(uploaded_file, encoding="latin-1")

        st.success(f"Archivo cargado: {df_nuevo.shape[0]} filas × {df_nuevo.shape[1]} columnas")
        st.dataframe(df_nuevo.head(), use_container_width=True)

        # Alinear al esquema esperado (si está disponible)
        if expected_cols:
            df_aligned, missing, extra = align_columns(df_nuevo.copy(), expected_cols)
            if missing:
                st.warning(f"Se agregaron {len(missing)} columnas faltantes con 0: {missing}")
            if extra:
                st.info(f"Se ignoraron {len(extra)} columnas no usadas: {extra}")
        else:
            df_aligned = df_nuevo

        # Botón de predicción
        if st.button("🔮 Realizar predicciones"):
            with st.spinner("Transformando y prediciendo..."):
                try:
                    X_new = preprocessor.transform(df_aligned)
                except Exception as e:
                    st.error(
                        "Fallo al transformar con el preprocesador. "
                        "Suele deberse a incompatibilidad de versiones o nombres/formato de columnas."
                    )
                    st.exception(e)
                    st.stop()

                # Predicción
                try:
                    preds = modelo.predict(X_new)
                except Exception as e:
                    st.error("Fallo al ejecutar predict() del modelo.")
                    st.exception(e)
                    st.stop()

                # Probabilidades (si el modelo las soporta)
                if hasattr(modelo, "predict_proba"):
                    try:
                        probs = modelo.predict_proba(X_new)[:, 1]
                    except Exception:
                        probs = np.full(len(preds), np.nan)
                else:
                    probs = np.full(len(preds), np.nan)

                df_result = df_nuevo.copy()
                # Mapear a etiquetas comprensibles
                df_result["Predicción"] = np.where(preds == 1, "Diabético", "No diabético")
                if not np.isnan(probs).all():
                    df_result["Probabilidad (%)"] = np.round(probs * 100, 2)

                st.subheader("📊 Resultados")
                st.dataframe(df_result, use_container_width=True)

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Descargar resultados (CSV)",
                    csv_out,
                    "predicciones_diabetes.csv",
                    "text/csv"
                )

    except Exception as e:
        st.error("Error general procesando el CSV.")
        st.exception(e)
else:
    st.info("Sube un CSV con los datos a evaluar.")

# ===================== Consejos de compatibilidad (opcional) =====================
with st.expander("🧩 Ayuda: compatibilidad de versiones (opcional)"):
    st.markdown(
        "- Si ves errores tipo **BitGenerator** al cargar artefactos, alinea las versiones de **numpy** y "
        "**scikit-learn** entre el entrenamiento y el despliegue, o re-exporta los artefactos con el stack del deployment.\n"
        "- En Streamlit Cloud, mantén en la **raíz**:\n"
        "  - `runtime.txt` → `3.12`\n"
        "  - `requirements.txt` mínimo (por ejemplo):\n"
        "    ```
        streamlit==1.39.0
        pandas==2.2.3
        numpy==1.26.4
        scikit-learn==1.5.2
        joblib==1.4.2
        ```\n"
        "- Verifica arriba en el diagnóstico que los archivos y versiones coincidan."
    )
