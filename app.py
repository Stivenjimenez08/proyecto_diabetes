# app.py
# =============================================================================
# App Streamlit para predicción de Diabetes (CSV)
# Preferencia: usar pipeline único (pipeline_diabetes.joblib).
# Fallback: modelo + preprocesador por separado (verificación de compatibilidad).
#
# Estructura recomendada del repo:
#   / (raíz)
#   ├─ app.py
#   ├─ pipeline_diabetes.joblib     <-- preferido (si existe)
#   ├─ modelo_diabetes.joblib       <-- opcional (fallback)
#   ├─ preprocessor.joblib          <-- opcional (fallback)
#   ├─ requirements.txt
#   └─ runtime.txt
# =============================================================================

import streamlit as st
st.set_page_config(page_title="Predicción de Diabetes", layout="centered")
# ^ Debe ser el PRIMER comando de Streamlit

# --- Imports estándar ---
import os
import sys
import platform
import numpy as np
import pandas as pd
import joblib

# --- Rutas robustas (relativas a este archivo) ---
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPE_PATH  = os.path.join(BASE_DIR, "pipeline_diabetes.joblib")  # preferido
MODEL_PATH = os.path.join(BASE_DIR, "modelo_diabetes.joblib")    # fallback
PREPROC_PATH = os.path.join(BASE_DIR, "preprocessor.joblib")     # fallback

# ===================== Panel de diagnóstico =====================
with st.expander("🔎 Diagnóstico del entorno (clic para abrir)", expanded=True):
    st.write({"Python": sys.version.split()[0], "SO": platform.platform()})
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

# ===================== Helpers =====================
def _sanitize_onehot_for_numpy2(preprocessor):
    """
    Parche: en NumPy 2.x, OneHotEncoder usa np.isnan internamente y falla con enteros/bools.
    Convertimos categories_ numéricas a float para evitar el TypeError.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline

    if not hasattr(preprocessor, "transformers_"):
        return

    def _patch(enc: OneHotEncoder):
        if not hasattr(enc, "categories_"):
            return
        fixed = []
        for cats in enc.categories_:
            try:
                if np.issubdtype(cats.dtype, np.number) or np.issubdtype(cats.dtype, np.bool_):
                    fixed.append(cats.astype(float))
                else:
                    fixed.append(cats)
            except Exception:
                fixed.append(cats)
        enc.categories_ = fixed

    for _, trans, _ in preprocessor.transformers_:
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(trans, Pipeline):
                for _, step in trans.steps:
                    try:
                        from sklearn.preprocessing import OneHotEncoder
                        if isinstance(step, OneHotEncoder):
                            _patch(step)
                    except Exception:
                        pass
            else:
                from sklearn.preprocessing import OneHotEncoder
                if isinstance(trans, OneHotEncoder):
                    _patch(trans)
        except Exception:
            pass

def _get_expected_cols_from_pre(pre):
    try:
        if hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    except Exception:
        pass
    return None

def align_columns(df_in: pd.DataFrame, expected: list[str]):
    cols_in = df_in.columns.tolist()
    missing = [c for c in expected if c not in cols_in]
    extra   = [c for c in cols_in if c not in expected]
    # columnas faltantes con 0 (neutral para numéricas y dummies)
    for c in missing:
        df_in[c] = 0
    # eliminar columnas no usadas
    if extra:
        df_in = df_in.drop(columns=extra, errors="ignore")
    # reordenar
    return df_in[expected], missing, extra

def pre_n_features_out(pre) -> int | None:
    """Intento de estimar cuántas columnas produce el preprocesador (para validar contra el modelo)."""
    try:
        if hasattr(pre, "get_feature_names_out"):
            return len(pre.get_feature_names_out())
    except Exception:
        pass
    # heurística por transformadores
    try:
        if hasattr(pre, "transformers_"):
            total = 0
            for _, trans, cols in pre.transformers_:
                if trans is None:
                    continue
                try:
                    from sklearn.pipeline import Pipeline
                    if isinstance(trans, Pipeline):
                        last = trans.steps[-1][1]
                        if hasattr(last, "categories_"):  # OHE
                            cats = last.categories_
                            # considerar drop si aplica
                            drop_idx = getattr(last, "drop_idx_", None)
                            total += sum(len(c) for c in cats) - (0 if drop_idx is None else len(drop_idx))
                        elif hasattr(last, "n_features_in_"):
                            total += last.n_features_in_
                        else:
                            total += len(cols) if cols is not None else 0
                    else:
                        if hasattr(trans, "categories_"):  # OHE directo
                            cats = trans.categories_
                            total += sum(len(c) for c in cats)
                        elif hasattr(trans, "n_features_in_"):
                            total += trans.n_features_in_
                        else:
                            total += len(cols) if cols is not None else 0
                except Exception:
                    total += len(cols) if cols is not None else 0
            return total
    except Exception:
        return None
    return None

# ===================== Carga de artefactos =====================
@st.cache_resource(show_spinner=True)
def load_artifacts():
    # 1) Preferir pipeline único
    if os.path.exists(PIPE_PATH):
        try:
            pipe = joblib.load(PIPE_PATH)
            # si el pipe tiene un step 'pre', aplicar parche numpy2
            try:
                from sklearn.pipeline import Pipeline as SkPipe
                if hasattr(pipe, "named_steps") and "pre" in pipe.named_steps:
                    _sanitize_onehot_for_numpy2(pipe.named_steps["pre"])
            except Exception:
                pass
            return {"mode": "pipeline", "pipe": pipe, "model": None, "pre": None}
        except Exception as e:
            raise RuntimeError(f"Error cargando pipeline ({PIPE_PATH}): {e}")

    # 2) Fallback: artefactos por separado
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró modelo: {MODEL_PATH}")
    if not os.path.exists(PREPROC_PATH):
        raise FileNotFoundError(f"No se encontró preprocesador: {PREPROC_PATH}")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error cargando modelo ({MODEL_PATH}): {e}")

    try:
        pre = joblib.load(PREPROC_PATH)
        _sanitize_onehot_for_numpy2(pre)  # parche NumPy 2.x
    except Exception as e:
        raise RuntimeError(f"Error cargando preprocesador ({PREPROC_PATH}): {e}")

    return {"mode": "separate", "pipe": None, "model": model, "pre": pre}

# Intento de carga
try:
    art = load_artifacts()
except Exception as e:
    st.error("❌ No se pudo cargar artefactos.")
    st.caption("Revisa que exista pipeline_diabetes.joblib o el par modelo+preprocesador.")
    st.exception(e)
    st.stop()

# ===================== UI Principal =====================
st.title("🩺 Predicción de Diabetes (CSV)")
st.markdown("Sube un **CSV** con las columnas que espera el modelo/preprocesador.")

# Columnas esperadas para alinear el CSV
expected_cols = None
try:
    if art["mode"] == "pipeline":
        # intentar tomar 'pre' dentro del pipeline
        pre = None
        if hasattr(art["pipe"], "named_steps"):
            pre = art["pipe"].named_steps.get("pre", None)
        expected_cols = _get_expected_cols_from_pre(pre) if pre is not None else None
    else:
        expected_cols = _get_expected_cols_from_pre(art["pre"])
except Exception:
    expected_cols = None

if expected_cols:
    plantilla_df = pd.DataFrame(columns=expected_cols)
    st.download_button(
        "📥 Descargar plantilla CSV (columnas esperadas)",
        data=plantilla_df.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_prediccion_diabetes.csv",
        mime="text/csv",
    )
else:
    st.info("Si tu CSV no coincide con el esquema del entrenamiento, la transformación puede fallar.")

# ===================== Cargador de CSV y Predicción =====================
def align_if_possible(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if expected_cols:
        return align_columns(df.copy(), expected_cols)
    return df.copy(), [], []

uploaded_file = st.file_uploader("📂 Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Lectura con fallback de encoding
        try:
            df_nuevo = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df_nuevo = pd.read_csv(uploaded_file, encoding="latin-1")

        st.success(f"Archivo: {df_nuevo.shape[0]} filas × {df_nuevo.shape[1]} columnas")
        st.dataframe(df_nuevo.head(), use_container_width=True)

        # Alinear al esquema del preprocesador si lo conocemos
        df_aligned, missing, extra = align_if_possible(df_nuevo)
        if missing:
            st.warning(f"Se agregaron {len(missing)} columnas faltantes con 0: {missing}")
        if extra:
            st.info(f"Se ignoraron {len(extra)} columnas no usadas: {extra}")

        # Validación de compatibilidad (solo en modo separado)
        if art["mode"] == "separate":
            n_pre_out = pre_n_features_out(art["pre"])
            n_model_in = getattr(art["model"], "n_features_in_", None)
            if n_pre_out is not None and n_model_in is not None and n_pre_out != n_model_in:
                st.error(
                    f"Incompatibilidad entre artefactos: el preprocesador produce {n_pre_out} columnas, "
                    f"pero el modelo espera {n_model_in}. "
                    "Usa artefactos de la MISMA corrida o exporta un Pipeline único (recomendado)."
                )
                st.stop()

        # Botón de predicción
        if st.button("🔮 Realizar predicciones"):
            with st.spinner("Transformando y prediciendo..."):
                if art["mode"] == "pipeline":
                    # Predicción directa con pipeline
                    try:
                        preds = art["pipe"].predict(df_aligned)
                    except Exception as e:
                        st.error("Fallo al predecir con el pipeline.")
                        st.exception(e); st.stop()

                    probs = None
                    if hasattr(art["pipe"], "predict_proba"):
                        try:
                            probs = art["pipe"].predict_proba(df_aligned)[:, 1]
                        except Exception:
                            probs = None

                else:
                    # Transformación + predicción con artefactos separados
                    try:
                        X_new = art["pre"].transform(df_aligned)
                    except Exception as e:
                        st.error("Fallo al transformar con el preprocesador.")
                        st.exception(e); st.stop()

                    try:
                        preds = art["model"].predict(X_new)
                    except Exception as e:
                        st.error("Fallo al ejecutar predict() del modelo.")
                        st.exception(e); st.stop()

                    probs = None
                    if hasattr(art["model"], "predict_proba"):
                        try:
                            probs = art["model"].predict_proba(X_new)[:, 1]
                        except Exception:
                            probs = None

                # Resultado
                df_result = df_nuevo.copy()
                df_result["Predicción"] = np.where(np.asarray(preds) == 1, "Diabético", "No diabético")
                if probs is not None:
                    df_result["Probabilidad (%)"] = np.round(np.asarray(probs) * 100, 2)

                st.subheader("📊 Resultados")
                st.dataframe(df_result, use_container_width=True)
                st.download_button("💾 Descargar resultados (CSV)",
                                   df_result.to_csv(index=False).encode("utf-8"),
                                   "predicciones_diabetes.csv", "text/csv")

    except Exception as e:
        st.error("Error general procesando el CSV.")
        st.exception(e)
else:
    st.info("Sube un CSV con los datos a evaluar.")
