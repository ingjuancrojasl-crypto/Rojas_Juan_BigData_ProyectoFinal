"""
Dashboard Streamlit — Escalamiento de quejas financieras.

Versión inicial: verifica que la app lee el Parquet procesado y que MongoDB
responde. Las 5 secciones completas (Resumen, Datos, Visualizaciones, Modelo,
Insights y reporte) se construyen en la siguiente fase del proyecto.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config  # noqa: E402

st.set_page_config(page_title="Escalamiento de quejas financieras", layout="wide")
st.title("Escalamiento de quejas financieras en Colombia")
st.caption("Fuente: Superintendencia Financiera de Colombia — Smartsupervision (datos.gov.co, xyy7-rn7p)")


@st.cache_data
def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(config.ARCHIVO_PARQUET_LIMPIO)


if not config.ARCHIVO_PARQUET_LIMPIO.exists():
    st.warning("Aún no existe el Parquet limpio. Espere a que el servicio de procesamiento termine el ETL "
               "y recargue la página.")
    st.stop()

datos = cargar_datos()
total_quejas = int(datos["cantidad_quejas"].sum())
quejas_escaladas = int((datos["cantidad_quejas"] * datos["escalada"]).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", f"{len(datos):,}".replace(",", "."))
col2.metric("Quejas", f"{total_quejas:,}".replace(",", "."))
col3.metric("Quejas escaladas", f"{quejas_escaladas / total_quejas:.1%}")
col4.metric("Registros escalados", f"{datos['escalada'].mean():.1%}")

st.subheader("Muestra del dataset limpio")
st.dataframe(datos.head(200), width="stretch")

with st.expander("Estado de los servicios"):
    try:
        from pymongo import MongoClient
        cliente = MongoClient(os.getenv("MONGO_URI", config.MONGO_URI), serverSelectionTimeoutMS=3000)
        cliente.admin.command("ping")
        st.success("MongoDB responde correctamente")
        cliente.close()
    except Exception as error:  # la app sigue funcionando aunque Mongo no esté listo
        st.error(f"No se pudo conectar a MongoDB: {error}")
