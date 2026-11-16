"""
Dashboard Streamlit — Gravedad de los siniestros de transporte en Colombia (2015-2024).

Versión inicial: verifica que la app lee los Parquet procesados y que MongoDB
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

st.set_page_config(page_title="Siniestros de transporte en Colombia", layout="wide")
st.title("Gravedad de los siniestros de transporte en Colombia, 2015-2024")
st.caption("Fuentes: Medicina Legal (lesiones y muertes por eventos de transporte) y Policía Nacional "
           "(lesiones y homicidios culposos en accidente de tránsito) — datos.gov.co")


@st.cache_data
def cargar(ruta: str) -> pd.DataFrame:
    return pd.read_parquet(ruta)


if not (config.ARCHIVO_UNIFICADO.exists() and config.ARCHIVO_DETALLE.exists()):
    st.warning("Aún no existen los Parquet procesados. Espere a que el servicio de procesamiento termine el ETL "
               "y recargue la página.")
    st.stop()

unificado = cargar(str(config.ARCHIVO_UNIFICADO))
victimas = cargar(str(config.ARCHIVO_DETALLE))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Víctimas (4 fuentes)", f"{len(unificado):,}".replace(",", "."))
col2.metric("Víctimas Medicina Legal", f"{len(victimas):,}".replace(",", "."))
col3.metric("Muertes (Medicina Legal)", f"{(victimas['desenlace'] == 'Muerte').sum():,}".replace(",", "."))
con_gravedad = victimas[victimas["gravedad"] != config.SIN_INFORMACION]
col4.metric("Lesiones graves (> 30 días)", f"{(con_gravedad['gravedad'] == 'Lesión grave').mean():.1%}")

st.subheader("Víctimas por fuente y desenlace")
st.dataframe(pd.crosstab(unificado["fuente"], unificado["desenlace"], margins=True), width="stretch")

st.subheader("Muestra de la base analítica (Medicina Legal)")
st.dataframe(victimas.head(200), width="stretch")

with st.expander("Estado de los servicios"):
    try:
        from pymongo import MongoClient
        cliente = MongoClient(os.getenv("MONGO_URI", config.MONGO_URI), serverSelectionTimeoutMS=3000)
        cliente.admin.command("ping")
        st.success("MongoDB responde correctamente")
        cliente.close()
    except Exception as error:
        st.error(f"No se pudo conectar a MongoDB: {error}")
