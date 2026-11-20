"""Dashboard Streamlit — Gravedad de los siniestros de transporte en Colombia (2015-2024).

Cinco secciones: Resumen, Datos, Visualizaciones, Modelo e Insights y reporte.
La app SOLO LEE resultados ya procesados: Parquet del ETL (data/processed), el modelo guardado (models/)
y el documento de análisis en MongoDB. No repite el ETL ni reentrena el modelo.

Ejecutar:  streamlit run app/app.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import config  # noqa: E402

SI = config.SIN_INFORMACION
AZUL, ROJO, VERDE, GRIS = "#3a65a9", "#c62828", "#1f9e89", "#9e9e9e"
COLOR_DESENLACE = {"Lesión": AZUL, "Muerte": ROJO}
COLOR_GRAVEDAD = {"Lesión leve": "#4ea1e8", "Lesión grave": "#1f4e96", "Muerte": ROJO}
ORDEN_GRAVEDAD = ["Lesión leve", "Lesión grave", "Muerte"]
ORDEN_DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
FUENTE_DATOS = "Fuente: INMLCF y Policía Nacional (datos.gov.co), 2015-2024. Elaboración propia."

st.set_page_config(page_title="Siniestros de transporte en Colombia", page_icon="🚦", layout="wide")
st.markdown("""
<style>
  /* ---------- Contenido ---------- */
  .block-container {padding-top: 3.2rem; max-width: 1400px;}

  /* ---------- Encabezado (banner azul) ---------- */
  .banner {background: linear-gradient(120deg, #0d2a5c 0%, #1f4e96 60%, #3a65a9 100%); color: #ffffff;
           padding: 22px 28px; border-radius: 14px; margin-bottom: 18px;
           box-shadow: 0 6px 18px rgba(13, 42, 92, 0.25);}
  .banner h1 {color: #ffffff; font-size: 1.9rem; margin: 0 0 6px 0; padding: 0; line-height: 1.25;}
  .banner p {color: #dbe7f7; margin: 0; font-size: 0.92rem;}

  /* ---------- Barra lateral azul ---------- */
  section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0d2a5c 0%, #163d7a 55%, #1f4e96 100%);}
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] small,
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {color: #ffffff !important;}
  section[data-testid="stSidebar"] hr {border-color: rgba(255, 255, 255, 0.25);}
  /* Cajas de selección: fondo blanco y texto oscuro para que se lean */
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {background: #ffffff; border-radius: 8px;}
  section[data-testid="stSidebar"] div[data-baseweb="select"] span,
  section[data-testid="stSidebar"] div[data-baseweb="select"] div {color: #1f2937 !important;}
  section[data-testid="stSidebar"] span[data-baseweb="tag"] {background: #3a65a9 !important;}
  section[data-testid="stSidebar"] span[data-baseweb="tag"] span {color: #ffffff !important;}
  /* Control deslizante de años en tonos claros */
  section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
  section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
  section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {color: #ffffff !important;}
  .lateral-titulo {font-size: 1.15rem; font-weight: 700; color: #ffffff; margin: 0;}
  .lateral-sub {font-size: 0.85rem; color: #c9dbf5; margin: 2px 0 0 0;}
  .lateral-estado {background: rgba(255, 255, 255, 0.10); border-radius: 8px; padding: 10px 12px;
                   font-size: 0.82rem; color: #ffffff; line-height: 1.55;}

  /* ---------- Pestañas: distribuidas y resaltadas en azul (Streamlit nuevo y anterior) ---------- */
  div[role="tablist"] {display: flex; gap: 8px; background: #eef3fa; padding: 6px; border-radius: 12px;
                       border: 1px solid #dbe5f3;}
  div[role="tablist"] [role="tab"] {flex: 1 1 0; display: flex; justify-content: center; align-items: center;
                       height: 46px; margin: 0; border-radius: 9px; background: transparent; cursor: pointer;
                       transition: background 0.15s ease, box-shadow 0.15s ease;}
  div[role="tablist"] [role="tab"] p {font-size: 1rem; font-weight: 600; color: #1f4e96; margin: 0;}
  div[role="tablist"] [role="tab"]:hover {background: #dbe7f7;}
  div[role="tablist"] [role="tab"][aria-selected="true"] {background: linear-gradient(120deg, #1f4e96, #3a65a9);
                       box-shadow: 0 3px 10px rgba(31, 78, 150, 0.35);}
  div[role="tablist"] [role="tab"][aria-selected="true"] p {color: #ffffff;}
  .react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"]
      {display: none !important;}
  div[data-testid="stTabs"] [role="tabpanel"] {padding-top: 14px;}

  /* ---------- Indicadores (tarjetas) ---------- */
  div[data-testid="stMetric"] {background: #ffffff; border-left: 5px solid #3a65a9; padding: 12px 16px;
                               border-radius: 10px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);}
  div[data-testid="stMetricValue"] {font-size: 1.65rem; color: #1f4e96; font-weight: 700;}
  div[data-testid="stMetricLabel"] p {color: #5f6368; font-weight: 600;}

  /* ---------- Tarjetas de insights ---------- */
  .tarjeta {background: #ffffff; border-left: 5px solid #3a65a9; padding: 12px 16px; border-radius: 10px;
            margin-bottom: 12px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);}
  .tarjeta .tema {color: #c62828; font-weight: 700; font-size: 0.85rem;}
  .tarjeta .titulo {font-weight: 700; margin: 2px 0 4px 0; color: #0d2a5c;}
  .tarjeta .pie {color: #5f6368; font-size: 0.8rem;}
  h2, h3, h4 {color: #0d2a5c;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------
def miles(valor) -> str:
    return f"{int(round(float(valor))):,}".replace(",", ".")


def dec(valor, decimales: int = 3) -> str:
    return f"{float(valor):.{decimales}f}".replace(".", ",")


def pct(valor, decimales: int = 1) -> str:
    return f"{dec(valor, decimales)} %"


def estilo_grafica(fig: go.Figure, titulo: str, subtitulo: str = "", alto: int = 430) -> go.Figure:
    """Estilo común (igual al de las gráficas del notebook): título centrado en dos líneas, cuadrícula punteada."""
    texto = f"<b>{titulo}</b>" + (f"<br><span style='font-size:13px'>{subtitulo}</span>" if subtitulo else "")
    fig.update_layout(
        title=dict(text=texto, x=0.5, xanchor="center", font=dict(size=16)),
        template="simple_white", height=alto, margin=dict(t=80, l=10, r=10, b=85),
        legend=dict(title_font=dict(size=12), bordercolor="#cccccc", borderwidth=1),
        font=dict(family="Arial", size=12, color="#222222"), hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#333333")
    fig.update_yaxes(showgrid=True, gridcolor="#d9d9d9", griddash="dot", showline=True, linecolor="#333333")
    fig.add_annotation(text=FUENTE_DATOS, xref="paper", yref="paper", x=0, y=0, yshift=-75, showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=10, color="#777777"))
    return fig


def orden_edad(valor: str) -> int:
    numeros = re.findall(r"\d+", str(valor))
    return int(numeros[0]) if numeros else -1  # "Fetos" queda antes de (0 a 4)


# ---------------------------------------------------------------------------
# Carga de resultados (con caché: se leen una sola vez)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo datos procesados (Parquet)…")
def cargar_parquet() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    unificado = pd.read_parquet(config.ARCHIVO_UNIFICADO)
    victimas = pd.read_parquet(config.ARCHIVO_DETALLE)
    cubos = {r.stem: pd.read_parquet(r) for r in sorted(config.DIR_AGREGACIONES.glob("*.parquet"))}
    return unificado, victimas, cubos


@st.cache_resource(show_spinner="Cargando el modelo guardado…")
def cargar_modelo():
    import modelo
    return modelo.cargar_modelo()


@st.cache_data(ttl=300, show_spinner="Consultando MongoDB…")
def cargar_documento() -> tuple[dict | None, str]:
    """Lee el documento de análisis desde MongoDB; si no responde, usa la copia en reports/."""
    try:
        import mongo_analisis
        cliente, coleccion = mongo_analisis.conectar(timeout_ms=3000)
        documento = coleccion.find_one({"_id": config.MONGO_ID_DOCUMENTO})
        cliente.close()
        if documento:
            return documento, "MongoDB"
        mensaje = "MongoDB respondió, pero el documento no existe (ejecute la sección 6 del notebook)."
    except Exception as error:  # noqa: BLE001
        mensaje = f"MongoDB no disponible ({type(error).__name__})."
    copia = config.DIR_REPORTES / "documento_mongodb.json"
    if copia.exists():
        return json.loads(copia.read_text(encoding="utf-8")), f"copia local (reports/documento_mongodb.json). {mensaje}"
    return None, mensaje


faltan = [p.name for p in (config.ARCHIVO_UNIFICADO, config.ARCHIVO_DETALLE) if not p.exists()]
if faltan:
    st.error(f"No existen los datos procesados {faltan}. Ejecute primero el notebook (ETL) o espere a que el "
             "servicio de procesamiento termine, y recargue la página.")
    st.stop()

unificado, victimas, cubos = cargar_parquet()
documento, origen_documento = cargar_documento()
try:
    modelo_ajustado, meta_modelo = cargar_modelo()
except Exception as error:  # noqa: BLE001
    modelo_ajustado, meta_modelo = None, None
    st.warning(f"No se pudo cargar el modelo de models/ ({type(error).__name__}). Ejecute la sección 5 del notebook.")

ANIO_MIN, ANIO_MAX = int(unificado["anio"].min()), int(unificado["anio"].max())
REGIONES = sorted(r for r in unificado["region"].astype(str).unique() if r != SI)
MEDIOS = sorted(m for m in victimas["medio_grupo"].astype(str).unique() if m != SI)

# ---------------------------------------------------------------------------
# Encabezado y filtros globales (barra lateral)
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='banner'><h1>🚦 Gravedad de los siniestros de transporte en Colombia</h1>"
    "<p>Proyecto final — Gerencia y almacenamiento en Big Data · Juan Rojas · Medicina Legal (INMLCF) y "
    f"Policía Nacional · <a href='https://www.datos.gov.co' style='color:#9cc4ff'>datos.gov.co</a> · "
    f"{ANIO_MIN}-{ANIO_MAX}</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<p class='lateral-titulo'>🚦 Siniestros de transporte</p>"
                f"<p class='lateral-sub'>Colombia · {ANIO_MIN}-{ANIO_MAX}</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 🔎 Filtros")
    st.caption("Se aplican a las pestañas **Datos** y **Visualizaciones**.")
    anios = st.slider("📅 Años", ANIO_MIN, ANIO_MAX, (ANIO_MIN, ANIO_MAX))
    regiones = st.multiselect("🗺️ Región natural", REGIONES, placeholder="Todas")
    medios = st.multiselect("🏍️ Medio de transporte", MEDIOS, placeholder="Todos")
    st.divider()
    estado_doc = "MongoDB ✅" if origen_documento == "MongoDB" else "copia local ⚠️"
    st.markdown(
        "<div class='lateral-estado'>"
        f"<b>Documento de análisis:</b> {estado_doc}<br>"
        f"<b>Modelo:</b> {meta_modelo['nombre_modelo'] if meta_modelo else 'no disponible'}<br>"
        "<span style='color:#c9dbf5'>La app no repite el ETL ni reentrena: solo lee Parquet, MongoDB y el "
        "modelo guardado.</span></div>", unsafe_allow_html=True)

def filtrar(df: pd.DataFrame) -> pd.DataFrame:
    mascara = df["anio"].between(*anios)
    if regiones and "region" in df:
        mascara &= df["region"].astype(str).isin(regiones)
    if medios and "medio_grupo" in df:
        mascara &= df["medio_grupo"].astype(str).isin(medios)
    return df[mascara]


uni_f = filtrar(unificado)
vic_f = filtrar(victimas)
con_gravedad = vic_f[vic_f["gravedad"].astype(str) != SI]

pestanas = st.tabs(["📊 Resumen", "🗂️ Datos", "📈 Visualizaciones", "🤖 Modelo", "💡 Insights y reporte"])

# ===========================================================================
# 1. RESUMEN
# ===========================================================================
with pestanas[0]:
    st.subheader("Pregunta de negocio")
    st.info(documento["metadata"]["pregunta_negocio"] if documento else
            "¿Qué características de la víctima, del vehículo y del contexto se asocian con la gravedad de un "
            "siniestro de transporte?")

    st.markdown("#### Indicadores del dataset")
    total_ml = victimas[victimas["gravedad"].astype(str) != SI]
    c = st.columns(5)
    c[0].metric("Víctimas (4 fuentes)", miles(len(unificado)))
    c[1].metric("Muertes (Medicina Legal)", miles((victimas["desenlace"] == "Muerte").sum()))
    c[2].metric("Lesionados (Medicina Legal)", miles((victimas["desenlace"] == "Lesión").sum()))
    c[3].metric("Letalidad", pct((total_ml["gravedad"] == "Muerte").mean() * 100),
                help="Porcentaje de víctimas de Medicina Legal con gravedad conocida que fallecen.")
    c[4].metric("Fuentes unificadas", "4", help="INMLCF lesiones y muertes; Policía lesiones y homicidios.")

    if meta_modelo:
        st.markdown("#### Indicadores del modelo")
        m = meta_modelo["metricas_prueba"]
        base = max(x["f1_macro"] for x in meta_modelo["comparacion_modelos"] if x["modelo"].startswith("Línea base"))
        st.caption(f"Modelo elegido: **{meta_modelo['nombre_modelo']}** · validación temporal (prueba 2023-2024)")
        c = st.columns(5)
        c[0].metric("Variables usadas", str(len(meta_modelo["variables"])))
        c[1].metric("F1 macro", dec(m["f1_macro"]), f"+{dec((m['f1_macro'] - base) * 100, 1)} pts vs línea base")
        c[2].metric("Recall de Muerte", pct(m["recall_Muerte"] * 100, 0), help="De cada 100 muertes, cuántas detecta.")
        c[3].metric("Accuracy", dec(m["accuracy"]))
        c[4].metric("Criterio de éxito", "✅ Cumple" if meta_modelo["criterio_exito"]["cumple_criterio"] else "❌ No")

    serie = (unificado.groupby(["anio", "fuente", "desenlace"], observed=True).size().rename("victimas").reset_index())
    serie["serie"] = serie["fuente"].astype(str) + " — " + serie["desenlace"].astype(str)
    izquierda, derecha = st.columns(2)
    with izquierda:
        fig = px.line(serie[serie["desenlace"] == "Muerte"], x="anio", y="victimas", color="fuente", markers=True,
                      color_discrete_sequence=[ROJO, "#ef8a62"], labels={"anio": "Año", "victimas": "Muertes",
                                                                         "fuente": "Fuente"})
        fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="#bdbdbd", opacity=0.35, line_width=0,
                      annotation_text="Pandemia", annotation_position="top left")
        st.plotly_chart(estilo_grafica(fig, "Muertes por Año", "Medicina Legal y Policía Nacional"), width="stretch")
    with derecha:
        fig = px.line(serie[serie["desenlace"] == "Lesión"], x="anio", y="victimas", color="fuente", markers=True,
                      color_discrete_sequence=[AZUL, "#67a9cf"], labels={"anio": "Año", "victimas": "Lesionados",
                                                                         "fuente": "Fuente"})
        fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor="#bdbdbd", opacity=0.35, line_width=0,
                      annotation_text="Pandemia", annotation_position="top left")
        st.plotly_chart(estilo_grafica(fig, "Lesionados por Año", "Medicina Legal y Policía Nacional"), width="stretch")

# ===========================================================================
# 2. DATOS
# ===========================================================================
with pestanas[1]:
    st.subheader("Muestra del dataset limpio")
    fuente_tabla = st.radio("Dataset", ["Detalle Medicina Legal (variables del modelo)",
                                        "Unificado (4 fuentes)"], horizontal=True)
    if fuente_tabla.startswith("Detalle"):
        columnas = ["anio", "mes", "dia_semana", "departamento", "region", "municipio", "zona", "sexo",
                    "grupo_edad_quinquenal", "condicion_victima", "medio_grupo", "clase_accidente", "objeto_grupo",
                    "causa_categoria", "desenlace", "gravedad"]
        base_tabla = vic_f[columnas]
    else:
        base_tabla = uni_f.drop(columns=["archivo_origen", "medio_incoherente"], errors="ignore")
    c1, c2, c3 = st.columns([1, 1, 2])
    desenlaces = c1.multiselect("Desenlace", sorted(base_tabla["desenlace"].astype(str).unique()), placeholder="Todos")
    tamano = c2.select_slider("Filas de la muestra", [100, 500, 1000, 5000, 10000], value=1000)
    texto = c3.text_input("Buscar municipio o departamento", placeholder="Ej.: Medellín")
    if desenlaces:
        base_tabla = base_tabla[base_tabla["desenlace"].astype(str).isin(desenlaces)]
    if texto:
        patron = texto.strip().lower()
        base_tabla = base_tabla[base_tabla["municipio"].astype(str).str.lower().str.contains(patron, regex=False)
                                | base_tabla["departamento"].astype(str).str.lower().str.contains(patron, regex=False)]
    muestra = base_tabla.sample(min(tamano, len(base_tabla)), random_state=42) if len(base_tabla) else base_tabla
    st.caption(f"{miles(len(base_tabla))} registros cumplen los filtros; se muestra una muestra aleatoria de "
               f"{miles(len(muestra))}.")
    st.dataframe(muestra, width="stretch", height=360, hide_index=True)
    st.download_button("⬇️ Descargar muestra (CSV)", muestra.to_csv(index=False).encode("utf-8-sig"),
                       "muestra_siniestros.csv", "text/csv")

    st.divider()
    st.subheader("Tablas de agregación (cubos OLAP)")
    nombres_cubos = {
        "cubo_temporal": "Temporal: año × mes × día × fuente × desenlace",
        "cubo_geografico": "Geográfico: año × región × departamento × fuente × desenlace",
        "cubo_perfil_victima": "Perfil de la víctima: sexo × edad × medio",
        "cubo_gravedad_medicina_legal": "Gravedad (Medicina Legal): condición × medio × clase × objeto × zona",
        "cubo_causas": "Causas: categoría × desenlace × gravedad",
        "cobertura_entre_fuentes": "Cobertura entre fuentes: Policía vs Medicina Legal por departamento",
    }
    disponibles = [c for c in nombres_cubos if c in cubos]
    elegido = st.selectbox("Cubo", disponibles, format_func=lambda c: nombres_cubos[c])
    cubo = filtrar(cubos[elegido]) if "anio" in cubos[elegido] else cubos[elegido]
    dimensiones = [c for c in cubo.columns if c not in ("victimas", "Medicina Legal", "Policía Nacional",
                                                         "razon_policia_vs_medicina_legal")]
    c1, c2 = st.columns([2, 1])
    agrupar = c1.multiselect("Agrupar por (roll-up)", dimensiones, default=dimensiones[:2])
    if "victimas" in cubo and agrupar:
        tabla_cubo = (cubo.groupby(agrupar, observed=True)["victimas"].sum().reset_index()
                      .sort_values("victimas", ascending=False))
        if "desenlace" in agrupar and len(agrupar) > 1 and c2.checkbox("Pivotar desenlace en columnas", value=True):
            otras = [a for a in agrupar if a != "desenlace"]
            tabla_cubo = tabla_cubo.pivot_table(index=otras, columns="desenlace", values="victimas", aggfunc="sum",
                                                observed=True, fill_value=0).reset_index()
            if {"Lesión", "Muerte"} <= set(tabla_cubo.columns):
                tabla_cubo["letalidad_%"] = (tabla_cubo["Muerte"] / (tabla_cubo["Lesión"] + tabla_cubo["Muerte"])
                                             * 100).round(1)
    else:
        tabla_cubo = cubo
    st.dataframe(tabla_cubo, width="stretch", height=340, hide_index=True)
    st.download_button("⬇️ Descargar tabla (CSV)", tabla_cubo.to_csv(index=False).encode("utf-8-sig"),
                       f"{elegido}.csv", "text/csv")

    if documento:
        with st.expander("Agregaciones guardadas en MongoDB"):
            for ag in documento["agregaciones"]:
                st.markdown(f"**{ag['nombre']}** · {ag['operacion_olap']} · medida: {ag['medida']}")
                st.dataframe(pd.DataFrame(ag["resultado"]), width="stretch", hide_index=True)

# ===========================================================================
# 3. VISUALIZACIONES
# ===========================================================================
with pestanas[2]:
    st.caption(f"Filtros activos: años {anios[0]}-{anios[1]} · región: {', '.join(regiones) or 'todas'} · "
               f"medio: {', '.join(medios) or 'todos'}. Pase el cursor sobre las gráficas para ver los valores.")
    if con_gravedad.empty:
        st.warning("No hay víctimas con los filtros seleccionados. Amplíe los filtros de la barra lateral.")
    else:

        # 3.1 Evolución anual por fuente y desenlace
        serie_f = uni_f.groupby(["anio", "fuente", "desenlace"], observed=True).size().rename("victimas").reset_index()
        fig = px.line(serie_f, x="anio", y="victimas", color="desenlace", line_dash="fuente", markers=True,
                      color_discrete_map=COLOR_DESENLACE,
                      labels={"anio": "Año", "victimas": "Número de víctimas", "desenlace": "Desenlace", "fuente": "Fuente"})
        st.plotly_chart(estilo_grafica(fig, "Evolución Anual de las Víctimas", "Por fuente y desenlace"), width="stretch")

        izquierda, derecha = st.columns(2)
        # 3.2 Medio de transporte
        with izquierda:
            medio = (pd.crosstab(vic_f["medio_grupo"].astype(str), vic_f["desenlace"].astype(str), normalize="columns")
                     * 100).drop(index=SI, errors="ignore").reset_index().melt(id_vars="medio_grupo", var_name="desenlace",
                                                                              value_name="porcentaje")
            fig = px.bar(medio, y="medio_grupo", x="porcentaje", color="desenlace", barmode="group", orientation="h",
                         color_discrete_map=COLOR_DESENLACE, text=medio["porcentaje"].map(lambda v: f"{v:.0f} %"),
                         labels={"medio_grupo": "Medio de transporte", "porcentaje": "% de las víctimas",
                                 "desenlace": "Desenlace"})
            fig.update_traces(textposition="outside", textfont_size=11, cliponaxis=False)
            fig.update_xaxes(range=[0, medio["porcentaje"].max() * 1.2])
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(estilo_grafica(fig, "Víctimas por Medio de Transporte", "Lesionados y fallecidos (%)"),
                            width="stretch")
        # 3.3 Gravedad por edad
        with derecha:
            edad = con_gravedad.groupby(["grupo_edad_quinquenal", "gravedad"], observed=True).size().reset_index(name="n")
            edad = edad[edad["grupo_edad_quinquenal"].astype(str) != SI]
            edad["porcentaje"] = edad["n"] / edad.groupby("grupo_edad_quinquenal", observed=True)["n"].transform("sum") * 100
            orden = sorted(edad["grupo_edad_quinquenal"].astype(str).unique(), key=orden_edad)
            fig = px.bar(edad, y="grupo_edad_quinquenal", x="porcentaje", color="gravedad", orientation="h",
                         color_discrete_map=COLOR_GRAVEDAD, category_orders={"gravedad": ORDEN_GRAVEDAD,
                                                                             "grupo_edad_quinquenal": orden[::-1]},
                         labels={"grupo_edad_quinquenal": "Edad de la víctima", "porcentaje": "% de las víctimas",
                                 "gravedad": "Gravedad"}, hover_data={"n": True})
            st.plotly_chart(estilo_grafica(fig, "Gravedad del Siniestro según la Edad", "Distribución 100 %", alto=520),
                            width="stretch")

        izquierda, derecha = st.columns(2)
        # 3.4 Letalidad por objeto de colisión
        with izquierda:
            objeto = (con_gravedad[con_gravedad["objeto_grupo"].astype(str) != SI]
                      .groupby("objeto_grupo", observed=True)["gravedad"]
                      .agg(letalidad=lambda g: (g == "Muerte").mean() * 100, victimas="size").reset_index()
                      .sort_values("letalidad"))
            fig = px.bar(objeto, y="objeto_grupo", x="letalidad", orientation="h", color="letalidad",
                         color_continuous_scale="Reds", text=objeto["letalidad"].map(lambda v: f"{v:.0f} %"),
                         hover_data={"victimas": ":,"},
                         labels={"objeto_grupo": "Objeto de colisión", "letalidad": "% de víctimas que fallecen"})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, objeto["letalidad"].max() * 1.18])
            st.plotly_chart(estilo_grafica(fig, "Letalidad según el Objeto de Colisión", "Porcentaje de fallecidos"),
                            width="stretch")
        # 3.5 Causas
        with derecha:
            causas = vic_f[~vic_f["causa_categoria"].astype(str).isin([SI, "Otras / no especificadas"])]
            causas = (pd.crosstab(causas["causa_categoria"].astype(str), causas["desenlace"].astype(str),
                                  normalize="columns") * 100).reset_index().melt(id_vars="causa_categoria",
                                                                                var_name="desenlace", value_name="porcentaje")
            fig = px.bar(causas, y="causa_categoria", x="porcentaje", color="desenlace", barmode="group", orientation="h",
                         color_discrete_map=COLOR_DESENLACE, text=causas["porcentaje"].map(lambda v: f"{v:.0f} %"),
                         labels={"causa_categoria": "Causa", "porcentaje": "% de víctimas con causa registrada",
                                 "desenlace": "Desenlace"})
            fig.update_traces(textposition="outside", textfont_size=10, cliponaxis=False)
            fig.update_xaxes(range=[0, causas["porcentaje"].max() * 1.2])
            fig.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(estilo_grafica(fig, "Causas en Lesionados y Fallecidos", "Víctimas con causa registrada"),
                            width="stretch")

        # 3.6 Mapa de calor región × zona
        heat = con_gravedad[(con_gravedad["region"].astype(str) != SI) & (con_gravedad["zona"].astype(str) != SI)]
        heat = heat.groupby(["region", "zona"], observed=True)["gravedad"].agg(
            letalidad=lambda g: (g == "Muerte").mean() * 100, victimas="size").reset_index()
        totales_region = heat.groupby("region", observed=True)["victimas"].transform("sum")
        heat = heat[(heat["victimas"] >= 30) & (totales_region >= 1000)]  # regiones pequeñas (Insular) se excluyen
        matriz = heat.pivot(index="region", columns="zona", values="letalidad")
        conteo = heat.pivot(index="region", columns="zona", values="victimas")
        texto_celdas = pd.DataFrame(
            [[f"<b>{matriz.iat[i, j]:.0f} %</b><br>({miles(conteo.iat[i, j])})" if pd.notna(matriz.iat[i, j]) else ""
              for j in range(matriz.shape[1])] for i in range(matriz.shape[0])])
        fig = go.Figure(go.Heatmap(z=matriz.values, x=matriz.columns, y=matriz.index, colorscale="Viridis",
                                   text=texto_celdas.values, texttemplate="%{text}", xgap=3, ygap=3,
                                   colorbar=dict(title="% fallecidos", ticksuffix=" %"),
                                   hovertemplate="%{y} · %{x}<br>Letalidad: %{z:.1f} %<extra></extra>"))
        fig = estilo_grafica(fig, "Letalidad por Región Natural y Zona del Hecho", "Porcentaje de víctimas fallecidas "
                             "(regiones con 1.000 víctimas o más)", alto=460)
        fig.update_xaxes(title="Zona del hecho")
        fig.update_yaxes(title="Región natural", showgrid=False)
        st.plotly_chart(fig, width="stretch")

        # 3.7 Departamentos
        deptos = (uni_f[uni_f["desenlace"] == "Muerte"].groupby(["departamento", "fuente"], observed=True).size()
                  .rename("muertes").reset_index())
        top = deptos.groupby("departamento", observed=True)["muertes"].sum().nlargest(15).index
        deptos = deptos[deptos["departamento"].isin(top)]
        fig = px.bar(deptos, y="departamento", x="muertes", color="fuente", barmode="group", orientation="h",
                     color_discrete_sequence=[ROJO, "#ef8a62"],
                     category_orders={"departamento": list(top.astype(str))},
                     labels={"departamento": "Departamento", "muertes": "Muertes", "fuente": "Fuente"})
        st.plotly_chart(estilo_grafica(fig, "Los 15 Departamentos con más Muertes", "Comparación entre fuentes",
                                       alto=620), width="stretch")

# ===========================================================================
# 4. MODELO
# ===========================================================================
with pestanas[3]:
    if not meta_modelo:
        st.warning("El modelo no está disponible.")
    else:
        m = meta_modelo["metricas_prueba"]
        st.markdown(f"**Tarea:** {meta_modelo['tarea']} ({', '.join(meta_modelo['clases'])}). "
                    f"**Validación temporal:** entrenamiento {meta_modelo['periodo_entrenamiento'][0]}-"
                    f"{meta_modelo['periodo_entrenamiento'][1]} ({miles(meta_modelo['registros_entrenamiento'])} "
                    f"víctimas), prueba {meta_modelo['periodo_prueba'][0]}-{meta_modelo['periodo_prueba'][1]} "
                    f"({miles(meta_modelo['registros_prueba'])} víctimas).")
        c = st.columns(5)
        for col, (clave, nombre) in zip(c, [("accuracy", "Accuracy"), ("precision_macro", "Precisión macro"),
                                            ("recall_macro", "Recall macro"), ("f1_macro", "F1 macro"),
                                            ("recall_Muerte", "Recall de Muerte")]):
            col.metric(nombre, dec(m[clave]))

        st.subheader("Comparación de modelos")
        comparacion = pd.DataFrame(meta_modelo["comparacion_modelos"])
        izquierda, derecha = st.columns([3, 2])
        with izquierda:
            comparacion_g = comparacion.assign(modelo=comparacion["modelo"].str.replace(": ", ":<br>", regex=False))
            largo = comparacion_g.melt(id_vars="modelo", value_vars=["f1_macro", "recall_Muerte", "accuracy"],
                                     var_name="métrica", value_name="valor")
            largo["métrica"] = largo["métrica"].map({"f1_macro": "F1 macro", "recall_Muerte": "Recall de Muerte",
                                                     "accuracy": "Accuracy"})
            fig = px.bar(largo, x="modelo", y="valor", color="métrica", barmode="group", text_auto=".2f",
                         color_discrete_sequence=[AZUL, ROJO, GRIS],
                         labels={"modelo": "Modelo", "valor": "Valor (0 a 1)", "métrica": "Métrica"})
            fig.update_xaxes(tickangle=0)
            fig.add_hline(y=0.70, line_dash="dash", line_color=ROJO,
                          annotation_text="Meta recall de Muerte = 0,70", annotation_position="top left")
            st.plotly_chart(estilo_grafica(fig, "Desempeño en Datos de Prueba", "Modelos frente a líneas base"),
                            width="stretch")
        with derecha:
            columnas = ["modelo", "accuracy", "precision_macro", "recall_macro", "f1_macro", "recall_Muerte",
                        "precision_Muerte", "segundos_entrenamiento"]
            st.dataframe(comparacion[columnas].round(3), width="stretch", hide_index=True)
            criterio = meta_modelo["criterio_exito"]
            st.success(f"**Criterio de éxito:** mejora de {dec(criterio['mejora_vs_linea_base'] * 100, 1)} puntos en F1 "
                       f"macro (meta ≥ 10) y recall de Muerte {dec(criterio['recall_Muerte'])} (meta ≥ 0,70) → "
                       f"**{'CUMPLE' if criterio['cumple_criterio'] else 'NO CUMPLE'}**")

        izquierda, derecha = st.columns(2)
        with izquierda:
            etiquetas = meta_modelo["matriz_confusion"]["etiquetas"]
            valores = np.array(meta_modelo["matriz_confusion"]["valores"])
            porcentaje = valores / valores.sum(axis=1, keepdims=True) * 100
            texto = [[f"<b>{porcentaje[i, j]:.0f} %</b><br>({miles(valores[i, j])})" for j in range(3)] for i in range(3)]
            fig = go.Figure(go.Heatmap(z=porcentaje, x=etiquetas, y=etiquetas, colorscale="Viridis", text=texto,
                                       texttemplate="%{text}", xgap=3, ygap=3,
                                       colorbar=dict(title="% de la clase real", ticksuffix=" %"),
                                       hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{z:.1f} %<extra></extra>"))
            fig = estilo_grafica(fig, "Matriz de Confusión", f"{meta_modelo['nombre_modelo']} — prueba 2023-2024")
            fig.update_xaxes(title="Gravedad predicha")
            fig.update_yaxes(title="Gravedad real", autorange="reversed", showgrid=False)
            st.plotly_chart(fig, width="stretch")
        with derecha:
            ruta_imp = config.DIR_MODELOS / "importancia_variables.csv"
            if ruta_imp.exists():
                imp = pd.read_csv(ruta_imp).sort_values("importancia")
                fig = px.bar(imp, x="importancia", y="variable_negocio", orientation="h", color="importancia",
                             color_continuous_scale="Blues", text=imp["importancia"].map(lambda v: f"{v:.3f}"),
                             labels={"importancia": "Caída del F1 macro", "variable_negocio": "Variable"})
                fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, imp["importancia"].max() * 1.2])
                fig.update_traces(textposition="outside", cliponaxis=False)
                st.plotly_chart(estilo_grafica(fig, "Importancia de las Variables", "Importancia por permutación"),
                                width="stretch")

        # ---- Formulario de predicción -------------------------------------
        st.divider()
        st.subheader("🔮 Simulador: probabilidad de cada nivel de gravedad")
        st.caption("Describa la víctima y el siniestro. El modelo guardado estima la probabilidad de lesión leve, "
                   "lesión grave o muerte. Es una estimación estadística (asociación), no un diagnóstico.")
        import modelo as modulo_modelo
        base = modulo_modelo.PERFIL_BASE

        def opciones(columna: str) -> list[str]:
            valores = [v for v in victimas[columna].astype(str).unique() if v != SI]
            return sorted(valores, key=orden_edad) if columna == "grupo_edad_quinquenal" else sorted(valores)

        def indice(lista: list[str], valor: str) -> int:
            return lista.index(valor) if valor in lista else 0

        with st.form("prediccion"):
            st.markdown("**Víctima**")
            c = st.columns(3)
            sexo = c[0].selectbox("Sexo", o := opciones("sexo"), index=indice(o, base["sexo"]))
            edad_v = c[1].selectbox("Grupo de edad", o := opciones("grupo_edad_quinquenal"),
                                    index=indice(o, base["grupo_edad_quinquenal"]))
            condicion = c[2].selectbox("Condición", o := opciones("condicion_victima"),
                                       index=indice(o, base["condicion_victima"]))
            st.markdown("**Vehículo y tipo de siniestro**")
            c = st.columns(4)
            medio_v = c[0].selectbox("Medio de transporte", o := opciones("medio_grupo"), index=indice(o, base["medio_grupo"]))
            servicio = c[1].selectbox("Servicio del vehículo", o := opciones("servicio_vehiculo"),
                                      index=indice(o, base["servicio_vehiculo"]))
            clase = c[2].selectbox("Clase de accidente", o := opciones("clase_accidente"),
                                   index=indice(o, base["clase_accidente"]))
            objeto_v = c[3].selectbox("Objeto de colisión", o := opciones("objeto_grupo"),
                                      index=indice(o, base["objeto_grupo"]))
            st.markdown("**Contexto**")
            c = st.columns(4)
            departamento = c[0].selectbox("Departamento", o := opciones("departamento"),
                                          index=indice(o, base["departamento"]))
            zona_v = c[1].selectbox("Zona", o := opciones("zona"), index=indice(o, base["zona"]))
            dia = c[2].selectbox("Día de la semana", ORDEN_DIAS, index=indice(ORDEN_DIAS, base["dia_semana"]))
            mes = c[3].slider("Mes", 1, 12, int(base["mes"]))
            enviado = st.form_submit_button("Calcular probabilidad", type="primary")

        # Contexto policial: mediana del departamento elegido (dato del año anterior en el municipio)
        contexto = victimas[victimas["departamento"].astype(str) == departamento][
            ["lesionados_policia_anio_ant", "muertes_policia_anio_ant"]].median()
        perfil = {"sexo": sexo, "grupo_edad_quinquenal": edad_v, "condicion_victima": condicion, "medio_grupo": medio_v,
                  "servicio_vehiculo": servicio, "clase_accidente": clase, "objeto_grupo": objeto_v, "zona": zona_v,
                  "dia_semana": dia, "departamento": departamento, "mes": float(mes),
                  "fin_de_semana": float(dia in ("Sábado", "Domingo")),
                  "lesionados_policia_anio_ant": float(contexto.iloc[0]),
                  "muertes_policia_anio_ant": float(contexto.iloc[1])}
        X = pd.DataFrame([perfil])[modulo_modelo.VARIABLES]
        probabilidades = pd.Series(modelo_ajustado.predict_proba(X)[0], index=modelo_ajustado.classes_)[ORDEN_GRAVEDAD]
        p_base = pd.Series(modelo_ajustado.predict_proba(pd.DataFrame([base])[modulo_modelo.VARIABLES])[0],
                           index=modelo_ajustado.classes_)
        prediccion = modelo_ajustado.predict(X)[0]

        izquierda, derecha = st.columns([2, 3])
        with izquierda:
            st.metric("Clase predicha", prediccion)
            st.metric("Probabilidad de muerte", pct(probabilidades["Muerte"] * 100),
                      f"{dec(probabilidades['Muerte'] / p_base['Muerte'], 1)} veces el perfil base",
                      delta_color="inverse")
            st.caption("Perfil base: motociclista hombre de 25-29 años que choca con un automóvil en la cabecera "
                       f"municipal de Antioquia (P(muerte) = {pct(p_base['Muerte'] * 100)}). Contexto policial usado: "
                       f"mediana del departamento ({miles(perfil['lesionados_policia_anio_ant'])} lesionados y "
                       f"{miles(perfil['muertes_policia_anio_ant'])} muertes en el municipio el año anterior).")
        with derecha:
            fig = px.bar(x=probabilidades.values * 100, y=probabilidades.index, orientation="h",
                         color=probabilidades.index, color_discrete_map=COLOR_GRAVEDAD,
                         text=[f"{v * 100:.1f} %" for v in probabilidades.values],
                         labels={"x": "Probabilidad estimada (%)", "y": "Gravedad", "color": "Gravedad"})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, xaxis_range=[0, 110])
            st.plotly_chart(estilo_grafica(fig, "Probabilidad Estimada por Nivel de Gravedad", "Perfil ingresado",
                                           alto=320), width="stretch")

# ===========================================================================
# 5. INSIGHTS Y REPORTE
# ===========================================================================
with pestanas[4]:
    if not documento:
        st.error(f"No se pudo leer el documento de análisis: {origen_documento}")
    else:
        if origen_documento == "MongoDB":
            st.success(f"Insights y recomendaciones leídos desde **MongoDB** (base `{config.MONGO_DB}`, colección "
                       f"`{config.MONGO_COLECCION}`). Análisis del {str(documento['metadata']['fecha_analisis'])[:10]}.")
        else:
            st.warning(f"Se muestra la {origen_documento}")

        st.subheader("Insights")
        temas = sorted({i["tema"] for i in documento["insights"]})
        elegidos = st.multiselect("Filtrar por tema", temas, placeholder="Todos los temas")
        insights = [i for i in documento["insights"] if not elegidos or i["tema"] in elegidos]
        columnas = st.columns(2)
        for n, i in enumerate(insights):
            columnas[n % 2].markdown(
                f"<div class='tarjeta'><div class='tema'>Insight {i['id']} · {i['tema']}</div>"
                f"<div class='titulo'>{i['titulo']}</div><div>{i['descripcion']}</div>"
                f"<div class='pie'>Gráfica de soporte: {i['visualizacion']}</div></div>", unsafe_allow_html=True)

        st.subheader("Recomendaciones")
        prioridades = st.radio("Prioridad", ["Todas", "Alta", "Media"], horizontal=True)
        recomendaciones = pd.DataFrame(documento["recomendaciones"])
        if prioridades != "Todas":
            recomendaciones = recomendaciones[recomendaciones["prioridad"] == prioridades]
        recomendaciones["insights_relacionados"] = recomendaciones["insights_relacionados"].map(
            lambda x: ", ".join(map(str, x)))
        st.dataframe(recomendaciones[["id", "prioridad", "tema", "recomendacion", "sustento", "insights_relacionados"]],
                     width="stretch", hide_index=True,
                     column_config={"recomendacion": st.column_config.TextColumn("Recomendación", width="large"),
                                    "sustento": st.column_config.TextColumn("Sustento", width="medium")})

        st.subheader("Reporte PDF")
        ruta_pdf = config.DIR_REPORTES / "Rojas_Juan_Reporte_Siniestros.pdf"
        if ruta_pdf.exists():
            c1, c2 = st.columns([1, 3])
            c1.download_button("⬇️ Descargar reporte PDF", ruta_pdf.read_bytes(), ruta_pdf.name, "application/pdf",
                               type="primary")
            c2.caption(f"Generado automáticamente desde MongoDB con ReportLab · {ruta_pdf.stat().st_size / 1024:,.0f} "
                       "KB · contenido: portada, dataset, calidad, agregaciones, 11 gráficas, modelo, insights y "
                       "recomendaciones.")
        else:
            st.warning("Aún no existe el PDF. Ejecute la sección 7 del notebook o `python src/reporte_pdf.py`.")
