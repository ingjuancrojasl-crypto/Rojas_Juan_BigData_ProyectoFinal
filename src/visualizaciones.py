"""
Visualizaciones del análisis exploratorio (sección 4).

Principios aplicados (Actividad 7):
  - Tipo de gráfico según la tarea: líneas para el tiempo, barras horizontales ordenadas para comparar
    categorías, barras 100 % apiladas para proporciones ordinales, mapa de calor para la relación entre
    dos variables categóricas. Nunca doble eje: dos magnitudes distintas van en paneles separados.
  - Paleta accesible y validada para daltonismo: azul #2a78d6 y naranja #eb6834 para dos series;
    escala ordinal de un solo tono (azules) para la gravedad; gris para lo que no se resalta.
  - Alto data-ink ratio: sin bordes superior/derecho, cuadrícula fina solo en el eje de valores,
    etiquetas directas solo donde aportan.
  - Títulos que enuncian el hallazgo (calculados con los datos), subtítulo con la métrica y ejes en español.

Cada función devuelve (figura, ficha) y guarda la figura en reports/figuras/. La ficha (título, hallazgo,
archivo) alimenta luego el documento de MongoDB, el reporte PDF y el dashboard.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SI = config.SIN_INFORMACION
DIR_FIGURAS = config.DIR_REPORTES / "figuras"

# ---------------------------------------------------------------------------
# Paleta (validada con el script de accesibilidad: CVD ΔE ≥ 24, contraste ≥ 3:1)
# ---------------------------------------------------------------------------
AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS = "#c3c2b7"              # categorías no resaltadas
TINTA = "#0b0b0b"             # títulos
TINTA_2 = "#52514e"           # subtítulos, etiquetas
TINTA_SUAVE = "#898781"       # ejes, notas
CUADRICULA = "#e1e0d9"
FONDO = "#fcfcfb"
GRAVEDAD = {"Lesión leve": "#86b6ef", "Lesión grave": "#2a78d6", "Muerte": "#0d366b"}   # ordinal, un tono
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azules", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
COLOR_FUENTE = {"Medicina Legal": AZUL, "Policía Nacional": NARANJA}
NOTA_ML = ("Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF), Lesiones y Muertes por eventos "
           "de transporte 2015-2024, datos.gov.co (ezhf-hscf, s65h-7665).")
NOTA_4F = ("Fuentes: INMLCF, Lesiones y Muertes por eventos de transporte (ezhf-hscf, s65h-7665); Policía Nacional, "
           "Lesiones y Homicidios en accidente de tránsito (72sg-cybi, ha6j-pa2r). datos.gov.co, 2015-2024.")


def aplicar_estilo() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.edgecolor": GRIS, "axes.linewidth": 1, "axes.labelcolor": TINTA_2,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titlesize": 11, "axes.titlecolor": TINTA,
        "grid.color": CUADRICULA, "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 110, "savefig.dpi": 150,
    })


def _miles(x, _pos=None) -> str:
    return f"{x:,.0f}".replace(",", ".")


def _dec(x: float, decimales: int = 1) -> str:
    """Número con coma decimal (formato colombiano)."""
    return f"{x:.{decimales}f}".replace(".", ",")


def _pct(x: float) -> str:
    """Proporción (0-1) como porcentaje con espacio: 0.25 -> '25 %'."""
    return f"{x * 100:.0f} %"


def _encabezado(fig, titulo: str, subtitulo: str, nota: str) -> None:
    """Encabezado centrado y pie de fuente.

    - Línea 1 (negrita, centrada): título descriptivo de la gráfica.
    - Línea 2 (centrada): hallazgo principal calculado con los datos (título con insight).
    - Pie (izquierda): fuente de los datos y notas metodológicas.
    Las posiciones se calculan en pulgadas para que el texto nunca se monte sobre el gráfico.
    """
    ancho_pulg, alto_pulg = fig.get_size_inches()
    lineas_titulo = textwrap.wrap(titulo, int(ancho_pulg * 8.6))
    lineas_sub = textwrap.wrap(subtitulo, int(ancho_pulg * 11.5))
    y = 0.12                                                 # pulgadas desde el borde superior
    fig.text(0.5, 1 - y / alto_pulg, "\n".join(lineas_titulo), ha="center", va="top", fontsize=14,
             fontweight="bold", color=TINTA, linespacing=1.25)
    y += 0.27 * len(lineas_titulo) + 0.06
    fig.text(0.5, 1 - y / alto_pulg, "\n".join(lineas_sub), ha="center", va="top", fontsize=10.5, color=TINTA_2,
             linespacing=1.3)
    y += 0.21 * len(lineas_sub) + 0.15
    fig._margen_superior = 1 - y / alto_pulg

    lineas_nota = [linea for parte in nota.split("\n") for linea in textwrap.wrap(parte, int(ancho_pulg * 17))]
    fig.text(0.01, 0.01, "\n".join(lineas_nota), ha="left", va="bottom", fontsize=8, color=TINTA_SUAVE,
             linespacing=1.35)
    fig._margen_inferior = (0.14 * len(lineas_nota) + 0.1) / alto_pulg


def _cerrar(fig, nombre: str, titulo: str, hallazgo: str, pregunta: str, tipo: str) -> dict:
    fig.tight_layout(rect=[0, getattr(fig, "_margen_inferior", 0.045), 1, getattr(fig, "_margen_superior", 0.85)])
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", facecolor=fig.get_facecolor())
    return {"nombre": nombre, "titulo": titulo, "tipo_grafico": tipo, "pregunta": pregunta, "hallazgo": hallazgo,
            "archivo": str(ruta.relative_to(config.RAIZ_PROYECTO)).replace("\\", "/")}


# ---------------------------------------------------------------------------
# Estilo de referencia (marco completo, cuadrícula punteada, título centrado en dos líneas)
# ---------------------------------------------------------------------------
REF_AZUL = "#3a65a9"      # azul acero (validado con #1f9e89: CVD ΔE 17,6; contraste ≥ 3:1)
REF_VERDE = "#1f9e89"     # verde azulado para la segunda serie
REF_AREA = "#dde4ef"      # relleno bajo la serie principal
REF_ROJO = "#c62828"      # anotaciones destacadas


def _estilo_referencia(ax) -> None:
    """Bordes solo en los ejes X e Y (izquierdo e inferior) y cuadrícula punteada visible sobre el área sombreada."""
    ax.set_facecolor("#ffffff")
    for nombre, lado in ax.spines.items():
        lado.set_visible(nombre in ("left", "bottom"))
        lado.set_color("#555555")
        lado.set_linewidth(0.9)
    ax.grid(True, axis="both", linestyle=(0, (3, 3)), linewidth=0.6, color="#c4c4c4")
    ax.set_axisbelow(False)          # la cuadrícula (zorder 2,5) queda sobre el área (1) y debajo de las líneas (3)
    ax.tick_params(axis="both", labelsize=8, colors="#333333", length=3, width=0.8)


def _encabezado_referencia(fig, linea_1: str, linea_2: str) -> None:
    """Título centrado en dos líneas, ambas en negrita (como la gráfica de referencia)."""
    alto = fig.get_size_inches()[1]
    fig.text(0.5, 1 - 0.12 / alto, linea_1, ha="center", va="top", fontsize=12.5, fontweight="bold", color=TINTA)
    fig.text(0.5, 1 - 0.40 / alto, linea_2, ha="center", va="top", fontsize=11, fontweight="bold", color=TINTA)
    fig._margen_superior = 1 - 0.66 / alto


def _con_gravedad(victimas: pd.DataFrame) -> pd.DataFrame:
    return victimas[victimas["gravedad"] != SI]


# ---------------------------------------------------------------------------
# 1. Evolución temporal (cubo temporal)
# ---------------------------------------------------------------------------
def evolucion_anual(cubo_temporal: pd.DataFrame):
    """Estilo de referencia: marco completo, cuadrícula punteada, línea azul acero con puntos blancos,
    área sombreada, periodo destacado con línea punteada y anotaciones en rojo."""
    anual = cubo_temporal.groupby(["anio", "desenlace", "fuente"], observed=True)["victimas"].sum().unstack("fuente")
    les, mue = anual.xs("Lesión", level="desenlace"), anual.xs("Muerte", level="desenlace")
    ml_mue, ml_les = mue["Medicina Legal"], les["Medicina Legal"]
    anio_max = int(ml_mue.idxmax())
    var_mue = ml_mue[anio_max] / ml_mue.iloc[0] - 1
    var_les = ml_les.iloc[-1] / ml_les.iloc[0] - 1
    caida_2020 = ml_les[2020] / ml_les[2019] - 1
    titulo = (f"Las muertes en siniestros viales llegan a su máximo en {anio_max} (+{_pct(var_mue)} frente a 2015), "
              f"mientras las lesiones forenses siguen {_pct(abs(var_les))} por debajo")

    colores = {"Medicina Legal": REF_AZUL, "Policía Nacional": REF_VERDE}
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    fig.patch.set_facecolor("#ffffff")
    paneles = [(ejes[0], les, "Lesionados", "Número de lesionados", 10_000),
               (ejes[1], mue, "Fallecidos", "Número de fallecidos", 1_000)]
    for ax, datos, nombre, etiqueta_y, paso_y in paneles:
        _estilo_referencia(ax)
        tope = datos.values.max() * 1.22
        ax.set_ylim(0, tope)
        ax.set_xlim(datos.index[0] - 0.4, datos.index[-1] + 0.4)
        # Periodo de pandemia: sombreado, líneas punteadas y etiqueta arriba (como la referencia)
        # Gris semitransparente POR ENCIMA del área azul (zorder 1,5): tiñe también la zona sombreada
        ax.axvspan(2019.5, 2020.5, color="#8c8c8c", alpha=0.28, linewidth=0, zorder=1.5)
        for borde in (2019.5, 2020.5):
            ax.axvline(borde, color="#6f6f6f", linestyle=":", linewidth=1.2, zorder=1.6)
        ax.text(2020, tope * 0.975, "Pandemia (2020)", ha="center", va="top", fontsize=7.5, color="#555555", zorder=4)
        # Área sombreada bajo la serie principal (Medicina Legal) y líneas con puntos blancos
        ax.fill_between(datos.index, datos["Medicina Legal"], color=REF_AREA, alpha=0.9, linewidth=0, zorder=1)
        for fuente, color in colores.items():
            ax.plot(datos.index, datos[fuente], color=color, linewidth=2, marker="o", markersize=7,
                    markerfacecolor="#ffffff", markeredgecolor=color, markeredgewidth=1.6, label=fuente, zorder=3)
        ax.set_title(nombre, loc="center", fontsize=10.5, fontweight="bold", color=TINTA, pad=6)
        ax.set_xticks(datos.index)
        ax.set_xlabel("Año", fontsize=9, color="#333333")
        ax.set_ylabel(etiqueta_y, fontsize=9, color="#333333")
        ax.yaxis.set_major_locator(mticker.MultipleLocator(paso_y))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_miles))

    # Anotaciones en rojo: mínimo de lesionados (pandemia) y máximo de fallecidos
    ejes[0].annotate(f"Mínimo: 2020\n{_miles(ml_les[2020])} lesionados", (2020, ml_les[2020]), xytext=(48, -6),
                     textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold", color=REF_ROJO,
                     arrowprops={"arrowstyle": "->", "color": REF_ROJO, "linewidth": 1})
    ejes[1].annotate(f"Máximo: {anio_max}\n{_miles(ml_mue[anio_max])} fallecidos", (anio_max, ml_mue[anio_max]),
                     xytext=(-30, 20), textcoords="offset points", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=REF_ROJO, arrowprops={"arrowstyle": "->", "color": REF_ROJO, "linewidth": 1})
    ejes[0].legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#ffffff", edgecolor="#cccccc",
                   framealpha=1)

    _encabezado_referencia(fig, "Evolución Anual de Víctimas de Siniestros de Transporte",
                           "Lesionados y Fallecidos — Medicina Legal y Policía Nacional, Colombia (2015-2024)")
    fig.text(0.01, 0.01, "Fuentes: Medicina Legal y Policía Nacional (datos.gov.co), 2015-2024.", ha="left",
             va="bottom", fontsize=8, color=TINTA_SUAVE)
    fig._margen_inferior = 0.06
    hallazgo = (f"Las muertes registradas por Medicina Legal pasaron de {_miles(ml_mue.iloc[0])} en 2015 a "
                f"{_miles(ml_mue[anio_max])} en {anio_max}. Las lesiones cayeron {_pct(abs(caida_2020))} en 2020 por la "
                f"pandemia y no recuperaron el nivel previo: en 2024 son {_pct(abs(var_les))} menos que en 2015.")
    return fig, _cerrar(fig, "01_evolucion_anual", titulo, hallazgo, "P1: evolución 2015-2024", "Líneas (paneles)")


# ---------------------------------------------------------------------------
# 2. Distribución por medio de transporte
# ---------------------------------------------------------------------------
def _barras_referencia(ax, valores: pd.Series, conteos: pd.Series, mapa_color: str, titulo_panel: str,
                       etiqueta_x: str, etiqueta_y: str | None) -> None:
    """Barras horizontales ordenadas (mayor arriba) con color secuencial según el valor y valores en negrita."""
    _estilo_referencia(ax)
    ax.grid(False)
    ax.grid(True, axis="x", linestyle=(0, (3, 3)), linewidth=0.6, color="#c4c4c4")
    ax.set_axisbelow(True)
    orden = valores.sort_values()
    escala = plt.get_cmap(mapa_color)
    # Intensidad proporcional al valor dentro del panel (de 0,35 a 0,95 de la escala, para que la más clara se vea)
    intensidad = 0.35 + 0.60 * (orden - orden.min()) / max(orden.max() - orden.min(), 1e-9)
    ax.barh(orden.index, orden.values, color=[escala(i) for i in intensidad], height=0.72, edgecolor="#ffffff",
            linewidth=0.8, zorder=3)
    for posicion, (categoria, valor) in enumerate(orden.items()):
        ax.text(valor + orden.max() * 0.012, posicion, f"{valor:.0f} %  ({_miles(conteos[categoria])})",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color="#222222", zorder=4)
    ax.set_xlim(0, orden.max() * 1.32)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0, symbol=" %"))
    ax.set_title(titulo_panel, loc="center", fontsize=10.5, fontweight="bold", color=TINTA, pad=6)
    ax.set_xlabel(etiqueta_x, fontsize=9, color="#333333")
    if etiqueta_y:
        ax.set_ylabel(etiqueta_y, fontsize=9, color="#333333")
    ax.tick_params(axis="y", labelsize=8.5, length=0)


def distribucion_medio(victimas: pd.DataFrame):
    """Barras divergentes desde cero: lesionados a la derecha (azul) y fallecidos a la izquierda (rojo),
    con intensidad del color según el valor (estilo de la encuesta de satisfacción de referencia)."""
    datos = victimas[~victimas["medio_grupo"].isin([SI, "Otros medios"])]
    conteos = pd.crosstab(datos["medio_grupo"], datos["desenlace"])
    tabla = conteos.div(conteos.sum()).mul(100).sort_values("Lesión")      # la mayor queda arriba
    moto = tabla.loc["Motocicleta"]
    titulo = (f"Las víctimas en motocicleta son {moto['Lesión']:.0f} % de los lesionados y "
              f"{moto['Muerte']:.0f} % de los fallecidos")

    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for nombre, lado in ax.spines.items():
        lado.set_visible(nombre == "bottom")
        lado.set_color("#222222")
    ax.grid(True, axis="y", linestyle=(0, (4, 4)), linewidth=0.6, color="#dcdcdc")
    ax.set_axisbelow(True)

    y = np.arange(len(tabla))
    maximo = tabla.values.max()
    for columna, signo, mapa in [("Lesión", 1, "Blues"), ("Muerte", -1, "Reds")]:
        valores = tabla[columna]
        escala = plt.get_cmap(mapa)
        intensidad = 0.45 + 0.50 * (valores - valores.min()) / max(valores.max() - valores.min(), 1e-9)
        ax.barh(y, signo * valores, height=0.62, color=[escala(i) for i in intensidad], edgecolor="#111111",
                linewidth=1, zorder=3)
        for posicion, (categoria, valor) in zip(y, valores.items()):
            texto = f"{valor:.0f} %\n({_miles(conteos.loc[categoria, columna])})"
            ax.text(signo * (valor + maximo * 0.025), posicion, texto, va="center", ha="left" if signo > 0 else "right",
                    fontsize=10, fontweight="bold", color="#111111", zorder=4, linespacing=1.15)
    ax.axvline(0, color="#9a9a9a", linewidth=1, zorder=4)

    limite = np.ceil(maximo * 1.3 / 10) * 10
    ax.set_xlim(-limite, limite)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{abs(v):.0f} %"))
    ax.set_yticks(y, [textwrap.fill(c, 14) for c in tabla.index])
    ax.tick_params(axis="y", labelsize=10.5, length=4, colors="#111111")
    ax.tick_params(axis="x", labelsize=10, colors="#111111")
    ax.set_xlabel("Porcentaje de las víctimas (← fallecidos  |  lesionados →)", fontsize=10.5, color="#111111")
    ax.set_ylabel("Medio de transporte", fontsize=10.5, color="#111111")
    ax.text(-limite * 0.5, len(tabla) - 0.45, "Fallecidos", ha="center", fontsize=11, fontweight="bold",
            color=plt.get_cmap("Reds")(0.85))
    ax.text(limite * 0.5, len(tabla) - 0.45, "Lesionados", ha="center", fontsize=11, fontweight="bold",
            color=plt.get_cmap("Blues")(0.85))
    ax.set_ylim(-0.6, len(tabla) - 0.1)

    alto = fig.get_size_inches()[1]
    fig.text(0.5, 1 - 0.1 / alto, "Distribución de las Víctimas por Medio de Transporte", ha="center", va="top",
             fontsize=17, fontweight="bold", color="#111111")
    fig._margen_superior = 1 - 0.55 / alto
    fig.text(0.01, 0.01, "Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (datos.gov.co), 2015-2024 · "
             "Peatón = víctima sin vehículo · Entre paréntesis: número de víctimas", ha="left", va="bottom",
             fontsize=8.5, color="#9a9a9a")
    fig._margen_inferior = 0.05
    peaton = tabla.loc["Peatón (sin vehículo)"]
    hallazgo = (f"La motocicleta domina: {moto['Lesión']:.0f} % de los lesionados y {moto['Muerte']:.0f} % de los "
                f"fallecidos. Los peatones pesan más entre los fallecidos ({peaton['Muerte']:.0f} %) que entre los "
                f"lesionados ({peaton['Lesión']:.0f} %).")
    return fig, _cerrar(fig, "02_distribucion_medio", titulo, hallazgo, "P2/P3: perfil de la víctima y vehículo",
                        "Gráfico de mariposa (barras enfrentadas) con escala secuencial")


# ---------------------------------------------------------------------------
# 3. Distribución por edad (lesionados vs fallecidos)
# ---------------------------------------------------------------------------
ORDEN_EDAD = ["(00 a 04)", "(05 a 09)", "(10 a 14)", "(15 a 17)", "(18 a 19)", "(20 a 24)", "(25 a 29)",
              "(30 a 34)", "(35 a 39)", "(40 a 44)", "(45 a 49)", "(50 a 54)", "(55 a 59)", "(60 a 64)",
              "(65 a 69)", "(70 a 74)", "(75 a 79)", "(80 y más)"]


def distribucion_edad(victimas: pd.DataFrame):
    """Barras verticales agrupadas (azul = lesionados, rojo = fallecidos) con el valor sobre cada barra,
    leyenda con recuadro y título centrado en dos líneas (estilo de la gráfica de referencia por año y sexo)."""
    tabla = (pd.crosstab(victimas["grupo_edad_quinquenal"], victimas["desenlace"], normalize="columns").mul(100)
             .reindex(ORDEN_EDAD))
    jovenes = tabla.loc[["(20 a 24)", "(25 a 29)", "(30 a 34)"], "Lesión"].sum()
    mayores = tabla.loc[ORDEN_EDAD[13:]].sum()
    titulo = (f"Los adultos de 20 a 34 años son el {jovenes:.0f} % de los lesionados, pero los mayores de 60 pesan "
              f"{mayores['Muerte'] / mayores['Lesión']:.0f} veces más entre los fallecidos que entre los lesionados")

    fig, ax = plt.subplots(figsize=(13, 5.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for nombre, lado in ax.spines.items():
        lado.set_visible(nombre in ("left", "bottom"))
        lado.set_color("#333333")
    ax.grid(True, axis="y", linestyle=(0, (4, 4)), linewidth=0.6, color="#d9d9d9")
    ax.set_axisbelow(True)

    x = np.arange(len(tabla))
    ancho = 0.42
    for columna, nombre, color, desplazamiento in [("Lesión", "Lesionados", REF_AZUL, -ancho / 2),
                                                    ("Muerte", "Fallecidos", REF_ROJO, ancho / 2)]:
        barras = ax.bar(x + desplazamiento, tabla[columna], width=ancho, color=color, edgecolor="#ffffff",
                        linewidth=0.6, label=nombre, zorder=3)
        for barra, valor in zip(barras, tabla[columna]):
            ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.15, _dec(valor) + " %", ha="center",
                    va="bottom", rotation=90, fontsize=7.5, fontweight="bold", color="#222222", zorder=4)

    ax.set_xticks(x, [e.strip("()").replace(" a ", "-") for e in tabla.index], fontsize=8.5)
    ax.set_xlim(-0.6, len(tabla) - 0.4)
    ax.set_ylim(0, tabla.values.max() * 1.18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0, symbol=" %"))
    ax.tick_params(axis="both", colors="#222222", labelsize=8.5)
    ax.set_xlabel("Grupo de edad (años)", fontsize=10, color="#222222")
    ax.set_ylabel("Porcentaje de las víctimas", fontsize=10, color="#222222")
    ax.legend(title="Desenlace", loc="upper left", fontsize=9, title_fontsize=9.5, frameon=True,
              facecolor="#ffffff", edgecolor="#333333", framealpha=1, borderpad=0.8)

    _encabezado_referencia(fig, "Víctimas por Grupo de Edad y Desenlace",
                           "Lesionados y Fallecidos — Medicina Legal, Colombia (2015-2024)")
    fig.text(0.01, 0.01, "Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (datos.gov.co), 2015-2024.",
             ha="left", va="bottom", fontsize=8, color=TINTA_SUAVE)
    fig._margen_inferior = 0.05
    hallazgo = (f"El grupo de 20 a 34 años concentra el {jovenes:.0f} % de los lesionados. Los mayores de 60 son "
                f"{mayores['Lesión']:.0f} % de los lesionados pero {mayores['Muerte']:.0f} % de los fallecidos.")
    return fig, _cerrar(fig, "03_distribucion_edad", titulo, hallazgo, "P2: perfil de la víctima",
                        "Barras verticales agrupadas")


# ---------------------------------------------------------------------------
# 4. Relación edad × gravedad (barras 100 % apiladas, ordinal)
# ---------------------------------------------------------------------------
GRAVEDAD_CUALITATIVA = {"Lesión leve": "#4ea1e8", "Lesión grave": "#1f4e96", "Muerte": "#c62828"}
ETIQUETAS_GRAVEDAD = {"Lesión leve": "Lesión leve (0-30 días)", "Lesión grave": "Lesión grave (más de 30 días)",
                      "Muerte": "Fallecidos"}


def gravedad_por_edad(victimas: pd.DataFrame):
    """Barras 100 % apiladas con colores cualitativos (azul claro, azul oscuro y rojo), eje Y izquierdo con las
    edades y eje Y derecho con el porcentaje de fallecidos de cada grupo."""
    datos = _con_gravedad(victimas)
    tabla = (pd.crosstab(datos["grupo_edad_quinquenal"], datos["gravedad"], normalize="index").mul(100)
             .reindex(ORDEN_EDAD)[list(GRAVEDAD_CUALITATIVA)])
    m80, m2534 = tabla.loc["(80 y más)", "Muerte"], tabla.loc[["(25 a 29)", "(30 a 34)"], "Muerte"].mean()
    titulo = (f"El riesgo de morir crece con la edad: a los 80 años o más fallece el {m80:.0f} % de las víctimas, "
              f"{m80 / m2534:.0f} veces más que entre los 25 y 34 años")

    fig, ax = plt.subplots(figsize=(13, 7.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for nombre, lado in ax.spines.items():
        lado.set_visible(nombre in ("left", "bottom", "right"))
        lado.set_color("#333333")

    y = np.arange(len(tabla))
    izquierda = np.zeros(len(tabla))
    for nivel, color in GRAVEDAD_CUALITATIVA.items():
        ax.barh(y, tabla[nivel], left=izquierda, height=0.66, color=color, edgecolor="#ffffff", linewidth=1.2,
                label=ETIQUETAS_GRAVEDAD[nivel], zorder=3)
        izquierda += tabla[nivel].values

    etiquetas_edad = [e.strip("()") + " años" for e in tabla.index]
    ax.set_yticks(y, etiquetas_edad)
    ax.set_ylim(len(tabla) - 0.5, -0.5)                    # el grupo más joven arriba
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0, symbol=" %"))
    # Cuadrícula del eje X (sin cuadrícula en Y): líneas verticales punteadas cada 10 %, por encima de las barras
    ax.grid(False)
    for posicion in range(10, 100, 10):
        ax.axvline(posicion, color="#9e9e9e", linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.8, zorder=4)
    ax.tick_params(axis="both", labelsize=8.5, colors="#222222")
    ax.set_xlabel("Porcentaje de las víctimas del grupo de edad", fontsize=10, color="#222222")
    ax.set_ylabel("Edades de las víctimas", fontsize=10, color="#222222")

    # Segundo eje Y (derecha): porcentaje de fallecidos por grupo, en las mismas posiciones
    derecho = ax.twinx()
    derecho.set_ylim(ax.get_ylim())
    derecho.set_yticks(y, [f"{v:.0f} %" for v in tabla["Muerte"]])
    derecho.tick_params(axis="y", labelsize=8.5, colors="#111111", length=3)
    for etiqueta in derecho.get_yticklabels():
        etiqueta.set_fontweight("bold")
    derecho.set_ylabel("Porcentaje de fallecidos", fontsize=10, color="#111111", rotation=270, labelpad=16)
    for nombre, lado in derecho.spines.items():
        lado.set_visible(nombre == "right")
        lado.set_color("#333333")

    ax.legend(title="Gravedad", loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=9,
              title_fontsize=9.5, frameon=True, facecolor="#ffffff", edgecolor="#333333", framealpha=1)

    _encabezado_referencia(fig, "Gravedad del Siniestro según la Edad de la Víctima",
                           "Lesión Leve, Lesión Grave y Fallecidos — Medicina Legal, Colombia (2015-2024)")
    fig._margen_superior = 1 - 0.62 / fig.get_size_inches()[1]          # deja la leyenda justo bajo el título
    fig.text(0.01, 0.01, "Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (datos.gov.co), 2015-2024.",
             ha="left", va="bottom", fontsize=8, color=TINTA_SUAVE)
    fig._margen_inferior = 0.04
    hallazgo = (f"La proporción de muertes es estable (~16 %) entre los 20 y 49 años y se dispara desde los 60: "
                f"{tabla.loc['(60 a 64)', 'Muerte']:.0f} % a los 60-64 y {m80:.0f} % a los 80 o más. "
                f"En niños de 5 a 9 años es solo {tabla.loc['(05 a 09)', 'Muerte']:.0f} %.")
    return fig, _cerrar(fig, "04_gravedad_por_edad", titulo, hallazgo, "P2: relación edad-gravedad",
                        "Barras 100 % apiladas (colores cualitativos)")


# ---------------------------------------------------------------------------
# 5. Comparación por categoría: letalidad según objeto de colisión
# ---------------------------------------------------------------------------
def letalidad_objeto(victimas: pd.DataFrame):
    """Barras horizontales ordenadas (mayor arriba) en rojo secuencial según el valor, cuadrícula en el eje X
    y valores en negrita (estilo de la gráfica de referencia 'Top 10 ciudades')."""
    datos = _con_gravedad(victimas)
    datos = datos[datos["objeto_grupo"] != SI]
    tabla = (datos.groupby("objeto_grupo", observed=True)["gravedad"]
             .agg(muertes=lambda s: (s == "Muerte").mean() * 100, victimas="size"))
    tabla = tabla[(tabla["victimas"] >= 1000) & (tabla.index != "Otros")].sort_values("muertes")
    promedio = (datos["gravedad"] == "Muerte").mean() * 100
    top = tabla.index[-1]
    titulo = (f"Chocar contra un objeto fijo es el evento más letal: muere el {tabla.loc[top, 'muertes']:.0f} % de las "
              f"víctimas, {_dec(tabla.loc[top, 'muertes'] / promedio)} veces el promedio")

    fig, ax = plt.subplots(figsize=(12, 6.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for nombre, lado in ax.spines.items():
        lado.set_visible(nombre in ("left", "bottom"))
        lado.set_color("#333333")
    ax.grid(True, axis="x", linestyle=(0, (4, 4)), linewidth=0.6, color="#d0d0d0")
    ax.set_axisbelow(True)

    escala = plt.get_cmap("Reds")
    valores = tabla["muertes"]
    intensidad = 0.30 + 0.65 * (valores - valores.min()) / max(valores.max() - valores.min(), 1e-9)
    ax.barh(tabla.index, valores, height=0.78, color=[escala(i) for i in intensidad], edgecolor="#ffffff",
            linewidth=0.8, zorder=3)
    for posicion, (valor, n) in enumerate(zip(valores, tabla["victimas"])):
        ax.text(valor + valores.max() * 0.012, posicion, f"{valor:.0f} %  ({_miles(n)})", va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#222222", zorder=4)

    ax.set_xlim(0, valores.max() * 1.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0, symbol=" %"))
    ax.tick_params(axis="both", labelsize=8.5, colors="#222222")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Porcentaje de víctimas que fallecen", fontsize=9.5, color="#222222")
    ax.set_ylabel("Objeto de colisión", fontsize=9.5, color="#222222")

    _encabezado_referencia(fig, "Letalidad de los Siniestros según el Objeto de Colisión",
                           "Porcentaje de Víctimas Fallecidas — Medicina Legal, Colombia (2015-2024)")
    fig.text(0.01, 0.01, "Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (datos.gov.co), 2015-2024. "
             "Entre paréntesis: número de víctimas. 'Sin colisión' agrupa volcamientos y caídas.", ha="left",
             va="bottom", fontsize=8, color=TINTA_SUAVE)
    fig._margen_inferior = 0.05
    carga = tabla.loc["Vehículo de carga pesada"]
    hallazgo = (f"Objeto fijo ({tabla.loc[top, 'muertes']:.0f} %) y sin colisión, es decir volcamientos y caídas "
                f"({tabla.loc['Sin colisión', 'muertes']:.0f} %), son los más letales; el promedio es "
                f"{promedio:.0f} %. Chocar con un vehículo de carga pesada mata al {carga['muertes']:.0f} % de las "
                f"víctimas, frente al {tabla.loc['Automóvil o camioneta', 'muertes']:.0f} % con un automóvil.")
    return fig, _cerrar(fig, "05_letalidad_objeto", titulo, hallazgo, "P3: vehículos y objetos más letales",
                        "Barras horizontales ordenadas con escala secuencial")


# ---------------------------------------------------------------------------
# 6. Causas en lesionados y fallecidos
# ---------------------------------------------------------------------------
def causas_lesionados_fallecidos(victimas: pd.DataFrame):
    """Barras horizontales agrupadas (azul = lesionados, rojo = fallecidos), cuadrícula en el eje X, valores en
    negrita y recuadro de convenciones (estilo de las gráficas de referencia)."""
    datos = victimas[~victimas["causa_categoria"].isin([SI, "Otras / no especificadas"])]
    tabla = pd.crosstab(datos["causa_categoria"], datos["desenlace"], normalize="columns").mul(100).sort_values("Lesión")
    conteos = pd.crosstab(datos["causa_categoria"], datos["desenlace"])
    top_muerte = tabla["Muerte"].nlargest(2)
    titulo = (f"{top_muerte.index[0]} y {top_muerte.index[1].lower()} explican el {top_muerte.sum():.0f} % de las "
              f"muertes con causa conocida; en lesionados pesan también las maniobras peligrosas")
    con_causa_muerte = (victimas.loc[victimas["desenlace"] == "Muerte", "causa_categoria"] != SI).mean()

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for nombre, lado in ax.spines.items():
        lado.set_visible(nombre in ("left", "bottom"))
        lado.set_color("#333333")
    ax.grid(True, axis="x", linestyle=(0, (4, 4)), linewidth=0.6, color="#d0d0d0")
    ax.set_axisbelow(True)

    y = np.arange(len(tabla))
    alto = 0.4
    maximo = tabla.values.max()
    for columna, nombre, color, posiciones in [("Lesión", "Lesionados", REF_AZUL, y + alto / 2),
                                                ("Muerte", "Fallecidos", REF_ROJO, y - alto / 2)]:
        ax.barh(posiciones, tabla[columna], height=alto, color=color, edgecolor="#ffffff", linewidth=0.6,
                label=nombre, zorder=3)
        for posicion, (categoria, valor) in zip(posiciones, tabla[columna].items()):
            ax.text(valor + maximo * 0.01, posicion, f"{valor:.0f} %  ({_miles(conteos.loc[categoria, columna])})",
                    va="center", ha="left", fontsize=8, fontweight="bold", color="#222222", zorder=4)

    ax.set_yticks(y, tabla.index)
    ax.set_ylim(-0.6, len(tabla) - 0.4)
    ax.set_xlim(0, maximo * 1.25)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0, symbol=" %"))
    ax.tick_params(axis="both", labelsize=8.5, colors="#222222")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Porcentaje de las víctimas con causa registrada", fontsize=9.5, color="#222222")
    ax.set_ylabel("Categoría de causa", fontsize=9.5, color="#222222")
    ax.legend(title="Desenlace", loc="lower right", fontsize=9, title_fontsize=9.5, frameon=True,
              facecolor="#ffffff", edgecolor="#333333", framealpha=1, borderpad=0.8)

    _encabezado_referencia(fig, "Causas de los Siniestros de Transporte en Lesionados y Fallecidos",
                           "Víctimas con Causa Registrada — Medicina Legal, Colombia (2015-2024)")
    fig.text(0.01, 0.01, "Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (datos.gov.co), 2015-2024. "
             f"Entre paréntesis: número de víctimas. Se excluyen 'Sin información' y 'Otras'; solo el "
             f"{_pct(con_causa_muerte)} de las muertes tiene causa registrada.", ha="left", va="bottom",
             fontsize=8, color=TINTA_SUAVE)
    fig._margen_inferior = 0.05
    hallazgo = (f"Entre los fallecidos con causa registrada, {top_muerte.index[0].lower()} "
                f"({top_muerte.iloc[0]:.0f} %) y {top_muerte.index[1].lower()} ({top_muerte.iloc[1]:.0f} %) dominan. "
                f"Entre los lesionados predominan las maniobras peligrosas ({tabla.loc['Maniobras peligrosas', 'Lesión']:.0f} %)"
                f" y la imprudencia del peatón ({tabla.loc['Imprudencia del peatón', 'Lesión']:.0f} %). Las causas de "
                f"muerte se registran con un catálogo más corto, por lo que la comparación es indicativa.")
    return fig, _cerrar(fig, "06_causas", titulo, hallazgo, "P4: causas más frecuentes", "Barras horizontales agrupadas")


# ---------------------------------------------------------------------------
# 7. Relación región × zona (mapa de calor)
# ---------------------------------------------------------------------------
def letalidad_region_zona(victimas: pd.DataFrame):
    """Mapa de calor con escala viridis, celdas separadas por líneas blancas y valores en negrita
    (estilo de la gráfica de referencia de precio promedio por categoría)."""
    datos = _con_gravedad(victimas)
    zonas = {"Cabecera municipal": "Cabecera municipal", "Centro poblado": "Centro poblado",
             "Parte rural (vereda y campo)": "Rural (vereda y campo)"}
    datos = datos[datos["zona"].isin(zonas) & (datos["region"] != SI) & (datos["region"] != "Insular")]
    datos = datos.assign(zona_corta=datos["zona"].astype(str).map(zonas))
    tasa = pd.crosstab(datos["region"], datos["zona_corta"], values=datos["gravedad"] == "Muerte",
                       aggfunc="mean").mul(100)[list(zonas.values())]
    n = pd.crosstab(datos["region"], datos["zona_corta"])[list(zonas.values())]
    tasa = tasa.loc[tasa["Rural (vereda y campo)"].sort_values(ascending=False).index]
    n = n.loc[tasa.index]
    razon = (tasa["Rural (vereda y campo)"] / tasa["Cabecera municipal"])
    titulo = (f"Un siniestro en zona rural es entre {_dec(razon.min())} y {_dec(razon.max())} veces más letal que en la "
              "cabecera municipal, en todas las regiones")

    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.patch.set_facecolor("#ffffff")
    mapa = plt.get_cmap("viridis")
    vmin, vmax = tasa.values.min(), tasa.values.max()
    imagen = ax.imshow(tasa.values, cmap=mapa, aspect="auto", vmin=vmin, vmax=vmax)
    # Líneas blancas entre celdas
    ax.set_xticks(np.arange(-0.5, tasa.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, tasa.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=3.5)
    ax.tick_params(which="minor", length=0)
    for i in range(tasa.shape[0]):
        for j in range(tasa.shape[1]):
            valor = tasa.iat[i, j]
            claro = (valor - vmin) / (vmax - vmin) > 0.6          # en celdas verdes/amarillas el texto va en negro
            color = "#1a1a1a" if claro else "#ffffff"
            ax.text(j, i - 0.08, f"{valor:.0f} %", ha="center", va="center", fontsize=15, fontweight="bold",
                    color=color)
            ax.text(j, i + 0.22, f"({_miles(n.iat[i, j])} víctimas)", ha="center", va="center", fontsize=8.5,
                    color=color)
    ax.set_xticks(range(tasa.shape[1]), tasa.columns, fontsize=11)
    ax.set_yticks(range(tasa.shape[0]), tasa.index, fontsize=10.5)
    ax.tick_params(which="major", length=4, colors="#333333")
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xlabel("Zona del hecho", fontsize=11, color="#333333")
    ax.set_ylabel("Región natural", fontsize=11, color="#333333")
    barra = fig.colorbar(imagen, ax=ax, fraction=0.04, pad=0.04, shrink=0.75,
                         format=mticker.PercentFormatter(decimals=0, symbol=" %"))
    barra.set_label("Porcentaje de víctimas que fallecen", fontsize=10.5, color="#333333")
    barra.ax.tick_params(labelsize=9, colors="#333333")
    barra.outline.set_visible(False)

    _encabezado_referencia(fig, "Letalidad de los Siniestros por Región Natural y Zona del Hecho",
                           "Porcentaje de Víctimas Fallecidas — Medicina Legal, Colombia (2015-2024)")
    fig.text(0.01, 0.01, "Fuente: Instituto Nacional de Medicina Legal y Ciencias Forenses (datos.gov.co), 2015-2024. "
             "Se excluye la región Insular por tener menos de 600 víctimas.", ha="left", va="bottom", fontsize=8,
             color=TINTA_SUAVE)
    fig._margen_inferior = 0.04
    hallazgo = (f"En la cabecera municipal muere entre el {tasa['Cabecera municipal'].min():.0f} % y el "
                f"{tasa['Cabecera municipal'].max():.0f} % de las víctimas; en zona rural, entre el "
                f"{tasa['Rural (vereda y campo)'].min():.0f} % y el {tasa['Rural (vereda y campo)'].max():.0f} %. "
                f"La región más crítica en zona rural es {tasa.index[0]} "
                f"({tasa['Rural (vereda y campo)'].iloc[0]:.0f} %).")
    return fig, _cerrar(fig, "07_letalidad_region_zona", titulo, hallazgo, "P5: dónde son más graves",
                        "Mapa de calor (escala viridis)")


# ---------------------------------------------------------------------------
# 8. Cobertura entre fuentes (cubo geográfico)
# ---------------------------------------------------------------------------
def cobertura_fuentes(cubo_geografico: pd.DataFrame):
    """Estilo de la gráfica 1: bordes solo en X e Y, cuadrícula punteada sobre el área, puntos blancos,
    franja de pandemia, línea de referencia en 1 y anotación del máximo."""
    anual = cubo_geografico.groupby(["anio", "desenlace", "fuente"], observed=True)["victimas"].sum().unstack("fuente")
    razon = (anual["Policía Nacional"] / anual["Medicina Legal"]).unstack("desenlace")
    titulo = (f"La Policía registra hasta {_dec(razon['Lesión'].max())} veces más lesionados que Medicina Legal, pero "
              f"siempre menos muertes: ninguna fuente sola dimensiona el problema")

    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.patch.set_facecolor("#ffffff")
    _estilo_referencia(ax)
    tope = 2.25
    ax.set_ylim(0, tope)
    ax.set_xlim(razon.index[0] - 0.4, razon.index[-1] + 0.4)

    ax.axvspan(2019.5, 2020.5, color="#8c8c8c", alpha=0.28, linewidth=0, zorder=1.5)
    for borde in (2019.5, 2020.5):
        ax.axvline(borde, color="#6f6f6f", linestyle=":", linewidth=1.2, zorder=1.6)
    ax.text(2020, tope * 0.975, "Pandemia (2020)", ha="center", va="top", fontsize=8, color="#555555", zorder=4)

    ax.fill_between(razon.index, razon["Lesión"], color=REF_AREA, alpha=0.9, linewidth=0, zorder=1)
    ax.axhline(1, color="#444444", linewidth=1.2, zorder=2.6)
    ax.text(2021.1, 1.03, "Ambas fuentes registran lo mismo (razón = 1)", ha="left", va="bottom",
            fontsize=8, color="#444444", zorder=4)
    for desenlace, nombre, color in [("Lesión", "Lesionados", REF_AZUL), ("Muerte", "Fallecidos", REF_ROJO)]:
        ax.plot(razon.index, razon[desenlace], color=color, linewidth=2, marker="o", markersize=7,
                markerfacecolor="#ffffff", markeredgecolor=color, markeredgewidth=1.6, label=nombre, zorder=3)

    anio_max = int(razon["Lesión"].idxmax())
    ax.annotate(f"Máximo: {anio_max}\n{_dec(razon['Lesión'][anio_max], 2)} veces", (anio_max, razon["Lesión"][anio_max]),
                xytext=(48, 6), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold",
                color="#222222", arrowprops={"arrowstyle": "->", "color": "#222222", "linewidth": 1}, zorder=5)

    ax.set_xticks(razon.index)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: _dec(v, 2)))
    ax.set_xlabel("Año", fontsize=9, color="#333333")
    ax.set_ylabel("Víctimas Policía / víctimas Medicina Legal", fontsize=9, color="#333333")
    ax.legend(title="Desenlace", loc="upper right", fontsize=8.5, title_fontsize=9, frameon=True,
              facecolor="#ffffff", edgecolor="#333333", framealpha=1)

    _encabezado_referencia(fig, "Cobertura entre Fuentes: Policía Nacional frente a Medicina Legal",
                           "Razón de Víctimas Registradas — Lesionados y Fallecidos, Colombia (2015-2024)")
    fig.text(0.01, 0.01, "Fuentes: Medicina Legal y Policía Nacional (datos.gov.co), 2015-2024.", ha="left",
             va="bottom", fontsize=8, color=TINTA_SUAVE)
    fig._margen_inferior = 0.05
    hallazgo = (f"En muertes la razón está entre {_dec(razon['Muerte'].min(), 2)} y {_dec(razon['Muerte'].max(), 2)}: no toda "
                f"muerte llega a denuncia. En lesiones va de {_dec(razon['Lesión'].min(), 2)} a {_dec(razon['Lesión'].max(), 2)}: "
                "muchos lesionados denuncian sin pasar por examen forense.")
    return fig, _cerrar(fig, "08_cobertura_fuentes", titulo, hallazgo, "P6: cobertura entre fuentes",
                        "Líneas con línea de referencia")
