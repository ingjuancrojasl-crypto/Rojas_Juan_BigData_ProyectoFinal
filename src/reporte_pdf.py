"""Sección 7: reporte PDF automático con ReportLab.

El reporte se arma solo con el documento de análisis guardado en MongoDB (sección 6): no se escribe ningún
texto de resultados a mano. Si se vuelve a ejecutar el pipeline con otros datos, el PDF cambia solo.

Uso:
    python src/reporte_pdf.py          # lee MongoDB y escribe reports/Rojas_Juan_Reporte_Siniestros.pdf
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ARCHIVO_REPORTE = config.DIR_REPORTES / "Rojas_Juan_Reporte_Siniestros.pdf"

# Paleta igual a la de las gráficas
AZUL = colors.HexColor("#3a65a9")
AZUL_OSCURO = colors.HexColor("#1f3c6e")
AZUL_CLARO = colors.HexColor("#e8eef8")
ROJO = colors.HexColor("#c62828")
ROJO_CLARO = colors.HexColor("#fbe9e7")
VERDE = colors.HexColor("#1f9e89")
GRIS = colors.HexColor("#5f6368")
GRIS_CLARO = colors.HexColor("#f3f4f6")
LINEA = colors.HexColor("#d0d4da")

ANCHO_PAGINA, ALTO_PAGINA = letter
MARGEN = 2 * cm
ANCHO_UTIL = ANCHO_PAGINA - 2 * MARGEN


# ---------------------------------------------------------------------------
# Fuentes: DejaVu Sans (viene con matplotlib) para tildes y símbolos como ≥ y ×
# ---------------------------------------------------------------------------
def _registrar_fuentes() -> tuple[str, str]:
    try:
        import matplotlib
        carpeta = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVu", str(carpeta / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(carpeta / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:  # sin matplotlib: fuentes estándar de PDF
        return "Helvetica", "Helvetica-Bold"


FUENTE, FUENTE_NEGRITA = _registrar_fuentes()


def _estilos() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    e = lambda nombre, **kw: ParagraphStyle(nombre, parent=base, **({"fontName": FUENTE} | kw))  # noqa: E731
    return {
        "titulo_portada": e("tp", fontName=FUENTE_NEGRITA, fontSize=24, leading=30, textColor=colors.white),
        "subtitulo_portada": e("sp", fontSize=13, leading=18, textColor=colors.white),
        "h1": e("h1", fontName=FUENTE_NEGRITA, fontSize=16, leading=20, textColor=AZUL_OSCURO, spaceBefore=4,
                spaceAfter=8),
        "h2": e("h2", fontName=FUENTE_NEGRITA, fontSize=12, leading=15, textColor=AZUL, spaceBefore=10, spaceAfter=5,
                keepWithNext=1),
        "cuerpo": e("cuerpo", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=6),
        "nota": e("nota", fontSize=8, leading=10.5, textColor=GRIS, spaceAfter=4),
        "celda": e("celda", fontSize=7.8, leading=9.6),
        "celda_negrita": e("celda_n", fontName=FUENTE_NEGRITA, fontSize=7.8, leading=9.6),
        "encabezado_tabla": e("et", fontName=FUENTE_NEGRITA, fontSize=7.8, leading=9.6, textColor=colors.white),
        "titulo_figura": e("tf", fontName=FUENTE_NEGRITA, fontSize=9, leading=11.5, textColor=AZUL_OSCURO),
        "pie_figura": e("pf", fontSize=8.3, leading=11, textColor=colors.HexColor("#333333"), alignment=TA_JUSTIFY),
        "kpi_valor": e("kv", fontName=FUENTE_NEGRITA, fontSize=17, leading=20, alignment=TA_CENTER, textColor=AZUL_OSCURO),
        "kpi_etiqueta": e("ke", fontSize=7.8, leading=9.5, alignment=TA_CENTER, textColor=GRIS),
    }


EST = _estilos()


# ---------------------------------------------------------------------------
# Formato de números (estilo colombiano: punto de miles, coma decimal)
# ---------------------------------------------------------------------------
def miles(valor) -> str:
    try:
        return f"{int(round(float(valor))):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(valor)


def dec(valor, decimales: int = 3) -> str:
    try:
        return f"{float(valor):.{decimales}f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(valor)


def pct(valor, decimales: int = 1) -> str:
    return f"{dec(valor, decimales)} %"


def _p(texto, estilo: str = "celda") -> Paragraph:
    texto = str(texto).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(texto, EST[estilo])


# ---------------------------------------------------------------------------
# Bloques reutilizables
# ---------------------------------------------------------------------------
def tabla(encabezados: list[str], filas: list[list], anchos: list[float] | None = None,
          numericas: tuple[int, ...] = (), resaltar: dict[int, colors.Color] | None = None) -> Table:
    """Tabla con encabezado azul, filas alternas y columnas numéricas alineadas a la derecha."""
    datos = [[_p(h, "encabezado_tabla") for h in encabezados]]
    for fila in filas:
        datos.append([c if isinstance(c, Paragraph) else _p(c) for c in fila])
    anchos = anchos or [ANCHO_UTIL / len(encabezados)] * len(encabezados)
    t = Table(datos, colWidths=anchos, repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINEA),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(datos)):
        if i % 2 == 0:
            estilo.append(("BACKGROUND", (0, i), (-1, i), GRIS_CLARO))
    for fila, color in (resaltar or {}).items():
        estilo.append(("BACKGROUND", (0, fila), (-1, fila), color))
    t.setStyle(TableStyle(estilo))
    for col in numericas:  # alinear a la derecha los números
        for fila in datos[1:]:
            fila[col].style = ParagraphStyle("num", parent=fila[col].style, alignment=2)
        datos[0][col].style = ParagraphStyle("numh", parent=EST["encabezado_tabla"], alignment=2)
    return t


def figura(ruta: Path, titulo: str, pie: str, ancho: float = ANCHO_UTIL, alto_max: float = 10.5 * cm) -> KeepTogether:
    """Imagen escalada sin deformar, con título arriba y lectura (hallazgo) abajo."""
    elementos = [_p(titulo, "titulo_figura"), Spacer(1, 3)]
    if ruta.exists():
        w, h = ImageReader(str(ruta)).getSize()
        escala = min(ancho / w, alto_max / h)
        elementos.append(Image(str(ruta), width=w * escala, height=h * escala))
    else:
        elementos.append(_p(f"[No se encontró la imagen {ruta.name}]", "nota"))
    elementos += [Spacer(1, 3), Paragraph(f"<b>Lectura:</b> {pie}", EST["pie_figura"]), Spacer(1, 12)]
    return KeepTogether(elementos)


def tarjetas_kpi(kpis: list[tuple[str, str]]) -> Table:
    celdas = [[Paragraph(valor, EST["kpi_valor"]) for valor, _ in kpis],
              [Paragraph(etiqueta, EST["kpi_etiqueta"]) for _, etiqueta in kpis]]
    t = Table(celdas, colWidths=[ANCHO_UTIL / len(kpis)] * len(kpis))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AZUL_CLARO),
        ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def caja(texto: str, color_fondo=AZUL_CLARO, color_borde=AZUL) -> Table:
    t = Table([[Paragraph(texto, EST["cuerpo"])]], colWidths=[ANCHO_UTIL])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color_fondo), ("LINEBEFORE", (0, 0), (0, -1), 3, color_borde),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ---------------------------------------------------------------------------
# Lienzo con "Página X de Y", encabezado y portada
# ---------------------------------------------------------------------------
class _LienzoNumerado(rl_canvas.Canvas):
    documento: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._paginas = []

    def showPage(self):
        self._paginas.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._paginas)
        for estado in self._paginas:
            self.__dict__.update(estado)
            if self._pageNumber == 1:
                self._portada()
            else:
                self._marco(total)
            super().showPage()
        super().save()

    def _marco(self, total: int):
        self.setStrokeColor(AZUL)
        self.setLineWidth(1.2)
        self.line(MARGEN, ALTO_PAGINA - 1.35 * cm, ANCHO_PAGINA - MARGEN, ALTO_PAGINA - 1.35 * cm)
        self.setFont(FUENTE, 7.5)
        self.setFillColor(GRIS)
        self.drawString(MARGEN, ALTO_PAGINA - 1.15 * cm, "Gravedad de los siniestros de transporte en Colombia (2015-2024)")
        self.drawRightString(ANCHO_PAGINA - MARGEN, ALTO_PAGINA - 1.15 * cm,
                             "Gerencia y almacenamiento en Big Data — " + ", ".join(self.documento["metadata"]["autores"]))
        self.drawString(MARGEN, 1.1 * cm, "Reporte generado automáticamente desde MongoDB")
        self.drawRightString(ANCHO_PAGINA - MARGEN, 1.1 * cm, f"Página {self._pageNumber} de {total}")

    def _portada(self):
        meta = self.documento["metadata"]
        modelo = self.documento["modelo"]
        # Franja superior
        self.setFillColor(AZUL_OSCURO)
        self.rect(0, ALTO_PAGINA - 11 * cm, ANCHO_PAGINA, 11 * cm, stroke=0, fill=1)
        self.setFillColor(ROJO)
        self.rect(0, ALTO_PAGINA - 11.25 * cm, ANCHO_PAGINA, 0.25 * cm, stroke=0, fill=1)
        self.setFillColor(colors.white)
        self.setFont(FUENTE, 10)
        self.drawString(MARGEN, ALTO_PAGINA - 2.6 * cm, "PROYECTO FINAL · " + meta.get("curso", "").upper())
        self.setFont(FUENTE_NEGRITA, 25)
        self.drawString(MARGEN, ALTO_PAGINA - 4.6 * cm, "Gravedad de los siniestros")
        self.drawString(MARGEN, ALTO_PAGINA - 5.75 * cm, "de transporte en Colombia")
        self.setFont(FUENTE, 13)
        self.drawString(MARGEN, ALTO_PAGINA - 7.0 * cm,
                        f"Análisis de {miles(meta['dataset']['registros'])} víctimas, {meta['dataset']['periodo']}")
        self.drawString(MARGEN, ALTO_PAGINA - 7.75 * cm, "Medicina Legal (INMLCF) y Policía Nacional — datos.gov.co")
        self.setFont(FUENTE_NEGRITA, 12)
        self.drawString(MARGEN, ALTO_PAGINA - 9.4 * cm, "Autor: " + ", ".join(meta["autores"]))

        # Pregunta de negocio
        y = ALTO_PAGINA - 13 * cm
        self.setFillColor(AZUL_OSCURO)
        self.setFont(FUENTE_NEGRITA, 11)
        self.drawString(MARGEN, y, "Pregunta de negocio")
        p = Paragraph(meta["pregunta_negocio"], ParagraphStyle("pq", fontName=FUENTE, fontSize=11, leading=15,
                                                              textColor=colors.HexColor("#222222")))
        _, alto = p.wrap(ANCHO_UTIL, 6 * cm)
        p.drawOn(self, MARGEN, y - 0.3 * cm - alto)

        # Tres cifras clave
        y = y - alto - 2.6 * cm
        cifras = [(miles(meta["dataset"]["registros"]), "víctimas analizadas"),
                  (str(len(meta["dataset"]["fuentes"])), "fuentes oficiales unificadas"),
                  (pct(modelo["metricas"]["recall_Muerte"] * 100, 0), "de las muertes detectadas por el modelo")]
        ancho = ANCHO_UTIL / 3
        for i, (valor, texto) in enumerate(cifras):
            x = MARGEN + i * ancho
            self.setFillColor(AZUL_CLARO)
            self.rect(x + 3, y - 0.4 * cm, ancho - 6, 2.3 * cm, stroke=0, fill=1)
            self.setFillColor(ROJO if i == 2 else AZUL_OSCURO)
            self.setFont(FUENTE_NEGRITA, 20)
            self.drawCentredString(x + ancho / 2, y + 0.85 * cm, valor)
            self.setFillColor(GRIS)
            self.setFont(FUENTE, 8.5)
            self.drawCentredString(x + ancho / 2, y + 0.1 * cm, texto)

        # Trazabilidad
        self.setFillColor(GRIS)
        self.setFont(FUENTE, 8.5)
        fecha = datetime.fromisoformat(str(meta["fecha_analisis"])).strftime("%d/%m/%Y %H:%M")
        lineas = [f"Fecha del análisis: {fecha}    ·    Reporte generado: {datetime.now():%d/%m/%Y %H:%M}",
                  f"Fuente: MongoDB — base '{config.MONGO_DB}', colección '{config.MONGO_COLECCION}'",
                  f"Documento: '{self.documento['_id']}'",
                  "Generado automáticamente con ReportLab (src/reporte_pdf.py): ningún resultado está escrito a mano."]
        for i, linea in enumerate(lineas):
            self.drawString(MARGEN, 3.6 * cm - i * 0.5 * cm, linea)


# ---------------------------------------------------------------------------
# Secciones del reporte
# ---------------------------------------------------------------------------
def _ruta(archivo: str) -> Path:
    return config.RAIZ_PROYECTO / archivo


def _viz(documento: dict, prefijo: str) -> dict:
    return next(v for v in documento["visualizaciones_generadas"] if v["nombre"].startswith(prefijo))


def seccion_resumen_ejecutivo(doc: dict) -> list:
    meta, modelo, est = doc["metadata"], doc["modelo"], doc["estadisticas_descriptivas"]
    m = modelo["metricas"]
    base = max(v["f1_macro"] for v in modelo["linea_base"].values())
    kpis = [(miles(meta["dataset"]["registros"]), "víctimas (2015-2024)"),
            (pct(est["letalidad_general_pct"]), "letalidad (Medicina Legal)"),
            (dec(m["f1_macro"]), f"F1 macro del modelo (base {dec(base)})"),
            (pct(m["recall_Muerte"] * 100, 0), "recall de Muerte")]
    criterio = modelo["criterio_exito"]
    texto_criterio = ("<b>Sí cumple</b>" if criterio["cumple_criterio"] else "<b>No cumple</b>")
    elementos = [
        Paragraph("Resumen ejecutivo", EST["h1"]),
        Paragraph(f"Se unificaron <b>{len(meta['dataset']['fuentes'])} datasets oficiales</b> de datos.gov.co "
                  f"(Medicina Legal y Policía Nacional) con <b>{miles(meta['dataset']['registros_crudos'])}</b> "
                  f"registros crudos, que tras el ETL quedan en <b>{miles(meta['dataset']['registros'])}</b> víctimas "
                  f"de siniestros de transporte entre {meta['dataset']['periodo']}. Con las víctimas de Medicina Legal "
                  f"se entrenó un modelo ({modelo['algoritmo']}) que clasifica la gravedad del siniestro en lesión "
                  f"leve, lesión grave o muerte. {texto_criterio} el criterio de éxito definido al inicio "
                  f"({criterio['definicion']}).", EST["cuerpo"]),
        Spacer(1, 4), tarjetas_kpi(kpis), Spacer(1, 12),
        Paragraph("Hallazgos principales", EST["h2"]),
    ]
    for insight in doc["insights"][:5]:
        elementos.append(Paragraph(f"• <b>{insight['titulo']}.</b>", EST["cuerpo"]))
    elementos += [Paragraph("Acciones prioritarias", EST["h2"])]
    for r in [r for r in doc["recomendaciones"] if r["prioridad"] == "Alta"]:
        elementos.append(Paragraph(f"• {r['recomendacion']}", EST["cuerpo"]))
    return elementos + [PageBreak()]


def seccion_dataset(doc: dict) -> list:
    meta, est = doc["metadata"], doc["estadisticas_descriptivas"]
    ds = meta["dataset"]
    filas = [[f["titulo"], f["entidad"].split("(")[-1].rstrip(")"), f["id_datos_gov"], miles(f["registros"]),
              str(f["columnas"])] for f in ds["fuentes"]]
    filas.append([_p("Total crudo", "celda_negrita"), "", "", _p(miles(ds["registros_crudos"]), "celda_negrita"), ""])
    por_fuente = [[r["fuente"], r["desenlace"], miles(r["victimas"])] for r in est["victimas_por_fuente_y_desenlace"]]
    anios = est["victimas_por_anio"]
    columnas = [c for c in anios[0] if c != "anio"]
    filas_anio = [[str(a["anio"])] + [miles(a[c]) for c in columnas] for a in anios]
    return [
        Paragraph("1. Resumen del dataset", EST["h1"]),
        Paragraph(f"<b>Fuente:</b> {ds['fuente']} ({ds['url']}). <b>Periodo:</b> {ds['periodo']}. "
                  f"<b>Dataset unificado:</b> {miles(ds['registros'])} registros y {ds['columnas']} columnas "
                  f"(una fila por víctima). Detalle de Medicina Legal: {miles(ds['registros_medicina_legal'])} "
                  f"víctimas; Policía Nacional: {miles(ds['registros_policia'])} víctimas.", EST["cuerpo"]),
        Paragraph("Fuentes originales", EST["h2"]),
        tabla(["Dataset", "Entidad", "Id", "Registros", "Columnas"], filas,
              [7.2 * cm, 2.3 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm], numericas=(3, 4)),
        Paragraph("Víctimas por fuente y desenlace (dataset unificado)", EST["h2"]),
        tabla(["Fuente", "Desenlace", "Víctimas"], por_fuente, [6 * cm, 4 * cm, 3 * cm], numericas=(2,)),
        Paragraph("Víctimas por año", EST["h2"]),
        tabla(["Año"] + columnas, filas_anio, [2 * cm] + [(ANCHO_UTIL - 2 * cm) / len(columnas)] * len(columnas),
              numericas=tuple(range(1, len(columnas) + 1))),
        Paragraph("Medicina Legal registra a quien pasa por examen forense; la Policía, a quien queda en una denuncia. "
                  "Por eso las dos fuentes se comparan, pero no se suman.", EST["nota"]),
        PageBreak(),
    ]


def seccion_calidad(doc: dict) -> list:
    cal = doc["calidad_datos"]
    problemas = [[p["fuente"], p["categoria"], p["problema"], miles(p["registros_afectados"]), p["tratamiento"]]
                 for p in cal["problemas_detectados"]]
    eliminados = [[e["motivo"], miles(e["registros"])] for e in cal["detalle_eliminados"]]
    eliminados.append([_p("Total eliminado", "celda_negrita"), _p(miles(cal["registros_eliminados"]), "celda_negrita")])
    pasos = [[p["fuente"], p["tecnica"], p["etapa"], miles(p["antes"]), miles(p["despues"]), miles(p["diferencia"])]
             for p in cal["pasos_etl"]]
    return [
        Paragraph("2. Problemas de calidad resueltos", EST["h1"]),
        Paragraph(f"El diagnóstico del ETL (CRISP-DM: Data Preparation) encontró <b>{cal['total_problemas']} "
                  f"problemas</b> de calidad. Todos se trataron con las técnicas: "
                  f"{', '.join(t.lower() for t in cal['tecnicas_aplicadas'])}. Se eliminaron "
                  f"<b>{miles(cal['registros_eliminados'])}</b> registros que no corresponden al estudio.",
                  EST["cuerpo"]),
        Paragraph("Registros eliminados y motivo", EST["h2"]),
        tabla(["Motivo", "Registros"], eliminados, [10 * cm, 3 * cm], numericas=(1,)),
        Paragraph("Problemas detectados y tratamiento aplicado", EST["h2"]),
        tabla(["Fuente", "Categoría", "Problema", "Afectados", "Tratamiento"], problemas,
              [2.3 * cm, 2.1 * cm, 5.2 * cm, 2.0 * cm, 4.4 * cm], numericas=(3,)),
        Paragraph("Conteos antes y después de cada paso del ETL", EST["h2"]),
        tabla(["Fuente", "Técnica", "Paso", "Antes", "Después", "Diferencia"], pasos,
              [2.6 * cm, 2.3 * cm, 5.3 * cm, 1.9 * cm, 1.9 * cm, 2 * cm], numericas=(3, 4, 5)),
        Paragraph("Las filas con conteos de columnas o de celdas (por ejemplo, 'Esquema común' o 'Celdas vacías') "
                  "miden esas unidades, no registros.", EST["nota"]),
        PageBreak(),
    ]


def seccion_agregaciones(doc: dict) -> list:
    titulos = {
        "muertes_por_region_y_fuente": "Muertes por región natural y fuente",
        "letalidad_por_medio_transporte": "Letalidad por medio de transporte (Medicina Legal)",
        "letalidad_por_zona": "Letalidad por zona del hecho (Medicina Legal)",
        "victimas_por_dia_semana": "Víctimas por día de la semana (Policía Nacional)",
        "top5_causas_por_desenlace": "Cinco causas más frecuentes por desenlace (víctimas con causa registrada)",
    }
    elementos = [Paragraph("3. Tablas de agregación (OLAP)", EST["h1"]),
                 Paragraph("Agregaciones calculadas sobre los cubos OLAP del ETL y guardadas en MongoDB. La "
                           "letalidad es el porcentaje de víctimas con gravedad conocida que fallecen.", EST["cuerpo"])]
    for ag in doc["agregaciones"]:
        resultado = ag["resultado"]
        columnas = list(resultado[0].keys())
        numericas = tuple(i for i, c in enumerate(columnas) if isinstance(resultado[0][c], (int, float)))
        filas = []
        for r in resultado:
            filas.append(["—" if r[c] is None else pct(r[c]) if c.endswith("_pct")
                          else (miles(r[c]) if i in numericas else r[c]) for i, c in enumerate(columnas)])
        encabezados = [c.replace("_pct", " (%)").replace("_", " ").capitalize() for c in columnas]
        ancho = min(ANCHO_UTIL, 4.2 * cm * len(columnas))
        elementos.append(KeepTogether([
            Paragraph(titulos.get(ag["nombre"], ag["nombre"]), EST["h2"]),
            Paragraph(f"Operación: {ag['operacion_olap']} · Dimensiones: {', '.join(ag['dimensiones'])} · "
                      f"Medida: {ag['medida']}", EST["nota"]),
            tabla(encabezados, filas, [ancho / len(columnas)] * len(columnas), numericas=numericas),
        ]))
    return elementos + [PageBreak()]


def seccion_visualizaciones(doc: dict) -> list:
    exploratorias = [v for v in doc["visualizaciones_generadas"] if v["nombre"][:2] < "09"]
    elementos = [Paragraph("4. Análisis exploratorio: visualizaciones", EST["h1"]),
                 Paragraph(f"Se presentan las {len(exploratorias)} gráficas del análisis exploratorio. Cada título "
                           "resume el hallazgo y la lectura explica qué muestra.", EST["cuerpo"])]
    for v in exploratorias:
        elementos.append(figura(_ruta(v["archivo"]), f"{v['titulo']}", v["hallazgo"]))
    return elementos + [PageBreak()]


def seccion_modelo(doc: dict) -> list:
    mo = doc["modelo"]
    val, cri = mo["validacion"], mo["criterio_exito"]
    metricas = [("accuracy", "Accuracy"), ("precision_macro", "Prec. macro"), ("recall_macro", "Recall macro"),
                ("f1_macro", "F1 macro"), ("recall_Muerte", "Recall Muerte"), ("precision_Muerte", "Prec. Muerte"),
                ("f1_Muerte", "F1 Muerte")]
    comparacion = sorted(mo["comparacion_modelos"], key=lambda c: c["f1_macro"], reverse=True)
    filas = [[c["modelo"]] + [dec(c[k]) for k, _ in metricas] for c in comparacion]
    fila_elegida = 1 + [c["modelo"] for c in comparacion].index(mo["algoritmo"])
    clases = mo["matriz_confusion"]["etiquetas"]
    importantes = [[v["variable_negocio"], dec(v["importancia"], 4)] for v in mo["variables_importantes"]]
    perfiles = [[p["perfil"], pct(p["Muerte"] * 100), f"×{dec(p['veces_riesgo_muerte_vs_base'], 1)}"]
                for p in mo["perfiles_riesgo"]]
    excluidas = [[k, v] for k, v in mo["variables_excluidas_por_fuga"].items()]
    v9, v10, v11 = _viz(doc, "09"), _viz(doc, "10"), _viz(doc, "11")
    return [
        Paragraph("5. Resultados del modelo", EST["h1"]),
        Paragraph(f"<b>Tarea:</b> {mo['tarea']} ({', '.join(clases)}). <b>Validación:</b> {val['tipo'].lower()}: "
                  f"entrenamiento {val['entrenamiento'][0]}-{val['entrenamiento'][1]} "
                  f"({miles(val['registros_entrenamiento'])} víctimas) y prueba {val['prueba'][0]}-{val['prueba'][1]} "
                  f"({miles(val['registros_prueba'])} víctimas). <b>Modelo elegido:</b> {mo['algoritmo']} "
                  f"({len(mo['variables_usadas'])} variables).", EST["cuerpo"]),
        Paragraph("Comparación con las líneas base (datos de prueba)", EST["h2"]),
        tabla(["Modelo"] + [n for _, n in metricas], filas, [4.0 * cm] + [1.9 * cm] * len(metricas),
              numericas=tuple(range(1, len(metricas) + 1)), resaltar={fila_elegida: AZUL_CLARO}),
        Spacer(1, 6),
        caja(f"<b>Criterio de éxito:</b> {cri['definicion']}. Mejora en F1 macro: "
             f"<b>{dec(cri['mejora_vs_linea_base'] * 100, 1)} puntos</b> "
             f"({'cumple' if cri['cumple_mejora_F1'] else 'no cumple'}); recall de Muerte: "
             f"<b>{dec(cri['recall_Muerte'])}</b> ({'cumple' if cri['cumple_recall_muerte'] else 'no cumple'}). "
             f"<b>Resultado: {'CUMPLE' if cri['cumple_criterio'] else 'NO CUMPLE'}.</b>",
             AZUL_CLARO if cri["cumple_criterio"] else ROJO_CLARO, VERDE if cri["cumple_criterio"] else ROJO),
        Spacer(1, 10),
        figura(_ruta(v9["archivo"]), v9["titulo"], v9["hallazgo"], alto_max=9.5 * cm),
        PageBreak(),
        Paragraph("Variables más importantes", EST["h2"]),
        figura(_ruta(v10["archivo"]), v10["titulo"], v10["hallazgo"], alto_max=7.6 * cm),
        Paragraph("Perfiles de riesgo", EST["h2"]),
        figura(_ruta(v11["archivo"]), v11["titulo"], v11["hallazgo"], alto_max=7.6 * cm),
        PageBreak(),
        Paragraph("Probabilidad estimada de muerte por perfil", EST["h2"]),
        tabla(["Perfil", "P(muerte)", "Veces el perfil base"], perfiles, [11.5 * cm, 2.3 * cm, 3.2 * cm],
              numericas=(1, 2), resaltar={1: GRIS_CLARO}),
        Paragraph("Importancia por permutación (caída del F1 macro al desordenar cada variable)", EST["h2"]),
        tabla(["Variable", "Importancia"], importantes, [8 * cm, 3 * cm], numericas=(1,)),
        Paragraph("Variables excluidas para evitar fuga de información", EST["h2"]),
        tabla(["Variable", "Motivo"], excluidas, [5.5 * cm, 11.5 * cm]),
        Paragraph("El modelo muestra asociaciones, no causas. Los perfiles cambian una característica a la vez frente "
                  "al perfil base.", EST["nota"]),
        PageBreak(),
    ]


def seccion_insights(doc: dict) -> list:
    elementos = [Paragraph("6. Insights", EST["h1"]),
                 Paragraph(f"{len(doc['insights'])} hallazgos calculados con los datos y almacenados en MongoDB.",
                           EST["cuerpo"])]
    for i in doc["insights"]:
        elementos.append(KeepTogether([
            caja(f"<font color='#c62828'><b>Insight {i['id']} · {i['tema']}</b></font><br/>"
                 f"<b>{i['titulo']}.</b><br/>{i['descripcion']}<br/>"
                 f"<font size='7.5' color='#5f6368'>Gráfica de soporte: {i['visualizacion']}</font>"),
            Spacer(1, 4)]))
    return elementos


def seccion_recomendaciones(doc: dict) -> list:
    color_prioridad = {"Alta": "#c62828", "Media": "#3a65a9", "Baja": "#5f6368"}
    filas = []
    for r in doc["recomendaciones"]:
        prioridad = Paragraph(f"<font color='{color_prioridad.get(r['prioridad'], '#333333')}'><b>{r['prioridad']}"
                              f"</b></font>", EST["celda"])
        filas.append([str(r["id"]), prioridad, r["tema"], r["recomendacion"], r["sustento"],
                      ", ".join(str(x) for x in r["insights_relacionados"])])
    return [
        Paragraph("7. Recomendaciones", EST["h1"]),
        Paragraph("Acciones sugeridas a partir de los insights, ordenadas por prioridad.", EST["cuerpo"]),
        tabla(["#", "Prioridad", "Tema", "Recomendación", "Sustento", "Insight"], filas,
              [0.6 * cm, 1.9 * cm, 2.6 * cm, 5.8 * cm, 4.6 * cm, 1.6 * cm]),
        Spacer(1, 8),
        caja("<b>Limitaciones.</b> Los resultados describen asociaciones en los registros oficiales, no relaciones "
             "causales. La hora y la causa del siniestro faltan en buena parte de las muertes, y Medicina Legal y "
             "Policía registran poblaciones distintas, por lo que sus cifras se comparan pero no se suman.",
             GRIS_CLARO, GRIS),
    ]


# ---------------------------------------------------------------------------
# Lectura desde MongoDB y generación
# ---------------------------------------------------------------------------
def obtener_documento(coleccion=None) -> dict:
    """Lee el documento de análisis desde MongoDB (la fuente del reporte)."""
    import mongo_analisis
    cliente = None
    if coleccion is None:
        cliente, coleccion = mongo_analisis.conectar()
    try:
        documento = coleccion.find_one({"_id": config.MONGO_ID_DOCUMENTO})
    finally:
        if cliente is not None:
            cliente.close()
    if documento is None:
        raise LookupError(f"No existe el documento '{config.MONGO_ID_DOCUMENTO}' en MongoDB. Ejecute la sección 6.")
    return documento


def generar_reporte(documento: dict | None = None, ruta: Path = ARCHIVO_REPORTE) -> Path:
    """Genera el PDF. Si no se pasa el documento, lo lee de MongoDB."""
    documento = documento or obtener_documento()
    config.crear_directorios()
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pdf = SimpleDocTemplate(str(ruta), pagesize=letter, leftMargin=MARGEN, rightMargin=MARGEN,
                            topMargin=2 * cm, bottomMargin=1.8 * cm,
                            title="Gravedad de los siniestros de transporte en Colombia (2015-2024)",
                            author=", ".join(documento["metadata"]["autores"]),
                            subject="Proyecto final — Gerencia y almacenamiento en Big Data")
    historia = [PageBreak()]  # la página 1 es la portada, dibujada por el lienzo
    for seccion in (seccion_resumen_ejecutivo, seccion_dataset, seccion_calidad, seccion_agregaciones,
                    seccion_visualizaciones, seccion_modelo, seccion_insights, seccion_recomendaciones):
        historia += seccion(documento)

    class Lienzo(_LienzoNumerado):
        pass
    Lienzo.documento = documento
    pdf.build(historia, canvasmaker=Lienzo)
    return ruta


def contar_paginas(ruta: Path = ARCHIVO_REPORTE) -> int:
    contenido = Path(ruta).read_bytes()
    return contenido.count(b"/Type /Page\n") or contenido.count(b"/Type /Page ")


if __name__ == "__main__":
    salida = generar_reporte()
    print(f"Reporte generado: {salida} ({salida.stat().st_size / 1024:,.0f} KB)")
