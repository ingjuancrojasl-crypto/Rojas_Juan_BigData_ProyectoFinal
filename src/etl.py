"""
ETL de siniestros de transporte en Colombia (CRISP-DM: Data Understanding + Data Preparation).

Integra 4 fuentes oficiales de datos.gov.co, periodo 2015-2024:
  - Medicina Legal (INMLCF): lesiones y muertes por eventos de transporte (registro forense, ~38 variables).
  - Policía Nacional: lesiones culposas y homicidios culposos en accidente de tránsito (denuncias, 9 variables).

Técnicas ETL:
  1. Limpieza        -> filtrar conductas de tránsito y periodo, duplicados, valores incoherentes, nulos,
                        columnas constantes o casi vacías.
  2. Normalización   -> nombres de columnas, catálogos de texto (mayúsculas, tildes, variantes por año),
                        códigos DANE, fechas, tipos y nivel de detalle (una fila = una víctima).
  3. Agregación      -> cubos OLAP y comparación de cobertura entre fuentes.
  4. Enriquecimiento -> gravedad (leve / grave / muerte), categoría de causa, grupos de vehículo y objeto,
                        región, franja horaria, fin de semana y contexto policial del municipio (año anterior).

Salidas:
  data/processed/victimas_medicina_legal.parquet  -> base del modelo (todas las variables armonizadas)
  data/processed/siniestros_unificado.parquet     -> las 4 fuentes con las columnas comunes
  data/processed/agregaciones/*.parquet           -> cubos OLAP
  data/processed/resumen_etl.json                 -> conteos antes/después y problemas detectados
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SI = config.SIN_INFORMACION

# ---------------------------------------------------------------------------
# Esquemas de columnas
# ---------------------------------------------------------------------------
RENOMBRAR_INML = {
    "ID": "id_origen", "Año del hecho": "anio", "Sexo de la victima": "sexo",
    "Grupo de edad quinquenal": "grupo_edad_quinquenal", "Grupo de Edad Quinquenal": "grupo_edad_quinquenal",
    "Grupo Mayor Menor de Edad": "mayor_menor_edad", "Grupo de Edad judicial": "grupo_edad_judicial",
    "Ciclo Vital": "ciclo_vital", "País de Nacimiento": "pais_nacimiento", "Escolaridad": "escolaridad",
    "Estado Civil": "estado_civil", "Tipo de Discapacidad": "tipo_discapacidad",
    "Pertenencia Étnica": "pertenencia_etnica", "Orientación Sexual": "orientacion_sexual",
    "Identidad de Género": "identidad_genero", "Transgénero": "transgenero",
    "Pertenencia Grupal": "pertenencia_grupal", "Mes del hecho": "mes_nombre", "Dia del hecho": "dia_semana",
    "Rango de Hora del Hecho X 3 Horas": "rango_hora", "Rango de Hora del Hecho (X3 Horas)": "rango_hora",
    "Código Dane Municipio": "codigo_dane_municipio", "Municipio del hecho DANE": "municipio",
    "Departamento del hecho DANE": "departamento", "Código Dane Departamento": "codigo_dane_departamento",
    "Localidad del Hecho": "localidad", "Zona del Hecho": "zona", "Escenario del Hecho": "escenario",
    "Actividad Durante el Hecho": "actividad", "Circunstancia del Hecho Detallada": "circunstancia",
    "Contexto del Hecho": "contexto", "Manera de Muerte": "manera_muerte",
    "Mecanismo Causal de la Lesión no Fatal": "mecanismo_causal", "Mecanismo Causal de la Lesión Fatal": "mecanismo_causal",
    "Diagnostico Topográfico de la Lesión no Fatal": "diagnostico_topografico",
    "Diagnostico Topográfico de la Lesión Fatal": "diagnostico_topografico",
    "Condición de la Víctima (AT)": "condicion_victima", "Medio de Desplazamiento o Transporte": "medio_transporte",
    "Servicio del Vehículo": "servicio_vehiculo",
    "Clase o Tipo de Accidente": "clase_accidente", "Clase o Tipo de Accidente de Transporte": "clase_accidente",
    "Objeto de Colisión": "objeto_colision", "Servicio del Objeto de Colisión": "servicio_objeto",
    "Días de Incapacidad Medicolegal": "dias_incapacidad", "Pueblo Indígena": "pueblo_indigena",
    "Ancestro Racial": "ancestro_racial",
}
RENOMBRAR_POLICIA = {
    "DEPARTAMENTO": "departamento_policia", "MUNICIPIO": "municipio_policia", "CODIGO DANE": "codigo_dane_policia",
    "ARMAS MEDIOS": "armas_medios", "FECHA HECHO": "fecha_texto", "GENERO": "genero",
    "GRUPO ETARIO": "grupo_etario", "GRUPO ETARÍO": "grupo_etario",
    "DESCRIPCIÓN CONDUCTA": "conducta", "CANTIDAD": "cantidad",
}
# Columnas que no pasan a la base analítica y motivo (se documenta en el notebook).
COLUMNAS_DESCARTADAS_INML = {
    "contexto": "constante (un solo valor en el archivo de lesiones)",
    "manera_muerte": "constante (un solo valor en el archivo de muertes)",
    "orientacion_sexual": "61-83 % sin información; no se registraba en años anteriores",
    "identidad_genero": "73-78 % sin información; no se registraba en años anteriores",
    "transgenero": "98-99 % sin información",
    "localidad": "87-93 % sin información (solo aplica a ciudades con localidades)",
    "tipo_discapacidad": "solo existe en el archivo de lesiones",
    "ancestro_racial": "solo existe en el archivo de muertes",
    "pueblo_indigena": "29-38 % sin información y 'No había sido implementada' en años antiguos",
    "pertenencia_grupal": "catálogo de 43-73 valores sin estructura comparable",
    "grupo_edad_judicial": "redundante con grupo_edad_quinquenal",
    "mayor_menor_edad": "redundante con ciclo_vital",
    "pais_nacimiento": "98 % Colombia: casi sin variación",
}

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
         "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# ---------------------------------------------------------------------------
# Reglas de armonización (sobre texto normalizado: minúsculas, sin tildes)
# ---------------------------------------------------------------------------
REGLAS_CAUSA = [  # (categoría, patrón) en orden de prioridad
    ("Embriaguez o consumo de sustancias", r"embriag"),
    ("Exceso de velocidad", r"exceso de velocidad"),
    ("Desobedecer señales y normas", r"desobedecer|semaforo|contravia|prelacion|vias prohibidas|violacion normas"
                                     r"|carriles destinados|anden|calzadas o carriles"),
    ("Maniobras peligrosas", r"adelantar|cambio de carril|frenar|reverso|distancia de seguridad|entre vehiculos"
                             r"|uno al lado|distante de la acera|remolque"),
    ("Imprudencia del peatón", r"cruzar|pararse sobre|salir por delante|jugar en la via|transitar por la calzada"
                               r"|de peaton"),
    ("Fallas mecánicas o del vehículo", r"falla|mantenimiento|vehiculo varado|reflectivas|carga sobresaliente"
                                        r"|reparacion"),
    ("Condiciones de la vía o el clima", r"condiciones|niebla|obstaculos|mal estacionado"),
    ("Distracción o fatiga", r"cansancio|dormido|telefono|mensajes|pantallas|fumar|aprendizaje"),
    ("Conducta del pasajero o transporte inseguro", r"descender|colgado|sobrecupo|transportar|platon|asiento"
                                                    r"|sujetar|horcajadas|de pasajero|izquierda del conductor"
                                                    r"|explosivos|aprovisionamiento|pasajero"),
]
CAUSAS_NO_ESPECIFICADAS = r"^(otras?|otro|otras circunstancias no especificadas|de motociclista o ciclista" \
                          r"|desastre no natural.*)$"

REGLAS_MEDIO = [
    ("Motocicleta", r"moto|cuatrimoto|bicimotor"),
    ("Bicicleta y micromovilidad", r"bici|patineta|segway"),
    ("Vehículo automotor", r"automovil|bus|camion|tracto|volqueta|camioneta|campero|microbus|articulado"
                           r"|ambulancia|remolque|patrulla|grua|carro de golf"),
    ("Peatón (sin vehículo)", r"^no aplica$|^peaton$"),
    ("Otros medios", r"avion|helicoptero|aeronave|acuatic|tren|cable|traccion|maquinaria|semoviente|animal|objeto"),
]
REGLAS_OBJETO = [
    ("Sin colisión", r"^no aplica$"),
    ("Objeto fijo", r"objeto fijo|poste|muro|arbol|inmueble|estacionado"),
    ("Vehículo de carga pesada", r"camion|tracto|volqueta|remolque|maquinaria"),
    ("Bus o buseta", r"bus|articulado"),
    ("Automóvil o camioneta", r"automovil|camioneta|campero|ambulancia|carro de golf|patrulla"),
    ("Motocicleta", r"moto|cuatrimoto"),
    ("Bicicleta o tracción humana", r"bici|patineta|segway|traccion humana|humano"),
    ("Animal", r"animal|semoviente|traccion animal"),
    ("Otros", r"tren|acuatic|avion|helicoptero|aeronave|cable|otro"),
]
MAPA_CLASE = {
    "choque": "Choque", "choque con otro vehiculo": "Choque",
    "choque con objeto fijo": "Choque con objeto fijo o en movimiento",
    "choque con objeto fijo o en movimiento": "Choque con objeto fijo o en movimiento",
    "caida de ocupante": "Caída del ocupante", "caida del ocupante": "Caída del ocupante",
    "lesion dentro de vehiculo": "Lesión dentro del vehículo",
    "caida del vehiculo a precipicio": "Caída del vehículo a precipicio",
    "caida del vehiculo a un precipicio": "Caída del vehículo a precipicio",
    "volcamiento": "Volcamiento", "atropello": "Atropello",
}
MEDIOS_POLICIA = {"VEHICULO": "Vehículo automotor", "MOTO": "Motocicleta", "BICICLETA": "Bicicleta y micromovilidad",
                  "TREN": "Otros medios", "SIN EMPLEO DE ARMAS": SI, "NO REPORTADO": SI}
MEDIOS_POLICIA_INCOHERENTES = {"CONTUNDENTES", "ESCOPOLAMINA", "ARMA BLANCA / CORTOPUNZANTE", "ARMA DE FUEGO"}
EDAD_COMUN_POLICIA = {"MENORES": "Menores (0-11)", "ADOLESCENTES": "Adolescentes (12-17)", "ADULTOS": "Adultos (18 y más)"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def clave(texto) -> str:
    """Texto comparable: minúsculas, sin tildes, sin paréntesis, espacios simples."""
    if pd.isna(texto):
        return ""
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"\s*([,/–-])\s*", r" \1 ", s)
    return re.sub(r"\s+", " ", s).strip()


ES_SIN_INFO = re.compile(r"^(sin informacion|no sabe / no informa|no reporta|no reportado|por determinar|)$")


def es_sin_informacion(serie: pd.Series) -> pd.Series:
    """Marca los valores vacíos o equivalentes a 'Sin información'.

    Evalúa cada valor distinto una sola vez (en vez de fila por fila), lo que lo hace
    cientos de veces más rápido en columnas de millones de filas con pocos valores distintos.
    """
    unicos = serie.dropna().unique()
    marcas = {v: bool(ES_SIN_INFO.match(clave(v))) for v in unicos}
    return serie.map(marcas).fillna(True).astype(bool)


def pct_sin_informacion(serie: pd.Series) -> float:
    return float(es_sin_informacion(serie).mean() * 100)


def mapear_unicos(serie: pd.Series, funcion) -> pd.Series:
    """Aplica `funcion` una vez por valor distinto y propaga el resultado (mucho más rápido que fila a fila)."""
    unicos = pd.Series(serie.unique())
    return serie.map(dict(zip(unicos, unicos.map(funcion))))


def mapear_por_reglas(serie: pd.Series, reglas: list[tuple[str, str]], por_defecto: str = "Otros") -> pd.Series:
    """Asigna una categoría a cada valor único (rápido: se evalúa una vez por valor distinto)."""
    def categoria(valor):
        k = clave(valor)
        if ES_SIN_INFO.match(k):
            return SI
        for nombre, patron in reglas:
            if re.search(patron, k):
                return nombre
        return por_defecto
    unicos = {v: categoria(v) for v in serie.dropna().unique()}
    return serie.map(unicos).fillna(SI)


def canonico(serie: pd.Series) -> pd.Series:
    """Unifica variantes de escritura: por cada clave elige la escritura más frecuente."""
    texto = serie.fillna(SI).astype(str).str.strip()
    claves = mapear_unicos(texto, lambda v: "sin informacion" if ES_SIN_INFO.match(clave(v)) else clave(v))
    tabla = (pd.DataFrame({"t": texto, "k": claves}).value_counts().reset_index()
             .drop_duplicates("k").set_index("k")["t"])
    tabla["sin informacion"] = SI
    return claves.map(tabla)


def memoria_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1e6, 1)


def _paso(fuente: str, tecnica: str, etapa: str, antes, despues, detalle: str = "") -> dict:
    return {"fuente": fuente, "tecnica": tecnica, "etapa": etapa, "antes": int(antes), "despues": int(despues),
            "diferencia": int(despues) - int(antes), "detalle": detalle}


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------
def cargar_fuentes() -> dict[str, pd.DataFrame]:
    """Lee los 4 CSV crudos como texto (sin inferir tipos) para no perder ceros iniciales ni formatos."""
    fuentes = {}
    for nombre, fuente in config.FUENTES.items():
        if not fuente["archivo"].exists():
            raise FileNotFoundError(f"Falta {fuente['archivo']}. Ejecute: python src/descargar_datos.py")
        fuentes[nombre] = pd.read_csv(fuente["archivo"], dtype=str, encoding="utf-8-sig", keep_default_na=False,
                                      na_values=[""])
    return fuentes


def perfil_fuentes(fuentes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    filas = []
    for nombre, df in fuentes.items():
        anios = (df["Año del hecho"].astype(int) if "Año del hecho" in df
                 else pd.to_datetime(df["FECHA HECHO"], format="%d/%m/%Y").dt.year)
        filas.append({"fuente": nombre, "entidad": config.FUENTES[nombre]["entidad"],
                      "id_datos_gov": config.FUENTES[nombre]["id"], "filas": len(df), "columnas": df.shape[1],
                      "disco_mb": round(config.FUENTES[nombre]["archivo"].stat().st_size / 1e6, 1),
                      "memoria_mb": memoria_mb(df), "periodo": f"{anios.min()}-{anios.max()}"})
    return pd.DataFrame(filas).set_index("fuente")


# ---------------------------------------------------------------------------
# Data Understanding: diagnóstico de calidad (todo se mide sobre los datos crudos)
# ---------------------------------------------------------------------------
def diagnosticar_calidad(fuentes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    L = fuentes["inml_lesiones"].rename(columns=RENOMBRAR_INML)
    M = fuentes["inml_muertes"].rename(columns=RENOMBRAR_INML)
    PL = fuentes["policia_lesiones"].rename(columns=RENOMBRAR_POLICIA)
    PH = fuentes["policia_homicidios_transito"].rename(columns=RENOMBRAR_POLICIA)
    P = []

    nombres_distintos = sum(1 for a in fuentes["inml_lesiones"].columns for b in fuentes["inml_muertes"].columns
                            if a != b and RENOMBRAR_INML.get(a) == RENOMBRAR_INML.get(b))
    P.append(("Medicina Legal", "Esquema", "Columnas equivalentes con nombre distinto entre lesiones y muertes",
              nombres_distintos, "Normalización: renombrar a un esquema común en snake_case"))

    no_transito = int((~PL["conducta"].str.contains("ACCIDENTE DE TRANSITO", na=False)).sum())
    P.append(("Policía lesiones", "Alcance", "Filas que NO son de tránsito (lesiones personales, riñas, etc.)",
              no_transito, "Limpieza: filtrar conducta 'EN ACCIDENTE DE TRANSITO'"))

    fuera = 0
    for df in (PL, PH):
        anio = pd.to_datetime(df["fecha_texto"], format="%d/%m/%Y").dt.year
        fuera += int((~anio.between(config.ANIO_INICIO, config.ANIO_FIN)).sum())
    P.append(("Policía", "Alcance", f"Filas fuera del periodo común {config.ANIO_INICIO}-{config.ANIO_FIN}",
              fuera, "Limpieza: filtrar al periodo común de las 4 fuentes"))

    dup_inml = int(L.drop(columns="id_origen").duplicated().sum() + M.drop(columns="id_origen").duplicated().sum())
    P.append(("Medicina Legal", "Duplicados", "Filas idénticas en las 38-39 columnas (sin contar el ID)", dup_inml,
              "Limpieza: eliminar (con tantas variables, coincidir en todo es un registro repetido)"))

    dup_ph = int(PH.duplicated().sum())
    P.append(("Policía homicidios", "Duplicados aparentes", "Filas idénticas en las 9 columnas (2010-2026)", dup_ph,
              "Se conservan: con solo 9 variables, dos víctimas distintas del mismo día y municipio se ven iguales"))

    transito = PL["conducta"].str.contains("EN ACCIDENTE DE TRANSITO", na=False)
    multi = int((PL.loc[transito, "cantidad"].astype(int) > 1).sum())
    P.append(("Policía lesiones", "Granularidad", "Filas de tránsito que agrupan varias víctimas (CANTIDAD > 1, "
              f"hasta {PL.loc[transito, 'cantidad'].astype(int).max()})",
              multi, "Normalización: expandir a una fila por víctima"))

    incoh = int(PL.loc[PL["conducta"].str.contains("ACCIDENTE DE TRANSITO", na=False), "armas_medios"]
                .isin(MEDIOS_POLICIA_INCOHERENTES).sum() + PH["armas_medios"].isin(MEDIOS_POLICIA_INCOHERENTES).sum())
    P.append(("Policía", "Valores incoherentes", "Medio = escopolamina, arma blanca, arma de fuego o contundente "
              "en un accidente de tránsito", incoh, "Limpieza: medio -> 'Sin información' y bandera"))

    sin_genero = int(PL["genero"].isin(["NO REPORTA"]).sum() + PL["genero"].isna().sum() + PH["genero"].isna().sum())
    P.append(("Policía", "Nulos", "Registros sin género (vacío o 'NO REPORTA')", sin_genero,
              "Limpieza: 'Sin información'"))

    for col, desc in [("rango_hora", "hora del hecho"), ("circunstancia", "causa (circunstancia)"),
                      ("actividad", "actividad durante el hecho"), ("escolaridad", "escolaridad")]:
        sl, sm = es_sin_informacion(L[col]), es_sin_informacion(M[col])
        pl, pm = sl.mean() * 100, sm.mean() * 100
        P.append(("Medicina Legal", "Registro desigual", f"Sin dato de {desc}: {pl:.0f} % en lesiones vs "
                  f"{pm:.0f} % en muertes", int(sl.sum() + sm.sum()),
                  "Modelo: no usar como predictor (fuga de información); causas solo con registros con dato"))

    for col in ["mes_nombre", "dia_semana", "escenario", "zona", "clase_accidente", "medio_transporte",
                "objeto_colision", "circunstancia", "condicion_victima"]:
        valores = pd.concat([L[col], M[col]]).dropna().unique()
        variantes = len(valores) - len({clave(v) for v in valores})
        if variantes:
            P.append(("Medicina Legal", "Texto inconsistente", f"'{col}': valores que solo cambian en "
                      "mayúsculas, tildes o espacios", variantes, "Normalización: etiqueta canónica"))

    cambio = pd.crosstab(L["anio"], L["clase_accidente"])[["Choque", "Choque con otro vehículo"]]
    anios_choque = cambio.index[cambio["Choque"] > 1000]
    P.append(("Medicina Legal", "Cambio de catálogo", "'Choque con otro vehículo' pasa a llamarse 'Choque' desde "
              f"{anios_choque.min()}", int(cambio.values.sum()), "Normalización: unificar en 'Choque'"))

    causas = pd.concat([L["circunstancia"], M["circunstancia"]]).dropna().unique()
    P.append(("Medicina Legal", "Catálogo extenso", "Causas distintas (137 en lesiones, 65 en muertes, con variantes)",
              len(causas), "Enriquecimiento: agrupar en 9 categorías de causa comparables"))

    dias = L["dias_incapacidad"].value_counts()
    P.append(("Medicina Legal lesiones", "Cambio de catálogo", "Días de incapacidad con 9 etiquetas distintas "
              "para 4 conceptos (p. ej. 'Cero', 'Cero días', 'Sin días de incapacidad')", len(dias),
              "Enriquecimiento: gravedad leve (0-30 días) / grave (> 30 días)"))

    codigos_cortos = int((L["codigo_dane_municipio"].str.len() < 5).sum() + (M["codigo_dane_municipio"].str.len() < 5).sum())
    P.append(("Medicina Legal", "Formato de códigos", "Códigos DANE de municipio sin el cero inicial (4 dígitos)",
              codigos_cortos, "Normalización: completar a 5 dígitos"))
    P.append(("Policía", "Formato de códigos", "Código DANE de 8 dígitos y municipio con sufijo '(CT)'",
              int(PL["municipio_policia"].str.contains(r"\(CT\)", na=False).sum()),
              "Normalización: 5 primeros dígitos y nombre oficial DANE"))

    constantes = [c for c in ["contexto", "manera_muerte"] if c in L or c in M]
    P.append(("Medicina Legal", "Columnas sin aporte", "Columnas constantes o con > 60 % sin dato",
              len(COLUMNAS_DESCARTADAS_INML), "Limpieza: excluir de la base analítica (motivo documentado)"))
    return pd.DataFrame(P, columns=["fuente", "categoria", "problema", "registros_afectados", "tratamiento"])


# ---------------------------------------------------------------------------
# Técnica 1: Limpieza
# ---------------------------------------------------------------------------
def limpiar_inml(L_crudo: pd.DataFrame, M_crudo: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pasos = []
    L = L_crudo.rename(columns=RENOMBRAR_INML).assign(archivo_origen="inml_lesiones")
    M = M_crudo.rename(columns=RENOMBRAR_INML).assign(archivo_origen="inml_muertes")
    pasos.append(_paso("Medicina Legal", "Limpieza", "Esquema común (columnas)", L_crudo.shape[1] + M_crudo.shape[1],
                       len(set(L.columns) | set(M.columns)) - 1, "Renombrado a snake_case; 5 pares de columnas equivalentes con nombre distinto"))
    df = pd.concat([L, M], ignore_index=True)
    filas = len(df)

    df = df[~df.drop(columns=["id_origen"]).duplicated()]
    pasos.append(_paso("Medicina Legal", "Limpieza", "Eliminar duplicados (sin ID)", filas, len(df)))

    filas = len(df)
    df = df[df["anio"].astype(int).between(config.ANIO_INICIO, config.ANIO_FIN)]
    pasos.append(_paso("Medicina Legal", "Limpieza", "Validar periodo", filas, len(df)))

    columnas = df.shape[1]
    df = df.drop(columns=[c for c in COLUMNAS_DESCARTADAS_INML if c in df])
    pasos.append(_paso("Medicina Legal", "Limpieza", "Excluir columnas constantes o casi vacías (columnas)",
                       columnas, df.shape[1], "Ver COLUMNAS_DESCARTADAS_INML"))

    nulos = int(df.isna().sum().sum())
    df = df.fillna(SI)
    pasos.append(_paso("Medicina Legal", "Limpieza", "Celdas vacías -> 'Sin información'", nulos,
                       int(df.isna().sum().sum())))
    return df.reset_index(drop=True), pasos


def limpiar_policia(PL_crudo: pd.DataFrame, PH_crudo: pd.DataFrame,
                    anio_inicio: int = config.ANIO_INICIO) -> tuple[pd.DataFrame, list[dict]]:
    pasos = []
    PL = PL_crudo.rename(columns=RENOMBRAR_POLICIA).assign(archivo_origen="policia_lesiones", desenlace="Lesión")
    PH = PH_crudo.rename(columns=RENOMBRAR_POLICIA).assign(archivo_origen="policia_homicidios_transito",
                                                          desenlace="Muerte")
    filas = len(PL)
    PL = PL[PL["conducta"].str.contains("EN ACCIDENTE DE TRANSITO", na=False)]
    pasos.append(_paso("Policía lesiones", "Limpieza", "Filtrar conducta de tránsito", filas, len(PL),
                       "Se excluyen lesiones personales (art. 111, 119...) y 8.651 'ART. 120' sin mención de tránsito"))

    df = pd.concat([PL, PH], ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha_texto"], format="%d/%m/%Y", errors="coerce")
    filas = len(df)
    df = df[df["fecha"].notna() & df["fecha"].dt.year.between(anio_inicio, config.ANIO_FIN)]
    pasos.append(_paso("Policía", "Limpieza", f"Filtrar periodo {anio_inicio}-{config.ANIO_FIN}", filas, len(df)))

    incoherentes = df["armas_medios"].isin(MEDIOS_POLICIA_INCOHERENTES)
    df["medio_incoherente"] = incoherentes.astype("int8")
    pasos.append(_paso("Policía", "Limpieza", "Marcar medio incoherente con tránsito", 0, int(incoherentes.sum()),
                       "Escopolamina, armas, contundentes -> medio 'Sin información'"))

    nulos = int(df[["genero", "grupo_etario"]].isna().sum().sum() + (df["genero"] == "NO REPORTA").sum())
    df["genero"] = df["genero"].replace({"NO REPORTA": np.nan}).fillna(SI)
    df["grupo_etario"] = df["grupo_etario"].fillna(SI)
    pasos.append(_paso("Policía", "Limpieza", "Género/edad vacíos -> 'Sin información'", nulos, 0))
    return df.reset_index(drop=True), pasos


# ---------------------------------------------------------------------------
# Técnica 2: Normalización
# ---------------------------------------------------------------------------
def _formato_hora(valor: str) -> str:
    m = re.search(r"(\d{2}:\d{2})\s*a\s*(\d{2}:\d{2})", str(valor))
    return f"({m.group(1)} a {m.group(2)})" if m else SI


def normalizar_inml(df_limpio: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pasos = []
    df = df_limpio.copy()
    memoria = memoria_mb(df)

    texto = ["sexo", "grupo_edad_quinquenal", "ciclo_vital", "escolaridad", "estado_civil", "pertenencia_etnica",
             "mes_nombre", "dia_semana", "municipio", "departamento", "zona", "escenario", "actividad",
             "circunstancia", "mecanismo_causal", "diagnostico_topografico", "condicion_victima", "medio_transporte",
             "servicio_vehiculo", "clase_accidente", "objeto_colision", "servicio_objeto", "dias_incapacidad"]
    total_antes = total_despues = 0
    for col in texto:
        antes = df[col].nunique()
        df[col] = canonico(df[col])
        total_antes, total_despues = total_antes + antes, total_despues + df[col].nunique()
    pasos.append(_paso("Medicina Legal", "Normalización", "Unificar escritura en 23 columnas de texto (valores únicos)",
                       total_antes, total_despues, "Mayúsculas, tildes, espacios, paréntesis"))

    antes = df["clase_accidente"].nunique()
    df["clase_accidente"] = mapear_unicos(df["clase_accidente"], lambda v: MAPA_CLASE.get(clave(v), v))
    pasos.append(_paso("Medicina Legal", "Normalización", "Clase de accidente: catálogos por año", antes,
                       df["clase_accidente"].nunique(), "'Choque con otro vehículo' (2015-2018) = 'Choque' (2019-2024)"))

    antes = df["rango_hora"].nunique()
    df["rango_hora"] = mapear_unicos(df["rango_hora"], _formato_hora)
    pasos.append(_paso("Medicina Legal", "Normalización", "Formato de rango horario", antes, df["rango_hora"].nunique(),
                       "'18:00 a 20:59', '(18:00 a  20:59)' -> '(18:00 a 20:59)'"))

    antes = df["zona"].nunique()
    df["zona"] = mapear_unicos(df["zona"], lambda v: "Centro poblado" if "centro poblado" in clave(v) else v)
    pasos.append(_paso("Medicina Legal", "Normalización", "Zona del hecho", antes, df["zona"].nunique()))

    df["sexo"] = df["sexo"].replace({"Indeterminado": SI})
    df["condicion_victima"] = df["condicion_victima"].str.capitalize().replace({"Sin información": SI})
    df["mes"] = mapear_unicos(df["mes_nombre"], lambda v: MESES.get(clave(v), 0)).astype("int8")
    df["dia_semana"] = mapear_unicos(df["dia_semana"], lambda v: next((d for d in DIAS if clave(d) == clave(v)), SI))

    cortos = int((df["codigo_dane_municipio"].str.len() < 5).sum())
    df["codigo_dane_municipio"] = df["codigo_dane_municipio"].str.zfill(5)
    df["codigo_dane_departamento"] = df["codigo_dane_departamento"].str.zfill(2)
    pasos.append(_paso("Medicina Legal", "Normalización", "Códigos DANE a 5 y 2 dígitos", cortos,
                       int((df["codigo_dane_municipio"].str.len() < 5).sum())))

    df["anio"] = df["anio"].astype("int16")
    df["id_origen"] = df["id_origen"].astype("int32")
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            df[col] = df[col].astype("category")
    pasos.append(_paso("Medicina Legal", "Normalización", "Tipos optimizados (MB en memoria)", round(memoria),
                       round(memoria_mb(df))))
    return df, pasos


def normalizar_policia(df_limpio: pd.DataFrame, nombres_dane: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pasos = []
    df = df_limpio.copy()

    df["codigo_dane_municipio"] = df["codigo_dane_policia"].str[:5]
    df["codigo_dane_departamento"] = df["codigo_dane_policia"].str[:2]
    municipios = nombres_dane.drop_duplicates("codigo_dane_municipio").set_index("codigo_dane_municipio")
    departamentos = nombres_dane.drop_duplicates("codigo_dane_departamento").set_index("codigo_dane_departamento")
    df["municipio"] = df["codigo_dane_municipio"].map(municipios["municipio"])
    sin_cruce = int(df["municipio"].isna().sum())
    df["municipio"] = df["municipio"].fillna(df["municipio_policia"].str.replace(r"\s*\(CT\)", "", regex=True))
    df["departamento"] = (df["codigo_dane_departamento"].map(departamentos["departamento"])
                          .fillna(df["departamento_policia"].str.title()))
    pasos.append(_paso("Policía", "Normalización", "Municipio y departamento con nombre oficial DANE (vía código)",
                       len(df), len(df) - sin_cruce, f"{sin_cruce:,} filas con municipio sin víctimas en Medicina "
                       "Legal: se limpia el nombre de la Policía"))

    df["sexo"] = df["genero"].replace({"MASCULINO": "Hombre", "FEMENINO": "Mujer"})
    df["grupo_edad_comun"] = df["grupo_etario"].map(EDAD_COMUN_POLICIA).fillna(SI)
    df["medio_grupo"] = df["armas_medios"].map(MEDIOS_POLICIA).fillna(SI)
    df["anio"] = df["fecha"].dt.year.astype("int16")
    df["mes"] = df["fecha"].dt.month.astype("int8")
    df["dia_semana"] = df["fecha"].dt.dayofweek.map(dict(enumerate(DIAS)))

    filas = len(df)
    df["cantidad"] = df["cantidad"].astype(int)
    df = df.loc[df.index.repeat(df["cantidad"])].reset_index(drop=True)
    pasos.append(_paso("Policía", "Normalización", "Expandir CANTIDAD: una fila por víctima", filas, len(df),
                       "Misma unidad de análisis que Medicina Legal"))
    return df, pasos


# ---------------------------------------------------------------------------
# Técnica 4: Enriquecimiento
# ---------------------------------------------------------------------------
def _gravedad(fila_archivo: str, dias: str) -> str:
    if fila_archivo == "inml_muertes":
        return "Muerte"
    k = clave(dias)
    if k in ("31 a 90", "mas de 90"):
        return "Lesión grave"
    if k in ("1 a 30", "cero", "cero dias", "sin dias de incapacidad"):
        return "Lesión leve"
    return SI


def _franja(hora: str) -> str:
    m = re.match(r"\((\d{2})", str(hora))
    if not m:
        return SI
    h = int(m.group(1))
    return "Madrugada (0-6 h)" if h < 6 else "Mañana (6-12 h)" if h < 12 else "Tarde (12-18 h)" if h < 18 else "Noche (18-24 h)"


def categorizar_causa(circunstancia: str) -> str:
    k = clave(circunstancia)
    if ES_SIN_INFO.match(k):
        return SI
    if re.match(CAUSAS_NO_ESPECIFICADAS, k):
        return "Otras / no especificadas"
    for nombre, patron in REGLAS_CAUSA:
        if re.search(patron, k):
            return nombre
    return "Otras / no especificadas"


def contexto_policial(policia: pd.DataFrame) -> pd.DataFrame:
    """Víctimas registradas por la Policía en cada municipio y año (se usa con un año de rezago)."""
    tabla = (policia.groupby(["codigo_dane_municipio", "anio", "desenlace"]).size()
             .unstack("desenlace", fill_value=0)
             .rename(columns={"Lesión": "lesionados_policia_anio_ant", "Muerte": "muertes_policia_anio_ant"})
             .reset_index())
    tabla["anio"] = tabla["anio"] + 1          # el valor de 2018 se asocia a los hechos de 2019
    return tabla


def enriquecer_inml(df_normal: pd.DataFrame, policia_todos_los_anios: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pasos = []
    df = df_normal.copy()
    columnas = df.shape[1]

    df["desenlace"] = np.where(df["archivo_origen"] == "inml_muertes", "Muerte", "Lesión")
    pares = df["archivo_origen"].astype(str) + "|" + df["dias_incapacidad"].astype(str)
    df["gravedad"] = mapear_unicos(pares, lambda v: _gravedad(*v.split("|", 1)))
    pasos.append(_paso("Medicina Legal", "Enriquecimiento", "Gravedad (leve/grave/muerte) con dato", len(df),
                       int((df["gravedad"] != SI).sum()), "Leve: 0-30 días de incapacidad; grave: > 30 días"))

    causas = {c: categorizar_causa(c) for c in df["circunstancia"].astype(str).unique()}
    df["causa_categoria"] = df["circunstancia"].astype(str).map(causas)
    pd.Series(causas, name="causa_categoria").rename_axis("circunstancia").sort_index().to_csv(
        config.ARCHIVO_CATALOGO_CAUSAS, encoding="utf-8")
    pasos.append(_paso("Medicina Legal", "Enriquecimiento", "Causas: valores distintos -> categorías",
                       df["circunstancia"].nunique(), df["causa_categoria"].nunique(),
                       "Catálogo aplicado en data/reference/catalogo_causas.csv"))

    df["medio_grupo"] = mapear_por_reglas(df["medio_transporte"].astype(str), REGLAS_MEDIO, "Otros medios")
    df["objeto_grupo"] = mapear_por_reglas(df["objeto_colision"].astype(str), REGLAS_OBJETO, "Otros")
    pasos.append(_paso("Medicina Legal", "Enriquecimiento", "Medio y objeto de colisión agrupados (categorías)",
                       df["medio_transporte"].nunique() + df["objeto_colision"].nunique(),
                       df["medio_grupo"].nunique() + df["objeto_grupo"].nunique()))

    ciclo = {"Primera Infancia": "Menores (0-11)", "Infancia": "Menores (0-11)", "Adolescencia": "Adolescentes (12-17)",
             "Juventud": "Adultos (18 y más)", "Adultez": "Adultos (18 y más)", "Adulto Mayor": "Adultos (18 y más)"}
    df["grupo_edad_comun"] = mapear_unicos(df["ciclo_vital"].astype(str),
                                           lambda v: next((g for k, g in ciclo.items() if k.lower() in v.lower()), SI))
    df["adulto_mayor"] = df["ciclo_vital"].astype(str).str.contains("Adulto Mayor").astype("int8")

    df["franja_horaria"] = mapear_unicos(df["rango_hora"].astype(str), _franja)
    df["fin_de_semana"] = df["dia_semana"].isin(["Sábado", "Domingo"]).astype("int8")
    df["trimestre"] = np.where(df["mes"] > 0, (df["mes"] - 1) // 3 + 1, 0).astype("int8")
    escenario = {"Calle urbana": r"calle", "Carretera": r"carretera", "Vía pública (sin especificar)": r"via publica"}
    df["escenario_grupo"] = mapear_por_reglas(df["escenario"].astype(str), list(escenario.items()), "Otros lugares")

    regiones = pd.read_csv(config.ARCHIVO_REGIONES, dtype=str)[["codigo_dane_departamento", "region"]]
    df = df.merge(regiones, on="codigo_dane_departamento", how="left")
    df["region"] = df["region"].fillna(SI)
    pasos.append(_paso("Medicina Legal", "Enriquecimiento", "Cruce con regiones naturales (filas con región)", len(df),
                       int((df["region"] != SI).sum())))

    contexto = contexto_policial(policia_todos_los_anios)
    df = df.merge(contexto, on=["codigo_dane_municipio", "anio"], how="left")
    for col in ["lesionados_policia_anio_ant", "muertes_policia_anio_ant"]:
        df[col] = df[col].fillna(0).astype("int32")
    pasos.append(_paso("Medicina Legal", "Enriquecimiento", "Contexto policial del municipio (año anterior)", len(df),
                       int((df["lesionados_policia_anio_ant"] + df["muertes_policia_anio_ant"] > 0).sum()),
                       "Filas cuyo municipio tuvo víctimas policiales el año previo"))

    for col in ["desenlace", "gravedad", "causa_categoria", "medio_grupo", "objeto_grupo", "grupo_edad_comun",
                "franja_horaria", "escenario_grupo", "region", "codigo_dane_municipio", "codigo_dane_departamento",
                "municipio", "departamento", "dia_semana"]:
        df[col] = df[col].astype("category")
    pasos.append(_paso("Medicina Legal", "Enriquecimiento", "Columnas de la base analítica", columnas, df.shape[1]))
    return df, pasos


# ---------------------------------------------------------------------------
# Unificación de las 4 fuentes (esquema común)
# ---------------------------------------------------------------------------
COLUMNAS_UNIFICADAS = ["fuente", "archivo_origen", "desenlace", "gravedad", "anio", "mes", "dia_semana",
                       "codigo_dane_departamento", "departamento", "region", "codigo_dane_municipio", "municipio",
                       "sexo", "grupo_edad_comun", "medio_grupo", "medio_incoherente"]


def unificar(inml: pd.DataFrame, policia: pd.DataFrame) -> pd.DataFrame:
    a = inml.assign(fuente="Medicina Legal", medio_incoherente=0)
    regiones = pd.read_csv(config.ARCHIVO_REGIONES, dtype=str)[["codigo_dane_departamento", "region"]]
    b = policia.merge(regiones, on="codigo_dane_departamento", how="left").assign(
        fuente="Policía Nacional", gravedad=lambda d: np.where(d["desenlace"] == "Muerte", "Muerte", "Lesión (sin clasificar)"))
    b["region"] = b["region"].fillna(SI)
    tabla = pd.concat([a[COLUMNAS_UNIFICADAS].astype({c: str for c in COLUMNAS_UNIFICADAS if c not in
                                                      ("anio", "mes", "medio_incoherente")}),
                       b[COLUMNAS_UNIFICADAS]], ignore_index=True)
    for col in COLUMNAS_UNIFICADAS:
        if col in ("anio",):
            tabla[col] = tabla[col].astype("int16")
        elif col in ("mes", "medio_incoherente"):
            tabla[col] = tabla[col].astype("int8")
        else:
            tabla[col] = tabla[col].astype("category")
    return tabla


# ---------------------------------------------------------------------------
# Técnica 3: Agregación (cubos OLAP)
# ---------------------------------------------------------------------------
def construir_cubos_olap(unificado: pd.DataFrame, inml: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def cubo(df, dims, medida="victimas"):
        return df.groupby(dims, observed=True).size().rename(medida).reset_index()

    geo = cubo(unificado, ["anio", "region", "departamento", "fuente", "desenlace"])
    tiempo = cubo(unificado, ["anio", "mes", "dia_semana", "fuente", "desenlace"])
    perfil = cubo(unificado, ["anio", "fuente", "desenlace", "sexo", "grupo_edad_comun", "medio_grupo"])

    gravedad = (inml.groupby(["anio", "condicion_victima", "medio_grupo", "clase_accidente", "objeto_grupo",
                              "zona", "gravedad"], observed=True).size().rename("victimas").reset_index())
    con_causa = inml[~inml["causa_categoria"].isin([SI])]
    causas = (con_causa.groupby(["anio", "causa_categoria", "desenlace", "gravedad"], observed=True).size()
              .rename("victimas").reset_index())

    cobertura = (unificado.groupby(["anio", "departamento", "desenlace", "fuente"], observed=True).size()
                 .unstack("fuente", fill_value=0).reset_index())
    cobertura["razon_policia_vs_medicina_legal"] = (cobertura["Policía Nacional"]
                                                    / cobertura["Medicina Legal"].replace(0, np.nan)).round(3)
    return {"cubo_geografico": geo, "cubo_temporal": tiempo, "cubo_perfil_victima": perfil,
            "cubo_gravedad_medicina_legal": gravedad, "cubo_causas": causas, "cobertura_entre_fuentes": cobertura}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def guardar_resultados(inml: pd.DataFrame, unificado: pd.DataFrame, cubos: dict, resumen: dict) -> None:
    config.crear_directorios()
    inml.to_parquet(config.ARCHIVO_DETALLE, index=False, compression="snappy")
    unificado.to_parquet(config.ARCHIVO_UNIFICADO, index=False, compression="snappy")
    for nombre, cubo in cubos.items():
        cubo.to_parquet(config.DIR_AGREGACIONES / f"{nombre}.parquet", index=False)
    config.ARCHIVO_RESUMEN_ETL.write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")


def ejecutar_etl() -> dict:
    inicio = time.perf_counter()
    crudos = cargar_fuentes()
    perfil = perfil_fuentes(crudos)
    problemas = diagnosticar_calidad(crudos)

    inml_limpio, p1 = limpiar_inml(crudos["inml_lesiones"], crudos["inml_muertes"])
    inml_normal, p2 = normalizar_inml(inml_limpio)

    # La Policía se limpia sin filtrar años para poder calcular el contexto del año anterior (2014 -> 2015).
    policia_ext, _ = limpiar_policia(crudos["policia_lesiones"], crudos["policia_homicidios_transito"],
                                     anio_inicio=config.ANIO_INICIO - 1)
    nombres = inml_normal[["codigo_dane_municipio", "municipio", "codigo_dane_departamento", "departamento"]].astype(str)
    policia_ext, _ = normalizar_policia(policia_ext, nombres)

    policia_limpio, p3 = limpiar_policia(crudos["policia_lesiones"], crudos["policia_homicidios_transito"])
    policia, p4 = normalizar_policia(policia_limpio, nombres)

    inml, p5 = enriquecer_inml(inml_normal, policia_ext)
    unificado = unificar(inml, policia)
    cubos = construir_cubos_olap(unificado, inml)

    resumen = {
        "fuentes": perfil.reset_index().to_dict(orient="records"),
        "filas_crudas": int(perfil["filas"].sum()),
        "filas_unificadas": len(unificado),
        "filas_medicina_legal": len(inml),
        "filas_policia_victimas": len(policia),
        "problemas_detectados": problemas.to_dict(orient="records"),
        "pasos_etl": p1 + p2 + p3 + p4 + p5,
        "agregaciones": {n: len(c) for n, c in cubos.items()},
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    guardar_resultados(inml, unificado, cubos, resumen)
    return resumen


if __name__ == "__main__":
    r = ejecutar_etl()
    print(f"ETL terminado en {r['segundos']} s")
    print(f"  Filas crudas (4 archivos): {r['filas_crudas']:,}")
    print(f"  Base unificada: {r['filas_unificadas']:,} víctimas | Medicina Legal: {r['filas_medicina_legal']:,}"
          f" | Policía: {r['filas_policia_victimas']:,}")
    for p in r["pasos_etl"]:
        print(f"  [{p['fuente']} · {p['tecnica']}] {p['etapa']}: {p['antes']:,} -> {p['despues']:,}")
