"""
ETL del dataset de quejas financieras (CRISP-DM: Data Understanding + Data Preparation).

El módulo expone una función por cada etapa para que el notebook pueda ejecutarlas
paso a paso y mostrar los conteos antes/después. `ejecutar_etl()` corre todo el
proceso de punta a punta (lo usa el contenedor de procesamiento).

Técnicas ETL aplicadas:
  1. Limpieza        -> nombres de columnas, nulos, valores inválidos, duplicados exactos
  2. Normalización   -> textos (mayúsculas/tildes/puntuación), tipos de datos, escala log
  3. Agregación      -> consolidación de registros equivalentes + cubos OLAP
  4. Enriquecimiento -> catálogo de tipos de entidad SFC, regiones/código DANE,
                        categoría de motivo y variables derivadas para el modelo
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

# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------
MAPA_COLUMNAS = {
    "Año_Creacion": "anio",
    "Mes_Creacion": "mes",
    "Tipo_Entidad": "tipo_entidad",
    "Codigo_Entidad": "codigo_entidad",
    "Instancia_Recepcion": "instancia_recepcion",
    "Motivo": "motivo",
    "Producto": "producto",
    "Departamento": "departamento",
    "Municipio": "municipio",
    "Cantidad_quejas_recibidas": "cantidad_quejas",
}
COLUMNAS_TEXTO = ["instancia_recepcion", "motivo", "producto", "departamento", "municipio"]
COLUMNAS_DIMENSION = ["anio", "mes", "tipo_entidad", "codigo_entidad", "instancia_recepcion",
                      "motivo", "producto", "departamento", "municipio"]

# Sinónimos que la normalización automática no detecta (revisión manual del catálogo).
SINONIMOS_MOTIVO = {
    "OTROS": "OTROS MOTIVOS",
    "DISTRIBUCION DE PORTAFOLIOS": "DISTRIBUCION DE PORTAFOLIO",
}

# Reglas (en orden de prioridad) para agrupar los 300+ motivos en categorías de negocio.
REGLAS_CATEGORIA_MOTIVO = [
    ("Fraude y transacciones no reconocidas",
     r"NO RECONOCID|SUPLANTACI|FRAUD|FALSIFIC|FALSOS|SEGURIDAD EN CANALES|SIN ORDEN DEL CLIENTE"),
    ("Pensiones y cesantías",
     r"PENSION|APORTE|HISTORIA LABORAL|AFILIACI|BONO|MESADA|CESANT|BEPS|INVALIDEZ|INCAPACIDAD"
     r"|PRESTACION|FUNERARIO|PSAP|SEMANAS|COLOMBIA MAYOR|DOBLE ASESOR"),
    ("Seguros y siniestros", r"SINIESTRO|INDEMNIZACI|POLIZA|PRIMA|SEGURO|DICTAMEN|DIAGNOSTICO|AUTORIZACION DE SERVICIOS"),
    ("Crédito, cobranza y centrales de riesgo",
     r"CENTRALES DE RIESGO|COBRANZA|EMBARGO|NORMALIZACION DE CREDITO|ACUERDOS DE PAGO|GARANTIA|PAZ Y SALVO"
     r"|REPORTE DE PAGO|DACION|CREDITO|PAGO ANTICIPADO"),
    ("Cobros, tarifas y tasas",
     r"COBRO|COMISION|GASTOS|TASA|TARIFA|IMPUESTO|FACTURA|PENALI|DESCUENTO"),
    ("Errores en pagos y transacciones",
     r"TRANSACCI|PAGO|TRANSFERENCIA|DEBITO AUTOMATICO|RECAUDO|CHEQUE|REMESA|MONEDA|MONETIZACI|CAJERO|BILLETES|LLAVE"),
    ("Canales y calidad del servicio",
     r"CANAL|ATENCION|MAL TRATO|COMUNICACION|SERVICIO|PROTOCOLO|QUEJAS|RECLAMOS|PETICIONES|ASESOR COMERCIAL|FUNCIONARIO"),
    ("Información, asesoría y datos",
     r"INFORMACION|ASESORIA|EXTRACTO|PUBLICIDAD|CERTIFICA|DATOS PERSONALES|INFORME|RESERVA|CLAUSULA|CONTRATO"),
    ("Gestión de productos y vinculación",
     r"VINCULACI|BLOQUEO|APERTURA|CANCELACI|TARJETA|SALDO|LIQUIDACION|PRODUCTO|DEPOSITO|TRASLADO|RETIRO|PORTAFOLIO"
     r"|RENTABILIDAD|RENDIMIENTO|SARLAFT|VALORES|TITULO|DIVIDENDO|INVERSION|ALMACEN|MERCANCIA"),
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def clave_texto(texto: str) -> str:
    """Clave de comparación: sin tildes, mayúsculas, sin puntuación final ni espacios repetidos."""
    if pd.isna(texto):
        return texto
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    limpio = re.sub(r"\s+", " ", sin_tildes).strip().upper()
    return re.sub(r"[.\s]+$", "", limpio)


def memoria_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1e6, 1)


def meses_faltantes_e_incompletos(df: pd.DataFrame, umbral: float = 0.2) -> tuple[list[str], list[str]]:
    """Detecta meses sin datos y meses cuyo volumen es < `umbral` × la mediana mensual."""
    mensual = df.groupby(["anio", "mes"])["cantidad_quejas"].sum()
    etiqueta = lambda a, m: f"{a}-{m:02d}"  # noqa: E731
    presentes = {etiqueta(a, m) for a, m in mensual.index}
    anio_min, anio_max = int(df["anio"].min()), int(df["anio"].max())
    ultimo_mes = int(df.loc[df["anio"] == anio_max, "mes"].max())
    esperados = [etiqueta(a, m) for a in range(anio_min, anio_max + 1) for m in range(1, 13)
                 if not (a == anio_max and m > ultimo_mes)]
    faltantes = [p for p in esperados if p not in presentes]
    incompletos = [etiqueta(a, m) for (a, m), v in mensual.items() if v < umbral * mensual.median()]
    return faltantes, incompletos


def _registro(etapa: str, tecnica: str, antes: int, despues: int, detalle: str) -> dict:
    return {"etapa": etapa, "tecnica": tecnica, "antes": int(antes), "despues": int(despues),
            "diferencia": int(despues) - int(antes), "detalle": detalle}


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------
def cargar_crudo(ruta: Path = config.ARCHIVO_RAW, optimizado: bool = False) -> pd.DataFrame:
    """Lee el CSV crudo. Con `optimizado=True` usa tipos category/enteros pequeños."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecute primero: python src/descargar_datos.py")
    if not optimizado:
        return pd.read_csv(ruta, encoding="utf-8-sig")
    tipos = {"Año_Creacion": "int16", "Mes_Creacion": "int8", "Tipo_Entidad": "int16",
             "Codigo_Entidad": "int16", "Cantidad_quejas_recibidas": "int32",
             "Instancia_Recepcion": "category", "Motivo": "category", "Producto": "category",
             "Departamento": "category", "Municipio": "category"}
    return pd.read_csv(ruta, encoding="utf-8-sig", dtype=tipos)


# ---------------------------------------------------------------------------
# Data Understanding
# ---------------------------------------------------------------------------
def perfil_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos, % nulos y valores únicos por columna."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "pct_nulos": (df.isna().mean() * 100).round(2),
        "valores_unicos": df.nunique(),
    })


def diagnosticar_calidad(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de problemas de calidad a partir de mediciones sobre los datos crudos."""
    df = df_crudo.rename(columns=MAPA_COLUMNAS)
    problemas = []

    columnas_problema = [c for c in df_crudo.columns if c != c.lower() or not c.isascii()]
    problemas.append(("Nombres de columna", "Mayúsculas, tildes y 'ñ' en encabezados; archivo con BOM",
                      len(columnas_problema), "Limpieza: renombrar a snake_case y leer con utf-8-sig"))

    nulos_ubicacion = int(df["departamento"].isna().sum())
    problemas.append(("Nulos", "Registros sin departamento ni municipio", nulos_ubicacion,
                      f"Limpieza: imputar '{config.VALOR_SIN_UBICACION}' (no se eliminan: 44 % escaló a la SFC)"))

    motivos = df["motivo"].dropna().unique()
    motivos_variantes = len(motivos) - len({clave_texto(m) for m in motivos})
    problemas.append(("Inconsistencia de texto", "Motivos repetidos que solo cambian en mayúsculas, tildes o puntuación",
                      motivos_variantes, "Normalización: clave sin tildes y etiqueta canónica más frecuente"))

    filas_mayus = int(df["motivo"].str.fullmatch(r"[^a-z]+").sum())
    problemas.append(("Catálogo mixto", "Filas con motivos de un catálogo distinto escrito en MAYÚSCULAS sostenidas",
                      filas_mayus, "Normalización: unificar a la etiqueta canónica"))

    codigos_repetidos = int((df.groupby("codigo_entidad")["tipo_entidad"].nunique() > 1).sum())
    problemas.append(("Llave ambigua", "Códigos de entidad que se repiten en varios tipos de entidad",
                      codigos_repetidos, "Enriquecimiento: llave compuesta id_entidad = tipo-código"))

    municipios = df[["departamento", "municipio"]].dropna().drop_duplicates()
    homonimos = int(municipios.assign(k=municipios["municipio"].map(clave_texto))
                    .groupby("k")["departamento"].nunique().gt(1).sum())
    problemas.append(("Llave ambigua", "Nombres de municipio que existen en más de un departamento",
                      homonimos, "Enriquecimiento: llave compuesta departamento|municipio"))

    sin_nombre = int(df["tipo_entidad"].nunique())
    problemas.append(("Códigos sin descripción", "Tipos de entidad solo como número", sin_nombre,
                      "Enriquecimiento: cruce con catálogo de tipos de entidad SFC"))

    q99 = df["cantidad_quejas"].quantile(0.99)
    atipicos = int((df["cantidad_quejas"] > q99).sum())
    problemas.append(("Distribución sesgada", f"Cantidad de quejas > p99 ({q99:,.0f}); máximo {df['cantidad_quejas'].max():,}",
                      atipicos, "Normalización: log1p y bandera de volumen atípico (no se eliminan, son reales)"))

    faltantes, incompletos = meses_faltantes_e_incompletos(df)
    problemas.append(("Meses faltantes", f"Meses sin ningún registro: {', '.join(faltantes) or 'ninguno'}",
                      len(faltantes), "Análisis: no interpolar; comparar años con promedios mensuales"))
    filas_incompletas = int(df.assign(p=df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2))
                            ["p"].isin(incompletos).sum())
    problemas.append(("Meses con carga parcial", f"Meses con < 20 % del volumen mediano: {', '.join(incompletos) or 'ninguno'}",
                      filas_incompletas, "Limpieza: bandera mes_incompleto; se excluyen de series y del modelo"))

    invalidos = int((~df["mes"].between(1, 12)).sum() + (df["cantidad_quejas"] <= 0).sum())
    problemas.append(("Valores inválidos", "Mes fuera de 1-12 o cantidad ≤ 0", invalidos,
                      "Limpieza: regla de validación (se eliminarían si aparecen)"))

    problemas.append(("Duplicados exactos", "Filas idénticas en todas las columnas", int(df.duplicated().sum()),
                      "Limpieza: eliminar duplicados exactos"))

    problemas.append(("Tipos ineficientes", "Textos como object y enteros de 64 bits", int(memoria_mb(df_crudo)),
                      "Normalización: category + enteros pequeños (valor = MB en memoria)"))

    return pd.DataFrame(problemas, columns=["categoria", "problema", "registros_afectados", "tratamiento"])


# ---------------------------------------------------------------------------
# Técnica 1: Limpieza
# ---------------------------------------------------------------------------
def limpiar(df_crudo: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    registros = []
    filas_inicio = len(df_crudo)

    df = df_crudo.rename(columns=MAPA_COLUMNAS).copy()
    registros.append(_registro("Renombrar columnas", "Limpieza", len(df_crudo.columns), len(df.columns),
                               "Encabezados a snake_case sin tildes ni 'ñ'"))

    for columna in COLUMNAS_TEXTO:
        df[columna] = df[columna].astype("string").str.strip()

    nulos_antes = int(df[["departamento", "municipio"]].isna().any(axis=1).sum())
    df["sin_ubicacion"] = df["departamento"].isna().astype("int8")
    df[["departamento", "municipio"]] = df[["departamento", "municipio"]].fillna(config.VALOR_SIN_UBICACION)
    registros.append(_registro("Imputar ubicación faltante", "Limpieza", nulos_antes,
                               int(df[["departamento", "municipio"]].isna().any(axis=1).sum()),
                               f"Nulos -> '{config.VALOR_SIN_UBICACION}' + bandera sin_ubicacion"))

    validos = df["mes"].between(1, 12) & (df["cantidad_quejas"] > 0) & df["anio"].between(2000, 2100)
    filas_antes = len(df)
    df = df[validos]
    registros.append(_registro("Eliminar valores inválidos", "Limpieza", filas_antes, len(df),
                               "Mes en 1-12, año plausible y cantidad > 0"))

    filas_antes = len(df)
    df = df.drop_duplicates()
    registros.append(_registro("Eliminar duplicados exactos", "Limpieza", filas_antes, len(df),
                               "Filas idénticas en todas las columnas"))

    _, incompletos = meses_faltantes_e_incompletos(df)
    etiqueta_periodo = df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2)
    df["mes_incompleto"] = etiqueta_periodo.isin(incompletos).astype("int8")
    registros.append(_registro("Marcar meses con carga parcial", "Limpieza", 0, int(df["mes_incompleto"].sum()),
                               f"Filas en meses incompletos ({', '.join(incompletos)}); se conservan con bandera"))

    registros.append(_registro("Total limpieza (filas)", "Limpieza", filas_inicio, len(df), ""))
    return df.reset_index(drop=True), registros


# ---------------------------------------------------------------------------
# Técnica 2: Normalización
# ---------------------------------------------------------------------------
def _etiquetas_canonicas(serie: pd.Series, pesos: pd.Series, sinonimos: dict | None = None,
                         grupo: pd.Series | None = None) -> pd.Series:
    """Para cada clave normalizada elige la escritura original con más quejas.

    `grupo` limita la unificación dentro de otra columna (ej. municipio dentro de su
    departamento), para no fusionar homónimos como Chimá (Córdoba) y Chima (Santander).
    """
    claves = serie.map(clave_texto)
    if sinonimos:
        claves = claves.replace(sinonimos)
    if grupo is not None:
        claves = grupo.astype(str) + "||" + claves
    tabla = (pd.DataFrame({"original": serie, "clave": claves, "peso": pesos})
             .groupby(["clave", "original"], observed=True)["peso"].sum().reset_index()
             .sort_values("peso", ascending=False).drop_duplicates("clave"))
    canonica = dict(zip(tabla["clave"], tabla["original"]))
    return claves.map(canonica)


def normalizar(df_limpio: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    registros = []
    df = df_limpio.copy()
    memoria_antes = memoria_mb(df)

    for columna in ["motivo", "producto", "municipio", "instancia_recepcion"]:
        unicos_antes = df[columna].nunique()
        sinonimos = SINONIMOS_MOTIVO if columna == "motivo" else None
        grupo = df["departamento"] if columna == "municipio" else None
        df[columna] = _etiquetas_canonicas(df[columna], df["cantidad_quejas"], sinonimos, grupo)
        registros.append(_registro(f"Unificar escritura de {columna}", "Normalización",
                                   unicos_antes, df[columna].nunique(),
                                   "Valores únicos antes/después (tildes, mayúsculas, puntuación)"))

    # Corrección ortográfica puntual en productos ("Creditos de consumo" -> "Créditos de consumo").
    filas_sin_tilde = int(df["producto"].str.contains(r"^Creditos\b", regex=True).sum())
    df["producto"] = df["producto"].str.replace(r"^Creditos\b", "Créditos", regex=True)
    registros.append(_registro("Corregir tildes en producto", "Normalización", filas_sin_tilde,
                               int(df["producto"].str.contains(r"^Creditos\b", regex=True).sum()),
                               "Filas con 'Creditos' sin tilde antes/después"))

    # Escala logarítmica para la cantidad de quejas (distribución muy sesgada).
    df["log_cantidad_quejas"] = np.log1p(df["cantidad_quejas"]).astype("float32")
    umbral_atipico = df["cantidad_quejas"].quantile(0.99)
    df["volumen_atipico"] = (df["cantidad_quejas"] > umbral_atipico).astype("int8")
    registros.append(_registro("Escala log y bandera de atípicos", "Normalización",
                               0, int(df["volumen_atipico"].sum()),
                               f"log1p(cantidad); atípico si cantidad > p99 = {umbral_atipico:,.0f}"))

    # Tipos eficientes.
    df = df.astype({"anio": "int16", "mes": "int8", "tipo_entidad": "int16",
                    "codigo_entidad": "int16", "cantidad_quejas": "int32"})
    for columna in COLUMNAS_TEXTO:
        df[columna] = df[columna].astype("category")
    registros.append(_registro("Optimizar tipos (MB en memoria)", "Normalización",
                               round(memoria_antes), round(memoria_mb(df)),
                               "category para textos y enteros de 8/16/32 bits"))
    return df, registros


# ---------------------------------------------------------------------------
# Técnica 3: Agregación
# ---------------------------------------------------------------------------
def consolidar(df_normalizado: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Tras unificar textos, varias filas describen la misma combinación: se suman sus quejas."""
    filas_antes = len(df_normalizado)
    quejas_antes = int(df_normalizado["cantidad_quejas"].sum())
    df = (df_normalizado
          .groupby(COLUMNAS_DIMENSION, observed=True, as_index=False)
          .agg(cantidad_quejas=("cantidad_quejas", "sum"), sin_ubicacion=("sin_ubicacion", "max"),
               mes_incompleto=("mes_incompleto", "max")))
    df["cantidad_quejas"] = df["cantidad_quejas"].astype("int32")
    df["log_cantidad_quejas"] = np.log1p(df["cantidad_quejas"]).astype("float32")
    umbral = df_normalizado["cantidad_quejas"].quantile(0.99)
    df["volumen_atipico"] = (df["cantidad_quejas"] > umbral).astype("int8")
    for columna in COLUMNAS_TEXTO:
        df[columna] = df[columna].astype("category")
    registros = [
        _registro("Consolidar registros equivalentes", "Agregación", filas_antes, len(df),
                  "Filas con las mismas 9 dimensiones tras normalizar"),
        _registro("Control: total de quejas", "Agregación", quejas_antes, int(df["cantidad_quejas"].sum()),
                  "Debe ser igual: la consolidación no pierde quejas"),
    ]
    return df, registros


def _cubo(df: pd.DataFrame, dimensiones: list[str]) -> pd.DataFrame:
    cubo = (df.groupby(dimensiones, observed=True)
            .agg(registros=("cantidad_quejas", "size"),
                 total_quejas=("cantidad_quejas", "sum"),
                 quejas_escaladas=("quejas_escaladas", "sum"))
            .reset_index())
    cubo["tasa_escalamiento"] = (cubo["quejas_escaladas"] / cubo["total_quejas"]).round(4)
    return cubo.sort_values("total_quejas", ascending=False).reset_index(drop=True)


def construir_cubos_olap(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agregaciones multidimensionales que luego sirven de filtros en el dashboard."""
    base = df.assign(quejas_escaladas=df["cantidad_quejas"] * df["escalada"])
    return {
        "cubo_anio_region_departamento_producto": _cubo(base, ["anio", "region", "departamento", "producto"]),
        "cubo_anio_sector_instancia_categoria": _cubo(base, ["anio", "sector", "instancia_recepcion", "categoria_motivo"]),
        "serie_mensual_sector": _cubo(base[base["mes_incompleto"] == 0], ["periodo", "sector"])
        .sort_values(["periodo", "sector"]).reset_index(drop=True),
        "ranking_entidades": _cubo(base, ["id_entidad", "nombre_tipo_entidad", "sector"]),
    }


# ---------------------------------------------------------------------------
# Técnica 4: Enriquecimiento
# ---------------------------------------------------------------------------
def categorizar_motivo(motivo: str) -> str:
    clave = clave_texto(motivo)
    for categoria, patron in REGLAS_CATEGORIA_MOTIVO:
        if re.search(patron, clave):
            return categoria
    return "Otros motivos"


def enriquecer(df_consolidado: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    registros = []
    df = df_consolidado.copy()
    columnas_antes = df.shape[1]

    # Catálogo de tipos de entidad.
    tipos = pd.read_csv(config.ARCHIVO_TIPOS_ENTIDAD)
    df = df.merge(tipos.rename(columns={"codigo_tipo_entidad": "tipo_entidad"}), on="tipo_entidad", how="left")
    sin_cruce = int(df["nombre_tipo_entidad"].isna().sum())
    df[["nombre_tipo_entidad", "sector"]] = df[["nombre_tipo_entidad", "sector"]].fillna("Sin clasificar")
    registros.append(_registro("Cruce con catálogo de tipos de entidad", "Enriquecimiento",
                               len(df), len(df) - sin_cruce, "Filas con nombre y sector de entidad"))

    # Regiones naturales y código DANE.
    regiones = pd.read_csv(config.ARCHIVO_REGIONES, dtype={"codigo_dane_departamento": "string"})
    df["departamento"] = df["departamento"].astype("string")
    df = df.merge(regiones, on="departamento", how="left")
    df["region"] = df["region"].fillna(config.VALOR_SIN_UBICACION)
    registros.append(_registro("Cruce con regiones naturales (DANE)", "Enriquecimiento", len(df),
                               int((df["region"] != config.VALOR_SIN_UBICACION).sum()),
                               "Filas con región asignada (el resto no tiene ubicación)"))

    # Llaves compuestas para resolver ambigüedades.
    df["id_entidad"] = df["tipo_entidad"].astype(str) + "-" + df["codigo_entidad"].astype(str)
    df["id_municipio"] = df["departamento"] + " | " + df["municipio"].astype("string")

    # Variables derivadas.
    df["periodo"] = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1))
    df["trimestre"] = ((df["mes"] - 1) // 3 + 1).astype("int8")
    categorias = {m: categorizar_motivo(m) for m in df["motivo"].astype(str).unique()}
    df["categoria_motivo"] = df["motivo"].astype(str).map(categorias)
    df["escalada"] = (df["instancia_recepcion"].astype(str) != config.INSTANCIA_NO_ESCALADA).astype("int8")

    for columna in ["departamento", "municipio", "region", "nombre_tipo_entidad", "sector", "fuente_nombre",
                    "id_entidad", "id_municipio", "categoria_motivo", "codigo_dane_departamento"]:
        df[columna] = df[columna].astype("category")

    registros.append(_registro("Columnas agregadas", "Enriquecimiento", columnas_antes, df.shape[1],
                               "nombre/sector de entidad, región, DANE, llaves, periodo, trimestre, "
                               "categoría de motivo y variable objetivo 'escalada'"))
    return df, registros


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
COLUMNAS_SALIDA = [
    "periodo", "anio", "mes", "trimestre", "mes_incompleto",
    "tipo_entidad", "nombre_tipo_entidad", "sector", "fuente_nombre", "codigo_entidad", "id_entidad",
    "instancia_recepcion", "escalada",
    "motivo", "categoria_motivo", "producto",
    "region", "codigo_dane_departamento", "departamento", "municipio", "id_municipio", "sin_ubicacion",
    "cantidad_quejas", "log_cantidad_quejas", "volumen_atipico",
]


def guardar_resultados(df: pd.DataFrame, cubos: dict[str, pd.DataFrame], resumen: dict) -> None:
    config.crear_directorios()
    df[COLUMNAS_SALIDA].to_parquet(config.ARCHIVO_PARQUET_LIMPIO, index=False, compression="snappy")
    for nombre, cubo in cubos.items():
        cubo.to_parquet(config.DIR_AGREGACIONES / f"{nombre}.parquet", index=False)
    config.ARCHIVO_RESUMEN_ETL.write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")


def ejecutar_etl() -> dict:
    inicio = time.perf_counter()
    df_crudo = cargar_crudo()
    problemas = diagnosticar_calidad(df_crudo)

    df_limpio, reg_limpieza = limpiar(df_crudo)
    df_normal, reg_normal = normalizar(df_limpio)
    df_consol, reg_consol = consolidar(df_normal)
    df_final, reg_enriq = enriquecer(df_consol)
    cubos = construir_cubos_olap(df_final)

    resumen = {
        "filas_crudas": len(df_crudo),
        "columnas_crudas": df_crudo.shape[1],
        "filas_finales": len(df_final),
        "columnas_finales": len(COLUMNAS_SALIDA),
        "registros_eliminados": len(df_crudo) - len(df_limpio),
        "registros_consolidados": len(df_normal) - len(df_consol),
        "total_quejas": int(df_final["cantidad_quejas"].sum()),
        "problemas_detectados": problemas.to_dict(orient="records"),
        "pasos_etl": reg_limpieza + reg_normal + reg_consol + reg_enriq,
        "agregaciones": {n: len(c) for n, c in cubos.items()},
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    guardar_resultados(df_final, cubos, resumen)
    return resumen


if __name__ == "__main__":
    resultado = ejecutar_etl()
    print(f"ETL terminado en {resultado['segundos']} s")
    print(f"  {resultado['filas_crudas']:,} filas crudas -> {resultado['filas_finales']:,} filas finales")
    print(f"  Parquet: {config.ARCHIVO_PARQUET_LIMPIO.relative_to(config.RAIZ_PROYECTO)}")
    for paso in resultado["pasos_etl"]:
        print(f"  [{paso['tecnica']}] {paso['etapa']}: {paso['antes']:,} -> {paso['despues']:,}")
