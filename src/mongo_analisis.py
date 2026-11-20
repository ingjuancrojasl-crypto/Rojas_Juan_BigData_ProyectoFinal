"""Sección 6: documento de análisis en MongoDB.

Construye un documento JSON con la estructura de la Actividad 1 ampliada con los resultados del modelo
(metadata, calidad_datos, estadisticas_descriptivas, agregaciones, modelo, insights,
visualizaciones_generadas, recomendaciones), lo guarda en MongoDB y ofrece consultas de ejemplo.

Todo se calcula a partir de los archivos que dejan el ETL (data/processed), el modelo (models/) y las
visualizaciones (reports/figuras), de modo que el módulo también funciona fuera del notebook (Docker, dashboard).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SI = config.SIN_INFORMACION
ARCHIVO_DOCUMENTO = config.DIR_REPORTES / "documento_mongodb.json"
ARCHIVO_VISUALIZACIONES = config.DIR_REPORTES / "figuras" / "visualizaciones.json"
AUTORES = ["Juan Rojas"]
PREGUNTA_NEGOCIO = ("¿Qué características de la víctima, del vehículo y del contexto se asocian con la gravedad de un "
                    "siniestro de transporte (lesión leve, lesión grave o muerte), y cuáles son las causas más "
                    "frecuentes en lesionados y fallecidos?")

# Etapas del ETL que eliminan filas (las demás cambian columnas, valores o celdas)
_ETAPAS_QUE_ELIMINAN = {
    "Eliminar duplicados": "Registros duplicados (Medicina Legal)",
    "Filtrar conducta de tránsito": "Denuncias que no son de tránsito (Policía)",
    "Filtrar periodo": "Registros fuera de 2015-2024 (Policía)",
    "Municipio y departamento": "Municipios sin código DANE válido (Policía)",
}


def _registros(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista de diccionarios con tipos nativos de Python (MongoDB no acepta tipos de numpy)."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _pct(parte: float, total: float) -> float:
    return round(float(parte) / float(total) * 100, 1) if total else 0.0


def _miles(valor: float) -> str:
    return f"{int(valor):,}".replace(",", ".")


def _dec(valor: float, decimales: int = 1) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def _letalidad(df: pd.DataFrame, columna: str) -> pd.Series:
    """Porcentaje de víctimas (con gravedad conocida) que fallecen, por categoría."""
    datos = df[(df["gravedad"].astype(str) != SI) & (df[columna].astype(str) != SI)]
    return datos.groupby(datos[columna].astype(str))["gravedad"].apply(lambda g: (g == "Muerte").mean() * 100).round(1)


# ---------------------------------------------------------------------------
# Carga de los resultados del pipeline
# ---------------------------------------------------------------------------
def cargar_insumos() -> dict:
    """Lee los resultados del ETL, del modelo y de las visualizaciones."""
    faltan = [str(r.relative_to(config.RAIZ_PROYECTO)) for r in
              (config.ARCHIVO_DETALLE, config.ARCHIVO_UNIFICADO, config.ARCHIVO_RESUMEN_ETL,
               config.DIR_MODELOS / "metadata_modelo.json", ARCHIVO_VISUALIZACIONES) if not r.exists()]
    if faltan:
        raise FileNotFoundError(f"Faltan resultados de secciones anteriores: {faltan}. Ejecute primero las secciones 3-5.")
    import modelo  # importación diferida: solo se necesita aquí
    modelo_ajustado, metadata_modelo = modelo.cargar_modelo()
    return {
        "victimas": pd.read_parquet(config.ARCHIVO_DETALLE),
        "unificado": pd.read_parquet(config.ARCHIVO_UNIFICADO),
        "resumen_etl": json.loads(config.ARCHIVO_RESUMEN_ETL.read_text(encoding="utf-8")),
        "metadata_modelo": metadata_modelo,
        "importancia": pd.read_csv(modelo.ARCHIVO_IMPORTANCIA),
        "perfiles": modelo.perfiles_de_riesgo(modelo_ajustado),
        "visualizaciones": json.loads(ARCHIVO_VISUALIZACIONES.read_text(encoding="utf-8")),
        "cubos": {r.stem: pd.read_parquet(r) for r in sorted(config.DIR_AGREGACIONES.glob("*.parquet"))},
    }


# ---------------------------------------------------------------------------
# Secciones del documento
# ---------------------------------------------------------------------------
def seccion_metadata(insumos: dict) -> dict:
    resumen, unificado = insumos["resumen_etl"], insumos["unificado"]
    fuentes = []
    for fila in resumen["fuentes"]:
        info = config.FUENTES[fila["fuente"]]
        fuentes.append({"nombre": fila["fuente"], "titulo": info["titulo"], "entidad": info["entidad"],
                        "id_datos_gov": info["id"], "url": info["pagina"], "registros": int(fila["filas"]),
                        "columnas": int(fila["columnas"]), "periodo": fila.get("periodo")})
    return {
        "fecha_analisis": datetime.now().isoformat(timespec="seconds"),
        "autores": AUTORES,
        "proyecto": "Gravedad de los siniestros de transporte en Colombia (2015-2024)",
        "curso": "Gerencia y almacenamiento en Big Data",
        "pregunta_negocio": PREGUNTA_NEGOCIO,
        "dataset": {
            "fuente": "datos.gov.co — Medicina Legal (INMLCF) y Policía Nacional: 4 datasets unificados",
            "url": "https://www.datos.gov.co",
            "registros": int(len(unificado)),
            "columnas": int(unificado.shape[1]),
            "registros_crudos": int(resumen["filas_crudas"]),
            "registros_medicina_legal": int(resumen["filas_medicina_legal"]),
            "registros_policia": int(resumen["filas_policia_victimas"]),
            "periodo": f"{config.ANIO_INICIO}-{config.ANIO_FIN}",
            "fuentes": fuentes,
        },
        "almacenamiento": {"formato": "Parquet (snappy)", "procesamiento": "pandas",
                           "archivos": [str(config.ARCHIVO_DETALLE.relative_to(config.RAIZ_PROYECTO)),
                                        str(config.ARCHIVO_UNIFICADO.relative_to(config.RAIZ_PROYECTO))]},
    }


def seccion_calidad(insumos: dict) -> dict:
    resumen = insumos["resumen_etl"]
    eliminados = []
    for paso in resumen["pasos_etl"]:
        for clave, descripcion in _ETAPAS_QUE_ELIMINAN.items():
            if paso["etapa"].startswith(clave) and paso["antes"] > paso["despues"]:
                eliminados.append({"motivo": descripcion, "registros": int(paso["antes"] - paso["despues"])})
    return {
        "problemas_detectados": resumen["problemas_detectados"],
        "total_problemas": len(resumen["problemas_detectados"]),
        "registros_eliminados": int(sum(e["registros"] for e in eliminados)),
        "detalle_eliminados": eliminados,
        "pasos_etl": resumen["pasos_etl"],
        "tecnicas_aplicadas": sorted({p["tecnica"] for p in resumen["pasos_etl"]}),
    }


def seccion_estadisticas(insumos: dict) -> dict:
    victimas, unificado = insumos["victimas"], insumos["unificado"]
    con_gravedad = victimas[victimas["gravedad"].astype(str) != SI]
    por_anio = (unificado.groupby(["anio", "fuente", "desenlace"], observed=True).size()
                .unstack(["fuente", "desenlace"]).fillna(0).astype(int))
    por_anio.columns = [f"{fuente} - {desenlace}" for fuente, desenlace in por_anio.columns]
    distribucion = lambda serie: {str(k): _pct(v, len(serie)) for k, v in serie.value_counts().items()}  # noqa: E731
    return {
        "victimas_por_fuente_y_desenlace": _registros(
            unificado.groupby(["fuente", "desenlace"], observed=True).size().rename("victimas").reset_index()),
        "victimas_por_anio": _registros(por_anio.reset_index()),
        "gravedad_medicina_legal_pct": distribucion(con_gravedad["gravedad"].astype(str)),
        "letalidad_general_pct": _pct((con_gravedad["gravedad"] == "Muerte").sum(), len(con_gravedad)),
        "sexo_pct_por_desenlace": {d: distribucion(g["sexo"].astype(str)) for d, g in
                                   victimas.groupby(victimas["desenlace"].astype(str))},
        "medio_transporte_pct_por_desenlace": {d: distribucion(g["medio_grupo"].astype(str)) for d, g in
                                               victimas.groupby(victimas["desenlace"].astype(str))},
        "dias_incapacidad_lesiones_pct": distribucion(
            victimas.loc[victimas["desenlace"] == "Lesión", "dias_incapacidad"].astype(str)),
        "victimas_por_grupo_edad": {str(k): int(v) for k, v in victimas["grupo_edad_comun"].value_counts().items()},
        # Lista de registros (no diccionario): nombres como "Bogotá, D.C." tienen puntos, que MongoDB no admite en claves
        "departamentos_con_mas_muertes": [{"departamento": str(k), "muertes": int(v)} for k, v in victimas.loc[
            victimas["desenlace"] == "Muerte", "departamento"].value_counts().head(10).items()],
    }


def seccion_agregaciones(insumos: dict) -> list[dict]:
    cubos, victimas = insumos["cubos"], insumos["victimas"]
    geo, temporal, causas = cubos["cubo_geografico"], cubos["cubo_temporal"], cubos["cubo_causas"]

    muertes_region = (geo[geo["desenlace"] == "Muerte"].groupby(["region", "fuente"], observed=True)["victimas"]
                      .sum().unstack().reset_index())
    orden = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dias = (temporal[temporal["fuente"] == "Policía Nacional"]
            .pivot_table(index="dia_semana", columns="desenlace", values="victimas", aggfunc="sum", observed=True)
            .reindex(orden).reset_index())
    con_causa = causas[causas["causa_categoria"].astype(str) != SI]
    top_causas = (con_causa.groupby(["desenlace", "causa_categoria"], observed=True)["victimas"].sum().reset_index()
                  .sort_values(["desenlace", "victimas"], ascending=[True, False]).groupby("desenlace").head(5))
    return [
        {"nombre": "muertes_por_region_y_fuente", "operacion_olap": "Roll-up (departamento -> región)",
         "dimensiones": ["region", "fuente"], "medida": "victimas (muertes)", "resultado": _registros(muertes_region)},
        {"nombre": "letalidad_por_medio_transporte", "operacion_olap": "Slice (Medicina Legal) + agregación",
         "dimensiones": ["medio_grupo"], "medida": "% de víctimas que fallecen",
         "resultado": _registros(_letalidad(victimas, "medio_grupo").rename("letalidad_pct").reset_index())},
        {"nombre": "letalidad_por_zona", "operacion_olap": "Slice + agregación", "dimensiones": ["zona"],
         "medida": "% de víctimas que fallecen",
         "resultado": _registros(_letalidad(victimas, "zona").rename("letalidad_pct").reset_index())},
        {"nombre": "victimas_por_dia_semana", "operacion_olap": "Pivot (día × desenlace) sobre el cubo temporal",
         "dimensiones": ["dia_semana", "desenlace"], "medida": "victimas (Policía Nacional)",
         "resultado": _registros(dias)},
        {"nombre": "top5_causas_por_desenlace", "operacion_olap": "Dice (causas con dato) + ranking",
         "dimensiones": ["desenlace", "causa_categoria"], "medida": "victimas",
         "resultado": _registros(top_causas)},
    ]


def seccion_modelo(insumos: dict) -> dict:
    meta, importancia, perfiles = insumos["metadata_modelo"], insumos["importancia"], insumos["perfiles"]
    comparacion = meta["comparacion_modelos"]
    lineas_base = {c["modelo"]: {k: v for k, v in c.items() if k != "modelo"} for c in comparacion
                   if c["modelo"].startswith("Línea base")}
    return {
        "tarea": meta["tarea"],
        "algoritmo": meta["nombre_modelo"],
        "clases": meta["clases"],
        "validacion": {"tipo": "Temporal", "entrenamiento": meta["periodo_entrenamiento"],
                       "prueba": meta["periodo_prueba"], "registros_entrenamiento": meta["registros_entrenamiento"],
                       "registros_prueba": meta["registros_prueba"]},
        "metricas": meta["metricas_prueba"],
        "linea_base": lineas_base,
        "comparacion_modelos": comparacion,
        "criterio_exito": {"definicion": "F1 macro ≥ mejor línea base + 0,10 y recall de Muerte ≥ 0,70",
                           **{k: (bool(v) if isinstance(v, bool) else v) for k, v in meta["criterio_exito"].items()}},
        "matriz_confusion": meta["matriz_confusion"],
        "variables_importantes": _registros(importancia[["variable", "variable_negocio", "importancia", "desviacion"]]
                                            .round(5)),
        "variables_usadas": meta["variables"],
        "variables_excluidas_por_fuga": meta["variables_excluidas"],
        "perfiles_riesgo": _registros(perfiles.round(4).reset_index()),
        "archivo": meta["archivo"],
        "version_sklearn": meta["version_sklearn"],
    }


def seccion_insights(insumos: dict) -> list[dict]:
    """Hallazgos calculados con los datos (no escritos a mano), cada uno con tema, cifra y gráfica de soporte."""
    victimas, unificado, meta = insumos["victimas"], insumos["unificado"], insumos["metadata_modelo"]
    perfiles = insumos["perfiles"]["veces_riesgo_muerte_vs_base"]
    ml = unificado[unificado["fuente"] == "Medicina Legal"]
    muertes = ml[ml["desenlace"] == "Muerte"].groupby("anio").size()
    lesiones = ml[ml["desenlace"] == "Lesión"].groupby("anio").size()
    zona = _letalidad(victimas, "zona")
    medio = _letalidad(victimas, "medio_grupo")
    objeto = _letalidad(victimas, "objeto_grupo")
    mayor = _letalidad(victimas.assign(adulto_mayor=victimas["adulto_mayor"].map({1: "60+", 0: "<60"})), "adulto_mayor")
    reparto = pd.crosstab(victimas["medio_grupo"], victimas["desenlace"], normalize="columns") * 100
    con_causa = victimas[victimas["causa_categoria"].astype(str) != SI]
    causas = pd.crosstab(con_causa["causa_categoria"], con_causa["desenlace"], normalize="columns") * 100
    causas = causas.drop(index="Otras / no especificadas", errors="ignore")
    sin_causa_muertes = _pct((victimas.loc[victimas["desenlace"] == "Muerte", "causa_categoria"] == SI).sum(),
                             (victimas["desenlace"] == "Muerte").sum())
    policia = unificado[(unificado["fuente"] == "Policía Nacional") & (unificado["desenlace"] == "Muerte")]
    razon = (policia.groupby("anio").size() / muertes).round(2)
    m = meta["metricas_prueba"]
    base_f1 = max(c["f1_macro"] for c in meta["comparacion_modelos"] if c["modelo"].startswith("Línea base"))
    anio_max = int(muertes.idxmax())
    fallecidos = victimas[victimas["desenlace"] == "Muerte"]
    pct_hombres_muertes = _pct((fallecidos["sexo"] == "Hombre").sum(), len(fallecidos))

    return [
        {"id": 1, "tema": "Tendencia temporal",
         "titulo": f"Las muertes crecen {_pct(muertes[anio_max] - muertes.iloc[0], muertes.iloc[0]):.0f} % entre "
                   f"{muertes.index[0]} y {anio_max}, mientras las lesiones no recuperan el nivel previo a la pandemia",
         "descripcion": f"Medicina Legal pasó de {_miles(muertes.iloc[0])} muertes en {muertes.index[0]} a {_miles(muertes[anio_max])} "
                        f"en {anio_max}. Las lesiones cayeron {abs(_pct(lesiones[2020] - lesiones[2019], lesiones[2019])):.0f} % "
                        f"en 2020 y en {lesiones.index[-1]} siguen "
                        f"{abs(_pct(lesiones.iloc[-1] - lesiones.iloc[0], lesiones.iloc[0])):.0f} % por debajo de "
                        f"{lesiones.index[0]}.",
         "metrica": {"muertes_inicio": int(muertes.iloc[0]), "muertes_maximo": int(muertes[anio_max]),
                     "anio_maximo": anio_max}, "visualizacion": "01_evolucion_anual"},
        {"id": 2, "tema": "Medio de transporte",
         "titulo": f"La motocicleta concentra el {reparto.loc['Motocicleta', 'Muerte']:.0f} % de las muertes y el "
                   f"{reparto.loc['Motocicleta', 'Lesión']:.0f} % de las lesiones",
         "descripcion": f"Los peatones son el {reparto.loc['Peatón (sin vehículo)', 'Muerte']:.0f} % de los fallecidos y "
                        f"tienen la mayor letalidad entre los usuarios frecuentes de la vía "
                        f"({medio['Peatón (sin vehículo)']:.0f} % de los peatones víctimas muere, frente a "
                        f"{medio['Motocicleta']:.0f} % de los motociclistas).",
         "metrica": {"pct_muertes_moto": round(float(reparto.loc["Motocicleta", "Muerte"]), 1),
                     "letalidad_peaton_pct": float(medio["Peatón (sin vehículo)"])},
         "visualizacion": "02_distribucion_medio"},
        {"id": 3, "tema": "Zona y región",
         "titulo": f"En zona rural muere el {zona['Parte rural (vereda y campo)']:.0f} % de las víctimas, frente al "
                   f"{zona['Cabecera municipal']:.0f} % en la cabecera municipal",
         "descripcion": f"Un siniestro rural es {_dec(zona['Parte rural (vereda y campo)'] / zona['Cabecera municipal'])} "
                        "veces más letal que uno urbano: velocidades más altas y atención médica más lejana. El patrón "
                        "se repite en todas las regiones naturales.",
         "metrica": {"letalidad_rural_pct": float(zona["Parte rural (vereda y campo)"]),
                     "letalidad_cabecera_pct": float(zona["Cabecera municipal"])},
         "visualizacion": "07_letalidad_region_zona"},
        {"id": 4, "tema": "Objeto de colisión",
         "titulo": f"Chocar contra un objeto fijo es mortal en el {objeto['Objeto fijo']:.0f} % de los casos",
         "descripcion": f"Es la colisión más letal (postes, árboles, muros), muy por encima del choque con automóvil "
                        f"({objeto['Automóvil o camioneta']:.0f} %) y con vehículo de carga pesada "
                        f"({objeto['Vehículo de carga pesada']:.0f} %).",
         "metrica": {"letalidad_objeto_fijo_pct": float(objeto["Objeto fijo"])},
         "visualizacion": "05_letalidad_objeto"},
        {"id": 5, "tema": "Perfil de la víctima",
         "titulo": f"Las víctimas de 60 años o más mueren en el {mayor['60+']:.0f} % de los casos, frente al "
                   f"{mayor['<60']:.0f} % del resto",
         "descripcion": f"El modelo confirma el efecto de la edad: con todo lo demás igual, tener 80 años o más "
                        f"multiplica por {_dec(perfiles.iloc[5])} la probabilidad de muerte. Los hombres son el "
                        f"{pct_hombres_muertes:.0f} % "
                        "de los fallecidos.",
         "metrica": {"letalidad_60_mas_pct": float(mayor["60+"]), "veces_riesgo_80_mas": round(float(perfiles.iloc[5]), 2)},
         "visualizacion": "04_gravedad_por_edad"},
        {"id": 6, "tema": "Causas",
         "titulo": f"Exceso de velocidad y desobedecer las normas explican el "
                   f"{causas.loc[['Exceso de velocidad', 'Desobedecer señales y normas'], 'Muerte'].sum():.0f} % de las "
                   "muertes con causa registrada",
         "descripcion": f"En los lesionados predominan las maniobras peligrosas ({causas.loc['Maniobras peligrosas', 'Lesión']:.0f} %) "
                        f"y la imprudencia del peatón ({causas.loc['Imprudencia del peatón', 'Lesión']:.0f} %). Advertencia: "
                        f"el {sin_causa_muertes:.0f} % de las muertes no tiene causa registrada.",
         "metrica": {"pct_sin_causa_en_muertes": sin_causa_muertes},
         "visualizacion": "06_causas"},
        {"id": 7, "tema": "Calidad de las fuentes",
         "titulo": f"La Policía registra entre {_dec(razon.min(), 2)} y {_dec(razon.max(), 2)} muertes por cada muerte forense",
         "descripcion": "Ninguna fuente por sí sola dimensiona el fenómeno: no toda muerte llega a denuncia penal y no "
                        "todo lesionado pasa por examen forense. Además, la hora y la causa faltan en la mitad o más de "
                        "las muertes del INMLCF.",
         "metrica": {"razon_min": float(razon.min()), "razon_max": float(razon.max())},
         "visualizacion": "08_cobertura_fuentes"},
        {"id": 8, "tema": "Modelo",
         "titulo": f"El modelo detecta {m['recall_Muerte'] * 100:.0f} de cada 100 muertes y supera la línea base en "
                   f"{(m['f1_macro'] - base_f1) * 100:.0f} puntos de F1 macro",
         "descripcion": f"{meta['nombre_modelo']} con validación temporal (prueba 2023-2024). Las variables más "
                        f"influyentes son {', '.join(insumos['importancia']['variable_negocio'].head(3).str.lower())}. "
                        "Separar lesión leve de grave es lo más difícil.",
         "metrica": {"f1_macro": m["f1_macro"], "recall_muerte": m["recall_Muerte"], "f1_linea_base": base_f1},
         "visualizacion": "09_matriz_confusion"},
    ]


def seccion_recomendaciones(insights: list[dict], insumos: dict) -> list[dict]:
    perfiles = insumos["perfiles"]["veces_riesgo_muerte_vs_base"]
    return [
        {"id": 1, "tema": "Zona y región", "prioridad": "Alta", "insights_relacionados": [3, 4],
         "recomendacion": "Priorizar control de velocidad, señalización y atención prehospitalaria en vías rurales y "
                          "centros poblados, con ambulancias ubicadas según los tramos de mayor letalidad.",
         "sustento": f"Zona rural ×{_dec(perfiles.iloc[2])} y objeto fijo ×{_dec(perfiles.iloc[1])} en la probabilidad de muerte."},
        {"id": 2, "tema": "Medio de transporte", "prioridad": "Alta", "insights_relacionados": [2, 6],
         "recomendacion": "Programas de seguridad para motociclistas: formación práctica para la licencia, control de "
                          "velocidad y uso de casco certificado, enfocados en los departamentos con más víctimas.",
         "sustento": insights[1]["titulo"]},
        {"id": 3, "tema": "Perfil de la víctima", "prioridad": "Alta", "insights_relacionados": [2, 5],
         "recomendacion": "Proteger al peatón adulto mayor: cruces seguros, tiempos de semáforo más largos y "
                          "reducción de velocidad cerca de hospitales, plazas de mercado y centros de salud.",
         "sustento": insights[4]["titulo"]},
        {"id": 4, "tema": "Infraestructura", "prioridad": "Media", "insights_relacionados": [4],
         "recomendacion": "Instalar barreras de contención y despejar o proteger postes y árboles al borde de la vía "
                          "en los corredores con choques contra objetos fijos.",
         "sustento": insights[3]["titulo"]},
        {"id": 5, "tema": "Calidad de las fuentes", "prioridad": "Media", "insights_relacionados": [6, 7],
         "recomendacion": "Que el INMLCF y la Policía registren obligatoriamente la hora y la causa del siniestro, y "
                          "compartan un identificador común del caso para cruzar las fuentes.",
         "sustento": insights[6]["descripcion"]},
        {"id": 6, "tema": "Modelo", "prioridad": "Media", "insights_relacionados": [8],
         "recomendacion": "Usar el modelo como alerta temprana para priorizar recursos (no para decidir casos "
                          "individuales) y reentrenarlo cada año con los datos nuevos.",
         "sustento": insights[7]["titulo"]},
    ]


def construir_documento(insumos: dict | None = None) -> dict:
    insumos = insumos or cargar_insumos()
    insights = seccion_insights(insumos)
    documento = {
        "_id": config.MONGO_ID_DOCUMENTO,
        "metadata": seccion_metadata(insumos),
        "calidad_datos": seccion_calidad(insumos),
        "estadisticas_descriptivas": seccion_estadisticas(insumos),
        "agregaciones": seccion_agregaciones(insumos),
        "modelo": seccion_modelo(insumos),
        "insights": insights,
        "visualizaciones_generadas": insumos["visualizaciones"],
        "recomendaciones": seccion_recomendaciones(insights, insumos),
    }
    # Copia en disco (respaldo y entrada del reporte PDF y del dashboard si MongoDB no está disponible)
    config.crear_directorios()
    ARCHIVO_DOCUMENTO.write_text(json.dumps(documento, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return documento


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
def conectar(uri: str | None = None, timeout_ms: int = 5000):
    """Devuelve (cliente, colección). Falla con un mensaje claro si MongoDB no responde."""
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure, PyMongoError

    if os.getenv("MONGO_SIMULADO") == "1":
        # Solo para pruebas sin servidor (requiere 'pip install mongomock'); la entrega usa MongoDB real.
        import mongomock
        global _CLIENTE_SIMULADO
        _CLIENTE_SIMULADO = globals().get("_CLIENTE_SIMULADO") or mongomock.MongoClient()  # misma "base" en todo el proceso
        return _CLIENTE_SIMULADO, _CLIENTE_SIMULADO[config.MONGO_DB][config.MONGO_COLECCION]
    uri = uri or config.MONGO_URI
    cliente = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        cliente.admin.command("ping")
        # 'ping' responde aunque no haya usuario; listar colecciones exige autenticación y detecta ese caso aquí
        cliente[config.MONGO_DB].list_collection_names()
    except OperationFailure as error:
        env = config.RAIZ_PROYECTO / ".env"
        raise ConnectionError(
            f"MongoDB respondió, pero rechazó la autenticación ({error.code}: {str(error)[:80]}). "
            f"¿Existe el archivo .env? {env.exists()} -> {env}. Debe contener MONGO_USER y MONGO_PASSWORD del "
            "contenedor; después reinicie el kernel (Restart).") from error
    except PyMongoError as error:
        oculto = uri.split("@")[-1]
        raise ConnectionError(
            f"No se pudo conectar a MongoDB en '{oculto}' ({type(error).__name__}). Encienda el contenedor "
            "(docker compose up -d mongodb) y revise MONGO_USER / MONGO_PASSWORD / MONGO_PORT en el archivo .env "
            "de la raíz del proyecto.") from error
    return cliente, cliente[config.MONGO_DB][config.MONGO_COLECCION]


def guardar(documento: dict, coleccion) -> dict:
    """Inserta o reemplaza el documento (mismo _id): volver a ejecutar no crea duplicados."""
    resultado = coleccion.replace_one({"_id": documento["_id"]}, documento, upsert=True)
    coleccion.create_index("insights.tema")
    coleccion.create_index("recomendaciones.prioridad")
    return {"_id": documento["_id"], "accion": "insertado" if resultado.upserted_id is not None else "actualizado"}


def uri_sin_clave(uri: str | None = None) -> str:
    """URI para mostrar en pantalla sin la contraseña."""
    uri = uri or config.MONGO_URI
    if "@" not in uri:
        return uri
    inicio, resto = uri.split("://", 1)
    usuario = resto.split("@", 1)[0].split(":")[0]
    return f"{inicio}://{usuario}:****@{resto.split('@', 1)[1]}"


def ejecutar() -> dict:
    documento = construir_documento()
    cliente, coleccion = conectar()
    try:
        return guardar(documento, coleccion)
    finally:
        cliente.close()


if __name__ == "__main__":
    print(ejecutar())
