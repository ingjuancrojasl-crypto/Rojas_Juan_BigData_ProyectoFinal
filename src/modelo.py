"""
Modelo de clasificación de la gravedad de un siniestro de transporte (sección 5: Modeling + Evaluation).

Tarea: clasificación multiclase de `gravedad` ∈ {Lesión leve, Lesión grave, Muerte} con las víctimas de
Medicina Legal que tienen gravedad conocida.

Prevención de fuga de información (data leakage):
  1. Solo se usan predictores registrados de forma comparable en lesiones y muertes. Se excluyen hora,
     causa, actividad, escolaridad, estado civil, pertenencia étnica y servicio del objeto (faltan mucho más
     en las muertes), el escenario (se codifica distinto en cada archivo) y todo lo que se conoce DESPUÉS del
     hecho: días de incapacidad (define la clase), mecanismo causal y diagnóstico de la lesión.
  2. En las variables usadas, el valor 'Sin información' se reemplaza por el más frecuente aprendido en el
     entrenamiento (SimpleImputer dentro del Pipeline), para que la ausencia del dato no delate el desenlace.
  3. División temporal: entrenamiento 2015-2022 y prueba 2023-2024. Todos los ajustes (imputación,
     codificación, escalado) se aprenden solo con el entrenamiento, dentro del Pipeline.
  4. El contexto policial del municipio usa el año ANTERIOR al hecho.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

SI = config.SIN_INFORMACION
CLASES = ["Lesión leve", "Lesión grave", "Muerte"]

VARIABLES_CATEGORICAS = ["sexo", "grupo_edad_quinquenal", "condicion_victima", "medio_grupo", "servicio_vehiculo",
                         "clase_accidente", "objeto_grupo", "zona", "dia_semana", "departamento"]
VARIABLES_NUMERICAS = ["mes", "fin_de_semana", "lesionados_policia_anio_ant", "muertes_policia_anio_ant"]
VARIABLES = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS

# Variables excluidas y motivo (se documentan en el notebook y en MongoDB)
VARIABLES_EXCLUIDAS = {
    "dias_incapacidad": "Define la clase (lesión leve/grave): usarla sería copiar la respuesta",
    "mecanismo_causal, diagnostico_topografico": "Se conocen después del hecho (resultado del examen forense)",
    "rango_hora / franja_horaria": "Sin dato en el 50 % de las muertes frente al 6 % de las lesiones",
    "causa (circunstancia)": "Sin dato en el 66 % de las muertes frente al 10 % de las lesiones",
    "actividad, escolaridad, estado civil, pertenencia étnica, servicio del objeto":
        "Faltan entre 14 y 35 puntos porcentuales más en las muertes",
    "escenario": "Se codifica distinto en cada archivo ('Vía pública' en muertes, 'Calle' en lesiones)",
    "anio": "Se usa para dividir entrenamiento y prueba; como predictor no generaliza a años futuros",
}

NOMBRES_NEGOCIO = {
    "sexo": "Sexo de la víctima", "grupo_edad_quinquenal": "Edad de la víctima",
    "condicion_victima": "Condición (conductor, pasajero, peatón)", "medio_grupo": "Medio de transporte",
    "servicio_vehiculo": "Servicio del vehículo", "clase_accidente": "Clase de accidente",
    "objeto_grupo": "Objeto de colisión", "zona": "Zona (urbana o rural)", "dia_semana": "Día de la semana",
    "departamento": "Departamento", "mes": "Mes", "fin_de_semana": "Fin de semana",
    "lesionados_policia_anio_ant": "Lesionados en el municipio (año anterior)",
    "muertes_policia_anio_ant": "Muertes en el municipio (año anterior)",
}

ARCHIVO_MODELO = config.DIR_MODELOS / "modelo_gravedad.joblib"
ARCHIVO_METADATA = config.DIR_MODELOS / "metadata_modelo.json"
ARCHIVO_IMPORTANCIA = config.DIR_MODELOS / "importancia_variables.csv"
PESO_EXTRA_MUERTE = 1.3          # Gradient Boosting: peso adicional a la clase más costosa de pasar por alto


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def preparar_datos(victimas: pd.DataFrame) -> dict:
    """Filtra víctimas con gravedad conocida, selecciona variables y divide por tiempo."""
    datos = victimas[victimas["gravedad"].astype(str).isin(CLASES)].copy()
    X = datos[VARIABLES].copy()
    for columna in VARIABLES_CATEGORICAS:
        X[columna] = X[columna].astype(str)
    for columna in VARIABLES_NUMERICAS:
        X[columna] = X[columna].astype(float)
    y = datos["gravedad"].astype(str)
    entrenamiento = datos["anio"].between(*config.ANIOS_ENTRENAMIENTO)
    prueba = datos["anio"].between(*config.ANIOS_PRUEBA)
    return {"X_train": X[entrenamiento], "y_train": y[entrenamiento], "X_test": X[prueba], "y_test": y[prueba],
            "anio_train": datos.loc[entrenamiento, "anio"], "anio_test": datos.loc[prueba, "anio"]}


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def _imputador():
    return SimpleImputer(missing_values=SI, strategy="most_frequent")


def crear_modelos() -> dict:
    log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    prep_logistica = ColumnTransformer([
        ("cat", Pipeline([("imputar", _imputador()),
                          ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]),
         VARIABLES_CATEGORICAS),
        ("num", Pipeline([("log", log), ("escalar", StandardScaler())]), VARIABLES_NUMERICAS),
    ])
    prep_arboles = ColumnTransformer([
        ("cat", Pipeline([("imputar", _imputador()),
                          ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]),
         VARIABLES_CATEGORICAS),
        ("num", "passthrough", VARIABLES_NUMERICAS),
    ])
    return {
        "Línea base: clase mayoritaria": DummyClassifier(strategy="most_frequent"),
        "Línea base: estratificada": DummyClassifier(strategy="stratified", random_state=config.SEMILLA),
        "Regresión logística": Pipeline([
            ("preparacion", prep_logistica),
            ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preparacion", prep_arboles),
            ("modelo", HistGradientBoostingClassifier(
                categorical_features=list(range(len(VARIABLES_CATEGORICAS))), max_iter=400, learning_rate=0.08,
                max_leaf_nodes=63, early_stopping=True, random_state=config.SEMILLA)),
        ]),
    }


def _pesos_gradient_boosting(y: pd.Series) -> np.ndarray:
    clases = np.array(sorted(y.unique()))
    pesos = dict(zip(clases, compute_class_weight("balanced", classes=clases, y=y)))
    pesos["Muerte"] *= PESO_EXTRA_MUERTE
    return y.map(pesos).to_numpy()


def evaluar(nombre: str, y_real: pd.Series, y_pred: np.ndarray, segundos: float) -> dict:
    precision, recall, f1, soporte = precision_recall_fscore_support(y_real, y_pred, labels=CLASES, zero_division=0)
    return {
        "modelo": nombre, "accuracy": accuracy_score(y_real, y_pred),
        "precision_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)),
        "f1_macro": f1_score(y_real, y_pred, average="macro", labels=CLASES),
        **{f"f1_{c}": float(v) for c, v in zip(CLASES, f1)},
        **{f"recall_{c}": float(v) for c, v in zip(CLASES, recall)},
        **{f"precision_{c}": float(v) for c, v in zip(CLASES, precision)},
        "segundos_entrenamiento": round(segundos, 1),
    }


def entrenar_y_comparar(datos: dict) -> tuple[pd.DataFrame, dict, dict]:
    """Entrena líneas base y modelos; devuelve tabla de métricas, modelos ajustados y predicciones."""
    resultados, ajustados, predicciones = [], {}, {}
    for nombre, modelo in crear_modelos().items():
        inicio = time.perf_counter()
        if nombre == "Gradient Boosting":
            modelo.fit(datos["X_train"], datos["y_train"], modelo__sample_weight=_pesos_gradient_boosting(datos["y_train"]))
        else:
            modelo.fit(datos["X_train"], datos["y_train"])
        segundos = time.perf_counter() - inicio
        predicho = modelo.predict(datos["X_test"])
        resultados.append(evaluar(nombre, datos["y_test"], predicho, segundos))
        ajustados[nombre], predicciones[nombre] = modelo, predicho
    tabla = pd.DataFrame(resultados).set_index("modelo")
    return tabla, ajustados, predicciones


def criterio_exito(tabla: pd.DataFrame, minimo_mejora: float = 0.10, minimo_recall_muerte: float = 0.70) -> pd.DataFrame:
    base = tabla.loc[tabla.index.str.startswith("Línea base"), "f1_macro"].max()
    evaluacion = tabla[["f1_macro", "recall_Muerte"]].copy()
    evaluacion["mejora_vs_linea_base"] = evaluacion["f1_macro"] - base
    evaluacion["cumple_mejora_F1"] = evaluacion["mejora_vs_linea_base"] >= minimo_mejora
    evaluacion["cumple_recall_muerte"] = evaluacion["recall_Muerte"] >= minimo_recall_muerte
    evaluacion["cumple_criterio"] = evaluacion["cumple_mejora_F1"] & evaluacion["cumple_recall_muerte"]
    return evaluacion


def elegir_mejor(evaluacion: pd.DataFrame) -> str:
    candidatos = evaluacion[evaluacion["cumple_criterio"]]
    if candidatos.empty:
        candidatos = evaluacion[~evaluacion.index.str.startswith("Línea base")]
    return candidatos["f1_macro"].idxmax()


# ---------------------------------------------------------------------------
# Control de fuga: qué pasaría si se usaran las variables mal diligenciadas
# ---------------------------------------------------------------------------
def experimento_fuga(victimas: pd.DataFrame) -> pd.DataFrame:
    """Compara el modelo honesto con uno que incluye variables con registro desigual entre lesiones y muertes."""
    datos = victimas[victimas["gravedad"].astype(str).isin(CLASES)].copy()
    con_fuga = ["franja_horaria", "causa_categoria", "escolaridad", "actividad"]
    filas = []
    for etiqueta, categoricas in [("Sin variables con fuga (modelo del proyecto)", VARIABLES_CATEGORICAS),
                                  ("Con hora, causa, escolaridad y actividad (fuga)", VARIABLES_CATEGORICAS + con_fuga)]:
        X = datos[categoricas + VARIABLES_NUMERICAS].copy()
        for c in categoricas:
            X[c] = X[c].astype(str)
        y = datos["gravedad"].astype(str)
        tr, te = datos["anio"].between(*config.ANIOS_ENTRENAMIENTO), datos["anio"].between(*config.ANIOS_PRUEBA)
        prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                   categoricas), ("num", "passthrough", VARIABLES_NUMERICAS)])
        modelo = Pipeline([("prep", prep), ("modelo", HistGradientBoostingClassifier(
            categorical_features=list(range(len(categoricas))), class_weight="balanced", max_iter=300,
            learning_rate=0.08, early_stopping=True, random_state=config.SEMILLA))])
        modelo.fit(X[tr], y[tr])
        pred = modelo.predict(X[te])
        recall_muerte = precision_recall_fscore_support(y[te], pred, labels=["Muerte"], zero_division=0)[1][0]
        filas.append({"escenario": etiqueta, "f1_macro": f1_score(y[te], pred, average="macro"),
                      "recall_Muerte": recall_muerte})
    return pd.DataFrame(filas).set_index("escenario")


# ---------------------------------------------------------------------------
# Interpretación
# ---------------------------------------------------------------------------
def importancia_variables(modelo, datos: dict, muestra: int = 25_000) -> pd.DataFrame:
    """Importancia por permutación: cuánto cae el F1 macro al desordenar cada variable (en datos de prueba)."""
    X = datos["X_test"].sample(min(muestra, len(datos["X_test"])), random_state=config.SEMILLA)
    y = datos["y_test"].loc[X.index]
    resultado = permutation_importance(modelo, X, y, scoring="f1_macro", n_repeats=3,
                                       random_state=config.SEMILLA, n_jobs=1)
    tabla = pd.DataFrame({"variable": X.columns, "importancia": resultado.importances_mean,
                          "desviacion": resultado.importances_std})
    tabla["variable_negocio"] = tabla["variable"].map(NOMBRES_NEGOCIO)
    return tabla.sort_values("importancia", ascending=False).reset_index(drop=True)


PERFIL_BASE = {"sexo": "Hombre", "grupo_edad_quinquenal": "(25 a 29)", "condicion_victima": "Conductor",
               "medio_grupo": "Motocicleta", "servicio_vehiculo": "Particular", "clase_accidente": "Choque",
               "objeto_grupo": "Automóvil o camioneta", "zona": "Cabecera municipal", "dia_semana": "Miércoles",
               "departamento": "Antioquia", "mes": 6.0, "fin_de_semana": 0.0,
               "lesionados_policia_anio_ant": 300.0, "muertes_policia_anio_ant": 40.0}


def perfiles_de_riesgo(modelo) -> pd.DataFrame:
    """Probabilidad estimada de cada nivel de gravedad para perfiles típicos (lectura de negocio).

    Parte de un perfil base y cambia UNA característica a la vez, para aislar su efecto.
    """
    cambios = [
        ("Perfil base: motociclista de 25-29 años, choque con automóvil en la ciudad", {}),
        ("… si choca contra un objeto fijo", {"objeto_grupo": "Objeto fijo", "clase_accidente": "Choque con objeto fijo o en movimiento"}),
        ("… si el choque ocurre en zona rural", {"zona": "Parte rural (vereda y campo)"}),
        ("… si choca con un vehículo de carga pesada", {"objeto_grupo": "Vehículo de carga pesada"}),
        ("… si es un peatón atropellado", {"condicion_victima": "Peatón", "medio_grupo": "Peatón (sin vehículo)",
                                          "clase_accidente": "Atropello", "servicio_vehiculo": "No aplica"}),
        ("… si la víctima tiene 80 años o más", {"grupo_edad_quinquenal": "(80 y más)"}),
        ("… si ocurre un domingo", {"dia_semana": "Domingo", "fin_de_semana": 1.0}),
        ("… si es mujer", {"sexo": "Mujer"}),
        ("Peor combinación: peatón de 80+ atropellado por carga pesada en zona rural",
         {"condicion_victima": "Peatón", "medio_grupo": "Peatón (sin vehículo)", "clase_accidente": "Atropello",
          "servicio_vehiculo": "No aplica", "grupo_edad_quinquenal": "(80 y más)",
          "objeto_grupo": "Vehículo de carga pesada", "zona": "Parte rural (vereda y campo)"}),
    ]
    filas = []
    for descripcion, cambio in cambios:
        perfil = {**PERFIL_BASE, **cambio}
        X = pd.DataFrame([perfil])[VARIABLES]
        probabilidades = modelo.predict_proba(X)[0]
        filas.append({"perfil": descripcion, **dict(zip(modelo.classes_, probabilidades))})
    tabla = pd.DataFrame(filas).set_index("perfil")[CLASES]
    tabla["veces_riesgo_muerte_vs_base"] = tabla["Muerte"] / tabla["Muerte"].iloc[0]
    return tabla


# ---------------------------------------------------------------------------
# Guardado (para el dashboard: carga y predice sin reentrenar)
# ---------------------------------------------------------------------------
def guardar_modelo(modelo, nombre: str, tabla: pd.DataFrame, evaluacion: pd.DataFrame, importancia: pd.DataFrame,
                   matriz: np.ndarray, datos: dict) -> dict:
    config.crear_directorios()
    joblib.dump(modelo, ARCHIVO_MODELO, compress=3)
    importancia.to_csv(ARCHIVO_IMPORTANCIA, index=False, encoding="utf-8")
    metadata = {
        "nombre_modelo": nombre, "tarea": "Clasificación multiclase de la gravedad del siniestro",
        "clases": CLASES, "variables": VARIABLES, "variables_categoricas": VARIABLES_CATEGORICAS,
        "variables_numericas": VARIABLES_NUMERICAS, "variables_excluidas": VARIABLES_EXCLUIDAS,
        "periodo_entrenamiento": list(config.ANIOS_ENTRENAMIENTO), "periodo_prueba": list(config.ANIOS_PRUEBA),
        "registros_entrenamiento": int(len(datos["X_train"])), "registros_prueba": int(len(datos["X_test"])),
        "metricas_prueba": tabla.loc[nombre].round(4).to_dict(),
        "comparacion_modelos": tabla.round(4).reset_index().to_dict(orient="records"),
        "criterio_exito": evaluacion.loc[nombre].astype(object).to_dict(),
        "matriz_confusion": {"etiquetas": CLASES, "valores": matriz.tolist()},
        "importancia_variables": importancia.head(10).round(5).to_dict(orient="records"),
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "version_sklearn": sklearn.__version__, "archivo": str(ARCHIVO_MODELO.relative_to(config.RAIZ_PROYECTO)),
    }
    ARCHIVO_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metadata


def cargar_modelo():
    return joblib.load(ARCHIVO_MODELO), json.loads(ARCHIVO_METADATA.read_text(encoding="utf-8"))


def ejecutar_modelado() -> dict:
    victimas = pd.read_parquet(config.ARCHIVO_DETALLE)
    datos = preparar_datos(victimas)
    tabla, ajustados, predicciones = entrenar_y_comparar(datos)
    evaluacion = criterio_exito(tabla)
    mejor = elegir_mejor(evaluacion)
    matriz = confusion_matrix(datos["y_test"], predicciones[mejor], labels=CLASES)
    importancia = importancia_variables(ajustados[mejor], datos)
    return guardar_modelo(ajustados[mejor], mejor, tabla, evaluacion, importancia, matriz, datos)


if __name__ == "__main__":
    meta = ejecutar_modelado()
    print(f"Mejor modelo: {meta['nombre_modelo']} | F1 macro {meta['metricas_prueba']['f1_macro']:.3f} | "
          f"recall Muerte {meta['metricas_prueba']['recall_Muerte']:.3f}")
    print(f"Guardado en {meta['archivo']}")
