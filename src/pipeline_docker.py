"""Orquestación del servicio 'procesamiento' en Docker.

Con un solo `docker compose up --build`, este script deja listos todos los resultados que usa la app:

  * Si faltan los datos procesados o el modelo (por ejemplo, en un clon nuevo del repositorio), o si el modelo
    se entrenó con otra versión de scikit-learn, ejecuta el NOTEBOOK COMPLETO sin intervención
    (descarga → ETL → EDA y gráficas → modelo → MongoDB → PDF). La copia ejecutada queda en
    reports/notebook_ejecutado_docker.ipynb, como evidencia.
  * Si los resultados ya existen, solo guarda el documento de análisis en MongoDB (el volumen de Mongo puede
    estar vacío) y regenera el PDF desde MongoDB: tarda segundos.

Al terminar crea una marca (/tmp/pipeline_listo) que el healthcheck de Docker usa para arrancar la app.
Variables: FORZAR_PIPELINE=1 obliga a ejecutar el notebook completo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

NOTEBOOK = config.RAIZ_PROYECTO / "notebooks" / "Rojas_Juan_BigData_ProyectoFinal.ipynb"
SALIDA_NOTEBOOK = "notebook_ejecutado_docker"
MARCA = Path(os.getenv("MARCA_PIPELINE", "/tmp/pipeline_listo"))


def registrar(mensaje: str) -> None:
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


def motivo_para_ejecutar_notebook() -> str | None:
    """Devuelve por qué hay que ejecutar el notebook completo, o None si los resultados sirven."""
    if os.getenv("FORZAR_PIPELINE", "0") == "1":
        return "FORZAR_PIPELINE=1"
    requeridos = [config.ARCHIVO_UNIFICADO, config.ARCHIVO_DETALLE, config.DIR_MODELOS / "modelo_gravedad.joblib",
                  config.DIR_MODELOS / "metadata_modelo.json", config.DIR_REPORTES / "figuras" / "visualizaciones.json"]
    faltan = [str(r.relative_to(config.RAIZ_PROYECTO)) for r in requeridos if not r.exists()]
    if faltan:
        return f"faltan resultados: {', '.join(faltan)}"
    import sklearn
    version_modelo = json.loads((config.DIR_MODELOS / "metadata_modelo.json").read_text(encoding="utf-8")) \
        .get("version_sklearn")
    if version_modelo != sklearn.__version__:
        return (f"el modelo se entrenó con scikit-learn {version_modelo} y el contenedor tiene {sklearn.__version__}; "
                "se reentrena para que el modelo cargue sin incompatibilidades")
    return None


def ejecutar_notebook() -> None:
    comando = ["jupyter", "nbconvert", "--to", "notebook", "--execute", str(NOTEBOOK),
               "--output-dir", str(config.DIR_REPORTES), "--output", SALIDA_NOTEBOOK,
               "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=python3"]
    registrar("Ejecutando el notebook completo (descarga, ETL, gráficas, modelo, MongoDB y PDF). "
              "Tarda varios minutos…")
    subprocess.run(comando, check=True)
    registrar(f"Notebook ejecutado sin errores: reports/{SALIDA_NOTEBOOK}.ipynb")


def actualizar_mongo_y_pdf() -> None:
    import mongo_analisis
    import reporte_pdf
    resultado = mongo_analisis.ejecutar()
    registrar(f"Documento {resultado['accion']} en MongoDB: {resultado['_id']}")
    ruta = reporte_pdf.generar_reporte()
    registrar(f"Reporte PDF generado desde MongoDB: {ruta.relative_to(config.RAIZ_PROYECTO)}")


def main() -> int:
    inicio = time.perf_counter()
    MARCA.unlink(missing_ok=True)
    registrar(f"MongoDB: {__import__('mongo_analisis').uri_sin_clave()}")
    motivo = motivo_para_ejecutar_notebook()
    try:
        if motivo:
            registrar(f"Se ejecuta el pipeline completo porque {motivo}.")
            ejecutar_notebook()
        else:
            registrar("Los datos procesados y el modelo ya existen: no se repite el ETL ni el entrenamiento.")
            actualizar_mongo_y_pdf()
    except Exception as error:  # noqa: BLE001
        registrar(f"ERROR: {type(error).__name__}: {error}")
        registrar("La app no arrancará hasta corregirlo. Revise: docker compose logs procesamiento")
        return 1
    MARCA.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    registrar(f"PIPELINE LISTO en {time.perf_counter() - inicio:.0f} s. Dashboard: http://localhost:8501")
    return 0


if __name__ == "__main__":
    sys.exit(main())
