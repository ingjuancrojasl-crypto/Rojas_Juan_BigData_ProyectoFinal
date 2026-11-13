"""
Configuración central del proyecto.

Todas las rutas son relativas a la raíz del repositorio, de modo que el código
funciona igual en el equipo del estudiante, en el del docente y dentro de Docker.
"""
from pathlib import Path
import os

# Raíz del proyecto: carpeta que contiene src/, data/, notebooks/, etc.
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Rutas de datos
# ---------------------------------------------------------------------------
DIR_DATOS = RAIZ_PROYECTO / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_PROCESSED = DIR_DATOS / "processed"
DIR_REFERENCIA = DIR_DATOS / "reference"
DIR_AGREGACIONES = DIR_PROCESSED / "agregaciones"
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_REPORTES = RAIZ_PROYECTO / "reports"

ARCHIVO_RAW = DIR_RAW / "quejas.csv"
ARCHIVO_METADATA_DESCARGA = DIR_RAW / "metadata_descarga.json"
ARCHIVO_PARQUET_LIMPIO = DIR_PROCESSED / "quejas_limpio.parquet"
ARCHIVO_RESUMEN_ETL = DIR_PROCESSED / "resumen_etl.json"

ARCHIVO_TIPOS_ENTIDAD = DIR_REFERENCIA / "tipos_entidad_sfc.csv"
ARCHIVO_REGIONES = DIR_REFERENCIA / "regiones_departamentos.csv"

# ---------------------------------------------------------------------------
# Fuentes del dataset
# ---------------------------------------------------------------------------
DATASET_ID = "xyy7-rn7p"
DATASET_NOMBRE = (
    "Quejas interpuestas por los consumidores financieros en contra de las "
    "entidades vigiladas en la plataforma Smartsupervision"
)
DATASET_ENTIDAD = "Superintendencia Financiera de Colombia (SFC)"
URL_PAGINA_DATASET = f"https://www.datos.gov.co/d/{DATASET_ID}"

# Fuente oficial (descarga completa) y respaldo en GitHub (CSV comprimido).
URL_OFICIAL = f"https://www.datos.gov.co/api/views/{DATASET_ID}/rows.csv?accessType=DOWNLOAD"
URL_RESPALDO_GITHUB = (
    "https://raw.githubusercontent.com/ingjuancrojasl-crypto/"
    "Rojas_Juan_BigData_ProyectoFinal/main/quejas.zip"
)

# ---------------------------------------------------------------------------
# Parámetros del análisis
# ---------------------------------------------------------------------------
INSTANCIA_NO_ESCALADA = "Entidad vigilada"   # la queja se atiende en la propia entidad
VALOR_SIN_UBICACION = "SIN INFORMACIÓN"
SEMILLA = 42

# ---------------------------------------------------------------------------
# MongoDB (las credenciales llegan por variables de entorno desde .env)
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "quejas_financieras")
MONGO_COLECCION = os.getenv("MONGO_COLECCION", "analisis")


def crear_directorios() -> None:
    """Crea las carpetas de trabajo si no existen."""
    for carpeta in (DIR_RAW, DIR_PROCESSED, DIR_AGREGACIONES, DIR_MODELOS, DIR_REPORTES):
        carpeta.mkdir(parents=True, exist_ok=True)
