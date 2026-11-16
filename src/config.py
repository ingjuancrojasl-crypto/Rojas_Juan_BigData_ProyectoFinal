"""
Configuración central del proyecto.

Todas las rutas son relativas a la raíz del repositorio, de modo que el código
funciona igual en el equipo del estudiante, en el del docente y dentro de Docker.
"""
from pathlib import Path
import os

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DIR_DATOS = RAIZ_PROYECTO / "data"
DIR_RAW = DIR_DATOS / "raw"
DIR_PROCESSED = DIR_DATOS / "processed"
DIR_REFERENCIA = DIR_DATOS / "reference"
DIR_AGREGACIONES = DIR_PROCESSED / "agregaciones"
DIR_MODELOS = RAIZ_PROYECTO / "models"
DIR_REPORTES = RAIZ_PROYECTO / "reports"

ARCHIVO_METADATA_DESCARGA = DIR_RAW / "metadata_descarga.json"
ARCHIVO_UNIFICADO = DIR_PROCESSED / "siniestros_unificado.parquet"   # 4 fuentes, columnas comunes
ARCHIVO_DETALLE = DIR_PROCESSED / "victimas_medicina_legal.parquet"   # Medicina Legal, todas las variables
ARCHIVO_RESUMEN_ETL = DIR_PROCESSED / "resumen_etl.json"
ARCHIVO_REGIONES = DIR_REFERENCIA / "regiones_departamentos.csv"
ARCHIVO_CATALOGO_CAUSAS = DIR_REFERENCIA / "catalogo_causas.csv"

# ---------------------------------------------------------------------------
# Fuentes (datos.gov.co) con respaldo comprimido en GitHub
# ---------------------------------------------------------------------------
_BASE_DESCARGA = "https://www.datos.gov.co/api/views/{id}/rows.csv?accessType=DOWNLOAD"
_BASE_RESPALDO = ("https://raw.githubusercontent.com/ingjuancrojasl-crypto/"
                  "Rojas_Juan_BigData_ProyectoFinal/main/respaldo_datos/{nombre}.zip")

FUENTES = {
    "inml_lesiones": {
        "id": "ezhf-hscf",
        "titulo": "Lesiones por eventos de transporte. Colombia, años 2015 a 2024. Cifras definitivas",
        "entidad": "Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF)",
        "filas_minimas": 300_000,
    },
    "inml_muertes": {
        "id": "s65h-7665",
        "titulo": "Muertes por eventos de transporte. Colombia, años 2015 a 2024. Cifras definitivas",
        "entidad": "Instituto Nacional de Medicina Legal y Ciencias Forenses (INMLCF)",
        "filas_minimas": 60_000,
    },
    "policia_lesiones": {
        "id": "72sg-cybi",
        "titulo": "Reporte Lesiones Personales y en Accidente de Tránsito Policía Nacional",
        "entidad": "Dirección General de la Policía Nacional (DIPON)",
        "filas_minimas": 1_000_000,
    },
    "policia_homicidios_transito": {
        "id": "ha6j-pa2r",
        "titulo": "Homicidios accidente de tránsito Policía Nacional",
        "entidad": "Dirección General de la Policía Nacional (DIPON)",
        "filas_minimas": 80_000,
    },
}
for _nombre, _fuente in FUENTES.items():
    _fuente["archivo"] = DIR_RAW / f"{_nombre}.csv"
    _fuente["pagina"] = f"https://www.datos.gov.co/d/{_fuente['id']}"
    _fuente["url_oficial"] = _BASE_DESCARGA.format(id=_fuente["id"])
    _fuente["url_respaldo"] = _BASE_RESPALDO.format(nombre=_nombre)

# ---------------------------------------------------------------------------
# Parámetros del análisis
# ---------------------------------------------------------------------------
ANIO_INICIO, ANIO_FIN = 2015, 2024          # periodo común a las 4 fuentes
ANIOS_ENTRENAMIENTO = (2015, 2022)          # validación temporal del modelo
ANIOS_PRUEBA = (2023, 2024)
SIN_INFORMACION = "Sin información"
SEMILLA = 42

# ---------------------------------------------------------------------------
# MongoDB (credenciales por variables de entorno desde .env)
# ---------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "siniestros_transporte")
MONGO_COLECCION = os.getenv("MONGO_COLECCION", "analisis")


def crear_directorios() -> None:
    for carpeta in (DIR_RAW, DIR_PROCESSED, DIR_AGREGACIONES, DIR_MODELOS, DIR_REPORTES):
        carpeta.mkdir(parents=True, exist_ok=True)
