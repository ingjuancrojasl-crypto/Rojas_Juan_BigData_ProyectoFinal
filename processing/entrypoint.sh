#!/bin/sh
# Orden de ejecución del servicio de procesamiento:
#   1. Descargar el dataset a data/raw/ (si no existe)
#   2. Ejecutar el ETL y escribir el Parquet en data/processed/ (si no existe)
#   3. Dejar Jupyter Lab disponible para ejecutar el notebook completo
set -e
cd /proyecto

python src/descargar_datos.py

if [ ! -f data/processed/quejas_limpio.parquet ] || [ "${FORZAR_ETL:-0}" = "1" ]; then
    echo "Ejecutando ETL..."
    python src/etl.py
else
    echo "Parquet limpio ya existe; se omite el ETL (use FORZAR_ETL=1 para repetirlo)."
fi

exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN}" --notebook-dir=/proyecto
