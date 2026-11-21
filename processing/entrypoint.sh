#!/bin/sh
# Servicio de procesamiento. Orden de ejecución:
#   1. src/pipeline_docker.py deja listos los resultados:
#        - clon nuevo (sin Parquet ni modelo): ejecuta el notebook completo
#          (descarga -> ETL -> gráficas -> modelo -> MongoDB -> PDF)
#        - resultados existentes: solo guarda el documento en MongoDB y regenera el PDF
#      Al terminar crea /tmp/pipeline_listo; con esa marca el healthcheck pasa a "healthy" y arranca la app.
#   2. Jupyter Lab queda disponible en http://localhost:8888 para revisar o volver a ejecutar el notebook.
cd /proyecto

if python src/pipeline_docker.py; then
    echo "Procesamiento terminado. Jupyter Lab: http://localhost:${JUPYTER_PORT:-8888}/?token=${JUPYTER_TOKEN:-siniestros}"
else
    echo "El procesamiento falló (ver mensajes anteriores). Jupyter Lab queda abierto para diagnosticar."
fi

exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.token="${JUPYTER_TOKEN:-siniestros}" --notebook-dir=/proyecto
