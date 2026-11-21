# Servicio de procesamiento: Jupyter Lab + pandas + scikit-learn
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /proyecto

COPY processing/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY processing/entrypoint.sh /usr/local/bin/entrypoint.sh
# Quita finales de línea CRLF (si el archivo pasó por Windows) y lo hace ejecutable
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
