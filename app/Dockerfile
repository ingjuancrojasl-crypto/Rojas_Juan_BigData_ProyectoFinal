# Servicio de la aplicación: dashboard Streamlit (solo lee resultados ya procesados)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /proyecto

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
