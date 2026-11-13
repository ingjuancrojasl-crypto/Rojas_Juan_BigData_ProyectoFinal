# Escalamiento de quejas financieras en Colombia
### Pipeline integral de Big Data con Machine Learning, almacenamiento híbrido y dashboard contenerizado

**Proyecto final — Gerencia y almacenamiento en Big Data** · Autor: **Juan Rojas** · Modalidad individual

**Video de sustentación:** _[pendiente: pegar aquí el enlace no listado]_

---

## 1. Descripción

El proyecto construye un producto de datos completo sobre el dataset oficial **"Quejas interpuestas por los consumidores financieros en contra de las entidades vigiladas en la plataforma Smartsupervision"** de la Superintendencia Financiera de Colombia (SFC).

| Dato | Valor |
|---|---|
| Fuente | SFC vía [datos.gov.co](https://www.datos.gov.co/d/xyy7-rn7p) (id `xyy7-rn7p`) |
| Periodo | Enero de 2023 a septiembre de 2026 |
| Volumen | 1.400.425 filas × 10 columnas (≈ 10 millones de quejas) |
| Peso | 221,8 MB en CSV · 9 MB en Parquet limpio |

**Pregunta de negocio:** ¿qué características de una queja financiera (producto, motivo, tipo de entidad, ubicación y momento) aumentan la probabilidad de que **escale** al Defensor del Consumidor o a la Superintendencia en lugar de resolverse en la propia entidad?

**Tarea de ML:** clasificación binaria de la variable `escalada` (instancia de recepción distinta de *Entidad vigilada*), con validación temporal (2023-2025 entrenamiento, 2026 prueba).

## 2. Arquitectura

```
                 ┌─────────────────────────────── red interna de Docker ───────────────────────────────┐
  datos.gov.co ─►│  procesamiento (Jupyter + pandas)  ──insights──►  mongodb (mongo:7.0)               │
  (o GitHub)     │      │  descarga → ETL → modelo → PDF                    ▲                           │
                 │      ▼                                                    │ lee insights              │
                 │  volumen ./data  ./models  ./reports  ───── lee ────►  app (Streamlit)               │
                 └──────────────────────────────────────────────────────────────────────────────────────┘
```

- **procesamiento:** descarga el dataset, ejecuta el ETL y deja Jupyter Lab listo para correr el notebook.
- **mongodb:** almacena el documento de análisis (metadata, calidad, estadísticas, modelo, insights, recomendaciones). Persistencia en el volumen `mongo_datos`.
- **app:** dashboard Streamlit que **solo lee** resultados ya procesados (Parquet, modelo, MongoDB y PDF); no repite el ETL ni reentrena.

**Decisión pandas vs. Spark:** se usa **pandas**. Con tipos optimizados el dataset ocupa ~25 MB en memoria (−91 %), se lee en ~2 s y el ETL completo tarda < 30 s en una máquina de 2 núcleos. Las mediciones están en la sección 2 del notebook.

## 3. Estructura del repositorio

```
Rojas_Juan_BigData_ProyectoFinal/
├── README.md
├── docker-compose.yml
├── .env.example
├── data/
│   ├── raw/                 # Vacía en la entrega; la llena src/descargar_datos.py
│   ├── processed/           # Parquet limpio + agregaciones OLAP
│   └── reference/           # Catálogos de enriquecimiento (tipos de entidad SFC, regiones DANE)
├── notebooks/
│   └── Rojas_Juan_BigData_ProyectoFinal.ipynb
├── src/
│   ├── config.py            # Rutas relativas y parámetros
│   ├── descargar_datos.py   # Descarga a data/raw/ (oficial con respaldo en GitHub)
│   └── etl.py               # Diagnóstico de calidad y 4 técnicas ETL
├── processing/              # Dockerfile y requirements del servicio Jupyter
├── models/                  # Modelo entrenado
├── reports/                 # Reporte PDF generado
└── app/
    ├── Dockerfile
    └── app.py
```

## 4. Requisitos

- Docker Desktop (o Docker Engine) con **Docker Compose v2**.
- ~4 GB de RAM libres y ~1 GB de disco.
- Conexión a internet la primera vez (descarga de imágenes y del dataset).

## 5. Cómo ejecutar

```bash
git clone https://github.com/ingjuancrojasl-crypto/Rojas_Juan_BigData_ProyectoFinal.git
cd Rojas_Juan_BigData_ProyectoFinal
cp .env.example .env          # en Windows PowerShell: copy .env.example .env
docker compose up --build
```

### Orden de ejecución
1. **mongodb** arranca y espera a estar sano (`healthcheck` con `ping`).
2. **procesamiento** descarga el dataset a `data/raw/` (si no existe), ejecuta el ETL (si no existe el Parquet) y abre Jupyter Lab.
3. **app** arranca Streamlit y lee los resultados.

### Puertos

| Servicio | URL | Notas |
|---|---|---|
| Jupyter Lab | http://localhost:8888 | Token = `JUPYTER_TOKEN` del `.env` |
| Dashboard | http://localhost:8501 | |
| MongoDB | `localhost:27017` | Usuario y clave del `.env` |

### Cómo verificar que todo funciona
1. `docker compose ps` muestra los tres servicios en estado `running` y mongodb como `healthy`.
2. Los logs de procesamiento (`docker compose logs procesamiento`) terminan con `ETL terminado` y la URL de Jupyter.
3. Existe `data/processed/quejas_limpio.parquet`.
4. En http://localhost:8501 aparecen los indicadores (1.400.348 registros, 10.041.272 quejas) y, en *Estado de los servicios*, el mensaje "MongoDB responde correctamente".
5. En Jupyter, abrir `notebooks/Rojas_Juan_BigData_ProyectoFinal.ipynb` y ejecutar **Kernel → Restart & Run All** sin errores.

### Ejecutar sin Docker (opcional)
```bash
pip install -r processing/requirements.txt
python src/descargar_datos.py
python src/etl.py
```

## 6. Problemas de infraestructura documentados

### 6.1 La fuente oficial rechaza la descarga (HTTP 403)
- **Síntoma:** `src/descargar_datos.py` mostró `Falló la fuente 'oficial': Tunnel connection failed: 403 Forbidden` al ejecutarse en un entorno con salida a internet restringida por proxy.
- **Diagnóstico:**
  1. La URL abría en el navegador, así que el dataset seguía publicado.
  2. `curl` desde el mismo entorno devolvía `CONNECT tunnel failed, response 403`: el bloqueo estaba en el proxy de red, no en datos.gov.co.
  3. Además, la URL `.../resource/xyy7-rn7p.csv` solo devolvía **1.000 filas** (límite por defecto de la API Socrata), lo que habría dado un dataset incompleto sin ningún error visible.
- **Solución:** el script intenta primero la descarga completa oficial (`/api/views/.../rows.csv?accessType=DOWNLOAD`) y, si falla, usa un respaldo comprimido en GitHub (20 MB). También valida que lleguen más de 1.000.000 de filas para no aceptar la vista limitada de la API, y registra la fuente usada en `data/raw/metadata_descarga.json`.

### 6.2 `entrypoint.sh` falla en Windows por finales de línea CRLF
- **Síntoma:** al clonar en Windows, el contenedor de procesamiento puede terminar con `/bin/sh^M: bad interpreter` o `no such file or directory`.
- **Diagnóstico:** Git en Windows convierte por defecto `LF` → `CRLF`; el `\r` final hace que Linux no reconozca el intérprete.
- **Solución:** `.gitattributes` fuerza `*.sh text eol=lf`, de modo que el script siempre se descarga con finales de línea Unix.

## 7. Estado del proyecto

| Paso de la guía | Estado |
|---|---|
| 1. Business Understanding y selección del dataset | ✅ |
| 2. Decisión pandas vs. Spark | ✅ |
| 3. ETL con CRISP-DM (4 técnicas, conteos antes/después, Parquet) | ✅ |
| 4. Análisis exploratorio y visualizaciones | ⏳ |
| 5. Machine Learning | ⏳ |
| 6. Insights en MongoDB | ⏳ |
| 7. Reporte PDF | ⏳ |
| 8. Dashboard Streamlit (5 secciones) | ⏳ versión inicial |
| 9. Docker | ✅ estructura base |
