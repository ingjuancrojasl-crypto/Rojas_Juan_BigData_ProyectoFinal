# Gravedad de los siniestros de transporte en Colombia (2015-2024)
### Pipeline integral de Big Data con Machine Learning, almacenamiento híbrido y dashboard contenerizado

**Proyecto final — Gerencia y almacenamiento en Big Data** · Autor: **Juan Rojas** · Modalidad individual

**Dashboard en línea:** https://siniestros-colombia-jrojas.streamlit.app/ (Streamlit Community Cloud; los insights se muestran desde la copia exportada del documento de MongoDB)

**Video de sustentación:** _[pendiente: pegar aquí el enlace no listado]_

---

## 1. Descripción

El proyecto integra **4 fuentes oficiales** de [datos.gov.co](https://www.datos.gov.co) sobre víctimas de siniestros de transporte y construye un producto de datos completo: ETL, análisis, modelo de Machine Learning, almacenamiento en MongoDB, reporte PDF y dashboard.

| Fuente | Entidad | Id | Filas crudas | Periodo | Qué registra |
|---|---|---|---|---|---|
| [Lesiones por eventos de transporte](https://www.datos.gov.co/d/ezhf-hscf) | Medicina Legal (INMLCF) | `ezhf-hscf` | 342.796 | 2015-2024 | Víctimas lesionadas con examen médico-legal (39 variables) |
| [Muertes por eventos de transporte](https://www.datos.gov.co/d/s65h-7665) | Medicina Legal (INMLCF) | `s65h-7665` | 73.403 | 2015-2024 | Víctimas fallecidas con necropsia (38 variables) |
| [Lesiones personales y en accidente de tránsito](https://www.datos.gov.co/d/72sg-cybi) | Policía Nacional | `72sg-cybi` | 1.381.416 | 2010-2026 | Denuncias de lesiones (se filtran las de tránsito) |
| [Homicidios accidente de tránsito](https://www.datos.gov.co/d/ha6j-pa2r) | Policía Nacional | `ha6j-pa2r` | 98.430 | 2010-2026 | Denuncias de homicidio culposo en tránsito |

Tras filtrar tránsito y el periodo común **2015-2024**: **767.313 registros**, equivalentes a **960.623 víctimas** (algunas filas de la Policía agrupan varias víctimas). Peso: 492 MB en CSV · 14 MB en Parquet.

**Pregunta de negocio:** ¿Qué características de la víctima, del vehículo y del contexto se asocian con la gravedad de un siniestro de transporte (lesión leve, lesión grave o muerte), y cuáles son las causas más frecuentes en lesionados y fallecidos?

**Tarea de ML:** clasificación multiclase de `gravedad` (lesión leve: 0-30 días de incapacidad; lesión grave: > 30 días; muerte), con validación temporal (2015-2022 entrenamiento, 2023-2024 prueba).

## 2. Arquitectura

```
                 ┌─────────────────────────────── red interna de Docker ───────────────────────────────┐
  GitHub (.zip) ►│  procesamiento (Jupyter + pandas)  ──documento──►  mongodb (mongo:7.0)              │
  (o datos.gov.co│      │  descarga → ETL → gráficas → modelo → PDF        ▲                           │
   como respaldo)│      ▼                                                    │ lee insights              │
                 │  volumen ./data  ./models  ./reports  ───── lee ────►  app (Streamlit)               │
                 └──────────────────────────────────────────────────────────────────────────────────────┘
```

- **procesamiento:** ejecuta `src/pipeline_docker.py`. En un clon nuevo corre el **notebook completo sin intervención** (descarga desde GitHub con datos.gov.co como respaldo → ETL → gráficas → modelo → MongoDB → PDF); si los resultados ya existen, solo guarda el documento en MongoDB y regenera el PDF. Después deja Jupyter Lab abierto.
- **mongodb:** almacena el documento de análisis (metadata, calidad, estadísticas, modelo, insights, recomendaciones). Persistencia en el volumen `mongo_datos`.
- **app:** dashboard Streamlit que **solo lee** resultados ya procesados (Parquet, modelo, MongoDB y PDF).

**Decisión pandas vs. Spark:** **pandas**. Los 4 CSV ocupan ~640 MB en memoria como texto y ~43 MB con tipos `category` (−93 %); se leen en ~5 s y el ETL completo tarda ~20 segundos con un pico de ~2,2 GB de RAM. Mediciones en la sección 2 del notebook.

## 3. Estructura del repositorio

```
Rojas_Juan_BigData_ProyectoFinal/
├── README.md
├── docker-compose.yml
├── .env.example
├── data/
│   ├── raw/                  # Vacía en la entrega; la llena src/descargar_datos.py
│   ├── processed/            # Parquet: base analítica, base unificada y cubos OLAP
│   └── reference/            # Regiones (DANE) y catálogo de causas aplicado
├── notebooks/
│   └── Rojas_Juan_BigData_ProyectoFinal.ipynb
├── src/
│   ├── config.py             # Rutas relativas, fuentes y parámetros
│   ├── descargar_datos.py    # Descarga de las 4 fuentes (GitHub, con datos.gov.co como respaldo)
│   ├── etl.py                # Diagnóstico de calidad, unificación y 4 técnicas ETL
│   ├── visualizaciones.py    # Gráficas del análisis exploratorio y del modelo
│   ├── modelo.py             # Preparación, entrenamiento, evaluación y guardado del modelo
│   ├── mongo_analisis.py     # Documento de análisis en MongoDB (sección 6)
│   ├── reporte_pdf.py        # Reporte PDF automático con ReportLab (sección 7)
│   └── pipeline_docker.py    # Orquestación del servicio de procesamiento en Docker
├── *.zip                     # 4 CSV originales comprimidos (fuente de descarga del proyecto)
├── processing/               # Dockerfile, requirements y entrypoint del servicio de procesamiento
├── .streamlit/config.toml    # Tema azul del dashboard
├── .dockerignore             # Excluye datos y modelos del build de Docker
├── models/                   # Modelo entrenado
├── reports/                  # Reporte PDF y figuras/ (gráficas PNG + registro visualizaciones.json)
└── app/
    ├── Dockerfile
    └── app.py
```

## 4. Requisitos

- Docker Desktop (o Docker Engine) con **Docker Compose v2**.
- ~5 GB de RAM asignados a Docker (el procesamiento llega a ~3 GB) y ~3 GB de disco.
- Puertos libres: 8888, 8501 y 27017 (se pueden cambiar en el `.env`).
- Conexión a internet la primera vez (imágenes de Docker y ~26 MB de datos comprimidos).

## 5. Cómo ejecutar

```bash
git clone https://github.com/ingjuancrojasl-crypto/Rojas_Juan_BigData_ProyectoFinal.git
cd Rojas_Juan_BigData_ProyectoFinal
cp .env.example .env          # en Windows PowerShell: copy .env.example .env
docker compose up --build
```

### Orden de ejecución
1. **mongodb** arranca con el usuario y la clave del `.env` y pasa a `healthy` cuando responde a `ping`.
2. **procesamiento** espera a MongoDB y ejecuta `src/pipeline_docker.py`:
   - **primera vez** (sin Parquet ni modelo): ejecuta el notebook completo — descarga (~26 MB), ETL, 11 gráficas, modelo, documento en MongoDB y PDF. Tarda unos **5-10 minutos**; la copia ejecutada queda en `reports/notebook_ejecutado_docker.ipynb`;
   - **siguientes veces**: no repite el ETL ni el entrenamiento; solo guarda el documento en MongoDB y regenera el PDF (segundos).
   Al terminar pasa a `healthy` y deja Jupyter Lab abierto.
3. **app** espera a que procesamiento esté `healthy` y arranca Streamlit.

Para repetir todo desde cero: `FORZAR_PIPELINE=1` en el `.env` (o borrar `data/processed/`).

### Puertos

| Servicio | URL | Credenciales |
|---|---|---|
| Dashboard (app) | http://localhost:8501 | — |
| Jupyter Lab (procesamiento) | http://localhost:8888/?token=TOKEN | `JUPYTER_TOKEN` del `.env` |
| MongoDB | `mongodb://localhost:27017` (Compass) | `MONGO_USER` / `MONGO_PASSWORD` del `.env` |

Dentro de la red de Docker los servicios se comunican por nombre (`mongodb:27017`), no por `localhost`.

### Volúmenes
| Volumen | Uso |
|---|---|
| `mongo_datos` (con nombre) | Persistencia de MongoDB: el documento sobrevive a `docker compose down` |
| `./` → `/proyecto` (procesamiento, lectura y escritura) | Código, `data/`, `models/` y `reports/` compartidos con el equipo |
| `./` → `/proyecto:ro` (app, solo lectura) | La app solo lee resultados |

### Cómo verificar que todo funciona
1. `docker compose ps` muestra los tres servicios `Up` y los tres `(healthy)`.
2. `docker compose logs procesamiento` muestra `PIPELINE LISTO` y la URL de Jupyter.
3. Existen `data/processed/siniestros_unificado.parquet`, `models/modelo_gravedad.joblib` y `reports/Rojas_Juan_Reporte_Siniestros.pdf`.
4. En http://localhost:8501 aparecen las 5 pestañas con 960.623 víctimas y, en la barra lateral, **"Documento de análisis: MongoDB ✅"**; en *Insights y reporte*, el botón descarga el PDF.
5. En MongoDB Compass (`mongodb://USUARIO:CLAVE@localhost:27017/?authSource=admin`) existe la base `siniestros_transporte` con la colección `analisis`.
6. Para detener: `docker compose down` (conserva los datos de MongoDB); `docker compose down -v` también borra el volumen.

### Ejecutar sin Docker (opcional)
```bash
pip install -r processing/requirements.txt
python src/descargar_datos.py
python src/etl.py
python src/reporte_pdf.py          # requiere MongoDB encendido y la sección 6 ejecutada
pip install -r app/requirements.txt
streamlit run app/app.py           # abre http://localhost:8501
```

### Dashboard (Streamlit)
| Pestaña | Contenido |
|---|---|
| 📊 Resumen | KPIs del dataset y del modelo, evolución anual de muertes y lesionados |
| 🗂️ Datos | Muestra filtrable del dataset limpio (descarga CSV) y cubos OLAP con roll-up y pivot |
| 📈 Visualizaciones | 7 gráficas interactivas (Plotly) con filtros de año, región y medio de transporte |
| 🤖 Modelo | Métricas, comparación con las líneas base, matriz de confusión, importancia y simulador de predicción |
| 💡 Insights y reporte | Insights y recomendaciones leídos desde MongoDB (filtro por tema y prioridad) y descarga del PDF |

**Versión en línea:** https://siniestros-colombia-jrojas.streamlit.app/ — desplegada desde este repositorio (archivo principal `app/app.py`, Python 3.11). Como en la nube no hay acceso al MongoDB local, la pestaña *Insights y reporte* usa `reports/documento_mongodb.json`, la copia exportada del mismo documento, y lo indica en pantalla. Cada cambio subido a GitHub actualiza la app.

La app **no repite el ETL ni reentrena**: lee los Parquet de `data/processed/`, el modelo de `models/` y el documento de MongoDB (si MongoDB no responde, muestra la copia `reports/documento_mongodb.json` y lo avisa).

## 6. Problemas de infraestructura documentados

### 6.1 La descarga oficial falla o llega incompleta
- **Síntoma:** `src/descargar_datos.py` mostró `falló (Tunnel connection failed: 403 Forbidden)` en un entorno con salida a internet restringida por proxy. Además, la URL `.../resource/<id>.csv` devuelve solo **1.000 filas** (límite por defecto de la API Socrata), sin ningún error visible.
- **Diagnóstico:** la URL abría en el navegador (el dataset seguía publicado) pero `curl` desde el mismo entorno devolvía `CONNECT tunnel failed, response 403`: el bloqueo estaba en el proxy de red.
- **Solución:** los 4 CSV originales se publicaron comprimidos en la raíz del repositorio y el script los descarga desde GitHub como fuente principal (~26 MB). Si GitHub falla, usa la descarga completa oficial (`/api/views/<id>/rows.csv?accessType=DOWNLOAD`, no la vista limitada a 1.000 filas). En ambos casos valida un mínimo de filas por fuente y registra la fuente usada en `data/raw/metadata_descarga.json`.

### 6.2 `entrypoint.sh` falla en Windows por finales de línea CRLF
- **Síntoma:** al clonar en Windows, el contenedor de procesamiento puede terminar con `/bin/sh^M: bad interpreter`.
- **Diagnóstico:** Git en Windows convierte `LF` → `CRLF`; el `\r` final impide que Linux reconozca el intérprete.
- **Solución:** `.gitattributes` fuerza `*.sh text eol=lf`.

### 6.3 El script de descarga se detiene en Windows sin mostrar el motivo
- **Síntoma:** en el notebook, `subprocess.run(...descargar_datos.py)` terminaba con `CalledProcessError` y sin mensaje útil.
- **Diagnóstico:** el error real ocurría en el proceso hijo: la consola de Windows usa la codificación cp1252 y no puede imprimir caracteres como `✓` o `→` (`UnicodeEncodeError`). Además, los respaldos estaban en la raíz del repositorio y no en la carpeta que esperaba el script.
- **Solución:** el script reconfigura su salida a UTF-8, el notebook llama a `descargar_datos.descargar_todo()` en el mismo proceso (los errores se ven completos en la celda) y la URL de respaldo apunta a la raíz del repositorio.

### 6.4 MongoDB rechaza la escritura aunque la conexión responde
- **Síntoma:** en la sección 6, la conexión mostraba éxito, pero al guardar aparecía `AutoReconnect: connection closed` y luego `OperationFailure: Command insert requires authentication`. Antes, con el contenedor apagado, el error era `ServerSelectionTimeoutError`.
- **Diagnóstico:** el comando `ping` responde aunque no haya usuario; el archivo `.env` no existía, así que la URI se armaba sin credenciales (`mongodb://localhost:27017/`) y el servidor rechazaba toda escritura.
- **Solución:** crear `.env` en la raíz con `MONGO_USER`/`MONGO_PASSWORD` (desde PowerShell con `Set-Content` para evitar el `.env.txt` del Bloc de notas), reiniciar el kernel y validar la autenticación al conectar (`list_collection_names()`), que ahora muestra un mensaje claro con la ruta esperada del `.env`.

### 6.5 Conflicto del puerto 27017 con otro contenedor de MongoDB
- **Síntoma:** `docker compose up` falla con `Bind for 0.0.0.0:27017 failed: port is already allocated`.
- **Diagnóstico:** `docker ps` mostró el contenedor `mongodb` de una actividad anterior (imagen `mongo:latest`) publicado en `27017:27017`; dos contenedores no pueden publicar el mismo puerto del equipo.
- **Solución:** detener el otro contenedor (`docker stop mongodb`) o cambiar `MONGO_PORT` en el `.env` (por ejemplo `27018`). El puerto interno no cambia: los servicios usan `mongodb:27017` dentro de la red de Docker. Además, el proyecto fija `mongo:7.0` en lugar de `latest` para que todos usen la misma versión.

### 6.6 El build de Docker enviaba ~500 MB de datos
- **Síntoma:** `docker compose up --build` tardaba en "transferring context" antes de empezar a construir.
- **Diagnóstico:** el *build context* es la carpeta del proyecto; Docker enviaba `data/raw/` (~470 MB), los .zip y los modelos aunque los Dockerfile solo copian `requirements.txt` y el `entrypoint.sh`.
- **Solución:** `.dockerignore` excluye `data/`, `models/`, `reports/`, `*.zip`, `.env` y cachés; el proyecto se monta como volumen al ejecutar.

### 6.7 Límite de 25 MB por archivo en la carga web de GitHub
- **Síntoma:** los CSV originales (hasta 238 MB) no se pueden subir al repositorio.
- **Solución:** cada fuente se comprime por separado en un .zip en la raíz del repositorio (entre 0,9 y 12 MB cada uno) y `data/raw/` queda fuera del control de versiones (`.gitignore`).

## 7. Estado del proyecto

| Paso de la guía | Estado |
|---|---|
| 1. Business Understanding y selección del dataset | ✅ |
| 2. Decisión pandas vs. Spark | ✅ |
| 3. ETL con CRISP-DM (unificación de 4 fuentes, 4 técnicas, Parquet, OLAP) | ✅ |
| 4. Análisis exploratorio y visualizaciones (8 gráficas + 3 consultas OLAP) | ✅ |
| 5. Machine Learning | ✅ |
| 6. Insights en MongoDB | ✅ |
| 7. Reporte PDF | ✅ |
| 8. Dashboard Streamlit (5 secciones) | ✅ |
| 9. Docker Compose (3 servicios, un solo comando) | ✅ |
