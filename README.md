# Gravedad de los siniestros de transporte en Colombia (2015-2024)
### Pipeline integral de Big Data con Machine Learning, almacenamiento híbrido y dashboard contenerizado

**Proyecto final — Gerencia y almacenamiento en Big Data** · Autor: **Juan Rojas** · Modalidad individual

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
  GitHub (.zip) ►│  procesamiento (Jupyter + pandas)  ──insights──►  mongodb (mongo:7.0)               │
  (o datos.gov.co│      │  descarga → ETL → modelo → PDF                    ▲                           │
   como respaldo)│      ▼                                                    │ lee insights              │
                 │  volumen ./data  ./models  ./reports  ───── lee ────►  app (Streamlit)               │
                 └──────────────────────────────────────────────────────────────────────────────────────┘
```

- **procesamiento:** descarga las 4 fuentes desde los .zip del repositorio en GitHub (con datos.gov.co como respaldo), ejecuta el ETL y deja Jupyter Lab listo para correr el notebook.
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
│   └── visualizaciones.py    # Gráficas del análisis exploratorio (principios de la Actividad 7)
├── *.zip                     # 4 CSV originales comprimidos (fuente de descarga del proyecto)
├── processing/               # Dockerfile y requirements del servicio Jupyter
├── models/                   # Modelo entrenado
├── reports/                  # Reporte PDF y figuras/ (gráficas PNG + registro visualizaciones.json)
└── app/
    ├── Dockerfile
    └── app.py
```

## 4. Requisitos

- Docker Desktop (o Docker Engine) con **Docker Compose v2**.
- ~4 GB de RAM libres para el contenedor de procesamiento y ~2 GB de disco.
- Conexión a internet la primera vez (imágenes de Docker y ~26 MB de datos comprimidos).

## 5. Cómo ejecutar

```bash
git clone https://github.com/ingjuancrojasl-crypto/Rojas_Juan_BigData_ProyectoFinal.git
cd Rojas_Juan_BigData_ProyectoFinal
cp .env.example .env          # en Windows PowerShell: copy .env.example .env
docker compose up --build
```

### Orden de ejecución
1. **mongodb** arranca y espera a estar sano (`healthcheck` con `ping`).
2. **procesamiento** descarga las 4 fuentes desde GitHub a `data/raw/` (si faltan), ejecuta el ETL (si no existen los Parquet) y abre Jupyter Lab.
3. **app** arranca Streamlit y lee los resultados.

### Puertos

| Servicio | URL | Notas |
|---|---|---|
| Jupyter Lab | http://localhost:8888 | Token = `JUPYTER_TOKEN` del `.env` |
| Dashboard | http://localhost:8501 | |
| MongoDB | `localhost:27017` | Usuario y clave del `.env` |

### Cómo verificar que todo funciona
1. `docker compose ps` muestra los tres servicios en `running` y mongodb como `healthy`.
2. `docker compose logs procesamiento` termina con `ETL terminado` y la URL de Jupyter.
3. Existen `data/processed/siniestros_unificado.parquet` y `data/processed/victimas_medicina_legal.parquet`.
4. En http://localhost:8501 aparecen las 5 pestañas del dashboard con 960.623 víctimas (4 fuentes) y, en la barra lateral, "Documento de análisis: MongoDB ✅".
5. En Jupyter, abrir `notebooks/Rojas_Juan_BigData_ProyectoFinal.ipynb` y ejecutar **Kernel → Restart & Run All** sin errores.

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

### 6.5 Límite de 25 MB por archivo en la carga web de GitHub
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
| 9. Docker | ✅ estructura base |
