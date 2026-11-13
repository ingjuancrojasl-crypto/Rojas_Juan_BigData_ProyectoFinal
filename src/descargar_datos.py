"""
Descarga el dataset de quejas financieras y lo deja en data/raw/quejas.csv.

Estrategia:
  1. Intenta la fuente oficial (datos.gov.co, descarga completa ~220 MB).
  2. Si falla, usa el respaldo comprimido en GitHub (~20 MB .zip).
  3. Registra en data/raw/metadata_descarga.json la URL usada, la fecha y el peso.

Uso:
    python src/descargar_datos.py            # descarga solo si no existe
    python src/descargar_datos.py --forzar   # vuelve a descargar
    python src/descargar_datos.py --fuente github
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

TAMANO_BLOQUE = 1024 * 1024  # 1 MB
FILAS_ESPERADAS_MINIMAS = 1_000_000  # control para no aceptar la vista de 1.000 filas de la API


def _descargar_bytes(url: str, timeout: int = 300) -> bytes:
    """Descarga una URL completa en memoria mostrando el avance en MB."""
    solicitud = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (proyecto-academico)"})
    with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
        buffer = io.BytesIO()
        descargado = 0
        while True:
            bloque = respuesta.read(TAMANO_BLOQUE)
            if not bloque:
                break
            buffer.write(bloque)
            descargado += len(bloque)
            print(f"\r  descargados {descargado / 1e6:,.1f} MB", end="", flush=True)
        print()
        return buffer.getvalue()


def _contar_filas(ruta_csv: Path) -> int:
    """Cuenta filas de datos (sin encabezado) leyendo por bloques."""
    with open(ruta_csv, "rb") as archivo:
        return sum(bloque.count(b"\n") for bloque in iter(lambda: archivo.read(TAMANO_BLOQUE), b"")) - 1


def descargar_oficial(destino: Path) -> None:
    print(f"Fuente oficial: {config.URL_OFICIAL}")
    contenido = _descargar_bytes(config.URL_OFICIAL)
    destino.write_bytes(contenido)


def descargar_github(destino: Path) -> None:
    print(f"Respaldo GitHub: {config.URL_RESPALDO_GITHUB}")
    contenido = _descargar_bytes(config.URL_RESPALDO_GITHUB)
    with zipfile.ZipFile(io.BytesIO(contenido)) as archivo_zip:
        csvs = [n for n in archivo_zip.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError("El .zip no contiene ningún CSV")
        with archivo_zip.open(csvs[0]) as origen, open(destino, "wb") as salida:
            shutil.copyfileobj(origen, salida)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forzar", action="store_true", help="descargar aunque el archivo ya exista")
    parser.add_argument("--fuente", choices=["auto", "oficial", "github"], default="auto")
    args = parser.parse_args()

    config.crear_directorios()
    destino = config.ARCHIVO_RAW

    if destino.exists() and not args.forzar:
        print(f"Ya existe {destino.relative_to(config.RAIZ_PROYECTO)} "
              f"({destino.stat().st_size / 1e6:,.1f} MB). Use --forzar para volver a descargar.")
        return

    fuentes = {"oficial": [("oficial", descargar_oficial)],
               "github": [("github", descargar_github)],
               "auto": [("oficial", descargar_oficial), ("github", descargar_github)]}[args.fuente]

    temporal = destino.with_suffix(".tmp")
    fuente_usada, url_usada = None, None
    for nombre, funcion in fuentes:
        try:
            funcion(temporal)
            filas = _contar_filas(temporal)
            if filas < FILAS_ESPERADAS_MINIMAS:
                raise RuntimeError(f"solo llegaron {filas:,} filas (¿vista limitada de la API?)")
            fuente_usada = nombre
            url_usada = config.URL_OFICIAL if nombre == "oficial" else config.URL_RESPALDO_GITHUB
            break
        except Exception as error:  # se intenta la siguiente fuente
            print(f"  ✗ Falló la fuente '{nombre}': {error}")
            temporal.unlink(missing_ok=True)

    if fuente_usada is None:
        sys.exit("No se pudo descargar el dataset desde ninguna fuente.")

    temporal.replace(destino)
    metadata = {
        "dataset": config.DATASET_NOMBRE,
        "entidad": config.DATASET_ENTIDAD,
        "pagina": config.URL_PAGINA_DATASET,
        "fuente_usada": fuente_usada,
        "url_descarga": url_usada,
        "fecha_descarga": datetime.now().isoformat(timespec="seconds"),
        "filas": filas,
        "peso_mb": round(destino.stat().st_size / 1e6, 1),
    }
    config.ARCHIVO_METADATA_DESCARGA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Guardado {destino.relative_to(config.RAIZ_PROYECTO)}: {filas:,} filas, {metadata['peso_mb']} MB")


if __name__ == "__main__":
    main()
