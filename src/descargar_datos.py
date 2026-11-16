"""
Descarga las 4 fuentes del proyecto y las deja en data/raw/.

Para cada fuente:
  1. Intenta la descarga completa oficial (datos.gov.co).
  2. Si falla, usa el respaldo comprimido del repositorio en GitHub.
  3. Valida que el archivo tenga el mínimo de filas esperado (evita aceptar
     vistas parciales de la API, que por defecto devuelve solo 1.000 filas).
  4. Registra fuente, URL, fecha, filas y peso en data/raw/metadata_descarga.json.

Uso:
    python src/descargar_datos.py             # descarga solo lo que falte
    python src/descargar_datos.py --forzar    # vuelve a descargar todo
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

BLOQUE = 1024 * 1024


def _descargar(url: str, timeout: int = 600) -> bytes:
    solicitud = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (proyecto-academico)"})
    with urllib.request.urlopen(solicitud, timeout=timeout) as respuesta:
        buffer, total = io.BytesIO(), 0
        while bloque := respuesta.read(BLOQUE):
            buffer.write(bloque)
            total += len(bloque)
            print(f"\r    {total / 1e6:,.1f} MB", end="", flush=True)
        print()
        return buffer.getvalue()


def _contar_filas(ruta: Path) -> int:
    with open(ruta, "rb") as archivo:
        return sum(b.count(b"\n") for b in iter(lambda: archivo.read(BLOQUE), b"")) - 1


def _oficial(fuente: dict, destino: Path) -> str:
    destino.write_bytes(_descargar(fuente["url_oficial"]))
    return fuente["url_oficial"]


def _respaldo(fuente: dict, destino: Path) -> str:
    with zipfile.ZipFile(io.BytesIO(_descargar(fuente["url_respaldo"]))) as archivo_zip:
        nombre_csv = next(n for n in archivo_zip.namelist() if n.lower().endswith(".csv"))
        with archivo_zip.open(nombre_csv) as origen, open(destino, "wb") as salida:
            shutil.copyfileobj(origen, salida)
    return fuente["url_respaldo"]


def descargar_fuente(nombre: str, fuente: dict, modo: str, forzar: bool) -> dict:
    destino: Path = fuente["archivo"]
    if destino.exists() and not forzar:
        print(f"✓ {nombre}: ya existe ({destino.stat().st_size / 1e6:,.1f} MB)")
        return {"fuente_usada": "existente", "url_descarga": fuente["url_oficial"], "filas": _contar_filas(destino)}

    metodos = {"oficial": [("oficial", _oficial)], "github": [("github", _respaldo)],
               "auto": [("oficial", _oficial), ("github", _respaldo)]}[modo]
    temporal = destino.with_suffix(".tmp")
    for etiqueta, metodo in metodos:
        print(f"→ {nombre}: intentando fuente {etiqueta}")
        try:
            url = metodo(fuente, temporal)
            filas = _contar_filas(temporal)
            if filas < fuente["filas_minimas"]:
                raise RuntimeError(f"solo {filas:,} filas (mínimo esperado {fuente['filas_minimas']:,})")
            temporal.replace(destino)
            print(f"✓ {nombre}: {filas:,} filas, {destino.stat().st_size / 1e6:,.1f} MB")
            return {"fuente_usada": etiqueta, "url_descarga": url, "filas": filas}
        except Exception as error:
            print(f"  ✗ falló ({error})")
            temporal.unlink(missing_ok=True)
    raise SystemExit(f"No se pudo descargar '{nombre}' desde ninguna fuente.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forzar", action="store_true")
    parser.add_argument("--fuente", choices=["auto", "oficial", "github"], default="auto")
    args = parser.parse_args()
    config.crear_directorios()

    previa = {}
    if config.ARCHIVO_METADATA_DESCARGA.exists():
        previa = json.loads(config.ARCHIVO_METADATA_DESCARGA.read_text(encoding="utf-8"))

    metadata = {}
    for nombre, fuente in config.FUENTES.items():
        resultado = descargar_fuente(nombre, fuente, args.fuente, args.forzar)
        if resultado["fuente_usada"] == "existente" and nombre in previa:
            metadata[nombre] = previa[nombre]
            continue
        metadata[nombre] = {
            "titulo": fuente["titulo"], "entidad": fuente["entidad"], "pagina": fuente["pagina"],
            "fuente_usada": resultado["fuente_usada"], "url_descarga": resultado["url_descarga"],
            "fecha_descarga": datetime.now().isoformat(timespec="seconds"), "filas": resultado["filas"],
            "peso_mb": round(fuente["archivo"].stat().st_size / 1e6, 1),
        }
    config.ARCHIVO_METADATA_DESCARGA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadatos en {config.ARCHIVO_METADATA_DESCARGA.relative_to(config.RAIZ_PROYECTO)}")


if __name__ == "__main__":
    main()
