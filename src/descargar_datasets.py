"""Descarga datasets PUBLICOS de EPP a  descargas/<nombre>/  (se ejecuta en SU computador).

Uso:
  python -m src.descargar_datasets zenodo                      # dataset de higiene (cofia/guantes)
  python -m src.descargar_datasets kaggle  --id mugheesahmad/sh17-dataset-for-ppe-detection
  python -m src.descargar_datasets roboflow --api-key XXX --workspace proof-87nsb \\
                                           --proyecto food-industry-ppe --version 1
Despues convierta cada uno (ver README, seccion "Datos publicos"):
  python -m src.convertir_yolo_elementos --dataset descargas/<nombre> --elementos cofia,guantes

Requisitos por fuente:
  zenodo   : ninguno (usa la API publica de Zenodo). Registro por defecto 16329852, el dataset del
             articulo "Benchmarking Lightweight YOLO Object Detectors for Real-Time Hygiene
             Compliance Monitoring" (Sensors 2025). Revise la licencia en la pagina del registro.
  kaggle   : pip install kaggle  +  su token en ~/.kaggle/kaggle.json (cuenta gratuita de Kaggle).
  roboflow : pip install roboflow  +  su API key gratuita (Roboflow > Settings > API Keys) y el
             numero de version del proyecto (visible en la pagina del dataset en Universe).
Cite cada dataset y su licencia en el informe. SH17 es CC BY-NC-SA (solo uso educativo/investigacion).
"""
import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESTINO = ROOT / "descargas"
ZENODO_API = "https://zenodo.org/api/records/"


def descargar_archivo(url, destino, tam=None):
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  descargando {destino.name}" + (f" ({tam / 1e6:.0f} MB)" if tam else "") + " ...")
    req = urllib.request.Request(url, headers={"User-Agent": "agrointep-descarga/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(destino, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def descomprimir(zip_path, destino):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    zip_path.unlink()
    print(f"  descomprimido en {destino}")


def zenodo(a):
    api = a.api or ZENODO_API
    base = Path(a.salida or DESTINO / f"zenodo{a.registro}")
    print(f"Consultando Zenodo, registro {a.registro} ...")
    with urllib.request.urlopen(f"{api}{a.registro}", timeout=60) as r:
        meta = json.load(r)
    md = meta.get("metadata", {})
    lic = md.get("license", {})
    print(f"Titulo : {md.get('title')}\nLicencia: {lic.get('id', lic) if lic else 'no indicada'}")
    archivos = meta.get("files", [])
    if not archivos:
        raise SystemExit("El registro no lista archivos (puede ser de acceso restringido).")
    for f in archivos:
        print(f"  - {f.get('key')}  {f.get('size', 0) / 1e6:.0f} MB")
    for f in archivos:
        key = f["key"]
        url = (f.get("links") or {}).get("self") or \
            f"https://zenodo.org/records/{a.registro}/files/{key}?download=1"
        destino = base / key
        descargar_archivo(url, destino, f.get("size"))
        if destino.suffix.lower() == ".zip":
            descomprimir(destino, base)
    print(f"\nListo: {base}\nRevise los nombres de clase en el archivo data.yaml de esa carpeta.")


def kaggle(a):
    base = Path(a.salida or DESTINO / a.id.split("/")[-1])
    try:
        subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", a.id, "-p",
                        str(base), "--unzip"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("Fallo la descarga con Kaggle. Verifique: pip install kaggle, su token en "
                         "~/.kaggle/kaggle.json y haber aceptado los terminos del dataset en la web.")
    print(f"\nListo: {base}")


def roboflow(a):
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("Falta el paquete:  pip install roboflow")
    base = Path(a.salida or DESTINO / a.proyecto)
    proyecto = Roboflow(api_key=a.api_key).workspace(a.workspace).project(a.proyecto)
    proyecto.version(a.version).download("yolov8", location=str(base))
    print(f"\nListo: {base}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="fuente", required=True)
    z = sub.add_parser("zenodo")
    z.add_argument("--registro", default="16329852")
    z.add_argument("--api", help=argparse.SUPPRESS)  # solo para pruebas
    z.add_argument("--salida")
    k = sub.add_parser("kaggle")
    k.add_argument("--id", required=True, help="usuario/dataset")
    k.add_argument("--salida")
    r = sub.add_parser("roboflow")
    r.add_argument("--api-key", required=True)
    r.add_argument("--workspace", required=True)
    r.add_argument("--proyecto", required=True)
    r.add_argument("--version", type=int, required=True)
    r.add_argument("--salida")
    a = ap.parse_args()
    {"zenodo": zenodo, "kaggle": kaggle, "roboflow": roboflow}[a.fuente](a)


if __name__ == "__main__":
    main()
