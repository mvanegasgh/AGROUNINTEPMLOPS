"""EDA + pipeline de limpieza + extraccion de caracteristicas.

Uso: python -m src.eda_limpieza [--raw dataset/raw]
Entrada : dataset/raw/<clase>/*.jpg
Salidas : dataset/limpio/<clase>/, data/features.csv, reports/eda.png,
          reports/inventario_completo.csv
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import features
from .common import CLEAN, DATA, IMG_EXT, RAW, REPORTS, ROOT


def inventariar(raw, min_lado, umbral_blur):
    filas, vistos = [], set()
    clases = sorted(p for p in raw.iterdir() if p.is_dir())
    if not clases:
        raise SystemExit(f"No hay carpetas de clase en {raw}")
    for cdir in clases:
        for f in sorted(cdir.rglob("*")):
            if f.suffix.lower() not in IMG_EXT:
                continue
            fila = dict(path=str(f), clase=cdir.name, motivo="ok",
                        grupo=f.stem.split("__")[0] if "__" in f.stem else None)
            data = f.read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                fila["motivo"] = "corrupta"
                filas.append(fila)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            md5 = hashlib.md5(data).hexdigest()
            fila.update(alto=h, ancho=w, brillo=float(gray.mean()),
                        nitidez=float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            if md5 in vistos:
                fila["motivo"] = "duplicada"
            elif min(h, w) < min_lado:
                fila["motivo"] = "muy_pequena"
            elif fila["nitidez"] < umbral_blur:
                fila["motivo"] = "borrosa"
            elif fila["brillo"] < 25:
                fila["motivo"] = "muy_oscura"
            elif fila["brillo"] > 240:
                fila["motivo"] = "sobreexpuesta"
            vistos.add(md5)
            filas.append(fila)
    return pd.DataFrame(filas)


def graficar(inv, limpio):
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    inv.groupby("clase").size().plot.bar(ax=ax[0], title="Imagenes por clase (antes)")
    limpio.groupby("clase").size().plot.bar(ax=ax[1], color="seagreen",
                                            title="Imagenes por clase (despues)")
    inv.dropna(subset=["brillo"]).boxplot(column="brillo", by="clase", ax=ax[2])
    ax[2].set_title("Brillo por clase")
    desc = inv[inv.motivo != "ok"].groupby("motivo").size()
    if len(desc):
        desc.plot.barh(ax=ax[3], color="indianred")
    ax[3].set_title("Descartadas por motivo")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(REPORTS / "eda.png", dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--min-lado", type=int, default=64)
    ap.add_argument("--umbral-blur", type=float, default=20.0)
    a = ap.parse_args()

    inv = inventariar(Path(a.raw), a.min_lado, a.umbral_blur)
    inv.to_csv(REPORTS / "inventario_completo.csv", index=False)
    limpio = inv[inv.motivo == "ok"].copy()

    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    filas = []
    for i, r in enumerate(limpio.itertuples()):
        dst = CLEAN / r.clase / f"{i:05d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imdecode(np.fromfile(r.path, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(dst), img)
        g = r.grupo if isinstance(r.grupo, str) else f"sin_grupo_{i}"
        fila = {"path": dst.relative_to(ROOT).as_posix(), "clase": r.clase, "grupo": g}
        fila.update({f"f_{k}": v for k, v in enumerate(features.extract(img))})
        filas.append(fila)
    pd.DataFrame(filas).to_csv(DATA / "features.csv", index=False)
    graficar(inv, limpio)

    print(f"Total: {len(inv)} | validas: {len(limpio)} | descartadas: {len(inv) - len(limpio)}")
    desc = inv[inv.motivo != "ok"].motivo.value_counts()
    print(desc.to_string() if len(desc) else "Sin descartes")
    conteo = limpio.clase.value_counts()
    print("\nImagenes por clase (limpias):\n" + conteo.to_string())
    if len(conteo) > 4:
        print("AVISO: el plan recomienda maximo 3-4 clases.")
    if conteo.max() / conteo.min() > 3:
        print("AVISO: desbalance > 3:1; considera mas imagenes de la clase minoritaria.")
    fd = pd.DataFrame(filas)
    propios = fd[~fd.grupo.str.startswith(("pub", "sin_grupo"))]
    print(f"\nPersonas/sesiones propias: {propios.grupo.nunique()} | publicas: "
          f"{fd.grupo.str.startswith('pub').sum()} imgs | sin grupo: "
          f"{fd.grupo.str.startswith('sin_grupo').sum()} imgs")
    if propios.grupo.nunique() < 4:
        print("AVISO: menos de 4 personas propias; el test por persona sera poco confiable (ideal >= 6).")
    por_clase = propios.groupby("clase").grupo.nunique()
    if len(por_clase) and por_clase.min() < 3:
        print("AVISO: alguna clase tiene menos de 3 personas distintas:\n" + por_clase.to_string())
    print("\nCaracteristicas: data/features.csv | Grafica: reports/eda.png")


if __name__ == "__main__":
    main()
