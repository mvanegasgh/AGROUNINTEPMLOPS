"""Convierte un dataset publico YOLO (cajas) a  dataset/publico/<elemento>/<si|no>/  para el
entrenamiento POR ELEMENTO (src/multietiqueta.py).

A diferencia de convertir_yolo.py, cada elemento se etiqueta por separado: una imagen con
"hairnet" y sin anotacion de guantes aporta solo la etiqueta de cofia; el resto no cuenta.

Uso:
  python -m src.convertir_yolo_elementos --dataset descargas/hygiene-ppe --elementos cofia,guantes
Opciones:
  --mapeo mapeo_clases.json   vocabulario (pos/neg) de cada EPP; edite segun las clases del dataset
  --max-por-clase 800         tope de imagenes por elemento y por si/no (equilibra el entrenamiento)
Si en una imagen un elemento aparece a la vez como presente y ausente, se omite por ambigua.
Revise la licencia del dataset y citelo en el informe.
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(n):
    return re.sub(r"[\s_]+", "-", str(n).strip().lower())


def cargar_nombres(root):
    yamls = list(root.rglob("data.yaml")) + list(root.rglob("*.yaml"))
    if not yamls:
        raise SystemExit("No se encontro data.yaml con los nombres de clase.")
    names = yaml.safe_load(yamls[0].read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [norm(n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--elementos", default="cofia,guantes,bata")
    ap.add_argument("--mapeo", default=str(ROOT / "mapeo_clases.json"))
    ap.add_argument("--salida", default=str(ROOT / "dataset" / "publico"))
    ap.add_argument("--max-por-clase", type=int, default=0, help="0 = sin limite")
    a = ap.parse_args()

    root, salida = Path(a.dataset), Path(a.salida)
    elementos = [e.strip() for e in a.elementos.split(",") if e.strip()]
    mapeo = json.loads(Path(a.mapeo).read_text(encoding="utf-8"))
    for e in elementos:
        if e not in mapeo:
            raise SystemExit(f"'{e}' no esta en {a.mapeo}. Disponibles: {list(mapeo)}")
    nombres = cargar_nombres(root)
    pos = {n: e for e, v in mapeo.items() for n in v["pos"]}
    neg = {n: e for e, v in mapeo.items() for n in v["neg"]}
    print("Clases del dataset:", nombres)
    print("Reconocidas  +:", {n: pos[n] for n in nombres if n in pos},
          " -:", {n: neg[n] for n in nombres if n in neg})

    tag = re.sub(r"[^a-z0-9]", "", root.name.lower())[:12] or "ds"
    cuenta, ambiguas, n = Counter(), 0, 0
    for lbl in sorted(root.rglob("*.txt")):
        if lbl.parent.name != "labels":
            continue
        img = next((p for p in (lbl.parent.parent / "images").glob(lbl.stem + ".*")
                    if p.suffix.lower() in IMG_EXT), None)
        if img is None:
            continue
        ids = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        vistos = {nombres[i] for i in ids if i < len(nombres)}
        for e in elementos:
            hay_si = any(pos.get(v) == e for v in vistos)
            hay_no = any(neg.get(v) == e for v in vistos)
            if hay_si == hay_no:  # ninguno (no se sabe) o ambos (ambiguo)
                ambiguas += hay_si and hay_no
                continue
            val = "si" if hay_si else "no"
            if a.max_por_clase and cuenta[(e, val)] >= a.max_por_clase:
                continue
            n += 1
            dst = salida / e / val / f"pub{tag}{n}{img.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)
            cuenta[(e, val)] += 1

    print("\nImagenes por elemento:")
    for e in elementos:
        print(f"  {e}: si={cuenta[(e, 'si')]}  no={cuenta[(e, 'no')]}")
    print("Omitidas por ambiguas:", ambiguas)
    for e in elementos:
        if bool(cuenta[(e, "si")]) != bool(cuenta[(e, "no")]):
            lado = "presente" if cuenta[(e, "si")] else "ausente"
            print(f"AVISO: de '{e}' solo hay imagenes con el elemento {lado}. Entrenar con ellas "
                  f"sesga al modelo (aprenderia que siempre esta {lado}); use ese elemento solo si "
                  "tiene tambien el otro lado, o confie en sus fotos propias.")
    if not cuenta:
        print("Sin resultados: revise --elementos y el mapeo de clases (mapeo_clases.json).")
    print("Siguiente paso: python -m src.multietiqueta")


if __name__ == "__main__":
    main()
