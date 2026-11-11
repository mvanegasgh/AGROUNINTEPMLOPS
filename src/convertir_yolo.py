"""Convierte un dataset publico en formato YOLO (cajas) a carpetas por clase (1 etiqueta por imagen).

Uso:
  python -m src.convertir_yolo --dataset descargas/ppe-zaintures --requeridos cofia,guantes
Opciones:
  --mapeo mapeo_clases.json    vocabulario (pos/neg) de cada EPP; edítelo segun las clases del dataset
  --ausencia-implicita         si un EPP requerido no aparece anotado, se asume AUSENTE
                               (solo si el dataset anota todo lo visible; si no, puede etiquetar mal)

Reglas por imagen (sobre los EPP --requeridos):
  completo     : todos presentes (anotados) y ninguno ausente
  falta_<epp>  : exactamente un EPP ausente (clase 'no-...' o ausencia implicita)
  sin_epp      : todos ausentes
  (otros casos se omiten por ambiguos)
Las imagenes salen como  dataset/raw/<clase>/pub<dataset>_<n>__<clase>_<n>.jpg  -> grupo 'pub...':
van SIEMPRE a entrenamiento; el test se reserva para personas de su camara.
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

from .common import IMG_EXT, RAW, ROOT


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
    ap.add_argument("--requeridos", required=True, help="ej. cofia,guantes")
    ap.add_argument("--mapeo", default=str(ROOT / "mapeo_clases.json"))
    ap.add_argument("--salida", default=str(RAW))
    ap.add_argument("--ausencia-implicita", action="store_true")
    ap.add_argument("--max-por-clase", type=int, default=0, help="0 = sin limite")
    a = ap.parse_args()

    root, salida = Path(a.dataset), Path(a.salida)
    req = [r.strip() for r in a.requeridos.split(",") if r.strip()]
    mapeo = json.loads(Path(a.mapeo).read_text(encoding="utf-8"))
    for r in req:
        if r not in mapeo:
            raise SystemExit(f"'{r}' no esta en {a.mapeo}. Disponibles: {list(mapeo)}")
    nombres = cargar_nombres(root)
    pos = {n: e for e, v in mapeo.items() for n in v["pos"]}
    neg = {n: e for e, v in mapeo.items() for n in v["neg"]}
    print("Clases del dataset:", nombres)
    print("Reconocidas  +:", {n: pos[n] for n in nombres if n in pos},
          " -:", {n: neg[n] for n in nombres if n in neg})

    tag = re.sub(r"[^a-z0-9]", "", root.name.lower())[:12] or "ds"
    cuenta, omitidas, n = Counter(), Counter(), 0
    for lbl in sorted(root.rglob("*.txt")):
        if lbl.parent.name != "labels":
            continue
        img = next((p for p in (lbl.parent.parent / "images").glob(lbl.stem + ".*")
                    if p.suffix.lower() in IMG_EXT), None)
        if img is None:
            continue
        ids = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
        vistos = {nombres[i] for i in ids if i < len(nombres)}
        presentes = {pos[v] for v in vistos if v in pos}
        ausentes = {neg[v] for v in vistos if v in neg}
        if a.ausencia_implicita and vistos:
            ausentes |= {r for r in req if r not in presentes and r not in ausentes}
        ok_pos = [r for r in req if r in presentes and r not in ausentes]
        falt = [r for r in req if r in ausentes]

        if len(ok_pos) == len(req):
            clase = "completo"
        elif len(req) > 1 and len(falt) == len(req):
            clase = "sin_epp"
        elif len(falt) == 1 and len(ok_pos) + 1 == len(req):  # (con 1 requerido: falta_<epp>)
            clase = f"falta_{falt[0]}"
        else:
            omitidas["ambigua"] += 1
            continue
        if a.max_por_clase and cuenta[clase] >= a.max_por_clase:
            continue
        n += 1
        dst = salida / clase / f"pub{tag}{n}__{clase}_{n}{img.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst)
        cuenta[clase] += 1

    print("\nImagenes por clase:", dict(cuenta) or "ninguna")
    print("Omitidas por ambiguas:", omitidas["ambigua"])
    if not cuenta:
        print("Sin resultados: revise --requeridos, el mapeo o pruebe --ausencia-implicita.")
    elif len(cuenta) < 2:
        print("AVISO: solo una clase; este dataset sirve poco solo. Combine con fotos propias.")


if __name__ == "__main__":
    main()
