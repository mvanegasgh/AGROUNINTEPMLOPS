"""Construye el dataset desde la camara del punto de acceso (la MISMA toma del sistema real).

Uso: python -m src.capturar_dataset --persona ana --clase completo
Teclas: ESPACIO = iniciar/pausar | q = salir. Se detiene sola al llegar a --max fotogramas.

Los archivos se guardan como  dataset/raw/<clase>/<persona>__<clase>_<ms>.jpg
El nombre de la persona permite separar train/test POR PERSONA (evaluacion honesta).
Use el mismo nombre para la misma persona en todas las clases.
"""
import argparse
import re
import time

import cv2

from .common import RAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True, help="identificador (ana, juan, p03...) - sin datos reales sensibles")
    ap.add_argument("--clase", required=True, help="completo | falta_cofia | falta_bata | sin_epp ...")
    ap.add_argument("--fuente", default="0")
    ap.add_argument("--cada", type=float, default=0.5, help="segundos entre fotogramas")
    ap.add_argument("--max", type=int, default=60, help="fotogramas por tanda (60 = 30 s)")
    a = ap.parse_args()

    persona = re.sub(r"[^a-z0-9]+", "", a.persona.lower())
    if not persona or persona.startswith("pub"):
        raise SystemExit("Nombre de persona invalido (no puede estar vacio ni empezar por 'pub').")
    destino = RAW / a.clase
    destino.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(int(a.fuente) if a.fuente.isdigit() else a.fuente)
    grabando, n, t_ult = False, 0, 0.0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if grabando and time.time() - t_ult >= a.cada:
            cv2.imwrite(str(destino / f"{persona}__{a.clase}_{int(time.time() * 1000)}.jpg"), frame)
            n += 1
            t_ult = time.time()
            if n >= a.max:
                grabando = False
        vista = frame.copy()
        estado = "GRABANDO" if grabando else ("LISTO" if n >= a.max else "PAUSA")
        cv2.putText(vista, f"{persona} / {a.clase}: {n}/{a.max} {estado} (ESPACIO)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 220) if grabando else (255, 255, 255), 2)
        cv2.imshow("Captura de dataset", vista)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or (n >= a.max and not grabando and k == 32):
            break
        if k == 32 and n < a.max:
            grabando = not grabando
    cap.release()
    cv2.destroyAllWindows()
    print(f"{n} fotogramas guardados en {destino} (persona: {persona})")


if __name__ == "__main__":
    main()
