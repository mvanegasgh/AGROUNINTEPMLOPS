"""API REST (FastAPI) para la verificacion de EPP.

Uso local:  uvicorn api.main:app --reload      ->  http://localhost:8000/docs
Variables:  EPP_BACKEND = multi (defecto, un resultado por elemento) | cnn | clasico
            EPP_UMBRAL = 0.70 (denegar)   EPP_UMBRAL_APROBAR = 0.80 (aprobar)

POST /predict  (multipart, campo 'archivo' = imagen)  -> ingreso (aprobado | denegado | incierto),
                                                         resultado, faltantes, elementos (ok|falta|incierto
                                                         por elemento, modelo multi), clase, confianza,
                                                         probabilidades, requiere_revision, latencia_ms
GET  /health   -> estado del servicio
GET  /stats    -> conteos y latencia (monitoreo). La API NO guarda imagenes, solo contadores.
"""
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.predictores import UMBRAL_APROBAR, UMBRAL_DEFECTO, crear_predictor, decidir

BACKEND = os.getenv("EPP_BACKEND", "multi")
UMBRAL = float(os.getenv("EPP_UMBRAL", UMBRAL_DEFECTO))
UMBRAL_AP = float(os.getenv("EPP_UMBRAL_APROBAR", UMBRAL_APROBAR))
MAX_BYTES = 5 * 1024 * 1024

modelo = {}
lock = threading.Lock()
stats = {"inicio": time.time(), "total": 0, "revision": 0, "errores": 0,
         "lat_total_ms": 0.0, "lat_max_ms": 0.0, "por_clase": Counter()}


@asynccontextmanager
async def ciclo_de_vida(app):
    modelo["predictor"] = crear_predictor(BACKEND)  # se carga una sola vez al arrancar
    yield
    modelo.clear()


app = FastAPI(title="AGROINTEP - Verificacion de EPP", version="1.0", lifespan=ciclo_de_vida)


@app.get("/health")
def health():
    return {"estado": "ok", "backend": BACKEND, "modelo_cargado": "predictor" in modelo,
            "umbral": UMBRAL, "umbral_aprobar": UMBRAL_AP}


@app.get("/stats")
def get_stats():
    with lock:
        n = stats["total"]
        return {"backend": BACKEND, "uptime_s": round(time.time() - stats["inicio"]),
                "predicciones": n, "requieren_revision": stats["revision"],
                "errores": stats["errores"],
                "latencia_media_ms": round(stats["lat_total_ms"] / n, 1) if n else None,
                "latencia_max_ms": round(stats["lat_max_ms"], 1),
                "por_clase": dict(stats["por_clase"])}


@app.post("/predict")
def predict(archivo: UploadFile = File(...)):
    datos = archivo.file.read(MAX_BYTES + 1)
    if len(datos) > MAX_BYTES:
        raise HTTPException(413, "Imagen demasiado grande (maximo 5 MB).")
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        with lock:
            stats["errores"] += 1
        raise HTTPException(400, "El archivo no es una imagen valida.")

    t0 = time.time()
    probs = modelo["predictor"].predecir(img)
    lat = (time.time() - t0) * 1000
    categoria, texto, estados, clase, conf = decidir(modelo["predictor"].tipo, probs, UMBRAL,
                                                     UMBRAL_AP)
    revision = categoria == "incierto"
    if estados is not None:  # modelo multi: faltantes por elemento
        faltantes = [e for e, s in estados.items() if s == "falta"]
    elif clase == "sin_epp":  # modelo por estado
        faltantes = ["todos"]
    elif clase.startswith("falta_") and categoria == "denegado":
        faltantes = clase[6:].split("_")
    else:
        faltantes = []

    with lock:
        stats["total"] += 1
        stats["revision"] += revision
        stats["lat_total_ms"] += lat
        stats["lat_max_ms"] = max(stats["lat_max_ms"], lat)
        stats["por_clase"][clase] += 1
    return {"tipo": modelo["predictor"].tipo, "ingreso": categoria, "resultado": texto,
            "faltantes": faltantes, "elementos": estados, "clase": clase,
            "confianza": round(conf, 4),
            "probabilidades": {k: round(v, 4) for k, v in probs.items()},
            "requiere_revision": revision, "latencia_ms": round(lat, 1)}
