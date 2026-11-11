"""Predictores y logica de decision compartidos por el video en vivo y la API.

Cada predictor recibe un fotograma BGR (como lo entrega OpenCV) y devuelve un diccionario
{clase: probabilidad}. Las clases son las del entrenamiento: completo, falta_<epp>, sin_epp.

No importa nada pesado al cargar el modulo: scikit-learn / TensorFlow / requests solo se
importan cuando se crea el backend correspondiente.
"""
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
UMBRAL_DEFECTO = 0.70  # confianza promedio < umbral -> INCIERTO (README)
UMBRAL_APROBAR = 0.80  # para APROBAR el ingreso se exige mas confianza que para denegarlo
IMG = 224


class PredictorClasico:
    """Mejor modelo de ML clasico (models/clasico.joblib, lo guarda benchmark_fase1.py)."""

    tipo = "estado"

    def __init__(self, ruta=None):
        import joblib

        from . import features  # mismas caracteristicas con las que se entreno

        self.features = features
        self.pipe = joblib.load(ruta or MODELS / "clasico.joblib")
        self.clases = [str(c) for c in self.pipe.classes_]

    def predecir(self, frame_bgr):
        x = np.asarray(self.features.extract(frame_bgr), dtype=float).reshape(1, -1)
        p = self.pipe.predict_proba(x)[0]
        return dict(zip(self.clases, (float(v) for v in p)))


class PredictorCNN:
    """MobileNetV2 con Transfer Learning (models/epp_mobilenet.keras + classes.json).
    El reescalado esta dentro del modelo: aqui solo se envian pixeles 0-255 en RGB."""

    tipo = "estado"

    def __init__(self, ruta=None):
        import tensorflow as tf

        self.modelo = tf.keras.models.load_model(ruta or MODELS / "epp_mobilenet.keras")
        self.clases = json.loads((MODELS / "classes.json").read_text(encoding="utf-8"))

    def predecir(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR).astype("float32")
        p = self.modelo(x[None], training=False).numpy()[0]
        return dict(zip(self.clases, (float(v) for v in p)))


class PredictorMulti:
    """Modelo MULTI-ETIQUETA (models/epp_multi.keras + elementos.json): evalua cada elemento por
    separado y devuelve {elemento: probabilidad de que este PRESENTE}."""

    tipo = "elementos"

    def __init__(self, ruta=None):
        import tensorflow as tf

        # compile=False: la perdida enmascarada solo se necesita para entrenar
        self.modelo = tf.keras.models.load_model(ruta or MODELS / "epp_multi.keras",
                                                 compile=False)
        meta = json.loads((MODELS / "elementos.json").read_text(encoding="utf-8"))
        self.elementos, self.img = meta["elementos"], meta.get("img", IMG)

    def predecir(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.img, self.img), interpolation=cv2.INTER_LINEAR)
        p = self.modelo(x.astype("float32")[None], training=False).numpy()[0]
        return dict(zip(self.elementos, (float(v) for v in p)))


class PredictorAPI:
    """Envia el fotograma a la API desplegada (POST multipart, campo 'archivo').
    El tipo (estado | elementos) lo decide el modelo que corre en la API."""

    tipo = "estado"

    def __init__(self, url, timeout=5.0):
        import requests

        self.requests, self.url, self.timeout = requests, url, timeout

    def predecir(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("no se pudo codificar el fotograma")
        r = self.requests.post(self.url, files={"archivo": ("frame.jpg", buf.tobytes(),
                                                            "image/jpeg")}, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        self.tipo = j.get("tipo", "estado")
        return {k: float(v) for k, v in j["probabilidades"].items()}


def crear_predictor(backend, url=None, modelo=None):
    if backend == "clasico":
        return PredictorClasico(modelo)
    if backend == "cnn":
        return PredictorCNN(modelo)
    if backend == "multi":
        return PredictorMulti(modelo)
    if backend == "api":
        if not url:
            raise ValueError("el backend 'api' requiere --url")
        return PredictorAPI(url)
    raise ValueError(f"backend desconocido: {backend}")


class Suavizador:
    """Promedio temporal de las probabilidades de los ultimos N fotogramas: evita que un
    solo fotograma ruidoso cambie el resultado en pantalla."""

    def __init__(self, ventana=10, minimo=3):
        self.buf, self.minimo = deque(maxlen=ventana), minimo

    def agregar(self, probs):
        self.buf.append(probs)

    def limpiar(self):
        self.buf.clear()

    def __len__(self):
        return len(self.buf)

    def promedio(self):
        """Probabilidades promedio de la ventana, o None si aun hay pocas muestras."""
        if len(self.buf) < self.minimo:
            return None
        return {c: float(np.mean([p.get(c, 0.0) for p in self.buf])) for c in self.buf[0]}

    def resultado(self):
        """(clase, confianza, probabilidades_promedio) para modelos de estado."""
        prom = self.promedio()
        if prom is None:
            return None
        clase = max(prom, key=prom.get)
        return clase, prom[clase], prom


def interpretar(clase, confianza, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR):
    """(categoria, texto) segun la clase y la confianza.
    Categorias: aprobado | denegado | incierto (este ultimo pide revision manual).

    Aprobar exige mas confianza que denegar: el error grave es dejar entrar a quien no cumple.
    Una clase combinada como `falta_cofia_guantes` se muestra como FALTA: COFIA, GUANTES."""
    if clase == "completo":
        if confianza >= max(umbral, umbral_aprobar):
            return "aprobado", "INGRESO APROBADO"
        return "incierto", "INCIERTO - REVISION MANUAL"
    if confianza < umbral:
        return "incierto", "INCIERTO - REVISION MANUAL"
    if clase == "sin_epp":
        return "denegado", "INGRESO DENEGADO - FALTAN TODOS LOS EPP"
    if clase.startswith("falta_"):
        return "denegado", "INGRESO DENEGADO - FALTA: " + ", ".join(clase[6:].split("_")).upper()
    return "denegado", "INGRESO DENEGADO - " + clase.upper()


def evaluar_elementos(probs, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR):
    """Decision a partir de {elemento: P(presente)}. Devuelve (categoria, texto, estados).

    Por elemento: ok si P >= umbral_aprobar | falta si P <= 1 - umbral | incierto en medio.
    Ingreso: aprobado solo si TODOS son ok; denegado si alguno falta (se nombran los que faltan);
    incierto si ninguno falta pero alguno no es seguro (revision manual)."""
    estados = {}
    for e, p in probs.items():
        estados[e] = "ok" if p >= umbral_aprobar else ("falta" if p <= 1 - umbral else "incierto")
    faltan = [e.upper() for e, s in estados.items() if s == "falta"]
    dudan = [e.upper() for e, s in estados.items() if s == "incierto"]
    if faltan:
        return "denegado", "INGRESO DENEGADO - FALTA: " + ", ".join(faltan), estados
    if dudan:
        return "incierto", "INCIERTO - REVISAR: " + ", ".join(dudan), estados
    return "aprobado", "INGRESO APROBADO", estados


def decidir(tipo, probs, umbral=UMBRAL_DEFECTO, umbral_aprobar=UMBRAL_APROBAR):
    """Unifica los dos tipos de modelo. Devuelve (categoria, texto, estados, clase, confianza);
    `estados` es {elemento: ok|falta|incierto} (None en modelos de estado) y `clase` usa la
    convencion del proyecto (completo | falta_<e>[_<e2>] | incierto_<e>)."""
    if tipo == "elementos":
        cat, txt, estados = evaluar_elementos(probs, umbral, umbral_aprobar)
        faltan = [e for e, s in estados.items() if s == "falta"]
        dudan = [e for e, s in estados.items() if s == "incierto"]
        clase = ("falta_" + "_".join(faltan)) if faltan else (
            ("incierto_" + "_".join(dudan)) if dudan else "completo")
        return cat, txt, estados, clase, min(max(p, 1 - p) for p in probs.values())
    clase = max(probs, key=probs.get)
    cat, txt = interpretar(clase, probs[clase], umbral, umbral_aprobar)
    return cat, txt, None, clase, probs[clase]
