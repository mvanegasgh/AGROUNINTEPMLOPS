"""Descriptores clasicos de OpenCV (Fase 1): vector numerico por imagen.

extract(img_bgr) -> lista de floats, siempre del mismo largo (N_FEATURES). Se usa igual al
entrenar (eda_limpieza.py) y en vivo (PredictorClasico): no cambie esta funcion sin reentrenar.

Contenido (largo total 72 + 48 + 16 + 16 = 152):
  - histograma de color HSV global (8 x 3 x 3)                 -> color de bata / guantes / cofia
  - media H, S, V en una cuadricula 4x4                        -> DONDE esta cada color
  - densidad de bordes (Canny) en la cuadricula 4x4            -> textura / contornos
  - brillo medio por celda de la cuadricula 4x4                -> forma general
"""
import cv2
import numpy as np

LADO = 128
GRID = 4
N_FEATURES = 8 * 3 * 3 + GRID * GRID * 3 + GRID * GRID + GRID * GRID


def extract(img_bgr):
    img = cv2.resize(img_bgr, (LADO, LADO), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / max(hist.sum(), 1.0)

    bordes = cv2.Canny(gris, 80, 160) > 0
    paso = LADO // GRID
    medias, dens, brillo = [], [], []
    for i in range(GRID):
        for j in range(GRID):
            sl = (slice(i * paso, (i + 1) * paso), slice(j * paso, (j + 1) * paso))
            medias.extend(hsv[sl].reshape(-1, 3).mean(axis=0) / np.array([180.0, 255.0, 255.0]))
            dens.append(float(bordes[sl].mean()))
            brillo.append(float(gris[sl].mean()) / 255.0)
    vec = np.concatenate([hist, medias, dens, brillo]).astype(float)
    assert vec.size == N_FEATURES
    return vec.tolist()
