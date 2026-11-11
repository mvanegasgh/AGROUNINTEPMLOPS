"""Rutas y utilidades compartidas por las etapas de la Fase 1 y 2 (modelo por estado).

Estructura de datos:
  dataset/raw/<estado>/<persona>__*.jpg   capturas propias (estado = completo | falta_<e> | sin_epp)
  dataset/raw/<estado>/pub...__*.jpg      publicos convertidos por convertir_yolo.py (solo train)
  dataset/limpio/<estado>/                imagenes validas tras eda_limpieza.py
  data/features.csv                       path, clase, grupo, f_0 ... f_n
El modelo por elemento (src/multietiqueta.py) lee dataset/raw y dataset/publico directamente.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
CLEAN = ROOT / "dataset" / "limpio"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEST_SIZE = 0.30  # mismo split en Fase 1 y Fase 2 -> comparacion justa

for d in (DATA, REPORTS, MODELS):
    d.mkdir(parents=True, exist_ok=True)


def mlflow_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())


def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA / "features.csv")


def feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("f_")]


def split_grupos(df: pd.DataFrame, frac: float, seed: int = SEED):
    """Divide en (resto, fraccion) sin que un mismo grupo (persona) quede en ambos lados.
    Los fotogramas consecutivos de un video son casi identicos: separar al azar infla las metricas."""
    clases = set(df["clase"])
    n_splits = max(2, min(round(1 / frac), df["grupo"].nunique()))
    if df["grupo"].nunique() >= 2:
        y, g = df["clase"].to_numpy(dtype=object), df["grupo"].to_numpy(dtype=object)
        for k in range(30):  # busca una particion donde ambos lados tengan todas las clases
            sgkf = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + k)
            for tr, te in sgkf.split(df, y, g):
                a, b = df.iloc[tr], df.iloc[te]
                if set(a["clase"]) == clases and set(b["clase"]) == clases:
                    return a, b
    print("AVISO: no hay suficientes personas/grupos para separar por persona; se usa division "
          "aleatoria (las metricas pueden ser optimistas). Capture mas personas con --persona.")
    return train_test_split(df, test_size=frac, stratify=df["clase"], random_state=seed)


def _es_propio(grupo: pd.Series) -> pd.Series:
    return ~grupo.astype(str).str.startswith(("pub", "sin_grupo"))


def split(df: pd.DataFrame):
    """Train/test por persona. Los datos publicos (grupo 'pub...') y sin grupo van SIEMPRE a
    entrenamiento: el test usa unicamente personas de su camara."""
    propios = df[_es_propio(df["grupo"])]
    if propios.empty:
        return split_grupos(df, TEST_SIZE)
    _, te = split_grupos(propios, TEST_SIZE)
    es_test = _es_propio(df["grupo"]) & df["grupo"].isin(set(te["grupo"]))
    return df[~es_test], df[es_test]


def resumen_split(train: pd.DataFrame, test: pd.DataFrame):
    print("Split por persona/sesion (sin fuga entre train y test):")
    for nombre, d in (("train", train), ("test ", test)):
        cnt = ", ".join(f"{k}={v}" for k, v in d["clase"].value_counts().sort_index().items())
        print(f"  {nombre}: {len(d)} imgs, {d['grupo'].nunique()} grupos | {cnt}")
    assert not (set(train["grupo"]) & set(test["grupo"])), "fuga: un grupo esta en train y test"
