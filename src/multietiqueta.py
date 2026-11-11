"""Entrenamiento MULTI-ETIQUETA: un modelo que evalua CADA elemento por separado.

Uso:
  python -m src.multietiqueta                                  # cofia, guantes, bata
  python -m src.multietiqueta --elementos cofia,guantes,bata --img 224 --epochs-cabeza 10 --epochs-fine 10
Salida: models/epp_multi.keras, models/elementos.json, reports/multi_metricas.csv  (+ MLflow)

Datos (el nombre de la carpeta da la etiqueta de cada elemento):
  dataset/raw/<estado>/<persona>__*.jpg      estado = completo | sin_epp | falta_<e>[_<e2>]
                                             -> los tres elementos quedan etiquetados.
  dataset/publico/<elemento>/<si|no>/*.jpg   datos publicos: solo UN elemento etiquetado; en los
                                             demas la perdida los ignora (ver convertir_yolo_elementos).
Evaluacion honesta: el test usa solo personas de su camara, separadas por persona.
Reporta, por elemento, recall/precision de "FALTA" y, a nivel de ingreso, los FALSOS APROBADOS
(aprobar a quien no cumple) con las mismas reglas y umbrales que usa el video en vivo.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

from .predictores import UMBRAL_APROBAR, UMBRAL_DEFECTO, evaluar_elementos

ROOT = Path(__file__).resolve().parent.parent
RAW, PUBLICO = ROOT / "dataset" / "raw", ROOT / "dataset" / "publico"
MODELS, REPORTS = ROOT / "models", ROOT / "reports"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def mlflow_uri():
    return os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())


def estado_a_etiquetas(estado, elementos):
    """completo / sin_epp / falta_<e>[_<e2>] -> {elemento: 1 presente | 0 ausente}, o None."""
    if estado == "completo":
        return {e: 1 for e in elementos}
    if estado == "sin_epp":
        return {e: 0 for e in elementos}
    if estado.startswith("falta_"):
        faltan = estado[6:].split("_")
        if all(f in elementos for f in faltan):
            return {e: 0 if e in faltan else 1 for e in elementos}
    return None


def imagenes(carpeta):
    return sorted(f for f in carpeta.rglob("*") if f.suffix.lower() in IMG_EXT)


def construir_tabla(elementos):
    filas, omitidas = [], []
    if RAW.exists():
        for cdir in sorted(p for p in RAW.iterdir() if p.is_dir()):
            et = estado_a_etiquetas(cdir.name, elementos)
            if et is None:
                omitidas.append(cdir.name)
                continue
            for i, f in enumerate(imagenes(cdir)):
                grupo = f.stem.split("__")[0] if "__" in f.stem else f"sin_grupo_{cdir.name}_{i}"
                if grupo.startswith("pub"):
                    omitidas.append(f"{cdir.name}/{f.name} (publico: use dataset/publico/...)")
                    continue
                filas.append({"path": str(f), "grupo": grupo, "origen": "propio", "estado": cdir.name,
                              "y": [et[e] for e in elementos]})
    if PUBLICO.exists():
        for edir in sorted(p for p in PUBLICO.iterdir() if p.is_dir() and p.name in elementos):
            for val, y1 in (("si", 1), ("no", 0)):
                for f in imagenes(edir / val):
                    y = [-1] * len(elementos)  # -1 = desconocido: no cuenta en la perdida
                    y[elementos.index(edir.name)] = y1
                    filas.append({"path": str(f), "grupo": f"pub_{edir.name}", "origen": "publico",
                                  "estado": f"pub_{edir.name}_{val}", "y": y})
    if omitidas:
        print(f"AVISO: {len(omitidas)} carpetas/archivos omitidos (nombre no valido para "
              f"{elementos}): {omitidas[:5]}")
    return pd.DataFrame(filas)


def dividir_personas(df, frac, seed=SEED):
    """(resto, seleccion) separando por persona; la seleccion toma ~frac de las personas."""
    personas = sorted(df.grupo.unique())
    k = max(1, math.ceil(len(personas) * frac))
    elegidas = set(random.Random(seed).sample(personas, k))
    es = df.grupo.isin(elegidas)
    return df[~es], df[es]


def hacer_ds(rutas, y, entrenar, img, batch=32):
    def cargar(p, t):
        im = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        im.set_shape([None, None, 3])
        return tf.image.resize(im, (img, img)), t

    ds = tf.data.Dataset.from_tensor_slices((list(rutas), np.asarray(y, dtype="float32")))
    if entrenar:
        ds = ds.shuffle(2048, seed=SEED)
    return ds.map(cargar, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def perdida_enmascarada(y_true, y_pred):
    """Entropia cruzada binaria que ignora las etiquetas desconocidas (-1)."""
    mask = tf.cast(y_true >= 0, tf.float32)
    bce = tf.keras.backend.binary_crossentropy(tf.clip_by_value(y_true, 0.0, 1.0), y_pred)
    return tf.reduce_sum(bce * mask, axis=-1) / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)


def construir(n, img, pesos):
    base = tf.keras.applications.MobileNetV2((img, img, 3), include_top=False, weights=pesos)
    base.trainable = False
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                               tf.keras.layers.RandomRotation(0.05),
                               tf.keras.layers.RandomZoom(0.1),
                               tf.keras.layers.RandomBrightness(0.2)], name="augmentacion")
    inp = tf.keras.Input((img, img, 3))
    x = aug(inp)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # = preprocess de MobileNetV2
    # con pesos preentrenados la base va en modo inferencia (BatchNorm congelada);
    # sin ellos (solo pruebas offline) debe aprender, incluida su BatchNorm
    x = base(x, training=False) if pesos else base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n, activation="sigmoid")(x)  # P(elemento presente), independiente
    return tf.keras.Model(inp, out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elementos", default="cofia,guantes,bata")
    ap.add_argument("--img", type=int, default=224, help="mas alto (p. ej. 288) ayuda con guantes")
    ap.add_argument("--epochs-cabeza", type=int, default=10)
    ap.add_argument("--epochs-fine", type=int, default=10)
    ap.add_argument("--paciencia", type=int, default=4, help="EarlyStopping (epocas sin mejorar)")
    ap.add_argument("--sin-publicos", action="store_true",
                    help="ignora dataset/publico/ (para comparar: si no mejoran el test, descartelos)")
    ap.add_argument("--pesos", default="imagenet", help="imagenet | none (sin descargar pesos)")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    elementos = [e.strip() for e in a.elementos.split(",") if e.strip()]

    df = construir_tabla(elementos)
    if df.empty:
        raise SystemExit(f"No hay imagenes validas en {RAW} (ni en {PUBLICO}).")
    propios, publicos = df[df.origen == "propio"], df[df.origen == "publico"]
    if a.sin_publicos:
        publicos = publicos.iloc[0:0]
    if len(publicos):
        print(f"Datos publicos: {len(publicos)} imagenes (solo entrenamiento; el test es de su camara)")
    n_pers = propios.grupo.nunique()
    if n_pers < 3:
        raise SystemExit(f"Hay {n_pers} personas propias; se necesitan al menos 3 (train, val y "
                         "test por persona). Ideal: 6 o mas.")
    resto, test = dividir_personas(propios, 0.30)
    train_p, val = dividir_personas(resto, 0.20, SEED + 1)
    train = pd.concat([train_p, publicos])
    print(f"Personas propias: {n_pers} | train {train_p.grupo.nunique()} (+{len(publicos)} publicas) "
          f"| val {val.grupo.nunique()} | test {test.grupo.nunique()}")
    assert not (set(train.grupo) & set(test.grupo)) and not (set(val.grupo) & set(test.grupo))
    Y = np.stack(train.y.to_numpy())
    for j, e in enumerate(elementos):
        print(f"  {e}: presente={int((Y[:, j] == 1).sum())} ausente={int((Y[:, j] == 0).sum())} "
              f"sin etiqueta={int((Y[:, j] < 0).sum())}")

    ds = lambda d, ent: hacer_ds(d.path, np.stack(d.y.to_numpy()), ent, a.img)  # noqa: E731
    ds_tr, ds_va, ds_te = ds(train, True), ds(val, False), ds(test, False)

    pesos = None if a.pesos.lower() == "none" else a.pesos
    model, base = construir(len(elementos), a.img, pesos)
    if pesos is None:  # sin preentrenar la base debe aprender (solo pruebas offline)
        base.trainable = True
        for capa in base.layers:  # estadisticas de BatchNorm que converjan rapido
            if isinstance(capa, tf.keras.layers.BatchNormalization):
                capa.momentum = 0.9
    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("agrointep-fase2")
    with mlflow.start_run(run_name="MobileNetV2_multietiqueta"):
        mlflow.log_params({"base": "MobileNetV2", "salidas": ",".join(elementos), "img": a.img,
                           "n_train": len(train), "n_val": len(val), "n_test": len(test),
                           "n_publicas": len(publicos)})
        es = tf.keras.callbacks.EarlyStopping(patience=a.paciencia, restore_best_weights=True)
        model.compile(tf.keras.optimizers.Adam(1e-3), perdida_enmascarada)
        model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs_cabeza, callbacks=[es], verbose=2)
        if pesos:  # fine-tuning solo tiene sentido con pesos preentrenados
            base.trainable = True
            for capa in base.layers[:-30]:
                capa.trainable = False
            for capa in base.layers:
                if isinstance(capa, tf.keras.layers.BatchNormalization):
                    capa.trainable = False
            model.compile(tf.keras.optimizers.Adam(1e-5), perdida_enmascarada)
            model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs_fine, callbacks=[es], verbose=2)

        P = model.predict(ds_te, verbose=0)
        Yt = np.stack(test.y.to_numpy())  # el test es 100% de su camara: todo etiquetado
        filas, met = [], {}
        for j, e in enumerate(elementos):
            falta_real, falta_pred = (Yt[:, j] == 0).astype(int), (P[:, j] < 0.5).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(falta_real, falta_pred, labels=[1],
                                                            zero_division=0)
            filas.append({"elemento": e, "recall_falta": rc[0], "precision_falta": pr[0],
                          "f1_falta": f1[0], "n_falta_real": int(falta_real.sum())})
            met.update({f"recall_falta_{e}": rc[0], f"precision_falta_{e}": pr[0],
                        f"f1_falta_{e}": f1[0]})
        # Decision de ingreso con las MISMAS reglas y umbrales del video en vivo
        cumple_real = (Yt == 1).all(axis=1)
        dec = [evaluar_elementos(dict(zip(elementos, p)), UMBRAL_DEFECTO, UMBRAL_APROBAR)[0]
               for p in P]
        aprob = np.array([d == "aprobado" for d in dec])
        niega = np.array([d == "denegado" for d in dec])
        inc = np.array([d == "incierto" for d in dec])
        met.update({
            "falsos_aprobados_tasa": float(aprob[~cumple_real].mean()) if (~cumple_real).any() else 0.0,
            "aprobados_correctos_tasa": float(aprob[cumple_real].mean()) if cumple_real.any() else 0.0,
            "denegados_indebidos_tasa": float(niega[cumple_real].mean()) if cumple_real.any() else 0.0,
            "incierto_tasa": float(inc.mean())})
        img0 = next(iter(ds_te))[0][:1]
        t0 = time.time()
        for _ in range(20):
            model(img0, training=False)
        met["latencia_ms"] = (time.time() - t0) / 20 * 1000
        mlflow.log_metrics(met)

        model.save(MODELS / "epp_multi.keras")
        (MODELS / "elementos.json").write_text(json.dumps({"elementos": elementos, "img": a.img}),
                                               encoding="utf-8")
        pd.DataFrame(filas).to_csv(REPORTS / "multi_metricas.csv", index=False)
        mlflow.log_artifact(str(MODELS / "elementos.json"))

    pd.set_option("display.width", 160)
    print("\nPor elemento (clase positiva = FALTA; el recall es lo critico):")
    print(pd.DataFrame(filas).round(3).to_string(index=False))
    print("\nDecision de ingreso en el test (mismas reglas que el video en vivo):")
    for k in ("falsos_aprobados_tasa", "aprobados_correctos_tasa", "denegados_indebidos_tasa",
              "incierto_tasa", "latencia_ms"):
        print(f"  {k}: {met[k]:.3f}")
    print("\nfalsos_aprobados_tasa = fraccion de fotogramas SIN cumplir que el sistema aprobaria.")


if __name__ == "__main__":
    main()
