"""Fase 2 (Semana 8+): CNN con Transfer Learning (MobileNetV2) y seguimiento en MLflow.

Uso: python -m src.transfer_learning [--epochs-cabeza 10 --epochs-fine 10]
Salida: models/epp_mobilenet.keras, models/classes.json
El reescalado va DENTRO del modelo: la API solo envia pixeles 0-255.
"""
import argparse
import json
import time

import matplotlib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score)
from sklearn.utils.class_weight import compute_class_weight

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import (MODELS, REPORTS, ROOT, SEED, load_features, mlflow_uri, resumen_split,
                     split, split_grupos)

IMG = 224
AUTOTUNE = tf.data.AUTOTUNE


def dataset(rutas, etiquetas, entrenamiento, batch=32):
    def cargar(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return tf.image.resize(img, (IMG, IMG)), y

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    if entrenamiento:
        ds = ds.shuffle(2048, seed=SEED)
    return ds.map(cargar, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def construir(n_clases):
    base = tf.keras.applications.MobileNetV2((IMG, IMG, 3), include_top=False,
                                             weights="imagenet")
    base.trainable = False
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ], name="augmentacion")
    inp = tf.keras.Input((IMG, IMG, 3))
    x = aug(inp)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # = preprocess de MobileNetV2
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_clases, activation="softmax")(x)
    return tf.keras.Model(inp, out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-cabeza", type=int, default=10)
    ap.add_argument("--epochs-fine", type=int, default=10)
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)

    df = load_features()
    clases = sorted(df.clase.unique())
    idx = {c: i for i, c in enumerate(clases)}
    train, test = split(df)  # mismo test que en la Fase 1
    resumen_split(train, test)
    train, val = split_grupos(train, 0.15)  # validacion tambien por persona

    def listas(d):
        return [str(ROOT / p) for p in d.path], d.clase.map(idx).values

    (ptr, ytr), (pva, yva), (pte, yte) = listas(train), listas(val), listas(test)
    ds_tr, ds_va, ds_te = (dataset(ptr, ytr, True), dataset(pva, yva, False),
                           dataset(pte, yte, False))
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(clases)), y=ytr)))

    model, base = construir(len(clases))
    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("agrointep-fase2")
    with mlflow.start_run(run_name="MobileNetV2_TL"):
        mlflow.log_params({"base": "MobileNetV2", "img": IMG, "epocas_cabeza": a.epochs_cabeza,
                           "epocas_fine": a.epochs_fine, "n_train": len(ytr)})
        es = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

        # Etapa 1: entrenar solo la cabeza clasificadora
        model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs_cabeza, class_weight=cw,
                  callbacks=[es], verbose=2)

        # Etapa 2: fine-tuning de las ultimas 30 capas (BatchNorm congelada)
        base.trainable = True
        for capa in base.layers[:-30]:
            capa.trainable = False
        for capa in base.layers:
            if isinstance(capa, tf.keras.layers.BatchNormalization):
                capa.trainable = False
        model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs_fine, class_weight=cw,
                  callbacks=[es], verbose=2)

        probs = model.predict(ds_te, verbose=0)
        pred = probs.argmax(1)
        rep = classification_report(yte, pred, target_names=clases, output_dict=True,
                                    zero_division=0)
        met = {"test_accuracy": accuracy_score(yte, pred),
               "test_f1_macro": f1_score(yte, pred, average="macro")}
        met.update({f"recall_{c}": rep[c]["recall"] for c in clases})
        img0 = next(iter(ds_te))[0][:1]
        t0 = time.time()
        for _ in range(20):
            model(img0, training=False)
        met["latencia_ms"] = (time.time() - t0) / 20 * 1000
        mlflow.log_metrics(met)

        ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=clases, cmap="Blues")
        ruta = REPORTS / "cm_MobileNetV2.png"
        plt.savefig(ruta, dpi=110)
        plt.close()
        mlflow.log_artifact(str(ruta))

        model.save(MODELS / "epp_mobilenet.keras")
        (MODELS / "classes.json").write_text(json.dumps(clases), encoding="utf-8")
        mlflow.log_artifact(str(MODELS / "classes.json"))

    print(classification_report(yte, pred, target_names=clases, zero_division=0))
    print({k: round(float(v), 3) for k, v in met.items()})


if __name__ == "__main__":
    main()
