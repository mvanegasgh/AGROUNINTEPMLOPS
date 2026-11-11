"""Fase 1 (Semana 5): MLP en Keras sobre las caracteristicas clasicas.
Uso: python -m src.mlp_keras [--epochs 150]
"""
import argparse

import matplotlib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import (REPORTS, SEED, feature_cols, load_features, mlflow_uri, resumen_split,
                     split, split_grupos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)

    df = load_features()
    cols = feature_cols(df)
    train, test = split(df)
    resumen_split(train, test)
    tr_df, va_df = split_grupos(train, 0.2)   # validacion tambien separada por persona
    le = LabelEncoder().fit(df.clase)
    sc = StandardScaler().fit(tr_df[cols].values)
    Xtr, ytr = sc.transform(tr_df[cols].values), le.transform(tr_df.clase)
    Xva, yva = sc.transform(va_df[cols].values), le.transform(va_df.clase)
    Xte, yte = sc.transform(test[cols].values), le.transform(test.clase)
    n = len(le.classes_)
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(n), y=ytr)))

    model = tf.keras.Sequential([
        tf.keras.layers.Input((Xtr.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("agrointep-fase1")
    with mlflow.start_run(run_name="MLP_Keras"):
        h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=32,
                      class_weight=cw, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=15,
                                                                  restore_best_weights=True)])
        pred = model.predict(Xte, verbose=0).argmax(1)
        _, rec, _, _ = precision_recall_fscore_support(yte, pred, labels=range(n),
                                                       zero_division=0)
        met = {"test_accuracy": accuracy_score(yte, pred),
               "test_f1_macro": f1_score(yte, pred, average="macro"),
               "epocas": len(h.history["loss"])}
        met.update({f"recall_{c}": r for c, r in zip(le.classes_, rec)})
        mlflow.log_params({"arquitectura": "128-64", "dropout": 0.3})
        mlflow.log_metrics(met)
        ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=le.classes_,
                                                cmap="Blues")
        ruta = REPORTS / "cm_MLP_Keras.png"
        plt.savefig(ruta, dpi=110)
        plt.close()
        mlflow.log_artifact(str(ruta))
    print({k: round(float(v), 3) for k, v in met.items()})


if __name__ == "__main__":
    main()
