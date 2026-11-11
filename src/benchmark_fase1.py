"""Fase 1 (Primer corte): comparativa de modelos de ML clasico + PCA + K-Means.

Uso: python -m src.benchmark_fase1
Registra cada modelo en MLflow (experimento 'agrointep-fase1').
"""
import time
import warnings

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, adjusted_rand_score,
                             confusion_matrix, f1_score, homogeneity_score,
                             normalized_mutual_info_score, precision_recall_fscore_support,
                             silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import (MODELS, REPORTS, SEED, feature_cols, load_features, mlflow_uri,
                     resumen_split, split)


def figura_pca(X, y, clases):
    Xs = StandardScaler().fit_transform(X)
    p = PCA(random_state=SEED).fit(Xs)
    Z = p.transform(Xs)[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(np.cumsum(p.explained_variance_ratio_), marker="o", ms=3)
    ax[0].axhline(0.95, ls="--", c="gray")
    ax[0].set(title="Varianza acumulada (PCA)", xlabel="componentes", ylabel="varianza")
    for c in clases:
        m = y == c
        ax[1].scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6, label=c)
    ax[1].legend()
    ax[1].set(title="Proyeccion PCA 2D", xlabel="PC1", ylabel="PC2")
    plt.tight_layout()
    ruta = REPORTS / "pca.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta


def pureza(y, clusters):
    return pd.crosstab(clusters, y).max(axis=1).sum() / len(y)


def main():
    df = load_features()
    cols = feature_cols(df)
    clases = sorted(df.clase.unique())
    train, test = split(df)
    resumen_split(train, test)
    gtr = train.grupo.to_numpy(dtype=object)
    Xtr, ytr = train[cols].values, train.clase.to_numpy(dtype=object)
    Xte, yte = test[cols].values, test.clase.to_numpy(dtype=object)

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment("agrointep-fase1")
    n_splits = max(2, min(5, len(set(gtr))))
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)  # CV tambien por persona

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(300, class_weight="balanced",
                                               random_state=SEED, n_jobs=-1),
        "SVM_RBF": SVC(C=3, class_weight="balanced", probability=True, random_state=SEED),
    }
    filas, ajustados = [], {}
    for nombre, clf in modelos.items():
        pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(0.95, random_state=SEED)),
                         ("clf", clf)])
        with mlflow.start_run(run_name=nombre):
            cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, groups=gtr, scoring="f1_macro")
            t0 = time.time()
            pipe.fit(Xtr, ytr)
            t_fit = time.time() - t0
            pred = pipe.predict(Xte)
            _, rec, _, _ = precision_recall_fscore_support(yte, pred, labels=clases,
                                                           zero_division=0)
            met = {"cv_f1_macro_mean": cv_f1.mean(), "cv_f1_macro_std": cv_f1.std(),
                   "test_accuracy": accuracy_score(yte, pred),
                   "test_f1_macro": f1_score(yte, pred, average="macro"),
                   "tiempo_fit_s": t_fit, "n_componentes_pca": pipe["pca"].n_components_}
            met.update({f"recall_{c}": r for c, r in zip(clases, rec)})
            mlflow.log_param("modelo", nombre)
            mlflow.log_metrics(met)
            ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=clases),
                                   display_labels=clases).plot(cmap="Blues")
            ruta = REPORTS / f"cm_{nombre}.png"
            plt.savefig(ruta, dpi=110)
            plt.close()
            mlflow.log_artifact(str(ruta))
            ajustados[nombre] = pipe
            filas.append({"modelo": nombre, "tipo": "supervisado", **met})

    # No supervisado: K-Means con k = n. de clases (las etiquetas solo se usan para evaluar)
    X_all, y_all = df[cols].values, df.clase.to_numpy(dtype=object)
    Z = Pipeline([("sc", StandardScaler()),
                  ("pca", PCA(0.95, random_state=SEED))]).fit_transform(X_all)
    km = KMeans(len(clases), n_init=10, random_state=SEED).fit(Z)
    with mlflow.start_run(run_name="KMeans"):
        met = {"silhouette": silhouette_score(Z, km.labels_),
               "ARI": adjusted_rand_score(y_all, km.labels_),
               "NMI": normalized_mutual_info_score(y_all, km.labels_),
               "homogeneidad": homogeneity_score(y_all, km.labels_),
               "pureza": pureza(y_all, km.labels_)}
        mlflow.log_param("k", len(clases))
        mlflow.log_metrics(met)
        mlflow.log_artifact(str(figura_pca(X_all, y_all, clases)))
        filas.append({"modelo": "KMeans", "tipo": "no_supervisado", **met})

    res = pd.DataFrame(filas)
    # El mejor modelo clasico (por F1 en validacion cruzada) alimenta el prototipo en vivo
    sup = res[res.tipo == "supervisado"]
    mejor = sup.loc[sup.cv_f1_macro_mean.idxmax(), "modelo"]
    joblib.dump(ajustados[mejor], MODELS / "clasico.joblib")
    print(f"Mejor modelo clasico (guardado para video en vivo): {mejor}\n")
    res.to_csv(REPORTS / "benchmark_fase1.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[["modelo", "tipo", "cv_f1_macro_mean", "test_f1_macro", "test_accuracy",
               "silhouette", "ARI", "pureza"]].round(3).to_string(index=False))
    print("\nRecall por clase (clave: no omitir un EPP obligatorio):")
    print(res[["modelo"] + [f"recall_{c}" for c in clases]].dropna().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
