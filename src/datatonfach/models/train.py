"""Train a FAISS KNN fire-risk classifier on DinoV2 features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ..config import FEATURES_DIR, LABEL_NAMES, MODELS_DIR, ensure_dirs
from .knn import FaissKMeans, FaissKNeighbors

MODEL_NAME = "dinov2_knn_firerisk"
KMEANS_NAME = "dinov2_kmeans_firerisk"


def train(k: int = 11, n_splits: int = 5, show_plots: bool = False) -> None:
    """Run stratified k-fold CV and persist the best model + KMeans helper."""
    ensure_dirs()
    X = np.load(FEATURES_DIR / "features.npy")
    y = np.load(FEATURES_DIR / "labels_cls.npy")

    kmeans = FaissKMeans()
    kmeans.fit(X, k=16)

    label_ids = np.arange(0, len(LABEL_NAMES)) + 1

    models_objs: dict[int, FaissKNeighbors] = {}
    df_metrics: list[pd.DataFrame] = []

    kf = StratifiedKFold(n_splits=n_splits)
    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = FaissKNeighbors(k=k)
        model.fit(X_train, y_train)
        models_objs[i] = model
        y_pred = model.predict(X_test)

        test_report_str = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
        test_report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=label_ids, normalize="true")
        acc_per_class = np.diagonal(cm)

        df_report = pd.DataFrame(test_report)[LABEL_NAMES].T
        df_report["accuracy"] = acc_per_class
        df_report["fold"] = i
        df_metrics.append(df_report.reset_index(drop=False))
        print(test_report_str)

        if show_plots:
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true", display_labels=LABEL_NAMES, cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix Kfold {i}")
            plt.show()

    combined = _aggregate_metrics(pd.concat(df_metrics, ignore_index=True))

    best_fold = int(pd.concat(df_metrics).groupby("fold")["f1-score"].mean().argmax())
    best_model = models_objs[best_fold]

    combined.to_csv(MODELS_DIR / f"{MODEL_NAME}_metrics.csv", encoding="utf-8-sig")
    best_model.save(MODELS_DIR / MODEL_NAME)
    kmeans.save(MODELS_DIR / KMEANS_NAME)
    print(combined)


def _aggregate_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics_cols = ["precision", "recall", "f1-score", "accuracy"]
    avg = df_metrics.groupby(["index"]).mean()
    std = df_metrics.groupby(["index"]).std()

    combined = avg[metrics_cols + ["support"]].copy()
    for col in metrics_cols:
        combined[col] = avg[col].map("{:,.2f}".format) + " ± " + std[col].map("{:,.2f}".format)

    support = np.tile(avg["support"].values.reshape(-1, 1), (1, len(metrics_cols)))
    macro_avg = list(avg[metrics_cols].values.mean(axis=0))
    total_support = support.sum(axis=0).mean()
    macro_avg.append(total_support)

    weighted = avg[metrics_cols].values * support
    weighted = weighted.sum(axis=0) / support.sum(axis=0).reshape(1, -1)
    weighted_list = list(np.squeeze(weighted))
    weighted_list.append(total_support)

    combined = combined.T
    combined["macro avg"] = [f"{np.round(v, 3)}" for v in macro_avg]
    combined["weighted avg"] = [f"{np.round(v, 3)}" for v in weighted_list]
    return combined.T
