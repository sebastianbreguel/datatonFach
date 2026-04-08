"""Validate predicted rasters against bbox ground-truth CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from ..config import DATA_DIR
from ..features.dataset import risk_cls_from_boxes
from ..io import load_raster, reverse_axis
from ..models.predict import create_labelmap, reduce_output_clases


def run_validation(bbox_path: str | Path | None = None, pred_raster_path: str | Path | None = None, show_plots: bool = False) -> str:
    """Compute a classification report between a predicted raster and bbox labels."""
    bbox_path = Path(bbox_path) if bbox_path else DATA_DIR / "valpo" / "bbox_labels.csv"
    pred_raster_path = Path(pred_raster_path) if pred_raster_path else DATA_DIR / "predictions" / "prediction_1712318539_986473.tif"

    label_names, _ = create_labelmap()
    y_pred_img, _ = load_raster(pred_raster_path)
    y_pred_img = reverse_axis(y_pred_img)

    risk_cls_resized = risk_cls_from_boxes(str(bbox_path), label_names)
    mask = risk_cls_resized > 0

    y_true = reduce_output_clases(risk_cls_resized[mask])
    y_pred = reduce_output_clases(y_pred_img[mask])

    label_names_with_bg = ["background"] + label_names
    y_true_str = [label_names_with_bg[int(idx)] for idx in y_true]
    y_pred_str = [label_names_with_bg[int(idx)] for idx in y_pred]

    report = classification_report(y_true_str, y_pred_str)
    print(report)
    if show_plots:
        ConfusionMatrixDisplay.from_predictions(y_true_str, y_pred_str, normalize="true", cmap=plt.cm.Blues)
        plt.show()
    return report
