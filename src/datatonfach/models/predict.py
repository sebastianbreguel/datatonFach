"""Run KNN prediction over a full raster using sliding windows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from ..config import DATA_DIR, DEFAULT_PATCH_SIZE, DEFAULT_STRIDE, MODELS_DIR, PREDICTIONS_DIR, ensure_dirs
from ..features.featurizer import DeepFeaturizer
from ..io import load_raster, reverse_axis, save_raster
from .knn import FaissKMeans, FaissKNeighbors
from .sliding_window import image_from_patches, sliding_window
from .train import KMEANS_NAME, MODEL_NAME


def create_labelmap() -> tuple[list[str], list[int]]:
    """Reduced label map used at inference time (merges very_low into low)."""
    label_names = ["high", "low", "moderate", "non-burnable", "very_high", "low", "water"]
    label_ids = list(np.arange(0, len(label_names)) + 1)
    return label_names, label_ids


def reduce_output_clases(y_pred_img: np.ndarray) -> np.ndarray:
    """Collapse duplicated classes in the prediction map."""
    _, old_ids = create_labelmap()
    _, new_ids = create_labelmap()
    for old_id, new_idx in zip(old_ids, new_ids, strict=True):
        if new_idx != old_id:
            y_pred_img[y_pred_img == old_id] = new_idx
    return y_pred_img


def predict_raster(
    input_raster: str | Path,
    backbone_size: str = "base",
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> Path:
    """Predict a class map for a composed GeoTIFF and save it under PREDICTIONS_DIR."""
    ensure_dirs()
    img_composed, profile = load_raster(input_raster)
    img_composed = reverse_axis(img_composed)

    featurizer = DeepFeaturizer(backbone_size=backbone_size)
    knn = FaissKNeighbors(k=11)
    knn.load(MODELS_DIR / MODEL_NAME)
    kmeans = FaissKMeans()
    kmeans.load(MODELS_DIR / KMEANS_NAME)

    h, w = img_composed.shape[0:2]
    h_new = int(patch_size[0] * np.ceil(h / patch_size[0]))
    w_new = int(patch_size[0] * np.ceil(w / patch_size[1]))

    padded = resize(img_composed, (h_new, w_new), order=0)
    padded[0:h, 0:w, :] = img_composed
    img_composed = padded

    im_patches = sliding_window(img_composed, patch_size, stride)
    features_all: list[np.ndarray] = []
    fmap_size: tuple[int, int] = (0, 0)
    for patch_i in tqdm(range(im_patches.shape[0])):
        im_patch = im_patches[patch_i, :, :, :]
        im_patch = resize(im_patch, (im_patch.shape[0] * 2, im_patch.shape[1] * 2), order=2)
        features = featurizer.get_features(im_patch)
        fmap_size = features.shape[0:2]
        features_all.append(features.reshape(-1, features.shape[-1]))
    x = np.vstack(features_all)

    y_pred_img = knn.predict(x).reshape(-1, 1)
    y_pred_img = y_pred_img.reshape(-1, fmap_size[0], fmap_size[1], 1)
    original_shape = (img_composed.shape[0], img_composed.shape[1], 1)
    y_pred_img = image_from_patches(y_pred_img, original_shape, patch_size, stride)
    y_pred_img = y_pred_img[0:h, 0:w, :]

    ts = str(datetime.now().timestamp()).replace(".", "_")
    output_fn = PREDICTIONS_DIR / f"prediction_{ts}.tif"

    profile_new = profile.copy()
    profile_new["count"] = 1
    profile_new["nodata"] = 0
    save_raster(reverse_axis(y_pred_img), output_fn, profile_new)
    return output_fn


def predict_default() -> Path:
    """Convenience wrapper that predicts on data/valpo/composed.tif."""
    return predict_raster(DATA_DIR / "valpo" / "composed.tif")
