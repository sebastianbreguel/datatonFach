"""Build (features, labels) training arrays from composed rasters and bbox labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm

from ..config import DATA_DIR, DEFAULT_PATCH_SIZE, DEFAULT_STRIDE, FEATURES_DIR, LABEL_NAMES, ensure_dirs
from ..io import load_raster, reverse_axis
from ..models.sliding_window import sliding_window
from .featurizer import DeepFeaturizer


def create_features_dataset(
    featurizer: DeepFeaturizer,
    img_composed: np.ndarray,
    risk_cls_resized: np.ndarray,
    patch_size: tuple[int, int] = DEFAULT_PATCH_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide over the image, extract DinoV2 features and align labels."""
    im_patches = sliding_window(img_composed, patch_size, stride)
    label_patches = sliding_window(risk_cls_resized, patch_size, stride)

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for patch_i in tqdm(range(im_patches.shape[0])):
        im_patch = im_patches[patch_i, :, :, :]
        im_patch = resize(im_patch, (im_patch.shape[0] * 2, im_patch.shape[1] * 2), order=2)

        features = featurizer.get_features(im_patch)
        fmap_size = features.shape[0:2]
        label_cls = resize(label_patches[patch_i, :, :, :], fmap_size, order=0)

        mask = (label_cls > 0).flatten()
        if mask.max():
            feats.append(features.reshape(-1, features.shape[-1])[mask])
            labels.append(label_cls.reshape(-1, 1)[mask])

    return np.vstack(feats), np.vstack(labels)


def risk_cls_from_boxes(bbox_path: str, label_names: list[str]) -> np.ndarray:
    """Rasterize a CSV of labeled bounding boxes into a class map."""
    df_bbox = pd.read_csv(bbox_path)
    h, w = df_bbox.iloc[0]["image_height"], df_bbox.iloc[0]["image_width"]
    risk_cls_resized = np.zeros((h, w, 1), dtype=int)
    for _, row in df_bbox.iterrows():
        label = label_names.index(row["label_name"]) + 1
        ymin, ymax = row["bbox_y"], row["bbox_y"] + row["bbox_height"]
        xmin, xmax = row["bbox_x"], row["bbox_x"] + row["bbox_width"]
        risk_cls_resized[ymin:ymax, xmin:xmax] = label
    return risk_cls_resized


def build_training_dataset(train_folders: list[str] | None = None, backbone_size: str = "base") -> None:
    """Generate features.npy / labels_cls.npy under data/features/."""
    ensure_dirs()
    train_folders = train_folders or ["valpo", "USA", "USA2"]
    featurizer = DeepFeaturizer(backbone_size=backbone_size)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for folder in train_folders:
        input_dir = DATA_DIR / folder
        img_composed, _ = load_raster(input_dir / "composed.tif")
        img_composed = reverse_axis(img_composed)

        bbox_path = input_dir / "bbox_labels.csv"
        if bbox_path.exists():
            risk_cls_resized = risk_cls_from_boxes(str(bbox_path), LABEL_NAMES)
        else:
            risk_cls, _ = load_raster(input_dir / "whp2023_cls_conus_crop.tif")
            risk_cls = reverse_axis(risk_cls)
            risk_cls_resized = resize(risk_cls, img_composed.shape[0:2], order=0, preserve_range=True)

        x, y = create_features_dataset(featurizer, img_composed, risk_cls_resized)
        xs.append(x)
        ys.append(y)

    np.save(FEATURES_DIR / "features.npy", np.vstack(xs))
    np.save(FEATURES_DIR / "labels_cls.npy", np.vstack(ys))
