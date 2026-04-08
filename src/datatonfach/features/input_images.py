"""Build false-RGB composed images from elevation + NDVI bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import io

from ..config import DATA_DIR
from ..io import load_raster, reverse_axis, save_raster


def create_input_image(elevation: np.ndarray, ndvi_mean: np.ndarray, ndvi_std: np.ndarray) -> np.ndarray:
    """Compose a 3-channel image from elevation, NDVI mean and NDVI std."""
    ndvi_mean = np.nan_to_num(ndvi_mean, nan=0)
    ndvi_std = np.nan_to_num(ndvi_std, nan=0)

    ch0 = np.squeeze(elevation).copy()
    e_min = ch0.min()
    e_max = 1000
    ch0 = (ch0 - e_min) / (e_max + e_min)

    ch1 = np.squeeze(ndvi_mean).copy()
    ch1 = np.clip(ch1, np.percentile(ch1, 5), np.percentile(ch1, 99))

    ch2 = np.squeeze(ndvi_std).copy()
    ch2 = np.clip(ch2, np.percentile(ch2, 5), np.percentile(ch2, 99))

    h, w = ch1.shape[0], ch1.shape[1]
    ch0 = ch0[:h, :w]
    ch1 = ch1[:h, :w]
    ch2 = ch2[:h, :w]

    img_composed = np.dstack((ch0, ch1, ch2))
    return np.clip(img_composed, 0, 1)


def build_composed_images(folders: list[str] | None = None) -> None:
    """Generate composed.tif/.jpg and bands_stats.csv for the given folders."""
    folders = folders or ["USA", "USA2", "valpo"]
    stats = []
    for folder in folders:
        input_folder = DATA_DIR / folder
        img_composed_path = input_folder / "composed.tif"

        elevation, _ = load_raster(input_folder / "elevation.tif")
        elevation = reverse_axis(elevation)
        ndvi_mean, profile = load_raster(input_folder / "NDVI_mean.tif")
        ndvi_mean = reverse_axis(ndvi_mean)
        ndvi_std, _ = load_raster(input_folder / "NDVI_stdDev.tif")
        ndvi_std = reverse_axis(ndvi_std)

        stats.append(pd.DataFrame(elevation.ravel()).describe([0.05, 0.1, 0.95, 0.99]).T)
        stats.append(pd.DataFrame(ndvi_mean.ravel()).describe([0.05, 0.1, 0.95, 0.99]).T)
        stats.append(pd.DataFrame(ndvi_std.ravel()).describe([0.05, 0.1, 0.95, 0.99]).T)

        img_composed = create_input_image(elevation, ndvi_mean, ndvi_std)
        profile_new = profile.copy()
        profile_new["dtype"] = str(img_composed.dtype)
        profile_new["count"] = 3
        save_raster(reverse_axis(img_composed), img_composed_path, profile_new)
        io.imsave(str(img_composed_path).replace(".tif", ".jpg"), img_composed)

    df_stats = pd.concat(stats)
    bands = ["elevation", "ndvi_mean", "ndvi_std"]
    df_stats["band"] = bands * len(folders)
    df_stats["raster"] = [f for f in folders for _ in bands]
    df_stats.to_csv(DATA_DIR / "bands_stats.csv")
