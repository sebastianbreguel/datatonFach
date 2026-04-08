"""GeoTIFF I/O and axis helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def load_raster(tif_path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a GeoTIFF using rasterio.

    Returns:
        data: channel-first numpy array.
        profile: rasterio profile with geospatial metadata.
    """
    with rasterio.open(str(tif_path)) as dataset:
        data = dataset.read()
        profile = dataset.profile
    return data, profile


def save_raster(data: np.ndarray, output_path: str | Path, profile: dict) -> None:
    """Save a numpy array to a GeoTIFF (channel-first)."""
    profile_new = profile.copy()
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    profile_new["count"] = min(data.shape)
    with rasterio.open(str(output_path), "w", **profile_new) as dst:
        dst.write(data)


def reverse_axis(raster: np.ndarray) -> np.ndarray:
    """Swap between channel-first and channel-last layouts."""
    ch = min(raster.shape)
    channel_first = list(raster.shape).index(ch) == 0
    if channel_first:
        return np.transpose(raster, (1, 2, 0))
    return np.transpose(raster, (2, 0, 1))
