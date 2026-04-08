"""Count pixels per dNBR severity class from a GeoTIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# dNBR ranges (scaled by 1000)
DNBR_RANGES: dict[str, tuple[int, int]] = {
    "enhaced_regrowth_high": (-500, -251),
    "enhaced_regrowth_low": (-250, -101),
    "unburned": (-100, 99),
    "low_severity": (100, 269),
    "moderate_low_severity": (270, 439),
    "moderate_high_severity": (440, 659),
    "high_severity": (660, 1300),
}


def cuantizacion(tif_path: str | Path) -> dict[str, int]:
    """Count pixels falling within each dNBR severity range."""
    counts = {k: 0 for k in DNBR_RANGES}
    image_array = np.array(Image.open(str(tif_path)))
    for category, (min_val, max_val) in DNBR_RANGES.items():
        pixels_in_range = np.logical_and(image_array >= min_val, image_array <= max_val)
        counts[category] = int(np.sum(pixels_in_range))
    return counts
