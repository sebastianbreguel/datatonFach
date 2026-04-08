"""Smoke tests for geo helpers — no external files, no GPU."""

from __future__ import annotations

import numpy as np

from datatonfach.io.geo import reverse_axis
from datatonfach.models.sliding_window import image_from_patches, sliding_window


def test_reverse_axis_roundtrip() -> None:
    channel_first = np.zeros((3, 10, 12), dtype=np.float32)
    channel_last = reverse_axis(channel_first)
    assert channel_last.shape == (10, 12, 3)
    assert reverse_axis(channel_last).shape == (3, 10, 12)


def test_sliding_window_shapes() -> None:
    img = np.random.rand(224, 448, 3).astype(np.float32)
    patches = sliding_window(img, (224, 224), 224)
    assert patches.shape == (2, 224, 224, 3)


def test_image_from_patches_roundtrip() -> None:
    img = np.random.rand(224, 448, 3).astype(np.float32)
    patches = sliding_window(img, (224, 224), 224)
    recon = image_from_patches(patches, img.shape, (224, 224), 224)
    assert recon.shape == img.shape
    np.testing.assert_allclose(recon, img)
