"""Sliding-window patchification utilities."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize


def calculate_stride(image_shape: tuple[int, ...], patch_size: tuple[int, int]) -> int:
    """Largest stride that tiles the image evenly for the given patch size."""
    height, width = image_shape[:2]
    patch_height, patch_width = patch_size
    height_divisors = [i for i in range(1, min(height, patch_height) + 1) if height % i == 0 and patch_height % i == 0]
    width_divisors = [i for i in range(1, min(width, patch_width) + 1) if width % i == 0 and patch_width % i == 0]
    return max(height_divisors + width_divisors)


def sliding_window(image: np.ndarray, patch_size: tuple[int, int], stride: int) -> np.ndarray:
    """Extract overlapping patches from a (H, W, C) image."""
    height, width = image.shape[:2]
    num_rows = (height - patch_size[0]) // stride + 1
    num_cols = (width - patch_size[1]) // stride + 1
    patches = []
    for y in range(0, num_rows * stride, stride):
        for x in range(0, num_cols * stride, stride):
            patches.append(image[y : y + patch_size[0], x : x + patch_size[1]])
    return np.array(patches)


def image_from_patches(
    patches: np.ndarray,
    original_shape: tuple[int, ...],
    patch_size: tuple[int, int],
    stride: int,
    overlap_avg: bool = False,
) -> np.ndarray:
    """Reconstruct an image from patches produced by :func:`sliding_window`."""
    height, width = original_shape[0:2]
    num_rows = (height - patch_size[0]) // stride + 1
    num_cols = (width - patch_size[1]) // stride + 1

    reconstructed_image = np.zeros(original_shape, dtype=patches.dtype)
    patch_count = 0
    for y in range(0, num_rows * stride, stride):
        for x in range(0, num_cols * stride, stride):
            patch = patches[patch_count]
            if patch.shape[0:2] != patch_size[0:2]:
                patch = resize(patch, patch_size, order=0)
            if overlap_avg:
                current_patch = reconstructed_image[y : y + patch_size[0], x : x + patch_size[1], :]
                patch = (patch + current_patch) / 2
            reconstructed_image[y : y + patch_size[0], x : x + patch_size[1], :] = patch
            patch_count += 1
    return reconstructed_image
