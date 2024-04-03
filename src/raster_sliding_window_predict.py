# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:33:54 2024

@author: Mico
"""

import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

import geo_utils as geo
from featurizer_dinov2 import DeepFeaturizer

def calculate_stride(image_shape, patch_size):
    height, width = image_shape[:2]
    patch_height, patch_width = patch_size
    height_divisors = [i for i in range(1, min(height, patch_height) + 1) if height % i == 0 and patch_height % i == 0]
    width_divisors = [i for i in range(1, min(width, patch_width) + 1) if width % i == 0 and patch_width % i == 0]
    stride = max(height_divisors + width_divisors)
    return stride

def image_from_patches(patches, original_shape, patch_size, stride, overlap_avg=False):
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
                current_patch = reconstructed_image[y:y+patch_size[0], x:x+patch_size[1], :]
                patch = (patch + current_patch)/2
            reconstructed_image[y:y+patch_size[0], x:x+patch_size[1], :] = patch
            patch_count += 1
    return reconstructed_image


def sliding_window(image, patch_size, stride):
    height, width = image.shape[:2]
    num_rows = (height - patch_size[0]) // stride + 1
    num_cols = (width - patch_size[1]) // stride + 1
    patches = []
    for y in range(0, num_rows * stride, stride):
        for x in range(0, num_cols * stride, stride):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)
    return np.array(patches)

featurizer = DeepFeaturizer()

raster_path = os.path.join('..', 'data', 'composed.tif')
raster, profile = geo.load_raster(raster_path)

raster = geo.reverse_axis(raster)

patch_size = (224, 224)
stride = 25

patches = sliding_window(raster, patch_size, stride)
predictions = []
for patch_i in tqdm(range(0, patches.shape[0])):
    patch = patches[patch_i, :, :, :]
    pred = featurizer.predict(patch)
    predictions.append(pred)
predictions = np.array(predictions)

im_pred = image_from_patches(patches=predictions,
                             original_shape=raster.shape,
                             patch_size=patch_size,
                             stride=stride)

profile_new = profile.copy()
profile_new['nodata'] = 0
geo.save_raster(geo.reverse_axis(im_pred), 'pred.tif', profile_new)
