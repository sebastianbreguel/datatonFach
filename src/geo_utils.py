# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:03:05 2024

@author: Mico
"""

import numpy as np
import rasterio


def load_raster(tif_path):
    """ Load a GeoTIFF using rasterio

    Args:
        tif_path (str): path to the raster file.

    Returns:
        data (np.array): image.
        profile (dict): rasterio profile with geospatial metadata.

    """
    with rasterio.open(tif_path) as dataset:
        data = dataset.read()
        profile = dataset.profile
    return data, profile


def save_raster(data, output_path, profile):
    """ Save a GeoTIFF from a numpy array

    Args:
        data (np.array): numpy array with channel first format (ch, w, h).
        output_path (str): path to save the raster.
        profile (dict): rasterio profile with geospatial metadata.
    """
    profile_new = profile.copy()
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    number_of_bands = min(data.shape)
    profile_new['count'] = number_of_bands
    with rasterio.open(output_path, 'w', **profile_new) as dst:
        dst.write(data)


def reverse_axis(raster):
    ch = min(raster.shape)
    channel_first = list(raster.shape).index(ch) == 0
    if channel_first:
        raster = np.transpose(raster, (1, 2, 0))
    else:
        raster = np.transpose(raster, (2, 0, 1))
    return raster