# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:35:50 2024

@author: Mico
"""

import os
from skimage import io
import numpy as np
import pandas as pd

import geo_utils as geo

def create_input_image(elevation, ndvi_mean, ndvi_std):
    ndvi_mean = np.nan_to_num(ndvi_mean, nan=0)
    ndvi_std = np.nan_to_num(ndvi_std, nan=0)
    ch0 = np.squeeze(elevation).copy()
    e_min = ch0.min()
    e_max = 1000
    ch0 = (ch0 - e_min) / (e_max + e_min)

    ch1 = np.squeeze(ndvi_mean).copy()
    lower_bound = np.percentile(ch1, 5)
    upper_bound = np.percentile(ch1, 99)
    ch1 = np.clip(ch1, lower_bound, upper_bound)

    #ch1 = (ch1 + 1) / 2

    ch2 = np.squeeze(ndvi_std).copy()
    lower_bound = np.percentile(ch2, 5)
    upper_bound = np.percentile(ch2, 99)
    ch2 = np.clip(ch2, lower_bound, upper_bound)

    h, w = ch1.shape[0], ch1.shape[1]

    ch0 = ch0[:h,:w]
    ch1 = ch1[:h,:w]
    ch2 = ch2[:h,:w]

    img_composed = np.dstack((ch0, ch1, ch2))
    img_composed = np.clip(img_composed, 0, 1)
    return img_composed

if __name__ == "__main__":
    input_folders = ['USA', 'USA2', 'valpo']
    stats = []
    for folder in input_folders:
        input_folder = os.path.join('..', 'data', folder)
        img_composed_path = os.path.join(input_folder, 'composed.tif') 
       
        elevation = os.path.join(input_folder, 'elevation.tif')
        ndvi_mean = os.path.join(input_folder, 'NDVI_mean.tif')
        ndvi_std = os.path.join(input_folder, 'NDVI_stdDev.tif')
        
        elevation, _ = geo.load_raster(elevation)
        elevation = geo.reverse_axis(elevation)
        ndvi_mean, profile = geo.load_raster(ndvi_mean)
        ndvi_mean = geo.reverse_axis(ndvi_mean)
        ndvi_std, _ = geo.load_raster(ndvi_std)
        ndvi_std = geo.reverse_axis(ndvi_std)
        
        stats += [pd.DataFrame(elevation.ravel()).describe([0.05, 0.1, 0.95, 0.99]).T]
        stats += [pd.DataFrame(ndvi_mean.ravel()).describe([0.05, 0.1, 0.95, 0.99]).T]
        stats += [pd.DataFrame(ndvi_std.ravel()).describe([0.05, 0.1, 0.95, 0.99]).T]
    
        img_composed = create_input_image(elevation, ndvi_mean, ndvi_std)
        profile_new = profile.copy()
        profile_new['dtype'] = str(img_composed.dtype)
        profile_new['count'] = 3
        geo.save_raster(geo.reverse_axis(img_composed), img_composed_path, profile_new)
        
        io.imsave(img_composed_path.replace('.tif', '.jpg'), img_composed)

stats = pd.concat(stats)
bands = ['elevation', 'ndvi_mean', 'ndvi_std']
stats['band'] = bands*len(input_folders)
rasters_names = []
for folder in input_folders:
    rasters_names += [folder]*len(bands)
stats['raster'] = rasters_names


stats.to_csv('bands_stats.csv')
