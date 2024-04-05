# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:04:55 2024

@author: Mico
"""

import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import pandas as pd

import geo_utils as geo
from featurizer_dinov2 import DeepFeaturizer
from raster_sliding_window_predict import sliding_window


def create_features_dataset(featurizer, img_composed, risk_cls_resized, patch_size=(224, 224), stride=112):
    im_patches = sliding_window(img_composed, patch_size, stride)
    label_patches = sliding_window(risk_cls_resized, patch_size, stride)

    X = []
    y = []

    for patch_i in tqdm(range(0, im_patches.shape[0])):
        im_patch = im_patches[patch_i, :, :, :]
        im_patch = resize(im_patch, (im_patch.shape[0]*2, im_patch.shape[1]*2), order=2)

        features = featurizer.get_features(im_patch)
        fmap_size = features.shape[0:2]
        label_cls = resize(label_patches[patch_i, :, :, :], fmap_size, order=0)

        mask = (label_cls > 0).flatten()
        if mask.max():
            label_flat = label_cls.reshape(-1, 1)[mask]
            features_flat = features.reshape(-1, features.shape[-1])[mask]

            X.append(features_flat)
            y.append(label_flat)

    X = np.vstack(X)
    y = np.vstack(y)

    return X, y


def risk_cls_from_boxes(bbox_path, label_names):
    df_bbox = pd.read_csv(bbox_path)
    h, w = df_bbox.iloc[0]['image_height'], df_bbox.iloc[0]['image_width']
    risk_cls_resized = np.zeros((h, w, 1), dtype=int)
    for _, row in df_bbox.iterrows():
        label = label_names.index(row['label_name'])+1
        ymin, ymax = row['bbox_y'], row['bbox_y'] + row['bbox_height']
        xmin, xmax = row['bbox_x'], row['bbox_x'] + row['bbox_width']
        risk_cls_resized[ymin:ymax, xmin:xmax] = label
    return risk_cls_resized


if __name__ == "__main__":
    featurizer = DeepFeaturizer(backbone_size='base')
    train_folders = ['valpo', 'USA', 'USA2']
    label_names = ['high', 'low', 'moderate', 'non-burnable', 'very_high', 'very_low', 'water']
    label_ids = np.arange(0, len(label_names)) + 1
    Xs, ys = [], []
    for folder in train_folders:
        input_dir = os.path.join('..', 'data', folder)
        img_composed_path = os.path.join(input_dir, 'composed.tif')

        img_composed, profile = geo.load_raster(img_composed_path)
        img_composed = geo.reverse_axis(img_composed)

        bbox_path = os.path.join(input_dir, 'bbox_labels.csv')
        if os.path.exists(bbox_path):
            risk_cls_resized = risk_cls_from_boxes(bbox_path, label_names)
        else:
            risk_cls = os.path.join(input_dir, 'whp2023_cls_conus_crop.tif')
            risk_cls, label_profile = geo.load_raster(risk_cls)
            risk_cls = geo.reverse_axis(risk_cls)
            risk_cls_resized = resize(risk_cls, img_composed.shape[0:2], order=0, preserve_range=True)

        X, y = create_features_dataset(featurizer, img_composed, risk_cls_resized)

        Xs.append(X)
        ys.append(y)

    Xs = np.vstack(Xs)
    ys = np.vstack(ys)

    output_dir = os.path.join('..', 'data', 'features')

    np.save(os.path.join(output_dir, 'features.npy'), Xs)
    np.save(os.path.join(output_dir, 'labels_cls.npy'), ys)
