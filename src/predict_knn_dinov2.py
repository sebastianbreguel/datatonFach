# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:20:10 2024

@author: Mico
"""
import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import label2rgb
from tqdm import tqdm
from datetime import datetime

import geo_utils as geo
from faiss_knn import FaissKNeighbors, FaissKMeans
from featurizer_dinov2 import DeepFeaturizer
from raster_sliding_window_predict import sliding_window, image_from_patches


def create_labelmap(reduced=False):
    label_names = ['high', 'low', 'moderate', 'non-burnable', 'very_high', 'low', 'water']
    label_ids = list(np.arange(0, len(label_names)) + 1)
    return label_names, label_ids

def reduce_output_clases(y_pred_img):
    label_names, label_ids = create_labelmap(reduced=False)
    new_labels, new_ids = create_labelmap(reduced=True)

    for old_id, new_idx in zip(label_ids, new_ids):
        if new_idx != old_id:
            y_pred_img[y_pred_img == old_id] = new_idx
    return y_pred_img


if __name__ == "__main__":
    input_raster_path = os.path.join('..', 'data', 'valpo', 'composed.tif')
    #input_raster_path = os.path.join('..', 'data', 'USA', 'composed.tif')
    #input_raster_path = os.path.join('..', 'data', 'USA2', 'composed.tif')

    img_composed, profile = geo.load_raster(input_raster_path)

    img_composed = geo.reverse_axis(img_composed)
    io.imshow(img_composed)
    io.show()

    model_name = 'dinov2_knn_firerisk'
    models_dir = os.path.join('..', 'models')
    model_path = os.path.join(models_dir, model_name)
    kmeans_name = 'dinov2_kmeans_firerisk'
    kmeans_model_path = os.path.join(models_dir, kmeans_name)

    featurizer = DeepFeaturizer(backbone_size='base')
    model = FaissKNeighbors(k=11)
    model.load(model_path)

    kmeans = FaissKMeans()
    kmeans.load(kmeans_model_path)

    X = []
    patch_size = (224, 224)
    stride = 112

    h, w = img_composed.shape[0:2]
    h_new = int(patch_size[0] * np.ceil(h / patch_size[0]))
    w_new = int(patch_size[0] * np.ceil(w / patch_size[1])) 

    img_composed_aux = resize(img_composed, (h_new, w_new), order=0)
    img_composed_aux[0:h, 0:w,:] = img_composed
    img_composed = img_composed_aux

    im_patches = sliding_window(img_composed, patch_size, stride)
    for patch_i in tqdm(range(0, im_patches.shape[0])):
        im_patch = im_patches[patch_i, :, :, :]
        im_patch = resize(im_patch, (im_patch.shape[0]*2, im_patch.shape[1]*2), order=2)
    
        features = featurizer.get_features(im_patch)
        fmap_size = features.shape[0:2]
        features_flat = features.reshape(-1, features.shape[-1])
        X.append(features_flat)
    X = np.vstack(X)

    #y_pred_img = kmeans.predict(X).reshape(-1, 1)
    y_pred_img = model.predict(X).reshape(-1, 1)
    y_pred_img = y_pred_img.reshape(-1, fmap_size[0], fmap_size[1], 1)
    original_shape = (img_composed.shape[0], img_composed.shape[1]) + (1,)
    y_pred_img = image_from_patches(y_pred_img,  original_shape, patch_size, stride)

    y_pred_img = y_pred_img[0:h, 0:w, :]
    io.imshow(img_composed)
    io.show()

    io.imshow(label2rgb(np.squeeze(y_pred_img).astype(np.uint8)))
    io.show()

    y_pred_img_less_classes = reduce_output_clases(y_pred_img)
    io.imshow(label2rgb(np.squeeze(y_pred_img_less_classes).astype(np.uint8)))
    io.show()

    ts = datetime.now().timestamp()
    ts = str(ts).replace('.', '_')
    output_fn = f'prediction_{ts}.tif'

    profile_new = profile.copy()
    profile_new['count'] = 1
    profile_new['nodata'] = 0
    geo.save_raster(geo.reverse_axis(y_pred_img), output_fn, profile_new)
