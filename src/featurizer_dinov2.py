# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:06:39 2024

@author: Mico
"""

import torch
import numpy as np
from skimage.transform import resize
from sklearn.decomposition import PCA


class DeepFeaturizer():
    def __init__(self, backbone_size='small'):
        self.model = load_model(backbone_size)

    def get_features(self, img):
        features, _, grid_shape = get_dense_descriptor(self.model, img)

        fmap_shape = grid_shape + (features.shape[-1],)
        features = features.reshape(fmap_shape)

        return features

    def predict(self, img):
        features, _, grid_shape = get_dense_descriptor(self.model, img)
        features_rgb = pca_colorize(features, grid_shape)
        
        return features_rgb
        
def load_model(backbone_size='small'):
    backbone_archs = {"small": "vits14",
                      "base": "vitb14",
                      "large": "vitl14",
                      "giant": "vitg14"}

    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    model.eval()
    model.cuda()
    return model

def get_grid_shape(h, w, patch_size):
    grid_shape = (h // patch_size), (w // patch_size)
    return grid_shape

def prepare_image(img, patch_size=14):
    mean = (0.485, 0.456, 0.406) # DinoV2 mean std originales
    std = (0.229, 0.224, 0.225)
    h, w, ch = img.shape
    grid_shape = get_grid_shape(h, w, patch_size)
    h_new = patch_size * grid_shape[0]
    w_new = patch_size * grid_shape[1]

    if h_new != h and w_new != w:
        img = resize(img, (h_new, w_new, ch), order=3)
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.as_tensor(img, dtype=torch.float32).cuda()
    return img, grid_shape

def get_dense_descriptor(model, img):
    with torch.no_grad():
        img_tensor, grid_shape = prepare_image(img)
        features_tensor = model.patch_embed(img_tensor)
        attention_tensor = model.get_intermediate_layers(img_tensor)[0]

        features = features_tensor.cpu().detach().numpy()
        attention = attention_tensor.cpu().detach().numpy()

    del img_tensor
    del features_tensor
    del attention_tensor

    torch.cuda.empty_cache()

    features = np.squeeze(features)
    attention = np.squeeze(attention, axis=0)
    return features, attention, grid_shape

def min_max_scale(data):
    return (data - data.min()) / (data.max() - data.min())

def pca_colorize(features, output_shape):
    pca = PCA(n_components=3)
    pca.fit(features)
    rgb = pca.transform(features)
    rgb = min_max_scale(rgb)
    rgb = rgb.reshape(output_shape + (3,))
    return rgb
