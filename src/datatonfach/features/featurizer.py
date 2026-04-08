"""DinoV2-based dense feature extractor."""

from __future__ import annotations

import numpy as np
import torch
from skimage.transform import resize
from sklearn.decomposition import PCA

_BACKBONE_ARCHS = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PATCH_SIZE = 14


class DeepFeaturizer:
    """Wraps a DinoV2 ViT and exposes dense patch features."""

    def __init__(self, backbone_size: str = "small") -> None:
        self.model = self._load_model(backbone_size)

    @staticmethod
    def _load_model(backbone_size: str) -> torch.nn.Module:
        backbone_name = f"dinov2_{_BACKBONE_ARCHS[backbone_size]}"
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model

    def get_features(self, img: np.ndarray) -> np.ndarray:
        """Return per-patch dense features shaped (gh, gw, dim)."""
        features, _, grid_shape = _get_dense_descriptor(self.model, img)
        return features.reshape(grid_shape + (features.shape[-1],))

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Return a false-RGB visualization of the PCA-projected features."""
        features, _, grid_shape = _get_dense_descriptor(self.model, img)
        return _pca_colorize(features, grid_shape)


def _get_grid_shape(h: int, w: int, patch_size: int) -> tuple[int, int]:
    return (h // patch_size, w // patch_size)


def _prepare_image(img: np.ndarray, patch_size: int = _PATCH_SIZE) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w, ch = img.shape
    grid_shape = _get_grid_shape(h, w, patch_size)
    h_new = patch_size * grid_shape[0]
    w_new = patch_size * grid_shape[1]

    if h_new != h and w_new != w:
        img = resize(img, (h_new, w_new, ch), order=3)
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    tensor = torch.as_tensor(img, dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor, grid_shape


def _get_dense_descriptor(model: torch.nn.Module, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    with torch.no_grad():
        img_tensor, grid_shape = _prepare_image(img)
        features_tensor = model.patch_embed(img_tensor)
        attention_tensor = model.get_intermediate_layers(img_tensor)[0]
        features = features_tensor.cpu().detach().numpy()
        attention = attention_tensor.cpu().detach().numpy()

    del img_tensor, features_tensor, attention_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.squeeze(features), np.squeeze(attention, axis=0), grid_shape


def _min_max_scale(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min())


def _pca_colorize(features: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    pca = PCA(n_components=3)
    pca.fit(features)
    rgb = pca.transform(features)
    rgb = _min_max_scale(rgb)
    return rgb.reshape(output_shape + (3,))
