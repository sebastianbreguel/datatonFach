"""FAISS-backed KNN classifier and KMeans helper."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissKNeighbors:
    """Fast approximate KNN classifier backed by FAISS."""

    def __init__(self, k: int = 5) -> None:
        self.index: faiss.Index | None = None
        self.y: np.ndarray | None = None
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, use_ivff: bool = True) -> None:
        if use_ivff:
            dim = X.shape[1]
            nlist = 60
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(X.astype(np.float32))
        else:
            self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = np.squeeze(y)

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        assert self.index is not None and self.y is not None
        self.index.add(X_new.astype(np.float32))
        self.y = np.concatenate([self.y, y_new])

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.index is not None and self.y is not None
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        return np.array([np.argmax(np.bincount(x)) for x in votes])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.index is not None and self.y is not None
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        n_classes = len(set(self.y))
        votes = self.y[indices]
        vote_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=votes)
        return vote_counts / self.k

    def save(self, filepath: str | Path) -> None:
        filepath = str(filepath)
        faiss.write_index(self.index, filepath + "_index.faiss")
        with open(filepath + "_data.pkl", "wb") as f:
            pickle.dump((self.y, self.k), f)

    def load(self, filepath: str | Path) -> None:
        filepath = str(filepath)
        self.index = faiss.read_index(filepath + "_index.faiss")
        with open(filepath + "_data.pkl", "rb") as f:
            self.y, self.k = pickle.load(f)


class FaissKMeans:
    """FAISS KMeans with an elbow-based k selection fallback."""

    def __init__(self) -> None:
        self.kmeans: faiss.Kmeans | None = None

    def fit(self, data: np.ndarray, k: int | None = None, elbow: float = 0.975) -> np.ndarray:
        sum_of_squared_dists: list[float] = []
        n_cluster_range = [k] if k else list(range(1, 15))

        algorithm: faiss.Kmeans | None = None
        labels: np.ndarray | None = None
        for n_clusters in n_cluster_range:
            algorithm = faiss.Kmeans(d=data.shape[-1], k=n_clusters, niter=300, nredo=10)
            algorithm.train(data.astype(np.float32))
            squared_distances, labels = algorithm.index.search(data.astype(np.float32), 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / data.shape[0])
            if len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]:
                break
        self.kmeans = algorithm
        assert labels is not None
        return labels

    def predict(self, data: np.ndarray) -> np.ndarray:
        assert self.kmeans is not None
        _, labels = self.kmeans.index.search(data.astype(np.float32), 1)
        return labels

    def save(self, filepath: str | Path) -> None:
        assert self.kmeans is not None
        np.save(str(filepath) + ".npy", self.kmeans.centroids)

    def load(self, filepath: str | Path) -> None:
        centroids = np.load(str(filepath) + ".npy")
        self.kmeans = faiss.Kmeans(d=centroids.shape[-1], k=centroids.shape[0], niter=1, nredo=10)
        self.kmeans.train(centroids.astype(np.float32))
