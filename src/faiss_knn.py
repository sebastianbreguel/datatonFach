# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:02:34 2024

@author: Mico
"""

import faiss
import pickle
import numpy as np

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = np.squeeze(y)
    
    def update(self, X_new, y_new):
        X_new = X_new.astype(np.float32)
        self.index.add(X_new)
        self.y = np.concatenate([self.y, y_new])

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

    def predict_proba(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        n_classes = len(set(self.y))

        votes = self.y[indices]
        vote_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes), axis=1, arr=votes)

        probas = vote_counts / self.k
        return probas

    def save(self, filepath):
        faiss.write_index(self.index, filepath + '_index.faiss')
        with open(filepath + '_data.pkl', 'wb') as f:
            pickle.dump((self.y, self.k), f)

    def load(self, filepath):
        self.index = faiss.read_index(filepath + '_index.faiss')
        with open(filepath + '_data.pkl', 'rb') as f:
            self.y, self.k = pickle.load(f)

class FaissKMeans:
    def __init__(self):
        self.kmeans = None

    def fit(self, data, k=None, elbow=0.975):
        sum_of_squared_dists = []
        if k:
            n_cluster_range = [k]
        else:
            n_cluster_range = list(range(1, 15))
    
        for n_clusters in n_cluster_range:
            algorithm = faiss.Kmeans(d=data.shape[-1], k=n_clusters, niter=300, nredo=10)
            algorithm.train(data.astype(np.float32))
            squared_distances, labels = algorithm.index.search(data.astype(np.float32), 1)
            objective = squared_distances.sum()
            sum_of_squared_dists.append(objective / data.shape[0])
            if (len(sum_of_squared_dists) > 1 and sum_of_squared_dists[-1] > elbow * sum_of_squared_dists[-2]):
                break
        self.kmeans = algorithm
        return labels

    def predict(self, data):
        _, labels = self.kmeans.index.search(data.astype(np.float32), 1)
        return labels

    def save(self, filepath):
        np.save(filepath + '.npy', self.kmeans.centroids)

    def load(self, filepath):
        centroids = np.load(filepath + '.npy')
        self.kmeans = faiss.Kmeans(d=centroids.shape[-1], k=centroids.shape[0], niter=1, nredo=10)
        self.kmeans.train(centroids.astype(np.float32))