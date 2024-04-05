# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:06:53 2024

@author: Mico
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from faiss_knn import FaissKNeighbors, FaissKMeans

features_dir = os.path.join('..', 'data', 'features')
X = np.load(os.path.join(features_dir, 'features.npy'))
y = np.load(os.path.join(features_dir, 'labels_cls.npy'))

kmeans = FaissKMeans()
clusters = kmeans.fit(X, k=16)


kf = StratifiedKFold(n_splits=5)

label_names = ['high', 'low', 'moderate', 'non-burnable', 'very_high', 'very_low', 'water']
label_ids = np.arange(0, len(label_names)) + 1

models_objs = {}
df_metrics = []

for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

    model = FaissKNeighbors(k=11)
    models_objs[i] = model

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_report_str, test_report = [classification_report(y_test, y_pred,
                                                          target_names=label_names,
                                                          output_dict=odict) for odict in [False, True]]

    cm = confusion_matrix(y_test, y_pred, labels=label_ids, normalize='true')
    acc_per_class = np.diagonal(cm)
    df_report = pd.DataFrame(test_report)
    df_report = df_report[label_names]
    df_report = df_report.T
    df_report['accuracy'] = acc_per_class
    df_report['fold'] = i
    df_metrics.append(df_report.reset_index(drop=False))
    print(test_report_str)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                            normalize='true',
                                            display_labels=label_names,
                                            cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix Kfold {i}')
    plt.show()

df_metrics = pd.concat(df_metrics, ignore_index=True)
model_avg_metrics = df_metrics.groupby(['index']).mean()
model_std_metrics = df_metrics.groupby(['index']).std()

metrics_cols = ['precision', 'recall', 'f1-score', 'accuracy']
combined_metrics = model_avg_metrics[metrics_cols + ['support']].copy()
for column in metrics_cols:
    avg = model_avg_metrics[column]
    std = model_std_metrics[column]
    combined_metrics[column] = avg.map('{:,.2f}'.format) + " ± " + std.map('{:,.2f}'.format)


support = np.tile(model_avg_metrics['support'].values.reshape(-1, 1), (1, len(metrics_cols)))
macro_avg = model_avg_metrics[metrics_cols].values.mean(axis=0)

total_support = support.sum(axis=0).mean()
macro_avg = list(macro_avg)
macro_avg.append(total_support)

weighted_avg = model_avg_metrics[metrics_cols].values
weighted_avg = weighted_avg * support
weighted_avg = weighted_avg.sum(axis=0) / support.sum(axis=0).reshape(1,-1) 
weighted_avg = list(np.squeeze(weighted_avg))
weighted_avg.append(total_support)


combined_metrics = combined_metrics.T

combined_metrics['macro avg'] = [f'{np.round(v, 3)}' for v in macro_avg]
combined_metrics['weighted avg'] = [f'{np.round(v, 3)}' for v in weighted_avg]

combined_metrics = combined_metrics.T

best_fold = df_metrics.groupby('fold')['f1-score'].mean().argmax()

best_model = models_objs[best_fold]
model_name = 'dinov2_knn_firerisk'
kmeans_name = 'dinov2_kmeans_firerisk' 
models_dir = os.path.join('..', 'models')
combined_metrics.to_csv(os.path.join(models_dir, f'{model_name}_metrics.csv'),
                        encoding='utf-8-sig')
best_model.save(os.path.join(models_dir, model_name))
kmeans.save(os.path.join(models_dir, kmeans_name))


print(combined_metrics)