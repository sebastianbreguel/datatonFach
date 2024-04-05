# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:33:40 2024

@author: Mico
"""

import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import geo_utils as geo
from predict_knn_dinov2 import reduce_output_clases, create_labelmap
from create_train_dataset import risk_cls_from_boxes

if __name__ == "__main__":
    bbox_path = os.path.join('..', 'data', 'valpo', 'bbox_labels.csv')
    pred_raster_path = 'prediction_1712318539_986473.tif'

    label_names, label_ids = create_labelmap()
    y_pred_img, profile = geo.load_raster(pred_raster_path)
    y_pred_img = geo.reverse_axis(y_pred_img)

    risk_cls_resized = risk_cls_from_boxes(bbox_path, label_names)
    mask = risk_cls_resized > 0

    y_true = risk_cls_resized[mask]
    y_pred = y_pred_img[mask]

    y_true = reduce_output_clases(y_true)
    y_pred = reduce_output_clases(y_pred)

    label_names_with_bg = ['background'] + label_names
    y_true = [label_names_with_bg[int(idx)] for idx in y_true]
    y_pred = [label_names_with_bg[int(idx)] for idx in y_pred]

    test_report_str, test_report = [classification_report(y_true, y_pred,
                                                          output_dict=odict) for odict in [False, True]]
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize='true',
                                            cmap=plt.cm.Blues)

    print(test_report_str)
