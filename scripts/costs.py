# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

import numpy as np

from scripts.proj1_helpers import predict_labels
from scripts.data_exploration import calculate_recall_precision_accuracy, create_confusion_matrix


def compute_mse(y, tx, w):
    """
    Compute mean square error.
    """
    e = y - tx @ w
    return 1/2 * np.mean(e**2)


def compute_rmse(y, tx, w):
    """
    Comput root mean square error
    """
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)


def log_likelihood(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx @ w)) - y * tx @ w)


def compute_accuracy(w, x, y_true, mode='default'):
    y_pred = predict_labels(w, x, mode)
    true_list = y_true - y_pred
    num_true = np.where(true_list == 0)
    acc = len(num_true[0]) / y_true.shape[0]

    return acc


def compute_f1score (w, x, y_true):
    y_pred = predict_labels(w, x)
    recall, precision, accuracy = calculate_recall_precision_accuracy(create_confusion_matrix(y_pred, y_true))
    return 2*(precision*recall)/(precision+recall)


def matthews_coeff (w, x, y_true):
    y_pred = predict_labels(w, x)
    cm = create_confusion_matrix(y_pred, y_true)
    # mc = (TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    mc = (cm[1,1]*cm[0,0]-cm[1,0]*cm[0,1])/np.sqrt((cm[1,1]+cm[1,0])*(cm[1,1]+cm[0,1])*(cm[0,0]+cm[1,0])*(cm[0,0]+cm[0,1]))
    return mc

