# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

import numpy as np

from scripts.proj1_helpers import predict_labels


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


def compute_accuracy(w, x, y_true):
    y_pred = predict_labels(w, x)
    true_list = y_true - y_pred
    num_true = np.where(true_list == 0)
    acc = len(num_true[0]) / y_true.shape[0]

    return acc