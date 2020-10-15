# -*- coding: utf-8 -*-
"""Functions used to compute the loss."""

import numpy as np


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