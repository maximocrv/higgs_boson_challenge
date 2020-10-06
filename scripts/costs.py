# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_mse(y, tx, w):
    e = y - tx @ w
    return 1/2 * np.mean(e**2)

def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)