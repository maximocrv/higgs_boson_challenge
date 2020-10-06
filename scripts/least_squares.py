# -*- coding: utf-8 -*-
"""Function to compute least squared MSE and weights."""
import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a,b)
    mse = compute_mse(y, tx, w)
    rmse = compute_rmse(y, tx, w)
    return mse, w
