# -*- coding: utf-8 -*-
"""Function used to perform ridge regression."""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    a = tx.T @ tx + 2 * lambda_ * tx.shape[0] * np.eye(tx.shape[1])
    b = tx.T @ y
    
    w = np.linalg.solve(a, b)
    
    mse = compute_mse(y, tx, w)
    rmse = compute_rmse(y, tx, w)
    
    return mse, w