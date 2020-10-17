import numpy as np


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def generate_batch(y, x, k_indices, k):
    """return the loss of ridge regression."""
    # indices calculation
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_indices = tr_indices.reshape(-1)
    # dividing x and y in training and testing set
    x_te = x[te_indices]
    x_tr = x[tr_indices]
    y_te = y[te_indices]
    y_tr = y[tr_indices]
    return x_tr, y_tr, x_te, y_te
