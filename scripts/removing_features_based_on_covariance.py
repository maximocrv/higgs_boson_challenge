import numpy as np
from scripts.proj1_helpers import load_csv_data
from scripts.data_exploration import covariance_matrix, set_nan

def set_cov_inf(cov):
    cov_ = cov
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if i >= j:
                cov_[i, j] = 0
    return cov_

def corr_col(cov, t):
    # initialize an array filled with the columns to be eliminated
    # t is the threshold correlation
    cde = []
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if abs(cov[i, j]) > t :
                v = np.array([i, j])
                cde.append(v)
    return cde


if __name__ == '__main__':
    y, x, ids = load_csv_data("data/train.csv")
    x = set_nan(x)
    cov = covariance_matrix(x)
    cov_ = set_cov_inf(cov)
    col_el = corr_col(cov_, t=0.6)
    print(col_el)
