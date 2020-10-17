import numpy as np
from scripts.proj1_helpers import *

y, x, ids = load_csv_data("data/train.csv")

def lin_dep(x):
    # define the matrix containing the inner products of the columns
    inn_prod = x.T @ x
    # define the matrix containing the products of the norms of the columns
    arr_norm = np.linalg.norm(x, axis=0)[..., np.newaxis]
    norm_prod = arr_norm @ arr_norm.T
    # define the difference matrix
    diff = inn_prod - norm_prod
    # define indices where the difference is = 0
    # the indices represents the linearly dependent columns
    ind_dep = []
    for i in range(diff.shape[0]):
        for j in range(diff.shape[0]):
            if np.abs(diff[i,j]) < 1E-1:
                id = np.array([i, j])
                ind_dep.append(id)
    return ind_dep

ind_dep = lin_dep(x)
print(lin_dep(x))
