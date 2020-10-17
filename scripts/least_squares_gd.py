import numpy as np

from scripts.helpers import batch_iter
from scripts.proj1_helpers import *
from scripts.build_polynomial import multi_build_poly
from scripts.implementations import stochastic_gradient_descent

y_tr, tx_tr, ids_tr = load_csv_data("data/train.csv")

# pre-processing (da alternare anche con gli altri metodi)
# togliamo tutte le righe dove è presente almeno un -999
min_inputs = np.min(tx_tr, axis=1)
min_input_ind = np.isin(min_inputs, -999)
tx_tr = tx_tr[~min_input_ind]
y_tr = y_tr[~min_input_ind]

# come posso scegliere la lambda da dare in input?
# posso adottare la tecnica della cross validation


def build_k_indices (y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
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
    # form data with polynomial degree
    x_tr = multi_build_poly(x_tr, degree)
    x_te = multi_build_poly(x_te, degree)
    # ridge regression
    mse_tr, w_tr = stochastic_gradient_descent(y_tr, x_tr, lambda_)
    y_pred = predict_labels(w_tr, x_te)
    true_list = y_te - y_pred
    num_true = np.where(true_list == 0)
    acc = len(num_true[0]) / y_tr.shape[0]
    return acc


def cross_validation_demo():
    seed = 1
    degrees = np.arange(3, 8)
    k_fold = 4
    lambdas = np.logspace(-5, 0, 10)
    # split data in k fold
    k_indices = build_k_indices(y_tr, k_fold, seed)
    # define lists to store the loss of training data and test data
    accuracy_ranking = np.zeros((len(lambdas), len(degrees)))
    # cross validation
    for h, lambda_ in enumerate(lambdas):
        for i, degree in enumerate(degrees):
            temp_acc = []
            for k in range(k_fold):
                acc = cross_validation(y_tr, tx_tr, k_indices, k, lambda_, degree)
                temp_acc.append(acc)
            #accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i] = np.mean(temp_acc)

    return accuracy_ranking