"""Hyperparameter tuning for regularized logistic regression."""
import numpy as np

from scripts.proj1_helpers import load_csv_data
from scripts.data_preprocessing import build_k_indices
from scripts.implementations import reg_logistic_regression, cross_validation

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", sub_sample=True, mode='one_hot')

seed = 1
degrees = np.arange(2, 6)
k_fold = 5
gammas = [1e-4, 1e-3, 1e-2, 2e-2]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

split_mode = 'jet_groups'
binary_mode = 'one_hot'
nan_mode = 'mode'

accuracy_ranking = np.zeros((len(gammas), len(degrees), len(lambdas)))

count = 0
for h, gamma in enumerate(gammas):
    for i, degree in enumerate(degrees):
        for j, lambda_ in enumerate(lambdas):
            count += 1
            temp_acc_tr = []
            temp_acc_te = []
            for k in range(k_fold):
                acc_tr, acc_te, loss_tr, loss_te = cross_validation(y_tr, x_tr, reg_logistic_regression, k_indices,
                                                                    k, degree, split_mode=split_mode,
                                                                    binary_mode=binary_mode, max_iters=200, gamma=gamma,
                                                                    lambda_=lambda_, w0=None, nan_mode=nan_mode)

                temp_acc_tr.append(acc_tr)
                temp_acc_te.append(acc_te)

            print(f'#: {count} / {len(gammas) * len(degrees) * len(lambdas)}, gamma = {gamma}, degree = {degree}, '
                  f'lambda = {lambda_}, accuracy = {np.mean(temp_acc_te)}')

            accuracy_ranking[h, i] = np.mean(temp_acc_te)
