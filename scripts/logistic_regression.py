"""Performing hyperparameter tuning for logistic regression."""
import numpy as np

from scripts.proj1_helpers import load_csv_data
from scripts.data_preprocessing import build_k_indices
from scripts.implementations import logistic_regression, cross_validation

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')

seed = 1
degrees = np.arange(7, 14, 3)
gammas = [0.15, 0.2, 0.25, 0.4]
# selected number of k folds and split data
k_fold = 5
k_indices = build_k_indices(y_tr, k_fold, seed)

nan_mode = 'median'
binary_mode = 'one_hot'
split_mode = 'jet_groups'
max_iters = 1000

accuracy_ranking_tr = np.zeros((len(gammas), len(degrees)))
accuracy_ranking_te = np.zeros((len(gammas), len(degrees)))

count = 0
for h, gamma in enumerate(gammas):
    for i, degree in enumerate(degrees):
        count += 1
        temp_acc_tr = []
        temp_acc_te = []
        for k in range(k_fold):
            acc_tr, acc_te, loss_tr, loss_te = cross_validation(y_tr, x_tr, logistic_regression, k_indices, k, degree,
                                                                binary_mode=binary_mode, split_mode=split_mode,
                                                                nan_mode=nan_mode, max_iters=max_iters, gamma=gamma,
                                                                w0=None)

            temp_acc_tr.append(acc_tr)
            temp_acc_te.append(acc_te)

        print(f'#: {count} / {len(gammas) * len(degrees)}, degree = {degree}, gamma = {gamma},'
              f' accuracy = {np.mean(temp_acc_te)}')

        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking_tr[h, i] = np.mean(temp_acc_tr)
        accuracy_ranking_te[h, i] = np.mean(temp_acc_te)
