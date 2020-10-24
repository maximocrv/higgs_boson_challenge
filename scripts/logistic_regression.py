import numpy as np

from scripts.proj1_helpers import load_csv_data
from scripts.data_preprocessing import build_k_indices
from scripts.implementations import logistic_regression_GD, logistic_regression_SGD, cross_validation
from scripts.utilities import obtain_best_params

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')

seed = 1
degrees = np.arange(4, 11, 3)
k_fold = 10
gammas = [1e-3, 1e-2, 1e-1]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

# set mode to either lr, lr_sgd or regularized_lr
mode = 'lr_GD'
assert mode == 'lr_GD' or mode == 'lr_SGD', "Please enter a valid mode (lr_GD, lr_SGD)"


nan_mode = 'median'
binary_mode = 'one_hot'
split_mode = 'jet_groups'
max_iters = 400

accuracy_ranking_tr = np.zeros((len(gammas), len(degrees)))
accuracy_ranking_te = np.zeros((len(gammas), len(degrees)))
accuracy_ranking_conf_interval = np.zeros((len(gammas), len(degrees)))

count = 0
for h, gamma in enumerate(gammas):
    for i, degree in enumerate(degrees):
        count += 1
        temp_acc_tr = []
        temp_acc_te = []
        for k in range(k_fold):
            if mode == 'lr_GD':
                acc_tr, acc_te = cross_validation(y_tr, x_tr, logistic_regression_GD, k_indices, k, degree,
                                                  binary_mode=binary_mode, split_mode=split_mode, nan_mode=nan_mode,
                                                  max_iters=max_iters, gamma=gamma, w0=None)
            elif mode == 'lr_SGD':
                acc_tr, acc_te = cross_validation(y_tr, x_tr, logistic_regression_SGD, k_indices, k, degree,
                                                  binary_mode=binary_mode, split_mode=split_mode, nan_mode=nan_mode,
                                                  max_iters=max_iters, gamma=gamma, w0=None)

            temp_acc_tr.append(acc_tr)
            temp_acc_te.append(acc_te)

        print(f'#: {count} / {len(gammas) * len(degrees)}, degree = {degree}, gamma = {gamma},'
              f' accuracy = {np.mean(temp_acc_te)}')

        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking_tr[h, i] = np.mean(temp_acc_tr)
        accuracy_ranking_te[h, i] = np.mean(temp_acc_te)
        accuracy_ranking_conf_interval[h, i] = np.mean(temp_acc_te) - 2 * np.std(temp_acc_te)

best_params = obtain_best_params(accuracy_ranking_te, gammas, degrees, None)
