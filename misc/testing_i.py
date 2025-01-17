"""Performing hyperparameter tuning using ridge regression."""
import numpy as np

from scripts.proj1_helpers import load_csv_data
from misc.implementations_misc import ridge_regression, cross_validation
from scripts.data_preprocessing import build_k_indices

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv")

seed = 1
degrees = np.arange(6, 16)
# degrees = np.array([5])
# lambdas = np.logspace(-6, -1, 6)
# lambdas = np.array([1e-4, 1e-3, 1e-2, 1e-1])
lambdas = np.array([1e-5, 1e-3, 1e-1])
# lambdas = np.array([1e-3])

k_fold = 5
k_indices = build_k_indices(y_tr, k_fold, seed)

nan_mode = 'mode'
binary_mode = 'default'
split_mode = 'default'

test_acc_sd = np.zeros((len(lambdas), len(degrees)))
accuracy_ranking_conf_interval = np.zeros((len(lambdas), len(degrees)))
accuracy_ranking_tr = np.zeros((len(lambdas), len(degrees)))
accuracy_ranking_te = np.zeros((len(lambdas), len(degrees)))

train_loss_rank = np.zeros((len(lambdas), len(degrees)))
test_loss_rank = np.zeros((len(lambdas), len(degrees)))
test_loss_l2_rank = np.zeros((len(lambdas), len(degrees)))
test_loss_sd_rank = np.zeros((len(lambdas), len(degrees)))

count = 0

for h, lambda_ in enumerate(lambdas):
    for i, degree in enumerate(degrees):
        count += 1
        temp_acc_te = []
        temp_acc_tr = []
        temp_loss_te = []
        temp_loss_l2_te = []
        temp_loss_tr = []
        for k in range(k_fold):
            # acc_tr, acc_te, loss_te, loss_te_l2, loss_tr = cross_validation(y_tr, x_tr, ridge_regression, k_indices, k, degree,
            #                                                        lambda_=lambda_, split_mode=split_mode,
            #                                                        binary_mode=binary_mode, nan_mode=nan_mode)
            acc_tr, acc_te, loss_tr, loss_te = cross_validation(y_tr, x_tr, ridge_regression, k_indices, k, degree,
                                                                   lambda_=lambda_, split_mode=split_mode,
                                                                   binary_mode=binary_mode, nan_mode=nan_mode)
            # RMSE NOW!!!
            temp_acc_te.append(acc_te)
            temp_acc_tr.append(acc_tr)
            # temp_loss_te.append(loss_te)
            # temp_loss_l2_te.append(loss_te_l2)
            temp_loss_tr.append(loss_tr)
            temp_loss_te.append(loss_te)
        print(f'#: {count} / {len(degrees) * len(lambdas)}, lambda: {lambda_}, degree: {degree}, '
              f'accuracy_tr = {np.mean(temp_acc_tr)}, accuracy_te = {np.mean(temp_acc_te)}')

        # test_acc_sd[h, i] = np.std(temp_acc_te)
        # accuracy_ranking_conf_interval[h, i] = np.mean(temp_acc_te) - 2 * np.std(temp_acc_te)
        # accuracy_ranking_tr[h, i] = np.mean(temp_acc_tr)
        # accuracy_ranking_te[h, i] = np.mean(temp_acc_te)

        train_loss_rank[h, i] = np.mean(temp_loss_tr)
        test_loss_rank[h, i] = np.mean(temp_loss_te)
        # test_loss_l2_rank[h, i] = np.mean(temp_loss_l2_te)
        # test_loss_sd_rank[h, i] = np.std(temp_loss_te)
