import numpy as np

from scripts.utilities import compute_accuracy
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.implementations import least_squares, least_squares_GD, least_squares_SGD, cross_validation
from scripts.data_preprocessing import build_k_indices


y_tr, x_tr, ids_tr = load_csv_data("data/train.csv" )


seed = 1
k_fold = 5
k_indices = build_k_indices(y_tr, k_fold, seed)

# method mode: can be ls, ls_gd, and ls_sgd
mode = 'ls'
degrees = np.arange(2, 13)
gammas = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
nan_mode = 'median'
split_mode = 'jet_groups' # 'default', entire dataset, or 'jet_groups'

assert mode == 'ls' or mode == 'ls_SGD' or mode == 'ls_GD', "Please enter a valid mode ('ls_GD', 'ls_SGD', 'ls')"

count = 0

if mode != 'ls':
    accuracy_ranking_te = np.zeros((len(gammas), len(degrees)))
    accuracy_ranking_tr = np.zeros((len(gammas), len(degrees)))
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            count += 1
            temp_acc_te = []
            temp_acc_tr = []
            for k in range(k_fold):
                if mode == 'ls_GD':
                    acc_tr, acc_te = cross_validation(y_tr, x_tr, least_squares_GD, k_indices, k, degree,
                                                      split_mode=split_mode, binary_mode='default', gamma=gamma,
                                                      w0=None, max_iters=100, nan_mode=nan_mode)
                elif mode == 'ls_SGD':
                    acc_tr, acc_te = cross_validation(y_tr, x_tr, least_squares_SGD, k_indices, k, degree,
                                                      split_mode=split_mode, binary_mode='default', gamma=gamma,
                                                      w0=None, max_iters=100, nan_mode=nan_mode)

                temp_acc_te.append(acc_te)
                temp_acc_tr.append(acc_tr)
            print(f'#: {count}/{len(gammas) * len(degrees)}, gamma: {gamma}, degree: {degree}, '
                  f'mean test accuracy = {np.mean(temp_acc_te)}, mean train accuracy = {np.mean(temp_acc_tr)}')
            # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking_te[h, i] = np.mean(temp_acc_te)
            accuracy_ranking_tr[h, i] = np.mean(temp_acc_tr)

elif mode == 'ls':
    # define lists to store the loss of training data and test data
    accuracy_ranking = np.zeros(len(degrees))
    # cross validation
    for i, degree in enumerate(degrees):
        count += 1
        temp_acc_te = []
        temp_acc_tr = []
        for k in range(k_fold):
            acc_tr, acc_te = cross_validation(y_tr, x_tr, least_squares, k_indices, k, degree, split_mode= split_mode,
                                              binary_mode='default', nan_mode=nan_mode)

            temp_acc_te.append(acc_te)
            temp_acc_tr.append(acc_tr)
        print(f'#: {count} / {len(degrees)}, degree: {degree}, mean test accuracy = {np.mean(temp_acc_te)},'
              f' mean train accuracy = {np.mean(temp_acc_tr)}')
        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking[i] = np.mean(temp_acc_te)


# max_ind = np.unravel_index(np.argmax(accuracy_ranking), accuracy_ranking.shape)
#
# gamma_ind = max_ind[0]
# gamma = gammas[gamma_ind]
#
# degree_ind = max_ind[1]
# degree = degrees[degree_ind]

# tx_tr_tot = build_poly(x_tr, 7)
# mse, w = least_squares(y_tr, tx_tr_tot)
#
# y_te, x_te, ids_te = load_csv_data("data/test.csv")
#
# x_te = standardize_data(x_te)
# tx_te_tot = build_poly(x_te, 7)
#
# y_te_pred = predict_labels(w, tx_te_tot)
#
# create_csv_submission(ids_te, y_te_pred, 'submission.csv')
