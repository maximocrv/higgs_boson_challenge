import numpy as np

from scripts.utilities import compute_accuracy
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.implementations import least_squares, least_squares_GD, least_squares_SGD, cross_validation
from scripts.data_preprocessing import build_k_indices, generate_batch, standardize_data, build_poly


y_tr, x_tr, ids_tr = load_csv_data("data/train.csv")

seed = 1
k_fold = 5
k_indices = build_k_indices(y_tr, k_fold, seed)

degrees = np.arange(1, 4)
gammas = [0.25, 0.27, 0.29, 0.31]

# set mode. can be ls, ls_gd, and ls_sgd
mode = 'ls'
assert mode == 'ls' or mode == 'ls_SGD' or mode == 'ls_GD', "Please enter a valid mode ('ls_GD', 'ls_SGD', 'ls')"

if mode != 'ls':
    accuracy_ranking = np.zeros((len(gammas), len(degrees)))
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            temp_acc = []
            for k in range(k_fold):
                if mode == 'ls_GD':
                    acc_tr, acc_te = cross_validation(y_tr, x_tr, least_squares_GD, k_indices, k, degree, split_mode='default',
                                                      gamma=gamma)
                elif mode == 'ls_SGD':
                    acc_tr, acc_te = cross_validation(y_tr, x_tr, least_squares_SGD, k_indices, k, degree,
                                                      split_mode='default', gamma=gamma)

                temp_acc.append(acc_te)
            print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
            # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i] = np.mean(temp_acc)

elif mode == 'ls':
    # define lists to store the loss of training data and test data
    accuracy_ranking = np.zeros(len(degrees))
    # cross validation
    for i, degree in enumerate(degrees):
        temp_acc = []
        for k in range(k_fold):
            acc_tr, acc_te = cross_validation(y_tr, x_tr, least_squares, k_indices, k, degree, split_mode='default')

            temp_acc.append(acc_te)
        print(f'#: {i + 1} / {len(degrees)}, accuracy = {np.mean(temp_acc)}')
        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking[i] = np.mean(temp_acc)

max_ind = np.unravel_index(np.argmax(accuracy_ranking), accuracy_ranking.shape)

gamma_ind = max_ind[0]
gamma = gammas[gamma_ind]

degree_ind = max_ind[1]
degree = degrees[degree_ind]

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
