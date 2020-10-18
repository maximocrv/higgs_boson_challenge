import numpy as np

from scripts.helpers import batch_iter
from scripts.proj1_helpers import *
from scripts.data_preprocessing import *
from scripts.model_batching import build_k_indices, generate_batch
from scripts.build_polynomial import multi_build_poly
from scripts.implementations import compute_accuracy, log_reg_gd, penalized_logistic_regression, \
    stochastic_gradient_descent

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')

x_tr = standardize_data(x_tr)

# x_tr = nan_to_mean(x_tr)

seed = 1
degrees = np.arange(3, 8)
k_fold = 10
gammas = [1e-5, 1e-3, 1e-2, 1e-1]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)
# define lists to store the loss of training data and test data
accuracy_ranking = np.zeros((len(gammas), len(degrees)))

# set mode to either lr or regularized_lr
mode = 'lr_sgd'

if mode == 'lr':
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            temp_acc = []
            for k in range(k_fold):
                _x_tr, _y_tr, _x_te, _y_te = generate_batch(y_tr, x_tr, k_indices, k)

                tx_tr = multi_build_poly(_x_tr, degree)
                tx_te = multi_build_poly(_x_te, degree)

                # w0 = np.random.randn(tx_tr.shape[1])
                w0 = np.random.randn(tx_tr.shape[1])

                # ridge regression
                nll, w_tr = log_reg_gd(_y_tr, tx_tr, w0, max_iters=30, gamma=gamma)

                acc = compute_accuracy(w_tr, tx_te, _y_te)

                temp_acc.append(acc)
            print(f'#: {h*len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
            #accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i] = np.mean(temp_acc)

elif mode == 'lr_sgd':
    # define lists to store the loss of training data and test data
    accuracy_ranking = np.zeros((len(gammas), len(degrees)))
    # cross validation
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            temp_acc = []
            for k in range(k_fold):
                _x_tr, _y_tr, _x_te, _y_te = generate_batch(y_tr, x_tr, k_indices, k)

                tx_tr = multi_build_poly(_x_tr, degree)
                tx_te = multi_build_poly(_x_te, degree)

                # w0 = np.random.randn(tx_tr.shape[1])
                w0 = np.random.randn(tx_tr.shape[1])

                # ridge regression
                mse_tr, w_tr = stochastic_gradient_descent(_y_tr, tx_tr, w0, max_iters=80, gamma=gamma, batch_size=1,
                                                           mode='logistic_reg')

                w = w_tr[-1]

                acc = compute_accuracy(w, tx_te, _y_te)

                temp_acc.append(acc)
            print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
            # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i] = np.mean(temp_acc)

elif mode == 'regularized_lr':
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            for j, lambda_ in enumerate(lambdas):
                temp_acc = []
                for k in range(k_fold):
                    _x_tr, _y_tr, _x_te, _y_te = generate_batch(y_tr, x_tr, k_indices, k)

                    tx_tr = multi_build_poly(_x_tr, degree)
                    tx_te = multi_build_poly(_x_te, degree)

                    # w0 = np.random.randn(tx_tr.shape[1])
                    w0 = np.random.randn(tx_tr.shape[1])

                    # ridge regression
                    nll, w_tr = penalized_logistic_regression(_y_tr, tx_tr, w0, gamma, lambda_)

                    acc = compute_accuracy(w_tr, tx_te, _y_te)

                    temp_acc.append(acc)
                print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
                # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
                accuracy_ranking[h, i] = np.mean(temp_acc)

# gamma = 1e-5
# degree = 5
# tx_tr_tot = multi_build_poly(x_tr, degree)
# w0 = np.random.randn(tx_tr_tot.shape[1])
# nll, w = stochastic_gradient_descent(y_tr, tx_tr_tot, w0, max_iters=80, gamma=gamma, batch_size=1,
#                                      mode='logistic_reg')
#
# y_te, x_te, ids_te = load_csv_data("data/test.csv", mode='one_hot')
#
# x_te = standardize_data(x_te)
# tx_te_tot = multi_build_poly(x_te, degree)
#
# y_te_pred = predict_labels(w, tx_te_tot, mode='one_hot')
# y_te_pred[y_te_pred == 0] = -1
#
# create_csv_submission(ids_te, y_te_pred, 'submission.csv')