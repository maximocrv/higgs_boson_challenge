import numpy as np

from scripts.utilities import compute_accuracy
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.data_preprocessing import standardize_data, build_k_indices, generate_batch, balance_fromnans, \
    build_poly, convert_nan
from scripts.implementations import logistic_regression_GD, reg_logistic_regression, least_squares_SGD, \
    reg_logistic_regression_GD

# streamlining hyperparameter tuning and testing across all optimization methods!!!!!!
# logistic regression sgd is best example of how to perform preprocessing etc
# standardizing continuous variables and leaving categorical variables be
# test feature elimination based on unprocessed highcorr features and nan to mean highcorr features
# test mean and median nan mode w logistic regression
# implement accuracy metric using distribution across folds (i.e. max(mean(acc) - 2*sd(acc)))
# PCA
# test all the above with ridge regression

nan_mode = 'median'
y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')
# balance dataset
y_tr, x_tr = balance_fromnans(y_tr, x_tr)


# Choice of variables to cut based on covariance and histograms
cut_features = np.array([9, 29, 3, 4])
cut_features2 = np.array([15, 18, 20])
cut_features3 = np.array([4, 5, 6, 12, 26, 27, 28])

# unprocessed highly correlated features
features = [5, 6, 12, 21, 22, 24, 25, 26, 27, 28, 29]
# nan to mean highly correlated features
# features = [2, 6, 7, 9, 11, 12, 16, 17, 19, 21, 22, 23, 29]
# highly correlated features no nans
# features = INSERT INDICES

# x_tr = np.delete(x_tr, cut_features, axis=1)
x_tr = np.delete(x_tr, features, axis=1)

# STANDARDIZE DATA AFTER GENERATING FEATURE EXPANSION VECTOR
x_tr = standardize_data(x_tr, nan_mode=nan_mode)

seed = 1
degrees = np.arange(2, 8)
k_fold = 5
gammas = [1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

# set mode to either lr, lr_sgd or regularized_lr
mode = 'lr_sgd'
# mode = 'submission'

if mode == 'lr':
    accuracy_ranking = np.zeros((len(gammas), len(degrees)))
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            temp_acc = []
            for k in range(k_fold):
                _x_tr, _y_tr, _x_te, _y_te = generate_batch(y_tr, x_tr, k_indices, k)

                tx_tr = build_poly(_x_tr, degree)
                tx_te = build_poly(_x_te, degree)

                # w0 = np.random.randn(tx_tr.shape[1])
                w0 = np.random.randn(tx_tr.shape[1])

                # ridge regression
                nll, w_tr = logistic_regression_GD(_y_tr, tx_tr, w0, max_iters=30, gamma=gamma)

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

                tx_tr = build_poly(_x_tr, degree)
                tx_te = build_poly(_x_te, degree)

                tx_tr = standardize_data(tx_tr[:, 1:])
                tx_tr = np.concatenate((np.ones((tx_tr.shape[0], 1)), tx_tr), axis=1)

                tx_te = standardize_data(tx_te[:, 1:])
                tx_te = np.concatenate((np.ones((tx_te.shape[0], 1)), tx_te), axis=1)

                # w0 = np.random.randn(tx_tr.shape[1])
                w0 = np.random.randn(tx_tr.shape[1])

                # ridge regression
                nll_tr, w_tr = least_squares_SGD(_y_tr, tx_tr, w0, max_iters=5000, gamma=gamma, batch_size=1,
                                                 mode='logistic_reg')

                w = w_tr[-1]

                acc = compute_accuracy(w, tx_te, _y_te, mode='one_hot')

                temp_acc.append(acc)
            print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
            # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i] = np.mean(temp_acc)

elif mode == 'regularized_lr':
    accuracy_ranking = np.zeros((len(gammas), len(degrees), len(lambdas)))
    for h, gamma in enumerate(gammas):
        for i, degree in enumerate(degrees):
            for j, lambda_ in enumerate(lambdas):
                temp_acc = []
                for k in range(k_fold):
                    _x_tr, _y_tr, _x_te, _y_te = generate_batch(y_tr, x_tr, k_indices, k)

                    tx_tr = build_poly(_x_tr, degree)
                    tx_te = build_poly(_x_te, degree)

                    tx_tr = standardize_data(tx_tr[:, 1:])
                    tx_tr = np.concatenate((np.ones((tx_tr.shape[0], 1)), tx_tr), axis=1)

                    tx_te = standardize_data(tx_te[:, 1:])
                    tx_te = np.concatenate((np.ones((tx_te.shape[0], 1)), tx_te), axis=1)

                    # w0 = np.random.randn(tx_tr.shape[1])
                    w0 = np.random.randn(tx_tr.shape[1])

                    # ridge regression
                    nll, w_tr = reg_logistic_regression_GD(_y_tr, tx_tr, w0, 5, gamma, lambda_)

                    acc = compute_accuracy(w_tr, tx_te, _y_te)

                    temp_acc.append(acc)
                print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
                # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
                accuracy_ranking[h, i, j] = np.mean(temp_acc)
elif mode == 'submission':
    print('gl hf')

max_ind = np.unravel_index(np.argmax(accuracy_ranking), accuracy_ranking.shape)

gamma_ind = max_ind[0]
gamma = gammas[gamma_ind]

degree_ind = max_ind[1]
degree = degrees[degree_ind]

if len(max_ind) > 2:
    lambda_ind = max_ind[2]
    lambda_ = lambdas[lambda_ind]
# # x_tr and y_tr already balanced from nans and standardized
# tx_tr_tot = multi_build_poly(x_tr, degree)
# # standardize again after polynomial basis expansion
# tx_tr_tot = standardize_data(tx_tr_tot[:, 1:])
# tx_tr_tot = np.concatenate((np.ones((tx_tr_tot.shape[0], 1)), tx_tr_tot), axis=1)
#
# w0 = np.random.randn(tx_tr_tot.shape[1])
# nll, w = stochastic_gradient_descent(y_tr, tx_tr_tot, w0, max_iters=50000, gamma=gamma, batch_size=1,
#                                      mode='logistic_reg')
# w = w[-1]
#
#
# y_te, x_te, ids_te = load_csv_data("data/test.csv", mode='one_hot')
# x_te = np.delete(x_te, features, axis=1)
# # nan to mean or median
# x_te = standardize_data(x_te, nan_mode=nan_mode)
#
# tx_te_tot = multi_build_poly(x_te, degree)
# tx_te_tot = standardize_data(tx_te_tot[:, 1:], nan_mode=nan_mode)
# tx_te_tot = np.concatenate((np.ones((tx_te_tot.shape[0], 1)), tx_te_tot), axis=1)
#
# y_te_pred = predict_labels(w, tx_te_tot, mode='one_hot')
# y_te_pred[y_te_pred == 0] = -1
#
# create_csv_submission(ids_te, y_te_pred, 'log_reg_sgd_submission.csv')
