import numpy as np

from scripts.utilities import compute_accuracy
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.data_preprocessing import standardize_data, build_k_indices, generate_batch, balance_fromnans, \
    build_poly, convert_nan
from scripts.implementations import logistic_regression_GD, penalized_logistic_regression, least_squares_SGD, \
    reg_logistic_regression_GD, cross_validation, reg_logistic_regression_SGD

# logistic regression sgd is best example of how to perform preprocessing etc
# standardizing continuous variables and leaving categorical variables be
# test feature elimination based on unprocessed highcorr features and nan to mean highcorr features
# test mean and median nan mode w logistic regression
# implement accuracy metric using distribution across folds (i.e. max(mean(acc) - 2*sd(acc)))
# PCA
# test all the above with ridge regression
y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", sub_sample=True, mode='one_hot')
# balance dataset
# y_tr, x_tr = balance_fromnans(y_tr, x_tr)


# Choice of variables to cut based on covariance and histograms
# cut_features = np.array([9, 29, 3, 4])
# cut_features2 = np.array([15, 18, 20])
# cut_features3 = np.array([4, 5, 6, 12, 26, 27, 28])
#
# # unprocessed highly correlated features
# features = [5, 6, 12, 21, 22, 24, 25, 26, 27, 28, 29]
# # nan to mean highly correlated features
# # features = [2, 6, 7, 9, 11, 12, 16, 17, 19, 21, 22, 23, 29]
# # highly correlated features no nans
# # features = INSERT INDICES
#
# # x_tr = np.delete(x_tr, cut_features, axis=1)
# feat = [21, 29]
# x_tr = np.delete(x_tr, features, axis=1)

# STANDARDIZE DATA AFTER GENERATING FEATURE EXPANSION VECTOR
# x_tr = standardize_data(x_tr, nan_mode=nan_mode)

seed = 1
degrees = np.arange(6, 8)
k_fold = 5
gammas = [1e-4, 1e-3, 1e-2, 2e-2]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

# set mode to either lr, lr_sgd or regularized_lr
nan_mode = 'mode'
mode = 'reg_lr_GD'
assert mode == 'reg_lr_GD' or mode == 'reg_lr_SGD', "Please enter a valid mode (reg_lr_GD, reg_lr_SGD)"
# mode = 'submission'

accuracy_ranking = np.zeros((len(gammas), len(degrees), len(lambdas)))

count = 0
for h, gamma in enumerate(gammas):
    for i, degree in enumerate(degrees):
        for j, lambda_ in enumerate(lambdas):
            count += 1
            temp_acc = []
            for k in range(k_fold):
                if mode == 'reg_lr_GD':
                    acc_tr, acc_te = cross_validation(y_tr, x_tr, reg_logistic_regression_GD, k_indices,
                                                      k, degree, split_mode='jet_groups', binary_mode='one_hot',
                                                      max_iters=1000, gamma=gamma, lambda_=lambda_, w0=None,
                                                      nan_mode=nan_mode)
                elif mode == 'reg_lr_SGD':
                    acc_tr, acc_te = cross_validation(y_tr, x_tr, reg_logistic_regression_SGD, k_indices, k, degree,
                                                      split_mode='jet_groups', binary_mode='one_hot', max_iters=5000,
                                                      gamma=gamma, lambda_=lambda_, w0=None, nan_mode=nan_mode)

                temp_acc.append(acc_te)
            print(f'#: {count} / {len(gammas) * len(degrees) * len(lambdas)}, gamma = {gamma}, degree = {degree}, '
                  f'lambda = {lambda_}, accuracy = {np.mean(temp_acc)}')

            # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i] = np.mean(temp_acc)

# gamma = np.argmax(accuracy_ranking, axis=0)
# degree = np.argmax(accuracy_ranking, axis=1)
# gamma = 1e-3
# degree = 5
# # x_tr and y_tr already balanced from nans and standardized
# tx_tr_tot = build_poly(x_tr, degree)
# # standardize again after polynomial basis expansion
# tx_tr_tot = standardize_data(tx_tr_tot[:, 1:])
# tx_tr_tot = np.concatenate((np.ones((tx_tr_tot.shape[0], 1)), tx_tr_tot), axis=1)
#
# w0 = np.random.randn(tx_tr_tot.shape[1])
# nll, w = reg_logistic_regression_SGD(y_tr, tx_tr_tot, w0, max_iters=50000, gamma=gamma, batch_size=1)
# w = w[-1]
#
#
# y_te, x_te, ids_te = load_csv_data("data/test.csv", mode='one_hot')
# x_te = np.delete(x_te, features, axis=1)
# # nan to mean or median
# x_te = standardize_data(x_te, nan_mode=nan_mode)
#
# tx_te_tot = build_poly(x_te, degree)
# tx_te_tot = standardize_data(tx_te_tot[:, 1:], nan_mode=nan_mode)
# tx_te_tot = np.concatenate((np.ones((tx_te_tot.shape[0], 1)), tx_te_tot), axis=1)
#
# y_te_pred = predict_labels(w, tx_te_tot, binary_mode='one_hot')
# y_te_pred[y_te_pred == 0] = -1

# create_csv_submission(ids_te, y_te_pred, 'log_reg_sgd_submission.csv')
