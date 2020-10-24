import numpy as np

from scripts.proj1_helpers import load_csv_data
from scripts.data_preprocessing import build_k_indices
from scripts.implementations import logistic_regression_GD, logistic_regression_SGD, cross_validation

# streamlining hyperparameter tuning and testing across all optimization methods!!!!!!
# logistic regression sgd is best example of how to perform preprocessing etc
# standardizing continuous variables and leaving categorical variables be
# test feature elimination based on unprocessed highcorr features and nan to mean highcorr features
# test mean and median nan mode w logistic regression
# implement accuracy metric using distribution across folds (i.e. max(mean(acc) - 2*sd(acc)))
# PCA
# test all the above with ridge regression
y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')
# balance dataset
# y_tr, x_tr = balance_fromnans(y_tr, x_tr)


# Choice of variables to cut based on covariance and histograms
# cut_features = np.array([9, 29, 3, 4])
# cut_features2 = np.array([15, 18, 20])
# cut_features3 = np.array([4, 5, 6, 12, 26, 27, 28])

# unprocessed highly correlated features
# features = [5, 6, 12, 21, 22, 24, 25, 26, 27, 28, 29]
# nan to mean highly correlated features
# features = [2, 6, 7, 9, 11, 12, 16, 17, 19, 21, 22, 23, 29]
# highly correlated features no nans
# features = INSERT INDICES

# x_tr = np.delete(x_tr, cut_features, axis=1)
# x_tr = np.delete(x_tr, features, axis=1)

# STANDARDIZE DATA AFTER GENERATING FEATURE EXPANSION VECTOR
# x_tr = standardize_data(x_tr, nan_mode=nan_mode)

seed = 1
degrees = np.arange(2, 8)
k_fold = 5
gammas = [1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

# set mode to either lr, lr_sgd or regularized_lr
mode = 'lr_SGD'
assert mode == 'lr_GD' or mode == 'lr_SGD', "Please enter a valid mode (lr_GD, lr_SGD)"
nan_mode = 'median'
binary_mode = 'one_hot'
split_mode = 'default'
max_iters = 100

count = 0
accuracy_ranking = np.zeros((len(gammas), len(degrees)))
for h, gamma in enumerate(gammas):
    for i, degree in enumerate(degrees):
        count += 1
        temp_acc = []
        for k in range(k_fold):
            if mode == 'lr_GD':
                acc_tr, acc_te = cross_validation(y_tr, x_tr, logistic_regression_GD, k_indices, k, degree,
                                                  binary_mode=binary_mode, split_mode=split_mode, nan_mode=nan_mode,
                                                  max_iters=max_iters, gamma=gamma, w0=None)

            elif mode == 'lr_SGD':
                acc_tr, acc_te = cross_validation(y_tr, x_tr, logistic_regression_SGD, k_indices, k, degree,
                                                  binary_mode=binary_mode, split_mode=split_mode, nan_mode=nan_mode,
                                                  max_iters=max_iters, gamma=gamma, w0=None)

            temp_acc.append(acc_tr)
        print(f'#: {count} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking[h, i] = np.mean(temp_acc)

max_ind = np.unravel_index(np.argmax(accuracy_ranking), accuracy_ranking.shape)

# gamma_ind = max_ind[0]
# gamma = gammas[gamma_ind]
#
# degree_ind = max_ind[1]
# degree = degrees[degree_ind]
#
# if len(max_ind) > 2:
#     lambda_ind = max_ind[2]
#     lambda_ = lambdas[lambda_ind]
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
