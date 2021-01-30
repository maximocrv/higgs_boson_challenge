import numpy as np

from scripts.implementations import compute_accuracy
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.data_preprocessing import standardize_data, build_k_indices, generate_batch, balance_fromnans, \
    multi_build_poly, convert_nan
from scripts.implementations import log_reg_gd, penalized_logistic_regression, stochastic_gradient_descent, \
    regularized_log_reg_gd

# logistic regression sgd is best example of how to perform preprocessing etc
# standardizing continuous variables and leaving categorical variables be
# test feature elimination based on unprocessed highcorr features and nan to mean highcorr features
# test mean and median nan mode w logistic regression
# PCA
# test all the above with ridge regression
nan_mode = 'median'
y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')
# balance dataset
y_tr, x_tr = balance_fromnans(y_tr, x_tr)


# Choice of variables to cut based on covariance and histograms
cut_features1 = np.array([15, 18, 20])
cut_features2 = np.array([15, 18, 20])
cut_features3 = np.array([4, 5, 6, 12, 26, 27, 28])

# unprocessed highly correlated features
#features = [5, 6, 12, 21, 22, 24, 25, 26, 27, 28, 29]
# nan to mean highly correlated features
# features = [2, 6, 7, 9, 11, 12, 16, 17, 19, 21, 22, 23, 29]
# highly correlated features no nans
# features = INSERT INDICES

x_tr = np.delete(x_tr, cut_features1, axis=1)
# x_tr = np.delete(x_tr, features, axis=1)

# STANDARDIZE DATA AFTER GENERATING FEATURE EXPANSION VECTOR
x_tr = standardize_data(x_tr, nan_mode=nan_mode)

seed = 1
degrees = np.arange(2, 6)
k_fold = 5
gammas = [1e-3, 3e-3, 7e-3, 1e-2, 5e-2 ]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

# set mode to either lr, lr_sgd or regularized_lr
mode = 'hjfjj'

if mode == 'lr':
    accuracy_ranking = np.zeros((len(gammas), len(degrees)))
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

                tx_tr = standardize_data(tx_tr[:, 1:])
                tx_tr = np.concatenate((np.ones((tx_tr.shape[0], 1)), tx_tr), axis=1)

                tx_te = standardize_data(tx_te[:, 1:])
                tx_te = np.concatenate((np.ones((tx_te.shape[0], 1)), tx_te), axis=1)

                # w0 = np.random.randn(tx_tr.shape[1])
                w0 = np.random.randn(tx_tr.shape[1])

                # ridge regression
                nll_tr, w_tr = stochastic_gradient_descent(_y_tr, tx_tr, w0, max_iters=30000, gamma=gamma, batch_size=1,
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

                    tx_tr = multi_build_poly(_x_tr, degree)
                    tx_te = multi_build_poly(_x_te, degree)

                    tx_tr = standardize_data(tx_tr[:, 1:])
                    tx_tr = np.concatenate((np.ones((tx_tr.shape[0], 1)), tx_tr), axis=1)

                    tx_te = standardize_data(tx_te[:, 1:])
                    tx_te = np.concatenate((np.ones((tx_te.shape[0], 1)), tx_te), axis=1)

                    # w0 = np.random.randn(tx_tr.shape[1])
                    w0 = np.random.randn(tx_tr.shape[1])

                    # ridge regression
                    nll, w_tr = regularized_log_reg_gd(_y_tr, tx_tr, w0, 5, gamma, lambda_)

                    acc = compute_accuracy(w_tr, tx_te, _y_te)

                    temp_acc.append(acc)
                print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
                # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
                accuracy_ranking[h, i, j] = np.mean(temp_acc)


gamma = 5e-3
degree = 4 #or maybe 4
# x_tr and y_tr already balanced from nans and standardized
tx_tr_tot = multi_build_poly(x_tr, degree)
# standardize again after polynomial basis expansion
tx_tr_tot = standardize_data(tx_tr_tot[:, 1:])
tx_tr_tot = np.concatenate((np.ones((tx_tr_tot.shape[0], 1)), tx_tr_tot), axis=1)

w0 = np.random.randn(tx_tr_tot.shape[1])
nll, w = stochastic_gradient_descent(y_tr, tx_tr_tot, w0, max_iters=10000, gamma=gamma, batch_size=1,
                                     mode='logistic_reg')
w = w[-1]


y_te, x_te, ids_te = load_csv_data("data/test.csv", mode='one_hot')

# Choice of variables to cut based on covariance and histograms

# nan to mean or median
x_te = standardize_data(x_te, nan_mode=nan_mode)
x_te = np.delete(x_te, cut_features1, axis=1)

tx_te_tot = multi_build_poly(x_te, degree)
tx_te_tot = standardize_data(tx_te_tot[:, 1:], nan_mode=nan_mode)
tx_te_tot = np.concatenate((np.ones((tx_te_tot.shape[0], 1)), tx_te_tot), axis=1)

y_te_pred = predict_labels(w, tx_te_tot, mode='one_hot')
y_te_pred[y_te_pred == 0] = -1

create_csv_submission(ids_te, y_te_pred, 'submission_median_cut9banana.csv')
