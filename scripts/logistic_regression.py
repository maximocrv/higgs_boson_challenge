import numpy as np

from scripts.costs import compute_accuracy
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.data_preprocessing import standardize_data, build_k_indices, generate_batch, balance_fromnans, \
    multi_build_poly, convert_nan
from scripts.implementations import log_reg_gd, penalized_logistic_regression, stochastic_gradient_descent, \
    regularized_log_reg_gd

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')


y_tr, x_tr = balance_fromnans(y_tr, x_tr)
#STANDARDIZE DATA AFTER GENERATING FEATURE EXPANSION VECTOR
# try the median or mean
x_tr = standardize_data(x_tr,'median')

# Choice of variables to cut based on covariance and histograms
cut_features = np.array([9, 29, 3,   4])
cut_features2 = np.array([15, 18, 20])
cut_features3 = np.array([4,5,6,12,26,27,28])
#x_tr = np.delete(x_tr, cut_features, axis=1)


seed = 1
degrees = np.arange(3, 8)
k_fold = 5
gammas = [1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

# set mode to either lr, lr_sgd or regularized_lr
mode = 'regularized_lr'

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
                    nll, w_tr = regularized_log_reg_gd(_y_tr, tx_tr, w0, max_iters=5, gamma, lambda_)

                    acc = compute_accuracy(w_tr, tx_te, _y_te)

                    temp_acc.append(acc)
                print(f'#: {h * len(degrees) + i + 1} / {len(gammas) * len(degrees)}, accuracy = {np.mean(temp_acc)}')
                # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
                accuracy_ranking[h, i, j] = np.mean(temp_acc)


gamma = 1e-1
degree = 5
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
x_te = standardize_data(x_te, 'median')
# x_te = np.delete(x_te, cut_features, axis=1)

tx_te_tot = multi_build_poly(x_te, degree)
tx_te_tot = standardize_data(tx_te_tot[:, 1:], nan_mode=)
tx_te_tot = np.concatenate((np.ones((tx_te_tot.shape[0], 1)), tx_te_tot), axis=1)

y_te_pred = predict_labels(w, tx_te_tot, mode='one_hot')
y_te_pred[y_te_pred == 0] = -1

create_csv_submission(ids_te, y_te_pred, 'log_reg_sgd_submission.csv')
