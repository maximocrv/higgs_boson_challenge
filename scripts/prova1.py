import numpy as np
from scripts.proj1_helpers import load_csv_data
from scripts.data_preprocessing import balance_fromnans, standardize_data, build_k_indices, generate_batch, \
    multi_build_poly
from scripts.implementations import regularized_log_reg_gd
from scripts.costs import compute_accuracy, compute_f1score

# load the data
y_tr, x_tr, ids_tr = load_csv_data("data/train.csv", mode='one_hot')

# balance dataset
y_tr, x_tr = balance_fromnans(y_tr, x_tr)

# cut features based on correlation
feat = [15, 18, 20, 25, 28]
x_tr = np.delete(x_tr, feat, axis=1)

# standardize data
x_tr = standardize_data(x_tr)

# initialize setting variables
seed = 1
degrees = np.arange(1, 4)
k_fold = 4
gammas = [1e-3, 1e-2, 1e-1]
lambdas = [1e-4, 1e-3, 1e-2, 1e-1]

# split data in k fold for cross validation
k_indices = build_k_indices(y_tr, k_fold, seed)

accuracy_ranking = np.zeros((len(gammas), len(degrees), len(lambdas)))
f1_ranking = np.zeros((len(gammas), len(degrees), len(lambdas)))

for h, gamma in enumerate(gammas):
    for i, degree in enumerate(degrees):
        for j, lambda_ in enumerate(lambdas):
            temp_acc = []
            temp_f1 = []
            for k in range(k_fold):
                # generate batch
                _x_tr, _y_tr, _x_te, _y_te = generate_batch(y_tr, x_tr, k_indices, k)

                # polynomial expansion of the features
                tx_tr = multi_build_poly(_x_tr, degree)
                tx_te = multi_build_poly(_x_te, degree)

                # standardize the features after polynomial expansion
                tx_tr = standardize_data(tx_tr[:, 1:])
                tx_te = standardize_data(tx_te[:, 1:])

                # initialize w0 with random values
                w0 = np.random.randn(tx_tr.shape[1])

                # regularized ridge regression
                # loss, w = regularized_log_reg_gd(y, tx, w0, max_iters, gamma, lambda_):
                nll, w_tr = regularized_log_reg_gd(_y_tr, tx_tr, w0, 5, gamma, lambda_)

                # compute accuracy
                acc = compute_accuracy(w_tr, tx_te, _y_te)

                # compute f1 score
                f1 = compute_f1score(w_tr, tx_te, _y_te)

                temp_acc.append(acc)
                temp_f1.append(f1)

            print(f'degree: {degree}, lambda = {lambda_}, gamma = {gamma}, accuracy = {np.mean(temp_acc)}, '
                  f'f1 score = {np.mean(temp_f1)}')

            # accuracy and f1 score rankings
            # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
            accuracy_ranking[h, i, j] = np.mean(temp_acc)
            f1_ranking[h, i, j] = np.mean(temp_f1)



