import numpy as np

from scripts.costs import compute_accuracy
from scripts.proj1_helpers import load_csv_data
from scripts.implementations import ridge_regression, cross_validation
from scripts.data_preprocessing import standardize_data, multi_build_poly, build_k_indices, generate_batch

# standardize data after polynomial basis expansion?
# function to generate test predictions?.....
# implement confusion matrix....

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv")

seed = 1
degrees = np.arange(6, 10)
lambdas = np.logspace(-5, -1, 5)
k_fold = 10
# split data in k fold
k_indices = build_k_indices(y_tr, k_fold, seed)
# define lists to store the loss of training data and test data
accuracy_ranking = np.zeros((len(lambdas), len(degrees)))

# cross validation
for h, lambda_ in enumerate(lambdas):
    for i, degree in enumerate(degrees):
        temp_acc = []
        for k in range(k_fold):
            acc_tr, acc_te = cross_validation(y_tr, x_tr, ridge_regression, k_indices, k, degree, mode='jet_groups',
                                              lambda_=lambda_)

            temp_acc.append(acc_te)
        print(f'#: {h*len(degrees) + i + 1} / {len(degrees) * len(lambdas)}, lambda: {lambda_}, degree: {degree}, '
              f'accuracy = {np.mean(temp_acc)}')
        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking[h, i] = np.mean(temp_acc)

# tx_tr_tot = multi_build_poly(x_tr, 7)
# mse, w = least_squares(y_tr, tx_tr_tot)
#
# y_te, x_te, ids_te = load_csv_data("data/test.csv")
#
# x_te = standardize_data(x_te)
# tx_te_tot = multi_build_poly(x_te, 7)
#
# y_te_pred = predict_labels(w, tx_te_tot)
#
# create_csv_submission(ids_te, y_te_pred, 'submission.csv')
