import numpy as np

from scripts.utilities import compute_accuracy
from scripts.proj1_helpers import load_csv_data
from scripts.implementations import ridge_regression, cross_validation
from scripts.data_preprocessing import standardize_data, build_poly, build_k_indices, generate_batch

# standardize data after polynomial basis expansion?
# function to generate test predictions?.....
# implement confusion matrix....

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv")

seed = 1
degrees = np.arange(6, 10)
lambdas = np.logspace(-4, -1, 5)
k_fold = 10
# split data in k fold
k_indices = build_k_indices(y_tr, k_fold, seed)
# define lists to store the loss of training data and test data
accuracy_ranking = np.zeros((len(lambdas), len(degrees)))

for h, lambda_ in enumerate(lambdas):
    for i, degree in enumerate(degrees):
        temp_acc = []
        for k in range(k_fold):
            acc_tr, acc_te = cross_validation(y_tr, x_tr, ridge_regression, k_indices, k, degree, split_mode='jet_groups',
                                              lambda_=lambda_)

            temp_acc.append(acc_te)
        print(f'#: {h*len(degrees) + i + 1} / {len(degrees) * len(lambdas)}, lambda: {lambda_}, degree: {degree}, '
              f'accuracy = {np.mean(temp_acc)}')
        # accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking[h, i] = np.mean(temp_acc)
