import numpy as np

from scripts.helpers import batch_iter
from scripts.proj1_helpers import *
from scripts.data_preprocessing import *
from scripts.model_batching import build_k_indices, generate_batch
from scripts.build_polynomial import multi_build_poly
from scripts.implementations import stochastic_gradient_descent, compute_accuracy

y_tr, x_tr, ids_tr = load_csv_data("data/train.csv")

x_tr = standardize_data(x_tr)

x_tr = nan_to_mean(x_tr)



seed = 1
degrees = np.arange(3, 8)
k_fold = 5
gammas = np.logspace(-5, 0, 10)
# split data in k fold
k_indices = build_k_indices(y_tr, k_fold, seed)
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
            mse_tr, w_tr = stochastic_gradient_descent(_y_tr, tx_tr, w0, max_iters=50, gamma=gamma, batch_size=1)

            acc = compute_accuracy(w_tr, tx_te, _y_te)

            temp_acc.append(acc)

        #accuracy_ranking[h,i]=np.mean(temp_acc)-2*np.std(temp_acc)
        accuracy_ranking[h, i] = np.mean(temp_acc)
