import numpy as np

from scripts.build_polynomial import build_poly
from scripts.proj1_helpers import *
from scripts.implementations import least_squares

y_tr, tx_tr, ids_tr = load_csv_data("data/train.csv")

min_inputs = np.min(tx_tr, axis=1)
min_input_ind = np.isin(min_inputs, -999)

tx_tr = tx_tr[~min_input_ind]


degree = 8
tx_tr = np.repeat(tx_tr[..., np.newaxis], degree, axis=-1)
tx_tr = tx_tr ** np.arange(1, degree + 1)
tx_tr = np.concatenate(tx_tr.transpose(2, 0, 1), axis=-1)
# tx = build_poly(tx, 5)
tx_tr = np.concatenate((np.ones((tx_tr.shape[0], 1)), tx_tr), axis=1)


y_tr = y_tr[~min_input_ind]

mse, w = least_squares(y_tr, tx_tr)

y_pred = predict_labels(w, tx_tr)
true_list = y_tr - y_pred
num_true = np.where(true_list == 0)

accuracy = len(num_true[0]) / y_tr.shape[0]

print(f'accuracy = {accuracy}')