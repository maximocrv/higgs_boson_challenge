import numpy as np

from scripts.build_polynomial import build_poly, multi_build_poly
from scripts.proj1_helpers import *
from scripts.implementations import least_squares

y_tr, tx_tr, ids_tr = load_csv_data("data/train.csv")

min_inputs = np.min(tx_tr, axis=1)
min_input_ind = np.isin(min_inputs, -999)

tx_tr = tx_tr[~min_input_ind]


tx_tr = multi_build_poly(tx_tr, 6)

y_tr = y_tr[~min_input_ind]

mse, w = least_squares(y_tr, tx_tr)

y_pred = predict_labels(w, tx_tr)
true_list = y_tr - y_pred
num_true = np.where(true_list == 0)

accuracy = len(num_true[0]) / y_tr.shape[0]

print(f'accuracy = {accuracy}')
