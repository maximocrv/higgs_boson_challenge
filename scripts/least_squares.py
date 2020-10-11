import numpy as np

from scripts.build_polynomial import build_poly
from scripts.proj1_helpers import *
from scripts.implementations import least_squares

y, tx, ids = load_csv_data("data/train.csv")

min_inputs = np.min(tx, axis=1)
min_input_ind = np.isin(min_inputs, -999)

tx = tx[~min_input_ind]

tx = build_poly(tx, 5)

y = y[~min_input_ind]

mse, w = least_squares(y, tx)

y_pred = predict_labels(w, tx)
true_list = y - y_pred
num_true = np.where(true_list == 0)

accuracy = len(num_true[0]) / y.shape[0]