# Script to run our best performing model with config etc etc
import numpy as np

from scripts.data_preprocessing import preprocess_data, split_data_jet, transform_data
from scripts.implementations import least_squares
from scripts.proj1_helpers import load_csv_data, predict_labels

# RUN SETTINGS
from scripts.utilities import compute_accuracy

split_mode = 'default'
binary_mode = 'one_hot'
nan_mode = 'median'
degree = 5

method = least_squares

x_tr, y_tr, inds_tr = load_csv_data('data/train.csv', mode=binary_mode)
x_te, y_te, inds_te = load_csv_data('data/test.csv', mode=binary_mode)


if split_mode == 'default':
    x_tr = preprocess_data(x_tr, nan_mode=nan_mode)
    x_te = preprocess_data(x_te, nan_mode=nan_mode)

    x_tr, x_te = transform_data(x_tr, x_te, degree)

    loss_tr, w = method(y_tr, x_tr)

    # y_tr_pred = predict_labels(w, x_tr, binary_mode=binary_mode)
    # y_te_pred = predict_labels(w, x_te, binary_mode=binary_mode)

    # loss_te = compute_mse(y_te, x_te, w)

    acc_tr = compute_accuracy(w, x_tr, y_tr, binary_mode=binary_mode)
    acc_te = compute_accuracy(w, x_te, y_te, binary_mode=binary_mode)

elif split_mode == 'jet_groups':
    y_train_pred = np.zeros(len(y_tr))
    y_test_pred = np.zeros(len(y_te))

    jet_groups_tr = split_data_jet(x_tr)
    jet_groups_te = split_data_jet(x_te)

    for jet_group_tr, jet_group_te in zip(jet_groups_tr, jet_groups_te):
        _x_tr = x_tr[jet_group_tr]
        _x_te = x_te[jet_group_te]
        _y_tr = y_tr[jet_group_tr]
        _y_te = y_te[jet_group_te]

        _x_tr = preprocess_data(_x_tr, nan_mode=nan_mode)
        _x_te = preprocess_data(_x_te, nan_mode=nan_mode)

        _x_tr, _x_te = transform_data(_x_tr, _x_te, degree)

        loss_tr, w = method(_y_tr, _x_tr)

        y_train_pred[jet_group_tr] = predict_labels(w, _x_tr, binary_mode=binary_mode)
        y_test_pred[jet_group_te] = predict_labels(w, _x_te, binary_mode=binary_mode)

        # loss_te = compute_mse(_y_te, _x_te, w)

    acc_tr = len(np.where(y_train_pred - y_tr == 0)[0]) / y_train_pred.shape[0]
    acc_te = len(np.where(y_test_pred - y_te == 0)[0]) / y_test_pred.shape[0]


