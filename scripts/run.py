"""Script to generate our best test prediction."""
import numpy as np

from scripts.data_preprocessing import preprocess_data, split_data_jet, transform_data
from scripts.proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from scripts.implementations import compute_accuracy, ridge_regression

# RUN SETTINGS
split_mode = 'jet_groups'
binary_mode = 'default'
nan_mode = 'mode'
degree = 13
lambda_ = 1e-3

method = ridge_regression

y_tr, x_tr, ids_tr = load_csv_data('data/train.csv', mode=binary_mode)
_nothing_, x_te, ids_te = load_csv_data('data/test.csv', mode=binary_mode)


y_tr_pred = np.zeros(len(y_tr))
y_te_pred = np.zeros(len(x_te))

jet_groups_tr = split_data_jet(x_tr)
jet_groups_te = split_data_jet(x_te)

for i, (jet_group_tr, jet_group_te) in enumerate(zip(jet_groups_tr, jet_groups_te)):
    _x_tr = x_tr[jet_group_tr]
    _x_te = x_te[jet_group_te]
    _y_tr = y_tr[jet_group_tr]
    # _y_te = y_te[jet_group_te]

    _x_tr = preprocess_data(_x_tr, nan_mode=nan_mode)
    _x_te = preprocess_data(_x_te, nan_mode=nan_mode)

    _x_tr, _x_te = transform_data(_x_tr, _x_te, degree)

    loss_tr, w = method(_y_tr, _x_tr, lambda_=lambda_)

    y_tr_pred[jet_group_tr] = predict_labels(w, _x_tr, binary_mode=binary_mode)
    y_te_pred[jet_group_te] = predict_labels(w, _x_te, binary_mode=binary_mode)

    # loss_te = compute_mse(_y_te, _x_te, w)

acc_tr = len(np.where(y_tr_pred - y_tr == 0)[0]) / y_tr_pred.shape[0]
    # acc_te = len(np.where(y_te_pred - y_te == 0)[0]) / y_te_pred.shape[0]


#y_te_pred[y_te_pred == 0] = -1
create_csv_submission(ids_te, y_te_pred, 'ridge_reg_submission8.csv')
