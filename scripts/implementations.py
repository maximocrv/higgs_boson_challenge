import numpy as np

from scripts.utilities import compute_mse, compute_rmse, compute_accuracy, compute_gradient, sigmoid, \
    compute_negative_log_likelihood_loss, compute_negative_log_likelihood_gradient, matthews_coeff
from scripts.data_preprocessing import batch_iter, build_poly, split_data_jet, preprocess_data
from scripts.proj1_helpers import predict_labels


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)

        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss,
        #                                                                        w0=w[0], w1=w[1]))

    return losses, ws


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Stochastic gradient descent algorithm."""
    losses = []
    ws = []

    w = initial_w

    for i, (batch_y, batch_tx) in enumerate(batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters)):
        grad = compute_gradient(batch_y, batch_tx, w)
        loss = compute_mse(batch_y, batch_tx, w)

        w = w - gamma * grad

        losses.append(loss)
        ws.append(w)

        # print(f'Gradient Descent ({i}/{max_iters-1}): loss={loss}, w0={w[0]}, w1={w[1]}')

    return losses, ws


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
    return mse, w


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    a = tx.T @ tx + 2 * lambda_ * tx.shape[0] * np.eye(tx.shape[1])
    b = tx.T @ y

    w = np.linalg.solve(a, b)

    mse = compute_mse(y, tx, w)

    return mse, w


def logistic_regression_GD(y, tx, w0, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    ws = [w0]
    losses = []
    w = w0

    for i in range(max_iters):
        loss = compute_negative_log_likelihood_loss(y, tx, w)
        grad = compute_negative_log_likelihood_gradient(y, tx, w)
        w = w - gamma * grad

        ws.append(w0)
        losses.append(loss)

    return loss, w


def logistic_regression_SGD(y, tx, w0, max_iters, gamma, batch_size=1):
    """Stochastic gradient descent algorithm."""
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    losses = []
    ws = [w0]

    w = w0

    for i, (batch_y, batch_tx) in enumerate(batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters)):
        grad = compute_negative_log_likelihood_loss(batch_y, batch_tx, w)
        loss = compute_negative_log_likelihood_gradient(batch_y, batch_tx, w)

        w = w - gamma * grad

        losses.append(loss)
        ws.append(w)

        # print(f'Stochastic Gradient Descent ({i}/{max_iters-1}): loss={loss}, w0={w[0]}, w1={w[1]}')

    return losses, ws


def reg_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and Hessian."""

    loss = np.sum(np.log(1 + np.exp(tx @ w)) - y * (tx @ w)) + lambda_ * np.linalg.norm(w) ** 2
    gradient = tx.T @ (sigmoid(tx @ w) - y) + 2 * lambda_ * w

    S = np.diag((sigmoid(tx @ w) * (1 - sigmoid(tx @ w))).flatten())
    hessian = tx.T @ S @ tx + np.diag(np.ones((1, 3)) * 2 * lambda_)

    return loss, gradient, hessian


def reg_logistic_regression_GD(y, tx, w0, max_iters, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    ws = [w0]
    losses = []
    w = w0

    for i in range(max_iters):
        loss, grad, hessian = reg_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad

        ws.append(w0)
        losses.append(loss)

    return loss, w


def reg_logistic_regression_SGD(y, tx, w0, max_iters, gamma, lambda_, batch_size=1):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    ws = [w0]
    losses = []
    w = w0
    for i, (batch_y, batch_tx) in enumerate(batch_iter(y, tx, batch_size=batch_size, num_batches=max_iters)):
        loss, grad, hessian = reg_logistic_regression(batch_y, batch_tx, w, lambda_)

        w = w - gamma * grad

        losses.append(loss)
        ws.append(w)

        # print(f'Stochastic Gradient Descent ({i}/{max_iters-1}): loss={loss}, w0={w[0]}, w1={w[1]}')

    return loss, w


def cross_validation(y, x, method, k_indices, k, degree, mode, **kwargs):
    """return the loss of ridge regression."""

    test_ind = k_indices[k]
    train_ind = k_indices[np.arange(len(k_indices)) != k].ravel()

    x_tr, y_tr = x[train_ind], y[train_ind]
    x_te, y_te = x[test_ind], y[test_ind]

    if mode == 'default':
        x_tr = preprocess_data(x_tr, degree=degree, mode='median')
        x_te = preprocess_data(x_te, degree=degree, mode='median')

        loss_tr, w = method(y_tr, x_tr, degree, **kwargs)

        y_tr_pred = predict_labels(w, x_tr, mode='default')
        y_te_pred = predict_labels(w, x_te, mode='default')

        loss_te = compute_mse(y_te, x_te, w)

        acc_tr = compute_accuracy(w, x_tr, y_tr, mode='default')
        acc_te = compute_accuracy(w, x_te, y_te, mode='default')

        mc_tr = matthews_coeff(w, x_tr, y_tr)
        mc_te = matthews_coeff(w, x_te, y_te)

    elif mode == 'jet_groups':
        y_train_pred = np.zeros(len(y_tr))
        y_test_pred = np.zeros(len(y_te))

        jet_groups_tr = split_data_jet(x_tr)
        jet_groups_te = split_data_jet(x_te)

        for jet_group_tr, jet_group_te in zip(jet_groups_tr, jet_groups_te):
            _x_tr = x_tr[jet_group_tr]
            _x_te = x_te[jet_group_te]
            _y_tr = y_tr[jet_group_tr]
            _y_te = y_te[jet_group_te]

            _x_tr = preprocess_data(_x_tr, mode='mode', degree=degree)
            _x_te = preprocess_data(_x_te, mode='mode', degree=degree)

            loss_tr, w = method(_y_tr, _x_tr, **kwargs)

            y_train_pred[jet_group_tr] = predict_labels(w, _x_tr, mode='default')
            y_test_pred[jet_group_te] = predict_labels(w, _x_te, mode='default')

            loss_te = compute_mse(_y_te, _x_te, w)

        acc_tr = len(np.where(y_train_pred - y_tr == 0)[0]) / y_train_pred.shape[0]
        acc_te = len(np.where(y_test_pred - y_te == 0)[0]) / y_test_pred.shape[0]

        mc_tr = matthews_coeff(w=None, x=None, y_true=y_tr, y_pred=y_train_pred)
        mc_te = matthews_coeff(w=None, x=None, y_true=y_te, y_pred=y_test_pred)

    return acc_tr, acc_te, mc_tr, mc_te
