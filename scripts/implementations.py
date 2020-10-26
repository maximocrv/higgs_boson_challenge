"""Contains the required optimization implementations with any required additional functions."""
import numpy as np

from scripts.data_preprocessing import batch_iter, split_data_jet, preprocess_data, transform_data
from scripts.proj1_helpers import predict_labels


def compute_mae(y, tx, w):
    """
    Computes the mean absolute error.

    :param y: Label data.
    :param tx: Input features.
    :param w: Model weights
    :return: Mean absolute error of the predicted vector with respect to the label data.
    """
    e = y - tx @ w
    return np.mean(np.abs(e))


def compute_mse(y, tx, w):
    """
    Compute mean square error.

    :param y: Label data.
    :param tx: Input features.
    :param w: Model weights.
    :return: Mean squared error of the predicted vector with respect to the label data.
    """
    e = y - tx @ w
    return 1/2 * np.mean(e**2)


def compute_rmse(y, tx, w):
    """
    Compute root mean square error.

    :param y: Label data.
    :param tx: Input features.
    :param w: Model weights
    :return: Root mean squared error of the predicted vector with respect to the label data.
    """
    mse = compute_mse(y, tx, w)
    return np.sqrt(2*mse)


def compute_gradient(y, tx, w):
    """
    Compute the gradient for least squares.

    :param y: Label data.
    :param tx: Input features.
    :param w: Model weights
    :return: Gradient matrix containing derivatives with respect to all the weights.
    """

    return -1 / y.shape[0] * tx.T @ (y - tx @ w)


def least_squares_GD(y, tx, w0, max_iters, gamma):
    """
    Perform least squares using gradient descent.

    :param y: Label data.
    :param tx: Input features.
    :param w0: Initial model weights.
    :param max_iters: Maximum number of iterations to perform.
    :param gamma: Step size for performing weight updates (i.e. learning rate)
    :return: Final loss and model weights.
    """
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    ws = [w0]
    losses = []
    w = w0
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        w = w - gamma * grad

        ws.append(w)
        losses.append(loss)

    return losses[-1], w


def least_squares_SGD(y, tx, w0, max_iters, gamma, batch_size=1):
    """
    Perform least squares using stochastic gradient descent.

    :param y:
    :param tx:
    :param w0:
    :param max_iters:
    :param gamma:
    :param batch_size:
    :return:
    """
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    losses = []
    ws = []

    w = w0
    for i in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad = compute_gradient(batch_y, batch_tx, w)
            loss = compute_mse(batch_y, batch_tx, w)

            w = w - gamma * grad

            losses.append(loss)
            ws.append(w)

    return losses[-1], w


def least_squares(y, tx):
    """
    Calculate least squares solution using the normal equations.

    :param y:
    :param tx:
    :return:
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
    return mse, w


def ridge_regression(y, tx, lambda_):
    """
    Calculate the weights using ridge regression.

    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    a = tx.T @ tx + 2 * lambda_ * tx.shape[0] * np.eye(tx.shape[1])
    b = tx.T @ y

    w = np.linalg.solve(a, b)

    mse = compute_mse(y, tx, w)

    return mse, w


def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))


def compute_negative_log_likelihood_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx @ w)) - y * (tx @ w))


def compute_negative_log_likelihood_gradient(y, tx, w):
    """compute the gradient of the negative log likelihood."""
    return tx.T @ (sigmoid(tx @ w) - y)


def calculate_hessian(tx, w):
    """return the Hessian of the loss function."""

    S = np.diag((sigmoid(tx @ w) * (1 - sigmoid(tx @ w))).flatten())

    return tx.T @ S @ tx


def logistic_regression(y, tx, w0, max_iters, gamma):
    """
    Perform gradient descent using logistic regression.

    :param y: Label data.
    :param tx: Input features.
    :param w0: Initialization weight.
    :param max_iters: Maximum number of gradient descent iterations to perform.
    :param gamma: Step size for updating weights (i.e. learning rate)
    :return:
    """
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    ws = [w0]
    losses = []
    w = w0

    for i in range(max_iters):
        loss = compute_negative_log_likelihood_loss(y, tx, w)
        grad = compute_negative_log_likelihood_gradient(y, tx, w)

        w = w - gamma * grad

        ws.append(w0)
        losses.append(loss)

    return losses[-1], w


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Compute the penalized logistic regression and its corresponding gradient.

    :param y: Label data.
    :param tx: Input features.
    :param w: Model weights.
    :param lambda_: Regularization parameter lambda.
    :return: Penalized loss and corresponding gradient.
    """
    loss = np.sum(np.log(1 + np.exp(tx @ w)) - y * (tx @ w)) + lambda_ * np.linalg.norm(w) ** 2
    gradient = tx.T @ (sigmoid(tx @ w) - y) + 2 * lambda_ * w

    return loss, gradient


def reg_logistic_regression(y, tx, w0, max_iters, gamma, lambda_):
    """
    Perform gradient descent using regularized logistic regression.

    :param y: Label data.
    :param tx: Input features.
    :param w0: Initialization weight.
    :param max_iters: Maximum number of gradient descent iterations to perform.
    :param gamma: Step size for updating weights (i.e. learning rate)
    :param lambda_: Regularization parameter lambda.
    :return: Penalized logistic regression loss and model weights.
    """
    if w0 is None:
        w0 = np.zeros(tx.shape[1])

    ws = [w0]
    losses = []
    w = w0

    for i in range(max_iters):
        loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad

        ws.append(w0)
        losses.append(loss)

    return losses[-1], w


def compute_accuracy(w, x, y_true, binary_mode='default'):
    """
    Compute model accuracy.

    :param w: Model weights.
    :param x: Inpute features.
    :param y_true: Label data.
    :param binary_mode: Either 'default' or 'one_hot' for models using {-1,1} or {0,1} predictions respectively.
    :return: Accuracy expressed as percentage of correct predictions.
    """
    y_pred = predict_labels(w, x, binary_mode)
    true_list = y_true - y_pred
    num_true = np.where(true_list == 0)
    acc = len(num_true[0]) / y_true.shape[0]

    return acc


def cross_validation(y, x, method, k_indices, k, degree, split_mode, binary_mode, nan_mode, **kwargs):
    """
    This function performs k-means cross validation for one single fold. It can be used for any of the implementations
    by providing the appropriate function name. Furthermore, make sure to provide the correct key word arguments to
    avoid encountering any errors.

    :param y: Labels.
    :param x: Feature set.
    :param method: Function name i.e. any of the implementations mentioned above.
    :param k_indices: Array containing subarray with all the indices for each subgroup.
    :param k: Particular fold on which to perform the training and testing evaluation.
    :param degree: Degree of the basis polynomial.
    :param split_mode: Either 'jet_groups' or 'default'.
    :param binary_mode: Either 'one_hot' or 'default' for {0,1} or {-1,1} predictions respectively.
    :param nan_mode: Either 'mean', 'mode', or 'median'.
    :param kwargs: Extra arguments pertaining to the selected method. Please refer to the documentation to see which
    additional arguments are required.
    :return: Training and testing accuracies, as well as training and testing losses.
    """
    test_ind = k_indices[k]
    train_ind = k_indices[np.arange(len(k_indices)) != k].ravel()

    x_tr, y_tr = x[train_ind], y[train_ind]
    x_te, y_te = x[test_ind], y[test_ind]

    if split_mode == 'default':
        x_tr = preprocess_data(x_tr, nan_mode=nan_mode)
        x_te = preprocess_data(x_te, nan_mode=nan_mode)

        x_tr, x_te = transform_data(x_tr, x_te, degree)

        loss_tr, w = method(y_tr, x_tr, **kwargs)

        loss_te = compute_mse(y_te, x_te, w)

        acc_tr = compute_accuracy(w, x_tr, y_tr, binary_mode=binary_mode)
        acc_te = compute_accuracy(w, x_te, y_te, binary_mode=binary_mode)

    elif split_mode == 'jet_groups':
        y_train_pred = np.zeros(len(y_tr))
        y_test_pred = np.zeros(len(y_te))

        jet_groups_tr = split_data_jet(x_tr)
        jet_groups_te = split_data_jet(x_te)

        loss_te_list = []
        loss_tr_list = []
        for jet_group_tr, jet_group_te in zip(jet_groups_tr, jet_groups_te):
            _x_tr = x_tr[jet_group_tr]
            _x_te = x_te[jet_group_te]
            _y_tr = y_tr[jet_group_tr]
            _y_te = y_te[jet_group_te]

            _x_tr = preprocess_data(_x_tr, nan_mode=nan_mode)
            _x_te = preprocess_data(_x_te, nan_mode=nan_mode)

            _x_tr, _x_te = transform_data(_x_tr, _x_te, degree)

            _loss_tr, w = method(_y_tr, _x_tr, **kwargs)

            y_train_pred[jet_group_tr] = predict_labels(w, _x_tr, binary_mode=binary_mode)
            y_test_pred[jet_group_te] = predict_labels(w, _x_te, binary_mode=binary_mode)

            _loss_te = compute_mse(_y_te, _x_te, w)

        loss_te_list.append(_loss_te)
        loss_tr_list.append(_loss_tr)

        acc_tr = len(np.where(y_train_pred - y_tr == 0)[0]) / y_train_pred.shape[0]
        acc_te = len(np.where(y_test_pred - y_te == 0)[0]) / y_test_pred.shape[0]

        loss_tr = np.mean(loss_tr_list)
        loss_te = np.mean(loss_te_list)

    return acc_tr, acc_te, loss_tr, loss_te
g