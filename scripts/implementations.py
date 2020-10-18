# -*- coding: utf-8 -*-
"""Function to compute least squared MSE and weights."""

import numpy as np

from scripts.costs import *
from scripts.helpers import batch_iter


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))

    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array([[w0[i]], [w1[j]]])
            losses[i, j] = (compute_mse(y, tx, w))

    return losses


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w)
    rmse = compute_rmse(y, tx, w)
    return mse, w


def compute_gradient(y, tx, w):
    """Compute the gradient."""

    return -1 / y.shape[0] * tx.T @ (y - tx @ w)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
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
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size=1):
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

    #         print(f'Gradient Descent ({i}/{max_iters-1}): loss={loss}, w0={w[0]}, w1={w[1]}')

    return losses, ws


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    a = tx.T @ tx + 2 * lambda_ * tx.shape[0] * np.eye(tx.shape[1])
    b = tx.T @ y

    w = np.linalg.solve(a, b)

    mse = compute_mse(y, tx, w)
    rmse = compute_rmse(y, tx, w)

    return mse, w


def sigmoid(t):
    """apply the sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))


def neg_log_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx @ w)) - y * (tx @ w))


def nll_grad(y, tx, w):
    """compute the gradient of the negative log likelihood."""
    return tx.T @ (sigmoid(tx @ w) - y)


def log_reg_gd(y, tx, w0, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    ws = [w0]
    losses = []
    w = w0

    for i in range(max_iters):
        loss = neg_log_loss(y, tx, w)
        grad = nll_grad(y, tx, w)
        w = w - gamma * grad

        ws.append(w0)
        losses.append(loss)

    return loss, w


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""

    S = np.diag((sigmoid(tx @ w) * (1 - sigmoid(tx @ w))).flatten())

    return tx.T @ S @ tx


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and Hessian."""

    loss = np.sum(np.log(1 + np.exp(tx @ w)) - y * tx @ w) + lambda_ * np.linalg.norm(w) ** 2
    gradient = tx.T @ (sigmoid(tx @ w) - y) + 2 * lambda_ * w

    S = np.diag((sigmoid(tx @ w) * (1 - sigmoid(tx @ w))).flatten())
    hessian = tx.T @ S @ tx + np.diag(np.ones((1, 3)) * 2 * lambda_)
    return loss, gradient, hessian


def regularized_log_reg_gd(y, tx, w0, max_iters, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    ws = [w0]
    losses = []
    w = w0

    for i in range(max_iters):
        loss, grad, hessian = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * grad

        ws.append(w0)
        losses.append(loss)

    return loss, w