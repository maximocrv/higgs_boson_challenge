"""This script contains utility functions used throughout the codebase."""
import numpy as np

from scripts.proj1_helpers import predict_labels


def compute_f1score(w, x, y_true):
    y_pred = predict_labels(w, x)
    recall, precision, accuracy = calculate_recall_precision_accuracy(create_confusion_matrix(y_pred, y_true))

    return 2*(precision*recall)/(precision+recall)


def matthews_coeff(w, x, y_true, _y_pred):
    if w is not None:
        y_pred = predict_labels(w, x)
    else:
        y_pred = _y_pred
    cm = create_confusion_matrix(y_pred, y_true)
    tn, tp, fp, fn = cm[0, 0], cm[1, 1], cm[1, 0], cm[0, 1]
    mc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return mc


def create_confusion_matrix(predicted, actual) -> np.ndarray:
    """
    Given predicted and actual arrays, calculate confusion matrix (not normalized).
    Predicted on first axis, actual on second axis, i.e. confusion_matrix[0, 1] is false negative
    """
    cm = np.zeros((2, 2), dtype=np.int)
    for i in range(len(predicted)):
        if actual[i] == -1 and predicted[i] == -1:
            cm[0, 0] += 1
        if actual[i] == -1 and not predicted[i] == -1:
            cm[1, 0] += 1
        if actual[i] == 1 and not predicted[i] == 1:
            cm[0, 1] += 1
        if actual[i] == 1 and predicted[i] == 1:
            cm[1, 1] += 1
    return cm


def calculate_recall_precision_accuracy(confusion_matrix: np.ndarray) -> (float, float, float):
    """From the confusion matrix, return recall, precision and accuracy for class True"""
    cm = confusion_matrix
    recall = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    precision = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    return recall, precision, accuracy


def obtain_best_params(accuracy_ranking, gammas, degrees, lambdas):
    max_ind = np.unravel_index(np.argmax(accuracy_ranking), accuracy_ranking.shape)

    degree_ind = max_ind[1]
    degree = degrees[degree_ind]

    if gammas is not None and lambdas is not None:
        gamma_ind = max_ind[0]
        gamma = gammas[gamma_ind]

        lambda_ind = max_ind[2]
        lambda_ = lambdas[lambda_ind]

        return gamma, degree, lambda_

    elif gammas is not None and lambdas is None:
        gamma_ind = max_ind[0]
        gamma = gammas[gamma_ind]

        return gamma, degree

    elif gammas is None and lambdas is not None:
        lambda_ind = max_ind[0]
        lambda_ = lambdas[lambda_ind]

        return degree, lambda_

    else:
        return degree
