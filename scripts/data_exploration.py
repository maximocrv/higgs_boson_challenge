from scripts.proj1_helpers import *

y, x, ids = load_csv_data("data/train.csv")


def count_nan(x):
    """

    :param x: input
    :return: ratios of nan
    """
    x[x == -999] = np.nan
    truth_array = np.isnan(x)
    return (np.sum(truth_array, axis=0) )/ x.shape[0]


def check_nan_positions(x,candidates):
    "check if nan occurs in the same place, outputs the percentage of values with nans in all the candidates columns"

    c= np.isnan(x[:,candidates])

    check = np.sum(c,1) == candidates.shape[0]

    value = np.sum(check)/ check.shape[0]

    return value


import numpy as np


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
    recall = cm[1, 1]/(cm[1, 1] + cm[0, 1])
    precision = cm[1, 1]/(cm[1, 1] + cm[1, 0])
    accuracy = (cm[0, 0] + cm[1, 1])/np.sum(cm)
    return recall, precision, accuracy


x[x == -999] = np.nan
candidates = np.array([0, 23, 24, 25, 4, 5, 6, 12, 26, 27, 28])
value1 = check_nan_positions(x, candidates)

candidates = np.array([0, 23, 24, 25])
value2 = check_nan_positions(x, candidates)

candidates = np.array([0, 4, 5, 6, 12, 26, 27, 28])
value3 = check_nan_positions(x, candidates)
