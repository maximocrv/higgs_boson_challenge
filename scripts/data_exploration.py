import numpy as np
import matplotlib.pyplot as plt
from scripts.proj1_helpers import *
from scripts.data_preprocessing import *


def count_nan(x):
    """
    counts nan values in each feature
    :param x: input
    :return: vector composed of ratios of nan for each feature
    """
    x = set_nan(x)
    truth_array = np.isnan(x)
    return (np.sum(truth_array, axis=0) )/ x.shape[0]


def check_nan_positions(x,candidates):
    "check if nan occurs in the same place, outputs the percentage of values with nans in all the candidates columns"

    c = np.isnan(x[:, candidates])

    return c


def check_nan_positions(x, features):
    "check if nan occurs in the same place, outputs the percentage of values with nans that are in all the chosen columns"

    c= np.isnan(x[:, features])
    check = np.sum(c,1) == features.shape[0]

    value = np.sum(check)/ check.shape[0]
    return value


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


def cut_datasets(y, x, features):
    "eliminates datapoints which have nans in specific features"

    c = np.isnan(x[:, features])
    check = np.sum(c, 1) == features.shape[0]
    check = np.logical_not(check)
    x = x[check, :]
    y = y[check]
    return y, x


def covariance_matrix(x):
    """
    computes the covariance matrix computed with pearson's coefficient.
    :param x: input 
    :return: covmat : covariance matrix
    """
    x= standardize_data(x)
    covmat = np.corrcoef(x.T)
    return covmat


def lin_dep(x):
    # define the matrix containing the inner products of the columns
    inn_prod = x.T @ x
    # define the matrix containing the products of the norms of the columns
    arr_norm = np.linalg.norm(x, axis=0)[..., np.newaxis]
    norm_prod = arr_norm @ arr_norm.T
    # define the difference matrix
    diff = inn_prod - norm_prod
    # define indices where the difference is = 0
    # the indices represents the linearly dependent columns
    ind_dep = []
    for i in range(diff.shape[0]):
        for j in range(diff.shape[0]):
            if i != j:
                if np.abs(diff[i,j]) < 1E-1:
                    id = np.array([i, j])
                    ind_dep.append(id)
    return ind_dep


if __name__ == '__main__':
    y, x, ids = load_csv_data("data/train.csv")

    "how are the nans connected?"
    x = set_nan(x)

    features1 = np.array([4, 5, 6, 12, 26, 27, 28])
    value1 = check_nan_positions(x, features1)

    features2 = np.array([23, 24, 25])
    value2 = check_nan_positions(x, features2)

    features3 = np.array([0, 23, 24, 25, 4, 5, 6, 12, 26, 27, 28])
    value3 = check_nan_positions(x, features3)

    "is the dataset balanced?"
    proportion_hits = np.sum(y[y == 1]) / y.shape[0]

    y1, x1 = cut_datasets(y, x, features1)
    y2, x2 = cut_datasets(y, x, features2)
    y3, x3 = cut_datasets(y, x, features3)

    "proportions of hits in the segments remaining after the cut"
    proportion_hits1 = np.sum(y1[y1 == 1]) / y1.shape[0]
    proportion_hits2 = np.sum(y2[y2 == 1]) / y2.shape[0]
    proportion_hits3 = np.sum(y3[y3 == 1]) / y3.shape[0]

    covariance = covariance_matrix(x)

    covariance[covariance > 0.99] = 0
    covariance[covariance > 0.6] = 10
    covariance[covariance < -0.6] = -10

    "are features linearly dependent"
    ind_dep = lin_dep(x)
    print(lin_dep(x))


# data visualization with histograms
    y, xbis, ids = load_csv_data("data/train.csv")
    col_means = np.nanmean(xbis, axis=0)
    col_sd = np.nanstd(xbis, axis=0)

    xbis = (xbis - col_means) / col_sd

    figure = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        k = xbis[:, i]
        plt.hist(k[~np.isnan(k)], bins='auto')
        plt.title(f'feature : {i}')
        plt.xlim(-5, 5)
