import numpy as np

from scripts.proj1_helpers import *

y, x, ids = load_csv_data("data/train.csv")


def nan_to_0(x):
    """
    Converts all -999 entries to 0's.

    :param x: Input dataset
    :return: Input dataset with all -999's replaced by 0's
    """
    x[x == -999] = 0
    return x


def remove_nan_features(x):
    """
    Removes feature columns containing -999 values.

    :param x: Input data
    :return: Returns input data with stripped columns
    """
    col_mins = np.min(x, axis=0)
    x = x[:, col_mins != -999]
    return x


def remove_nan_entries(x):
    """
    Removes rows containing features with value -999

    :param x: Input data
    :return: Input data without all the rows containing -999 entries
    """
    row_mins = np.min(x, axis=1)
    x = x[row_mins != -999, :]
    return x


def nan_to_mean(x):
    """
    Replace all -999 entries by the mean of their respective columns

    :param x: Input data
    :return: Input data containing column means in place of -999 entries
    """
    x[x == -999] = np.nan
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])

    return x


def standardize_data(x):
    """


    :param x:
    :return:
    """
    col_means = x.mean(axis=0)
    col_sd = x.std(axis=0)

    x = (x - col_means) / col_sd

    return x


def check_linearity(x):
    raise NotImplementedError


def eigen_corr(x):
    """
    Generate eigenvalue matrix for pearson correlation matrix.
    :param x:
    :return:
    """
    corr_mat = np.corrcoef(x)
    eig = np.linalg.eig(corr_mat)

    return eig