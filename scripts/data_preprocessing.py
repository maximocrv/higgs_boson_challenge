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


def set_nan(x):
    """
    Converts all -999 entries to nans.

    :param x: Input dataset
    :return: Input dataset with all -999's replaced by nans
    """
    x[x == -999] = np.nan
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
    x = set_nan(x)
    col_means = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_means, inds[1])

    return x


def standardize_data(x):
    """

    :param x:
    :return:
    """
    x = nan_to_mean(x)

    col_means = np.nanmean(x, axis=0)
    col_sd = np.nanstd(x, axis=0)

    # x[np.isnan(x)] =

    x = (x - col_means) / col_sd

    return x

def balance(y,x):
    """

    :param y: classification feature, 1 or -1
    :param x: input matrix
    :return: yv and xv, with equal hits and misses
    """
    x = set_nan(x)
    datalength = y.shape[0]
    hits = np.sum(y[y == 1])
    misses = - np.sum(y[y == -1])
    proportion_hits = hits / datalength
    diff = misses - hits

    features = np.array([23, 24, 25, 4, 5, 6, 12, 26, 27, 28])
    nancount = np.isnan(x[:, features])
    nancount_allfeat = np.sum(nancount, 1) == features.shape[0]
    misses_subgroup = np.sum(nancount_allfeat)
    assert misses_subgroup > diff, "not enough misses to cut that are nans in the selected features"
    all_indexes = np.argwhere(nancount_allfeat)
    cut_indexes = np.random.choice(all_indexes.flatten(), size=np.int(diff), replace=False )

    xv = np.delete(x, cut_indexes, axis=0)
    yv = np.delete(y, cut_indexes, axis=0)
    datalengthv = yv.shape[0]
    hitsv = np.sum(yv[yv == 1])
    missesv = - np.sum(yv[yv == -1])
    proportion_hitsv = hitsv / datalengthv
    diffv = missesv - hitsv
    return yv, xv

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
