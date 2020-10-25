import numpy as np


def set_nan(x):
    """
    Converts all -999 entries to nans.

    :param x: Input dataset
    :return: Input dataset with all -999's replaced by nans
    """
    x[x == -999] = np.nan
    return x


def remove_constant_columns(x):
    x = set_nan(x)

    nan_count = np.sum(np.isnan(x), axis=0)
    only_nans = np.where(nan_count == x.shape[0])
    x = np.delete(x, only_nans, axis=1)

    single_list = []
    for i in range(x.shape[1]):
        nan_rows = np.isnan(x[:, i])
        unique, counts = np.unique(x[~nan_rows, i], return_counts=True, axis=0)
        if len(unique) == 1:
            single_list.append(i)

    x = np.delete(x, single_list, axis=1)

    return x


def convert_nan(x, nan_mode='mode'):
    """
    Replace all -999 entries by the mean, median, or mode of their respective columns

    :param nan_mode:
    :param x: Input data
    :return: Input data containing column means in place of -999 entries
    """
    if nan_mode == 'mean':
        col_vals = np.nanmean(x, axis=0)

    elif nan_mode == 'median':
        col_vals = np.nanmedian(x, axis=0)

    elif nan_mode == 'mode':
        col_vals = np.zeros((1, x.shape[1]))

        for i in range(x.shape[1]):
            nan_rows = np.isnan(x[:, i])
            unique, counts = np.unique(x[~nan_rows, i], return_counts=True, axis=0)
            col_vals[:, i] = unique[counts.argmax()]

    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_vals, inds[1])
    return x


def standardize_data(x):
    """

    :param nan_mode:
    :param x:
    :return:
    """

    col_means = np.nanmean(x, axis=0)
    col_sd = np.nanstd(x, axis=0)

    x = (x - col_means) / col_sd

    return x, col_means, col_sd


def balance_all(y, x):
    """
    Balances the datasets by selecting and cutting a random subsets of misses, which are more abundant,
    to obtain an equal number of hits and misses
    :param y: input, categorical
    :param x: input features
    :return: 50/50 balanced random
    """
    x = set_nan(x)
    datalength = y.shape[0]
    hits = np.sum(y[y == 1])
    misses = - np.sum(y[y == -1])

    diff = misses - hits
    allmiss_indexes = np.argwhere(y == -1)
    cut_indexes = np.random.choice(allmiss_indexes.flatten(), size=np.int(diff), replace=False)
    xv = np.delete(x, cut_indexes, axis=0)
    yv = np.delete(y, cut_indexes, axis=0)

    # for testing : check if proportion hits = 0.5
    datalengthv = yv.shape[0]
    hitsv = np.sum(yv[yv == 1])
    missesv = - np.sum(yv[yv == -1])
    proportion_hitsv = hitsv / datalengthv
    diffv = missesv - hitsv
    return yv, xv


def balance_fromnans(y, x):
    """
    Balances the datasets by preferably cutting features with nans. To be used with the entire dataset and
    not with spit-number-of-jets specific subdatasets
    :param y: classification feature, 1 or -1
    :param x: input matrix
    :return: yv and xv, with equal hits and misses
    """
    x = set_nan(x)
    datalength = y.shape[0]
    hits = np.sum(y[y == 1])
    # misses = - np.sum(y[y == -1])
    misses = len(y[y == 0])
    proportion_hits = hits / datalength
    diff = misses - hits

    features = np.array([23, 24, 25, 4, 5, 6, 12, 26, 27, 28])
    nancount = np.isnan(x[:, features])
    # nancount_allfeat = (np.sum(nancount, 1) == features.shape[0]) & (y == -1)
    nancount_allfeat = (np.sum(nancount, 1) == features.shape[0]) & (y == 0)
    misses_subgroup = np.sum(nancount_allfeat)

    if misses_subgroup < diff:
        "not enough misses to cut that are nans in the selected features"
        amount = misses_subgroup
    else:
        amount = diff

    all_indexes = np.argwhere(nancount_allfeat)
    cut_indexes = np.random.choice(all_indexes.flatten(), size=np.int(amount), replace=False)

    xv = np.delete(x, cut_indexes, axis=0)
    yv = np.delete(y, cut_indexes, axis=0)

    # for testing : checks to verify the new ratio hit/tot
    datalengthv = yv.shape[0]
    hitsv = np.sum(yv[yv == 1])
    missesv = - np.sum(yv[yv == -1])
    proportion_hitsv = hitsv / datalengthv
    diffv = missesv - hitsv

    return yv, xv


def build_poly(x, degree):
    """polynomial basis functions for multidimensional input data x"""
    if degree == 0:
        x = np.ones((x.shape[0], 1))
    else:
        x = np.repeat(x[..., np.newaxis], degree, axis=-1)
        x = x ** np.arange(1, degree + 1)
        x = np.concatenate(x.transpose(2, 0, 1), axis=-1)
    return x


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)

    split = int(ratio * x.shape[0])
    train_ind = np.random.permutation(np.arange(x.shape[0]))[:split]
    test_ind = np.random.permutation(np.arange(x.shape[0]))[split:]

    x_tr, y_tr = x[train_ind], y[train_ind]
    x_te, y_te = x[test_ind], y[test_ind]

    return x_tr, y_tr, x_te, y_te


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def split_data_jet(x):
    """
    Splits the data depending on the value of the number of jets variable (22)
    :param x: input dataset
    :return: split datasets
    """
    ind_0 = x[:, 22] == 0
    ind_1 = x[:, 22] == 1
    ind_2 = np.logical_or(x[:, 22] == 2, x[:, 22] == 3)

    return ind_0, ind_1, ind_2


def preprocess_data(x, nan_mode):
    # remove unnecessary features, 22 -- > jet group number
    x = np.delete(x, [9, 15, 18, 20, 25, 28, 29], axis=1)
    # useless features, based on histograms (15, 18, 20, 25, 28) and linearity found with the covariance matrix (9,29)

    x = remove_constant_columns(x)

    x = convert_nan(x, nan_mode)

    return x


def remove_outliers(x):
    x_mean = np.mean(x, axis=0)
    x_sd = np.std(x, axis=0)

    lower_lim = x_mean - 6 * x_sd
    upper_lim = x_mean + 6 * x_sd

    testlower = np.any(x < lower_lim, axis=1)
    testupper = np.any(x > upper_lim, axis=1)

    outliers = np.logical_or(testlower, testupper)

    return x[~outliers]



def cross_channel_features(x):
    cross_x = np.zeros((x.shape[0], np.sum(np.arange(x.shape[1]))))

    count = 0
    for i in range(x.shape[1]):
        for j in range(i+1, x.shape[1]):
            cross_x[:, count] = x[:, i] * x[:, j]
            count += 1

    return cross_x


def transform_data(x_tr, x_te, degree):
    # x_tr_cross = cross_channel_features(x_tr)
    # x_te_cross = cross_channel_features(x_te)
    #
    # neg_cols = np.any(x_tr <= 0, axis=0)
    #
    # x_tr_log = np.log(x_tr[:, ~neg_cols])
    # x_te_log = np.log(x_te[:, ~neg_cols])

    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)

    # x_tr = np.concatenate((x_tr, x_tr_cross, x_tr_log), axis=1)
    # x_te = np.concatenate((x_te, x_te_cross, x_te_log), axis=1)

    x_tr, tr_mean, tr_sd = standardize_data(x_tr)
    x_te = (x_te - tr_mean) / tr_sd

    x_tr = np.concatenate((np.ones((x_tr.shape[0], 1)), x_tr), axis=1)
    x_te = np.concatenate((np.ones((x_te.shape[0], 1)), x_te), axis=1)

    return x_tr, x_te
