import numpy as np


def set_nan(x):
    """
    Converts all -999 entries to nans.

    :param x: Input dataset
    :return: Input dataset with all -999's replaced by nans
    """
    x[x == -999] = np.nan
    return x


def convert_nan(x, nan_mode='mode'):
    """
    Replace all -999 entries by the mean of their respective columns

    :param nan_mode:
    :param x: Input data
    :return: Input data containing column means in place of -999 entries
    """
    x = set_nan(x)

    nan_count = np.sum(np.isnan(x), axis=0)
    only_nans = np.where(nan_count == x.shape[0])
    x = np.delete(x, only_nans, axis=1)

    if nan_mode == 'mean':
        col_vals = np.nanmean(x, axis=0)

    elif nan_mode == 'median':
        col_vals = np.nanmedian(x, axis=0)

    elif nan_mode == 'mode':
        col_vals = np.zeros((1, x.shape[1]))
        single_list = []
        for i in range(x.shape[1]):
            nan_rows = np.isnan(x[:, i])
            unique, counts = np.unique(x[~nan_rows, i], return_counts=True, axis=0)
            if len(unique) == 1:
                single_list.append(i)
                col_vals[:, i] = np.nan
            else:
                col_vals[:, i] = unique[counts.argmax()]
        x = np.delete(x, single_list, axis=1)
        col_vals = col_vals[~np.isnan(col_vals)]

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

    return x


def balance_all(y, x):
    """

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

    # check if proportion hits = 0.5
    datalengthv = yv.shape[0]
    hitsv = np.sum(yv[yv == 1])
    missesv = - np.sum(yv[yv == -1])
    proportion_hitsv = hitsv / datalengthv
    diffv = missesv - hitsv
    return yv, xv


def balance_fromnans(y, x):
    """

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

    # checks to verify the new ratio hit/tot
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
        # tx = np.c_[np.ones(num_samples), x]
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


def generate_batch(y, x, k_indices, k):
    """return the loss of ridge regression."""
    # indices calculation
    te_indices = k_indices[k]
    tr_indices = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_indices = tr_indices.reshape(-1)
    # dividing x and y in training and testing set
    x_te = x[te_indices]
    x_tr = x[tr_indices]
    y_te = y[te_indices]
    y_tr = y[tr_indices]
    return x_tr, y_tr, x_te, y_te


def split_data_jet(x):
    ind_0 = x[:, 22] == 0
    ind_1 = x[:, 22] == 1
    ind_2 = np.logical_or(x[:, 22] == 2, x[:, 22] == 3)

    return ind_0, ind_1, ind_2


def preprocess_data(x, nan_mode, degree):
    # remove unnecessary features, 22 -- > jet group number
    x = np.delete(x, [15, 18, 20, 25, 28, 9, 29], axis=1)
    # useless features, based on histograms (15, 18, 20, 25, 28) and linearity found with the covariance matrix (9,29)

    x = convert_nan(x, nan_mode)

    x = build_poly(x, degree)

    x = standardize_data(x)

    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    return x
