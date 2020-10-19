import numpy as np

from scripts.proj1_helpers import load_csv_data

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


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.tile(x, (degree+1, 1)).transpose()
    pwrs = np.arange(0, degree+1)
    x = x**pwrs
    return x


def multi_build_poly(x, degree):
    """polynomial basis functions for multidimensional input data x"""
    if degree == 0:
        x = np.ones((x.shape[0], 1))
    else:
        x = np.repeat(x[..., np.newaxis], degree, axis=-1)
        x = x ** np.arange(1, degree + 1)
        x = np.concatenate(x.transpose(2, 0, 1), axis=-1)
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        #tx = np.c_[np.ones(num_samples), x]
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
