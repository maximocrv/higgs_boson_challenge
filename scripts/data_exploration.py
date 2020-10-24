import numpy as np

import matplotlib.pyplot as plt
from scripts.data_preprocessing import set_nan, standardize_data, convert_nan
from scripts.proj1_helpers import load_csv_data


def count_nan(x):
    """
    counts nan values in each feature
    :param x: input
    :return: vector composed of ratios of nan for each feature
    """
    x = set_nan(x)
    truth_array = np.isnan(x)
    return (np.sum(truth_array, axis=0)) / x.shape[0]


def check_nan_positions(x, features):
    """check if nan occurs in the same place, outputs the percentage of values with nans that are in all the chosen
    columns """

    c = np.isnan(x[:, features])
    check = np.sum(c, 1) == features.shape[0]

    value = np.sum(check) / check.shape[0]
    return value * 100


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
    x = convert_nan(x, nan_mode='median')
    x = standardize_data(x)
    covmat = np.corrcoef(x.T)
    return covmat


def principal_component_analysis(x):
    """
    
    :param y: 
    :param x: 
    :return: 
    """
    x = convert_nan(x, nan_mode='median')
    x = standardize_data(x)
    cov_matrix = np.cov(x.T)
    evalues, evectors = np.linalg.eig(cov_matrix)
    indexes = evalues.argsort()[::-1]  # Sort descending and get sorted indices
    evalues = evalues[indexes]
    evectors = evectors[:, indexes]
    explained_var = []
    for j in evalues:
        explanatory_power = (j / sum(evalues)) * 100
        explained_var.append(explanatory_power)
    return explained_var



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
                if np.abs(diff[i, j]) < 1E-1:
                    id = np.array([i, j])
                    ind_dep.append(id)
    return ind_dep


def set_cov_inf(cov):
    cov_ = cov
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if i >= j:
                cov_[i, j] = 0
    return cov_


def corr_col(cov, t):
    # initialize an array filled with the columns to be eliminated
    # t is the threshold correlation
    cde = []
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if abs(cov[i, j]) > t :
                v = np.array([i, j])
                cde.append(v)
    return cde


if __name__ == '__main__':
    y, x, ids = load_csv_data("data/train.csv")

    "how are the nans connected?"
    x = set_nan(x)
    nan_ratios = count_nan(x)
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

    "are features linearly dependent"
    ind_dep = lin_dep(x)
    print(lin_dep(x))

    # Data visualization
    # Standardization
    y, xbis, ids = load_csv_data("data/train.csv")

    xbis = set_nan(xbis)
    # Histograms to show the distribution of each feature, colored depending on hit/miss
    figure1 = plt.figure(1)
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        k = x[:, i]
        kt = k[y == 1]
        kf = k[y == -1]

        plt.hist(kt[~np.isnan(kt)], bins='auto', alpha=0.5, facecolor='b')
        plt.hist(kf[~np.isnan(kf)], bins='auto', alpha=0.5, facecolor='r')
        plt.title(f'feature : {i}')
        plt.axis('tight')

    # Lineplots of the ratios between hit and miss for different values, for each feature.
    # if this ratio is constant then the feature does not have predictive power
    figure2 = plt.figure(2)
    L = 50  # number of bins
    ratio = np.zeros(L)
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        k = x[:, i]
        kt = k[y == 1]
        kf = k[y == -1]
        kt = kt[~np.isnan(kt)]
        kf = kf[~np.isnan(kf)]
        kthist = np.histogram(kt, bins=L, range=(np.min(k[~np.isnan(k)]), np.max(k[~np.isnan(k)])))
        kfhist = np.histogram(kf, bins=L, range=(np.min(k[~np.isnan(k)]), np.max(k[~np.isnan(k)])))
        for j in range(L):
            if kfhist[0][j] == 0 or kthist[0][j] == 0:
                ratio[j] = 0
            else:
                ratio[j] = kthist[0][j] / kfhist[0][j]
        binz = kthist[1][0:L]
        plt.plot(binz, ratio)
        plt.title(f'feature : {i}')
        plt.ylim(0, 2)

    x = convert_nan(x, nan_mode='median')
    cov = covariance_matrix(x)
    cov_ = set_cov_inf(cov)
    col_el = corr_col(cov_, t=0.85)
    print(col_el)

    # Creation of an array with the numbered variables names
    """import pandas as pd

    df = pd.read_csv("data/train.csv")
    name_dict = {}
    for i, name in enumerate(df.columns):
        name_dict[f"{name}"] = f"{i - 2}_{name}"

    df = df.rename(columns=name_dict)"""

    """names = name_dict.values()
    df_names = pd.DataFrame(names)
    df_names.to_csv("names_vars.csv")
    names = np.array(list(names))
    names = names[..., np.newaxis].T"""

    # highly correlated features using unprocessed dataset [5, 6, 12, 21, 22, 24, 25, 26, 27, 28, 29]
    # highly correlated features using nan_to_mean dataset [2, 6, 7, 9, 11, 12, 16, 17, 19, 21, 22, 23, 29]
    # note that standardizing (i.e. transforming to standard normal RV has no effect on correlations)
