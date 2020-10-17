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

x[x == -999] = np.nan
candidates= np.array([0, 23,24,25, 4,5,6,12,26,27,28])
value1 = check_nan_positions(x,candidates)

candidates= np.array([0, 23,24,25])
value2 = check_nan_positions(x,candidates)

candidates= np.array([0, 4,5,6,12,26,27,28])
value3 = check_nan_positions(x,candidates)