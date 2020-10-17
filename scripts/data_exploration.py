import numpy as np

from scripts.proj1_helpers import *

y, x, ids = load_csv_data("data/train.csv")

def count_nan(x):
    """

    :param x:
    :return:
    """
    return np.sum(np.isnan(x),axis=0) / x.shape(0)


