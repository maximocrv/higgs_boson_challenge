# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.tile(x, (degree+1, 1)).transpose()
    pwrs = np.arange(0, degree+1)
    x = x**pwrs
    return x


def multi_build_poly(x, degree):
    """polynomial basis functions for multidimensional input data x"""
    x = np.repeat(x[..., np.newaxis], degree, axis=-1)
    x = x ** np.arange(1, degree + 1)
    x = np.concatenate(x.transpose(2, 0, 1), axis=-1)
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    return x
