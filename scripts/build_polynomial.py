# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.tile(x, (degree+1, 1)).transpose()
    pwrs = np.arange(0, degree+1)
    x = x**pwrs
    return x