# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False, mode='default'):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1) (or also one-hot encoding)
    yb = np.ones(len(y))
    if mode == 'default':
        yb[np.where(y == 'b')] = -1
    elif mode == 'one_hot':
        yb[np.where(y == 'b')] = 0
    
    # sub-sample
    if sub_sample:
        num = 10
        yb = yb[::num]
        input_data = input_data[::num]
        ids = ids[::num]

    return yb, input_data, ids


def predict_labels(weights, data, binary_mode='default'):
    """Generates class predictions given weights, and a test data matrix"""
    if binary_mode == 'default':
        y_pred = np.dot(data, weights)

        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1

    elif binary_mode == 'one_hot':
        y_pred = np.dot(data, weights)

        y_pred = np.clip(y_pred, -500, 500)
        y_pred = np.exp(y_pred) / (1 + np.exp(y_pred))

        y_pred[np.where(y_pred < 0.5)] = 0
        y_pred[np.where(y_pred >= 0.5)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
