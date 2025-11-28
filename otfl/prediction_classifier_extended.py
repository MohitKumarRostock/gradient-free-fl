import numpy as np
from .combine_multiple_autoencoders_extended import (
    combine_multiple_autoencoders_extended,
)  # Python version of combineMultipleAutoencodersExtended.m


def prediction_classifier_extended(y_data_new, CLF, distance_type):
    """
    Python version of:
        function [min_distance,labels_arr] = predictionClassifierExtended(y_data_new,CLF,distance_type)

    Parameters
    ----------
    y_data_new : ndarray, shape (D, N)
        New data samples (columns are samples).
    CLF : dict
        Classifier struct as returned by `classifier()`:
        keys: 'labels' (1D array-like), 'AE_arr_arr' (list of AE_arr per class).
    distance_type : str
        Distance type passed through to combine_multiple_autoencoders_extended.

    Returns
    -------
    min_distance : ndarray, shape (N,)
        Minimum distance across all classes for each sample.
    labels_arr : ndarray, shape (N,)
        Predicted labels for each sample.
    """
    y_data_new = np.asarray(y_data_new, dtype=float)

    C = len(CLF["AE_arr_arr"])
    _, N = y_data_new.shape

    distance_matrix = np.zeros((C, N), dtype=float)

    # Fill distance matrix: one row per class
    for i in range(C):
        distances = combine_multiple_autoencoders_extended(
            y_data_new, CLF["AE_arr_arr"][i], distance_type
        )  # should be shape (N,)
        distance_matrix[i, :] = distances

    # Minimum distance across classes (axis 0 = over rows/classes)
    min_distance = np.min(distance_matrix, axis=0)  # shape (N,)

    # Assign labels corresponding to min distances
    labels = np.asarray(CLF["labels"])
    labels_arr = np.zeros(N, dtype=labels.dtype)

    for i in range(C):
        mask = distance_matrix[i, :] == min_distance
        labels_arr[mask] = labels[i]

    return min_distance, labels_arr
