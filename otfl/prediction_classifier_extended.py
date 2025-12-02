import numpy as np
from joblib import Parallel, delayed

from .combine_multiple_autoencoders_extended import (
    combine_multiple_autoencoders_extended,
)


def prediction_classifier_extended(y_data_new, CLF, distance_type, n_jobs=1):
    """
    Python version of:
        [min_distance, labels_arr] = predictionClassifierExtended(y_data_new, CLF, distance_type)

    Parameters
    ----------
    y_data_new : ndarray, shape (D, N)
        New data samples (columns are samples).
    CLF : dict
        Classifier struct as returned by `classifier()`:
        keys: 'labels' (1D array-like), 'AE_arr_arr' (list of AE_arr per class).
    distance_type : str
        Distance type passed through to combine_multiple_autoencoders_extended.
    n_jobs : int, optional
        Number of parallel jobs over classes. 1 = no parallelism, -1 = all cores.

    Returns
    -------
    min_distance : ndarray, shape (N,)
        Minimum distance across all classes for each sample.
    labels_arr : ndarray, shape (N,)
        Predicted labels for each sample.
    """
    y_data_new = np.asarray(y_data_new, dtype=float)
    AE_arr_arr = CLF["AE_arr_arr"]
    labels = np.asarray(CLF["labels"])

    C = len(AE_arr_arr)
    _, N = y_data_new.shape

    # ------------------------------------------------------------
    # Fast path for the very common case: a single query (N = 1)
    # ------------------------------------------------------------
    if N == 1 and n_jobs == 1:
        best_dist = np.inf
        best_label = None
        for AE_arr, lab in zip(AE_arr_arr, labels):
            # distances shape: (N,) == (1,)
            dist = combine_multiple_autoencoders_extended(
                y_data_new, AE_arr, distance_type
            )[0]
            if dist < best_dist:
                best_dist = dist
                best_label = lab

        min_distance = np.array([best_dist], dtype=float)
        labels_arr = np.array([best_label], dtype=labels.dtype)
        return min_distance, labels_arr

    # ------------------------------------------------------------
    # General path: compute distances per class, optionally in parallel
    # ------------------------------------------------------------
    if n_jobs == 1:
        # Sequential: allocate once and fill
        distance_matrix = np.empty((C, N), dtype=float)
        for i, AE_arr in enumerate(AE_arr_arr):
            distance_matrix[i, :] = combine_multiple_autoencoders_extended(
                y_data_new, AE_arr, distance_type
            )
    else:
        # Parallel over classes
        results = Parallel(n_jobs=n_jobs)(
            delayed(combine_multiple_autoencoders_extended)(
                y_data_new, AE_arr, distance_type
            )
            for AE_arr in AE_arr_arr
        )
        # results is a list of arrays of shape (N,)
        distance_matrix = np.vstack(results)  # shape (C, N)

    # ------------------------------------------------------------
    # Vectorised label assignment
    # ------------------------------------------------------------
    # argmin over classes gives index of best class per sample
    best_idx = np.argmin(distance_matrix, axis=0)         # shape (N,)
    min_distance = distance_matrix[best_idx, np.arange(N)]
    labels_arr = labels[best_idx]

    return min_distance, labels_arr
