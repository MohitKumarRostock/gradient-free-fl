import numpy as np

def combine_multiple_classifiers(distance_arr, labels_arr_arr):
    """
    Python version of:
        function [min_distance,labels_arr] = combineMultipleClassifiers(distance_arr,labels_arr_arr)

    Parameters
    ----------
    distance_arr : list of array-like
        distance_arr[i] is a 1D array of distances for classifier i, length N.
    labels_arr_arr : list of array-like
        labels_arr_arr[i] is a 1D array of labels for classifier i, length N.

    Returns
    -------
    min_distance : ndarray, shape (N,)
        Element-wise minimum distance over all classifiers.
    labels_arr : ndarray, shape (N,)
        Labels corresponding to the (last) classifier achieving the minimum.
    """
    Q = len(distance_arr)
    # First distance vector defines N
    first_dist = np.asarray(distance_arr[0], dtype=float).ravel()
    N = first_dist.shape[0]

    # Initialize
    min_distance = np.full(N, np.inf, dtype=float)
    labels_arr = np.zeros(N, dtype=np.asarray(labels_arr_arr[0]).ravel().dtype)

    for i in range(Q):
        d_i = np.asarray(distance_arr[i], dtype=float).ravel()
        if d_i.shape[0] != N:
            raise ValueError("All distance arrays must have the same length.")

        # Element-wise minimum (like MATLAB min(min_distance, distance_arr{i}))
        new_min = np.minimum(min_distance, d_i)

        # Positions where this classifier attains the (updated) minimum
        mask = (d_i == new_min)

        # Update global minimum and labels at those positions
        min_distance = new_min
        labels_i = np.asarray(labels_arr_arr[i]).ravel()
        if labels_i.shape[0] != N:
            raise ValueError("All label arrays must have the same length.")

        labels_arr[mask] = labels_i[mask]

    return min_distance, labels_arr
