import numpy as np
from .autoencoder_filtering_extended import autoencoder_filtering_extended
# ^ Python version of AutoencoderFilteringExtended.m


def output_parallel_autoencoders_extended(y_data_new, AE_arr, distance_type):
    """
    Python version of:
        function [hat_y_data] = outputParallelAutoencodersExtended(y_data_new, AE_arr, distance_type)

    Parameters
    ----------
    y_data_new : ndarray, shape (D, N)
        New data samples (columns are samples).
    AE_arr : list
        List of autoencoder objects (e.g. dicts from `autoencoder()`).
    distance_type : str
        Distance type passed to `autoencoder_filtering_extended`.

    Returns
    -------
    hat_y_data : ndarray, shape (D, N)
        For each sample, the reconstruction from the autoencoder that
        gives the minimal distance (over AE_arr).
    """
    y_data_new = np.asarray(y_data_new, dtype=float)
    D, N = y_data_new.shape
    Q = len(AE_arr)

    distance_matrix = np.zeros((Q, N), dtype=float)
    hat_y_data_list = [None] * Q

    # Run all autoencoders
    for i in range(Q):
        hat_y_i, dist_i = autoencoder_filtering_extended(
            y_data_new, AE_arr[i], distance_type
        )
        hat_y_data_list[i] = hat_y_i          # shape (D, N)
        distance_matrix[i, :] = dist_i        # shape (N,)

    # Minimum distance per sample
    distance_min = np.min(distance_matrix, axis=0)  # (N,)

    # Initialize hat_y_data with zeros like first reconstruction
    hat_y_data = np.zeros_like(hat_y_data_list[0])

    # For each autoencoder, copy reconstructions where it attains the min
    for i in range(Q):
        mask = distance_matrix[i, :] == distance_min  # (N,)
        hat_y_data[:, mask] = hat_y_data_list[i][:, mask]

    return hat_y_data
