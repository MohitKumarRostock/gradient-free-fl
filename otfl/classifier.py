import numpy as np
from joblib import Parallel, delayed

from .parallel_autoencoders import parallel_autoencoders


def _build_autoencoders_for_label(lbl, y_data, label_data, subspace_dim, Nb, verbose):
    mask = (label_data == lbl)
    y_class = y_data[:, mask]

    if y_class.shape[1] == 0:
        # no samples for this label in this client
        if verbose:
            print(f"Skipping class {lbl}: no samples.")
        return None

    if verbose:
        print(f"Building an autoencoder for class = {lbl}...")

    AE_arr = parallel_autoencoders(y_class, subspace_dim, Nb)
    return AE_arr


def classifier(y_data, label_data, subspace_dim, Nb, n_jobs=1, verbose=True):
    """
    Build one (parallel) autoencoder model per label.

    Parameters
    ----------
    y_data : ndarray, shape (D, N)
    label_data : array_like, shape (N,)
    subspace_dim : int
    Nb : int
        Block size / cluster parameter for parallel_autoencoders.
    n_jobs : int, optional
        Number of parallel jobs (CPU cores). Use -1 for "all cores".
    verbose : bool
        Print progress messages.
    """
    y_data = np.ascontiguousarray(y_data, dtype=float)
    label_data = np.asarray(label_data).ravel()

    D, N = y_data.shape
    if label_data.size != N:
        raise ValueError(
            f"label_data length ({label_data.size}) must match number of samples ({N})."
        )

    labels = np.unique(label_data)

    if n_jobs == 1:
        # serial (original behaviour)
        AE_arr_arr = []
        for lbl in labels:
            AE_arr = _build_autoencoders_for_label(
                lbl, y_data, label_data, subspace_dim, Nb, verbose
            )
            AE_arr_arr.append(AE_arr)
    else:
        # parallel over labels
        results = Parallel(n_jobs=n_jobs)(
            delayed(_build_autoencoders_for_label)(
                lbl, y_data, label_data, subspace_dim, Nb, verbose
            )
            for lbl in labels
        )
        AE_arr_arr = results

    CLF = {
        "labels": labels,
        "AE_arr_arr": AE_arr_arr,
    }

    return CLF
