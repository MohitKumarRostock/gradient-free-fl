import numpy as np


def dirichlet_sample(alpha_vec):
    """
    Python version of:
        function [p] = dirichlet_sample(alph_vec)
    """
    alpha_vec = np.asarray(alpha_vec, dtype=float)
    g = np.random.gamma(shape=alpha_vec, scale=1.0)
    return g / g.sum()


def non_iid_label_distribution(alph, labels_arr, C, Q, verbose=True):
    """
    Python version of:
        function [client_indices] = non_iid_label_distribution(alph,labels_arr,C,Q)

    Parameters
    ----------
    alph : float
        Dirichlet concentration parameter.
    labels_arr : array_like, shape (N,)
        Label for each sample (expected to be in {1, 2, ..., C} as in MATLAB).
    C : int
        Number of classes.
    Q : int
        Number of clients.
    verbose : bool, optional
        If True, prints per-client stats like the MATLAB code.

    Returns
    -------
    client_indices : list of ndarray
        client_indices[q] contains the (0-based) indices of samples assigned
        to client q.
    """
    labels_arr = np.asarray(labels_arr).ravel()
    N = labels_arr.size

    # Initialize client index lists
    client_indices = [[] for _ in range(Q)]

    # For each class
    for i in range(1, C + 1):
        # idx_i = find(labels_arr == i);
        idx_i = np.where(labels_arr == i)[0]
        ni = idx_i.size
        if ni == 0:
            continue

        # q_i = dirichlet_sample(alph*ones(1,Q));
        q_i = dirichlet_sample(alph * np.ones(Q))

        # counts = mnrnd(ni,q_i);
        counts = np.random.multinomial(ni, q_i)

        # idx_i = idx_i(randperm(ni));
        idx_i = np.random.permutation(idx_i)

        s = 0
        for q in range(Q):
            c = counts[q]
            if c > 0:
                client_indices[q].extend(idx_i[s:s + c].tolist())
                s += c

    # Convert lists to numpy arrays and print stats
    for q in range(Q):
        client_indices[q] = np.array(client_indices[q], dtype=int)
        if verbose:
            print(f"Client {q + 1}: N={client_indices[q].size}")
            if client_indices[q].size > 0:
                # counts = histcounts(labels_arr(client_indices{q}),1:(C+1));
                counts, _ = np.histogram(labels_arr[client_indices[q]],
                                         bins=np.arange(1, C + 2))
            else:
                counts = np.zeros(C, dtype=int)
            print(counts)

    return client_indices
