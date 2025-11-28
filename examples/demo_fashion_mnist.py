# examples/demo_fashion_mnist.py

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

from otfl.non_iid_label_distribution import non_iid_label_distribution
from otfl.classifier import classifier
from otfl.prediction_classifier_extended import prediction_classifier_extended
from otfl.combine_multiple_classifiers import combine_multiple_classifiers


def main(
    Q=100,
    alph=5,
    subspace_dim=20,
    Nb=1000,
    use_subset=False,
    subset_train=10000,
    subset_test=2000,
):
    """
    Demo for OTFL library on Fashion-MNIST using the standard split:
    60 000 train / 10 000 test samples from keras.datasets.fashion_mnist.

    Parameters
    ----------
    Q : int
        Number of clients.
    alph : float
        Dirichlet concentration for non-IID label distribution.
    subspace_dim : int
        Subspace dimension for the autoencoders.
    Nb : int
        Block size for parallel_autoencoders.
    use_subset : bool
        If True, use only a subset of the standard train/test for speed.
    subset_train : int
        Number of training samples to keep if use_subset=True.
    subset_test : int
        Number of test samples to keep if use_subset=True.
    """
    np.random.seed(0)

    # ------------------------------------------------------------
    # 1. Load Fashion-MNIST from Keras (standard 60k / 10k split)
    # ------------------------------------------------------------
    print("Loading Fashion-MNIST via keras.datasets.fashion_mnist...")
    (X_train, y_train_raw), (X_test, y_test_raw) = fashion_mnist.load_data()
    # X_train: (60000, 28, 28), y_train_raw: (60000,)
    # X_test : (10000, 28, 28), y_test_raw : (10000,)

    # Optional: use only subsets for speed
    if use_subset:
        subset_train = min(subset_train, X_train.shape[0])
        subset_test = min(subset_test, X_test.shape[0])
        X_train = X_train[:subset_train]
        y_train_raw = y_train_raw[:subset_train]
        X_test = X_test[:subset_test]
        y_test_raw = y_test_raw[:subset_test]

    # Flatten images to 784-dim vectors and normalize to [0, 1]
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

    # Our library expects:
    # - data as (D, N)
    # - labels in {1, ..., C}
    C = 10
    XTrain = X_train.T                # (D, N_train)
    XTest = X_test.T                  # (D, N_test)
    YTrain = (y_train_raw.astype(int) + 1)  # 1..10
    YTest = (y_test_raw.astype(int) + 1)    # 1..10

    print(f"Using {XTrain.shape[1]} train samples and {XTest.shape[1]} test samples.")
    print(f"{C} classes, {Q} clients.")

    # ------------------------------------------------------------
    # 2. Non-IID label distribution across clients
    # ------------------------------------------------------------
    client_indices = non_iid_label_distribution(alph, YTrain, C, Q)

    # ------------------------------------------------------------
    # 3. Train local classifiers and collect their predictions
    # ------------------------------------------------------------
    min_distance_arr = [None] * Q
    labels_arr_arr = [None] * Q

    for i in range(Q):
        idx = client_indices[i]
        if idx.size == 0:
            print(f"Client {i + 1} has no samples after partitioning, skipping.")
            continue

        client_labels = YTrain[idx]
        counts, edges = np.histogram(client_labels, bins=np.arange(1, C + 2))

        # classes with single sample
        classes_with_single_sample = edges[:-1][counts < 2]

        # remove samples of singleton classes
        to_remove = np.isin(client_labels, classes_with_single_sample)
        idx = idx[~to_remove]
        client_indices[i] = idx

        if idx.size == 0:
            print(f"Client {i + 1} lost all samples after removing singletons, skipping.")
            continue

        # Train classifier on this client's local data
        CLF = classifier(
            XTrain[:, idx],
            YTrain[idx],
            subspace_dim=subspace_dim,
            Nb=Nb,
            n_jobs=-1,      # parallel over labels
            verbose=False,
        )

        # Get predictions on the global test set
        min_dist, labels_pred = prediction_classifier_extended(
            XTest, CLF, "folding"
        )
        min_distance_arr[i] = min_dist
        labels_arr_arr[i] = labels_pred

        print(f"processed data of client {i + 1}.")

    # ------------------------------------------------------------
    # 4. Combine classifiers and compute global accuracy
    # ------------------------------------------------------------
    valid_min = [d for d in min_distance_arr if d is not None]
    valid_labels = [l for l in labels_arr_arr if l is not None]

    _, hat_labels_test = combine_multiple_classifiers(valid_min, valid_labels)
    acc = np.mean(hat_labels_test == YTest)

    print(f"global accuracy on Fashion-MNIST = {acc:.6f}.")
    return acc, hat_labels_test


if __name__ == "__main__":
    # For quick experiments, you can set use_subset=True.
    # For full 60k/10k, set use_subset=False.
    main(alph=0.1, Nb = 100, use_subset=False)
