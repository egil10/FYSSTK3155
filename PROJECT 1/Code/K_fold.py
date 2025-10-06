import numpy as np

def k_fold_split(X, y, k=5, shuffle=True, random_state=None):
    """
    Split data into k folds for cross-validation.

    Args:
        X : array-like, shape (n_samples, n_features) or (n_samples,)
            Feature matrix or 1D input array.
        y : array-like, shape (n_samples,)
            Target values.
        k : int, optional (default=5)
            Number of folds.
        shuffle : bool, optional (default=True)
            Whether to shuffle data before splitting.
        random_state : int or None, optional
            Random seed for reproducibility.

    Returns:
        folds : list of (train_idx, val_idx)
            A list where each element contains the indices for one fold’s
            training and validation subsets.
    """

    n_samples = len(X)
    indices = np.arange(n_samples)

    # Shuffle for random fold assignment
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Compute fold sizes (handle uneven splits)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1  # distribute remainder

    # Build index splits
    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_idx, val_idx))
        current = stop

    return folds