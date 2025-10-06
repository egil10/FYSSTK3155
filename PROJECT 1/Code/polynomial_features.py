import numpy as np
def polynomial_features(x, p, intercept=False):
    """
    Generate a polynomial feature matrix from a 1D input array.

    Args:
        x : numpy.ndarray, shape (n_samples,)
            Input vector of samples.
        p : int
            Degree of the polynomial features to generate.
        intercept : bool, default=False
            If True, includes an intercept (bias) column of ones as the first feature.
            If False, only powers of `x` from 1 to `p` are included.

    Returns:

        X : numpy.ndarray, shape (n_samples, p+1) if intercept=True, else (n_samples, p)
            The polynomial feature matrix, where each column corresponds to 
            increasing powers of `x`.
    """

    n = len(x)
    # Condition on intercept:
    if intercept==True:
        X = np.zeros((n, p + 1))
        for i in range(p+1):
            X[:, i] = x ** i
            # First column becomes ones because for all x, x**i = 1 when i=0.
    else: 
        X = np.zeros((n, p))
        for i in range(p):
            X[:, i] = x ** (i+1)
    
    return X 