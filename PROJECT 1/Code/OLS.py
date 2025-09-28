import numpy as np

def fit_ols(X_train, y_train):
    """
    Fit an Ordinary Least Squares (OLS) regression model from scratch
    using the closed-form normal equation.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Training input features (without bias term).
    y_train : ndarray of shape (n_samples,)
        Training target values.
    """
    # Add intercept column (bias term)
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]

    return np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

