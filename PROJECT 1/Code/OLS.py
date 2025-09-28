import numpy as np

def fit_ols(X_train, y_train, X_test, y_test):
    """
    Fit an Ordinary Least Squares (OLS) regression model from scratch
    using the closed-form normal equation.

    Parameters
    ----------
    X_train : ndarray of shape (n_samples, n_features)
        Training input features (without bias term).
    y_train : ndarray of shape (n_samples,)
        Training target values.
    X_test : ndarray of shape (m_samples, n_features)
        Test input features (without bias term).
    y_test : ndarray of shape (m_samples,)
        Test target values.

    Returns
    -------
    beta_hat : ndarray of shape (n_features + 1,)
        Estimated regression coefficients (including intercept).
    y_pred : ndarray of shape (m_samples,)
        Predictions for the test data.
    metrics : dict
        Dictionary containing evaluation metrics:
        - "r2" : float, R² score on the test set
        - "mse" : float, Mean Squared Error on the test set
    """
    # Add intercept column (bias term)
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    
    return np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

