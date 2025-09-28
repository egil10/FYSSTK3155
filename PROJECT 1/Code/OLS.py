import numpy as np

def OLS_parameters(X, y):
    """
    Compute Ordinary Least Squares (OLS) regression coefficients 
    using the closed-form solution.

    This function solves the unpenalized least squares problem

        β̂_OLS = (XᵀX)⁻¹ Xᵀy

    where X is the design matrix (including an intercept column if desired) 
    and y is the response vector. OLS provides unbiased estimates of the 
    regression coefficients under the Gauss–Markov assumptions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The design matrix containing input features. 
        If an intercept is required, a column of ones must be added manually.
    y : ndarray of shape (n_samples,)
        The target values.

    Returns
    -------
    beta : ndarray of shape (n_features,)
        The estimated OLS regression coefficients.
    
    Notes
    -----
    - If XᵀX is not invertible (e.g. multicollinearity), consider using the 
      Moore–Penrose pseudoinverse via `np.linalg.pinv(X) @ y`.
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ y
