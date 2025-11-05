import numpy as np

def Ridge_parameters(X, y,lambda_=1):
    """
    Compute Ridge regression coefficients using the closed-form solution.

    This function solves the penalized least squares problem

        β̂_ridge = (XᵀX + λI)⁻¹ Xᵀy

    where X is the design matrix, y the response vector, λ ≥ 0 the 
    regularization parameter, and I the identity matrix. The ridge penalty
    shrinks coefficients to reduce variance and avoid multicollinearity issues
    in linear regression :contentReference[oaicite:0]{index=0}.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The design matrix containing input features.
    y : ndarray of shape (n_samples,)
        The target values.
    lambda_ : float, default=1
        Regularization strength λ. A higher value applies stronger shrinkage.

    Returns
    -------
    beta : ndarray of shape (n_features,)
        The estimated Ridge regression coefficients.
    
    """
    
    return np.linalg.pinv(X.T @ X+np.identity(X.shape[1])*lambda_) @ X.T @ y