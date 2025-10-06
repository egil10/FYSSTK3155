import numpy as np

def OLS_parameters(X, y):
    """
    Compute closed-form regression parameters for OLS regression.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target
        
    Returns:
        beta (numpy.ndarray): Optimal (closed-form) OLS parameters
    """

    return np.linalg.pinv(X.T @ X) @ X.T @ y
