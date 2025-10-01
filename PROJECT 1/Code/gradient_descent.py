import numpy as np

def gradient_descent_OLS(X, y, eta=0.01, num_iters=1000, print_num_iters = False):
    """
    Perform gradient descent to estimate coefficients for Ordinary Least Squares (OLS) regression.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        The input feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        The target values.
    eta : float, default=0.01
        Learning rate that controls the step size in the gradient descent updates.
    num_iters : int, default=1000
        Number of iterations to run the gradient descent algorithm.
    print_num_iters : bool, default=False
        If True, prints the actual number of iterations taken before convergence.

    Returns
    -------
    theta : numpy.ndarray, shape (n_features,)
        Estimated regression coefficients after gradient descent optimization.
    """

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    for t in range(num_iters):
        # Compute gradients for OSL and Ridge
        grad_OLS = 2/n_samples * X.T @ (X @ theta - y)
        # Update parameters theta
        theta_new = theta - eta*grad_OLS
        # Stopping criterion:
        if np.allclose(theta, theta_new, rtol=1e-12, atol=1e-12):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new
        else: theta = theta_new
    return theta

def gradient_descent_Ridge(X, y, eta=0.01, lam=1, num_iters=1000, print_num_iters = False):
    """
    Perform gradient descent to estimate coefficients for Ridge regression.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        The input feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        The target values.
    eta : float, default=0.01
        Learning rate that controls the step size in the gradient descent updates.
    lam : float, default=1
        Regularization strength (λ). Larger values shrink coefficients more strongly.
    num_iters : int, default=1000
        Maximum number of iterations for gradient descent.
    print_num_iters : bool, default=False
        If True, prints the actual number of iterations taken before convergence.

    Returns
    -------
    theta : numpy.ndarray, shape (n_features,)
        Estimated regression coefficients after Ridge regularized gradient descent.
    """
    n_samples, n_features = X.shape
    #Initialize theta
    theta = np.zeros(n_features)
    for t in range(num_iters):
        grad = 2 * (1/n_samples * X.T @ (X @ theta - y) + lam*theta)
        theta_new = theta - eta * grad
        # Stopping criterion:
        if np.allclose(theta, theta_new, rtol=1e-12, atol=1e-12):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new
        else: theta = theta_new
    return theta



def momentum_gradient_descent_OLS(X, y, eta=0.01, momentum = 0.3 ,num_iters=1000, print_num_iters = False):
    """
    Perform gradient descent with momentum to estimate coefficients for Ordinary Least Squares (OLS) regression.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        The input feature matrix.
    y : numpy.ndarray, shape (n_samples,)
        The target values.
    eta : float, default=0.01
        Learning rate that controls the step size in the gradient descent updates.
    num_iters : int, default=1000
        Number of iterations to run the gradient descent algorithm.
    print_num_iters : bool, default=False
        If True, prints the actual number of iterations taken before convergence.

    Returns
    -------
    theta : numpy.ndarray, shape (n_features,)
        Estimated regression coefficients after gradient descent optimization.
    """

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    change = np.zeros(n_features)
    for t in range(num_iters):
        # Compute gradients for OLS
        grad_OLS = 2/n_samples * X.T @ (X @ theta - y)
        # Compute new change
        new_change = eta*grad_OLS + momentum*change
        # Update parameters theta
        theta_new = theta - new_change
        change = new_change
        
        # Stopping criterion:
        if np.allclose(theta, theta_new, rtol=1e-12, atol=1e-12):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new
        else: theta = theta_new
    return theta
        
        
def momentum_gradient_descent_Ridge():
    pass