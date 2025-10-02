import numpy as np

def gradient_descent_OLS(X, y, eta=0.01, num_iters=1000, print_num_iters = False):
    """Perform gradient descent for OLS regression (fixed learning rate)

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the functin prints the number of iterations. Defaults to False.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
    """

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    for t in range(num_iters):
        # Compute gradients for OSL and Ridge
        grad_OLS = 2/n_samples * X.T @ (X @ theta - y)
        # Update parameters theta
        theta_new = theta - eta*grad_OLS
        # Stopping criterion:
        if np.allclose(theta, theta_new, rtol=1e-8, atol=1e-8):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new, t+1
        else: theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta, t+1

def gradient_descent_Ridge(X, y, eta=0.01, lam=1, num_iters=1000, print_num_iters = False):
    """Perform gradient descent for OLS regression (fixed learning rate)

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        lam (float, optional): Regularization parameter. Defaults to 1. 
        eta (float, optional): Learning rate. Defaults to 0.01.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the functin prints the number of iterations. Defaults to False.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
    """
    n_samples, n_features = X.shape
    #eta = eta/n_samples
    #Initialize theta
    theta = np.zeros(n_features)
    for t in range(num_iters):
        #grad = 2 * (1/n_samples * X.T @ (X @ theta - y) + lam*theta)
        # We drop the 1/n_samples term to get the same results as in closed form ridge
        grad = 2 * (X.T @ (X @ theta - y) + lam * theta)
        theta_new = theta - eta * grad
        # Stopping criterion:
        if np.allclose(theta, theta_new, rtol=1e-8, atol=1e-8):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new, t+1
        else: theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta, t+1



def momentum_gradient_descent_OLS(X, y, eta=0.01, momentum = 0.3 ,num_iters=1000, print_num_iters = False):
    """Gradient descent with momentum for OLS Regression

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        momentum (float, optional): _description_. Defaults to 0.3.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the functin prints the number of iterations. Defaults to False.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
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
        if np.allclose(theta, theta_new, rtol=1e-8, atol=1e-8):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new, t+1
        else: theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta, t+1
        
        
def momentum_gradient_descent_Ridge(X, y, eta=0.01, lam=1, momentum = 0.3, num_iters=1000, print_num_iters = False):
    """Gradient descent with momentum for Ridge Regression

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        lam (float, optional): Regularization parameter. Defaults to 1.
        momentum (float, optional): _description_. Defaults to 0.3.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the functin prints the number of iterations. Defaults to False.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
    """

    
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    change = np.zeros(n_features)
    for t in range(num_iters):
        # We drop the 1/n_samples term to get the same results as in closed form ridge
        grad_Ridge = 2 * (X.T @ (X @ theta - y) + lam * theta)
        # Compute new change
        new_change = eta*grad_Ridge + momentum*change
        # Update parameters theta
        theta_new = theta - new_change
        change = new_change
        
        # Stopping criterion:
        if np.allclose(theta, theta_new, rtol=1e-8, atol=1e-8):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new, t+1
        else: theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta, t+1

def ADAGrad_gradient_descent_OLS(X, y, eta=0.01 ,num_iters=1000, print_num_iters = False):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    r = np.zeros(n_features)
    eps = 1e-7
    
    for t in range(num_iters):
        grad = 2/n_samples * X.T @ (X @ theta - y)
        r += grad**2
        
        update = eta/(np.sqrt(r)+eps)*grad 
        
        theta_new = theta - update
        if np.allclose(theta, theta_new, rtol=1e-8, atol=1e-8):
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new, t+1
        else: theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta, t+1



def ADAGrad_gradient_descent_Ridge(
    X, y, eta=0.01, lam=1.0, num_iters=1000, print_num_iters=False,
    tol_theta=1e-8
):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features, dtype=float)
    r = np.zeros(n_features, dtype=float)
    eps = 1e-7

    for t in range(num_iters):
        grad = 2.0 * (X.T @ (X @ theta - y) + lam * theta)

        r += grad * grad
        step = eta * grad / (np.sqrt(r) + eps)
        theta_new = theta - step
        
        if np.linalg.norm(theta_new - theta) <= tol_theta * (1.0 + np.linalg.norm(theta)):
            if print_num_iters:
                print("Number of iterations:", t+1, "(small parameter change)")
            return theta_new, t+1

        theta = theta_new

    if print_num_iters:
        print("Number of iterations:", t+1)
    return theta, t+1

import numpy as np

def RMSProp_gradient_descent_OLS(
    X, y,
    eta=1e-3, rho=0.99, num_iters=50000,
    eps=1e-8,
    tol_grad=1e-5,            # 1) ||grad||_inf <= tol_grad
    tol_step=1e-8,            # 2) ||step||_2 <= tol_step * (1 + ||theta||_2)
    tol_rel_loss=1e-9,        # 3) |J_t - J_{t-1}| / J_{t-1} <= tol_rel_loss
    print_num_iters=False
):
    """
    RMSProp for OLS: min (1/n)||y - Xθ||^2
    Gradient: (2/n) X^T (Xθ - y)
    """
    n, p = X.shape
    theta = np.zeros(p, dtype=float)
    v = np.zeros(p, dtype=float)

    def obj(th):
        r = X @ th - y
        return (r @ r) / n

    J_prev = obj(theta)

    for t in range(1, num_iters + 1):
        # grad
        r = X @ theta - y
        grad = (2.0 / n) * (X.T @ r)

        # 1) liten gradient
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters: print("Stopper ved liten gradient etter", t, "iterasjoner")
            return theta, t

        # RMSProp-oppdatering
        v = rho * v + (1.0 - rho) * (grad * grad)
        step = eta * grad / (np.sqrt(v) + eps)
        theta_new = theta - step

        if not np.all(np.isfinite(theta_new)):
            raise FloatingPointError(f"Divergens (NaN/Inf) ved iterasjon {t}. Reduser eta/rho eller skaler X.")

        # 2) lite steg
        if np.linalg.norm(step) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters: print("Stopper ved lite steg etter", t, "iterasjoner")
            return theta_new, t

        # 3) liten relativ tap-endring
        J = obj(theta_new)
        if abs(J - J_prev) / (J_prev + 1e-12) <= tol_rel_loss:
            if print_num_iters: print("Stopper ved liten relativ tap-endring etter", t, "iterasjoner")
            return theta_new, t

        theta, J_prev = theta_new, J

    if print_num_iters:
        print("Nådde max iterasjoner:", num_iters)
    return theta, num_iters


"""
def RMSProp_gradient_descent_OLS(X, y, eta=0.01, rho=0.99 ,num_iters=1000, print_num_iters = False):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    v = np.zeros(n_features)
    eps = 1e-8
    tol = 1e-4
    
    for t in range(num_iters):
        grad = 2/n_samples * X.T @ (X @ theta - y)
        v = rho*v + (1-rho)*grad**2
        
        adjusted_grad = eta/(np.sqrt(v)+eps)*grad 

        theta_new = theta - adjusted_grad
        if np.linalg.norm(theta_new - theta, ord=np.inf) < tol:
            if print_num_iters:
               print("Number of iterations: ", t+1)
            return theta_new, t+1
            
        theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta, t+1
"""

def RMSProp_gradient_descent_Ridge(X, y, lam=1, eta=0.01, rho=0.9 ,num_iters=1000, print_num_iters = False):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    v = np.zeros(n_features)
    eps = 1e-8
    
    for t in range(num_iters):
        grad = 2 * (X.T @ (X @ theta - y) + lam * theta)
        v = rho*v + (1-rho)*grad**2
        
        adjusted_grad = eta/(np.sqrt(v)+eps)*grad 
        
        theta_new = theta - adjusted_grad
        if np.linalg.norm(theta_new - theta, ord=np.inf) < 1e-4:
            if print_num_iters:
                print("Number of iterations: ", t+1)
            return theta_new
        else: theta = theta_new
    if print_num_iters:
        print("Number of iterations: ", t+1)
    return theta

def ADAM_gradient_descent_OLS(X, y, eta=0.01, beta_1 = 0.9, beta_2 = 0.999 ,num_iters=1000, print_num_iters = False):
    pass