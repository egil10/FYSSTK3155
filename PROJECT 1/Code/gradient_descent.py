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
    """Perform gradient descent for Ridge regression (fixed learning rate)

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
        momentum (float, optional): Momentum parameter. Defaults to 0.3.
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
        momentum (float, optional): Momentum parameter. Defaults to 0.3.
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
    """Gradient descent with ADAGrad, OLS

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target data
        eta (float, optional): Learning rate. Defaults to 0.01.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, prints the number of iterations before convergence. Defaults to False.

    Returns:
        theta (numpy.ndarray): OLS parameters
        t (int): number of iterations 
    """
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
    tol_theta=1e-8):
    """ADAGrad gradient descent for Ridge regression

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate.  Defaults to 0.01.
        lam (float, optional): Regularization parameter. Defaults to 1.0.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the function prints the number of iterations before convergence. Defaults to False.
        tol_theta (float, optional): tolerance for checking convergence. Defaults to 1e-8.

    Returns:
        theta (numpy.ndarray): Ridge parameters
        t (int): number of iterations
    """
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

def RMSProp_gradient_descent_OLS(
    X, y,
    eta=1e-3, rho=0.99, num_iters=50000,
    eps=1e-8,
    tol_grad=1e-5,            
    tol_step=1e-8,            
    tol_rel_loss=1e-9,        
    print_num_iters=False):
    """RMSProp gradient descent, for OLS regression

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 1e-3.
        rho (float, optional): Defaults to 0.99.
        num_iters (int, optional): Number of iterations. Defaults to 50000.
        eps (_type_, optional): term to avoid division by zero. Defaults to 1e-8.
        tol_grad (_type_, optional): Tolerance for checking convergence, gradient. Defaults to 1e-5.
        tol_step (_type_, optional): Tolerance. Defaults to 1e-8.
        tol_rel_loss (_type_, optional): Tolerance. Defaults to 1e-9.
        print_num_iters (bool, optional): If true, the function prints the number of iterations before convergence. Defaults to False.

    Returns:
        theta (numpy.ndarray): OLS parameters
        t (int): Number of iterations
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

        # 1) Small gradient
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters: print("Number of iterations: ", t)
            return theta, t

        # RMSProp-update
        v = rho * v + (1.0 - rho) * (grad * grad)
        step = eta * grad / (np.sqrt(v) + eps)
        theta_new = theta - step


        # 2) lite steg
        if np.linalg.norm(step) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters: print("Number of iterations: ", t)
            return theta_new, t

        
        # 3) liten relativ tap-endring
        J = obj(theta_new)
        if abs(J - J_prev) / (J_prev + 1e-12) <= tol_rel_loss:
            if print_num_iters: print("Number of iterations: ", t)
            return theta_new, t
        theta, J_prev = theta_new, J

    if print_num_iters:
        print("Reached maximum iterations:", num_iters)
    return theta, num_iters

def RMSProp_gradient_descent_Ridge(
    X, y, lam=1.0,
    eta=1e-3, rho=0.99, num_iters=50000,
    eps=1e-8,
    tol_grad=1e-5,            
    tol_step=1e-8,            
    tol_rel_loss=1e-9,        
    print_num_iters=False
):
    """RMSProp gradient descent, Ridge regression

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        lam (float, optional): Regularization parameter. Defaults to 1.
        eta (float, optional): Learning rate. Defaults to 1e-3.
        rho (float, optional): Defaults to 0.99.
        num_iters (int, optional): Number of iterations. Defaults to 50000.
        eps (_type_, optional): term to avoid division by zero. Defaults to 1e-8.
        tol_grad (_type_, optional): Tolerance for checking convergence, gradient. Defaults to 1e-5.
        tol_step (_type_, optional): Tolerance. Defaults to 1e-8.
        tol_rel_loss (_type_, optional): Tolerance. Defaults to 1e-9.
        print_num_iters (bool, optional): If true, the function prints the number of iterations before convergence. Defaults to False.

    Returns:
        theta (numpy.ndarray): OLS parameters
        t (int): Number of iterations
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
        grad = 2.0 * (X.T @ (X @ theta - y) + lam * theta)

        # 1) Small gradient
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters: print("Number of iterations: ", t)
            return theta, t

        # RMSProp-update
        v = rho * v + (1.0 - rho) * (grad * grad)
        step = eta * grad / (np.sqrt(v) + eps)
        theta_new = theta - step


        # 2) lite steg
        if np.linalg.norm(step) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters: print("Number of iterations: ", t)
            return theta_new, t

        
        # 3) liten relativ tap-endring
        J = obj(theta_new)
        if abs(J - J_prev) / (J_prev + 1e-12) <= tol_rel_loss:
            if print_num_iters: print("Number of iterations: ", t)
            return theta_new, t
        theta, J_prev = theta_new, J

    if print_num_iters:
        print("Reached maximum iterations:", num_iters)
    return theta, num_iters


def ADAM_gradient_descent_OLS(X, y, eta=0.01, rho_1 = 0.9, rho_2 = 0.999, num_iters=1000, print_num_iters = False):
    """Gradient Descent with ADAM, OLS Regression.

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        rho1 (float, optional): Decay parameter 1. Defaults to 0.9.
        rho_2 (float, optional): Decay parameter 2. Defaults to 0.999.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the function prints the number of iterations before convergence. Defaults to False.
    """
    eps = 1e-8 # Small constant for numerical stability
    n, p = X.shape
    
    s = np.zeros(p)
    r = np.zeros(p)
    theta = np.zeros(p, dtype=float)
    tol_grad = 1e-6
    tol_step = 1e-8
    
    for t in range (1, num_iters+1):
        grad = 2/n * X.T @ (X @ theta - y)
        
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters:
                print("Stop: small gradient at", t)
            return theta, t
        
        s = rho_1*s + (1-rho_1)*grad
        r = rho_2*r + (1-rho_2)*grad**2
        
        s_unbiased = s/(1-rho_1**t)
        r_unbiased = r/(1-rho_2**t)
        
        update = eta * (s_unbiased/(np.sqrt(r_unbiased)+eps))
        
        theta_new = theta - update
        
        if np.linalg.norm(update) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters:
                print("Stop: small step at", t)
            return theta_new, t
        theta = theta_new
    
    if print_num_iters:
        print("Reached maximum iterations: ", t)
    
    return theta, num_iters

def ADAM_gradient_descent_Ridge(X, y, eta=0.01, lam = 1.0, rho_1 = 0.9, rho_2 = 0.999, num_iters=1000, print_num_iters = False):
    """Gradient descent with ADAM, Ridge regression.

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        lam (float, optional): Regularization parameter. Defaults to 1.0
        rho1 (float, optional): Decay parameter 1. Defaults to 0.9.
        rho_2 (float, optional): Decay parameter 2. Defaults to 0.999.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the function prints the number of iterations before convergence. Defaults to False.
    """
    eps = 1e-8 # Small constant for numerical stability
    n, p = X.shape
    
    s = np.zeros(p)
    r = np.zeros(p)
    theta = np.zeros(p, dtype=float)
    tol_grad = 1e-6
    tol_step = 1e-8
    
    for t in range (1, num_iters+1):
        grad = 2 * (X.T @ (X @ theta - y) + lam * theta)
        
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters:
                print("Stop: small gradient at", t)
            return theta, t
        
        s = rho_1*s + (1-rho_1)*grad
        r = rho_2*r + (1-rho_2)*grad**2
        
        s_unbiased = s/(1-rho_1**t)
        r_unbiased = r/(1-rho_2**t)
        
        update = eta * (s_unbiased/(np.sqrt(r_unbiased)+eps))
        
        theta_new = theta - update
        
        if np.linalg.norm(update) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters:
                print("Stop: small step at", t)
            return theta_new, t
        theta = theta_new
    
    if print_num_iters:
        print("Reached maximum iterations: ", t)
    
    return theta, num_iters

### LASSO ###

def soft_threshold(z, alpha):
    """Soft-thresholding operator, used for gradient descent with LASSO"""
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)


def gradient_descent_LASSO(X, y, eta=0.01, lam=1.0, num_iters=1000, print_num_iters=False):
    """Perform ISTA (proximal gradient descent) for LASSO regression

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        lam (float, optional): Regularization parameter. Defaults to 1. 
        eta (float, optional): Learning rate. Defaults to 0.01.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, prints number of iterations. Defaults to False.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
    """
    
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    for t in range(num_iters):
        # Gradient step (OLS part only)
        grad = (1/n_samples) * X.T @ (X @ theta - y)
        z = theta - eta * grad

        # Proximal step (soft-thresholding for L1)
        theta_new = soft_threshold(z, eta * lam)

        # Stopping criterion
        if np.allclose(theta, theta_new, rtol=1e-8, atol=1e-8):
            if print_num_iters:
                print("Number of iterations:", t+1)
            return theta_new, t+1
        else:
            theta = theta_new

    if print_num_iters:
        print("Number of iterations:", t+1)
    return theta, t+1


def momentum_gradient_descent_LASSO(X, y, eta=0.01, lam=1.0, momentum=0.3,
                                    num_iters=1000, print_num_iters=False):
    """Gradient descent with momentum for LASSO regression (proximal version)

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        lam (float, optional): Regularization parameter (lambda). Defaults to 1.0.
        momentum (float, optional): Momentum coefficient. Defaults to 0.3.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, the function prints the number of iterations.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
    """

    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    change = np.zeros(n_features)
    
    def lasso_cost(X, y, theta, lam):
        n = X.shape[0]
        return (1/(2*n)) * np.linalg.norm(y - X @ theta)**2 + lam * np.linalg.norm(theta, 1)

    tol = 1e-8
    prev_cost = np.inf
    
    for t in range(num_iters):
        # Gradient of OLS part only
        grad_OLS = (1/n_samples) * X.T @ (X @ theta - y)

        # Momentum update
        new_change = eta * grad_OLS + momentum * change
        z = theta - new_change

        # Proximal step (soft-thresholding for L1 penalty)
        theta_new = soft_threshold(z, eta * lam)
        # Update momentum buffer
        change = new_change

        cost = lasso_cost(X, y, theta_new, lam)
        
        if abs(prev_cost - cost) < tol:
            if print_num_iters:
                print(f"Converged after {t+1} iterations with cost {cost:.6f}")
            return theta_new, t+1
        
        theta = theta_new
        prev_cost = cost

    if print_num_iters:
        print("Number of iterations:", num_iters)
    return theta, num_iters


def ADAGrad_gradient_descent_LASSO(
    X, y, eta=0.01, lam=1.0, num_iters=1000, print_num_iters=False,
    tol_theta=1e-8):
    """ADAGrad gradient descent, LASSO (proximal version)

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        lam (float, optional): Regularization parameter (lambda). Defaults to 1.0.
        num_iters (int, optional): Number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, prints iteration count.
        tol_theta (float, optional): tolerance for checking convergence. Defaults to 1e-8.

    Returns:
        theta (numpy.ndarray): LASSO parameters
        t (int): number of iterations
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features, dtype=float)
    r = np.zeros(n_features, dtype=float)
    eps = 1e-7

    for t in range(num_iters):
        # Gradient of OLS part only
        grad = (1.0 / n_samples) * X.T @ (X @ theta - y)

        # Update squared gradient accumulator
        r += grad * grad

        # Compute adaptive step
        step = eta * grad / (np.sqrt(r) + eps)
        z = theta - step

        # Proximal step (soft-thresholding for L1 penalty)
        theta_new = soft_threshold(z, eta * lam / (np.sqrt(r) + eps))

        # Check convergence (parameter change small)
        if np.linalg.norm(theta_new - theta) <= tol_theta * (1.0 + np.linalg.norm(theta)):
            if print_num_iters:
                print("Number of iterations:", t+1, "(small parameter change)")
            return theta_new, t+1

        theta = theta_new

    if print_num_iters:
        print("Number of iterations:", num_iters)
    return theta, num_iters


def RMSProp_gradient_descent_LASSO(
    X, y, lam=1.0,
    eta=1e-3, rho=0.99, num_iters=50000,
    eps=1e-8,
    tol_grad=1e-5,            
    tol_step=1e-8,            
    tol_rel_loss=1e-9,        
    print_num_iters=False
):
    """RMSProp gradient descent, LASSO (proximal version with soft-thresholding)

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        lam (float, optional): Regularization parameter (lambda). Defaults to 1.0.
        eta (float, optional): Learning rate. Defaults to 1e-3.
        rho (float, optional): Decay rate for moving average. Defaults to 0.99.
        num_iters (int, optional): Maximum number of iterations. Defaults to 50000.
        eps (float, optional): Small term to avoid division by zero. Defaults to 1e-8.
        tol_grad (float, optional): Gradient tolerance (sup-norm). Defaults to 1e-5.
        tol_step (float, optional): Step-size tolerance. Defaults to 1e-8.
        tol_rel_loss (float, optional): Relative loss change tolerance. Defaults to 1e-9.
        print_num_iters (bool, optional): Print number of iterations before convergence. Defaults to False.

    Returns:
        theta (numpy.ndarray): LASSO parameters
        t (int): Number of iterations
    """
    n, p = X.shape
    theta = np.zeros(p, dtype=float)
    v = np.zeros(p, dtype=float)

    # LASSO cost function
    def obj(th):
        r = y - X @ th
        return (0.5 / n) * (r @ r) + lam * np.linalg.norm(th, 1)

    J_prev = obj(theta)

    for t in range(1, num_iters + 1):
        # Gradient of OLS part only
        grad = (1.0 / n) * X.T @ (X @ theta - y)

        # 1) Small gradient (OLS part only)
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters:
                print("Number of iterations: ", t, "(small gradient)")
            return theta, t

        # RMSProp update of squared gradient accumulator
        v = rho * v + (1.0 - rho) * (grad * grad)
        step = eta * grad / (np.sqrt(v) + eps)

        # Proximal step (soft-thresholding for L1 penalty)
        z = theta - step
        theta_new = soft_threshold(z, eta * lam / (np.sqrt(v) + eps))

        # 2) Small step
        if np.linalg.norm(theta_new - theta) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters:
                print("Number of iterations: ", t, "(small step)")
            return theta_new, t

        # 3) Small relative loss change
        J = obj(theta_new)
        if abs(J - J_prev) / (J_prev + 1e-12) <= tol_rel_loss:
            if print_num_iters:
                print("Number of iterations: ", t, "(small relative loss change)")
            return theta_new, t

        theta, J_prev = theta_new, J

    if print_num_iters:
        print("Reached maximum iterations:", num_iters)
    return theta, num_iters


def ADAM_gradient_descent_LASSO(
    X, y,
    eta=0.01, lam=1.0,
    rho_1=0.9, rho_2=0.999,
    num_iters=1000,
    print_num_iters=False
):
    """Gradient descent with ADAM, LASSO regression (proximal version).

    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target values
        eta (float, optional): Learning rate. Defaults to 0.01.
        lam (float, optional): Regularization parameter. Defaults to 1.0.
        rho_1 (float, optional): Decay parameter 1 (first moment). Defaults to 0.9.
        rho_2 (float, optional): Decay parameter 2 (second moment). Defaults to 0.999.
        num_iters (int, optional): Maximum number of iterations. Defaults to 1000.
        print_num_iters (bool, optional): If true, prints iteration info.

    Returns:
        theta (numpy.ndarray): Model parameters
        t (int): Number of iterations
    """
    eps = 1e-8
    n, p = X.shape
    
    s = np.zeros(p)  # first moment
    r = np.zeros(p)  # second moment
    theta = np.zeros(p, dtype=float)

    tol_grad = 1e-6
    tol_step = 1e-8

    for t in range(1, num_iters+1):
        # Gradient of OLS part only
        grad = (1.0/n) * X.T @ (X @ theta - y)

        # Stop if gradient is very small
        if np.linalg.norm(grad, ord=np.inf) <= tol_grad:
            if print_num_iters:
                print("Stop: small gradient at", t)
            return theta, t
        
        # Update biased moment estimates
        s = rho_1 * s + (1 - rho_1) * grad
        r = rho_2 * r + (1 - rho_2) * (grad**2)

        # Bias correction
        s_unbiased = s / (1 - rho_1**t)
        r_unbiased = r / (1 - rho_2**t)

        # ADAM step
        update = eta * (s_unbiased / (np.sqrt(r_unbiased) + eps))

        # Proximal step for LASSO
        z = theta - update
        theta_new = soft_threshold(z, eta * lam / (np.sqrt(r_unbiased) + eps))

        # Stop if step is very small
        if np.linalg.norm(theta_new - theta) <= tol_step * (1.0 + np.linalg.norm(theta)):
            if print_num_iters:
                print("Stop: small step at", t)
            return theta_new, t

        theta = theta_new
    
    if print_num_iters:
        print("Reached maximum iterations:", num_iters)
    
    return theta, num_iters
