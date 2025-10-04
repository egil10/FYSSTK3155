import numpy as np

### OLS ###


def sgd_ols(X, y, eta=0.01, n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent with fixed learning rate, OLS

    Args:
        X (ndarray): Feature-matrix (n_samples, n_features)
        y (ndarray): Target (n_samples,)
        eta (float): Learning rate
        n_epochs (int): Number of epochs
        M (int): Size of minibatches
        seed (int|None): RNG-seed

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of steps
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    rng = np.random.default_rng(seed)

    # Number of minibaches per epoch
    m = n_samples // M  

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for i in range(m):
            # Draw random 
            idx = rng.choice(n_samples, size=M, replace=False)

            Xb, yb = X[idx], y[idx]

            # gradient on minibatch
            grad = (2.0 / M) * Xb.T @ (Xb @ theta - yb)

            # Update parameters
            theta -= eta * grad
            steps += 1

    return theta, steps

def sgd_momentum_ols(X, y, eta=0.01, momentum=0.3, n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent with momentum for OLS.


    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        eta (float): Learning rate
        momentum (float): Momentum coefficient (beta)
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    change = np.zeros(n_features)
    rng = np.random.default_rng(seed)

    # number of minibatches per epoch (rounded down if not divisible)
    m = n_samples // M
    if m == 0:          # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for _ in range(m):
            idx = rng.choice(n_samples, size=M, replace=False)

            Xb, yb = X[idx], y[idx]
            # compute minibatch gradient for OLS
            grad = (2.0 / M) * Xb.T @ (Xb @ theta - yb)

            # momentum update
            new_change = eta * grad + momentum * change
            theta -= new_change
            change = new_change

            steps += 1

    return theta, steps


def sgd_ADAGrad_ols(X, y, eta=0.01, n_epochs=10, M=5, seed=6114, eps=1e-7):
    """
    Stochastic ADAGrad for OLS (fixed learning rate).

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        eta (float): Base learning rate (scaled per-parameter by ADAGrad)
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility
        eps (float): Small constant to avoid division by zero

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    # ADAGrad accumulator (per-parameter sum of squared gradients)
    r = np.zeros(n_features)

    rng = np.random.default_rng(seed)

    # number of fixed minibatches per epoch (rounded down)
    m = n_samples // M
    if m == 0:           # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for _ in range(m):
            idx = rng.choice(n_samples, size=M, replace=False)

            Xb, yb = X[idx], y[idx]

            # minibatch gradient for OLS
            grad = (2.0 / M) * Xb.T @ (Xb @ theta - yb)

            # ADAGrad accumulator update
            r += grad**2

            # per-parameter scaled step
            step = eta * grad / (np.sqrt(r) + eps)

            # parameter update
            theta -= step

            steps += 1

    return theta, steps


def sgd_RMSProp_ols(X, y, eta=1e-3, rho=0.99, n_epochs=10, M=5, seed=6114, eps=1e-8):
    """
    Stochastic RMSProp for OLS (fixed learning rate).

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        eta (float): Learning rate
        rho (float): Decay parameter for RMSProp moving average
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility
        eps (float): Small constant to avoid division by zero

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    v = np.zeros(n_features)         
    rng = np.random.default_rng(seed)

    # number of minibatches per epoch (rounded down if not divisible)
    m = n_samples // M
    if m == 0:          # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for _ in range(m):
            
            idx = rng.choice(n_samples, size=M, replace=False)

            Xb, yb = X[idx], y[idx]

            # minibatch gradient for OLS
            grad = (2.0 / M) * Xb.T @ (Xb @ theta - yb)

            # RMSProp accumulator update
            v = rho * v + (1.0 - rho) * (grad * grad)

            # compute scaled step
            step = eta * grad / (np.sqrt(v) + eps)

            # update parameters
            theta -= step

            steps += 1

    return theta, steps


def sgd_ADAM_ols(X, y, eta=1e-3, rho_1=0.9, rho_2=0.999,
                 n_epochs=10, M=5, seed=6114, eps=1e-8):
    """
    Stochastic ADAM for OLS (fixed learning rate).

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        eta (float): Base learning rate
        rho_1 (float): Exponential decay for first moment (beta1)
        rho_2 (float): Exponential decay for second moment (beta2)
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility
        eps (float): Small constant for numerical stability

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features, dtype=float)

    m1 = np.zeros(n_features, dtype=float)  # first moment
    m2 = np.zeros(n_features, dtype=float)  # second moment

    rng = np.random.default_rng(seed)

    # number of fixed minibatches per epoch (rounded down)
    m = n_samples // M
    if m == 0:          # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    t = 0  # time step for bias correction

    for _ in range(n_epochs):
        for _ in range(m):
            idx = rng.choice(n_samples, size=M, replace=False)
            Xb, yb = X[idx], y[idx]

            # minibatch gradient for OLS
            grad = (2.0 / M) * Xb.T @ (Xb @ theta - yb)

            # Adam moments
            m1 = rho_1 * m1 + (1.0 - rho_1) * grad
            m2 = rho_2 * m2 + (1.0 - rho_2) * (grad * grad)

            # bias correction
            t += 1
            m1_hat = m1 / (1.0 - rho_1**t)
            m2_hat = m2 / (1.0 - rho_2**t)

            # parameter update
            step = eta * m1_hat / (np.sqrt(m2_hat) + eps)
            theta -= step

            steps += 1

    return theta, steps

### Ridge ###
def sgd_Ridge(X, y, lam=1e-2, eta=0.01, n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent, fixed learning rate, Ridge

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target values. (n_samples,)
        lam (float): Regularization parameter (lambda)
        eta (float): Learning rate
        n_epochs (int): Number of epochs
        M (int): Minibatch-size
        seed (int): RNG-seed

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    rng = np.random.default_rng(seed)

    m = n_samples // M
    if m == 0:                
        M = n_samples
        m = 1

    steps = 0
    for _ in range(n_epochs):
        for _ in range(m):
            idx = rng.choice(n_samples, size=M, replace=False)

            Xb, yb = X[idx], y[idx]

            # Gradient
            # data-gradient
            grad_data = (2.0 / M) * Xb.T @ (Xb @ theta - yb)
            # ridge-term
            lam_eff   = lam / n_samples
            # total
            grad = grad_data + 2.0 * lam_eff * theta



            # oppdater
            theta -= eta * grad
            steps += 1

    return theta, steps


def sgd_momentum_Ridge(X, y, lam=1e-2, eta=0.01, momentum=0.3, 
                       n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent with momentum for Ridge regression (fixed learning rate).

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength (lambda)
        eta (float): Learning rate
        momentum (float): Momentum coefficient (beta)
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    change = np.zeros(n_features)   # momentum "velocity"
    rng = np.random.default_rng(seed)

    m = n_samples // M
    if m == 0:   # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for _ in range(n_epochs):
        for _ in range(m):
            # Draw random minibatch
            idx = rng.choice(n_samples, size=M, replace=False)
            Xb, yb = X[idx], y[idx]

            # gradient
            # data-gradient
            grad_data = (2.0 / M) * Xb.T @ (Xb @ theta - yb)
            # ridge-term
            lam_eff   = lam / n_samples
            # total
            grad = grad_data + 2.0 * lam_eff * theta

            # momentum update
            new_change = eta * grad + momentum * change
            theta -= new_change
            change = new_change

            steps += 1

    return theta, steps


def sgd_ADAGrad_Ridge(X, y, lam=1e-2, eta=0.01, n_epochs=10, M=5,
                      seed=6114, eps=1e-7):
    """
    Stochastic AdaGrad for Ridge (OLS + L2) with fixed learning rate.


    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength (lambda)
        eta (float): Base learning rate (per-parameter scaled by AdaGrad)
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility
        eps (float): Small constant to avoid division by zero

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    # AdaGrad accumulator (sum of squared gradients per parameter)
    r = np.zeros(n_features)

    rng = np.random.default_rng(seed)

    # number of fixed minibatches per epoch (rounded down)
    m = n_samples // M
    if m == 0:                 # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for _ in range(n_epochs):
        for _ in range(m):
            # Draw random minibatch
            idx = rng.choice(n_samples, size=M, replace=False)

            Xb, yb = X[idx], y[idx]
            
            # Gradient

            # data-gradient
            grad_data = (2.0 / M) * Xb.T @ (Xb @ theta - yb)
            # ridge-term
            lam_eff   = lam / n_samples
            # total
            grad = grad_data + 2.0 * lam_eff * theta

            # AdaGrad accumulator and parameter update
            r += grad ** 2
            step = eta * grad / (np.sqrt(r) + eps)
            theta -= step

            steps += 1

    return theta, steps


def sgd_RMSProp_Ridge(X, y, lam=1.0, eta=1e-4, rho=0.99,
                              n_epochs=1000, M=5, seed=6114, eps=1e-8):
    """
    Stochastic RMSProp for Ridge (OLS + L2).


    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength (lambda)
        eta (float): Base learning rate (per-parameter scaled by AdaGrad)
        rho (float): Decay parameter for RMSProp moving average
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed for reproducibility
        eps (float): Small constant to avoid division by zero

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of updates (= n_epochs * m)
    """
    n, p = X.shape
    theta = np.zeros(p)
    v = np.zeros(p)
    rng = np.random.default_rng(seed)

    m = max(1, n // M)

    steps = 0
    for _ in range(n_epochs):
        for _ in range(m):
            # Draw random minibatch
            idx = rng.choice(n, size=M, replace=False)
            Xb, yb = X[idx], y[idx]
            

            # Gradient
            
            # data-gradient
            grad_data = (2.0 / M) * Xb.T @ (Xb @ theta - yb)
            # ridge-term
            lam_eff   = lam / n
            # total
            grad = grad_data + 2.0 * lam_eff * theta

            # RMSProp
            v = rho * v + (1.0 - rho) * (grad * grad)
            step = eta * grad / (np.sqrt(v) + eps)
            theta -= step

            steps += 1

    return theta, steps


def sgd_ADAM_Ridge(X, y, lam=1.0, eta=1e-3, rho_1=0.9, rho_2=0.999,
                   n_epochs=10, M=5, seed=6114, eps=1e-8):
    """
    Stochastic Adam for Ridge regression (sum-loss version).

    Loss: J(θ) = ||y - Xθ||^2 + lam * ||θ||^2
    Gradient: ∇J = 2 X^T(Xθ - y) + 2 lam θ

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge penalty parameter (lambda)
        eta (float): Learning rate
        rho_1 (float): Exponential decay for first moment (beta1)
        rho_2 (float): Exponential decay for second moment (beta2)
        n_epochs (int): Number of epochs
        M (int): Minibatch size
        seed (int): RNG seed
        eps (float): Small constant for numerical stability

    Returns:
        theta (ndarray): Estimated parameters
        steps (int): Total number of parameter updates (= n_epochs * m)
    """
    n, p = X.shape
    theta = np.zeros(p, dtype=float)

    m1 = np.zeros(p, dtype=float)  # first moment
    m2 = np.zeros(p, dtype=float)  # second moment

    rng = np.random.default_rng(seed)

    m = n // M
    if m == 0:   # handle case M > n
        M = n
        m = 1

    steps = 0
    t = 0  # time step for bias correction

    for _ in range(n_epochs):
        for _ in range(m):
            # Draw random minibatch
            idx = rng.choice(n, size=M, replace=False)
            Xb, yb = X[idx], y[idx]


            # Gradient
            # data-gradient
            grad_data = (2.0 / M) * Xb.T @ (Xb @ theta - yb)
            # ridge-term
            lam_eff   = lam / n
            # total
            grad = grad_data + 2.0 * lam_eff * theta


            # --- Adam updates ---
            m1 = rho_1 * m1 + (1.0 - rho_1) * grad
            m2 = rho_2 * m2 + (1.0 - rho_2) * (grad * grad)

            t += 1
            m1_hat = m1 / (1.0 - rho_1**t)
            m2_hat = m2 / (1.0 - rho_2**t)

            step = eta * m1_hat / (np.sqrt(m2_hat) + eps)
            theta -= step

            steps += 1

    return theta, steps




