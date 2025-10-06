import numpy as np

### OLS ###


def sgd_ols(X, y, eta=0.01, n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent with fixed learning rate, OLS

    Args:
        X (ndarray): Feature-matrix (n_samples, n_features)
        y (ndarray): Target (n_samples,)
        eta (float): Learning rate. Defaults to 0.01.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Size of minibatches. Defaults to 5.
        seed (int|None): RNG-seed. Defaults to 6114

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
        eta (float): Learning rate. Defaults to 0.01.
        momentum (float): Momentum coefficient. Defaults to 0.3.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5. 
        seed (int): RNG seed for reproducibility. Defaults to 6114.

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
        eta (float): Learning rate. Defaults to 0.01.
        n_epochs (int): Number of epochs. Defaults to 10. 
        M (int): Minibatch size. Defaults to 5. 
        seed (int): RNG seed for reproducibility. Defaults t0 6114.
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
        eta (float): Learning rate. Defaults to 0.001
        rho (float): Decay parameter for RMSProp moving average. Defaults to 0.99.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5. 
        seed (int): RNG seed for reproducibility. Defaults to 6114.
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
    Stochastic gradient descent with ADAM for OLS.

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        eta (float): Learning rate. Defaults to 0.001
        rho_1 (float): Decay parameter 1. Defaults to 0.9.
        rho_2 (float): Decay parameter 2. Defaults to 0.999.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5. 
        seed (int): RNG seed for reproducibility. Defaults to 6114.
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
        lam (float): Regularization parameter. Defaults to 0.01.
        eta (float): Learning rate. Defaults to 0.01.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch-size. Defaults to 5.
        seed (int): RNG-seed. Defaults to 6114.

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
    Stochastic gradient descent with momentum for Ridge regression.

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength defaults to 0.01.
        eta (float): Learning rate. Defaults to 0.01. 
        momentum (float): Momentum coefficient. Defaults to 0.3.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5.
        seed (int): RNG seed for reproducibility. Defaults to 6114.

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
    Stochastic gradient descent with AdaGrad for Ridge.


    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength (lambda). Defaults to 0.01.
        eta (float): Learning rate. Defaults to 0.01.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5.
        seed (int): RNG seed for reproducibility. Defaults to 6114.
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
    Stochastic gradient descent with RMSProp for Ridge.


    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength (lambda). Defaults to 0.01.
        eta (float): Learning rate. Defaults to 0.0001.
        rho (float): Decay parameter for RMSProp moving average. Defaults to 0.99.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5.
        seed (int): RNG seed for reproducibility. Defaults to 6114.
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
    Stochastic gradient descent with ADAM for Ridge regression.

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge penalty parameter (lambda). Defaults to 1.0
        eta (float): Learning rate. Defaults to 0.001.
        rho_1 (float): Decay parameter 1. Defaults to 0.9.
        rho_2 (float): Decay parameter 2. Defaults to 0.99.
        n_epochs (int): Number of epochs. Defaults to 10.
        M (int): Minibatch size. Defaults to 5.
        seed (int): RNG seed. Defaults to 6114.
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


### LASSO ###

def soft_threshold(z, alpha):
    """Soft-thresholding operator, used for proximal gradient descent with LASSO"""
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)


def sgd_LASSO(X, y, lam=1.0, eta=1e-2, n_epochs=100, M=32, seed=6114):
    """
    Stochastic gradient descent, fixed learning rate, LASSO regression. 
    Proximal version.

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target values. (n_samples,)
        lam (float): Regularization parameter for LASSO. Defaults to 1.0.
        eta (float): Learning rate. Defaults to 0.01.
        n_epochs (int): Number of epochs. Defaults to 100.
        M (int): Minibatch-size. Defaults to 32.
        seed (int): RNG-seed. Defaults to 6114.

    Returns:
        theta (ndarray): Estimated parameters
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)
    rng = np.random.default_rng(seed)

    theta = np.zeros(p)

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))   # OLS-del
            theta = soft_threshold(theta - eta * grad, eta * lam)  # prox L1

    return theta

def sgd_momentum_LASSO(X, y, lam=1.0, eta=1e-2, momentum=0.3,
                       n_epochs=100, M=32, seed=6114):
    """
    Stochastic gradient descent with momentum for LASSO regression.
    Proximal version

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): LASSO regularization strength defaults to 1.0.
        eta (float): Learning rate. Defaults to 0.01. 
        momentum (float): Momentum coefficient. Defaults to 0.3.
        n_epochs (int): Number of epochs. Defaults to 100.
        M (int): Minibatch size. Defaults to 32.
        seed (int): RNG seed for reproducibility. Defaults to 6114.

    Returns:
        theta (ndarray): Estimated parameters
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)
    theta = np.zeros(p)
    change = np.zeros(p) 

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS gradient on minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))


            new_change = eta * grad + momentum * change
            z = theta - new_change

            # Proximal step
            theta = soft_threshold(z, eta * lam)

            # Update momentum
            change = new_change

    return theta


def sgd_ADAGrad_LASSO(X, y, lam=1.0, eta=1e-2, n_epochs=100, M=32, seed=6114, eps=1e-7):
    """
    Stochastic gradient descent with AdaGrad for LASSO.
    Proximal version

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): LASSO regularization strength (lambda). Defaults to 1.0.
        eta (float): Learning rate. Defaults to 0.01.
        n_epochs (int): Number of epochs. Defaults to 100.
        M (int): Minibatch size. Defaults to 32.
        seed (int): RNG seed for reproducibility. Defaults to 6114.
        eps (float): Small constant to avoid division by zero

    Returns:
        theta (ndarray): Estimated parameters
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)

    theta = np.zeros(p, dtype=float)
    r = np.zeros(p, dtype=float)

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS gradient on minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # Update r
            r += grad * grad

            # Adaptive step
            denom = np.sqrt(r) + eps
            step = eta * grad / denom
            z = theta - step

            # Proximal step
            theta = soft_threshold(z, (eta * lam) / denom)

    return theta



def sgd_RMSProp_LASSO(X, y, lam=1.0, eta=1e-3, rho=0.99,
                      n_epochs=100, M=32, seed=6114, eps=1e-8):
    """
    Stochastic gradient descent with RMSProp for LASSO.
    Proximal version

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge regularization strength (lambda). Defaults to 1.0.
        eta (float): Learning rate. Defaults to 0.001.
        rho (float): Decay parameter for RMSProp moving average. Defaults to 0.99.
        n_epochs (int): Number of epochs. Defaults to 100.
        M (int): Minibatch size. Defaults to 32.
        seed (int): RNG seed for reproducibility. Defaults to 6114.
        eps (float): Small constant to avoid division by zero

    Returns:
        theta (ndarray): Estimated parameters
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)

    theta = np.zeros(p, dtype=float)
    v = np.zeros(p, dtype=float)  

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS gradient on mini-batch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # Update v
            v = rho * v + (1.0 - rho) * (grad * grad)

            # Adaptive step
            denom = np.sqrt(v) + eps
            step = eta * grad / denom
            z = theta - step

            # Proximal step
            theta = soft_threshold(z, (eta * lam) / denom)

    return theta


def sgd_ADAM_LASSO(X, y, lam=1.0, eta=1e-2,
                   rho_1=0.9, rho_2=0.999,
                   n_epochs=100, M=32, seed=6114, eps=1e-8):
    """
    Stochastic gradient descent with ADAM for LASSO regression.
    Proximal version. 

    Args:
        X (ndarray): Feature matrix (n_samples, n_features)
        y (ndarray): Target vector (n_samples,)
        lam (float): Ridge penalty parameter (lambda). Defaults to 1.0
        eta (float): Learning rate. Defaults to 0.01.
        rho_1 (float): Decay parameter 1. Defaults to 0.9.
        rho_2 (float): Decay parameter 2. Defaults to 0.99.
        n_epochs (int): Number of epochs. Defaults to 100.
        M (int): Minibatch size. Defaults to 32.
        seed (int): RNG seed. Defaults to 6114.
        eps (float): Small constant for numerical stability

    Returns:
        theta (ndarray): Estimated parameters
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)

    theta = np.zeros(p, dtype=float)
    s = np.zeros(p, dtype=float)  # 1st moment
    r = np.zeros(p, dtype=float)  # 2nd moment
    t = 0  

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, M):
            idx = perm[start:start+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS gradient on minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # Update moments
            s = rho_1 * s + (1.0 - rho_1) * grad
            r = rho_2 * r + (1.0 - rho_2) * (grad * grad)

            # Bias-correction
            t += 1
            s_hat = s / (1.0 - rho_1**t)
            r_hat = r / (1.0 - rho_2**t)

            # ADAM-step
            denom = np.sqrt(r_hat) + eps
            step = eta * (s_hat / denom)
            z = theta - step

            # Proximal step 
            theta = soft_threshold(z, (eta * lam) / denom)

    return theta
