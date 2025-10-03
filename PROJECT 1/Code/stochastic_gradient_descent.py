import numpy as np

### OLS ###


def sgd_ols(X, y, eta=0.01, n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent for OLS med fast læringsrate.
    Bruker malen: i hver epoke gjøres m = n/M oppdateringer,
    hver gang med en tilfeldig valgt minibatch.

    Args:
        X (ndarray): Feature-matrise (n_samples, n_features)
        y (ndarray): Target (n_samples,)
        eta (float): Læringsrate
        n_epochs (int): Antall epoker
        M (int): Minibatch-størrelse
        seed (int|None): RNG-seed

    Returns:
        theta (ndarray): Estimerte parametre
        steps (int): Totalt antall oppdateringer (n_epochs * m)
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    rng = np.random.default_rng(seed)

    # antall minibatcher per epoke (avrund ned hvis ikke delelig)
    m = n_samples // M  

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for i in range(m):
            # trekk tilfeldig minibatch-indeks
            k = rng.integers(m)
            idx = np.arange(k * M, (k + 1) * M)  # "den k-te minibatchen"

            Xb, yb = X[idx], y[idx]

            # gradient på minibatch
            grad = (2.0 / M) * Xb.T @ (Xb @ theta - yb)

            # oppdater parametre
            theta -= eta * grad
            steps += 1

    return theta, steps

def sgd_momentum_ols(X, y, eta=0.01, momentum=0.3, n_epochs=10, M=5, seed=6114):
    """
    Stochastic gradient descent with momentum for OLS (fixed learning rate).
    Follows the same structure as sgd_ols: m = n//M updates per epoch,
    each update uses one randomly chosen minibatch out of the m fixed minibatches.

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
    change = np.zeros(n_features)          # velocity vector for momentum
    rng = np.random.default_rng(seed)

    # number of minibatches per epoch (rounded down if not divisible)
    m = n_samples // M
    if m == 0:          # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for _ in range(m):
            # pick one random minibatch index
            k = rng.integers(m)
            idx = np.arange(k * M, (k + 1) * M)

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
    Follows the same template as sgd_ols: m = n//M updates per epoch;
    each update uses one randomly chosen minibatch among the m fixed minibatches.

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
            # pick one random minibatch index among the fixed partitions
            k = rng.integers(m)
            idx = np.arange(k * M, (k + 1) * M)

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

import numpy as np

def sgd_RMSProp_ols(X, y, eta=1e-3, rho=0.99, n_epochs=10, M=5, seed=6114, eps=1e-8):
    """
    Stochastic RMSProp for OLS (fixed learning rate).
    Follows the same template as sgd_ols: m = n//M updates per epoch,
    each update uses one randomly chosen minibatch among the m fixed minibatches.

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
    v = np.zeros(n_features)          # running average of squared gradients
    rng = np.random.default_rng(seed)

    # number of minibatches per epoch (rounded down if not divisible)
    m = n_samples // M
    if m == 0:          # handle case M > n
        M = n_samples
        m = 1

    steps = 0
    for epoch in range(1, n_epochs + 1):
        for _ in range(m):
            # pick one random minibatch index
            k = rng.integers(m)
            idx = np.arange(k * M, (k + 1) * M)

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
    Uses the same template as your sgd_ols: m = n//M updates per epoch,
    each update picks one random minibatch among the m fixed minibatches.

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
            # pick one random minibatch index among the fixed partitions
            k = rng.integers(m)
            idx = np.arange(k * M, (k + 1) * M)
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



    