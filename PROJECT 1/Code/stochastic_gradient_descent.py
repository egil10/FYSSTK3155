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


### LASSO ###

def soft_threshold(z, alpha):
    """Soft-thresholding operator"""
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)


def sgd_LASSO(X, y, lam=1.0, eta=1e-2, n_epochs=100, M=32, seed=6114):
    """
    Minimerer (1/(2n))||y - Xθ||^2 + lam * ||θ||_1
    Oppdatering per minibatch:
        grad = (1/Mb) Xb^T (Xb θ - yb)
        θ <- S_{eta*lam}( θ - eta * grad )
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
    Stokastisk (minibatch) gradient descent med momentum for LASSO (proximal variant)

    Minimerer: (1/(2n))||y - Xθ||^2 + lam * ||θ||_1
    Per minibatch:
        grad = (1/Mb) Xb^T (Xb θ - yb)
        Δ    = eta * grad + momentum * Δ
        θ    <- S_{eta*lam}( θ - Δ )

    Args:
        X (np.ndarray): (n x p) designmatrise
        y (np.ndarray): (n,) mål
        lam (float): L1-regularisering (λ)
        eta (float): læringsrate
        momentum (float): momentumkoeffisient (typisk 0.3–0.9)
        n_epochs (int): antall passeringer over datasettet
        M (int): minibatch-størrelse
        seed (int): RNG-seed for shuffling

    Returns:
        theta (np.ndarray): Estimert parametervektor (p,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)
    theta = np.zeros(p)
    change = np.zeros(p)  # momentum-buffer (akkumulerer skalerte steg)

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS-gradient på minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # Momentum + gradientsteg (beholder "change" som allerede skalert med eta)
            new_change = eta * grad + momentum * change
            z = theta - new_change

            # Proximal-steg for L1
            theta = soft_threshold(z, eta * lam)

            # Oppdater momentum-buffer
            change = new_change

    return theta


def sgd_ADAGrad_LASSO(X, y, lam=1.0, eta=1e-2, n_epochs=100, M=32, seed=6114, eps=1e-7):
    """
    Stokastisk (minibatch) ADAGrad for LASSO (proximal variant)

    Minimerer: (1/(2n))||y - Xθ||^2 + lam * ||θ||_1
    Per minibatch:
        grad = (1/Mb) Xb^T (Xb θ - yb)
        r   <- r + grad ⊙ grad
        step = eta * grad / (sqrt(r) + eps)
        θ    <- S_{ eta*lam/(sqrt(r)+eps) }( θ - step )
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)

    theta = np.zeros(p, dtype=float)
    r = np.zeros(p, dtype=float)  # ADAGrad-akkumulator (per-parameter)

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS-gradient på minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # Oppdater akkumulert kvadratisk gradient
            r += grad * grad

            # Adaptivt steg (komponentvis)
            denom = np.sqrt(r) + eps
            step = eta * grad / denom
            z = theta - step

            # Proximal-steg (soft-thresholding) med komponentvise terskler
            theta = soft_threshold(z, (eta * lam) / denom)

    return theta



def sgd_RMSProp_LASSO(X, y, lam=1.0, eta=1e-3, rho=0.99,
                      n_epochs=100, M=32, seed=6114, eps=1e-8):
    """
    Stokastisk (minibatch) RMSProp for LASSO (proximal variant)

    Minimerer: (1/(2n))||y - Xθ||^2 + lam * ||θ||_1
    Per minibatch:
        grad  = (1/Mb) Xb^T (Xb θ - yb)
        v     = rho * v + (1-rho) * (grad ⊙ grad)
        step  = eta * grad / (sqrt(v) + eps)
        θ     <- S_{ eta*lam/(sqrt(v)+eps) }( θ - step )

    Args:
        X (np.ndarray): (n x p) designmatrise
        y (np.ndarray): (n,) mål
        lam (float): L1-regularisering (λ)
        eta (float): læringsrate
        rho (float): eksponentiell glatting av kvadrerte gradienter (0<rho<1)
        n_epochs (int): antall passeringer over datasettet
        M (int): minibatch-størrelse
        seed (int): RNG-seed for shuffling
        eps (float): liten konstant for numerisk stabilitet

    Returns:
        theta (np.ndarray): Estimert parametervektor (p,)
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)

    theta = np.zeros(p, dtype=float)
    v = np.zeros(p, dtype=float)  # RMSProp-akkumulator (per-parameter)

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for s in range(0, n, M):
            idx = perm[s:s+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS-gradient på minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # RMSProp-oppdatering av akkumulert kvadratisk gradient
            v = rho * v + (1.0 - rho) * (grad * grad)

            # Adaptivt steg (komponentvis)
            denom = np.sqrt(v) + eps
            step = eta * grad / denom
            z = theta - step

            # Proximal-steg (soft-thresholding) med komponentvise terskler
            theta = soft_threshold(z, (eta * lam) / denom)

    return theta


def sgd_ADAM_LASSO(X, y, lam=1.0, eta=1e-2,
                   rho_1=0.9, rho_2=0.999,
                   n_epochs=100, M=32, seed=6114, eps=1e-8):
    """
    Stokastisk (minibatch) ADAM for LASSO (proximal variant)

    Minimerer: (1/(2n))||y - Xθ||^2 + lam * ||θ||_1
    Per minibatch:
        grad = (1/Mb) Xb^T (Xb θ - yb)
        s <- ρ1 s + (1-ρ1) grad
        r <- ρ2 r + (1-ρ2) (grad ⊙ grad)
        ŝ = s / (1-ρ1^t), r̂ = r / (1-ρ2^t)
        step = eta * ŝ / (sqrt(r̂) + eps)
        θ <- S_{ eta*lam/(sqrt(r̂)+eps) }( θ - step )
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, p = X.shape
    M = min(M, n)

    rng = np.random.default_rng(seed)

    theta = np.zeros(p, dtype=float)
    s = np.zeros(p, dtype=float)  # 1. moment
    r = np.zeros(p, dtype=float)  # 2. moment
    t = 0  # globalt trinntall (per minibatch)

    for _ in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, M):
            idx = perm[start:start+M]
            Xb, yb = X[idx], y[idx]
            Mb = len(idx)

            # OLS-gradient på minibatch
            grad = (1.0 / Mb) * (Xb.T @ (Xb @ theta - yb))

            # Oppdater momenter
            s = rho_1 * s + (1.0 - rho_1) * grad
            r = rho_2 * r + (1.0 - rho_2) * (grad * grad)

            # Bias-korrigering
            t += 1
            s_hat = s / (1.0 - rho_1**t)
            r_hat = r / (1.0 - rho_2**t)

            # ADAM-steg (komponentvis)
            denom = np.sqrt(r_hat) + eps
            step = eta * (s_hat / denom)
            z = theta - step

            # Proximal-steg (soft-thresholding) med komponentvis terskel
            theta = soft_threshold(z, (eta * lam) / denom)

    return theta
