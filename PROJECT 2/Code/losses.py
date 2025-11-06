import numpy as np


### MSE ###
def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    Mean Squared Error, mean over batch (and features).
    """
    return np.mean((y_hat - y) ** 2)


def mse_deriv(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    dL/d(y_hat) for MSE with mean over all elements (batch and features).
    """
    denom = y_hat.size  # B * D (evt. flere dimensjoner)
    return (2.0 / denom) * (y_hat - y)


### BCE – BINARY (two variants)###

def bce(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary Cross-Entropy on probabilities p = sigmoid(z).
    Not often used. 
    Returns mean over batch.
    """
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def bce_deriv(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    dL/dp for BCE on probabilites.
    """
    B = p.shape[0]
    p = np.clip(p, eps, 1.0 - eps)
    return -(1.0 / B) * (y / p - (1.0 - y) / (1.0 - p))


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """
    Stable BCE on logits z (recommended for binary classification).
    L(z,y) = mean( max(z,0) - z*y + log(1 + exp(-|z|)) )
    """
    # Stability: log(1+exp(-|z|))
    abs_z = np.abs(z)
    loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-abs_z))
    return np.mean(loss)


def bce_with_logits_deriv(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    dL/dz for BCE with logits. (sigmoid(z) - y)/B
    """
    B = z.shape[0]
    # Sigmoid in a stable way:
    # s = 1 / (1 + exp(-z))
    s = 1.0 / (1.0 + np.exp(-z))
    return (s - y) / B



### Multiclass Cross-Entropy – SOFTMAX (two variants) ###


def cross_entropy(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Multiclass CE on probabilites p = softmax(z).
    y is one-hot: shape (B, K).
    Returns mean over batch.
    """
    p = np.clip(p, eps, 1.0)  

    return -np.mean(np.sum(y * np.log(p), axis=1))


def cross_entropy_deriv(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    dL/dp for multiclass CE on probabilities.
    """
    B = p.shape[0]
    p = np.clip(p, eps, 1.0)
    return -(y / p) / B


def cross_entropy_with_logits(Z: np.ndarray, Y: np.ndarray) -> float:
    """
    Stable multiclass CE on logits Z (recommended).
    Y is one-hot: (B, K).
    """
    # logsumexp along class axis (axis=1)
    Z_max = np.max(Z, axis=1, keepdims=True)
    Z_shift = Z - Z_max
    logsumexp = Z_max + np.log(np.sum(np.exp(Z_shift), axis=1, keepdims=True))
    log_softmax = Z - logsumexp  # (B, K)
    return -np.mean(np.sum(Y * log_softmax, axis=1))


def cross_entropy_with_logits_deriv(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    dL/dZ for multiclass CE with logits.
    """
    B = Z.shape[0]
    Z_max = np.max(Z, axis=1, keepdims=True)
    Z_shift = Z - Z_max
    expZ = np.exp(Z_shift)
    softmax = expZ / np.sum(expZ, axis=1, keepdims=True)
    return (softmax - Y) / B


