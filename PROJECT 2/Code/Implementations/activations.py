import numpy as np

### ReLU ###
def relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, 0.0)

def relu_deriv(z: np.ndarray) -> np.ndarray:
    # d/dz ReLU(z) = 1 for z>0, 0 else 
    return np.where(z > 0, 1.0, 0.0)

### Sigmoid (stable version) ###
def sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable implementation: Split on sign
    # Avoid overflow when z << 0 or z >> 0
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])          
    out[neg] = ez / (1.0 + ez)
    return out

def sigmoid_deriv(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (s - 1.0) * -1.0  

### Leaky ReLU ###
def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)

### Linear / Identity ###
def linear(z: np.ndarray) -> np.ndarray:
    return z

def linear_deriv(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z, dtype=float)

### Softmax (stable) ###
def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable softmax. Works for 1D and 2D input
 
    axis: which axis that represents classes
    """
    z = np.asarray(z, dtype=float)
    # Subtract from max along axis for stability
    z_max = np.max(z, axis=axis, keepdims=True)
    z_shifted = z - z_max
    e = np.exp(z_shifted)
    sum_e = np.sum(e, axis=axis, keepdims=True)
    return e / sum_e

