from __future__ import annotations
import numpy as np
from typing import List, Tuple

Array = np.ndarray
Layer = Tuple[Array, Array]          
Grads = List[Tuple[Array, Array]]    
Updates = List[Tuple[Array, Array]]  

### Helper functions
def _zeros_like_layers(layers: List[Layer]) -> Tuple[List[Array], List[Array]]:
    """Utility: allocate zeros with same shapes as layers' W and b."""
    vW = [np.zeros_like(W) for (W, _) in layers]
    vB = [np.zeros_like(b) for (_, b) in layers]
    return vW, vB



def _clip_by_global_norm(grads: Grads, max_norm: float) -> Grads:
    """Global-norm gradient clipping."""
    if (max_norm is None) or (max_norm <= 0.0):
        return grads
    # Compute global norm
    total = 0.0
    for (dW, db) in grads:
        total += np.sum(dW * dW) + np.sum(db * db)
    gnorm = np.sqrt(total) + 1e-12
    if gnorm <= max_norm:
        return grads
    scale = max_norm / gnorm
    return [(dW * scale, db * scale) for (dW, db) in grads]


class Optimizer:
    """Base class (interface)."""

    def __init__(self, lr: float = 1e-2, clip_norm: float | None = None):
        """Initialize optimizer with learning rate and optional gradient clipping.
        """
        self.lr = lr
        self.clip_norm = clip_norm

    def step(self, layers: List[Layer], grads: Grads) -> Updates:
        """
        Compute parameter updates based on current gradients.
        Applies global gradient clipping if enabled.
        """
        
        grads = _clip_by_global_norm(grads, self.clip_norm if self.clip_norm is not None else -1.0)
        return self._step_impl(layers, grads)

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        """Subclasses must implement this to define update rule"""
        raise NotImplementedError

    def reset_state(self):
        """Clear any internal moment/accumulators."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        """Apply standard update rule: param = param - lr * grad"""
        return [(-self.lr * dW, -self.lr * db) for (dW, db) in grads]


class Momentum(Optimizer):
    """SGD with momentum"""

    def __init__(self, lr: float = 1e-2, momentum: float = 0.9, clip_norm: float | None = None):
        super().__init__(lr=lr,  clip_norm=clip_norm)
        self.momentum = momentum
        self.vW: List[Array] | None = None
        self.vB: List[Array] | None = None

    def reset_state(self):
        """Reset stored momentum"""
        self.vW, self.vB = None, None

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        """Apply update rule:
        velocity = momentum * velocity - lr * grad
        param = param + velocity"""
        if self.vW is None:
            self.vW, self.vB = _zeros_like_layers(layers)

        updates: Updates = []
        for i, (dW, db) in enumerate(grads):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW
            self.vB[i] = self.momentum * self.vB[i] - self.lr * db

            updates.append((self.vW[i], self.vB[i]))
        return updates


class Adagrad(Optimizer):
    """Adagrad: per-parameter lr scaled by sqrt(sum(grad^2))."""

    def __init__(self, lr: float = 1e-2, eps: float = 1e-10,
                  clip_norm: float | None = None):
        super().__init__(lr=lr, clip_norm=clip_norm)
        self.eps = eps
        self.GW: List[Array] | None = None
        self.GB: List[Array] | None = None

    def reset_state(self):
        """Reset accumulated gradient squares."""
        self.GW, self.GB = None, None

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        """
        Apply the Adagrad update rule:
        adaptive_lr = lr / sqrt(sum(grad^2) + eps)
        param = param - adaptive_lr * grad
        """
        if self.GW is None:
            self.GW, self.GB = _zeros_like_layers(layers)

        updates: Updates = []
        for i, (dW, db) in enumerate(grads):
            self.GW[i] += dW * dW
            self.GB[i] += db * db

            updW = - self.lr * dW / (np.sqrt(self.GW[i]) + self.eps)
            updB = - self.lr * db / (np.sqrt(self.GB[i]) + self.eps)
            updates.append((updW, updB))
        return updates


class RMSprop(Optimizer):
    """RMSprop optimizer using an exponential moving average of squared gradients."""

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, eps: float = 1e-8,
                 clip_norm: float | None = None):
        super().__init__(lr=lr,  clip_norm=clip_norm)
        self.rho = rho
        self.eps = eps
        self.EW: List[Array] | None = None
        self.EB: List[Array] | None = None

    def reset_state(self):
        """Reset moving averages of squared gradients."""
        self.EW, self.EB = None, None

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        """
        Apply the RMSprop update rule:
        moving_avg = rho * moving_avg + (1 - rho) * grad^2
        param = param - lr * grad / sqrt(moving_avg + eps)
        """
        if self.EW is None:
            self.EW, self.EB = _zeros_like_layers(layers)

        updates: Updates = []
        for i, (dW, db) in enumerate(grads):
            self.EW[i] = self.rho * self.EW[i] + (1.0 - self.rho) * (dW * dW)
            self.EB[i] = self.rho * self.EB[i] + (1.0 - self.rho) * (db * db)

            updW = - self.lr * dW / (np.sqrt(self.EW[i]) + self.eps)
            updB = - self.lr * db / (np.sqrt(self.EB[i]) + self.eps)
            updates.append((updW, updB))
        return updates


class Adam(Optimizer):
    """Adam: momentum + RMSprop with bias correction."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 clip_norm: float | None = None):
        super().__init__(lr=lr, clip_norm=clip_norm)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW: List[Array] | None = None
        self.mB: List[Array] | None = None
        self.vW: List[Array] | None = None
        self.vB: List[Array] | None = None
        self.t = 0  # time step

    def reset_state(self):
        """Reset first and second moment buffers and time step."""
        self.mW = self.mB = self.vW = self.vB = None
        self.t = 0

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        """
        Apply the Adam update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
        """
        if self.mW is None:
            self.mW, self.mB = _zeros_like_layers(layers)
            self.vW, self.vB = _zeros_like_layers(layers)

        self.t += 1
        updates: Updates = []

        for i, (dW, db) in enumerate(grads):
            # First moment
            self.mW[i] = self.beta1 * self.mW[i] + (1.0 - self.beta1) * dW
            self.mB[i] = self.beta1 * self.mB[i] + (1.0 - self.beta1) * db
            # Second moment
            self.vW[i] = self.beta2 * self.vW[i] + (1.0 - self.beta2) * (dW * dW)
            self.vB[i] = self.beta2 * self.vB[i] + (1.0 - self.beta2) * (db * db)

            # Bias correction
            mW_hat = self.mW[i] / (1.0 - self.beta1 ** self.t)
            mB_hat = self.mB[i] / (1.0 - self.beta1 ** self.t)
            vW_hat = self.vW[i] / (1.0 - self.beta2 ** self.t)
            vB_hat = self.vB[i] / (1.0 - self.beta2 ** self.t)

            updW = - self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            updB = - self.lr * mB_hat / (np.sqrt(vB_hat) + self.eps)
            updates.append((updW, updB))

        return updates
