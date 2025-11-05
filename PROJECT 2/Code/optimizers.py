from __future__ import annotations
import numpy as np
from typing import List, Tuple

Array = np.ndarray
Layer = Tuple[Array, Array]          # (W, b)
Grads = List[Tuple[Array, Array]]    # (dW, db)
Updates = List[Tuple[Array, Array]]  # (ΔW, Δb) to be ADDED to parameters


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
        self.lr = lr
        self.clip_norm = clip_norm

    def step(self, layers: List[Layer], grads: Grads) -> Updates:
        """
        Given current layers and raw grads (dW, db), return (ΔW, Δb) updates to ADD.
        Subclasses implement _step_impl.
        """
        
        grads = _clip_by_global_norm(grads, self.clip_norm if self.clip_norm is not None else -1.0)
        return self._step_impl(layers, grads)

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        raise NotImplementedError

    def reset_state(self):
        """Clear any internal moment/accumulators."""
        pass


class SGD(Optimizer):
    """Plain SGD: θ <- θ - lr * g"""

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        return [(-self.lr * dW, -self.lr * db) for (dW, db) in grads]


class Momentum(Optimizer):
    """SGD with momentum (optional Nesterov). v = μ v - lr g; θ += v"""

    def __init__(self, lr: float = 1e-2, momentum: float = 0.9, nesterov: bool = False, clip_norm: float | None = None):
        super().__init__(lr=lr,  clip_norm=clip_norm)
        self.momentum = momentum
        self.nesterov = nesterov
        self.vW: List[Array] | None = None
        self.vB: List[Array] | None = None

    def reset_state(self):
        self.vW, self.vB = None, None

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
        if self.vW is None:
            self.vW, self.vB = _zeros_like_layers(layers)

        updates: Updates = []
        for i, ((dW, db), (W, b)) in enumerate(zip(grads, layers)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dW
            self.vB[i] = self.momentum * self.vB[i] - self.lr * db

            if self.nesterov:
                # Nesterov: use lookahead velocity
                updW = self.momentum * self.vW[i] - self.lr * dW
                updB = self.momentum * self.vB[i] - self.lr * db
                updates.append((updW, updB))
            else:
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
        self.GW, self.GB = None, None

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
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
    """RMSprop: EMA of squared grads; θ <- θ - lr * g / sqrt(E[g^2] + eps)"""

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, eps: float = 1e-8,
                 clip_norm: float | None = None):
        super().__init__(lr=lr,  clip_norm=clip_norm)
        self.rho = rho
        self.eps = eps
        self.EW: List[Array] | None = None
        self.EB: List[Array] | None = None

    def reset_state(self):
        self.EW, self.EB = None, None

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
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
        self.mW = self.mB = self.vW = self.vB = None
        self.t = 0

    def _step_impl(self, layers: List[Layer], grads: Grads) -> Updates:
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
