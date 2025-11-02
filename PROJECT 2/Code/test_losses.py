# test_losses.py
import numpy as np
import importlib

import losses
importlib.reload(losses)

from losses import (
    mse, mse_deriv,
    bce, bce_deriv,
    bce_with_logits, bce_with_logits_deriv,
    cross_entropy, cross_entropy_deriv,
    cross_entropy_with_logits, cross_entropy_with_logits_deriv,
    l1_penalty, l1_deriv,
    l2_penalty, l2_deriv
)

np.random.seed(42)

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def _softmax(Z, axis=1):
    Zm = Z - np.max(Z, axis=axis, keepdims=True)
    e = np.exp(Zm)
    return e / np.sum(e, axis=axis, keepdims=True)

def _finite_diff_grad(f, x, eps=1e-6):
    # f: function that takes x and returns scalar
    # x: ndarray
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad

def test_mse_and_deriv():
    B, D = 5, 3
    y_hat = np.random.randn(B, D)
    y = np.random.randn(B, D)

    # Lukket form
    val = mse(y_hat, y)
    val_manual = np.mean((y_hat - y) ** 2)
    assert np.allclose(val, val_manual, atol=1e-12)

    # Numerisk gradient sjekk w.r.t y_hat
    def f(yy):
        return np.mean((yy - y) ** 2)
    num_g = _finite_diff_grad(lambda arr: f(arr), y_hat.copy())
    ana_g = mse_deriv(y_hat, y)
    assert np.allclose(ana_g, num_g, atol=1e-6, rtol=1e-6)
    print("MSE: ok")

def test_bce_prob_and_logits():
    B = 8
    z = np.random.randn(B, 1)
    y = (np.random.rand(B, 1) > 0.5).astype(float)

    p = _sigmoid(z)

    # Konsistens: BCE(probs) ~ BCEWithLogits(logits)
    val_probs = bce(p, y)
    val_logits = bce_with_logits(z, y)
    assert np.allclose(val_probs, val_logits, atol=1e-8, rtol=1e-8)

    # Deriv w.r.t. logits: numerisk sjekk
    def f_logits(zz):
        return bce_with_logits(zz, y)
    num_g_logits = _finite_diff_grad(lambda arr: f_logits(arr), z.copy())
    ana_g_logits = bce_with_logits_deriv(z, y)
    assert np.allclose(ana_g_logits, num_g_logits, atol=1e-6, rtol=1e-6)

    # Deriv w.r.t. probs: numerisk sjekk
    def f_probs(pp):
        return bce(pp, y)
    num_g_probs = _finite_diff_grad(lambda arr: f_probs(arr), p.copy())
    ana_g_probs = bce_deriv(p, y)
    assert np.allclose(ana_g_probs, num_g_probs, atol=1e-6, rtol=1e-6)
    print("BCE (probs & logits): ok")

def test_ce_multiclass_prob_and_logits():
    B, K = 6, 4
    Z = np.random.randn(B, K)
    # one-hot labels
    labels = np.random.randint(0, K, size=(B,))
    Y = np.eye(K)[labels]

    P = _softmax(Z)

    # Konsistens i verdi
    val_probs = cross_entropy(P, Y)
    val_logits = cross_entropy_with_logits(Z, Y)
    assert np.allclose(val_probs, val_logits, atol=1e-8, rtol=1e-8)

    # Deriv w.r.t. logits: numerisk sjekk
    def f_logits(ZZ):
        return cross_entropy_with_logits(ZZ, Y)
    num_g_logits = _finite_diff_grad(lambda arr: f_logits(arr), Z.copy())
    ana_g_logits = cross_entropy_with_logits_deriv(Z, Y)
    assert np.allclose(ana_g_logits, num_g_logits, atol=1e-6, rtol=1e-6)

    # Deriv w.r.t. probs: numerisk sjekk
    def f_probs(PP):
        return cross_entropy(PP, Y)
    num_g_probs = _finite_diff_grad(lambda arr: f_probs(arr), P.copy())
    ana_g_probs = cross_entropy_deriv(P, Y)
    assert np.allclose(ana_g_probs, num_g_probs, atol=1e-6, rtol=1e-6)
    print("Cross-Entropy (probs & logits): ok")

def test_regularization_l2():
    # Små matriser uten null for tydelig gradient
    W = np.array([[0.5, -0.2], [0.3, -0.7]], dtype=float)
    b = np.array([[0.1, -0.4]], dtype=float)
    layers = [(W.copy(), b.copy())]
    lam = 0.05

    # Verdi
    val = l2_penalty(layers, lam)
    val_manual = lam * (np.sum(W**2) + np.sum(b**2))
    assert np.allclose(val, val_manual, atol=1e-12)

    # Deriv v.s. numerisk gradient
    def fW(Wvar):
        return lam * (np.sum(Wvar**2) + np.sum(b**2))
    def fb(bvar):
        return lam * (np.sum(W**2) + np.sum(bvar**2))

    num_gW = _finite_diff_grad(lambda arr: fW(arr), W.copy())
    num_gb = _finite_diff_grad(lambda arr: fb(arr), b.copy())
    ana_gW, ana_gb = l2_deriv(W, b, lam)
    assert np.allclose(ana_gW, num_gW, atol=1e-7, rtol=1e-7)
    assert np.allclose(ana_gb, num_gb, atol=1e-7, rtol=1e-7)
    print("L2 regularization: ok")

def test_regularization_l1():
    # Unngå nøyaktig 0 for å slippe ikke-differensierbare punkter
    W = np.array([[0.5, -0.2], [0.3, -0.7]], dtype=float)
    b = np.array([[0.1, -0.4]], dtype=float)
    layers = [(W.copy(), b.copy())]
    lam = 0.05

    # Verdi
    val = l1_penalty(layers, lam)
    val_manual = lam * (np.sum(np.abs(W)) + np.sum(np.abs(b)))
    assert np.allclose(val, val_manual, atol=1e-12)

    # Deriv v.s. numerisk gradient (subgradient, fungerer når elementer != 0)
    def fW(Wvar):
        return lam * (np.sum(np.abs(Wvar)) + np.sum(np.abs(b)))
    def fb(bvar):
        return lam * (np.sum(np.abs(W)) + np.sum(np.abs(bvar)))

    num_gW = _finite_diff_grad(lambda arr: fW(arr), W.copy(), eps=1e-6)
    num_gb = _finite_diff_grad(lambda arr: fb(arr), b.copy(), eps=1e-6)
    ana_gW, ana_gb = l1_deriv(W, b, lam)
    # Litt slakk pga. ikke-differensierbarhet ved 0
    assert np.allclose(ana_gW, num_gW, atol=1e-5, rtol=1e-5)
    assert np.allclose(ana_gb, num_gb, atol=1e-5, rtol=1e-5)
    print("L1 regularization: ok")

if __name__ == "__main__":
    print("Running losses.py tests...\n")
    test_mse_and_deriv()
    test_bce_prob_and_logits()
    test_ce_multiclass_prob_and_logits()
    test_regularization_l2()
    test_regularization_l1()
    print("\nAll losses tests passed.")
