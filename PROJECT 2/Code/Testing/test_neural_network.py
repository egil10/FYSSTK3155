import numpy as np
import sys
import os

# Legg Code-mappen på import-stien
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Implementations.neural_network import NeuralNetwork
from Implementations.activations import relu, relu_deriv, sigmoid, sigmoid_deriv, leaky_relu, \
    leaky_relu_deriv, linear, linear_deriv, softmax  
from Implementations.losses import mse, mse_deriv, bce_with_logits, bce_with_logits_deriv,  \
    cross_entropy, cross_entropy_deriv, cross_entropy_with_logits, cross_entropy_with_logits_deriv

rng = np.random.default_rng(0)

# ---------- 1) REGRESJON (MSE) ----------
print("\n=== REGRESJON (MSE) ===")
B, Din, H, Dout = 8, 5, 10, 1
X = rng.normal(size=(B, Din))
Y = rng.normal(size=(B, Dout))  # kontinuerlig mål

nn_reg = NeuralNetwork(
    network_input_size=Din,
    layer_output_sizes=[H, Dout],
    activation_funcs=[relu, linear],          # siste: lineær utgang
    activation_ders=[relu_deriv, linear_deriv],
    cost_fun=mse,
    cost_der=mse_deriv,
    seed=6114,
)

yhat = nn_reg.predict(X)
print("Predict shape:", yhat.shape)           # (B, 1)
loss0 = nn_reg.cost(X, Y)
print("MSE loss (før):", loss0)

# Ett enkelt “SGD-steg” manuelt
grads = nn_reg.compute_gradient(X, Y)
lr = 1e-2
updates = [(-lr * dW, -lr * db) for (dW, db) in grads]
nn_reg.update_weights(updates)
loss1 = nn_reg.cost(X, Y)
print("MSE loss (etter 1 steg):", loss1)

# ---------- 2) BINÆR KLASSIFISERING (BCE med logits) ----------
print("\n=== BINÆR KLASSIFISERING (BCE med logits) ===")
B, Din, H = 16, 6, 12
Xb = rng.normal(size=(B, Din))
Yb = rng.integers(0, 2, size=(B, 1)).astype(float)  # 0/1 targets

nn_bin = NeuralNetwork(
    network_input_size=Din,
    layer_output_sizes=[H, 1],
    activation_funcs=[relu, linear],                 # logits ut
    activation_ders=[relu_deriv, linear_deriv],
    cost_fun=bce_with_logits,
    cost_der=bce_with_logits_deriv,
    seed=6114,
)

logits = nn_bin.predict(Xb)                          # logits (ikke sigmoid her)
print("Logits shape:", logits.shape)                 # (B, 1)
bce0 = nn_bin.cost(Xb, Yb)
print("BCE(logits) (før):", bce0)
grads = nn_bin.compute_gradient(Xb, Yb)
updates = [(-1e-2 * dW, -1e-2 * db) for (dW, db) in grads]
nn_bin.update_weights(updates)
bce1 = nn_bin.cost(Xb, Yb)
print("BCE(logits) (etter 1 steg):", bce1)

# ---------- 3) MULTIKLASSE KLASSIFISERING (CE med logits) ----------
print("\n=== MULTIKLASSE (CE med logits) ===")
B, Din, H, C = 20, 10, 16, 4
Xm = rng.normal(size=(B, Din))
ym_idx = rng.integers(0, C, size=B)
Ym = np.eye(C)[ym_idx]                              # one-hot targets (B, C)

nn_mc = NeuralNetwork(
    network_input_size=Din,
    layer_output_sizes=[H, C],
    activation_funcs=[relu, linear],                # logits ut
    activation_ders=[relu_deriv, linear_deriv],
    cost_fun=cross_entropy_with_logits,
    cost_der=cross_entropy_with_logits_deriv,
    seed=6114,
)

logits_m = nn_mc.predict(Xm)
print("Logits shape:", logits_m.shape)              # (B, C)
ce0 = nn_mc.cost(Xm, Ym)
print("CE(logits) (før):", ce0)
grads = nn_mc.compute_gradient(Xm, Ym)
updates = [(-1e-2 * dW, -1e-2 * db) for (dW, db) in grads]
nn_mc.update_weights(updates)
ce1 = nn_mc.cost(Xm, Ym)
print("CE(logits) (etter 1 steg):", ce1)

# ---------- 4) SHAPE-SJEKKER ----------
print("\n=== SHAPE-SJEKKER ===")
def check_grad_shapes(nn, X, Y):
    grads = nn.compute_gradient(X, Y)
    for i, (dW, db) in enumerate(grads):
        W, b = nn.layers[i]
        assert dW.shape == W.shape, f"Layer {i} dW shape {dW.shape} != W shape {W.shape}"
        assert db.shape == b.shape, f"Layer {i} db shape {db.shape} != b shape {b.shape}"
        print(f"Layer {i}: dW {dW.shape} OK, db {db.shape} OK")
    print("Alle gradient-shapes OK.")

check_grad_shapes(nn_reg, X, Y)
check_grad_shapes(nn_bin, Xb, Yb)
check_grad_shapes(nn_mc, Xm, Ym)



