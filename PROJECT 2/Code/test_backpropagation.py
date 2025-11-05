# test_backpropagation.py
# Comprehensive backpropagation tests for NeuralNetwork implementation.


import numpy as np

from neural_network import NeuralNetwork
from activations import (
    relu, relu_deriv,
    sigmoid, sigmoid_deriv,
    linear, linear_deriv,
    softmax
)
from losses import (
    mse, mse_deriv,
    bce_with_logits, bce_with_logits_deriv,
    cross_entropy_with_logits, cross_entropy_with_logits_deriv
)
from optimizers import SGD, Adam

np.set_printoptions(precision=6, suppress=True)
np.random.seed(6114)


# -------------------------------
# Utility helpers
# -------------------------------

def flatten_params(layers):
    """Flatten all parameters (W, b) into a single vector; also return shapes & keys to restore."""
    vecs, shapes, keys = [], [], []
    for li, (W, b) in enumerate(layers):
        vecs.append(W.ravel())
        shapes.append(W.shape)
        keys.append((li, "W"))
        vecs.append(b.ravel())
        shapes.append(b.shape)
        keys.append((li, "b"))
    theta = np.concatenate(vecs).astype(np.float64)
    return theta, shapes, keys


def set_params_from_theta(nn: NeuralNetwork, theta, shapes, keys):
    """Write vector 'theta' back into nn.layers respecting original shapes."""
    i = 0
    new_layers = []
    param_map = {}
    for shape, (li, name) in zip(shapes, keys):
        n = int(np.prod(shape))
        block = theta[i:i+n].reshape(shape).astype(float)
        param_map.setdefault(li, {})[name] = block
        i += n
    for li, (W_old, b_old) in enumerate(nn.layers):
        W_new = param_map[li].get("W", W_old)
        b_new = param_map[li].get("b", b_old)
        new_layers.append((W_new, b_new))
    nn.layers = new_layers


def numerical_gradient(nn: NeuralNetwork, X, Y, eps=1e-5):
    """Central difference numerical gradient dL/dtheta."""
    theta, shapes, keys = flatten_params(nn.layers)

    def loss_from_theta(tt):
        set_params_from_theta(nn, tt, shapes, keys)
        L = nn.cost(X, Y)
        return float(L)

    g_num = np.zeros_like(theta, dtype=np.float64)
    for i in range(theta.size):
        t_plus = theta.copy();  t_plus[i] += eps
        t_minus = theta.copy(); t_minus[i] -= eps
        Lp = loss_from_theta(t_plus)
        Lm = loss_from_theta(t_minus)
        g_num[i] = (Lp - Lm) / (2.0 * eps)
    return g_num, theta, shapes, keys


def analytical_gradient(nn: NeuralNetwork, X, Y):
    """Collect analytical gradient from nn.compute_gradient and flatten."""
    grads = nn.compute_gradient(X, Y)
    flat = []
    for (dW, db) in grads:
        flat.append(dW.ravel())
        flat.append(db.ravel())
    return np.concatenate(flat).astype(np.float64)


def max_err(g_ana, g_num):
    abs_err = np.abs(g_ana - g_num)
    rel_err = abs_err / (np.abs(g_ana) + np.abs(g_num) + 1e-12)
    return abs_err.max(), np.median(abs_err), rel_err.max(), np.median(rel_err)


def clone_same_arch(nn_src: NeuralNetwork, cost_fun, cost_der):
    """Create a new NN with the same shapes and activations as nn_src."""
    in_dim = nn_src.layers[0][0].shape[0]
    out_sizes = [W.shape[1] for (W, _) in nn_src.layers]
    acts = nn_src.activation_funcs
    ders = nn_src.activation_ders
    nn_new = NeuralNetwork(
        network_input_size=in_dim,
        layer_output_sizes=out_sizes,
        activation_funcs=acts,
        activation_ders=ders,
        cost_fun=cost_fun,
        cost_der=cost_der,
        seed=6114,
    )
    theta, shapes, keys = flatten_params(nn_src.layers)
    set_params_from_theta(nn_new, theta, shapes, keys)
    return nn_new


def assert_close(name, g1, g2, atol=1e-6, rtol=1e-4):
    if not np.allclose(g1, g2, atol=atol, rtol=rtol):
        diff = np.abs(g1 - g2)
        raise AssertionError(f"{name} mismatch: max diff {diff.max():.3e}")


# -------------------------------
# 1) Numerical gradient checks
# -------------------------------

def run_numeric_checks():
    rng = np.random.default_rng(0)

    configs = [
        dict(
            name="MSE (regression)",
            Din=3, H=5, Dout=2,
            make_data=lambda: (rng.normal(size=(4, 3)), rng.normal(size=(4, 2))),
            acts=[relu, linear],
            ders=[relu_deriv, linear_deriv],
            cost=mse, dcost=mse_deriv
        ),
        dict(
            name="BCE-with-logits (binary)",
            Din=4, H=6, Dout=1,
            make_data=lambda: (rng.normal(size=(5, 4)), rng.integers(0, 2, size=(5, 1)).astype(float)),
            acts=[relu, linear],
            ders=[relu_deriv, linear_deriv],
            cost=bce_with_logits, dcost=bce_with_logits_deriv
        ),
        dict(
            name="CE-with-logits (multiclass)",
            Din=5, H=7, Dout=4,
            make_data=lambda: (
                rng.normal(size=(6, 5)),
                np.eye(4)[rng.integers(0, 4, size=6)]
            ),
            acts=[relu, linear],
            ders=[relu_deriv, linear_deriv],
            cost=cross_entropy_with_logits, dcost=cross_entropy_with_logits_deriv
        ),
    ]

    for cfg in configs:
        print(f"\n[Grad check] {cfg['name']}")
        X, Y = cfg["make_data"]()
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        nn = NeuralNetwork(
            network_input_size=cfg["Din"],
            layer_output_sizes=[cfg["H"], cfg["Dout"]],
            activation_funcs=cfg["acts"],
            activation_ders=cfg["ders"],
            cost_fun=cfg["cost"],
            cost_der=cfg["dcost"],
            seed=6114,
        )

        g_num, theta, shapes, keys = numerical_gradient(nn, X, Y, eps=1e-6)
        set_params_from_theta(nn, theta, shapes, keys)
        g_ana = analytical_gradient(nn, X, Y)

        abs_max, abs_med, rel_max, rel_med = max_err(g_ana, g_num)
        print(f"  max|diff|={abs_max:.3e}, max rel={rel_max:.3e}")
        assert abs_max < 3e-5 and rel_max < 3e-4, "Gradient check FAILED."

        # Optional: check L2 modifies gradients
        nn_l2 = NeuralNetwork(
            network_input_size=cfg["Din"],
            layer_output_sizes=[cfg["H"], cfg["Dout"]],
            activation_funcs=cfg["acts"],
            activation_ders=cfg["ders"],
            cost_fun=cfg["cost"],
            cost_der=cfg["dcost"],
            seed=6114,
            l2_lambda=1e-3,
        )
        g_num_l2, _, _, _ = numerical_gradient(nn_l2, X, Y, eps=1e-6)
        g_ana_l2 = analytical_gradient(nn_l2, X, Y)
        if np.allclose(g_ana_l2, g_ana):
            raise AssertionError("L2 regularization had no effect on gradients!")
        print("  +L2 -> gradient shift detected (OK).")

    print("\nNumeric gradient checks: OK.")


# -------------------------------
# 2) Loss scaling test
# -------------------------------

def run_scaling_test():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 3))
    Y = np.eye(3)[rng.integers(0, 3, size=4)]

    nn = NeuralNetwork(
        network_input_size=3,
        layer_output_sizes=[5, 3],
        activation_funcs=[relu, linear],
        activation_ders=[relu_deriv, linear_deriv],
        cost_fun=cross_entropy_with_logits,
        cost_der=cross_entropy_with_logits_deriv,
        seed=6114,
    )

    g_base = analytical_gradient(nn, X, Y)

    c = 3.7
    def cost_scaled(z, y): return c * cross_entropy_with_logits(z, y)
    def dcost_scaled(z, y): return c * cross_entropy_with_logits_deriv(z, y)

    nn_scaled = clone_same_arch(nn, cost_scaled, dcost_scaled)
    g_scaled = analytical_gradient(nn_scaled, X, Y)
    assert_close("scaled grads", g_scaled, c * g_base)
    print("\nLoss scaling test: OK.")


# -------------------------------
# 3) Softmax + CE closed-form
# -------------------------------

def run_softmax_ce_closed_form_test():
    rng = np.random.default_rng(2)
    B, Din, H, C = 6, 4, 8, 5
    X = rng.normal(size=(B, Din))
    Y = np.eye(C)[rng.integers(0, C, size=B)]

    nn = NeuralNetwork(
        network_input_size=Din,
        layer_output_sizes=[H, C],
        activation_funcs=[relu, linear],
        activation_ders=[relu_deriv, linear_deriv],
        cost_fun=cross_entropy_with_logits,
        cost_der=cross_entropy_with_logits_deriv,
        seed=6114,
    )

    Z = nn.predict(X)
    P = softmax(Z, axis=1)
    delta_last = (P - Y) / B

    (W0, b0), (W1, b1) = nn.layers
    A1 = relu(X @ W0 + b0)
    dW_exp = A1.T @ delta_last
    db_exp = np.sum(delta_last, axis=0)

    grads = nn.compute_gradient(X, Y)
    dW_last, db_last = grads[-1]

    assert_close("dW_last", dW_last, dW_exp)
    assert_close("db_last", db_last, db_exp)
    print("\nSoftmax+CE closed-form last-layer test: OK.")


# -------------------------------
# 4) Symmetry test
# -------------------------------

def run_symmetry_test():
    rng = np.random.default_rng(3)
    Din, H, C = 3, 6, 4
    x = rng.normal(size=(1, Din))
    y = np.eye(C)[[rng.integers(0, C)]]

    X_dup = np.vstack([x, x])
    Y_dup = np.vstack([y, y])

    nn = NeuralNetwork(
        network_input_size=Din,
        layer_output_sizes=[H, C],
        activation_funcs=[relu, linear],
        activation_ders=[relu_deriv, linear_deriv],
        cost_fun=cross_entropy_with_logits,
        cost_der=cross_entropy_with_logits_deriv,
        seed=6114
    )

    out = nn.predict(X_dup)
    assert np.allclose(out[0], out[1]), "Forward mismatch for identical inputs."

    grads_dup = nn.compute_gradient(X_dup, Y_dup)
    grads_one = nn.compute_gradient(x, y)
    for (dW_d, db_d), (dW_1, db_1) in zip(grads_dup, grads_one):
        assert_close("dW symmetry", dW_d, dW_1)
        assert_close("db symmetry", db_d, db_1)

    print("\nSymmetry test: OK.")


# -------------------------------
# 5) Optimizer one-step consistency
# -------------------------------

def run_optimizer_consistency_test():
    rng = np.random.default_rng(4)
    B, Din, H, Dout = 5, 2, 4, 1
    X = rng.normal(size=(B, Din))
    Y = rng.normal(size=(B, Dout))

    nn = NeuralNetwork(
        network_input_size=Din,
        layer_output_sizes=[H, Dout],
        activation_funcs=[relu, linear],
        activation_ders=[relu_deriv, linear_deriv],
        cost_fun=mse,
        cost_der=mse_deriv,
        seed=6114,
    )

    grads = nn.compute_gradient(X, Y)

    lr = 1e-2
    updates_manual = [(-lr * dW, -lr * db) for (dW, db) in grads]

    nn_manual = clone_same_arch(nn, nn.cost_fun, nn.cost_der)
    nn_manual.update_weights(updates_manual)

    nn_opt = clone_same_arch(nn, nn.cost_fun, nn.cost_der)
    opt = SGD(lr=lr)
    updates = opt.step(nn_opt.layers, grads)
    nn_opt.update_weights(updates)

    th_manual, _, _ = flatten_params(nn_manual.layers)
    th_opt, _, _ = flatten_params(nn_opt.layers)
    assert_close("SGD parameter vectors", th_opt, th_manual)

    print("\nOptimizer (SGD) single-step consistency: OK.")


# -------------------------------
# 6) Sanity training
# -------------------------------

def make_blobs(n=200, centers=2, dim=2, spread=0.5, seed=42):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-2, 2, size=(centers, dim))
    X_list, y_list = [], []
    for c in range(centers):
        Xc = rng.normal(loc=means[c], scale=spread, size=(n // centers, dim))
        yc = np.full(n // centers, c)
        X_list.append(Xc); y_list.append(yc)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    Y = np.eye(centers)[y]
    return X, Y, y


def run_sanity_training():
    X, Y, y_idx = make_blobs(n=300, centers=3, dim=2, spread=0.30, seed=7)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    nn = NeuralNetwork(
        network_input_size=2,
        layer_output_sizes=[32, 3],
        activation_funcs=[relu, linear],
        activation_ders=[relu_deriv, linear_deriv],
        cost_fun=cross_entropy_with_logits,
        cost_der=cross_entropy_with_logits_deriv,
        seed=6114,
    )

    opt = Adam(lr=0.01)
    theta_before, shapes, keys = flatten_params(nn.layers)

    epochs = 400
    hist = nn.fit(
        X, Y,
        epochs=epochs,
        batch_size=32,
        optimizer=opt,
        shuffle=True,
        seed=6114,
        log_every=None
    )

    assert "train_loss" in hist
    assert len(hist["train_loss"]) == epochs

    theta_after, _, _ = flatten_params(nn.layers)
    delta_params = np.linalg.norm(theta_after - theta_before)
    assert delta_params > 0.0, "Parameters did not change during training."

    final_loss = hist["train_loss"][-1]
    probs = softmax(nn.predict(X), axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == y_idx).mean()

    print(f"\nSanity training: final loss = {final_loss:.4e}, acc = {acc:.3f}")
    assert acc >= 0.97, "Training accuracy too low."
    assert final_loss < 0.05, "Final loss too high."
    print("Sanity training: OK.")


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    run_numeric_checks()
    run_scaling_test()
    run_softmax_ce_closed_form_test()
    run_symmetry_test()
    run_optimizer_consistency_test()
    run_sanity_training()
    print("\nAll backpropagation tests passed")
