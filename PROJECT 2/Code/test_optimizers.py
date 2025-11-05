import numpy as np
from neural_network import NeuralNetwork
from activations import sigmoid, sigmoid_deriv, linear, linear_deriv
from losses import mse, mse_deriv
from optimizers import SGD, RMSprop, Adam

# Enkel treningsfunksjon y = x^2
seed = 6114
rng = np.random.default_rng(seed)
X = rng.uniform(-1, 1, size=(200, 1))
Y = X**2

# Lager en liten modell (1 -> 10 -> 1)
def make_model():
    return NeuralNetwork(
        network_input_size=1,
        layer_output_sizes=[10, 1],
        activation_funcs=[sigmoid, linear],
        activation_ders=[sigmoid_deriv, linear_deriv],
        cost_fun=mse,
        cost_der=mse_deriv,
        seed=6114,
    )

# Treningsfunksjon som bruker fit()
def train_model(nn, optimizer, name, epochs=200, batch_size=32):
    print(f"\n=== {name.upper()} ===")
    history = nn.fit(
        X, Y,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        log_every=50
    )
    print(f"Final {name} loss: {history['train_loss'][-1]:.6f}")

# ---------- TEST 1: PLAIN SGD ----------
nn_sgd = make_model()
opt_sgd = SGD(lr=0.05)
train_model(nn_sgd, opt_sgd, name="SGD")

# ---------- TEST 2: RMSPROP ----------
nn_rms = make_model()
opt_rms = RMSprop(lr=0.01, rho=0.9)
train_model(nn_rms, opt_rms, name="RMSProp")

# ---------- TEST 3: ADAM ----------
nn_adam = make_model()
opt_adam = Adam(lr=0.01)
train_model(nn_adam, opt_adam, name="Adam")
