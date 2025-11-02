from typing import List, Tuple, Callable, Optional
import numpy as np

### Define functions for backpropagation (batch) and create_layers (batch) ###
# --- Initialiseringshjelpere ---
def _he_init(fan_in: int, fan_out: int, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(float)
    b = np.zeros((1, fan_out), dtype=float)
    return W, b

def _xavier_init(fan_in: int, fan_out: int, rng=None, gain: float = 1.0):
    if rng is None:
        rng = np.random.default_rng()
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    W = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(float)
    b = np.zeros((1, fan_out), dtype=float)
    return W, b

def _is_relu_like(act):
    # Tilpass gjerne med flere: leaky_relu, elu, gelu, etc.
    name = getattr(act, "__name__", "").lower()
    return "relu" in name  # fanger relu/leaky_relu

def _is_tanh_or_sigmoid(act):
    name = getattr(act, "__name__", "").lower()
    return ("tanh" in name) or ("sigmoid" in name)

def create_layers_batch(network_input_size, layer_output_sizes, activation_funcs=None, rng=None,
                        output_gain=0.5, class_priors=None):
    """
    Allmenn create_layers som velger init per lag basert på aktiveringsfunksjon:
      - ReLU-lignende: He-init
      - tanh/sigmoid: Xavier-init
      - Hvis ingen aktivering gjenkjennes: faller tilbake på Xavier
    Output-laget får Xavier med liten gain (default 0.5) for små logits/lineærutgang.
    Bias = 0 alltid, med unntak av klassifisering med skjeve klasser:
      hvis class_priors (shape (C,)) gis, settes siste bias = log(prior).
    
    Bruk:
      self.layers = create_layers_batch(input_dim, sizes, activation_funcs=self.activation_funcs)
    """
    if rng is None:
        rng = np.random.default_rng()

    layers = []
    fan_in = network_input_size

    # Hvis vi vet aktiveringsfunksjoner, bruk dem til å velge init per lag
    acts = list(activation_funcs) if activation_funcs is not None else [None] * len(layer_output_sizes)

    # Skjulte lag
    for fan_out, act in zip(layer_output_sizes[:-1], acts[:-1]):
        if act is not None and _is_relu_like(act):
            W, b = _he_init(fan_in, fan_out, rng=rng)
        elif act is not None and _is_tanh_or_sigmoid(act):
            W, b = _xavier_init(fan_in, fan_out, rng=rng, gain=1.0)
        else:
            # trygg default
            W, b = _xavier_init(fan_in, fan_out, rng=rng, gain=1.0)
        layers.append((W, b))
        fan_in = fan_out

    # Output-lag (små vekter uansett om det er lineært eller logits)
    out_dim = layer_output_sizes[-1]
    W_out, b_out = _xavier_init(fan_in, out_dim, rng=rng, gain=output_gain)

    # Hvis klassifisering med skjeve klasser: sett bias = log(prior)
    if class_priors is not None:
        pri = np.asarray(class_priors, dtype=float)
        pri = pri / np.sum(pri)
        b_out = np.log(pri)[None, :]

    layers.append((W_out, b_out))
    return layers


# Function used in backpropagation:
def feed_forward_saver_batch(inputs, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

def backpropagation_batch(
    inputs, layers, activation_funcs, target, activation_ders, cost_der):
    layer_inputs, zs, predict = feed_forward_saver_batch(inputs, layers, activation_funcs)

    layer_grads = [() for layer in layers]
    
    dC_dz_next = None
    
    B = inputs.shape[0]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict, target)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W_next, b_next) = layers[i + 1]
            dC_da = dC_dz_next @ W_next.T

        dC_dz = dC_da*activation_der(z)
        # Grads (average over batch)
        dC_dW = (layer_input.T @ dC_dz)
        dC_db = np.sum(dC_dz, axis=0, keepdims=True)

        layer_grads[i] = (dC_dW, dC_db)
        
        dC_dz_next = dC_dz

    return layer_grads



Array = np.ndarray
Layer = Tuple[Array, Array]          
Grads = List[Tuple[Array, Array]] 

class NeuralNetwork:
    def __init__(
        self,
        network_input_size: int,
        layer_output_sizes: List[int],
        activation_funcs: List[Callable[[Array], Array]],
        activation_ders: List[Callable[[Array], Array]],
        cost_fun: Callable,
        cost_der: Callable,
    ):
        """Setting up neural network with given layers, activation functions
        and cost function.

        Args:
            network_input_size (int): 
                Number of input neurons
            layer_output_sizes (List[int]): 
                List containing number of nodes in each layer
            activation_funcs (List[Callable[[Array], Array]]):
                List of activation functions (one for each layer)
            activation_ders (List[Callable[[Array], Array]]):
                List of derivatives of activation functions
            cost_fun (Callable): 
                Cost function. 
            cost_der (Callable): 
                Derivative of cost function
        """
        # Checking that the parameters line up (shapes):
        assert len(layer_output_sizes) == len(activation_funcs) == len(activation_ders), (
            "Number of layers, activation functions and derivatives of activation functions must be equal."
        )
        
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        
        self.layers = create_layers_batch(network_input_size, 
                                          layer_output_sizes,
                                          activation_funcs=self.activation_funcs)

    def predict(self, inputs):
        """Perform a forward pass through the neural network

        Args:
            inputs (Array): 
                Input data of shape (B, in_dim), where B is the batch size
        
        Returns: 
        np.ndarray:
            The network output after the final activation function. Typically 
            represents probabilities if the last activation is a softmax layer. 
        """
        a = inputs
        for (W, b), act in zip(self.layers, self.activation_funcs):
            a = act(a @ W + b)
        
        return a

    def cost(self, inputs, targets):
        """ Compute the loss for a given batch using the network's configured
        loss function.

        Args:
            inputs (np.ndarray): 
                Input data of shape (B, in_dim), where B is the batch size
            targets (np.ndarray):
                Target labels in one-hot (or appropriate) format matching the 
                output shape of the network, typically (B, out_dim).

        Returns:
        float: 
            The scalar loss value computed by `self.cost_fun`
        """
        preds = self.predict(inputs)
        return self.cost_fun(preds, targets)

    def _feed_forward_saver(self, inputs):
        """Perform a forward pass while storing intermediate values
        needed for backpropagation.

        Args:
            inputs (np.ndarray): 
                Input data of shape (B, in_dim), where B is the batch size.

        Returns:
        tuple:
            (layer_inputs, zs, a)
            - layer_inputs: list of activations before each layer (a_l)
            - zs: list of pre-activation values (z_l)
            - a: final output after the last activation
        """
        layer_inputs = []
        zs = []
        a = inputs
        
        for (W, b), act in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            
            z = a @ W + b
            zs.append(z)
            
            a = act(z)
        
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        """Compute parameter gradients for a given batch using the configured 
        cost derivative.

        This method is a thin wrapper around the existing `backpropagation_batch`
        implementation. It does not perform any optimization step; it only returns
        gradients so that an external optimizer can produce weight updates.

        Args:
            inputs (np.ndarray): 
                Input batch of shape (B, in_dim).
            targets (np.ndarray): 
                Target batch of shape (B, out_dim), typically one-hot for classification.

        Returns:
        list[tuple[np.ndarray, np.ndarray]]:
            A list of (dW, db) tuples, one per layer, matching `self.layers` order.
            Shapes:
              - dW: (in_dim_l, out_dim_l)
              - db: (1, out_dim_l)
              
        Notes:
        - Uses `self.cost_der` provided at construction time (e.g., softmax CE derivative).
        - For softmax + cross-entropy, ensure your last activation derivative is identity
          (so you don’t multiply by the softmax derivative again).
        """
        return backpropagation_batch(
            inputs=inputs,
            layers=self.layers,
            activation_funcs=self.activation_funcs,
            activation_ders=self.activation_ders,
            target=targets,
            cost_der=self.cost_der
            ) 

    def update_weights(self, layer_grads):
        """Apply parameter updates to the network layers.

        Args:
            layer_grads : list[tuple[np.ndarray, np.ndarray]]
                A list of (dW, db) tuples, one per layer, matching the ordering of `self.layers`.
                IMPORTANT: These should be *updates* (deltas), not raw gradients.
                For example:
                - Plain SGD:           (dW, db) = (-lr * gradW, -lr * gradb)
                - SGD with momentum:   (dW, db) = (vW, vB) where v = beta*v - lr*grad

        Notes:
        - This method does not implement any optimization logic (no lr, no momentum).
          It simply adds the provided updates to the current weights and biases.
        - Shapes must match each layer:
              W: (in_dim_l, out_dim_l),  b: (1, out_dim_l)
             dW: (in_dim_l, out_dim_l), db: (1, out_dim_l)
        """
        new_layers = []
        for (W, b), (dW, db) in zip(self.layers, layer_grads):
            new_W = W + dW
            new_b = b + db
            new_layers.append((new_W, new_b))
        
        self.layers = new_layers
        
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        optimizer=None,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        log_every: int = 1,
        callback=None,
        ):
        """
        Tren nettverket ved å koble sammen compute_gradient -> optimizer.step -> update_weights.
        Importerer ikke optimizere; du sender inn et optimizer-objekt utenfra.

        Args:
            X, Y: Treningsdata (B x D_in), (B x D_out).
            epochs: Antall epoker.
            batch_size: Minibatch-størrelse.
            optimizer: Objekt med signatur step(layers, grads) -> updates.
            X_val, Y_val: (valgfritt) valideringsdatasett for logging.
            shuffle: Om batcher skal shuffles per epoch.
            seed: (valgfritt) seed til shuffling.
            log_every: Logg hver n-te epoch.
            callback: (valgfritt) funksjon f(self, epoch, train_loss, val_loss) kalt etter hver epoch.

        Returns:
            history: dict med lister 'train_loss' og (hvis gitt val) 'val_loss'.
        """
        if optimizer is None:
            raise ValueError("Du må sende inn et optimizer-objekt (f.eks. Adam(lr=1e-3)).")

        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = {"train_loss": []}
        if X_val is not None and Y_val is not None:
            history["val_loss"] = []

        def iter_minibatches(Xa, Ya, bs, do_shuffle: bool):
            idx = np.arange(Xa.shape[0])
            if do_shuffle:
                rng.shuffle(idx)
            for start in range(0, Xa.shape[0], bs):
                sl = idx[start : start + bs]
                yield Xa[sl], Ya[sl]

        for epoch in range(1, epochs + 1):
            # --- trenings-epoch ---
            for xb, yb in iter_minibatches(X, Y, batch_size, shuffle):
                grads = self.compute_gradient(xb, yb)       # [(dW, db), ...]
                updates = optimizer.step(self.layers, grads) # [(ΔW, Δb), ...]
                self.update_weights(updates)                 # legg til oppdateringene

            # --- logging ---
            train_loss = self.cost(X, Y)
            history["train_loss"].append(train_loss)

            val_loss = None
            if X_val is not None and Y_val is not None:
                val_loss = self.cost(X_val, Y_val)
                history["val_loss"].append(val_loss)

            if (log_every is not None) and (epoch % log_every == 0):
                if val_loss is None:
                    print(f"Epoch {epoch:3d} | train: {train_loss:.6f}")
                else:
                    print(f"Epoch {epoch:3d} | train: {train_loss:.6f} | val: {val_loss:.6f}")

            if callback is not None:
                # callback(self, epoch, train_loss, val_loss)
                callback(self, epoch, train_loss, val_loss)

        return history


