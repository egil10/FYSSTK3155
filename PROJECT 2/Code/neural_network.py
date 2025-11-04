from typing import List, Tuple, Callable, Optional
import numpy as np

### Define functions for backpropagation (batch) and create_layers (batch) ###

def create_layers_batch(network_input_size, layer_output_sizes, activation_funcs=None, seed=6114):
    """
    Creates layers with simple initialization:
      - Weights from  Normal distribution(mean=0, std=0.01)
      - Biases = 0
    Activation functions and class priors are ignored.
    
    Returns:
      List of (W, b) tuples for each layer.
    """
    rng = np.random.default_rng(seed)
    

    layers = []
    fan_in = network_input_size

    for fan_out in layer_output_sizes:
        W = rng.normal(loc=0.0, scale=0.01, size=(fan_in, fan_out))
        b = np.zeros((fan_out,))
        layers.append((W, b))
        fan_in = fan_out

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
        seed: int
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
        self.seed = seed
        
        self.layers = create_layers_batch(network_input_size, 
                                          layer_output_sizes,
                                          activation_funcs=self.activation_funcs,
                                          seed = self.seed)

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
        
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            optimizer=None,
            shuffle: bool = True,
            seed: int | None = None,
            log_every: int = 1,
            grad_tol: float | None = None,
        ):
        if optimizer is None:
            raise ValueError("Du må sende inn et optimizer-objekt (f.eks. SGD(lr=1e-3)).")

        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = {"train_loss": []}

        def iter_minibatches(Xa, Ya, bs, do_shuffle: bool):
            idx = np.arange(Xa.shape[0])
            if do_shuffle:
                rng.shuffle(idx)
            for start in range(0, Xa.shape[0], bs):
                sl = idx[start:start + bs]
                yield Xa[sl], Ya[sl]

        # Full-batch hvis bs >= n og ingen stokastikk
        is_full_batch = (batch_size == n) and (shuffle is False)

        for epoch in range(1, epochs + 1):
            last_grads = None
            for xb, yb in iter_minibatches(X, Y, batch_size, shuffle):
                grads = self.compute_gradient(xb, yb)        # [(dW, db), ...]
                last_grads = grads
                updates = optimizer.step(self.layers, grads)  # [(ΔW, Δb), ...]
                self.update_weights(updates)

            train_loss = self.cost(X, Y)
            history["train_loss"].append(train_loss)

            if (log_every is not None) and (epoch % log_every == 0 or epoch == 1):
                print(f"Epoch {epoch:3d} | train: {train_loss:.6f}")

            if is_full_batch and grad_tol is not None and last_grads is not None:
                sqsum = sum(float(np.sum(dW * dW)) + float(np.sum(db * db)) for dW, db in last_grads)
                grad_norm = float(np.sqrt(sqsum))
                if not np.isfinite(grad_norm):
                    print(f"[Grad-norm stop] epoch {epoch} | ||grad|| is {grad_norm}.")
                    break
                if grad_norm < grad_tol:
                    print(f"[Grad-norm stop] epoch {epoch} | ||grad||={grad_norm:.3e} < {grad_tol:.3e}")
                    break

        return history
