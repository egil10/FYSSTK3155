from typing import List, Tuple, Callable, Optional
import numpy as np

### Define functions for backpropagation (batch) and create_layers (batch) ###

def create_layers_batch(network_input_size, layer_output_sizes, activation_funcs=None, seed=6114):
    """Initialize a list of layers for a neural network.
    Each layer consists of a weight matrix W and a bias vector b.  
    The weights are initialized from a normal distribution with mean 0 and 
    standard deviation 0.01, and all biases are set to zero.  
    The initialization does not depend on the activation function.

    Args:
        network_input_size (int): 
            Number of input features to the network.
        layer_output_sizes (list[int]): 
            A list specifying the number of neurons in each layer,
            including the output layer.
        activation_funcs (list[Callable], optional): 
            Placeholder for activation functions 
            (ignored in this simple initializer). Included only for interface compatibility 
            Defaults to None.
        seed (int, optional): 
            Random seed for reproducible layers. Defaults to 6114.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: 
            A list of tuples (W, b) for each layer, where:
            - W has shape (fan_in, fan_out)
            - b has shape (fan_out,)
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
    """Forward pass through all the layers (batch).
    Used in backpropagation

    Args:
        inputs (np.ndarray): 
            input data of shape (n_samples, n_features)
        layers (list[tuple[np.ndarray, np.ndarray]]): 
            (W, b) pair for each layer
        activation_funcs (list[Callable]): 
            Activation functions for each layer

    Returns:
        layer_inputs (list[np.ndarray]): 
            Activations entering each layer
        zs (list[np.ndarray]): 
            Pre-activation values for each layer
        a (np.ndarray): 
            Final network output
    """
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
    """Compute gradients for all layers using backpropagation.
    Args:
        inputs (np.ndarray): 
            input data of shape (batch_size, n_features)
        layers (list[tuple[np.ndarray, np.ndarray]]): 
            (W, b) pairs for each layer
        activation_funcs (list[Callable]): 
            Activation functions for each layer
        target (np.ndarray): 
            Target values, shape (batch_size, n_outputs)
        activation_ders (list[Callable]): 
            Derivatives of the activation functions
        cost_der (Callable): 
            Derivative of the cost function w.r.t. predictions

    Returns:
        layer_grads (list[tuple[np.ndarray, np.ndarray]]):  
            Gradients (dC/dW, dC/db) for each layer
    """

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
        dC_dW = (layer_input.T @ dC_dz)
        dC_db = np.sum(dC_dz, axis=0)

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
        seed: int, 
        initial_layers: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        l2_lambda: float = 0.0, 
        l1_lambda: float = 0.0,
        reg_on_bias: bool = False,
    ):
        """Initialize fully connected Neural Network

        Args:
            network_input_size (int): 
                Number of input features to the network.
            layer_output_sizes (List[int]): 
                Number of neurons in each layer, including the output layer.
            activation_funcs (List[Callable[[Array], Array]]):
                Activation functions for each layer
            activation_ders (List[Callable[[Array], Array]]): 
                Derivatives of the activation functions, used in backpropagation.
            cost_fun (Callable): 
                Cost function used in training
            cost_der (Callable): 
                Derivative of cost function w.r.t. predictions
            seed (int): 
                Random seed for reproducible weight initialization.
            initial_layers ([List[Tuple[np.ndarray, np.ndarray]]], optional): 
                Custom initialization for weights and biases.
                Each tuple should contain (W, b) for one layer.
                If not provided, layers are initialized with small random values. 
                Defaults to None.
            l2_lambda (float, optional): 
                Strength of L2 regularization. Defaults to 0.0.
            l1_lambda (float, optional): 
                Strength of L1 regularization. Defaults to 0.0.
            reg_on_bias (bool, optional): 
                If True, applies regularization terms to biases as well. 
                Defaults to False.
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
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.reg_on_bias = reg_on_bias
        
        if initial_layers is not None:
            fan_in = network_input_size
            for (W, b), fan_out in zip(initial_layers, layer_output_sizes):
                assert W.shape == (fan_in, fan_out), f"W has shape {W.shape}, expected {(fan_in, fan_out)}"
                if b.ndim == 2 and b.shape[0] == 1:
                    b = b.reshape(-1)
                assert b.shape == (fan_out,), f"b has shape {b.shape}, expected {(fan_out,)}"
                fan_in = fan_out
            self.layers = [(W.astype(np.float64).copy(), b.astype(np.float64).copy())
                           for (W, b) in initial_layers]
        else:
            self.layers = create_layers_batch(network_input_size, 
                                          layer_output_sizes,
                                          activation_funcs=self.activation_funcs,
                                          seed = self.seed)

    def predict(self, inputs):
        """Perform a forward pass through the network to produce predictions.

        Args:
            inputs (np.ndarray): 
                Input data of shape (batch_size, n_features).

        Returns:
            np.ndarray: 
                Network output after the final activation function.
        """

        a = inputs
        for (W, b), act in zip(self.layers, self.activation_funcs):
            a = act(a @ W + b)
        
        return a

    def cost(self, inputs, targets):
        """Compute the total loss for a given batch, including regularization.

        Args:
            inputs (np.ndarray): 
                Input data of shape (batch_size, n_features).
            targets (np.ndarray): 
                Target values or labels matching the output shape of the network,
                typically (batch_size, n_outputs).

        Returns:
            float: 
                The scalar loss value computed from the cost function,
                optionally including L1 and L2 regularization penalties.
        """

        preds = self.predict(inputs)
        base_loss = self.cost_fun(preds, targets)
        # Regularization terms:
        B = inputs.shape[0]
        l2_term = 0.0
        l1_term = 0.0
        if self.l2_lambda > 0.0:
            for (W, b) in self.layers:
                l2_term += np.sum(W ** 2)
                if self.reg_on_bias:
                    l2_term += np.sum(b ** 2)
            l2_term = (self.l2_lambda / B) * l2_term
            
        if self.l1_lambda > 0.0:
            for (W, b) in self.layers:
                l1_term += np.sum(np.abs(W))
                if self.reg_on_bias:
                    l1_term += np.sum(np.abs(b))
            l1_term = (self.l1_lambda / B) * l1_term
        
        return base_loss + l2_term + l1_term



    def compute_gradient(self, inputs, targets):
        """Compute the gradients of the cost function w.r.t. all network parameters.
        
        This performs a full backpropagation pass for the given batch and returns
        weight and bias gradients without applying any updates.

        Args:
            inputs (np.ndarray): 
                Input data of shape (batch_size, n_features).
            targets (np.ndarray): 
                Target values or labels of shape (batch_size, n_outputs).

        Returns:
            list[tuple[np.ndarray, np.ndarray]]:
                Gradients (dW, db) for each layer in self.layers.
                    - dW : same shape as corresponding weight matrix
                    - db : same shape as corresponding bias vector
        Notes:
            - Uses the cost derivative `self.cost_der` defined at initialization.
            - L1 and L2 regularization terms are added to the gradients if enabled.
            - Does not modify the model parameters. Use within an optimizer or training loop.
    
        """

        grads = backpropagation_batch(
            inputs=inputs,
            layers = self.layers,
            activation_funcs=self.activation_funcs,
            activation_ders=self.activation_ders,
            target=targets,
            cost_der=self.cost_der
        )
        # Regularization:
        B = inputs.shape[0]
        if self.l2_lambda > 0.0 or self.l1_lambda > 0.0:
            new_grads = []
            for (W, b), (dW, db) in zip(self.layers, grads):
                if self.l2_lambda > 0.0:
                    dW = dW + (2.0 * self.l2_lambda / B) * W
                    if self.reg_on_bias:
                        db = db + (2.0 * self.l2_lambda / B) * b
                if self.l1_lambda > 0.0:
                    dW = dW + (self.l1_lambda / B) * np.sign(W)
                    if self.reg_on_bias:
                        db = db + (self.l1_lambda / B) * np.sign(b)
                new_grads.append((dW, db))
            grads = new_grads
        
        return grads 

    def update_weights(self, layer_grads):
        """Apply weight and bias updates to all layers.

        Args:
            layer_grads (list[tuple[np.ndarray, np.ndarray]]):
                List of (dW, db) tuples matching the order of `self.layers`.
                These should represent *update steps* (e.g., from an optimizer),
                not raw gradients.
        Notes:
            - This method performs only the parameter update step (`W += dW`, `b += db`).
                Learning rate and optimizer logic must be handled externally.
            - The shapes of all updates must exactly match their corresponding parameters.
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
            seed: int = 6114,
            log_every: int = 1,
            grad_tol: float | None = None,
        ):
        """Train the neural network using minibatch gradient descent.

        Args:
            X (np.ndarray): 
                Training inputs of shape (n_samples, n_features).
            Y (np.ndarray): 
                Training targets of shape (n_samples, n_outputs).
            epochs (int, optional): 
                Number of training epochs. Defaults to 50.
            batch_size (int, optional): 
                Number of samples per mini-batch. Defaults to 32.
            optimizer (object): 
                Optimizer instance implementing a .step(layers, grads) method.
                Examples include SGD, Adam, or RMSProp.
            shuffle (bool, optional): 
                Whether to shuffle the data at the start of each epoch. 
                Defaults to True.
            seed (int, optional): 
                Random seed for reproducible shuffling. Defaults to 6114.
            log_every (int, optional): 
                Print training loss every `log_every` epochs. 
                Set to None to disable logging.
                Defaults to 1.
            grad_tol (float, optional): 
                Optional stopping criterion based on gradient norm. 
                Training stops early if the total gradient magnitude falls below this value.
                Defaults to None.

        Raises:
            ValueError: 
                If no optimizer is given

        Returns:
            dict:
                Training history containing:
                - "train_loss" : list of loss values per epoch.
        Notes:
            - Setting the batch size to the full dataset size and setting 
                shuffle=False yields deterministic (non-stochastic) full-batch
                gradient descent.
        """
        if optimizer is None:
            raise ValueError("You must provide an optimizer object (e.g., SGD(lr=1e-3)).")

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

        # Full-batch if bs >= n and no shuffling
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
