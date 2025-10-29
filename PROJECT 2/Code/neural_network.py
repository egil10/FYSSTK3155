from typing import List, Tuple, Callable, Optional

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
        
        self.layers = create_layers_batch(network_input_size, layer_output_sizes)

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