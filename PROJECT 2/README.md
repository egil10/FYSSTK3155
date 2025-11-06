# Project 2: Neural Networks for Classification and Regression

**Course:** Data Analysis and Machine Learning (FYS-STK3155/FYS4155)  
**Institution:** University of Oslo  
**Project Type:** Implementation and Analysis of Feed-Forward Neural Networks

## Project Overview

This project implements a complete feed-forward neural network (FFNN) framework from scratch, including the backpropagation algorithm, various activation functions, loss functions, and optimization methods. The implementation is applied to both regression and classification problems, with comprehensive comparisons against established machine learning methods from Project 1 and standard libraries.

The project addresses two primary tasks:
1. **Regression:** Fitting the one-dimensional Runge function using neural networks and comparing results with Ordinary Least Squares (OLS), Ridge, and Lasso regression from Project 1.
2. **Classification:** Multiclass classification on the MNIST dataset using softmax cross-entropy loss.

## Repository Structure

```
PROJECT 2/
├── Code/
│   ├── Implementations/          # Core neural network implementation
│   │   ├── neural_network.py     # Main FFNN class and backpropagation
│   │   ├── activations.py        # Activation functions (Sigmoid, ReLU, LeakyReLU, Softmax, Linear)
│   │   ├── losses.py             # Loss functions (MSE, Cross-Entropy, BCE)
│   │   ├── optimizers.py         # Optimization algorithms (SGD, RMSprop, Adam, Momentum, Adagrad)
│   │   ├── prepare_data.py       # Data preparation utilities
│   │   └── plot_style.py         # Plotting utilities and style configuration
│   ├── Code_Project1/            # Reused code from Project 1
│   │   ├── OLS.py                # Ordinary Least Squares regression
│   │   ├── Ridge.py              # Ridge regression
│   │   └── polynomial_features.py # Polynomial feature generation
│   ├── Notebooks/                # Jupyter notebooks for analysis
│   │   ├── Exa.ipynb             # Part a: Analytical warm-up (cost functions, activations)
│   │   ├── Exb.ipynb             # Part b: Basic FFNN implementation for regression
│   │   ├── Exb_with_noise.ipynb  # Part b: Regression with noise analysis
│   │   ├── Exc.ipynb             # Part c: Comparison with Scikit-Learn/TensorFlow
│   │   ├── Exd.ipynb             # Part d: Testing different activation functions and network depths
│   │   ├── Exe.ipynb             # Part e: Testing L1 and L2 regularization
│   │   └── Exf.ipynb             # Part f: Classification on MNIST dataset
│   ├── Plots/                    # Generated figures and visualizations
│   ├── Tables/                   # Generated tables and results
│   ├── Testing/                  # Unit tests for implementations
│   │   ├── test_activations.py
│   │   ├── test_backpropagation.py
│   │   ├── test_losses.py
│   │   ├── test_neural_network.py
│   │   └── test_optimizers.py
│   └── requirements.txt          # Python package dependencies
├── project2.ipynb                # Project assignment document
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or navigate to the project directory:
   ```bash
   cd "PROJECT 2/Code"
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   The core dependencies are:
   - `numpy`: Numerical computations and array operations
   - `matplotlib`: Plotting and visualization
   - `scikit-learn`: Data preprocessing, train-test splitting, and comparison models

3. For running Jupyter notebooks, install Jupyter:
   ```bash
   pip install jupyter
   ```

## Implementation Details

### Core Components

#### Neural Network Architecture (`neural_network.py`)
- Flexible feed-forward architecture supporting arbitrary numbers of layers and neurons
- Batch processing with efficient matrix operations
- Backpropagation algorithm for gradient computation
- Support for multiple activation functions per layer
- Configurable loss functions (MSE for regression, Cross-Entropy for classification)
- Optional L1 and L2 regularization

#### Activation Functions (`activations.py`)
- **Sigmoid**: Stable implementation with overflow protection
- **ReLU**: Rectified Linear Unit
- **Leaky ReLU**: Variant with small negative slope
- **Linear/Identity**: For output layers in regression
- **Softmax**: For multiclass classification output layers

All activation functions include corresponding derivative implementations for backpropagation.

#### Loss Functions (`losses.py`)
- **Mean Squared Error (MSE)**: For regression problems
- **Binary Cross-Entropy**: For binary classification (with stable logits variant)
- **Multiclass Cross-Entropy**: For classification with softmax (with stable logits variant)

All loss functions include gradient computations for backpropagation.

#### Optimization Algorithms (`optimizers.py`)
- **SGD**: Plain stochastic gradient descent
- **Momentum**: SGD with momentum
- **Adagrad**: Adaptive gradient algorithm
- **RMSprop**: Root Mean Square Propagation
- **Adam**: Adaptive Moment Estimation

All optimizers support gradient clipping and configurable learning rates.

### Key Features

- **Modular Design**: Separate modules for activations, losses, and optimizers enable easy experimentation
- **Reproducibility**: Random seed control for weight initialization and data splitting
- **Efficiency**: Vectorized operations using NumPy for batch processing
- **Extensibility**: Clean interfaces allow easy addition of new activation functions, losses, or optimizers
- **Validation**: Unit tests verify correctness of gradient computations and network behavior

## Notebook Descriptions

### Exa.ipynb: Analytical Warm-up
Derives and documents the mathematical expressions for:
- Cost functions: MSE, Binary Cross-Entropy, Multiclass Cross-Entropy (with and without L1/L2 regularization)
- Activation functions: Sigmoid, ReLU, Leaky ReLU (expressions and derivatives)
- Gradient computations for all components

### Exb.ipynb: Basic Neural Network for Regression
- Implements and trains FFNN on the one-dimensional Runge function
- Compares neural network performance with OLS regression from Project 1
- Analyzes learning curves for different optimizers (SGD, RMSprop, Adam)
- Explores optimal learning rates and network architectures (1-2 hidden layers, 50-100 neurons)

### Exb_with_noise.ipynb: Regression with Noise Analysis
- Extends Exb.ipynb with noise analysis
- Studies neural network robustness to noisy data
- Compares performance degradation with OLS under noise

### Exc.ipynb: Comparison with Standard Libraries
- Validates implementation against Scikit-Learn's MLPRegressor/MLPClassifier
- Optional comparison with TensorFlow/Keras or PyTorch
- Uses automatic differentiation (Autograd/JAX) to verify gradient computations

### Exd.ipynb: Activation Functions and Network Depth
- Systematic comparison of Sigmoid, ReLU, and Leaky ReLU activation functions
- Analysis of network depth (number of layers) and width (neurons per layer)
- Overfitting analysis and bias-variance trade-off considerations
- Hyperparameter heatmaps for learning rate and network architecture

### Exe.ipynb: Regularization Analysis
- Implements L1 and L2 regularization in the neural network
- Compares L2-regularized networks with Ridge regression from Project 1
- Compares L1-regularized networks with Lasso regression from Project 1
- Hyperparameter optimization for regularization strength (λ) and learning rate

### Exf.ipynb: MNIST Classification
- Applies the neural network to multiclass classification on the full MNIST dataset
- Uses softmax cross-entropy loss for 10-class classification
- Analyzes classification accuracy as a function of:
  - Network architecture (depth and width)
  - Activation functions
  - Learning rates and optimization algorithms
  - Regularization parameters
- Optional comparisons with Logistic Regression and standard ML libraries
- Optional confusion matrix analysis for best-performing model

## Results and Outputs

### Generated Artifacts

- **Plots/**: Contains PDF figures including:
  - Learning curves for different optimizers
  - Comparison plots between neural networks and OLS/Ridge/Lasso
  - Hyperparameter heatmaps
  - Network performance visualizations

- **Tables/**: Contains tabulated results and performance metrics

### Key Findings

The notebooks systematically explore:
1. Neural network performance relative to linear regression methods
2. Impact of activation function choice on training dynamics and final performance
3. Optimal network architectures for given problem complexity
4. Effectiveness of different optimization algorithms
5. Regularization effects and comparison with traditional methods
6. Classification performance on high-dimensional image data

## Usage

### Running Individual Notebooks

1. Navigate to the Code directory:
   ```bash
   cd "PROJECT 2/Code"
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open the desired notebook from the `Notebooks/` folder

4. Ensure the notebook kernel can access the `Implementations/` module (notebooks include path setup)

### Running Tests

Execute unit tests to verify implementation correctness:
```bash
cd "PROJECT 2/Code"
python -m pytest Testing/
```

### Reproducing Results

All notebooks use fixed random seeds for reproducibility. To reproduce exact results:
- Ensure NumPy random seed is set (default: 6114 for network initialization)
- Use the same train-test split random state (typically 42)
- Follow the exact hyperparameter values specified in the notebooks

## Project Requirements

This project fulfills the requirements for Project 2 of FYS-STK3155/FYS4155:

- ✅ Custom FFNN implementation with backpropagation
- ✅ Support for regression (MSE) and classification (Cross-Entropy)
- ✅ Multiple activation functions (Sigmoid, ReLU, LeakyReLU)
- ✅ Multiple optimization algorithms (SGD, RMSprop, Adam)
- ✅ L1 and L2 regularization
- ✅ Comparison with Project 1 methods (OLS, Ridge, Lasso)
- ✅ Comparison with standard ML libraries
- ✅ Comprehensive analysis and visualization

## References

- Nielsen, M. *Neural Networks and Deep Learning*. http://neuralnetworksanddeeplearning.com/
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*. https://www.deeplearningbook.org/
- Raschka, S., et al. *Machine Learning with PyTorch and Scikit-Learn*. https://sebastianraschka.com/blog/2022/ml-pytorch-book.html
- Scikit-Learn Documentation: https://scikit-learn.org/
- Course Lecture Notes: https://compphysics.github.io/MachineLearning/

## License

This project is part of coursework for FYS-STK3155/FYS4155 at the University of Oslo.

