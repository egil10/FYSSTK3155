# FYS-STK3155/FYS4155: Data Analysis and Machine Learning

**Course:** Data Analysis and Machine Learning (FYS-STK3155/FYS4155)  
**Institution:** University of Oslo  
**Academic Year:** 2024-2025

This repository contains implementations, analyses, and reports for the three main projects in the Data Analysis and Machine Learning course. The projects progressively build from fundamental linear regression methods to advanced neural network architectures, providing hands-on experience with both theoretical foundations and practical implementations.

## Repository Overview

This repository is organized into three main projects, each focusing on different aspects of machine learning:

- **PROJECT 1**: Linear Regression and Gradient Descent Methods
- **PROJECT 2**: Feed-Forward Neural Networks
- **PROJECT 3**: Advanced Topics (in progress)

Each project includes complete implementations, comprehensive analyses, Jupyter notebooks, generated visualizations, and detailed reports.

## Project Structure

```
FYSSTK3155/
├── PROJECT 1/              # Linear Regression and Optimization
│   ├── Code/               # Implementation and analysis code
│   ├── README.md           # Project 1 documentation
│   └── FYS_STK3155_PROJECT_1_The_Great_Regression.pdf
│
├── PROJECT 2/              # Neural Networks
│   ├── Code/               # Neural network implementation
│   ├── README.md           # Project 2 documentation
│   └── project2.ipynb      # Project assignment
│
├── PROJECT 3/              # Advanced Topics
│   └── (in progress)
│
└── README.md               # This file
```

## PROJECT 1: Linear Regression and Gradient Descent

**Focus:** Fundamental regression methods, optimization algorithms, and model evaluation techniques.

### Objectives

- Implement and analyze Ordinary Least Squares (OLS), Ridge, and LASSO regression
- Develop gradient descent optimization algorithms from scratch
- Study bias-variance tradeoff using resampling methods
- Compare closed-form solutions with iterative optimization methods

### Key Components

**Core Implementations:**
- `OLS.py`: Closed-form OLS parameter estimation
- `Ridge.py`: Closed-form Ridge regression with L2 regularization
- `gradient_descent.py`: Standard gradient descent for OLS, Ridge, and LASSO
- `stochastic_gradient_descent.py`: Stochastic variants with advanced optimizers (Momentum, Adagrad, RMSprop, Adam)
- `polynomial_features.py`: Polynomial feature matrix generation
- `prepare_data.py`: Data preparation and Runge function definition

**Analysis Notebooks:**
- `Exa.ipynb`: OLS regression analysis on Runge function
- `Exb.ipynb`: Ridge regression with hyperparameter tuning
- `Exc.ipynb`: Validation of gradient descent implementations
- `Exd.ipynb`: Advanced gradient descent methods comparison
- `Exe.ipynb`: LASSO regression implementation and validation
- `Exf.ipynb`: Stochastic gradient descent methods
- `Exg.ipynb`: Bias-variance analysis using bootstrap resampling
- `Exh.ipynb`: Bias-variance analysis using k-fold cross-validation

### Main Results

- Comprehensive comparison of OLS, Ridge, and LASSO regression methods
- Performance analysis of various gradient descent algorithms
- Systematic study of bias-variance tradeoff through resampling techniques
- Hyperparameter optimization for regularization parameters
- Validation against Scikit-Learn implementations

### Dependencies

- NumPy
- Scikit-Learn
- Matplotlib
- Pandas

For detailed information, see [PROJECT 1/README.md](PROJECT%201/README.md).

## PROJECT 2: Feed-Forward Neural Networks

**Focus:** Building neural networks from scratch, implementing backpropagation, and applying to regression and classification problems.

### Objectives

- Implement a complete feed-forward neural network framework from scratch
- Develop the backpropagation algorithm for gradient computation
- Test multiple activation functions (Sigmoid, ReLU, LeakyReLU)
- Implement various optimization algorithms (SGD, RMSprop, Adam)
- Apply neural networks to regression and classification tasks
- Compare neural network performance with traditional methods

### Key Components

**Core Implementations:**
- `neural_network.py`: Main FFNN class with backpropagation algorithm
- `activations.py`: Activation functions (Sigmoid, ReLU, LeakyReLU, Softmax, Linear) with derivatives
- `losses.py`: Loss functions (MSE, Cross-Entropy, BCE) with gradient computations
- `optimizers.py`: Optimization algorithms (SGD, Momentum, Adagrad, RMSprop, Adam)
- `prepare_data.py`: Data preparation utilities
- `plot_style.py`: Visualization utilities

**Analysis Notebooks:**
- `Exa.ipynb`: Analytical derivation of cost functions and activation derivatives
- `Exb.ipynb`: Basic neural network implementation for regression on Runge function
- `Exb_with_noise.ipynb`: Regression analysis with noise
- `Exc.ipynb`: Validation against Scikit-Learn and automatic differentiation
- `Exd.ipynb`: Activation function and network architecture analysis
- `Exe.ipynb`: L1 and L2 regularization analysis
- `Exf.ipynb`: MNIST classification using neural networks

**Testing:**
- Comprehensive unit tests for all components
- Gradient verification using automatic differentiation
- Validation against standard ML libraries

### Main Results

- Successful implementation of neural networks matching library performance
- Comparison of neural networks with OLS, Ridge, and Lasso regression
- Analysis of activation function impact on training dynamics
- Hyperparameter optimization for network architecture and learning rates
- Classification accuracy on MNIST dataset
- Regularization effects and comparison with traditional methods

### Dependencies

- NumPy
- Matplotlib
- Scikit-Learn

For detailed information, see [PROJECT 2/README.md](PROJECT%202/README.md).

## PROJECT 3: Advanced Topics

**Status:** In progress

This project will explore advanced machine learning topics and applications.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (for running analysis notebooks)

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FYSSTK3155
   ```

2. Install dependencies for a specific project:
   ```bash
   # For PROJECT 1
   cd "PROJECT 1/Code"
   pip install -r requirements.txt

   # For PROJECT 2
   cd "PROJECT 2/Code"
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Navigate to the desired project's `Code/` directory and open the relevant notebook.

### Global Dependencies

If working across multiple projects, install all dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas jupyter
```

## Usage Guidelines

### Running Notebooks

1. Navigate to the project's `Code/` directory
2. Start Jupyter Notebook: `jupyter notebook`
3. Open the desired analysis notebook
4. Ensure all Python modules in the `Code/` directory are accessible (notebooks include path setup)

### Reproducibility

All implementations use fixed random seeds for reproducibility:
- Network weight initialization: seed 6114 (Project 2)
- Data splitting: random_state 42 (standard)
- NumPy random operations: explicitly seeded

To reproduce exact results, ensure you use the same random seeds as specified in the notebooks.

### Code Organization

- **Modular Design**: Each project separates core implementations from analysis notebooks
- **Reusability**: Project 2 reuses optimization code from Project 1
- **Testing**: Project 2 includes comprehensive unit tests
- **Documentation**: Each project has detailed README files

## Learning Outcomes

Through these projects, students gain:

1. **Theoretical Understanding:**
   - Linear regression methods (OLS, Ridge, LASSO)
   - Gradient descent optimization algorithms
   - Neural network architecture and backpropagation
   - Bias-variance tradeoff and model complexity
   - Regularization techniques

2. **Practical Skills:**
   - Implementing machine learning algorithms from scratch
   - Hyperparameter tuning and model selection
   - Data preprocessing and feature engineering
   - Model evaluation and validation techniques
   - Visualization and result interpretation

3. **Technical Competencies:**
   - NumPy for numerical computations
   - Scikit-Learn for comparison and utilities
   - Jupyter Notebooks for interactive analysis
   - Version control and code organization
   - Scientific report writing

## Course Information

**Course Code:** FYS-STK3155 / FYS4155  
**Course Name:** Data Analysis and Machine Learning  
**Institution:** University of Oslo  
**Department:** Physics

**Course Resources:**
- Course website: [Machine Learning Course](http://www.uio.no/studier/emner/matnat/fys/FYS3155/index-eng.html)
- Lecture notes: [CompPhysics Machine Learning](https://compphysics.github.io/MachineLearning/)
- Textbook references: See individual project README files

## Contributing

This repository contains coursework submissions. For questions or issues related to the course material, please refer to the course website or contact the course instructors.

## License

This repository contains academic coursework for FYS-STK3155/FYS4155 at the University of Oslo. The code and reports are intended for educational purposes as part of the course requirements.

## Acknowledgments

- Course instructors and teaching assistants at the University of Oslo
- Authors of referenced textbooks and resources (Nielsen, Goodfellow et al., Raschka et al.)
- Scikit-Learn, NumPy, and Matplotlib development teams

## Contact

For questions about this repository or the implementations, please refer to the individual project README files or the course documentation.

---

**Note:** This repository is maintained as part of coursework for FYS-STK3155/FYS4155. All implementations are original work developed for learning and educational purposes.
