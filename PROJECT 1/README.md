# Project 1 FYS-STK3155

## Group Members:

Egil Furnes, Bror Johannes Tidemand Ruud, Ådne Rolstad

## Project Description
This project explores the challenge of linear regression on Runges function, using OLS, Ridge and LASSO regression. In addition, we implement methods for gradient descent, and use resampling methods such as bootstrap and cross-validation to explore the bias-variance tradeoff.

## Required Packages
To install the required packages, run the following command in terminal: !pip install -r requirements.txt


## Descripton of Notebooks and Python Files
### Python Files: 
- **gradient_descent.py**: Implementation of the following gradient descent methods for Ridge, OLS and LASSO regression: Standard GD, GD with momentum, ADAGrad, RMSProp, ADAM.
- **OLS.py**: Contains function that returns closed-form OLS parameters.
- **plot_style.py**: Sets code style for entire project, ensuring that every plot has the same styling.
- **polynomial_features.py**: Contains a function that returns a polynomial feature matrix, used in linear regression. Can return feature matrices either with or without intercept column.
- **prepare_data.py**: Contains a function for preparing the data used in the project. Performs train-test split, defines the Runge function, etc.
- **Ridge.py**: Contains function that returns the closed-form Ridge parameters.
- **Runge.py**: Used for plotting the Runge function during the project.
- **Set_latex_params.py**: Configuration for matplotlib plots in latex-style format.
- **stochastic_gradient_descent.py**: Implementation of stochastic gradient descent functions for Ridge, OLS and LASSO regression. Standard GD, GD with momentum, ADAGrad, RMSProp, ADAM.

### Notebooks:
- **Exa.ipynb**: Solves exercise a of the given problem set. OLS regression analysis, plotting MSE scores, R2-scores, parameters and fits for different polynomial degrees.
- **Exb.ipynb**: Solves exercise b of the given problem set. Ridge regression analysis, plotting MSE and R2 scores, parameters and fits for various lambda values. Calulate parameters for different polynomial degrees and lambdas to find optimal parameters
- **Exc.ipynb**: Solves exercise c of the given problem set. Testing and validation of standard gradient descent implementations for OLS and Ridge. Computing parameters, MSEs, and R2-scores for different parameters and degrees, and comparing with closed-form solutions for Ridge and OLS. Plotting results
- **Exd.ipynb**: Solves exercise d of the given problem set. Testing and validation of gradient descent implementations for OLS and Ridge. Implementations include GD with momentum, ADAGrad, RMSProp and ADAM. Computing parameters, MSEs, and R2-scores for different parameters and degrees, and comparing with closed-form solutions for Ridge and OLS. Plotting results, comparing different GD methods.
- **Exe.ipynb**: Solves exercise e of the given problem set. Testing and validation of gradient descent method for LASSO regression. Comparing results with Scikit-Learn. Plotting results.
- **Exf.ipynb**: Solves exercise f of the given problem set. Testing stochastic gradient descent methods for Ridge, OLS and LASSO regression. Comparing results with closed-form solutions (OLS and Ridge) and Scikit-learn (LASSO). Plotting results.
- **Exg.ipynb**: Solves exercise g of the given problem set. Study bias-variance tradeoff using bootstrap resampling.
- **Exh.ipynb**: Solves exercise h of the given problem set. Study bias-variance trade-off using k-fold cross validation.