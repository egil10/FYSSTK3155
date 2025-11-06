import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def prepare_data(n=100, noise_scale=0.01, if_noise=False):
    """Function for preparing data (Runge function)

    Args:
        n (int, optional): 
            Number of datapoints. Defaults to 100.
        noise_scale (int, optional): 
            Standard deviation of added Gaussian noise. Default is 0.01.
        if_noise (bool, optional):
            If True, adds noise to the output values before splitting. 
            Defaults to False.
        
    Returns:
        x (np.ndarray):
            Input values of shape (n, 1).
        y (np.ndarray):
            True Runge function values without noise.
        x_train, x_test  (np.ndarray):
            Training and test input values.
        y_train, y_test (np.ndarray)
            Corresponding target values (noisy if if_noise=True).
    """
    def f(x): 
        return 1/(1+25*x**2)
    rng = np.random.default_rng(seed=6114)
    x = np.linspace(-1,1,n)
    x = x.reshape(-1,1)
    y = f(x)

  
    noise = rng.normal(loc=0.0, scale=noise_scale, size=x.shape)
    y_noisy = y + noise

    if if_noise: 
        x_train, x_test, y_train, y_test = train_test_split(x, y_noisy, test_size=0.2, random_state=6114)
        return x, y, x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6114)

    return x, y, x_train, x_test, y_train, y_test