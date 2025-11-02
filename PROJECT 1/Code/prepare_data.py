import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def prepare_data(n=100, noise_scale=0.01):
    """Function for preparing data

    Args:
        n (int, optional): Number of datapoints. Defaults to 100.
        noise_scale (int, optional): Amplitude of random noise. Defaults to 0.01.
        
    Returns:
        x (np.array): 1d array containing x values.
        y (np.array): 1d array containing y values (Runges function).
        x_train (np.array): 1d array containing x training values (from train/test split).
        x_test (np.array): 1d array containing x testing values (from train/test split).
        y_train (np.array): 1d array containing y training values (from train/test split).
        y_test (np.array): 1d array containing y testing values (from train/test split).
        y_noisy (np.array): 1d array containing y values (Runges function) with added noise.
    """
    def f(x): 
        return 1/(1+25*x**2)
    rng = np.random.default_rng(seed=6114)
    x = np.linspace(-1,1,n)
    y = f(x)

    noise = rng.normal(loc=0.0, scale=noise_scale, size=x.shape)
    y_noisy = y + noise
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6114)
    
    return x, y, x_train, x_test, y_train, y_test, y_noisy

