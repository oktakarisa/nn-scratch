"""
Activation functions for neural network
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    
    Parameters
    ----------
    x : ndarray
        Input array
    
    Returns
    -------
    ndarray
        Output after applying sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Hyperbolic tangent activation function
    
    Parameters
    ----------
    x : ndarray
        Input array
    
    Returns
    -------
    ndarray
        Output after applying tanh function
    """
    return np.tanh(x)


def softmax(x):
    """
    Softmax activation function
    
    Parameters
    ----------
    x : ndarray, shape (batch_size, n_classes)
        Input array
    
    Returns
    -------
    ndarray, shape (batch_size, n_classes)
        Output probabilities for each class
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    
    Parameters
    ----------
    x : ndarray
        Input array (output of sigmoid function)
    
    Returns
    -------
    ndarray
        Derivative
    """
    sig = sigmoid(x)
    return sig * (1 - sig)


def tanh_derivative(x):
    """
    Derivative of tanh function
    
    Parameters
    ----------
    x : ndarray
        Input array
    
    Returns
    -------
    ndarray
        Derivative
    """
    return 1 - np.tanh(x) ** 2

