"""
Data preprocessing utilities for MNIST
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def flatten_images(X):
    """
    Flatten image data from (n_samples, height, width) to (n_samples, height*width)
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, height, width)
        Image data
    
    Returns
    -------
    ndarray, shape (n_samples, height*width)
        Flattened image data
    """
    return X.reshape(X.shape[0], -1)


def normalize_images(X):
    """
    Normalize image pixel values from [0, 255] to [0, 1]
    
    Parameters
    ----------
    X : ndarray
        Image data with pixel values in range [0, 255]
    
    Returns
    -------
    ndarray
        Normalized image data in range [0, 1]
    """
    X = X.astype(np.float64)
    X /= 255.0
    return X


def one_hot_encode(y):
    """
    Convert labels to one-hot encoding
    
    Parameters
    ----------
    y : ndarray, shape (n_samples,)
        Label array with integer class labels
    
    Returns
    -------
    ndarray, shape (n_samples, n_classes)
        One-hot encoded labels
    """
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    y_one_hot = enc.fit_transform(y[:, np.newaxis])
    return y_one_hot


def preprocess_mnist(X_train, X_test, y_train, y_test):
    """
    Complete preprocessing pipeline for MNIST data
    
    Parameters
    ----------
    X_train : ndarray
        Training images
    X_test : ndarray
        Test images
    y_train : ndarray
        Training labels
    y_test : ndarray
        Test labels
    
    Returns
    -------
    tuple
        Preprocessed (X_train, X_test, y_train_one_hot, y_test_one_hot)
    """
    # Flatten images
    X_train = flatten_images(X_train)
    X_test = flatten_images(X_test)
    
    # Normalize pixel values
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)
    
    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train)
    y_test_one_hot = one_hot_encode(y_test)
    
    return X_train, X_test, y_train_one_hot, y_test_one_hot

