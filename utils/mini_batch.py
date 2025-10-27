"""
Mini-batch iterator for stochastic gradient descent
"""
import numpy as np


class GetMiniBatch:
    """
    Iterator to get a mini-batch
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data
    y : ndarray, shape (n_samples, n_output)
        Correct answer value
    batch_size : int
        Batch size
    seed : int
        NumPy random seed
    """
    
    def __init__(self, X, y, batch_size=20, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self._X = X[shuffle_index]
        self._y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0] / self.batch_size).astype(np.int64)
    
    def __len__(self):
        return self._stop
    
    def __getitem__(self, item):
        p0 = item * self.batch_size
        p1 = item * self.batch_size + self.batch_size
        return self._X[p0:p1], self._y[p0:p1]
    
    def __iter__(self):
        self._counter = 0
        return self
    
    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter * self.batch_size
        p1 = self._counter * self.batch_size + self.batch_size
        self._counter += 1
        return self._X[p0:p1], self._y[p0:p1]

