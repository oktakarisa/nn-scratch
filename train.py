#!/usr/bin/env python3
"""
Neural Network from Scratch - Command Line Training Script
Train a 3-layer neural network on MNIST dataset from the command line.

Usage:
    python train.py
    python train.py --epochs 50 --lr 0.01 --batch_size 20
    python train.py --activation sigmoid --nodes1 800 --nodes2 400
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.activations import sigmoid, tanh, softmax
from utils.mini_batch import GetMiniBatch


# ============================================================================
# Activation Functions and Derivatives
# ============================================================================

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_derivative(x):
    """Derivative of tanh function"""
    return 1 - np.tanh(x) ** 2


# ============================================================================
# Neural Network Core Functions
# ============================================================================

def forward_propagation(X, W1, B1, W2, B2, W3, B3, activation='tanh'):
    """
    Perform forward propagation through the 3-layer network
    
    Parameters
    ----------
    X : ndarray, shape (batch_size, n_features)
        Input data
    W1, B1, W2, B2, W3, B3 : ndarrays
        Weights and biases for each layer
    activation : str
        Activation function to use ('sigmoid' or 'tanh')
    
    Returns
    -------
    Z3 : ndarray, shape (batch_size, n_output)
        Output probabilities
    cache : dict
        Intermediate values for backpropagation
    """
    # Choose activation function
    if activation == 'sigmoid':
        act_func = sigmoid
    else:
        act_func = tanh
    
    # Layer 1
    A1 = np.dot(X, W1) + B1
    Z1 = act_func(A1)
    
    # Layer 2
    A2 = np.dot(Z1, W2) + B2
    Z2 = act_func(A2)
    
    # Layer 3 (Output layer)
    A3 = np.dot(Z2, W3) + B3
    Z3 = softmax(A3)
    
    # Store values for backpropagation
    cache = {
        'X': X, 'A1': A1, 'Z1': Z1, 'A2': A2, 'Z2': Z2, 'A3': A3, 'Z3': Z3
    }
    
    return Z3, cache


def cross_entropy_loss(y_true, y_pred):
    """
    Compute cross-entropy loss
    
    Parameters
    ----------
    y_true : ndarray, shape (batch_size, n_classes)
        True labels (one-hot encoded)
    y_pred : ndarray, shape (batch_size, n_classes)
        Predicted probabilities
    
    Returns
    -------
    float
        Average loss over the batch
    """
    batch_size = y_true.shape[0]
    epsilon = 1e-7
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / batch_size
    return loss


def backward_propagation(cache, y_true, W1, B1, W2, B2, W3, B3, activation='tanh', lr=0.01):
    """
    Perform backpropagation and update weights
    
    Parameters
    ----------
    cache : dict
        Intermediate values from forward propagation
    y_true : ndarray, shape (batch_size, n_output)
        True labels (one-hot)
    W1, B1, W2, B2, W3, B3 : ndarrays
        Current weights and biases
    activation : str
        Activation function used ('sigmoid' or 'tanh')
    lr : float
        Learning rate
    
    Returns
    -------
    Updated weights and biases
    """
    batch_size = y_true.shape[0]
    X = cache['X']
    A1 = cache['A1']
    Z1 = cache['Z1']
    A2 = cache['A2']
    Z2 = cache['Z2']
    A3 = cache['A3']
    Z3 = cache['Z3']
    
    # Layer 3 gradients
    dA3 = (Z3 - y_true) / batch_size
    dB3 = np.sum(dA3, axis=0)
    dW3 = np.dot(Z2.T, dA3)
    dZ2 = np.dot(dA3, W3.T)
    
    # Layer 2 gradients
    if activation == 'sigmoid':
        dA2 = dZ2 * sigmoid_derivative(A2)
    else:  # tanh
        dA2 = dZ2 * tanh_derivative(A2)
    
    dB2 = np.sum(dA2, axis=0)
    dW2 = np.dot(Z1.T, dA2)
    dZ1 = np.dot(dA2, W2.T)
    
    # Layer 1 gradients
    if activation == 'sigmoid':
        dA1 = dZ1 * sigmoid_derivative(A1)
    else:  # tanh
        dA1 = dZ1 * tanh_derivative(A1)
    
    dB1 = np.sum(dA1, axis=0)
    dW1 = np.dot(X.T, dA1)
    
    # Update weights and biases
    W1 = W1 - lr * dW1
    B1 = B1 - lr * dB1
    W2 = W2 - lr * dW2
    B2 = B2 - lr * dB2
    W3 = W3 - lr * dW3
    B3 = B3 - lr * dB3
    
    return W1, B1, W2, B2, W3, B3


def predict(X, W1, B1, W2, B2, W3, B3, activation='tanh'):
    """
    Make predictions using the trained network
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Input data
    W1, B1, W2, B2, W3, B3 : ndarrays
        Trained weights and biases
    activation : str
        Activation function used
    
    Returns
    -------
    predictions : ndarray, shape (n_samples,)
        Predicted class labels
    """
    Z3, _ = forward_propagation(X, W1, B1, W2, B2, W3, B3, activation)
    predictions = np.argmax(Z3, axis=1)
    return predictions


# ============================================================================
# Neural Network Class
# ============================================================================

class NeuralNetwork:
    """
    3-layer Neural Network Classifier
    """
    
    def __init__(self, n_nodes1=400, n_nodes2=200, n_output=10, sigma=0.01,
                 lr=0.01, batch_size=20, epochs=50, activation='tanh', verbose=True):
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.n_output = n_output
        self.sigma = sigma
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.verbose = verbose
        
        # History for plotting
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
    def _initialize_weights(self, n_features):
        """Initialize weights and biases"""
        self.W1 = self.sigma * np.random.randn(n_features, self.n_nodes1)
        self.B1 = self.sigma * np.random.randn(self.n_nodes1)
        self.W2 = self.sigma * np.random.randn(self.n_nodes1, self.n_nodes2)
        self.B2 = self.sigma * np.random.randn(self.n_nodes2)
        self.W3 = self.sigma * np.random.randn(self.n_nodes2, self.n_output)
        self.B3 = self.sigma * np.random.randn(self.n_output)
        
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the neural network"""
        # Initialize weights
        self._initialize_weights(X.shape[1])
        
        if self.verbose:
            print("="*70)
            print("TRAINING NEURAL NETWORK")
            print("="*70)
            print(f"Architecture: {X.shape[1]} → {self.n_nodes1} → {self.n_nodes2} → {self.n_output}")
            print(f"Training samples: {X.shape[0]}")
            print(f"Validation samples: {X_val.shape[0] if X_val is not None else 0}")
            print(f"Batch size: {self.batch_size}")
            print(f"Learning rate: {self.lr}")
            print(f"Epochs: {self.epochs}")
            print(f"Activation: {self.activation}")
            print("="*70)
        
        # Training loop
        for epoch in range(self.epochs):
            # Mini-batch training
            mini_batch = GetMiniBatch(X, y, batch_size=self.batch_size, seed=epoch)
            
            epoch_loss = 0
            for mini_X, mini_y in mini_batch:
                # Forward propagation
                Z3, cache = forward_propagation(mini_X, self.W1, self.B1, self.W2, self.B2,
                                                 self.W3, self.B3, self.activation)
                
                # Compute loss
                loss = cross_entropy_loss(mini_y, Z3)
                epoch_loss += loss
                
                # Backward propagation
                self.W1, self.B1, self.W2, self.B2, self.W3, self.B3 = backward_propagation(
                    cache, mini_y, self.W1, self.B1, self.W2, self.B2, self.W3, self.B3,
                    self.activation, self.lr
                )
            
            # Average loss for epoch
            avg_loss = epoch_loss / len(mini_batch)
            self.train_loss_history.append(avg_loss)
            
            # Calculate training accuracy
            train_pred = self.predict(X)
            train_acc = accuracy_score(np.argmax(y, axis=1), train_pred)
            self.train_acc_history.append(train_acc)
            
            # Calculate validation metrics if provided
            if X_val is not None and y_val is not None:
                val_pred_prob, _ = forward_propagation(X_val, self.W1, self.B1, self.W2, self.B2,
                                                        self.W3, self.B3, self.activation)
                val_loss = cross_entropy_loss(y_val, val_pred_prob)
                self.val_loss_history.append(val_loss)
                
                val_pred = self.predict(X_val)
                val_acc = accuracy_score(np.argmax(y_val, axis=1), val_pred)
                self.val_acc_history.append(val_acc)
                
                if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            else:
                if self.verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                    print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                          f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        if self.verbose:
            print("="*70)
            print("TRAINING COMPLETE!")
            print("="*70)
    
    def predict(self, X):
        """Make predictions"""
        return predict(X, self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, self.activation)
    
    def plot_learning_curves(self, save_path='plots/learning_curves_cli.png'):
        """Plot and save learning curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs_range = range(1, len(self.train_loss_history) + 1)
        
        # Loss curve
        ax1.plot(epochs_range, self.train_loss_history, 'b-o', label='Training Loss', linewidth=2, markersize=4)
        ax1.plot(epochs_range, self.val_loss_history, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
        ax1.set_title('Learning Curve - Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs_range, self.train_acc_history, 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
        ax2.plot(epochs_range, self.val_acc_history, 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Curve - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
        plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main training function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a 3-layer Neural Network on MNIST')
    parser.add_argument('--nodes1', type=int, default=400, help='Number of nodes in first hidden layer')
    parser.add_argument('--nodes2', type=int, default=200, help='Number of nodes in second hidden layer')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--sigma', type=float, default=0.01, help='Weight initialization std')
    parser.add_argument('--activation', type=str, default='tanh', choices=['sigmoid', 'tanh'], 
                        help='Activation function')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("\n" + "="*70)
    print("NEURAL NETWORK FROM SCRATCH - MNIST CLASSIFICATION")
    print("="*70)
    
    # Load MNIST dataset
    print("\n[1/5] Loading MNIST dataset...")
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f"✓ Loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Preprocess data
    print("\n[2/5] Preprocessing data...")
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0
    
    # One-hot encode labels
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    y_train_one_hot = enc.fit_transform(y_train[:, np.newaxis])
    y_test_one_hot = enc.transform(y_test[:, np.newaxis])
    
    # Train/validation split
    X_train, X_val, y_train_one_hot, y_val_one_hot, y_train_orig, y_val_orig = train_test_split(
        X_train, y_train_one_hot, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )
    print(f"✓ Preprocessed: {X_train.shape[0]} train, {X_val.shape[0]} validation samples")
    
    # Initialize and train model
    print(f"\n[3/5] Training neural network...")
    model = NeuralNetwork(
        n_nodes1=args.nodes1,
        n_nodes2=args.nodes2,
        n_output=10,
        sigma=args.sigma,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        activation=args.activation,
        verbose=True
    )
    
    model.fit(X_train, y_train_one_hot, X_val, y_val_one_hot)
    
    # Evaluate on test set
    print(f"\n[4/5] Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Training Accuracy:   {model.train_acc_history[-1]:.4f} ({model.train_acc_history[-1]*100:.2f}%)")
    print(f"Validation Accuracy: {model.val_acc_history[-1]:.4f} ({model.val_acc_history[-1]*100:.2f}%)")
    print(f"Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*70)
    
    # Plot and save results
    if not args.no_plots:
        print(f"\n[5/5] Generating plots...")
        model.plot_learning_curves()
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                    xticklabels=range(10), yticklabels=range(10),
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix_cli.png', dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: plots/confusion_matrix_cli.png")
        plt.close()
        
        print("\n✓ All plots saved successfully!")
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

