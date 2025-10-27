# Neural Network from Scratch - MNIST Digit Classification

## Overview
This project implements a **3-layer neural network from scratch** using only NumPy to classify handwritten digits from the MNIST dataset. The implementation demonstrates the fundamental concepts of deep learning including forward propagation, backpropagation, and stochastic gradient descent.

## Project Structure
```
nn-scratch/
├── data/                     # MNIST data (downloaded automatically)
├── plots/                    # Auto-saved plots (learning curves, misclassified images)
├── nn_mnist.ipynb            # Main notebook with all implementations
├── utils/                    # Helper utilities
│   ├── activations.py        # Activation functions (sigmoid, tanh, softmax)
│   ├── mini_batch.py         # Mini-batch iterator for SGD
│   └── preprocessing.py      # Data preprocessing utilities
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Features
- **From-Scratch Implementation**: Neural network built using only NumPy
- **3-Layer Architecture**: Configurable hidden layer sizes (default: 400, 200 nodes)
- **Multiple Activation Functions**: Sigmoid and Hyperbolic Tangent (tanh)
- **Mini-batch SGD**: Efficient stochastic gradient descent with configurable batch size
- **Comprehensive Visualization**: Learning curves, accuracy plots, misclassification analysis
- **High Accuracy**: Achieves ~95-97% accuracy on MNIST test set

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or navigate to the project directory:
```bash
cd nn-scratch
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook nn_mnist.ipynb
```

## Usage

### Quick Start
Open `nn_mnist.ipynb` and run all cells sequentially. The notebook will:
1. Download and load MNIST dataset automatically
2. Preprocess and split data into train/validation/test sets
3. Initialize and train the neural network
4. Display learning curves and accuracy metrics
5. Visualize misclassified examples

### Key Parameters
You can modify these hyperparameters in the notebook:
- **Hidden Layer Sizes**: `n_nodes1=400`, `n_nodes2=200`
- **Learning Rate**: `lr=0.01`
- **Batch Size**: `batch_size=20`
- **Epochs**: `epochs=50`
- **Activation Function**: `'sigmoid'` or `'tanh'`
- **Weight Initialization**: `sigma=0.01` (standard deviation)

## Implementation Details

### Problem 1: Weight Initialization
- Gaussian (normal) distribution initialization
- Configurable standard deviation (σ = 0.01)
- Separate initialization for weights and biases

### Problem 2: Forward Propagation
- Three fully connected layers
- Choice of sigmoid or tanh activation for hidden layers
- Softmax activation for output layer
- Efficient matrix operations using NumPy

### Problem 3: Loss Function
- Cross-entropy loss for multi-class classification
- Numerical stability with log clipping (1e-7)

### Problem 4: Backpropagation
- Gradient computation for all layers
- Efficient use of matrix operations
- Stochastic gradient descent with mini-batches

### Problem 5: Prediction
- Argmax operation on softmax output
- Returns predicted class labels

### Problem 6: Training & Accuracy
- Mini-batch SGD implementation
- Training and validation accuracy tracking
- Epoch-wise loss monitoring

### Problem 7: Learning Curves
- Training vs validation loss plots
- Training vs validation accuracy plots
- Auto-saved to `plots/` directory

### Problem 8: Misclassification Analysis
- Visual display of incorrectly classified images
- Shows predicted vs actual labels
- Up to 36 examples displayed

## Results

Expected performance on MNIST:
- **Training Accuracy**: ~98-99%
- **Validation Accuracy**: ~95-97%
- **Test Accuracy**: ~95-97%

Training typically converges within 30-50 epochs with the default hyperparameters.

## Dataset

**MNIST** (Modified National Institute of Standards and Technology)
- **Training samples**: 48,000 (after 80/20 split)
- **Validation samples**: 12,000
- **Test samples**: 10,000
- **Image size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

The dataset is automatically downloaded via TensorFlow/Keras.

## Key Concepts Demonstrated

1. **Neural Network Fundamentals**
   - Multi-layer perceptron architecture
   - Weight initialization strategies
   - Forward and backward propagation

2. **Optimization**
   - Stochastic gradient descent
   - Mini-batch training
   - Learning rate scheduling

3. **Regularization & Validation**
   - Train/validation/test split
   - Learning curve analysis
   - Overfitting detection

4. **Mathematical Foundations**
   - Matrix operations
   - Gradient computation
   - Activation functions
   - Loss functions

## Extensions & Improvements

Possible enhancements (for learning):
- Implement ReLU activation function
- Add dropout for regularization
- Implement momentum or Adam optimizer
- Add L2 regularization
- Experiment with different architectures
- Add batch normalization
- Implement learning rate decay

## License
This project is for educational purposes.

## Author
Neural Network Assignment - Deep Learning from Scratch

## Acknowledgments
- MNIST dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- Assignment based on neural network fundamentals curriculum

