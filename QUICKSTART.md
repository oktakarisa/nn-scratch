# Quick Start Guide

## Getting Started with Neural Network from Scratch

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Launch Jupyter Notebook
```bash
jupyter notebook nn_mnist.ipynb
```

### Step 3: Run All Cells
In Jupyter, select `Cell → Run All` from the menu, or run cells sequentially.

## What to Expect

### Training Time
- On a modern CPU: ~10-15 minutes for 50 epochs
- On GPU: ~2-5 minutes

### Expected Accuracy
- Training: ~98-99%
- Validation: ~95-97%
- Test: ~95-97%

### Generated Files
After running the notebook, the following visualizations will be saved in `plots/`:
- `sample_images.png` - Sample MNIST images
- `learning_curves.png` - Training/validation loss and accuracy
- `misclassified_images.png` - Examples of misclassifications
- `confusion_matrix.png` - Confusion matrix on test set

## Understanding the Output

### Problems Covered
1. **Weight Initialization** - Gaussian distribution initialization
2. **Forward Propagation** - Complete forward pass implementation
3. **Loss Function** - Cross-entropy loss
4. **Backpropagation** - Gradient computation and updates
5. **Prediction** - Argmax-based prediction
6. **Training** - Mini-batch SGD training loop
7. **Learning Curves** - Visualization of training progress
8. **Misclassification** - Analysis of model errors

### Training Progress
You'll see output like:
```
Epoch   1/50 | Train Loss: 2.2156 | Train Acc: 0.2345 | Val Loss: 2.1876 | Val Acc: 0.2543
Epoch   6/50 | Train Loss: 0.5432 | Train Acc: 0.8567 | Val Loss: 0.5687 | Val Acc: 0.8432
...
Epoch  50/50 | Train Loss: 0.0987 | Train Acc: 0.9876 | Val Loss: 0.1234 | Val Acc: 0.9654
```

## Customization

### Hyperparameters
You can modify these in the notebook:

```python
model = ScratchSimpleNeuralNetworkClassifier(
    n_nodes1=400,        # First hidden layer size
    n_nodes2=200,        # Second hidden layer size
    n_output=10,         # Output classes (0-9)
    sigma=0.01,          # Weight initialization std dev
    lr=0.01,             # Learning rate
    batch_size=20,       # Mini-batch size
    epochs=50,           # Number of training epochs
    activation='tanh',   # 'tanh' or 'sigmoid'
    verbose=True         # Print progress
)
```

### Activation Functions
Try switching between:
- `'tanh'` - Hyperbolic tangent (default, usually better)
- `'sigmoid'` - Sigmoid function (slower convergence)

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size or use fewer samples
```python
# Use smaller subset of data
X_train_small = X_train[:10000]
y_train_small = y_train_one_hot[:10000]
```

### Issue: Training Too Slow
**Solution:** Reduce epochs or increase batch size
```python
model = ScratchSimpleNeuralNetworkClassifier(
    epochs=20,          # Fewer epochs
    batch_size=100,     # Larger batches
    ...
)
```

### Issue: Poor Accuracy
**Solutions:**
1. Increase epochs (try 100)
2. Adjust learning rate (try 0.001 or 0.1)
3. Change activation function
4. Modify network architecture (more nodes)

## Next Steps

### Experiments to Try
1. **Compare Activations**: Run with both 'tanh' and 'sigmoid'
2. **Learning Rate Search**: Try [0.001, 0.01, 0.05, 0.1]
3. **Network Size**: Experiment with different hidden layer sizes
4. **Longer Training**: Try 100 or 200 epochs

### Advanced Challenges
1. Implement momentum or Adam optimizer
2. Add L2 regularization
3. Implement dropout
4. Try on different datasets (Fashion-MNIST)
5. Implement ReLU activation

## Understanding Results

### Good Signs
- ✅ Training loss decreases steadily
- ✅ Validation accuracy > 95%
- ✅ Small gap between train and validation accuracy

### Warning Signs
- ⚠️ Training loss increases → learning rate too high
- ⚠️ Loss stays flat → learning rate too low or poor initialization
- ⚠️ Big gap between train/val accuracy → overfitting

## Resources

### Mathematical Foundations
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

### Code References
- `utils/activations.py` - Activation function implementations
- `utils/mini_batch.py` - Mini-batch iterator
- `utils/preprocessing.py` - Data preprocessing utilities

## Support

For questions or issues:
1. Check the main README.md
2. Review the notebook markdown cells for explanations
3. Examine the code comments for implementation details

Happy learning! 🎓

