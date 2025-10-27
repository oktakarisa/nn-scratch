# Command-Line Training Script Usage

## Quick Start

### Basic Training (Default Parameters)
```bash
python train.py
```

This will train with:
- 400 nodes in first hidden layer
- 200 nodes in second hidden layer
- Learning rate: 0.01
- Batch size: 20
- Epochs: 50
- Activation: tanh

---

## Custom Training Examples

### Example 1: Train with Sigmoid Activation
```bash
python train.py --activation sigmoid
```

### Example 2: More Epochs
```bash
python train.py --epochs 100
```

### Example 3: Larger Network
```bash
python train.py --nodes1 800 --nodes2 400
```

### Example 4: Faster Learning
```bash
python train.py --lr 0.05 --batch_size 50
```

### Example 5: Full Custom Configuration
```bash
python train.py --nodes1 512 --nodes2 256 --lr 0.02 --batch_size 32 --epochs 75 --activation tanh --seed 123
```

### Example 6: Quick Test (No Plots)
```bash
python train.py --epochs 10 --no-plots
```

---

## All Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--nodes1` | int | 400 | Number of nodes in first hidden layer |
| `--nodes2` | int | 200 | Number of nodes in second hidden layer |
| `--lr` | float | 0.01 | Learning rate |
| `--batch_size` | int | 20 | Mini-batch size |
| `--epochs` | int | 50 | Number of training epochs |
| `--sigma` | float | 0.01 | Weight initialization standard deviation |
| `--activation` | str | tanh | Activation function (sigmoid or tanh) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--no-plots` | flag | False | Skip generating plots |

---

## Output Files

The script generates:
- `plots/learning_curves_cli.png` - Training/validation loss and accuracy
- `plots/confusion_matrix_cli.png` - Confusion matrix on test set

---

## Expected Output

```
======================================================================
NEURAL NETWORK FROM SCRATCH - MNIST CLASSIFICATION
======================================================================

[1/5] Loading MNIST dataset...
✓ Loaded: 60000 training, 10000 test samples

[2/5] Preprocessing data...
✓ Preprocessed: 48000 train, 12000 validation samples

[3/5] Training neural network...
======================================================================
TRAINING NEURAL NETWORK
======================================================================
Architecture: 784 → 400 → 200 → 10
Training samples: 48000
Validation samples: 12000
Batch size: 20
Learning rate: 0.01
Epochs: 50
Activation: tanh
======================================================================
Epoch   1/50 | Train Loss: 0.5234 | Train Acc: 0.8542 | Val Loss: 0.3456 | Val Acc: 0.9012
Epoch   6/50 | Train Loss: 0.2145 | Train Acc: 0.9345 | Val Loss: 0.1987 | Val Acc: 0.9423
...
Epoch  50/50 | Train Loss: 0.0534 | Train Acc: 0.9876 | Val Loss: 0.0987 | Val Acc: 0.9734
======================================================================
TRAINING COMPLETE!
======================================================================

[4/5] Evaluating on test set...

======================================================================
FINAL RESULTS
======================================================================
Training Accuracy:   0.9876 (98.76%)
Validation Accuracy: 0.9734 (97.34%)
Test Accuracy:       0.9712 (97.12%)
======================================================================

[5/5] Generating plots...
Learning curves saved to: plots/learning_curves_cli.png
Confusion matrix saved to: plots/confusion_matrix_cli.png

✓ All plots saved successfully!

======================================================================
EXECUTION COMPLETE!
======================================================================
```

---

## Advantages Over Jupyter Notebook

✅ **Faster** - No browser overhead  
✅ **Scriptable** - Easy to automate experiments  
✅ **Remote-friendly** - Run on servers via SSH  
✅ **Reproducible** - Command-line arguments track configuration  
✅ **Batch processing** - Run multiple experiments easily  

---

## Tips

### Run Multiple Experiments
```bash
# Experiment 1: Sigmoid
python train.py --activation sigmoid --seed 1 > results_sigmoid.log

# Experiment 2: Tanh
python train.py --activation tanh --seed 1 > results_tanh.log

# Experiment 3: Larger network
python train.py --nodes1 800 --nodes2 400 --seed 1 > results_large.log
```

### Background Execution
```bash
# Run in background (Linux/Mac)
nohup python train.py --epochs 100 > training.log 2>&1 &

# Run in background (Windows PowerShell)
Start-Process python -ArgumentList "train.py --epochs 100" -NoNewWindow
```

---

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "utils module not found"
**Solution:** Make sure you're in the project root directory
```bash
cd nn-scratch
python train.py
```

### Issue: "Permission denied"
**Solution:** Make the script executable (Linux/Mac)
```bash
chmod +x train.py
./train.py
```

