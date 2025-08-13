# Enhanced Log-Normalized DANTE Alloy Design Model

## Overview

This enhanced version of the DANTE Alloy Design framework introduces logarithmic transformation and weighted loss functions to improve neural network performance on mechanical properties that span multiple orders of magnitude.

## Key Enhancements

### 1. Logarithmic Transformation
- **Young's modulus** and **yield strength** are log-transformed before normalization
- Helps neural networks handle properties spanning multiple orders of magnitude
- Improves numerical stability during training

### 2. Weighted MSE Loss Function
- Uses `exp(2y)` weights where `y` is the log-normalized label
- Emphasizes accurate prediction of higher-value materials
- Formula: `loss = mean(exp(2*y_true) * (y_true - y_pred)^2)`

### 3. Proper Inverse Transformation
- Neural network outputs are inverse-normalized then exponentiated
- Ensures final predictions are in original physical units (Pa)
- Maintains prediction accuracy across the full range

## New Classes and Functions

### LogNormalizedDataLoader
Enhanced data loader with logarithmic transformation capabilities.

```python
from data_loader import LogNormalizedDataLoader

# Initialize enhanced data loader
data_loader = LogNormalizedDataLoader()

# Process data with log transformation
processed_data = data_loader.process_data_with_log_transform(df)
X_elements, X_elements_with_Fe, X_compounds, Y_original, Y_log_normalized, Y_combined = processed_data
```

### LogNormalizedDualNetworkSurrogateModel
Enhanced neural network model with weighted loss and log transformation.

```python
from neural_models import LogNormalizedDualNetworkSurrogateModel

# Create enhanced model
log_model = LogNormalizedDualNetworkSurrogateModel(
    search_dims=3,          # 3D search space (Co, Mo, Ti)
    network_input_dims=4,   # 4D network input (Co, Mo, Ti, Fe)
    n_folds=5               # Cross-validation folds
)

# Train the model
trained_model = log_model(X_elements_with_Fe, Y_original, verbose=1)

# Make predictions (automatically handles inverse transformation)
predictions = trained_model.predict(test_samples)
```

## Usage Examples

### Basic Usage

```python
import numpy as np
import pandas as pd
from data_loader import LogNormalizedDataLoader
from neural_models import LogNormalizedDualNetworkSurrogateModel

# 1. Load and process data
data_loader = LogNormalizedDataLoader()
df = data_loader.load_data("../data.csv")
processed_data = data_loader.process_data_with_log_transform(df)
X_elements, X_elements_with_Fe, X_compounds, Y_original, Y_log_normalized, Y_combined = processed_data

# 2. Create and train model
model = LogNormalizedDualNetworkSurrogateModel(search_dims=3, network_input_dims=4)
trained_model = model(X_elements_with_Fe, Y_original)

# 3. Make predictions
predictions = trained_model.predict(X_elements_with_Fe)
```

### Advanced Usage with Optimization

```python
from alloy_objective import AlloyObjectiveFunction
from optimization import run_dante_optimization

# Create objective function
alloy_obj_func = AlloyObjectiveFunction(X_elements, X_elements_with_Fe, Y_combined)

# Run optimization with enhanced model
optimization_results = run_dante_optimization(
    alloy_obj_func, 
    trained_model,
    X_elements,
    max_iterations=100
)
```

## Files Structure

```
src/
├── neural_models.py                    # Contains LogNormalizedDualNetworkSurrogateModel
├── data_loader.py                      # Contains LogNormalizedDataLoader
├── test_log_normalized_model.py        # Test script for enhanced model
├── main_v5_log_normalized.ipynb        # Enhanced notebook
└── README_LOG_NORMALIZED.md            # This file
```

## Testing

Run the test script to verify the enhanced model:

```bash
cd src/
python test_log_normalized_model.py
```

Or use the enhanced Jupyter notebook:

```bash
jupyter notebook main_v5_log_normalized.ipynb
```

## Technical Details

### Data Transformation Pipeline

1. **Input**: Original mechanical properties (Pa scale)
2. **Log Transform**: `Y_log = log(Y_original)`
3. **Normalize**: `Y_normalized = StandardScaler().fit_transform(Y_log)`
4. **Train**: Neural network trains on normalized log values
5. **Predict**: Model outputs normalized log predictions
6. **Inverse Normalize**: `Y_log_pred = scaler.inverse_transform(Y_normalized_pred)`
7. **Exponentiate**: `Y_final = exp(Y_log_pred)`

### Weighted Loss Function

The weighted MSE loss function is defined as:

```python
def weighted_mse_loss(y_true, y_pred):
    weights = tf.exp(2.0 * y_true)  # Higher weights for larger values
    squared_errors = tf.square(y_true - y_pred)
    weighted_errors = weights * squared_errors
    return tf.reduce_mean(weighted_errors)
```

### Model Architecture

- **Input Layer**: 4D (Co, Mo, Ti, Fe compositions)
- **Shared Layers**: Dense(256) → LayerNorm → Dropout → Dense(128) → LayerNorm → Dropout
- **Elastic Branch**: Dense(64) → Dense(32) → Dense(1)
- **Yield Branch**: Dense(64) → Dense(32) → Dense(1)
- **Output**: Concatenated 2D (elastic modulus, yield strength)
- **Loss**: Weighted MSE with exp(2y) weights
- **Optimizer**: AdamW with weight decay

## Performance Benefits

1. **Better Numerical Stability**: Log transformation reduces the dynamic range
2. **Improved High-Value Predictions**: Weighted loss emphasizes important materials
3. **Enhanced Training Convergence**: More stable gradients during backpropagation
4. **Robust Performance**: Better handling of extreme values in the dataset

## Compatibility

The enhanced model is fully compatible with:
- Existing DANTE optimization framework
- Original objective functions
- Visualization tools
- All existing analysis scripts

Simply replace `DualNetworkSurrogateModel` with `LogNormalizedDualNetworkSurrogateModel` in your code.

## Notes

- The enhanced model automatically saves weights with prefix "log_dual_network"
- Cross-validation metrics are reported on both log scale and original scale
- Fallback RandomForest model is available when TensorFlow is not installed
- All predictions are automatically converted back to original physical units
