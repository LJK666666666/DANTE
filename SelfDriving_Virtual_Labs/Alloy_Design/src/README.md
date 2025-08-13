# DANTE Alloy Design Virtual Lab

This directory contains the modularized Python implementation of the DANTE (Deep Active Neural-network-based Tree Exploration) framework for alloy material composition optimization.

## Overview

The DANTE framework optimizes alloy compositions to achieve desired mechanical properties (elastic modulus and yield strength) through:

1. **Neural Surrogate Models**: Deep learning models that predict material properties from composition
2. **Active Learning**: Intelligent sampling strategy to efficiently explore the design space
3. **Tree Exploration**: Hierarchical search algorithm for systematic optimization
4. **Multi-objective Optimization**: Balancing multiple material properties

## File Structure

```
src/
├── main.py                 # Main execution script (command-line)
├── main.ipynb              # Interactive Jupyter notebook (recommended)
├── data_loader.py          # Data loading and preprocessing
├── alloy_objective.py      # Objective function definition
├── neural_models.py        # Neural network surrogate models
├── optimization.py         # DANTE optimization algorithms
├── visualization.py        # Comprehensive visualization tools
├── config.py              # Configuration parameters
├── run_example.py          # Example usage script
├── test_imports.py         # Module testing script
└── README.md              # This file
```

## Module Descriptions

### main.py
The main execution script for command-line usage:
- Loads and preprocesses data
- Creates objective functions
- Trains neural surrogate models
- Runs DANTE optimization
- Generates visualizations

### main.ipynb (Recommended)
Interactive Jupyter notebook that provides the best user experience:
- **Modular imports**: Uses functions from .py modules
- **Interactive visualization**: Inline plots and analysis
- **Step-by-step execution**: Run cells individually
- **Real-time feedback**: Immediate results and error handling
- **Educational**: Clear explanations and documentation
- **Lightweight**: Main logic in reusable modules

### data_loader.py
Handles data loading and preprocessing:
- `DataLoader`: Main class for data operations
- Loads alloy composition and property data from CSV
- Preprocesses features and targets
- Provides data quality checks and statistics

### alloy_objective.py
Defines the optimization objective function:
- `AlloyObjectiveFunction`: Objective function for alloy optimization
- Maps 3D compositions (Co, Mo, Ti) to performance metrics
- Uses nearest neighbor interpolation for smooth evaluation
- Supports scaling and noise injection for realistic simulation

### neural_models.py
Contains advanced neural network models:
- `ImprovedPhaseCompositionSurrogateModel`: Predicts compound ratios from elements
- `DualNetworkSurrogateModel`: Directly predicts mechanical properties
- Features improved loss functions, attention mechanisms, and ensemble learning

### optimization.py
Implements DANTE optimization algorithms:
- `run_dante_optimization()`: Main DANTE optimization function
- Integrates deep active learning and tree exploration
- Includes fallback to simple random optimization
- Provides comprehensive result analysis

### visualization.py
Comprehensive visualization tools:
- Data distribution analysis
- Model performance evaluation
- Composition-property relationships
- 3D visualization of design space
- Optimization convergence plots

### config.py
Configuration management:
- Centralized parameter settings
- Model hyperparameters
- Optimization settings
- Visualization preferences
- Environment configuration

## Usage

### Recommended: Interactive Jupyter Notebook

```bash
# Start Jupyter notebook
jupyter notebook main.ipynb
```

**Benefits of using main.ipynb:**
- 🎯 **Interactive Experience**: Run cells step-by-step
- 📊 **Inline Visualizations**: See plots directly in the notebook
- 🔍 **Real-time Analysis**: Immediate feedback and results
- 📝 **Documentation**: Built-in explanations and guidance
- 🛠️ **Debugging**: Easy to modify and experiment
- 💡 **Educational**: Perfect for learning and exploration

### Alternative: Command-line Usage

```bash
# Run the complete workflow
python main.py

# Run example with synthetic data
python run_example.py

# Test all modules
python test_imports.py
```

### Advanced Usage

```python
from data_loader import DataLoader
from alloy_objective import AlloyObjectiveFunction
from neural_models import DualNetworkSurrogateModel
from optimization import run_dante_optimization
from visualization import create_visualizations

# Load data
data_loader = DataLoader()
df = data_loader.load_data("../data.csv")
X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined = data_loader.process_data(df)

# Create objective function
obj_func = AlloyObjectiveFunction(X_elements, X_elements_with_Fe, Y_combined)

# Train neural model
model = DualNetworkSurrogateModel(input_dims=4, output_dims=2)
trained_model = model(X_elements_with_Fe, Y)

# Run optimization
results = run_dante_optimization(obj_func, trained_model, X_elements)

# Create visualizations
create_visualizations(X_elements_with_Fe, Y, X_compounds, trained_model, results)
```

## Requirements

### Core Dependencies
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Seaborn

### DANTE Framework
- dante (custom package)
  - dante.neural_surrogate
  - dante.deep_active_learning
  - dante.tree_exploration
  - dante.obj_functions
  - dante.utils

### Installation

1. Install core dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

2. Install DANTE framework (if available):
```bash
pip install dante-optimization
```

## Data Format

The input data should be a CSV file with the following columns:

### Element Compositions
- `Co`: Cobalt content (fraction)
- `Mo`: Molybdenum content (fraction)
- `Ti`: Titanium content (fraction)
- Note: Iron (Fe) content is calculated as 1 - (Co + Mo + Ti)

### Compound Compositions
- `Ni3Al`: Ni3Al compound fraction
- `Ni3Ti`: Ni3Ti compound fraction
- `Ni3V`: Ni3V compound fraction
- `NiTi`: NiTi compound fraction
- `NiTi2`: NiTi2 compound fraction

### Mechanical Properties
- `Elastic_Modulus`: Elastic modulus value
- `Yield_Strength`: Yield strength value

## Output

The framework generates several outputs:

### Model Files
- `../model_weights/`: Pre-trained neural network weights (from original notebook)
- `../model_weights/*_scalers.pkl`: Data preprocessing scalers
- `model_weights/`: Local model weights (newly trained models)

### Visualizations
- `figures/element_distributions.png`: Element composition distributions
- `figures/property_distributions.png`: Mechanical property distributions
- `figures/model_performance.png`: Model prediction accuracy
- `figures/3d_composition_space.png`: 3D visualization of design space
- `figures/optimization_convergence.png`: Optimization progress

### Results
- Optimization results (best compositions and properties)
- Model performance metrics
- Convergence analysis

## Key Features

### Improved Neural Models
- Enhanced loss functions with multiple constraints
- Attention mechanisms for better feature learning
- Layer normalization and advanced regularization
- Ensemble learning with cross-validation

### Advanced Optimization
- Deep active learning for efficient sampling
- Tree-based exploration for systematic search
- Multi-objective optimization capabilities
- Uncertainty quantification

### Comprehensive Analysis
- Data quality assessment
- Model validation and performance analysis
- Visualization of composition-property relationships
- Optimization convergence monitoring

### Smart Model Weight Management
- **Automatic Loading**: Pre-trained models from original notebook are automatically loaded
- **Path Intelligence**: System automatically finds weights in `../model_weights/` directory
- **Fallback Mechanism**: If pre-trained weights not found, trains new models
- **Dual Storage**: Original weights preserved, new weights saved locally

## Configuration

Modify `config.py` to customize:
- Neural network architectures
- Optimization parameters
- Visualization settings
- Data processing options

## Troubleshooting

### Common Issues

1. **DANTE modules not found**: Install the DANTE framework or use fallback optimization
2. **Data file not found**: Ensure data.csv is in the correct location
3. **Memory issues**: Reduce batch size or model complexity in config.py
4. **Convergence problems**: Adjust learning rates and training epochs

### Performance Tips

1. Use GPU acceleration for neural network training
2. Adjust batch sizes based on available memory
3. Use ensemble models for better predictions
4. Monitor convergence and adjust hyperparameters

## Citation

If you use this code in your research, please cite:

```
DANTE: Deep Active Neural-network-based Tree Exploration for Alloy Design
[Add appropriate citation information]
```

## License

[Add license information]

## Contact

[Add contact information for support]
