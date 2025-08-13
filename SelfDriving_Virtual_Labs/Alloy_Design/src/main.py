#!/usr/bin/env python3
"""
DANTE Alloy Design Virtual Lab - Main Script

This script demonstrates how to use the DANTE framework for alloy material 
composition optimization to achieve optimal mechanical properties (combination 
of elastic modulus and yield strength).

Author: DANTE Team
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add DANTE module to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

# Import local modules
from data_loader import DataLoader
from alloy_objective import AlloyObjectiveFunction
from neural_models import ImprovedPhaseCompositionSurrogateModel, DualNetworkSurrogateModel
from visualization import create_visualizations
from optimization import run_dante_optimization

# Import DANTE modules (optional)
try:
    from dante.neural_surrogate import SurrogateModel
    from dante.deep_active_learning import DeepActiveLearning
    from dante.obj_functions import ObjectiveFunction
    from dante.tree_exploration import TreeExploration
    from dante.utils import generate_initial_samples, Tracker
    print("Successfully imported DANTE modules!")
    DANTE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DANTE modules not available: {e}")
    print("Using fallback implementations.")
    DANTE_AVAILABLE = False


def main():
    """Main function to run the alloy design optimization."""
    print("=" * 60)
    print("DANTE Alloy Material Design Optimization")
    print("=" * 60)
    
    # Set visualization style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # Use original model weights directory
    weights_dir = Path("../model_weights")  # Go up one level to find original weights
    weights_dir.mkdir(exist_ok=True)
    print(f"Model weights directory: {weights_dir.absolute()}")
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_loader = DataLoader()
    
    # Check if data file exists
    data_path = "../data.csv"
    if not os.path.exists(data_path):
        print(f"Warning: Data file {data_path} not found!")
        # Try to find data file in parent directories
        parent_data_path = "../../../data.csv"
        if os.path.exists(parent_data_path):
            print("Found data file in parent directory, copying...")
            import shutil
            shutil.copy(parent_data_path, data_path)
            print("Copy completed!")
        else:
            print("Data file not found in parent directories either.")
            return
    
    # Load data
    df = data_loader.load_data(data_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Process data
    processed_data = data_loader.process_data(df)
    if processed_data is None:
        print("Failed to process data. Exiting.")
        return
    
    X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined = processed_data
    
    # Step 2: Create objective function
    print("\n2. Creating alloy optimization objective function...")
    alloy_obj_func = AlloyObjectiveFunction(X_elements, X_elements_with_Fe, Y_combined)
    print(f"Search space dimensions: {alloy_obj_func.dims}D (Co, Mo, Ti)")
    print(f"Search boundaries: Co[{alloy_obj_func.lb[0]:.2f}, {alloy_obj_func.ub[0]:.2f}], "
          f"Mo[{alloy_obj_func.lb[1]:.2f}, {alloy_obj_func.ub[1]:.2f}], "
          f"Ti[{alloy_obj_func.lb[2]:.2f}, {alloy_obj_func.ub[2]:.2f}]")
    
    # Test objective function
    test_point_3d = X_elements[0]
    print(f"\nTesting objective function:")
    print(f"3D test point: {test_point_3d}")
    print(f"Original performance value: {alloy_obj_func(test_point_3d)}")
    print(f"Scaled performance value: {alloy_obj_func(test_point_3d, apply_scaling=True)}")
    
    # Step 3: Train neural network surrogate models
    print("\n3. Training neural network surrogate models...")
    
    # Train improved phase composition model
    print("\n3.1 Training improved phase composition prediction model...")
    improved_phase_model = ImprovedPhaseCompositionSurrogateModel(
        input_dims=4,   # 4D element input
        output_dims=5,  # 5D compound output
        n_folds=5
    )
    
    trained_improved_phase_model = improved_phase_model(
        X_elements_with_Fe,  # 4D element features as input
        X_compounds,         # 5D compound ratios as target
        verbose=1
    )
    
    # Train dual network model
    print("\n3.2 Training dual network model...")
    dual_network_model = DualNetworkSurrogateModel(
        input_dims=4,   # 4D element input
        output_dims=2,  # 2D output (elastic modulus, yield strength)
        n_folds=5
    )
    
    trained_dual_model = dual_network_model(
        X_elements_with_Fe,  # 4D element features as input
        Y,                   # 2D mechanical properties as target
        verbose=1
    )
    
    # Step 4: Run DANTE optimization
    print("\n4. Running DANTE optimization...")
    optimization_results = run_dante_optimization(
        alloy_obj_func, 
        trained_dual_model,
        X_elements,
        max_iterations=50
    )
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(
        X_elements_with_Fe, 
        Y, 
        X_compounds, 
        trained_dual_model,
        optimization_results
    )
    
    print("\n" + "=" * 60)
    print("DANTE Alloy Design Optimization Completed!")
    print("=" * 60)
    print("\nMain improvements achieved:")
    print("  ✓ Improved loss function: MSE + probability constraints + KL divergence + smoothness constraints")
    print("  ✓ Enhanced network architecture: layer normalization + attention mechanism + improved residual blocks")
    print("  ✓ Optimized data preprocessing: probability distribution normalization + outlier handling")
    print("  ✓ Better training strategy: data augmentation + gradient clipping + adaptive learning rate")
    print("  ✓ Enhanced regularization: weight decay + Dropout + layer normalization")
    print("  ✓ Ensemble learning: 5-fold cross-validation ensemble model")


if __name__ == "__main__":
    main()
