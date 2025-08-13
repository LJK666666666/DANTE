#!/usr/bin/env python3
"""
Example Script for DANTE Alloy Design

This script demonstrates how to use the DANTE framework modules
with sample data or synthetic data if the original data is not available.

Author: DANTE Team
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from data_loader import DataLoader
from alloy_objective import AlloyObjectiveFunction
from neural_models import DualNetworkSurrogateModel
from optimization import run_simple_optimization
from visualization import create_visualizations
from config import get_config, validate_config


def create_synthetic_data(n_samples=100):
    """
    Create synthetic alloy data for demonstration purposes.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic alloy data
    """
    print(f"Creating synthetic data with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate element compositions (Co, Mo, Ti)
    # Ensure they sum to less than 1 (remaining is Fe)
    Co = np.random.uniform(0.05, 0.25, n_samples)
    Mo = np.random.uniform(0.02, 0.15, n_samples)
    Ti = np.random.uniform(0.01, 0.12, n_samples)
    
    # Ensure total doesn't exceed 1 (leave room for Fe)
    total = Co + Mo + Ti
    scale_factor = np.minimum(1.0, 0.8 / total)  # Scale to max 80% to leave room for Fe
    Co *= scale_factor
    Mo *= scale_factor
    Ti *= scale_factor
    
    # Generate compound compositions (simplified relationships)
    Ni3Al = 0.1 + 0.3 * Co + 0.1 * np.random.normal(0, 0.05, n_samples)
    Ni3Ti = 0.05 + 0.4 * Ti + 0.1 * np.random.normal(0, 0.03, n_samples)
    Ni3V = 0.02 + 0.2 * Mo + 0.1 * np.random.normal(0, 0.02, n_samples)
    NiTi = 0.03 + 0.3 * Ti + 0.1 * np.random.normal(0, 0.02, n_samples)
    NiTi2 = 0.01 + 0.2 * Ti + 0.1 * np.random.normal(0, 0.01, n_samples)
    
    # Ensure non-negative compounds
    Ni3Al = np.maximum(Ni3Al, 0.01)
    Ni3Ti = np.maximum(Ni3Ti, 0.01)
    Ni3V = np.maximum(Ni3V, 0.01)
    NiTi = np.maximum(NiTi, 0.01)
    NiTi2 = np.maximum(NiTi2, 0.01)
    
    # Generate mechanical properties with realistic relationships
    # Elastic modulus: influenced by all elements
    Elastic_Modulus = (200 + 50 * Co + 30 * Mo + 40 * Ti + 
                      10 * np.random.normal(0, 5, n_samples))
    
    # Yield strength: different relationships
    Yield_Strength = (300 + 100 * Co + 80 * Mo + 60 * Ti + 
                     20 * np.random.normal(0, 10, n_samples))
    
    # Ensure positive values
    Elastic_Modulus = np.maximum(Elastic_Modulus, 150)
    Yield_Strength = np.maximum(Yield_Strength, 200)
    
    # Create DataFrame
    data = {
        'Co': Co,
        'Mo': Mo,
        'Ti': Ti,
        'Ni3Al': Ni3Al,
        'Ni3Ti': Ni3Ti,
        'Ni3V': Ni3V,
        'NiTi': NiTi,
        'NiTi2': NiTi2,
        'Elastic_Modulus': Elastic_Modulus,
        'Yield_Strength': Yield_Strength
    }
    
    df = pd.DataFrame(data)
    
    print("Synthetic data created successfully!")
    print(f"Data shape: {df.shape}")
    print("\nData summary:")
    print(df.describe())
    
    return df


def run_example():
    """Run the complete DANTE alloy design example."""
    print("=" * 60)
    print("DANTE Alloy Design - Example Run")
    print("=" * 60)
    
    # Validate configuration
    try:
        validate_config()
        print("Configuration validated successfully!")
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return
    
    # Step 1: Load or create data
    print("\n1. Loading data...")
    data_loader = DataLoader()
    
    # Try to load real data first
    data_path = "../data.csv"
    if os.path.exists(data_path):
        print(f"Loading real data from {data_path}")
        df = data_loader.load_data(data_path)
    else:
        print("Real data not found, creating synthetic data...")
        df = create_synthetic_data(n_samples=150)
        
        # Save synthetic data for future use
        synthetic_path = "synthetic_alloy_data.csv"
        df.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to {synthetic_path}")
    
    if df is None:
        print("Failed to load or create data. Exiting.")
        return
    
    # Process data
    processed_data = data_loader.process_data(df)
    if processed_data is None:
        print("Failed to process data. Exiting.")
        return
    
    X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined = processed_data
    
    # Step 2: Create objective function
    print("\n2. Creating objective function...")
    try:
        alloy_obj_func = AlloyObjectiveFunction(X_elements, X_elements_with_Fe, Y_combined)
        print(f"Objective function created successfully!")
        print(f"Search space: {alloy_obj_func.dims}D")
        print(f"Bounds: {alloy_obj_func.lb} to {alloy_obj_func.ub}")
        
        # Test objective function
        test_point = X_elements[0]
        test_value = alloy_obj_func(test_point)
        print(f"Test evaluation: f({test_point}) = {test_value:.6f}")
        
    except Exception as e:
        print(f"Failed to create objective function: {e}")
        return
    
    # Step 3: Train neural network model
    print("\n3. Training neural network model...")
    try:
        # Use a simpler model for the example
        dual_model = DualNetworkSurrogateModel(
            input_dims=4,   # Co, Mo, Ti, Fe
            output_dims=2,  # Elastic modulus, Yield strength
            n_folds=3       # Reduced for faster training
        )
        
        print("Training dual network model...")
        trained_model = dual_model(X_elements_with_Fe, Y, verbose=1)
        print("Model training completed!")
        
    except Exception as e:
        print(f"Failed to train neural network: {e}")
        print("Continuing without neural model...")
        trained_model = None
    
    # Step 4: Run optimization
    print("\n4. Running optimization...")
    try:
        # Use simple optimization for the example
        optimization_results = run_simple_optimization(
            alloy_obj_func, 
            X_elements,
            max_iterations=30,  # Reduced for faster execution
            verbose=True
        )
        
        print("Optimization completed!")
        print(f"Best value found: {optimization_results['best_value']:.6f}")
        print(f"Best composition: {optimization_results['best_point']}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        optimization_results = None
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    try:
        if trained_model is not None:
            create_visualizations(
                X_elements_with_Fe, 
                Y, 
                X_compounds, 
                trained_model,
                optimization_results
            )
            print("Visualizations created successfully!")
        else:
            print("Skipping visualizations due to missing trained model.")
            
    except Exception as e:
        print(f"Visualization creation failed: {e}")
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Example Run Summary")
    print("=" * 60)
    print(f"Data samples: {len(df)}")
    print(f"Element features: {X_elements.shape[1]}")
    print(f"Compound features: {X_compounds.shape[1]}")
    print(f"Target properties: {Y.shape[1]}")
    
    if optimization_results:
        print(f"\nOptimization Results:")
        print(f"  Best objective value: {optimization_results['best_value']:.6f}")
        print(f"  Best composition (Co, Mo, Ti): {optimization_results['best_point']}")
        print(f"  Total evaluations: {optimization_results['total_evaluations']}")
    
    if trained_model:
        print(f"\nModel Training: Successful")
    else:
        print(f"\nModel Training: Failed or Skipped")
    
    print("\nFiles generated:")
    print("  - Model weights (if training successful)")
    print("  - Visualization figures (if successful)")
    print("  - Synthetic data (if used)")
    
    print("\nExample run completed!")


if __name__ == "__main__":
    try:
        run_example()
    except KeyboardInterrupt:
        print("\nExample run interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error during example run: {e}")
        import traceback
        traceback.print_exc()
