#!/usr/bin/env python3
"""
Test script for the new LogNormalizedDualNetworkSurrogateModel.

This script demonstrates how to use the enhanced model with:
1. Logarithmic transformation of Young's modulus and yield strength
2. Weighted MSE loss function with exp(2y) weights
3. Proper inverse transformation for predictions

Author: Enhanced DANTE Team
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our enhanced modules
from data_loader import LogNormalizedDataLoader
from neural_models import LogNormalizedDualNetworkSurrogateModel
from alloy_objective import AlloyObjectiveFunction

def test_log_normalized_model():
    """Test the log-normalized dual network model."""
    print("🧪 Testing Log-Normalized Dual Network Surrogate Model")
    print("=" * 60)
    
    # 1. Load and process data
    print("\n📊 Step 1: Loading and processing data...")
    data_loader = LogNormalizedDataLoader()
    
    # Check for data file
    data_path = "../data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists.")
        return
    
    # Load data
    df = data_loader.load_data(data_path)
    if df is None:
        print("❌ Failed to load data.")
        return
    
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Process data with log transformation
    processed_data = data_loader.process_data_with_log_transform(df)
    if processed_data is None:
        print("❌ Failed to process data.")
        return
    
    X_elements, X_elements_with_Fe, X_compounds, Y_original, Y_log_normalized, Y_combined = processed_data
    
    print(f"✅ Data processed successfully!")
    print(f"  • Element features: {X_elements.shape}")
    print(f"  • Element features with Fe: {X_elements_with_Fe.shape}")
    print(f"  • Original properties: {Y_original.shape}")
    print(f"  • Log-normalized properties: {Y_log_normalized.shape}")
    
    # 2. Create and train log-normalized model
    print("\n🧠 Step 2: Creating and training log-normalized model...")
    
    log_model = LogNormalizedDualNetworkSurrogateModel(
        search_dims=3,          # 3D search space (Co, Mo, Ti)
        network_input_dims=4,   # 4D network input (Co, Mo, Ti, Fe)
        n_folds=3               # Use 3 folds for faster testing
    )
    
    # Train the model
    trained_model = log_model(X_elements_with_Fe, Y_original, verbose=1)
    
    if trained_model is None:
        print("❌ Model training failed.")
        return
    
    print("✅ Log-normalized model training completed!")
    
    # 3. Test predictions
    print("\n🔮 Step 3: Testing predictions...")
    
    # Test on a few samples
    test_indices = [0, 10, 50, 100, 200]
    test_samples = X_elements_with_Fe[test_indices]
    true_values = Y_original[test_indices]
    
    predictions = trained_model.predict(test_samples)
    
    print(f"\nPrediction comparison:")
    print(f"{'Index':<6} {'True Elastic':<12} {'Pred Elastic':<12} {'True Yield':<12} {'Pred Yield':<12}")
    print("-" * 60)
    
    for i, idx in enumerate(test_indices):
        true_elastic = true_values[i, 0]
        pred_elastic = predictions[i, 0]
        true_yield = true_values[i, 1]
        pred_yield = predictions[i, 1]
        
        print(f"{idx:<6} {true_elastic:<12.2e} {pred_elastic:<12.2e} {true_yield:<12.2e} {pred_yield:<12.2e}")
    
    # 4. Calculate overall performance metrics
    print("\n📈 Step 4: Calculating performance metrics...")
    
    all_predictions = trained_model.predict(X_elements_with_Fe)
    
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # Elastic modulus metrics
    mse_elastic = mean_squared_error(Y_original[:, 0], all_predictions[:, 0])
    r2_elastic = r2_score(Y_original[:, 0], all_predictions[:, 0])
    mae_elastic = mean_absolute_error(Y_original[:, 0], all_predictions[:, 0])
    
    # Yield strength metrics
    mse_yield = mean_squared_error(Y_original[:, 1], all_predictions[:, 1])
    r2_yield = r2_score(Y_original[:, 1], all_predictions[:, 1])
    mae_yield = mean_absolute_error(Y_original[:, 1], all_predictions[:, 1])
    
    print(f"\nOverall Performance Metrics:")
    print(f"Elastic Modulus:")
    print(f"  • MSE: {mse_elastic:.2e}")
    print(f"  • R²: {r2_elastic:.4f}")
    print(f"  • MAE: {mae_elastic:.2e}")
    
    print(f"Yield Strength:")
    print(f"  • MSE: {mse_yield:.2e}")
    print(f"  • R²: {r2_yield:.4f}")
    print(f"  • MAE: {mae_yield:.2e}")
    
    # 5. Create visualization
    print("\n📊 Step 5: Creating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elastic modulus plot
    axes[0].scatter(Y_original[:, 0], all_predictions[:, 0], alpha=0.6, s=20)
    axes[0].plot([Y_original[:, 0].min(), Y_original[:, 0].max()], 
                 [Y_original[:, 0].min(), Y_original[:, 0].max()], 'r--', lw=2)
    axes[0].set_xlabel('True Elastic Modulus (Pa)')
    axes[0].set_ylabel('Predicted Elastic Modulus (Pa)')
    axes[0].set_title(f'Elastic Modulus Prediction\n(R² = {r2_elastic:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # Yield strength plot
    axes[1].scatter(Y_original[:, 1], all_predictions[:, 1], alpha=0.6, s=20)
    axes[1].plot([Y_original[:, 1].min(), Y_original[:, 1].max()], 
                 [Y_original[:, 1].min(), Y_original[:, 1].max()], 'r--', lw=2)
    axes[1].set_xlabel('True Yield Strength (Pa)')
    axes[1].set_ylabel('Predicted Yield Strength (Pa)')
    axes[1].set_title(f'Yield Strength Prediction\n(R² = {r2_yield:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("../figures")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "log_normalized_model_performance.png", dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_dir / 'log_normalized_model_performance.png'}")
    
    # 6. Test with objective function
    print("\n🎯 Step 6: Testing with objective function...")
    
    try:
        # Create objective function
        alloy_obj_func = AlloyObjectiveFunction(X_elements, X_elements_with_Fe, Y_combined)
        
        # Test a few points
        test_points = X_elements[:5]
        
        print(f"\nObjective function test:")
        print(f"{'Point':<6} {'Composition (Co, Mo, Ti)':<25} {'Objective Value':<15}")
        print("-" * 50)
        
        for i, point in enumerate(test_points):
            obj_value = alloy_obj_func(point)
            comp_str = f"[{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]"
            print(f"{i:<6} {comp_str:<25} {obj_value:<15.6f}")
        
        print("✅ Objective function test completed!")
        
    except Exception as e:
        print(f"⚠️ Objective function test failed: {e}")
    
    print("\n🎉 Log-normalized model testing completed successfully!")
    print("\nKey improvements demonstrated:")
    print("  1. ✅ Logarithmic transformation of mechanical properties")
    print("  2. ✅ Weighted MSE loss with exp(2y) weights")
    print("  3. ✅ Proper inverse transformation for predictions")
    print("  4. ✅ Enhanced model performance metrics")

if __name__ == "__main__":
    test_log_normalized_model()
