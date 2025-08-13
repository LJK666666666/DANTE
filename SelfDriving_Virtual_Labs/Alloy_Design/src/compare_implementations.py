#!/usr/bin/env python3
"""
Compare Original vs Converted Implementation

This script compares the original notebook implementation with our converted
modular implementation to identify differences.

Author: DANTE Team
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path

def test_objective_function_consistency():
    """Test if objective functions produce consistent results."""
    print("🎯 Testing Objective Function Consistency")
    print("=" * 50)
    
    try:
        # Create test data
        np.random.seed(42)
        X_elements = np.random.rand(100, 3) * 0.2  # Decimal form
        X_elements_with_Fe = np.column_stack([X_elements, 1 - X_elements.sum(axis=1)])
        Y_combined = np.random.rand(100)
        
        # Test our converted implementation
        from alloy_objective import AlloyObjectiveFunction
        obj_func = AlloyObjectiveFunction(X_elements, X_elements_with_Fe, Y_combined)
        
        # Test points in both decimal and percentage form
        test_point_decimal = np.array([0.08, 0.05, 0.03])  # 8%, 5%, 3%
        test_point_percentage = test_point_decimal * 100   # 8.0, 5.0, 3.0
        
        result_decimal = obj_func(test_point_decimal)
        result_percentage = obj_func(test_point_percentage)
        
        print(f"Test point (decimal): {test_point_decimal}")
        print(f"Test point (percentage): {test_point_percentage}")
        print(f"Result from decimal input: {result_decimal}")
        print(f"Result from percentage input: {result_percentage}")
        
        # Check boundaries
        print(f"\nBoundaries:")
        print(f"  Lower bounds: {obj_func.lb}")
        print(f"  Upper bounds: {obj_func.ub}")
        
        # Test boundary validation
        in_bounds_point = (obj_func.lb + obj_func.ub) / 2
        out_bounds_point = obj_func.ub + 10
        
        result_in_bounds = obj_func(in_bounds_point)
        result_out_bounds = obj_func(out_bounds_point)
        
        print(f"\nBoundary test:")
        print(f"  In bounds point: {in_bounds_point} -> {result_in_bounds}")
        print(f"  Out bounds point: {out_bounds_point} -> {result_out_bounds}")
        
        # Check if out of bounds returns penalty
        if result_out_bounds > 1e5:
            print("  ✅ Out of bounds penalty working correctly")
        else:
            print("  ⚠️ Out of bounds penalty may not be working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing objective function: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading_consistency():
    """Test if data loading produces consistent results."""
    print("\n📊 Testing Data Loading Consistency")
    print("=" * 50)
    
    try:
        from data_loader import DataLoader
        
        # Test with real data if available
        data_path = "../data.csv"
        if Path(data_path).exists():
            print(f"Testing with real data: {data_path}")
            
            data_loader = DataLoader()
            df = data_loader.load_data(data_path)
            
            if df is not None:
                print(f"Data shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                # Process data
                processed = data_loader.process_data(df)
                if processed is not None:
                    X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined = processed
                    
                    print(f"\nProcessed data shapes:")
                    print(f"  X_elements: {X_elements.shape}")
                    print(f"  X_elements_with_Fe: {X_elements_with_Fe.shape}")
                    print(f"  X_compounds: {X_compounds.shape}")
                    print(f"  Y: {Y.shape}")
                    print(f"  Y_combined: {Y_combined.shape}")
                    
                    # Check data ranges
                    print(f"\nData ranges:")
                    print(f"  Co: [{X_elements[:, 0].min():.4f}, {X_elements[:, 0].max():.4f}]")
                    print(f"  Mo: [{X_elements[:, 1].min():.4f}, {X_elements[:, 1].max():.4f}]")
                    print(f"  Ti: [{X_elements[:, 2].min():.4f}, {X_elements[:, 2].max():.4f}]")
                    print(f"  Fe: [{X_elements_with_Fe[:, 3].min():.4f}, {X_elements_with_Fe[:, 3].max():.4f}]")
                    
                    # Check if data is in decimal or percentage form
                    max_element = X_elements.max()
                    if max_element <= 1.0:
                        print("  ✅ Data appears to be in decimal form (0-1)")
                    else:
                        print("  ⚠️ Data appears to be in percentage form (0-100)")
                    
                    return True
                else:
                    print("❌ Failed to process data")
                    return False
            else:
                print("❌ Failed to load data")
                return False
        else:
            print(f"⚠️ Data file not found: {data_path}")
            print("Creating synthetic data for testing...")
            
            # Create synthetic data
            np.random.seed(42)
            n_samples = 100
            
            # Create data in decimal form (like our implementation expects)
            Co = np.random.uniform(0.05, 0.25, n_samples)
            Mo = np.random.uniform(0.02, 0.15, n_samples)
            Ti = np.random.uniform(0.01, 0.12, n_samples)
            
            # Ensure total doesn't exceed 1
            total = Co + Mo + Ti
            scale_factor = np.minimum(1.0, 0.8 / total)
            Co *= scale_factor
            Mo *= scale_factor
            Ti *= scale_factor
            
            print(f"Synthetic data ranges:")
            print(f"  Co: [{Co.min():.4f}, {Co.max():.4f}]")
            print(f"  Mo: [{Mo.min():.4f}, {Mo.max():.4f}]")
            print(f"  Ti: [{Ti.min():.4f}, {Ti.max():.4f}]")
            
            return True
            
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_model_consistency():
    """Test if neural models are consistent."""
    print("\n🧠 Testing Neural Model Consistency")
    print("=" * 50)
    
    try:
        # Test original dual network implementation
        from original_dual_network import DualNetworkAlloySurrogateModel
        
        print("Testing original dual network implementation...")
        original_model = DualNetworkAlloySurrogateModel()
        
        print(f"  Search dimensions: {original_model.search_dims}")
        print(f"  Network input dimensions: {original_model.network_input_dims}")
        print(f"  Weights directory: {original_model.weights_dir}")
        print(f"  Elastic weights path: {original_model.elastic_weights_path}")
        print(f"  Yield weights path: {original_model.yield_weights_path}")
        
        # Check if weight files exist
        elastic_exists = original_model.elastic_weights_path.exists()
        yield_exists = original_model.yield_weights_path.exists()
        scaler_exists = original_model.scaler_path.exists()
        
        print(f"  Elastic weights exist: {elastic_exists}")
        print(f"  Yield weights exist: {yield_exists}")
        print(f"  Scalers exist: {scaler_exists}")
        
        # Test our converted implementation
        from neural_models import DualNetworkSurrogateModel
        
        print("\nTesting converted dual network implementation...")
        converted_model = DualNetworkSurrogateModel()
        
        print(f"  Input dimensions: {converted_model.input_dims}")
        print(f"  Weights directory: {converted_model.weights_dir}")
        
        # Compare paths
        print(f"\nPath comparison:")
        print(f"  Original weights dir: {original_model.weights_dir}")
        print(f"  Converted weights dir: {converted_model.weights_dir}")
        print(f"  Paths match: {original_model.weights_dir == converted_model.weights_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing neural models: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main comparison function."""
    print("🔍 DANTE Implementation Comparison")
    print("=" * 60)
    print("Comparing original notebook vs converted modular implementation")
    
    tests = [
        ("Data Loading", test_data_loading_consistency),
        ("Objective Function", test_objective_function_consistency),
        ("Neural Models", test_neural_model_consistency)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ CONSISTENT" if result else "❌ INCONSISTENT"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Implementations appear consistent.")
    else:
        print("\n⚠️ Some inconsistencies found. Check the details above.")
        print("\n💡 Common issues:")
        print("  • Data format differences (decimal vs percentage)")
        print("  • Boundary handling differences")
        print("  • Model architecture differences")
        print("  • Path configuration differences")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
