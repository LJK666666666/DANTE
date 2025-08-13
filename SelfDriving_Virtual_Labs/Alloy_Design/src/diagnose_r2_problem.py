#!/usr/bin/env python3
"""
Diagnose R² Problem

This script diagnoses why there's a huge difference between the R² values
reported during model loading vs. during visualization.

Author: DANTE Team
Date: 2024
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def diagnose_r2_problem():
    """Diagnose the R² calculation problem."""
    print("🔍 Diagnosing R² Calculation Problem")
    print("=" * 60)
    
    try:
        # Load data
        from data_loader import DataLoader
        from neural_models import DualNetworkSurrogateModel
        
        data_loader = DataLoader()
        
        # Create synthetic data to test
        print("Creating test data...")
        np.random.seed(42)
        X_elements = np.random.rand(100, 3) * 10 + 5  # Percentage form
        Fe_content = 100.0 - X_elements.sum(axis=1)
        X_elements_with_Fe = np.column_stack([X_elements, Fe_content])
        Y_properties = np.random.rand(100, 2)  # Normalized properties
        
        print(f"Data shapes:")
        print(f"  X_elements: {X_elements.shape}")
        print(f"  X_elements_with_Fe: {X_elements_with_Fe.shape}")
        print(f"  Y_properties: {Y_properties.shape}")
        
        print(f"\nData ranges:")
        print(f"  X_elements: [{X_elements.min():.2f}, {X_elements.max():.2f}]")
        print(f"  Y_properties: [{Y_properties.min():.4f}, {Y_properties.max():.4f}]")
        
        # Create and train model
        print(f"\nCreating dual network model...")
        model = DualNetworkSurrogateModel(search_dims=3, network_input_dims=4)
        
        # Train model (will use fallback)
        trained_model = model(X_elements_with_Fe, Y_properties, verbose=0)
        
        print(f"Model type: {type(trained_model)}")
        print(f"Has ensemble_model: {hasattr(model, 'ensemble_model')}")
        if hasattr(model, 'ensemble_model'):
            print(f"Ensemble model type: {type(model.ensemble_model)}")
        
        # Test predictions
        print(f"\nTesting predictions...")
        
        # Method 1: Direct model prediction
        if hasattr(model, 'ensemble_model') and model.ensemble_model is not None:
            Y_pred_ensemble = model.ensemble_model.predict(X_elements_with_Fe)
            print(f"Ensemble prediction shape: {Y_pred_ensemble.shape}")
            print(f"Ensemble prediction range: [{Y_pred_ensemble.min():.4f}, {Y_pred_ensemble.max():.4f}]")
        
        # Method 2: Direct trained model prediction
        if hasattr(trained_model, 'predict'):
            try:
                Y_pred_direct = trained_model.predict(X_elements_with_Fe)
                print(f"Direct prediction shape: {Y_pred_direct.shape}")
                print(f"Direct prediction range: [{Y_pred_direct.min():.4f}, {Y_pred_direct.max():.4f}]")
            except Exception as e:
                print(f"Direct prediction failed: {e}")
        
        # Calculate R² scores
        print(f"\nCalculating R² scores...")
        
        if hasattr(model, 'ensemble_model') and model.ensemble_model is not None:
            Y_pred = model.ensemble_model.predict(X_elements_with_Fe)
            
            # Overall R²
            r2_overall = r2_score(Y_properties.flatten(), Y_pred.flatten())
            print(f"Overall R²: {r2_overall:.6f}")
            
            # Individual property R²
            r2_elastic = r2_score(Y_properties[:, 0], Y_pred[:, 0])
            r2_yield = r2_score(Y_properties[:, 1], Y_pred[:, 1])
            
            print(f"Elastic modulus R²: {r2_elastic:.6f}")
            print(f"Yield strength R²: {r2_yield:.6f}")
            
            # Check for data issues
            print(f"\nData consistency checks:")
            print(f"  Y_properties std: [{Y_properties[:, 0].std():.6f}, {Y_properties[:, 1].std():.6f}]")
            print(f"  Y_pred std: [{Y_pred[:, 0].std():.6f}, {Y_pred[:, 1].std():.6f}]")
            
            # Check if predictions are constant
            if Y_pred[:, 0].std() < 1e-10:
                print(f"  ⚠️ Elastic modulus predictions are nearly constant!")
            if Y_pred[:, 1].std() < 1e-10:
                print(f"  ⚠️ Yield strength predictions are nearly constant!")
            
            # Check for extreme values
            if np.any(np.abs(Y_pred) > 1e10):
                print(f"  ⚠️ Predictions contain extreme values!")
                print(f"  Max prediction: {Y_pred.max():.2e}")
                print(f"  Min prediction: {Y_pred.min():.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def explain_r2_problem():
    """Explain the R² problem."""
    print("\n📚 R² Problem Explanation")
    print("=" * 50)
    
    print("Possible causes of negative R² values:")
    print("  1. 🎯 Model predictions are worse than mean prediction")
    print("     - R² = 1 - (SS_res / SS_tot)")
    print("     - If SS_res > SS_tot, then R² < 0")
    print("     - This happens when model is very poor")
    print()
    print("  2. 🔢 Data scaling issues")
    print("     - Training data in one scale, test data in another")
    print("     - Scaler not applied correctly")
    print("     - Units mismatch (percentage vs decimal)")
    print()
    print("  3. 🤖 Model type mismatch")
    print("     - Loading neural network weights")
    print("     - But using linear regression fallback for prediction")
    print("     - Different models have different capabilities")
    print()
    print("  4. 📊 Data preprocessing inconsistency")
    print("     - Training with normalized data")
    print("     - Testing with raw data")
    print("     - Inverse transform not applied correctly")
    
    print("\nLikely cause in our case:")
    print("  • High R² (0.997) during loading: Using correct neural network")
    print("  • Negative R² (-38, -43) during viz: Using fallback linear model")
    print("  • Solution: Ensure visualization uses the same model as loading")

def suggest_fixes():
    """Suggest potential fixes."""
    print("\n🔧 Suggested Fixes")
    print("=" * 30)
    
    print("1. Check model consistency:")
    print("   - Verify same model is used for loading and visualization")
    print("   - Ensure TensorFlow availability is consistent")
    print()
    print("2. Check data preprocessing:")
    print("   - Verify scalers are applied consistently")
    print("   - Check data units (percentage vs decimal)")
    print()
    print("3. Debug prediction pipeline:")
    print("   - Print intermediate values")
    print("   - Check prediction ranges")
    print("   - Verify data shapes")
    print()
    print("4. Improve fallback model:")
    print("   - Use better fallback algorithm")
    print("   - Add proper data preprocessing")
    print("   - Ensure consistent scaling")

def main():
    """Main diagnosis function."""
    print("🔍 R² Problem Diagnosis")
    print("=" * 60)
    print("Investigating why R² values differ dramatically")
    
    tests = [
        ("R² Problem Diagnosis", diagnose_r2_problem),
        ("Problem Explanation", explain_r2_problem),
        ("Suggested Fixes", suggest_fixes)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Diagnosis Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ COMPLETED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests completed")
    
    print("\n🎯 Key Findings:")
    print("  • R² difference likely due to model type mismatch")
    print("  • Loading uses neural network (high R²)")
    print("  • Visualization uses fallback model (poor R²)")
    print("  • Need to ensure consistent model usage")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
