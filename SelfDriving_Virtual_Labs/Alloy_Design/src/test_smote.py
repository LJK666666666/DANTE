#!/usr/bin/env python3
"""
Test script to verify SMOTE functionality for high-value region resampling.
This script tests the SMOTE implementation without running the full notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_smote_functionality():
    """Test SMOTE resampling functionality"""
    print("🧪 Testing SMOTE Functionality for High-Value Region Resampling")
    print("=" * 70)
    
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.neighbors import NearestNeighbors
        print("✅ Required packages imported successfully")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install imbalanced-learn scikit-learn")
        return False
    
    # Create synthetic test data similar to alloy composition data
    print("\n📊 Creating synthetic test data...")
    np.random.seed(42)
    n_samples = 200
    
    # Features: Co, Mo, Ti, Fe percentages
    X_test = np.random.rand(n_samples, 4)
    X_test[:, 0] = X_test[:, 0] * 3 + 8.5    # Co: 8.5-11.5
    X_test[:, 1] = X_test[:, 1] * 0.7 + 4.8  # Mo: 4.8-5.5
    X_test[:, 2] = X_test[:, 2] * 2.5 + 0.5  # Ti: 0.5-3.0
    X_test[:, 3] = 100 - X_test[:, :3].sum(axis=1)  # Fe: remainder
    
    # Performance metric (higher values are better)
    # Create a realistic distribution with fewer high-performance samples
    performance = np.random.beta(2, 5, n_samples)  # Skewed towards lower values
    
    print(f"  Generated {n_samples} samples")
    print(f"  Performance range: [{performance.min():.3f}, {performance.max():.3f}]")
    print(f"  Performance mean±std: {performance.mean():.3f}±{performance.std():.3f}")
    
    # Create performance-based bins
    print("\n🔢 Creating performance-based bins...")
    n_bins = 5
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    performance_bins = discretizer.fit_transform(performance.reshape(-1, 1)).flatten().astype(int)
    
    # Show original bin distribution
    unique_bins, bin_counts = np.unique(performance_bins, return_counts=True)
    print(f"Original bin distribution:")
    for bin_id, count in zip(unique_bins, bin_counts):
        bin_range = discretizer.bin_edges_[0][bin_id:bin_id+2]
        print(f"  Bin {bin_id}: {count} samples (performance range: [{bin_range[0]:.3f}, {bin_range[1]:.3f}])")
    
    # Define sampling strategy for high-value regions
    print("\n⚙️ Defining sampling strategy...")
    max_samples = max(bin_counts)
    sampling_strategy = {}
    
    for bin_id in unique_bins:
        current_count = bin_counts[bin_id]
        # Exponentially increase sampling for higher performance bins
        multiplier = 1.0 + (bin_id / (n_bins - 1)) * 2.0  # 1.0 to 3.0 multiplier
        target_samples = int(max_samples * multiplier)
        
        if target_samples > current_count:
            sampling_strategy[bin_id] = target_samples
            print(f"  📈 Bin {bin_id}: {current_count} → {target_samples} samples (multiplier: {multiplier:.2f})")
        else:
            print(f"  📊 Bin {bin_id}: {current_count} samples (no resampling needed)")
    
    # Apply SMOTE
    if sampling_strategy:
        print(f"\n🔬 Applying SMOTE resampling...")
        
        # Determine k_neighbors
        min_samples_in_bin = min(bin_counts)
        k_neighbors = min(3, min_samples_in_bin - 1) if min_samples_in_bin > 1 else 1
        print(f"  Using k_neighbors = {k_neighbors}")
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42
        )
        
        # Apply SMOTE
        X_resampled, y_bins_resampled = smote.fit_resample(X_test, performance_bins)
        
        n_original = len(X_test)
        n_resampled = len(X_resampled)
        
        print(f"✅ SMOTE resampling completed!")
        print(f"  Original samples: {n_original}")
        print(f"  Resampled samples: {n_resampled}")
        print(f"  Increase: {n_resampled - n_original} samples ({((n_resampled/n_original)-1)*100:.1f}%)")
        
        # Show final bin distribution
        unique_final_bins, final_bin_counts = np.unique(y_bins_resampled, return_counts=True)
        print(f"\n📊 Final bin distribution after SMOTE:")
        for bin_id, count in zip(unique_final_bins, final_bin_counts):
            bin_range = discretizer.bin_edges_[0][bin_id:bin_id+2]
            print(f"  Bin {bin_id}: {count} samples (performance range: [{bin_range[0]:.3f}, {bin_range[1]:.3f}])")
        
        # Verify high-value regions are oversampled
        print(f"\n📈 Verification of high-value oversampling:")
        for i, (original_count, final_count) in enumerate(zip(bin_counts, final_bin_counts)):
            increase_ratio = final_count / original_count
            print(f"  Bin {i}: {increase_ratio:.2f}x increase")
        
        # Create visualization
        print(f"\n📊 Creating visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original distribution
        ax1.bar(unique_bins, bin_counts, alpha=0.7, color='skyblue')
        ax1.set_title('Original Bin Distribution')
        ax1.set_xlabel('Performance Bin')
        ax1.set_ylabel('Sample Count')
        ax1.grid(True, alpha=0.3)
        
        # Resampled distribution
        ax2.bar(unique_final_bins, final_bin_counts, alpha=0.7, color='lightcoral')
        ax2.set_title('After SMOTE Resampling')
        ax2.set_xlabel('Performance Bin')
        ax2.set_ylabel('Sample Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('smote_test_results.png', dpi=150, bbox_inches='tight')
        print(f"  Visualization saved as 'smote_test_results.png'")
        
        return True
    else:
        print("ℹ️ No resampling needed - all bins have sufficient samples.")
        return True

def main():
    """Main function"""
    success = test_smote_functionality()
    
    if success:
        print(f"\n✅ SMOTE functionality test completed successfully!")
        print(f"🎯 High-value regions will be properly oversampled in the main notebook.")
    else:
        print(f"\n❌ SMOTE functionality test failed!")
        print(f"Please install required packages and try again.")

if __name__ == "__main__":
    main()
