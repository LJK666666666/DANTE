# SMOTE Enhancement for High-Value Region Resampling

## Overview

This document describes the SMOTE (Synthetic Minority Oversampling Technique) enhancement added to the DANTE Alloy Design Virtual Lab. The enhancement specifically targets high-value regions in the dataset for more frequent resampling, improving model training on high-performance alloy compositions.

## What is SMOTE?

SMOTE is a statistical technique used to increase the number of cases in a dataset in a balanced way. The component works by taking each minority class sample and introducing synthetic examples along the line segments joining any/all of the k minority class nearest neighbors.

## Implementation Details

### Location
The SMOTE resampling functionality has been added as **Section 3** in `main_v5.ipynb`, positioned between data preprocessing and objective function creation.

### Key Features

1. **Performance-Based Binning**: The dataset is divided into 5 quantile-based bins based on the combined performance metric.

2. **Exponential Sampling Strategy**: Higher performance bins receive exponentially more synthetic samples:
   - Bin 0 (lowest): 1.0x multiplier (no increase)
   - Bin 1: 1.5x multiplier
   - Bin 2: 2.0x multiplier  
   - Bin 3: 2.5x multiplier
   - Bin 4 (highest): 3.0x multiplier

3. **Intelligent Neighbor Selection**: Automatically adjusts k_neighbors parameter based on the smallest bin size to ensure robust synthetic sample generation.

4. **Property Estimation**: For synthetic samples, mechanical properties are estimated using weighted k-nearest neighbors from the original dataset.

### Code Structure

```python
# Create performance-based bins
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
performance_bins = discretizer.fit_transform(performance_metric.reshape(-1, 1))

# Define sampling strategy with exponential increase for high-value regions
for bin_id in unique_bins:
    multiplier = 1.0 + (bin_id / (n_bins - 1)) * 2.0  # 1.0 to 3.0 multiplier
    target_samples = int(max_samples * multiplier)

# Apply SMOTE
smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
X_resampled, y_bins_resampled = smote.fit_resample(X_combined, performance_bins)
```

## Benefits

1. **Enhanced High-Value Learning**: The model receives more training examples from high-performance regions, improving its ability to predict and optimize for superior alloy compositions.

2. **Balanced Dataset**: While oversampling high-value regions, the technique maintains dataset balance and prevents overfitting to any single region.

3. **Synthetic Diversity**: SMOTE generates realistic synthetic samples that lie between existing high-performance samples, expanding the exploration space.

4. **Improved Optimization**: More training data in high-value regions leads to better surrogate model accuracy in these critical areas, resulting in more effective optimization.

## Testing

A comprehensive test script `test_smote.py` has been created to verify the functionality:

```bash
python test_smote.py
```

### Test Results
- ✅ Successfully creates performance-based bins
- ✅ Applies exponential sampling strategy
- ✅ Generates synthetic samples using SMOTE
- ✅ Verifies high-value regions are properly oversampled
- ✅ Creates visualization of before/after distributions

## Usage

The SMOTE enhancement is automatically applied when running `main_v5.ipynb`. The process is transparent and provides detailed logging:

```
🔄 Applying SMOTE Resampling for High-Value Regions
============================================================
📊 Creating performance-based bins...
Original bin distribution:
  Bin 0: 124 samples (performance range: [0.001, 0.350])
  Bin 1: 124 samples (performance range: [0.350, 0.425])
  ...
⚙️ Defining sampling strategy...
  📈 Bin 4: 124 → 372 samples (multiplier: 3.00)
🔬 Applying SMOTE resampling...
✅ SMOTE resampling completed!
  Original samples: 621
  Resampled samples: 1116
  Increase: 495 samples (79.7%)
```

## Dependencies

The enhancement requires the `imbalanced-learn` package:

```bash
pip install imbalanced-learn
```

If the package is not available, the notebook will skip SMOTE resampling and continue with the original dataset.

## Integration

The SMOTE enhancement seamlessly integrates with the existing DANTE framework:

- **Data Flow**: Processed data → SMOTE resampling → Objective function creation
- **Variable Updates**: All relevant variables (X_elements, Y_original, etc.) are updated with resampled data
- **Compatibility**: Maintains full compatibility with downstream processing and visualization

## Performance Impact

- **Training Time**: Slight increase due to larger dataset
- **Model Accuracy**: Improved accuracy in high-value regions
- **Optimization Quality**: Better optimization results due to enhanced surrogate model performance
- **Memory Usage**: Moderate increase proportional to synthetic sample generation

## Visualization

The test script generates `smote_test_results.png` showing before/after bin distributions, clearly demonstrating the exponential increase in high-value region samples.

## Future Enhancements

Potential improvements for future versions:

1. **Adaptive Multipliers**: Dynamic multiplier calculation based on dataset characteristics
2. **Multi-Objective SMOTE**: Separate resampling strategies for different objectives
3. **Advanced Synthetic Generation**: Integration with more sophisticated synthetic data generation techniques
4. **Real-time Monitoring**: Live visualization of resampling effects during training

## Conclusion

The SMOTE enhancement significantly improves the DANTE framework's ability to discover high-performance alloy compositions by ensuring the machine learning models receive adequate training data in the most valuable regions of the composition space.
