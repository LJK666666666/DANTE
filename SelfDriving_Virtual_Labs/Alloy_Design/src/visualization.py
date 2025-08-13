"""
Visualization Module for DANTE Alloy Design

This module provides comprehensive visualization functions for analyzing
alloy composition data, model performance, and optimization results.

Author: DANTE Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font for matplotlib (if needed)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")


def create_visualizations(X_elements_with_Fe, Y, X_compounds, trained_dual_model, optimization_results=None):
    """
    Create comprehensive visualizations for alloy design analysis.
    
    Args:
        X_elements_with_Fe (np.ndarray): 4D element features including Fe
        Y (np.ndarray): Mechanical properties (elastic modulus, yield strength)
        X_compounds (np.ndarray): Compound compositions
        trained_dual_model: Trained dual network model
        optimization_results (dict): DANTE optimization results
    """
    print("Creating comprehensive visualizations...")
    
    # Create figure directory
    import os
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Data distribution analysis
    create_data_distribution_plots(X_elements_with_Fe, Y, X_compounds, fig_dir)
    
    # 2. Model performance visualization
    create_model_performance_plots(X_elements_with_Fe, Y, trained_dual_model, fig_dir)
    
    # 3. Composition-property relationships
    create_composition_property_plots(X_elements_with_Fe, Y, fig_dir)
    
    # 4. 3D visualization
    create_3d_visualization(X_elements_with_Fe, Y, fig_dir)
    
    # 5. Optimization results (if available)
    if optimization_results is not None:
        create_optimization_plots(optimization_results, fig_dir)
    
    print(f"All visualizations saved to {fig_dir}/ directory")


def create_data_distribution_plots(X_elements_with_Fe, Y, X_compounds, fig_dir):
    """Create data distribution analysis plots."""
    print("Creating data distribution plots...")
    
    # Element composition distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    element_names = ['Co', 'Mo', 'Ti', 'Fe']
    
    for i, (ax, name) in enumerate(zip(axes.flat, element_names)):
        ax.hist(X_elements_with_Fe[:, i], bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
        ax.set_title(f'{name} Content Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{name} Fraction', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(X_elements_with_Fe[:, i])
        std_val = np.std(X_elements_with_Fe[:, i])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/element_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Mechanical properties distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    property_names = ['Elastic Modulus', 'Yield Strength']
    
    for i, (ax, name) in enumerate(zip(axes, property_names)):
        ax.hist(Y[:, i], bins=30, alpha=0.7, color=f'C{i+4}', edgecolor='black')
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(Y[:, i])
        std_val = np.std(Y[:, i])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/property_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compound composition distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    compound_names = ['Ni3Al', 'Ni3Ti', 'Ni3V', 'NiTi', 'NiTi2']
    
    for i, name in enumerate(compound_names):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        ax.hist(X_compounds[:, i], bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
        ax.set_title(f'{name} Content Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{name} Fraction', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(X_compounds[:, i])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/compound_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_model_performance_plots(X_elements_with_Fe, Y, trained_dual_model, fig_dir):
    """Create model performance visualization plots."""
    print("Creating model performance plots...")
    
    # Get model predictions
    # Check if model expects 3D or 4D input by testing the model type
    # TensorFlow models expect 4D input (Co, Mo, Ti, Fe)
    # Fallback models expect 3D input (Co, Mo, Ti)

    # Try 4D input first (for TensorFlow models)
    try:
        Y_pred = trained_dual_model.predict(X_elements_with_Fe, verbose=0)
    except Exception as e:
        # If 4D fails, try 3D input (for fallback models)
        try:
            X_elements = X_elements_with_Fe[:, :3]  # Take first 3 columns (Co, Mo, Ti)
            Y_pred = trained_dual_model.predict(X_elements, verbose=0)
        except TypeError:
            # Fallback for models that don't accept verbose parameter
            Y_pred = trained_dual_model.predict(X_elements)
    
    # Calculate metrics
    mse_elastic = mean_squared_error(Y[:, 0], Y_pred[:, 0])
    r2_elastic = r2_score(Y[:, 0], Y_pred[:, 0])
    mse_yield = mean_squared_error(Y[:, 1], Y_pred[:, 1])
    r2_yield = r2_score(Y[:, 1], Y_pred[:, 1])
    
    print(f"Model performance:")
    print(f"  Elastic modulus - R²: {r2_elastic:.4f}, MSE: {mse_elastic:.2e}")
    print(f"  Yield strength - R²: {r2_yield:.4f}, MSE: {mse_yield:.2e}")
    
    # Create prediction vs actual plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    property_names = ['Elastic Modulus', 'Yield Strength']
    metrics = [(r2_elastic, mse_elastic), (r2_yield, mse_yield)]
    
    for i, (ax, name, (r2, mse)) in enumerate(zip(axes, property_names, metrics)):
        # Scatter plot
        ax.scatter(Y[:, i], Y_pred[:, i], alpha=0.6, color=f'C{i}', s=50)
        
        # Perfect prediction line
        min_val = min(Y[:, i].min(), Y_pred[:, i].min())
        max_val = max(Y[:, i].max(), Y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {name}', fontsize=12)
        ax.set_ylabel(f'Predicted {name}', fontsize=12)
        ax.set_title(f'{name} Prediction\nR² = {r2:.4f}, MSE = {mse:.2e}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add correlation coefficient
        corr = np.corrcoef(Y[:, i], Y_pred[:, i])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residual plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (ax, name) in enumerate(zip(axes, property_names)):
        residuals = Y[:, i] - Y_pred[:, i]
        ax.scatter(Y_pred[:, i], residuals, alpha=0.6, color=f'C{i}', s=50)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel(f'Predicted {name}', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(f'{name} Residuals', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add residual statistics
        residual_std = np.std(residuals)
        ax.text(0.05, 0.95, f'Residual Std: {residual_std:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/residual_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_composition_property_plots(X_elements_with_Fe, Y, fig_dir):
    """Create composition-property relationship plots."""
    print("Creating composition-property relationship plots...")
    
    element_names = ['Co', 'Mo', 'Ti', 'Fe']
    property_names = ['Elastic Modulus', 'Yield Strength']
    
    # Correlation heatmap
    # Combine element compositions and properties
    data_combined = np.column_stack([X_elements_with_Fe, Y])
    column_names = element_names + property_names
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_combined.T)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(range(len(column_names)))
    ax.set_yticks(range(len(column_names)))
    ax.set_xticklabels(column_names, rotation=45, ha='right')
    ax.set_yticklabels(column_names)
    
    # Add correlation values
    for i in range(len(column_names)):
        for j in range(len(column_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Element-Property Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual element-property scatter plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, element in enumerate(element_names):
        for j, property_name in enumerate(property_names):
            ax = axes[j, i]
            ax.scatter(X_elements_with_Fe[:, i], Y[:, j], alpha=0.6, color=f'C{i}', s=50)
            ax.set_xlabel(f'{element} Content', fontsize=10)
            ax.set_ylabel(property_name, fontsize=10)
            ax.set_title(f'{element} vs {property_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(X_elements_with_Fe[:, i], Y[:, j], 1)
            p = np.poly1d(z)
            ax.plot(X_elements_with_Fe[:, i], p(X_elements_with_Fe[:, i]), "r--", alpha=0.8)
            
            # Add correlation coefficient
            corr = np.corrcoef(X_elements_with_Fe[:, i], Y[:, j])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/element_property_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_3d_visualization(X_elements_with_Fe, Y, fig_dir):
    """Create 3D visualization of composition-property space."""
    print("Creating 3D visualization...")

    # 3D scatter plot: Co, Mo, Ti vs combined performance
    fig = plt.figure(figsize=(15, 5))

    # Combined performance metric
    Y_combined = 0.5 * (Y[:, 0] - Y[:, 0].min()) / (Y[:, 0].max() - Y[:, 0].min()) + \
                 0.5 * (Y[:, 1] - Y[:, 1].min()) / (Y[:, 1].max() - Y[:, 1].min())

    # Plot 1: Co-Mo-Ti space colored by combined performance
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(X_elements_with_Fe[:, 0], X_elements_with_Fe[:, 1], X_elements_with_Fe[:, 2],
                         c=Y_combined, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('Co Content', fontsize=10)
    ax1.set_ylabel('Mo Content', fontsize=10)
    ax1.set_zlabel('Ti Content', fontsize=10)
    ax1.set_title('Co-Mo-Ti Space\n(Colored by Performance)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter, ax=ax1, shrink=0.5, aspect=20)

    # Plot 2: Co-Mo-Ti space colored by elastic modulus
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(X_elements_with_Fe[:, 0], X_elements_with_Fe[:, 1], X_elements_with_Fe[:, 2],
                          c=Y[:, 0], cmap='plasma', s=50, alpha=0.7)
    ax2.set_xlabel('Co Content', fontsize=10)
    ax2.set_ylabel('Mo Content', fontsize=10)
    ax2.set_zlabel('Ti Content', fontsize=10)
    ax2.set_title('Co-Mo-Ti Space\n(Colored by Elastic Modulus)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20)

    # Plot 3: Co-Mo-Ti space colored by yield strength
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(X_elements_with_Fe[:, 0], X_elements_with_Fe[:, 1], X_elements_with_Fe[:, 2],
                          c=Y[:, 1], cmap='coolwarm', s=50, alpha=0.7)
    ax3.set_xlabel('Co Content', fontsize=10)
    ax3.set_ylabel('Mo Content', fontsize=10)
    ax3.set_zlabel('Ti Content', fontsize=10)
    ax3.set_title('Co-Mo-Ti Space\n(Colored by Yield Strength)', fontsize=11, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/3d_composition_space.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2D projections
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Element pairs
    element_pairs = [('Co', 'Mo', 0, 1), ('Co', 'Ti', 0, 2), ('Mo', 'Ti', 1, 2)]
    properties = [('Combined Performance', Y_combined), ('Elastic Modulus', Y[:, 0])]

    for i, (prop_name, prop_values) in enumerate(properties):
        for j, (elem1, elem2, idx1, idx2) in enumerate(element_pairs):
            ax = axes[i, j]
            scatter = ax.scatter(X_elements_with_Fe[:, idx1], X_elements_with_Fe[:, idx2],
                               c=prop_values, cmap='viridis', s=50, alpha=0.7)
            ax.set_xlabel(f'{elem1} Content', fontsize=10)
            ax.set_ylabel(f'{elem2} Content', fontsize=10)
            ax.set_title(f'{elem1}-{elem2} Space\n(Colored by {prop_name})', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/2d_composition_projections.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_optimization_plots(optimization_results, fig_dir):
    """Create optimization results visualization."""
    print("Creating optimization results plots...")

    if 'convergence_history' in optimization_results:
        # Convergence plot
        fig, ax = plt.subplots(figsize=(12, 8))

        history = optimization_results['convergence_history']
        iterations = range(1, len(history) + 1)

        ax.plot(iterations, history, 'b-', linewidth=2, marker='o', markersize=6)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Objective Value', fontsize=12)
        ax.set_title('DANTE Optimization Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add final value annotation
        final_value = history[-1]
        ax.annotate(f'Final: {final_value:.4f}',
                   xy=(len(history), final_value),
                   xytext=(len(history)*0.8, final_value*1.1),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{fig_dir}/optimization_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

    if 'best_points' in optimization_results:
        # Best points visualization
        best_points = optimization_results['best_points']
        best_values = optimization_results['best_values']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        element_names = ['Co', 'Mo', 'Ti']

        for i, (ax, name) in enumerate(zip(axes, element_names)):
            ax.scatter(range(len(best_points)), [point[i] for point in best_points],
                      c=best_values, cmap='viridis', s=100, alpha=0.7)
            ax.set_xlabel('Optimization Step', fontsize=12)
            ax.set_ylabel(f'{name} Content', fontsize=12)
            ax.set_title(f'Best {name} Content Evolution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{fig_dir}/best_points_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_pareto_front(objectives, fig_dir):
    """Plot Pareto front for multi-objective optimization."""
    print("Creating Pareto front plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Assuming objectives is a 2D array with [elastic_modulus, yield_strength]
    ax.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=50, color='blue')
    ax.set_xlabel('Elastic Modulus', fontsize=12)
    ax.set_ylabel('Yield Strength', fontsize=12)
    ax.set_title('Pareto Front - Elastic Modulus vs Yield Strength', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(X_elements_with_Fe, Y, X_compounds, trained_dual_model, fig_dir):
    """Create a summary report with key statistics and insights."""
    print("Creating summary report...")

    # Calculate key statistics
    n_samples = X_elements_with_Fe.shape[0]
    element_names = ['Co', 'Mo', 'Ti', 'Fe']
    property_names = ['Elastic Modulus', 'Yield Strength']

    # Get model predictions
    if hasattr(trained_dual_model, 'ensemble_model') and trained_dual_model.ensemble_model is not None:
        Y_pred = trained_dual_model.ensemble_model.predict(X_elements_with_Fe)
    else:
        Y_pred = trained_dual_model.predict(X_elements_with_Fe)

    # Calculate performance metrics
    r2_scores = [r2_score(Y[:, i], Y_pred[:, i]) for i in range(2)]
    mse_scores = [mean_squared_error(Y[:, i], Y_pred[:, i]) for i in range(2)]

    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Element composition summary
    ax1 = axes[0, 0]
    element_means = np.mean(X_elements_with_Fe, axis=0)
    element_stds = np.std(X_elements_with_Fe, axis=0)

    x_pos = np.arange(len(element_names))
    bars = ax1.bar(x_pos, element_means, yerr=element_stds, capsize=5,
                   color=['C0', 'C1', 'C2', 'C3'], alpha=0.7)
    ax1.set_xlabel('Elements', fontsize=12)
    ax1.set_ylabel('Average Content', fontsize=12)
    ax1.set_title('Average Element Composition', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(element_names)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars, element_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Top-right: Property summary
    ax2 = axes[0, 1]
    property_means = np.mean(Y, axis=0)
    property_stds = np.std(Y, axis=0)

    x_pos = np.arange(len(property_names))
    bars = ax2.bar(x_pos, property_means, yerr=property_stds, capsize=5,
                   color=['C4', 'C5'], alpha=0.7)
    ax2.set_xlabel('Properties', fontsize=12)
    ax2.set_ylabel('Average Value', fontsize=12)
    ax2.set_title('Average Mechanical Properties', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(property_names)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, mean_val in zip(bars, property_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + property_stds[0]*0.1,
                f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')

    # Bottom-left: Model performance
    ax3 = axes[1, 0]
    x_pos = np.arange(len(property_names))
    bars = ax3.bar(x_pos, r2_scores, color=['C6', 'C7'], alpha=0.7)
    ax3.set_xlabel('Properties', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Model Performance (R² Scores)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(property_names)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, r2_val in zip(bars, r2_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{r2_val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Bottom-right: Data summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    Dataset Summary:
    • Total samples: {n_samples}
    • Element features: {X_elements_with_Fe.shape[1]}
    • Compound features: {X_compounds.shape[1]}
    • Target properties: {Y.shape[1]}

    Model Performance:
    • Elastic Modulus R²: {r2_scores[0]:.4f}
    • Yield Strength R²: {r2_scores[1]:.4f}
    • Elastic Modulus MSE: {mse_scores[0]:.2e}
    • Yield Strength MSE: {mse_scores[1]:.2e}

    Key Insights:
    • Average Co content: {element_means[0]:.3f} ± {element_stds[0]:.3f}
    • Average Mo content: {element_means[1]:.3f} ± {element_stds[1]:.3f}
    • Average Ti content: {element_means[2]:.3f} ± {element_stds[2]:.3f}
    • Average Fe content: {element_means[3]:.3f} ± {element_stds[3]:.3f}
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/summary_report.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Summary report created successfully!")
