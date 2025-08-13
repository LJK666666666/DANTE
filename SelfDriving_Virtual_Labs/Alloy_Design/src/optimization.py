"""
DANTE Optimization Module for Alloy Design

This module implements the DANTE optimization algorithm for alloy composition
optimization, integrating deep active learning, tree exploration, and neural
surrogate models.

Author: DANTE Team
Date: 2024
"""

import numpy as np
import time
from pathlib import Path

# Add DANTE module to path
import sys
import os
dante_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if dante_path not in sys.path:
    sys.path.append(dante_path)

# Import DANTE modules (selectively)
DANTE_AVAILABLE = False
try:
    # Try to import basic DANTE utilities first
    from dante.utils import generate_initial_samples, Tracker
    print("✅ DANTE utils imported successfully!")

    # Try to import optimization modules
    try:
        from dante.deep_active_learning import DeepActiveLearning
        from dante.tree_exploration import TreeExploration
        DANTE_AVAILABLE = True
        print("✅ DANTE optimization modules imported successfully!")
    except ImportError as e:
        print(f"⚠️ DANTE optimization modules not available: {e}")
        print("Using basic DANTE utilities only.")

except ImportError as e:
    print(f"Warning: Could not import DANTE modules: {e}")
    print("DANTE optimization will not be available.")
    DANTE_AVAILABLE = False


def run_dante_optimization(objective_function, surrogate_model, X_elements, 
                          max_iterations=50, initial_samples=20, verbose=True):
    """
    Run DANTE optimization for alloy composition optimization.
    
    Args:
        objective_function: Alloy objective function to optimize
        surrogate_model: Trained neural surrogate model
        X_elements (np.ndarray): Training element compositions
        max_iterations (int): Maximum number of optimization iterations
        initial_samples (int): Number of initial samples
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Optimization results including best points, values, and convergence history
    """
    print(f"\n{'='*60}")
    print("DANTE Alloy Composition Optimization")
    print(f"{'='*60}")
    
    try:
        # Initialize DANTE components
        print("Initializing DANTE components...")
        
        # Create tracker for monitoring optimization progress
        tracker = Tracker("alloy_optimization")
        
        # Generate initial samples within the search space
        print(f"Generating {initial_samples} initial samples...")
        initial_X = generate_initial_samples(
            objective_function.lb, 
            objective_function.ub, 
            initial_samples
        )
        
        # Evaluate initial samples
        print("Evaluating initial samples...")
        initial_Y = np.array([objective_function(x) for x in initial_X])
        
        # Initialize deep active learning component
        print("Initializing Deep Active Learning...")
        dal = DeepActiveLearning(
            surrogate_model=surrogate_model,
            acquisition_function='ei',  # Expected Improvement
            batch_size=5
        )
        
        # Initialize tree exploration component
        print("Initializing Tree Exploration...")
        tree_explorer = TreeExploration(
            objective_function=objective_function,
            max_depth=5,
            branching_factor=3
        )
        
        # Optimization loop
        print(f"\nStarting optimization loop (max {max_iterations} iterations)...")
        
        # Storage for results
        all_X = list(initial_X)
        all_Y = list(initial_Y)
        best_values = []
        best_points = []
        convergence_history = []
        
        current_best_idx = np.argmax(initial_Y)
        current_best_X = initial_X[current_best_idx]
        current_best_Y = initial_Y[current_best_idx]
        
        print(f"Initial best value: {current_best_Y:.6f}")
        print(f"Initial best point: {current_best_X}")
        
        for iteration in range(max_iterations):
            iteration_start_time = time.time()
            
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # Update surrogate model with current data
            X_current = np.array(all_X)
            Y_current = np.array(all_Y)
            
            # Deep Active Learning: suggest next points to evaluate
            if hasattr(dal, 'suggest_next_points'):
                candidate_points = dal.suggest_next_points(X_current, Y_current, n_points=3)
            else:
                # Fallback: random sampling with some bias towards promising regions
                candidate_points = generate_candidate_points(
                    objective_function, current_best_X, n_points=3
                )
            
            # Tree Exploration: explore promising regions
            if hasattr(tree_explorer, 'explore'):
                tree_points = tree_explorer.explore(current_best_X, depth=2)
                if len(tree_points) > 0:
                    candidate_points = np.vstack([candidate_points, tree_points[:2]])
            
            # Evaluate candidate points
            new_Y = []
            for point in candidate_points:
                # Ensure point is within bounds
                point_clipped = np.clip(point, objective_function.lb, objective_function.ub)
                y_val = objective_function(point_clipped)
                new_Y.append(y_val)
                all_X.append(point_clipped)
                all_Y.append(y_val)
            
            new_Y = np.array(new_Y)
            
            # Update best solution
            current_iteration_best_idx = np.argmax(new_Y)
            current_iteration_best_Y = new_Y[current_iteration_best_idx]
            
            if current_iteration_best_Y > current_best_Y:
                current_best_Y = current_iteration_best_Y
                current_best_X = candidate_points[current_iteration_best_idx]
                if verbose:
                    print(f"New best found! Value: {current_best_Y:.6f}")
                    print(f"Best point: {current_best_X}")
            
            # Record progress
            best_values.append(current_best_Y)
            best_points.append(current_best_X.copy())
            convergence_history.append(current_best_Y)
            
            # Update tracker
            tracker.update(iteration + 1, current_best_Y, current_best_X)
            
            iteration_time = time.time() - iteration_start_time
            
            if verbose:
                print(f"Iteration {iteration + 1} completed in {iteration_time:.2f}s")
                print(f"Current best: {current_best_Y:.6f}")
                print(f"Evaluated {len(candidate_points)} new points")
        
        # Final results
        print(f"\n{'='*60}")
        print("DANTE Optimization Completed!")
        print(f"{'='*60}")
        print(f"Total iterations: {max_iterations}")
        print(f"Total evaluations: {len(all_X)}")
        print(f"Final best value: {current_best_Y:.6f}")
        print(f"Final best point: {current_best_X}")
        print(f"Improvement: {((current_best_Y - initial_Y.max()) / initial_Y.max() * 100):.2f}%")
        
        # Prepare results dictionary
        results = {
            'best_value': current_best_Y,
            'best_point': current_best_X,
            'best_values': best_values,
            'best_points': best_points,
            'convergence_history': convergence_history,
            'all_X': np.array(all_X),
            'all_Y': np.array(all_Y),
            'total_evaluations': len(all_X),
            'initial_best': initial_Y.max(),
            'final_best': current_best_Y,
            'improvement_percent': ((current_best_Y - initial_Y.max()) / initial_Y.max() * 100)
        }
        
        return results
        
    except Exception as e:
        print(f"Error during DANTE optimization: {e}")
        print("Falling back to simple random optimization...")
        return run_simple_optimization(objective_function, X_elements, max_iterations, verbose)


def generate_candidate_points(objective_function, best_point, n_points=3, noise_scale=0.1):
    """
    Generate candidate points for evaluation.
    
    Args:
        objective_function: Objective function with bounds
        best_point (np.ndarray): Current best point
        n_points (int): Number of points to generate
        noise_scale (float): Scale of noise to add to best point
        
    Returns:
        np.ndarray: Candidate points
    """
    candidates = []
    
    # Generate points around the current best
    for _ in range(n_points):
        # Add noise to best point
        noise = np.random.normal(0, noise_scale, size=best_point.shape)
        candidate = best_point + noise
        
        # Clip to bounds
        candidate = np.clip(candidate, objective_function.lb, objective_function.ub)
        candidates.append(candidate)
    
    return np.array(candidates)


def run_simple_optimization(objective_function, X_elements, max_iterations=50, verbose=True):
    """
    Run simple random optimization as fallback.
    
    Args:
        objective_function: Objective function to optimize
        X_elements (np.ndarray): Training element compositions
        max_iterations (int): Maximum number of iterations
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Optimization results
    """
    print("Running simple random optimization...")
    
    # Initialize with random points
    best_value = -np.inf
    best_point = None
    convergence_history = []
    all_X = []
    all_Y = []
    
    for iteration in range(max_iterations):
        # Generate random point
        random_point = np.random.uniform(
            objective_function.lb, 
            objective_function.ub, 
            size=objective_function.dims
        )
        
        # Evaluate point
        value = objective_function(random_point)
        
        all_X.append(random_point)
        all_Y.append(value)
        
        # Update best
        if value > best_value:
            best_value = value
            best_point = random_point.copy()
            if verbose:
                print(f"Iteration {iteration + 1}: New best = {best_value:.6f}")
        
        convergence_history.append(best_value)
    
    results = {
        'best_value': best_value,
        'best_point': best_point,
        'convergence_history': convergence_history,
        'all_X': np.array(all_X),
        'all_Y': np.array(all_Y),
        'total_evaluations': len(all_X),
        'optimization_type': 'simple_random'
    }
    
    print(f"Simple optimization completed. Best value: {best_value:.6f}")
    
    return results


def analyze_optimization_results(results, objective_function):
    """
    Analyze and summarize optimization results.
    
    Args:
        results (dict): Optimization results
        objective_function: Objective function used
        
    Returns:
        dict: Analysis summary
    """
    print("\nAnalyzing optimization results...")
    
    analysis = {
        'convergence_rate': calculate_convergence_rate(results['convergence_history']),
        'exploration_diversity': calculate_exploration_diversity(results['all_X']),
        'final_improvement': results.get('improvement_percent', 0),
        'search_efficiency': len(results['all_X']) / results['total_evaluations'],
    }
    
    # Best point analysis
    best_point = results['best_point']
    print(f"\nBest composition found:")
    print(f"  Co: {best_point[0]:.4f}")
    print(f"  Mo: {best_point[1]:.4f}")
    print(f"  Ti: {best_point[2]:.4f}")
    print(f"  Fe: {1.0 - best_point.sum():.4f}")
    print(f"  Objective value: {results['best_value']:.6f}")
    
    return analysis


def calculate_convergence_rate(convergence_history):
    """Calculate the convergence rate of optimization."""
    if len(convergence_history) < 2:
        return 0.0
    
    # Calculate improvement rate
    initial_value = convergence_history[0]
    final_value = convergence_history[-1]
    
    if initial_value == 0:
        return float('inf') if final_value > 0 else 0.0
    
    return (final_value - initial_value) / initial_value


def calculate_exploration_diversity(all_X):
    """Calculate the diversity of explored points."""
    if len(all_X) < 2:
        return 0.0
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(all_X)):
        for j in range(i + 1, len(all_X)):
            dist = np.linalg.norm(all_X[i] - all_X[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0
