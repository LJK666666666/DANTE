"""
Alloy Objective Function Module for DANTE Optimization

This module defines the objective function for alloy composition optimization,
focusing on maximizing mechanical properties (elastic modulus and yield strength).

Author: DANTE Team
Date: 2024
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Add DANTE module to path
import sys
import os
dante_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if dante_path not in sys.path:
    sys.path.append(dante_path)

# Try to import DANTE modules, use fallback if not available
try:
    from dante.obj_functions import ObjectiveFunction
    DANTE_AVAILABLE = True
    print("✅ DANTE ObjectiveFunction imported successfully!")
except ImportError as e:
    print(f"Warning: DANTE ObjectiveFunction not available: {e}")
    print("Using fallback implementation.")
    DANTE_AVAILABLE = False

    # Fallback ObjectiveFunction class
    class ObjectiveFunction:
        """Fallback ObjectiveFunction class when DANTE is not available."""
        def __init__(self):
            pass


class AlloyObjectiveFunction(ObjectiveFunction):
    """
    Objective function for alloy composition optimization.

    This function takes 3D alloy compositions (Co, Mo, Ti) and returns
    a performance metric based on mechanical properties.

    Note: This implementation follows the original notebook design:
    - Uses percentage units (Co, Mo, Ti values are percentages)
    - Returns negative values (minimization problem)
    - Uses nearest neighbor lookup
    """

    def __init__(self, X_elements, X_elements_with_Fe, Y_combined, dims=3, turn=0.01):
        """
        Initialize the alloy objective function.

        Args:
            X_elements (np.ndarray): 3D element compositions (Co, Mo, Ti) in decimal form
            X_elements_with_Fe (np.ndarray): 4D element compositions including Fe
            Y_combined (np.ndarray): Combined performance metric
            dims (int): Search space dimensions
            turn (float): Turn parameter for DANTE
        """
        self.name = "alloy_optimization"

        # Store data in percentage form to match original notebook
        self.X_data_3d = X_elements  # Already in percentage form from data_loader
        self.X_data_4d = X_elements_with_Fe  # Already in percentage form from data_loader
        self.Y_data = Y_combined

        # Calculate data statistics for scaling
        self.max_val = np.max(Y_combined)
        self.min_val = np.min(Y_combined)

        # Set search boundaries based on training data range (in percentage form)
        co_min, co_max = X_elements[:, 0].min(), X_elements[:, 0].max()
        mo_min, mo_max = X_elements[:, 1].min(), X_elements[:, 1].max()
        ti_min, ti_max = X_elements[:, 2].min(), X_elements[:, 2].max()

        # Add small margins to boundaries (2% margin)
        margin = 0.02
        co_range = co_max - co_min
        mo_range = mo_max - mo_min
        ti_range = ti_max - ti_min

        co_min = max(co_min * 0.9, co_min - margin * co_range)  # Don't go below 90% of original min
        co_max = min(100.0, co_max + margin * co_range)  # Don't exceed 100%
        mo_min = max(mo_min * 0.9, mo_min - margin * mo_range)  # Don't go below 90% of original min
        mo_max = min(100.0, mo_max + margin * mo_range)  # Don't exceed 100%
        ti_min = max(ti_min * 0.9, ti_min - margin * ti_range)  # Don't go below 90% of original min
        ti_max = min(100.0, ti_max + margin * ti_range)  # Don't exceed 100%

        # Ensure Fe content remains reasonable (at least 60%)
        max_sum = 40.0  # Maximum sum of Co + Mo + Ti in percentage form
        current_max_sum = co_max + mo_max + ti_max

        if current_max_sum > max_sum:
            # Scale down upper boundaries proportionally
            scale_factor = max_sum / current_max_sum
            co_max *= scale_factor
            mo_max *= scale_factor
            ti_max *= scale_factor

        # Initialize parent class if available
        if DANTE_AVAILABLE:
            super().__init__(dims=dims, turn=turn)

        # Initialize boundary attributes
        self.lb = np.array([co_min, mo_min, ti_min])
        self.ub = np.array([co_max, mo_max, ti_max])
        self.dims = dims

        print(f"Alloy objective function initialized:")
        print(f"  Search space: {self.dims}D")
        print(f"  Boundaries: {self.lb} to {self.ub}")
        print(f"  Performance range: [{self.min_val:.4f}, {self.max_val:.4f}]")
        print(f"  Performance mean±std: {np.mean(Y_combined):.4f}±{np.std(Y_combined):.4f}")

    def __post_init__(self):
        """Post initialization - ensure boundaries are correctly initialized."""
        # Don't call super().__post_init__() as we have custom boundaries
        if DANTE_AVAILABLE:
            try:
                from dante.utils import Tracker
                self.tracker = Tracker("results_alloy")
            except ImportError:
                pass

    def convert_3d_to_4d(self, x_3d):
        """Convert 3D input (Co, Mo, Ti) to 4D (Co, Mo, Ti, Fe) in percentage form."""
        if x_3d.ndim == 1:
            # Single sample
            fe_content = 100.0 - np.sum(x_3d)  # Percentage form
            return np.append(x_3d, fe_content)
        else:
            # Multiple samples
            fe_content = 100.0 - np.sum(x_3d, axis=1)  # Percentage form
            return np.column_stack([x_3d, fe_content])
    
    def __call__(self, x, apply_scaling=False):
        """
        Evaluate the objective function at point x.

        Args:
            x (np.ndarray): 3D composition point [Co, Mo, Ti] in percentage form
            apply_scaling (bool): Whether to apply scaling to the result

        Returns:
            float: Objective function value (negative for minimization)
        """
        x = self._preprocess(x)

        # Strict boundary check - return penalty if out of bounds
        if np.any(x < self.lb) or np.any(x > self.ub):
            penalty = 1e6  # Large penalty value
            return penalty

        # Validate Fe content reasonableness (in percentage form)
        fe_content = 100.0 - np.sum(x)
        if fe_content < 60.0 or fe_content > 90.0:  # Fe should be 60-90%
            penalty = 1e6
            return penalty

        # Find nearest neighbor in training data (using 3D comparison)
        distances = np.linalg.norm(self.X_data_3d - x, axis=1)
        nearest_idx = np.argmin(distances)

        # Return positive value for maximization (optimization algorithm seeks maximum)
        result = self.Y_data[nearest_idx]

        if apply_scaling:
            return self.scaled(result)
        return result

    def scaled(self, y):
        """Scale original objective value to [0,1] range."""
        # Scale positive values to [0,1] range for maximization
        return (y - self.min_val) / (self.max_val - self.min_val)

    def _preprocess(self, x):
        """Preprocess input to ensure correct format."""
        if hasattr(x, 'ndim'):
            if x.ndim == 0:
                x = np.array([x])
            elif x.ndim > 1:
                x = x.flatten()
        return np.array(x)
    
    def evaluate_batch(self, X, apply_scaling=False):
        """
        Evaluate the objective function for a batch of points.
        
        Args:
            X (np.ndarray): Batch of 3D composition points
            apply_scaling (bool): Whether to apply scaling to the results
            
        Returns:
            np.ndarray: Objective function values
        """
        return self.__call__(X, apply_scaling=apply_scaling)
    
    def get_best_known_point(self):
        """
        Get the best known point from the training data.
        
        Returns:
            tuple: (best_x, best_y) where best_x is 3D composition and best_y is performance
        """
        best_idx = np.argmax(self.Y_combined)
        best_x = self.X_elements[best_idx]
        best_y = self.Y_combined[best_idx]
        
        return best_x, best_y
    
    def get_random_points(self, n_points):
        """
        Generate random points within the search space.
        
        Args:
            n_points (int): Number of random points to generate
            
        Returns:
            np.ndarray: Random points within bounds
        """
        random_points = np.random.uniform(
            low=self.lb, 
            high=self.ub, 
            size=(n_points, self.dims)
        )
        return random_points
    
    def validate_point(self, x):
        """
        Validate if a point is within the search space bounds.
        
        Args:
            x (np.ndarray): Point to validate
            
        Returns:
            bool: True if point is valid, False otherwise
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        return np.all((x >= self.lb) & (x <= self.ub))
    
    def get_search_space_info(self):
        """
        Get information about the search space.
        
        Returns:
            dict: Search space information
        """
        return {
            'dimensions': self.dims,
            'lower_bounds': self.lb.tolist(),
            'upper_bounds': self.ub.tolist(),
            'variable_names': ['Co', 'Mo', 'Ti'],
            'performance_range': [self.y_min, self.y_max],
            'performance_stats': {
                'mean': self.y_mean,
                'std': self.y_std,
                'min': self.y_min,
                'max': self.y_max
            }
        }
    
    def convert_3d_to_4d(self, x_3d):
        """
        Convert 3D composition (Co, Mo, Ti) to 4D (Co, Mo, Ti, Fe).
        
        Args:
            x_3d (np.ndarray): 3D composition
            
        Returns:
            np.ndarray: 4D composition with Fe content
        """
        if x_3d.ndim == 1:
            x_3d = x_3d.reshape(1, -1)
        
        # Calculate Fe content
        Fe_content = 1.0 - x_3d.sum(axis=1, keepdims=True)
        
        # Ensure Fe content is non-negative
        Fe_content = np.maximum(Fe_content, 0.0)
        
        # Combine with other elements
        x_4d = np.column_stack([x_3d, Fe_content])
        
        return x_4d
