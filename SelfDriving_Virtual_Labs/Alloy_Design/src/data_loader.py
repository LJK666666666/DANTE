"""
Data Loading and Preprocessing Module for DANTE Alloy Design

This module handles loading and preprocessing of alloy composition and 
mechanical property data for the DANTE optimization framework.

Author: DANTE Team
Date: 2024
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Data loader and preprocessor for alloy design data."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.scaler_elements = StandardScaler()
        self.scaler_compounds = StandardScaler()
        self.scaler_properties = StandardScaler()
        
    def load_data(self, data_path):
        """
        Load alloy data from CSV file.
        
        Args:
            data_path (str): Path to the data CSV file
            
        Returns:
            pd.DataFrame: Loaded data or None if failed
        """
        try:
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            print(f"Data loaded successfully! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Display basic statistics
            print("\nData overview:")
            print(df.describe())
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def process_data(self, df):
        """
        Process the loaded data into features and targets.

        Args:
            df (pd.DataFrame): Raw data

        Returns:
            tuple: (X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined) or None if failed
        """
        try:
            print("\nProcessing data...")

            # Check if this is the new format with 'sid' column
            if 'sid' in df.columns and 'phase_ratio_dict' in df.columns:
                return self.process_new_format_data(df)

            # Original format processing
            # Extract element compositions (Co, Mo, Ti)
            element_columns = ['Co', 'Mo', 'Ti']
            if not all(col in df.columns for col in element_columns):
                print(f"Missing element columns. Available: {list(df.columns)}")
                return None

            X_elements = df[element_columns].values
            print(f"Element features shape: {X_elements.shape}")

            # Calculate Fe content (assuming Fe + Co + Mo + Ti = 100%)
            # Convert to percentage form to match original notebook
            X_elements = X_elements * 100.0  # Convert to percentage
            Fe_content = 100.0 - X_elements.sum(axis=1)
            X_elements_with_Fe = np.column_stack([X_elements, Fe_content])
            print(f"Element features with Fe shape: {X_elements_with_Fe.shape}")

            # Extract compound compositions
            compound_columns = ['Ni3Al', 'Ni3Ti', 'Ni3V', 'NiTi', 'NiTi2']
            if not all(col in df.columns for col in compound_columns):
                print(f"Missing compound columns. Available: {list(df.columns)}")
                return None

            X_compounds = df[compound_columns].values
            print(f"Compound features shape: {X_compounds.shape}")

            # Extract mechanical properties
            property_columns = ['Elastic_Modulus', 'Yield_Strength']
            if not all(col in df.columns for col in property_columns):
                print(f"Missing property columns. Available: {list(df.columns)}")
                return None

            Y = df[property_columns].values
            print(f"Mechanical properties shape: {Y.shape}")

            # Create combined performance metric (weighted sum)
            # Normalize both properties to [0, 1] range first
            elastic_normalized = (Y[:, 0] - Y[:, 0].min()) / (Y[:, 0].max() - Y[:, 0].min())
            yield_normalized = (Y[:, 1] - Y[:, 1].min()) / (Y[:, 1].max() - Y[:, 1].min())

            # Combined metric: 0.5 * elastic_modulus + 0.5 * yield_strength
            Y_combined = 0.5 * elastic_normalized + 0.5 * yield_normalized
            print(f"Combined performance metric shape: {Y_combined.shape}")

            # Data quality checks
            print("\nData quality checks:")
            print(f"Element compositions sum check: {np.allclose(X_elements_with_Fe.sum(axis=1), 1.0)}")
            print(f"Element range: [{X_elements.min():.3f}, {X_elements.max():.3f}]")
            print(f"Compound range: [{X_compounds.min():.3f}, {X_compounds.max():.3f}]")
            print(f"Elastic modulus range: [{Y[:, 0].min():.3f}, {Y[:, 0].max():.3f}]")
            print(f"Yield strength range: [{Y[:, 1].min():.3f}, {Y[:, 1].max():.3f}]")
            print(f"Combined metric range: [{Y_combined.min():.3f}, {Y_combined.max():.3f}]")

            # Check for missing values
            if np.any(np.isnan(X_elements)) or np.any(np.isnan(X_compounds)) or np.any(np.isnan(Y)):
                print("Warning: Found NaN values in data!")

            return X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined

        except Exception as e:
            print(f"Error processing data: {e}")
            return None

    def process_new_format_data(self, df):
        """
        Process data in the new format with 'sid' and 'phase_ratio_dict' columns.

        Args:
            df (pd.DataFrame): Raw data in new format

        Returns:
            tuple: (X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined) or None if failed
        """
        try:
            print("Processing data in new format...")
            import json
            import re

            # Extract element compositions from 'sid' column
            # Format: Co8.50Mo5.15Ti2.60
            element_data = []
            for sid in df['sid']:
                # Parse element compositions using regex
                co_match = re.search(r'Co([\d.]+)', sid)
                mo_match = re.search(r'Mo([\d.]+)', sid)
                ti_match = re.search(r'Ti([\d.]+)', sid)

                co = float(co_match.group(1)) / 100.0 if co_match else 0.0
                mo = float(mo_match.group(1)) / 100.0 if mo_match else 0.0
                ti = float(ti_match.group(1)) / 100.0 if ti_match else 0.0

                element_data.append([co, mo, ti])

            X_elements = np.array(element_data)
            print(f"Element features shape: {X_elements.shape}")

            # Calculate Fe content (percentage form to match original notebook)
            X_elements = X_elements * 100.0  # Convert to percentage
            Fe_content = 100.0 - X_elements.sum(axis=1)
            X_elements_with_Fe = np.column_stack([X_elements, Fe_content])
            print(f"Element features with Fe shape: {X_elements_with_Fe.shape}")

            # Extract compound compositions from 'phase_ratio_dict' column
            compound_data = []
            compound_names = ['martensite', 'Fe2Mo', 'austenite', 'gamma_phase', 'Ni3Ti']

            for phase_dict_str in df['phase_ratio_dict']:
                # Parse JSON string
                phase_dict = json.loads(phase_dict_str)

                # Extract compound ratios
                compound_ratios = []
                for compound in compound_names:
                    ratio = phase_dict.get(compound, 0.0)
                    compound_ratios.append(ratio)

                compound_data.append(compound_ratios)

            X_compounds = np.array(compound_data)
            print(f"Compound features shape: {X_compounds.shape}")

            # Extract mechanical properties
            Y = np.column_stack([df['elastic'].values, df['yield'].values])
            print(f"Mechanical properties shape: {Y.shape}")

            # Create combined performance metric
            elastic_normalized = (Y[:, 0] - Y[:, 0].min()) / (Y[:, 0].max() - Y[:, 0].min())
            yield_normalized = (Y[:, 1] - Y[:, 1].min()) / (Y[:, 1].max() - Y[:, 1].min())
            Y_combined = 0.5 * elastic_normalized + 0.5 * yield_normalized
            print(f"Combined performance metric shape: {Y_combined.shape}")

            # Data quality checks
            print("\nData quality checks:")
            print(f"Element compositions sum check: {np.allclose(X_elements_with_Fe.sum(axis=1), 1.0)}")
            print(f"Element range: [{X_elements.min():.3f}, {X_elements.max():.3f}]")
            print(f"Compound range: [{X_compounds.min():.3f}, {X_compounds.max():.3f}]")
            print(f"Elastic modulus range: [{Y[:, 0].min():.2e}, {Y[:, 0].max():.2e}]")
            print(f"Yield strength range: [{Y[:, 1].min():.2e}, {Y[:, 1].max():.2e}]")
            print(f"Combined metric range: [{Y_combined.min():.3f}, {Y_combined.max():.3f}]")

            return X_elements, X_elements_with_Fe, X_compounds, Y, Y_combined

        except Exception as e:
            print(f"Error processing new format data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def normalize_data(self, X_elements, X_compounds, Y):
        """
        Normalize the data using StandardScaler.
        
        Args:
            X_elements (np.ndarray): Element features
            X_compounds (np.ndarray): Compound features  
            Y (np.ndarray): Target properties
            
        Returns:
            tuple: Normalized data (X_elements_norm, X_compounds_norm, Y_norm)
        """
        print("\nNormalizing data...")
        
        X_elements_norm = self.scaler_elements.fit_transform(X_elements)
        X_compounds_norm = self.scaler_compounds.fit_transform(X_compounds)
        Y_norm = self.scaler_properties.fit_transform(Y)
        
        print("Data normalization completed!")
        
        return X_elements_norm, X_compounds_norm, Y_norm
    
    def inverse_transform_properties(self, Y_norm):
        """
        Inverse transform normalized properties back to original scale.
        
        Args:
            Y_norm (np.ndarray): Normalized properties
            
        Returns:
            np.ndarray: Original scale properties
        """
        return self.scaler_properties.inverse_transform(Y_norm)
    
    def get_data_statistics(self, X_elements, X_compounds, Y):
        """
        Get comprehensive statistics about the data.
        
        Args:
            X_elements (np.ndarray): Element features
            X_compounds (np.ndarray): Compound features
            Y (np.ndarray): Target properties
            
        Returns:
            dict: Data statistics
        """
        stats = {
            'n_samples': X_elements.shape[0],
            'n_element_features': X_elements.shape[1],
            'n_compound_features': X_compounds.shape[1],
            'n_properties': Y.shape[1],
            'element_means': X_elements.mean(axis=0),
            'element_stds': X_elements.std(axis=0),
            'compound_means': X_compounds.mean(axis=0),
            'compound_stds': X_compounds.std(axis=0),
            'property_means': Y.mean(axis=0),
            'property_stds': Y.std(axis=0),
            'element_ranges': [(X_elements[:, i].min(), X_elements[:, i].max()) 
                              for i in range(X_elements.shape[1])],
            'compound_ranges': [(X_compounds[:, i].min(), X_compounds[:, i].max()) 
                               for i in range(X_compounds.shape[1])],
            'property_ranges': [(Y[:, i].min(), Y[:, i].max()) 
                               for i in range(Y.shape[1])]
        }
        
        return stats


class LogNormalizedDataLoader(DataLoader):
    """
    Enhanced data loader with logarithmic transformation for mechanical properties.

    This class extends the base DataLoader to apply logarithmic transformation
    to Young's modulus and yield strength before normalization, which can help
    with neural network training when dealing with properties that span multiple
    orders of magnitude.
    """

    def __init__(self):
        """Initialize the log-normalized data loader."""
        super().__init__()
        self.log_scaler_properties = StandardScaler()

    def log_transform_properties(self, Y):
        """
        Apply logarithmic transformation to mechanical properties.

        Args:
            Y (np.ndarray): Original mechanical properties [elastic_modulus, yield_strength]

        Returns:
            np.ndarray: Log-transformed properties
        """
        # Ensure positive values (add small epsilon if needed)
        epsilon = 1e-10
        Y_safe = np.maximum(Y, epsilon)

        # Take logarithm
        Y_log = np.log(Y_safe)

        print(f"Log transformation applied:")
        print(f"  Original range: [{Y.min():.2e}, {Y.max():.2e}]")
        print(f"  Log-transformed range: [{Y_log.min():.3f}, {Y_log.max():.3f}]")

        return Y_log

    def inverse_log_transform_properties(self, Y_log_normalized):
        """
        Inverse transform: denormalize then exponentiate.

        Args:
            Y_log_normalized (np.ndarray): Log-transformed and normalized properties

        Returns:
            np.ndarray: Original scale properties
        """
        # First inverse normalize
        Y_log = self.log_scaler_properties.inverse_transform(Y_log_normalized)

        # Then exponentiate
        Y_properties = np.exp(Y_log)

        return Y_properties

    def process_data_with_log_transform(self, df):
        """
        Process data with logarithmic transformation of properties.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            tuple: (X_elements, X_elements_with_Fe, X_compounds, Y_original, Y_log_normalized, Y_combined)
        """
        # First process data normally
        processed_data = self.process_data(df)

        if processed_data is None:
            return None

        X_elements, X_elements_with_Fe, X_compounds, Y_original, Y_combined = processed_data

        # Apply log transformation to properties
        print("\nApplying logarithmic transformation to mechanical properties...")
        Y_log = self.log_transform_properties(Y_original)

        # Normalize log-transformed properties
        Y_log_normalized = self.log_scaler_properties.fit_transform(Y_log)

        print(f"Log-normalized properties range: [{Y_log_normalized.min():.3f}, {Y_log_normalized.max():.3f}]")

        return X_elements, X_elements_with_Fe, X_compounds, Y_original, Y_log_normalized, Y_combined

    def normalize_log_data(self, X_elements, X_compounds, Y_log):
        """
        Normalize the data including log-transformed properties.

        Args:
            X_elements (np.ndarray): Element features
            X_compounds (np.ndarray): Compound features
            Y_log (np.ndarray): Log-transformed target properties

        Returns:
            tuple: Normalized data (X_elements_norm, X_compounds_norm, Y_log_norm)
        """
        print("\nNormalizing data with log-transformed properties...")

        X_elements_norm = self.scaler_elements.fit_transform(X_elements)
        X_compounds_norm = self.scaler_compounds.fit_transform(X_compounds)
        Y_log_norm = self.log_scaler_properties.fit_transform(Y_log)

        print("Log-normalized data normalization completed!")

        return X_elements_norm, X_compounds_norm, Y_log_norm
