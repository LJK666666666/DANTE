"""
Configuration Module for DANTE Alloy Design

This module contains configuration parameters and settings for the
DANTE alloy design optimization framework.

Author: DANTE Team
Date: 2024
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "model_weights"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    'data_file': 'data.csv',
    'element_columns': ['Co', 'Mo', 'Ti'],
    'compound_columns': ['Ni3Al', 'Ni3Ti', 'Ni3V', 'NiTi', 'NiTi2'],
    'property_columns': ['Elastic_Modulus', 'Yield_Strength'],
    'test_size': 0.2,
    'random_state': 42,
    'normalize_data': True
}

# Neural network configuration
NEURAL_CONFIG = {
    'phase_composition_model': {
        'input_dims': 4,  # Co, Mo, Ti, Fe
        'output_dims': 5,  # 5 compounds
        'n_folds': 5,
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'early_stopping_patience': 50,
        'lr_reduction_patience': 20,
        'lr_reduction_factor': 0.7,
        'min_lr': 1e-8
    },
    'dual_network_model': {
        'input_dims': 4,  # Co, Mo, Ti, Fe
        'output_dims': 2,  # Elastic modulus, Yield strength
        'n_folds': 5,
        'batch_size': 16,
        'epochs': 300,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'dropout_rate': 0.2,
        'early_stopping_patience': 30,
        'lr_reduction_patience': 15,
        'lr_reduction_factor': 0.8,
        'min_lr': 1e-8
    }
}

# Optimization configuration
OPTIMIZATION_CONFIG = {
    'dante': {
        'max_iterations': 50,
        'initial_samples': 20,
        'acquisition_function': 'ei',  # Expected Improvement
        'batch_size': 5,
        'tree_max_depth': 5,
        'tree_branching_factor': 3,
        'noise_level': 0.01,
        'scaling_factor': 100.0
    },
    'search_space': {
        'Co_range': [0.0, 0.3],  # Will be updated based on data
        'Mo_range': [0.0, 0.2],  # Will be updated based on data
        'Ti_range': [0.0, 0.15], # Will be updated based on data
        'constraint_tolerance': 1e-6
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'ggplot',
    'color_palette': 'husl',
    'save_format': 'png',
    'font_size': 12,
    'title_font_size': 14,
    'legend_font_size': 10,
    'grid_alpha': 0.3,
    'scatter_alpha': 0.7,
    'scatter_size': 50
}

# Model saving configuration
MODEL_CONFIG = {
    'save_models': True,
    'save_scalers': True,
    'save_history': True,
    'model_format': 'h5',
    'overwrite_existing': False,
    'backup_existing': True
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': True,
    'log_file': 'alloy_design.log',
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'monitor_memory': True,
    'monitor_time': True,
    'profile_code': False,
    'save_metrics': True,
    'metrics_file': 'performance_metrics.json'
}

# Validation configuration
VALIDATION_CONFIG = {
    'cross_validation_folds': 5,
    'validation_metrics': ['mse', 'r2', 'mae'],
    'significance_level': 0.05,
    'bootstrap_samples': 1000,
    'confidence_interval': 0.95
}

# Export configuration
EXPORT_CONFIG = {
    'export_results': True,
    'export_format': 'json',
    'export_models': True,
    'export_figures': True,
    'create_report': True,
    'report_format': 'html'
}

# Environment configuration
ENV_CONFIG = {
    'use_gpu': True,
    'gpu_memory_growth': True,
    'mixed_precision': False,
    'parallel_processing': True,
    'n_jobs': -1,  # Use all available cores
    'random_seed': 42
}

# Advanced settings
ADVANCED_CONFIG = {
    'data_augmentation': {
        'enabled': True,
        'noise_factor': 0.02,
        'augmentation_ratio': 0.1
    },
    'ensemble_methods': {
        'enabled': True,
        'n_models': 5,
        'voting_method': 'average',
        'diversity_threshold': 0.1
    },
    'hyperparameter_tuning': {
        'enabled': False,
        'method': 'bayesian',
        'n_trials': 100,
        'timeout': 3600  # 1 hour
    },
    'uncertainty_quantification': {
        'enabled': True,
        'method': 'ensemble',
        'confidence_level': 0.95
    }
}

# Quality control
QUALITY_CONFIG = {
    'data_quality_checks': True,
    'model_validation_checks': True,
    'result_sanity_checks': True,
    'outlier_detection': True,
    'outlier_threshold': 3.0,  # Standard deviations
    'missing_value_handling': 'interpolate'
}

# Integration settings
INTEGRATION_CONFIG = {
    'dante_integration': True,
    'external_tools': [],
    'api_endpoints': {},
    'database_connection': None
}


def get_config(section=None):
    """
    Get configuration parameters.
    
    Args:
        section (str): Configuration section name. If None, returns all configs.
        
    Returns:
        dict: Configuration parameters
    """
    all_configs = {
        'data': DATA_CONFIG,
        'neural': NEURAL_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'model': MODEL_CONFIG,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'validation': VALIDATION_CONFIG,
        'export': EXPORT_CONFIG,
        'environment': ENV_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'quality': QUALITY_CONFIG,
        'integration': INTEGRATION_CONFIG
    }
    
    if section is None:
        return all_configs
    elif section in all_configs:
        return all_configs[section]
    else:
        raise ValueError(f"Unknown configuration section: {section}")


def update_config(section, key, value):
    """
    Update a configuration parameter.
    
    Args:
        section (str): Configuration section name
        key (str): Parameter key
        value: New parameter value
    """
    config = get_config(section)
    if key in config:
        config[key] = value
        print(f"Updated {section}.{key} = {value}")
    else:
        raise ValueError(f"Unknown configuration key: {section}.{key}")


def validate_config():
    """Validate configuration parameters."""
    print("Validating configuration...")
    
    # Check required directories
    for directory in [DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
        if not directory.exists():
            print(f"Warning: Directory {directory} does not exist. Creating...")
            directory.mkdir(parents=True, exist_ok=True)
    
    # Validate neural network configuration
    neural_config = get_config('neural')
    for model_name, model_config in neural_config.items():
        if model_config['input_dims'] <= 0:
            raise ValueError(f"Invalid input_dims for {model_name}")
        if model_config['output_dims'] <= 0:
            raise ValueError(f"Invalid output_dims for {model_name}")
        if model_config['learning_rate'] <= 0:
            raise ValueError(f"Invalid learning_rate for {model_name}")
    
    # Validate optimization configuration
    opt_config = get_config('optimization')
    if opt_config['dante']['max_iterations'] <= 0:
        raise ValueError("Invalid max_iterations")
    if opt_config['dante']['initial_samples'] <= 0:
        raise ValueError("Invalid initial_samples")
    
    print("Configuration validation completed successfully!")


def print_config_summary():
    """Print a summary of current configuration."""
    print("\n" + "="*60)
    print("DANTE Alloy Design Configuration Summary")
    print("="*60)
    
    configs = get_config()
    for section_name, section_config in configs.items():
        print(f"\n{section_name.upper()} Configuration:")
        if isinstance(section_config, dict):
            for key, value in section_config.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {section_config}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Validate and print configuration when run as script
    validate_config()
    print_config_summary()
