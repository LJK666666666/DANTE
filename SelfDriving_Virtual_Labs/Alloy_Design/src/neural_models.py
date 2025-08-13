"""
Neural Network Models for DANTE Alloy Design

This module contains improved neural network surrogate models for:
1. Phase composition prediction (element -> compound ratios)
2. Dual network for direct property prediction (element -> properties)

Author: DANTE Team
Date: 2024
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import pickle
import gc

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    # Test basic functionality
    test_model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
    TF_AVAILABLE = True
    print("TensorFlow/Keras imported successfully!")
except (ImportError, Exception) as e:
    print(f"Warning: TensorFlow/Keras not available: {e}")
    print("Neural network models will use fallback implementations.")
    TF_AVAILABLE = False
    # Create dummy classes
    class keras:
        class Model:
            def __init__(self, *args, **kwargs):
                pass
        class Input:
            def __init__(self, *args, **kwargs):
                pass
        class layers:
            class Dense:
                def __init__(self, *args, **kwargs):
                    pass

# Add DANTE module to path
import sys
import os
dante_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if dante_path not in sys.path:
    sys.path.append(dante_path)

# Try to import DANTE modules
try:
    from dante.neural_surrogate import SurrogateModel
    DANTE_AVAILABLE = True
except ImportError:
    print("Warning: DANTE modules not available. Using fallback implementation.")
    DANTE_AVAILABLE = False

    # Fallback SurrogateModel class
    class SurrogateModel:
        """Fallback SurrogateModel class when DANTE is not available."""
        def __init__(self, input_dims=4, **kwargs):
            self.input_dims = input_dims


def clear_training_cache():
    """Clear training process memory cache."""
    # Clear Python garbage collection
    gc.collect()
    
    # Clear TensorFlow/Keras backend cache
    try:
        tf.keras.backend.clear_session()
    except:
        pass


class ImprovedPhaseCompositionSurrogateModel(SurrogateModel):
    """
    Improved phase composition prediction surrogate model:
    1. Improved loss function and constraints
    2. Enhanced network architecture  
    3. Better data preprocessing
    4. Optimized training strategy
    
    Specifically optimized for phase composition prediction
    """
    
    def __init__(self, input_dims=4, output_dims=5, n_folds=5, **kwargs):
        super().__init__(input_dims=input_dims, **kwargs)
        self.output_dims = output_dims  # 5D compound output
        self.n_folds = n_folds

        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            print("Warning: TensorFlow not available. Neural network functionality will be limited.")

        # Initialize scalers
        self.element_scaler = StandardScaler()
        self.compound_scaler = StandardScaler()

        # Model storage
        self.fold_models = []
        self.final_model = None
        self.ensemble_model = None

        # Training results
        self.cv_scores = []
        self.training_history = []

        # Weight file paths - use original model_weights directory
        self.weights_dir = Path("../model_weights")  # Go up one level to find original weights
        self.weights_dir.mkdir(exist_ok=True)
        self.model_name = "improved_phase_composition"
        self.final_weights_path = self.weights_dir / f"{self.model_name}_final.weights.h5"
        self.scaler_path = self.weights_dir / f"{self.model_name}_scalers.pkl"
        
    def save_scalers(self):
        """Save scalers."""
        scalers = {
            'element_scaler': self.element_scaler,
            'compound_scaler': self.compound_scaler
        }
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to: {self.scaler_path}")
    
    def load_scalers(self):
        """Load scalers."""
        if self.scaler_path.exists():
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            self.element_scaler = scalers['element_scaler']
            self.compound_scaler = scalers['compound_scaler']
            print(f"Scalers loaded from {self.scaler_path}")
            return True
        return False
        
    def improved_phase_composition_loss(self, y_true, y_pred):
        """
        Improved phase composition loss function - optimized for compound ratio prediction
        Combines multiple constraints and regularization terms
        """
        # Basic MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Probability distribution constraint - sum to 1
        sum_constraint = tf.reduce_mean(tf.square(tf.reduce_sum(y_pred, axis=1) - 1.0))
        
        # Non-negative constraint - prevent negative values
        negative_penalty = tf.reduce_mean(tf.maximum(0.0, -y_pred))
        
        # Smoothness constraint - prevent overly extreme predictions
        smoothness_penalty = tf.reduce_mean(tf.square(tf.nn.moments(y_pred, axes=[1])[1]))
        
        # KL divergence loss - maintain probability distribution characteristics
        epsilon = 1e-7
        y_true_normalized = y_true / (tf.reduce_sum(y_true, axis=1, keepdims=True) + epsilon)
        y_pred_normalized = y_pred / (tf.reduce_sum(y_pred, axis=1, keepdims=True) + epsilon)
        
        kl_loss = tf.reduce_mean(tf.reduce_sum(
            y_true_normalized * tf.math.log((y_true_normalized + epsilon) / (y_pred_normalized + epsilon)), 
            axis=1))
        
        # Weighted combined loss
        total_loss = (
            1.0 * mse_loss +           # Main loss
            0.5 * sum_constraint +     # Sum constraint
            0.3 * negative_penalty +   # Non-negative constraint  
            0.1 * smoothness_penalty + # Smoothness
            0.2 * kl_loss             # Distribution constraint
        )
        
        return total_loss
    
    def create_advanced_residual_block(self, x, units, dropout_rate=0.3, name_prefix=""):
        """Create improved residual block with more regularization."""
        shortcut = x
        
        # First layer - add layer normalization
        x = layers.Dense(units, activation=None, name=f'{name_prefix}_res_dense1')(x)
        x = layers.LayerNormalization(name=f'{name_prefix}_res_ln1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_res_relu1')(x)
        x = layers.Dropout(dropout_rate, name=f'{name_prefix}_res_dropout1')(x)
        
        # Second layer
        x = layers.Dense(units, activation=None, name=f'{name_prefix}_res_dense2')(x)
        x = layers.LayerNormalization(name=f'{name_prefix}_res_ln2')(x)
        
        # Dimension matching skip connection
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units, activation=None, name=f'{name_prefix}_res_shortcut')(shortcut)
            shortcut = layers.LayerNormalization(name=f'{name_prefix}_res_shortcut_ln')(shortcut)
        
        # Residual connection
        x = layers.Add(name=f'{name_prefix}_res_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name_prefix}_res_relu2')(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f'{name_prefix}_res_dropout2')(x)  # Lower final dropout
        
        return x
    
    def create_attention_layer(self, x, units, name_prefix=""):
        """Create simple attention mechanism."""
        # Calculate attention weights
        attention_weights = layers.Dense(units, activation='tanh', name=f'{name_prefix}_attention_tanh')(x)
        attention_weights = layers.Dense(units, activation='softmax', name=f'{name_prefix}_attention_softmax')(attention_weights)
        
        # Apply attention
        attended = layers.Multiply(name=f'{name_prefix}_attention_apply')([x, attention_weights])
        
        return attended
    
    def create_improved_phase_model(self, model_name="improved_phase_model"):
        """Create improved phase composition prediction network."""
        inputs = keras.Input(shape=(self.input_dims,), name=f'{model_name}_input')
        
        # Enhanced input layer - use deeper feature extraction
        x = layers.Dense(512, activation='relu', name=f'{model_name}_dense1')(inputs)
        x = layers.LayerNormalization(name=f'{model_name}_ln1')(x)
        x = layers.Dropout(0.4, name=f'{model_name}_dropout1')(x)
        
        # Feature extraction layers
        x = layers.Dense(384, activation='relu', name=f'{model_name}_dense2')(x)
        x = layers.LayerNormalization(name=f'{model_name}_ln2')(x)
        x = layers.Dropout(0.3, name=f'{model_name}_dropout2')(x)
        
        x = layers.Dense(256, activation='relu', name=f'{model_name}_dense3')(x)
        x = layers.LayerNormalization(name=f'{model_name}_ln3')(x)
        x = layers.Dropout(0.3, name=f'{model_name}_dropout3')(x)
        
        # Multiple improved residual blocks
        x = self.create_advanced_residual_block(x, 256, 0.25, f'{model_name}_block1')
        x = self.create_advanced_residual_block(x, 192, 0.25, f'{model_name}_block2')
        x = self.create_advanced_residual_block(x, 128, 0.2, f'{model_name}_block3')
        x = self.create_advanced_residual_block(x, 96, 0.15, f'{model_name}_block4')
        
        # Attention mechanism
        x = self.create_attention_layer(x, x.shape[-1], f'{model_name}_attention')
        
        # Final feature extraction
        x = layers.Dense(128, activation='relu', name=f'{model_name}_dense4')(x)
        x = layers.LayerNormalization(name=f'{model_name}_ln4')(x)
        x = layers.Dropout(0.1, name=f'{model_name}_dropout4')(x)
        
        x = layers.Dense(64, activation='relu', name=f'{model_name}_dense5')(x)
        x = layers.LayerNormalization(name=f'{model_name}_ln5')(x)
        
        # Special output layer - ensure probability distribution
        # First output to higher dimension, then compress
        x = layers.Dense(32, activation='relu', name=f'{model_name}_pre_output')(x)
        x = layers.LayerNormalization(name=f'{model_name}_pre_output_ln')(x)
        
        # Final output - use softmax to ensure probability distribution
        outputs = layers.Dense(self.output_dims, activation='softmax', name=f'{model_name}_output')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
        
        # Use optimized compilation configuration
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.001,  # Moderate learning rate
                weight_decay=0.01,    # Add weight decay
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                clipnorm=1.0          # Gradient clipping
            ),
            loss=self.improved_phase_composition_loss,
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_model(self) -> keras.Model:
        """Required implementation of abstract method from SurrogateModel."""
        return self.create_improved_phase_model("phase_composition_model")

    def advanced_data_preprocessing(self, x_elements, x_compounds):
        """Improved data preprocessing."""
        # Standardize element features
        x_elements_scaled = self.element_scaler.fit_transform(x_elements)

        # Improved compound data preprocessing
        # 1. Normalize to probability distribution
        x_compounds_normalized = x_compounds / (np.sum(x_compounds, axis=1, keepdims=True) + 1e-8)

        # 2. Handle outliers
        x_compounds_clipped = np.clip(x_compounds_normalized, 1e-6, 1.0)

        # 3. Re-normalize
        x_compounds_final = x_compounds_clipped / np.sum(x_compounds_clipped, axis=1, keepdims=True)

        # Standardize (but maintain probability distribution characteristics)
        x_compounds_scaled = self.compound_scaler.fit_transform(x_compounds_final)

        print(f"Data preprocessing statistics:")
        print(f"  Element feature range: [{x_elements_scaled.min():.3f}, {x_elements_scaled.max():.3f}]")
        print(f"  Compound feature range: [{x_compounds_scaled.min():.3f}, {x_compounds_scaled.max():.3f}]")
        print(f"  Compound probability sum check: {np.mean(np.sum(x_compounds_final, axis=1)):.6f} (should be close to 1.0)")

        return x_elements_scaled, x_compounds_scaled, x_compounds_final

    def create_improved_callbacks(self, fold_idx=None):
        """Create improved callback functions."""
        callbacks = []

        # Early stopping - more patient settings
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,  # Increase patience
            restore_best_weights=True,
            verbose=0,
            min_delta=1e-6
        )
        callbacks.append(early_stopping)

        # Learning rate scheduler - more fine-grained adjustment
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,      # More gentle decay
            patience=20,     # More patience
            min_lr=1e-8,
            verbose=0,
            cooldown=5
        )
        callbacks.append(lr_scheduler)

        return callbacks

    def __call__(self, x_elements, x_compounds, verbose=1):
        """Train improved phase composition prediction model."""
        print("\n=== Starting improved phase composition prediction model training ===")

        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            print("TensorFlow not available. Creating simple fallback model...")
            return self.create_fallback_model(x_elements, x_compounds)

        from sklearn.metrics import mean_squared_error, r2_score

        # Try to load existing model weights
        if self.final_weights_path.exists() and self.load_scalers():
            print(f"Found saved model weights: {self.final_weights_path}")
            print("Loading pre-trained model...")

            # Data preprocessing (using loaded scalers)
            x_elements_scaled = self.element_scaler.transform(x_elements)
            x_compounds_normalized = x_compounds / (np.sum(x_compounds, axis=1, keepdims=True) + 1e-8)

            # Create model and load weights
            final_model = self.create_improved_phase_model("final_improved_phase_model")
            try:
                final_model.load_weights(self.final_weights_path)
                print("Model weights loaded successfully!")

                # Validate loaded model performance
                final_pred = final_model.predict(x_elements_scaled, verbose=0)
                final_mse = mean_squared_error(x_compounds_normalized, final_pred)
                final_r2 = r2_score(x_compounds_normalized.flatten(), final_pred.flatten())

                print(f"Loaded model performance validation:")
                print(f"  MSE: {final_mse:.6f}")
                print(f"  R²: {final_r2:.6f}")

                # Create simple ensemble model (only contains final model)
                class SimpleEnsembleModel:
                    def __init__(self, model, parent):
                        self.model = model
                        self.parent = parent

                    def predict(self, x, verbose=0):
                        if x.ndim == 1:
                            x = x.reshape(1, -1)
                        x_elements_scaled = self.parent.element_scaler.transform(x)
                        predictions = self.model.predict(x_elements_scaled, verbose=verbose)
                        predictions = np.clip(predictions, 1e-6, 1.0)
                        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
                        return predictions

                self.final_model = final_model
                self.ensemble_model = SimpleEnsembleModel(final_model, self)

                print("Completed initialization using pre-trained model!")
                return self.ensemble_model

            except Exception as e:
                print(f"Failed to load model weights: {e}")
                print("Will retrain model...")

        # If no pre-trained model or loading failed, perform full training
        print("Starting full model training process...")

        # Perform improved cross-validation
        cv_scores, fold_models = self.perform_improved_cross_validation(
            x_elements, x_compounds, verbose)

        # Train final model on full dataset
        print("\nTraining final improved model on full dataset...")

        x_elements_scaled, x_compounds_scaled, x_compounds_normalized = self.advanced_data_preprocessing(
            x_elements, x_compounds)

        final_model = self.create_improved_phase_model("final_improved_phase_model")
        callbacks = self.create_improved_callbacks()

        # Final training
        final_history = final_model.fit(
            x_elements_scaled, x_compounds_normalized,
            batch_size=16,
            epochs=300,  # Moderate training epochs for testing
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluate final model
        final_pred = final_model.predict(x_elements_scaled, verbose=0)
        final_mse = mean_squared_error(x_compounds_normalized, final_pred)
        final_r2 = r2_score(x_compounds_normalized.flatten(), final_pred.flatten())

        print(f"\nFinal improved model performance:")
        print(f"  MSE: {final_mse:.6f}")
        print(f"  R²: {final_r2:.6f}")

        # Compare with original results
        original_mse = 0.729800
        original_r2 = 0.270200

        print(f"\nComparison with original model:")
        print(f"  MSE: {original_mse:.6f} → {final_mse:.6f} (improvement: {(original_mse-final_mse)/original_mse*100:.2f}%)")
        print(f"  R²: {original_r2:.6f} → {final_r2:.6f} (improvement: {(final_r2-original_r2)/original_r2*100:.2f}%)")

        # Create ensemble model
        ensemble_model = self.create_ensemble_model(fold_models)

        self.final_model = final_model
        self.ensemble_model = ensemble_model

        # Save model weights and scalers
        try:
            final_model.save_weights(self.final_weights_path)
            self.save_scalers()
            print(f"\nModel weights saved to: {self.final_weights_path}")
        except Exception as e:
            print(f"Failed to save model weights: {e}")

        return ensemble_model

    def create_fallback_model(self, x_elements, x_compounds):
        """Create a simple fallback model when TensorFlow is not available."""
        from sklearn.linear_model import LinearRegression

        print("Creating simple linear regression fallback model...")

        # Fit scalers
        x_elements_scaled = self.element_scaler.fit_transform(x_elements)
        x_compounds_normalized = x_compounds / (np.sum(x_compounds, axis=1, keepdims=True) + 1e-8)

        # Create simple linear regression model
        model = LinearRegression()
        model.fit(x_elements_scaled, x_compounds_normalized)

        # Create ensemble-like wrapper
        class SimpleFallbackModel:
            def __init__(self, model, parent):
                self.model = model
                self.parent = parent

            def predict(self, x, verbose=0):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x_scaled = self.parent.element_scaler.transform(x)
                predictions = self.model.predict(x_scaled)
                predictions = np.clip(predictions, 1e-6, 1.0)
                predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
                return predictions

        self.ensemble_model = SimpleFallbackModel(model, self)
        self.final_model = model

        print("Fallback model created successfully!")
        return model

    def perform_improved_cross_validation(self, x_elements, x_compounds, verbose=1):
        """Perform improved cross-validation."""
        print(f"Starting improved phase composition prediction {self.n_folds}-fold cross-validation...")

        # Improved data preprocessing
        x_elements_scaled, x_compounds_scaled, x_compounds_normalized = self.advanced_data_preprocessing(
            x_elements, x_compounds)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_elements_scaled)):
            print(f"\nTraining fold {fold + 1}/{self.n_folds} (improved version)...")

            # Data splitting
            x_train, x_val = x_elements_scaled[train_idx], x_elements_scaled[val_idx]
            y_train, y_val = x_compounds_normalized[train_idx], x_compounds_normalized[val_idx]

            # Data augmentation - add more reasonable noise
            noise_factor = 0.02
            x_train_aug = x_train + np.random.normal(0, noise_factor, x_train.shape)

            # Target data augmentation - maintain probability distribution characteristics
            y_train_aug = y_train + np.random.normal(0, 0.01, y_train.shape)
            y_train_aug = np.clip(y_train_aug, 1e-6, 1.0)  # Ensure non-negative
            y_train_aug = y_train_aug / np.sum(y_train_aug, axis=1, keepdims=True)  # Re-normalize

            # Create model
            model = self.create_improved_phase_model(f"improved_phase_fold_{fold}")

            # Create callbacks
            callbacks = self.create_improved_callbacks(fold)

            # Train model
            history = model.fit(
                x_train_aug, y_train_aug,
                validation_data=(x_val, y_val),
                batch_size=16,  # Smaller batch size
                epochs=300,     # Moderate training epochs for testing
                callbacks=callbacks,
                verbose=0 if verbose == 0 else 1
            )

            # Evaluate model
            y_pred = model.predict(x_val, verbose=0)

            # Calculate evaluation metrics
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val.flatten(), y_pred.flatten())

            # Additional evaluation metrics
            mae = np.mean(np.abs(y_val - y_pred))

            # Probability distribution quality check
            pred_sums = np.sum(y_pred, axis=1)
            sum_deviation = np.mean(np.abs(pred_sums - 1.0))

            cv_scores.append({
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'sum_deviation': sum_deviation
            })
            fold_models.append(model)

            print(f"  Fold {fold + 1} results - MSE: {mse:.6f}, R²: {r2:.6f}, MAE: {mae:.6f}")
            print(f"  Probability sum deviation: {sum_deviation:.6f}")

            # Clear memory
            clear_training_cache()

        # Calculate average scores
        avg_mse = np.mean([score['mse'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        avg_mae = np.mean([score['mae'] for score in cv_scores])
        avg_sum_dev = np.mean([score['sum_deviation'] for score in cv_scores])

        std_mse = np.std([score['mse'] for score in cv_scores])
        std_r2 = np.std([score['r2'] for score in cv_scores])

        print(f"\nImproved phase composition prediction cross-validation results:")
        print(f"  Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        print(f"  Average R²: {avg_r2:.6f} ± {std_r2:.6f}")
        print(f"  Average MAE: {avg_mae:.6f}")
        print(f"  Probability distribution quality: {avg_sum_dev:.6f}")

        # Performance improvement analysis
        original_mse = 0.729800
        original_r2 = 0.270200

        mse_improvement = (original_mse - avg_mse) / original_mse * 100
        r2_improvement = (avg_r2 - original_r2) / original_r2 * 100

        print(f"\nPerformance improvement analysis:")
        print(f"  MSE improvement: {mse_improvement:.2f}%")
        print(f"  R² improvement: {r2_improvement:.2f}%")

        self.cv_scores = cv_scores
        self.fold_models = fold_models

        return cv_scores, fold_models

    def create_ensemble_model(self, fold_models):
        """Create ensemble model."""
        class ImprovedEnsembleModel:
            def __init__(self, models, parent):
                self.models = models
                self.parent = parent

            def predict(self, x, verbose=0):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                elif x.ndim > 2:
                    x = x.reshape(x.shape[0], -1)

                # Standardize input
                x_scaled = self.parent.element_scaler.transform(x)

                predictions = []
                for model in self.models:
                    pred = model.predict(x_scaled, verbose=0)
                    predictions.append(pred)

                # Ensemble prediction
                predictions = np.array(predictions)
                mean_pred = np.mean(predictions, axis=0)

                # Ensure output is valid probability distribution
                mean_pred = np.clip(mean_pred, 1e-6, 1.0)
                mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)

                return mean_pred

            def summary(self):
                print(f"Improved phase composition prediction ensemble model contains {len(self.models)} sub-models")
                if len(self.models) > 0:
                    print("\nSingle model architecture:")
                    self.models[0].summary()

        return ImprovedEnsembleModel(fold_models, self)


class DualNetworkSurrogateModel(SurrogateModel):
    """
    Dual network surrogate model for direct property prediction.

    This model directly predicts mechanical properties (elastic modulus, yield strength)
    from element compositions without going through phase composition prediction.
    """

    def __init__(self, search_dims=3, network_input_dims=4, input_dims=None, output_dims=2, n_folds=5, **kwargs):
        # Handle backward compatibility
        if input_dims is not None:
            search_dims = input_dims

        if DANTE_AVAILABLE:
            super().__init__(input_dims=search_dims, **kwargs)

        self.input_dims = search_dims  # Search space dimensions
        self.search_dims = search_dims  # 3D search space
        self.network_input_dims = network_input_dims  # 4D network input
        self.output_dims = output_dims  # 2D output (elastic modulus, yield strength)
        self.n_folds = n_folds

        # Initialize scalers
        self.element_scaler = StandardScaler()
        self.property_scaler = StandardScaler()

        # Model storage
        self.fold_models = []
        self.final_model = None
        self.ensemble_model = None

        # Training results
        self.cv_scores = []
        self.training_history = []

        # Weight file paths - use original model_weights directory
        self.weights_dir = Path("../model_weights")  # Go up one level to find original weights
        self.weights_dir.mkdir(exist_ok=True)
        self.model_name = "dual_network"
        self.final_weights_path = self.weights_dir / f"{self.model_name}_final.weights.h5"
        self.scaler_path = self.weights_dir / f"{self.model_name}_scalers.pkl"

    def save_scalers(self):
        """Save scalers."""
        scalers = {
            'element_scaler': self.element_scaler,
            'property_scaler': self.property_scaler
        }
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to: {self.scaler_path}")

    def load_scalers(self):
        """Load scalers."""
        if self.scaler_path.exists():
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            self.element_scaler = scalers['element_scaler']
            self.property_scaler = scalers['property_scaler']
            print(f"Scalers loaded from {self.scaler_path}")
            return True
        return False

    def create_dual_network_model(self, model_name="dual_network_model"):
        """Create dual network model for direct property prediction."""
        inputs = keras.Input(shape=(self.network_input_dims,), name=f'{model_name}_input')

        # Shared feature extraction layers
        shared = layers.Dense(256, activation='relu', name=f'{model_name}_shared1')(inputs)
        shared = layers.LayerNormalization(name=f'{model_name}_shared_ln1')(shared)
        shared = layers.Dropout(0.3, name=f'{model_name}_shared_dropout1')(shared)

        shared = layers.Dense(128, activation='relu', name=f'{model_name}_shared2')(shared)
        shared = layers.LayerNormalization(name=f'{model_name}_shared_ln2')(shared)
        shared = layers.Dropout(0.2, name=f'{model_name}_shared_dropout2')(shared)

        # Elastic modulus branch
        elastic_branch = layers.Dense(64, activation='relu', name=f'{model_name}_elastic1')(shared)
        elastic_branch = layers.LayerNormalization(name=f'{model_name}_elastic_ln1')(elastic_branch)
        elastic_branch = layers.Dropout(0.2, name=f'{model_name}_elastic_dropout1')(elastic_branch)

        elastic_branch = layers.Dense(32, activation='relu', name=f'{model_name}_elastic2')(elastic_branch)
        elastic_branch = layers.LayerNormalization(name=f'{model_name}_elastic_ln2')(elastic_branch)

        elastic_output = layers.Dense(1, activation='linear', name=f'{model_name}_elastic_output')(elastic_branch)

        # Yield strength branch
        yield_branch = layers.Dense(64, activation='relu', name=f'{model_name}_yield1')(shared)
        yield_branch = layers.LayerNormalization(name=f'{model_name}_yield_ln1')(yield_branch)
        yield_branch = layers.Dropout(0.2, name=f'{model_name}_yield_dropout1')(yield_branch)

        yield_branch = layers.Dense(32, activation='relu', name=f'{model_name}_yield2')(yield_branch)
        yield_branch = layers.LayerNormalization(name=f'{model_name}_yield_ln2')(yield_branch)

        yield_output = layers.Dense(1, activation='linear', name=f'{model_name}_yield_output')(yield_branch)

        # Combine outputs
        outputs = layers.Concatenate(name=f'{model_name}_concat')([elastic_output, yield_output])

        model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                clipnorm=1.0
            ),
            loss='mse',
            metrics=['mae', 'mse']
        )

        return model

    def create_model(self) -> keras.Model:
        """Required implementation of abstract method from SurrogateModel."""
        return self.create_dual_network_model("dual_network_model")

    def __call__(self, x_elements, y_properties, verbose=1):
        """Train dual network model."""
        print("\n=== Starting dual network model training ===")

        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            print("TensorFlow not available. Creating simple fallback model...")
            return self.create_dual_fallback_model(x_elements, y_properties)

        # Try to load existing model weights
        if self.final_weights_path.exists() and self.load_scalers():
            print(f"Found saved model weights: {self.final_weights_path}")
            print("Loading pre-trained model...")

            # Data preprocessing (using loaded scalers)
            x_elements_scaled = self.element_scaler.transform(x_elements)
            y_properties_scaled = self.property_scaler.transform(y_properties)

            # Create model and load weights
            final_model = self.create_dual_network_model("final_dual_network_model")
            try:
                final_model.load_weights(self.final_weights_path)
                print("Model weights loaded successfully!")

                # Validate loaded model performance
                final_pred_scaled = final_model.predict(x_elements_scaled, verbose=0)
                final_pred = self.property_scaler.inverse_transform(final_pred_scaled)

                # Calculate MSE and R² on scaled data for fair comparison
                y_properties_scaled = self.property_scaler.transform(y_properties)
                final_mse_scaled = mean_squared_error(y_properties_scaled, final_pred_scaled)
                final_r2 = r2_score(y_properties.flatten(), final_pred.flatten())

                # Also calculate MSE on original scale (for reference)
                final_mse_original = mean_squared_error(y_properties, final_pred)

                print(f"Loaded model performance validation:")
                print(f"  MSE (scaled): {final_mse_scaled:.6f}")
                print(f"  MSE (original): {final_mse_original:.2e}")
                print(f"  R²: {final_r2:.6f}")
                print(f"  Note: Large original MSE due to unit mismatch (Pa vs normalized)")

                # Create simple ensemble model
                class SimpleDualEnsembleModel:
                    def __init__(self, model, parent):
                        self.model = model
                        self.parent = parent

                    def predict(self, x, verbose=0):
                        if x.ndim == 1:
                            x = x.reshape(1, -1)
                        x_scaled = self.parent.element_scaler.transform(x)
                        pred_scaled = self.model.predict(x_scaled, verbose=verbose)
                        predictions = self.parent.property_scaler.inverse_transform(pred_scaled)
                        return predictions

                self.final_model = final_model
                self.ensemble_model = SimpleDualEnsembleModel(final_model, self)

                print("Completed initialization using pre-trained model!")
                return self.ensemble_model

            except Exception as e:
                print(f"Failed to load model weights: {e}")
                print("Will retrain model...")

        # If no pre-trained model or loading failed, perform full training
        print("Starting full model training process...")

        # Debug: Check input dimensions
        print(f"Input data shape: {x_elements.shape}")
        print(f"Expected network input dims: {self.network_input_dims}")
        print(f"Search dims: {self.search_dims}")

        # Validate input dimensions
        if x_elements.shape[1] != self.network_input_dims:
            raise ValueError(f"Input data has {x_elements.shape[1]} dimensions, "
                           f"but model expects {self.network_input_dims} dimensions")

        # Data preprocessing
        x_elements_scaled = self.element_scaler.fit_transform(x_elements)
        y_properties_scaled = self.property_scaler.fit_transform(y_properties)

        # Perform cross-validation
        cv_scores, fold_models = self.perform_dual_cross_validation(
            x_elements_scaled, y_properties_scaled, y_properties, verbose)

        # Train final model on full dataset
        print("\nTraining final dual network model on full dataset...")

        final_model = self.create_dual_network_model("final_dual_network_model")

        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7, patience=20, min_lr=1e-8, verbose=0)
        ]

        # Final training
        final_history = final_model.fit(
            x_elements_scaled, y_properties_scaled,
            batch_size=16,
            epochs=300,
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluate final model
        final_pred_scaled = final_model.predict(x_elements_scaled, verbose=0)
        final_pred = self.property_scaler.inverse_transform(final_pred_scaled)

        final_mse = mean_squared_error(y_properties, final_pred)
        final_r2 = r2_score(y_properties.flatten(), final_pred.flatten())

        print(f"\nFinal dual network model performance:")
        print(f"  MSE: {final_mse:.6f}")
        print(f"  R²: {final_r2:.6f}")

        # Create ensemble model
        ensemble_model = self.create_dual_ensemble_model(fold_models)

        self.final_model = final_model
        self.ensemble_model = ensemble_model

        # Save model weights and scalers
        try:
            final_model.save_weights(self.final_weights_path)
            self.save_scalers()
            print(f"\nModel weights saved to: {self.final_weights_path}")
        except Exception as e:
            print(f"Failed to save model weights: {e}")

        return ensemble_model

    def create_dual_fallback_model(self, x_elements, y_properties):
        """Create a simple fallback model when TensorFlow is not available."""
        from sklearn.linear_model import LinearRegression

        print("Creating simple linear regression fallback model for dual network...")

        # Fit scalers
        x_elements_scaled = self.element_scaler.fit_transform(x_elements)
        y_properties_scaled = self.property_scaler.fit_transform(y_properties)

        # Create simple linear regression model
        model = LinearRegression()
        model.fit(x_elements_scaled, y_properties_scaled)

        # Create ensemble-like wrapper
        class SimpleDualFallbackModel:
            def __init__(self, model, parent):
                self.model = model
                self.parent = parent

            def predict(self, x, verbose=0):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x_scaled = self.parent.element_scaler.transform(x)
                pred_scaled = self.model.predict(x_scaled)
                predictions = self.parent.property_scaler.inverse_transform(pred_scaled)
                return predictions

        self.ensemble_model = SimpleDualFallbackModel(model, self)
        self.final_model = model

        print("Dual network fallback model created successfully!")
        return self.ensemble_model


class LogNormalizedDualNetworkSurrogateModel(DualNetworkSurrogateModel):
    """
    Enhanced dual network surrogate model with logarithmic transformation and weighted loss.

    Key improvements:
    1. Takes logarithm of Young's modulus and yield strength before normalization
    2. Uses weighted MSE loss with exp(2y) weights where y is log-normalized label
    3. Outputs are inverse-normalized and exponentiated to get final predictions
    """

    def __init__(self, search_dims=3, network_input_dims=4, input_dims=None, output_dims=2, n_folds=5, **kwargs):
        """Initialize log-normalized dual network surrogate model."""
        super().__init__(search_dims, network_input_dims, input_dims, output_dims, n_folds, **kwargs)

        # Additional scalers for log-transformed data
        self.log_property_scaler = StandardScaler()

        # File paths for log-normalized model
        self.model_name = "log_dual_network"
        self.final_weights_path = self.weights_dir / f"{self.model_name}_final.weights.h5"
        self.scaler_path = self.weights_dir / f"{self.model_name}_scalers.pkl"

    def log_transform_properties(self, y_properties):
        """Apply logarithmic transformation to properties."""
        # Ensure positive values (add small epsilon if needed)
        epsilon = 1e-10
        y_properties_safe = np.maximum(y_properties, epsilon)

        # Take logarithm
        y_log = np.log(y_properties_safe)

        return y_log

    def inverse_log_transform_properties(self, y_log_normalized):
        """Inverse transform: denormalize then exponentiate."""
        # First inverse normalize
        y_log = self.log_property_scaler.inverse_transform(y_log_normalized)

        # Then exponentiate
        y_properties = np.exp(y_log)

        return y_properties

    def weighted_mse_loss(self, y_true, y_pred):
        """
        Weighted MSE loss function with exp(2y) weights.

        Args:
            y_true: True log-normalized values
            y_pred: Predicted log-normalized values

        Returns:
            Weighted MSE loss
        """
        import tensorflow as tf

        # Calculate weights: exp(2 * y_true)
        weights = tf.exp(2.0 * y_true)

        # Calculate squared errors
        squared_errors = tf.square(y_true - y_pred)

        # Apply weights
        weighted_squared_errors = weights * squared_errors

        # Return mean weighted squared error
        return tf.reduce_mean(weighted_squared_errors)

    def save_scalers(self):
        """Save scalers including log property scaler."""
        scalers = {
            'element_scaler': self.element_scaler,
            'property_scaler': self.property_scaler,
            'log_property_scaler': self.log_property_scaler
        }
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Log-normalized scalers saved to: {self.scaler_path}")

    def load_scalers(self):
        """Load scalers including log property scaler."""
        if self.scaler_path.exists():
            with open(self.scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            self.element_scaler = scalers['element_scaler']
            self.property_scaler = scalers.get('property_scaler', StandardScaler())
            self.log_property_scaler = scalers.get('log_property_scaler', StandardScaler())
            print(f"Log-normalized scalers loaded from {self.scaler_path}")
            return True
        return False

    def create_log_dual_network_model(self, model_name="log_dual_network_model"):
        """Create dual network model with weighted loss for log-normalized data."""
        if not TF_AVAILABLE:
            print("TensorFlow not available for log dual network model")
            return None

        inputs = keras.Input(shape=(self.network_input_dims,), name=f'{model_name}_input')

        # Shared feature extraction layers
        shared = layers.Dense(256, activation='relu', name=f'{model_name}_shared1')(inputs)
        shared = layers.LayerNormalization(name=f'{model_name}_shared_ln1')(shared)
        shared = layers.Dropout(0.3, name=f'{model_name}_shared_dropout1')(shared)

        shared = layers.Dense(128, activation='relu', name=f'{model_name}_shared2')(shared)
        shared = layers.LayerNormalization(name=f'{model_name}_shared_ln2')(shared)
        shared = layers.Dropout(0.2, name=f'{model_name}_shared_dropout2')(shared)

        # Elastic modulus branch
        elastic_branch = layers.Dense(64, activation='relu', name=f'{model_name}_elastic1')(shared)
        elastic_branch = layers.LayerNormalization(name=f'{model_name}_elastic_ln1')(elastic_branch)
        elastic_branch = layers.Dropout(0.1, name=f'{model_name}_elastic_dropout1')(elastic_branch)

        elastic_branch = layers.Dense(32, activation='relu', name=f'{model_name}_elastic2')(elastic_branch)
        elastic_output = layers.Dense(1, activation='linear', name=f'{model_name}_elastic_output')(elastic_branch)

        # Yield strength branch
        yield_branch = layers.Dense(64, activation='relu', name=f'{model_name}_yield1')(shared)
        yield_branch = layers.LayerNormalization(name=f'{model_name}_yield_ln1')(yield_branch)
        yield_branch = layers.Dropout(0.1, name=f'{model_name}_yield_dropout1')(yield_branch)

        yield_branch = layers.Dense(32, activation='relu', name=f'{model_name}_yield2')(yield_branch)
        yield_output = layers.Dense(1, activation='linear', name=f'{model_name}_yield_output')(yield_branch)

        # Combine outputs
        outputs = layers.Concatenate(name=f'{model_name}_concat')([elastic_output, yield_output])

        model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

        # Compile model with weighted loss
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                clipnorm=1.0
            ),
            loss=self.weighted_mse_loss,  # Use weighted loss
            metrics=['mae', 'mse']
        )

        return model

    def create_model(self) -> keras.Model:
        """Required implementation of abstract method from SurrogateModel."""
        return self.create_log_dual_network_model("log_dual_network_model")

    def __call__(self, x_elements, y_properties, verbose=1):
        """Train log-normalized dual network model."""
        print("\n=== Starting log-normalized dual network model training ===")

        # Check if TensorFlow is available
        if not TF_AVAILABLE:
            print("TensorFlow not available. Creating simple fallback model...")
            return self.create_log_dual_fallback_model(x_elements, y_properties)

        # Apply log transformation to properties
        print("Applying logarithmic transformation to properties...")
        y_log = self.log_transform_properties(y_properties)
        print(f"Original property range: [{y_properties.min():.2e}, {y_properties.max():.2e}]")
        print(f"Log-transformed range: [{y_log.min():.3f}, {y_log.max():.3f}]")

        # Try to load existing model weights
        if self.final_weights_path.exists() and self.load_scalers():
            print(f"Found saved log-normalized model weights: {self.final_weights_path}")
            print("Loading pre-trained log-normalized model...")

            # Data preprocessing (using loaded scalers)
            x_elements_scaled = self.element_scaler.transform(x_elements)
            y_log_scaled = self.log_property_scaler.transform(y_log)

            # Create and load model
            model = self.create_log_dual_network_model("loaded_log_dual_network_model")
            if model is not None:
                try:
                    model.load_weights(self.final_weights_path)
                    print("Model weights loaded successfully!")

                    # Validate loaded model
                    pred_scaled = model.predict(x_elements_scaled, verbose=0)
                    pred_original = self.inverse_log_transform_properties(pred_scaled)

                    mse_scaled = mean_squared_error(y_log_scaled.flatten(), pred_scaled.flatten())
                    mse_original = mean_squared_error(y_properties.flatten(), pred_original.flatten())
                    r2 = r2_score(y_properties.flatten(), pred_original.flatten())

                    print(f"Loaded model performance validation:")
                    print(f"  MSE (log-scaled): {mse_scaled:.6f}")
                    print(f"  MSE (original): {mse_original:.2e}")
                    print(f"  R²: {r2:.6f}")

                    # Create ensemble wrapper
                    class LogNormalizedEnsembleWrapper:
                        def __init__(self, model, parent):
                            self.model = model
                            self.parent = parent

                        def predict(self, x, verbose=0):
                            if x.ndim == 1:
                                x = x.reshape(1, -1)
                            x_scaled = self.parent.element_scaler.transform(x)
                            pred_scaled = self.model.predict(x_scaled, verbose=verbose)
                            predictions = self.parent.inverse_log_transform_properties(pred_scaled)
                            return predictions

                    self.ensemble_model = LogNormalizedEnsembleWrapper(model, self)
                    self.final_model = model

                    print("Completed initialization using pre-trained log-normalized model!")
                    return self.ensemble_model

                except Exception as e:
                    print(f"Failed to load weights: {e}")
                    print("Training new model...")

        # Data preprocessing
        print("Preprocessing data for log-normalized training...")
        x_elements_scaled = self.element_scaler.fit_transform(x_elements)
        y_log_scaled = self.log_property_scaler.fit_transform(y_log)

        print(f"Element features scaled range: [{x_elements_scaled.min():.3f}, {x_elements_scaled.max():.3f}]")
        print(f"Log properties scaled range: [{y_log_scaled.min():.3f}, {y_log_scaled.max():.3f}]")

        # Perform cross-validation
        cv_scores, fold_models = self.perform_log_cross_validation(
            x_elements_scaled, y_log_scaled, y_properties, verbose)

        # Train final model on full dataset
        print("\nTraining final log-normalized dual network model on full dataset...")

        final_model = self.create_log_dual_network_model("final_log_dual_network_model")

        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7, patience=20, min_lr=1e-8, verbose=0)
        ]

        # Final training
        final_history = final_model.fit(
            x_elements_scaled, y_log_scaled,
            batch_size=16,
            epochs=300,
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluate final model
        final_pred_scaled = final_model.predict(x_elements_scaled, verbose=0)
        final_pred = self.inverse_log_transform_properties(final_pred_scaled)

        final_mse = mean_squared_error(y_properties, final_pred)
        final_r2 = r2_score(y_properties.flatten(), final_pred.flatten())

        print(f"\nFinal log-normalized dual network model performance:")
        print(f"  MSE: {final_mse:.6f}")
        print(f"  R²: {final_r2:.6f}")

        # Create ensemble model
        ensemble_model = self.create_log_dual_ensemble_model(fold_models)

        self.final_model = final_model
        self.ensemble_model = ensemble_model

        # Save model and scalers
        try:
            final_model.save_weights(self.final_weights_path)
            self.save_scalers()
            print(f"Log-normalized model saved to: {self.final_weights_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

        print("Log-normalized dual network model training completed!")
        return ensemble_model

    def perform_log_cross_validation(self, x_elements_scaled, y_log_scaled, y_properties_original, verbose=1):
        """Perform cross-validation for log-normalized model."""
        from sklearn.model_selection import KFold

        print(f"\nPerforming {self.n_folds}-fold cross-validation for log-normalized model...")

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(x_elements_scaled)):
            print(f"\nTraining fold {fold + 1}/{self.n_folds}...")

            # Split data
            x_train, x_val = x_elements_scaled[train_idx], x_elements_scaled[val_idx]
            y_train_log, y_val_log = y_log_scaled[train_idx], y_log_scaled[val_idx]
            y_val_orig = y_properties_original[val_idx]

            # Create model
            model = self.create_log_dual_network_model(f"log_dual_network_fold_{fold}")

            # Create callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-8, verbose=0)
            ]

            # Train model
            history = model.fit(
                x_train, y_train_log,
                validation_data=(x_val, y_val_log),
                batch_size=16,
                epochs=200,
                callbacks=callbacks,
                verbose=0 if verbose == 0 else 1
            )

            # Evaluate model
            val_pred_log = model.predict(x_val, verbose=0)
            val_pred_orig = self.inverse_log_transform_properties(val_pred_log)

            # Calculate metrics on original scale
            mse = mean_squared_error(y_val_orig, val_pred_orig)
            r2 = r2_score(y_val_orig.flatten(), val_pred_orig.flatten())

            # Calculate metrics on log scale
            mse_log = mean_squared_error(y_val_log, val_pred_log)
            r2_log = r2_score(y_val_log.flatten(), val_pred_log.flatten())

            fold_score = {
                'fold': fold + 1,
                'mse_original': mse,
                'r2_original': r2,
                'mse_log': mse_log,
                'r2_log': r2_log,
                'history': history.history
            }

            cv_scores.append(fold_score)
            fold_models.append(model)

            print(f"  Fold {fold + 1} - MSE (original): {mse:.6f}, R² (original): {r2:.4f}")
            print(f"  Fold {fold + 1} - MSE (log): {mse_log:.6f}, R² (log): {r2_log:.4f}")

        # Calculate average scores
        avg_mse_orig = np.mean([score['mse_original'] for score in cv_scores])
        avg_r2_orig = np.mean([score['r2_original'] for score in cv_scores])
        avg_mse_log = np.mean([score['mse_log'] for score in cv_scores])
        avg_r2_log = np.mean([score['r2_log'] for score in cv_scores])

        print(f"\nCross-validation results (log-normalized model):")
        print(f"  Average MSE (original): {avg_mse_orig:.6f} ± {np.std([score['mse_original'] for score in cv_scores]):.6f}")
        print(f"  Average R² (original): {avg_r2_orig:.4f} ± {np.std([score['r2_original'] for score in cv_scores]):.4f}")
        print(f"  Average MSE (log): {avg_mse_log:.6f} ± {np.std([score['mse_log'] for score in cv_scores]):.6f}")
        print(f"  Average R² (log): {avg_r2_log:.4f} ± {np.std([score['r2_log'] for score in cv_scores]):.4f}")

        self.cv_scores = cv_scores
        return cv_scores, fold_models

    def create_log_dual_ensemble_model(self, fold_models):
        """Create ensemble model for log-normalized predictions."""
        class LogNormalizedDualEnsemble:
            def __init__(self, models, parent):
                self.models = models
                self.parent = parent

            def predict(self, x, verbose=0):
                if x.ndim == 1:
                    x = x.reshape(1, -1)

                # Scale input
                x_scaled = self.parent.element_scaler.transform(x)

                # Get predictions from all models (log-normalized)
                predictions_log = []
                for model in self.models:
                    pred_log = model.predict(x_scaled, verbose=0)
                    predictions_log.append(pred_log)

                # Average log-normalized predictions
                avg_pred_log = np.mean(predictions_log, axis=0)

                # Convert back to original scale
                predictions_orig = self.parent.inverse_log_transform_properties(avg_pred_log)

                return predictions_orig

            def summary(self):
                print(f"Log-normalized dual network ensemble model contains {len(self.models)} sub-models")
                if len(self.models) > 0:
                    print("\nSingle model architecture:")
                    self.models[0].summary()

        return LogNormalizedDualEnsemble(fold_models, self)

    def create_log_dual_fallback_model(self, x_elements, y_properties):
        """Create fallback model when TensorFlow is not available."""
        from sklearn.ensemble import RandomForestRegressor

        print("Creating log-normalized fallback model using RandomForest...")

        # Apply log transformation
        y_log = self.log_transform_properties(y_properties)

        # Fit scalers
        x_elements_scaled = self.element_scaler.fit_transform(x_elements)
        y_log_scaled = self.log_property_scaler.fit_transform(y_log)

        # Create and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_elements_scaled, y_log_scaled)

        # Evaluate
        pred_log_scaled = model.predict(x_elements_scaled)
        pred_original = self.inverse_log_transform_properties(pred_log_scaled.reshape(-1, y_log.shape[1]))

        mse = mean_squared_error(y_properties, pred_original)
        r2 = r2_score(y_properties.flatten(), pred_original.flatten())

        print(f"Log-normalized fallback model performance:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R²: {r2:.6f}")

        # Create ensemble-like wrapper
        class LogNormalizedFallbackModel:
            def __init__(self, model, parent):
                self.model = model
                self.parent = parent

            def predict(self, x, verbose=0):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x_scaled = self.parent.element_scaler.transform(x)
                pred_log_scaled = self.model.predict(x_scaled)
                if pred_log_scaled.ndim == 1:
                    pred_log_scaled = pred_log_scaled.reshape(-1, self.parent.output_dims)
                predictions = self.parent.inverse_log_transform_properties(pred_log_scaled)
                return predictions

        self.ensemble_model = LogNormalizedFallbackModel(model, self)
        self.final_model = model

        print("Log-normalized dual network fallback model created successfully!")
        return self.ensemble_model

    def perform_dual_cross_validation(self, x_elements_scaled, y_properties_scaled, y_properties_original, verbose=1):
        """Perform cross-validation for dual network model."""
        print(f"Starting dual network {self.n_folds}-fold cross-validation...")

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_elements_scaled)):
            print(f"\nTraining fold {fold + 1}/{self.n_folds} (dual network)...")

            # Data splitting
            x_train, x_val = x_elements_scaled[train_idx], x_elements_scaled[val_idx]
            y_train, y_val = y_properties_scaled[train_idx], y_properties_scaled[val_idx]
            y_val_orig = y_properties_original[val_idx]

            # Data augmentation
            noise_factor = 0.02
            x_train_aug = x_train + np.random.normal(0, noise_factor, x_train.shape)
            y_train_aug = y_train + np.random.normal(0, 0.01, y_train.shape)

            # Create model
            model = self.create_dual_network_model(f"dual_network_fold_{fold}")

            # Create callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=15, min_lr=1e-8, verbose=0)
            ]

            # Train model
            history = model.fit(
                x_train_aug, y_train_aug,
                validation_data=(x_val, y_val),
                batch_size=16,
                epochs=200,
                callbacks=callbacks,
                verbose=0 if verbose == 0 else 1
            )

            # Evaluate model
            y_pred_scaled = model.predict(x_val, verbose=0)
            y_pred = self.property_scaler.inverse_transform(y_pred_scaled)

            # Calculate evaluation metrics
            mse = mean_squared_error(y_val_orig, y_pred)
            r2 = r2_score(y_val_orig.flatten(), y_pred.flatten())
            mae = np.mean(np.abs(y_val_orig - y_pred))

            # Individual property metrics
            elastic_mse = mean_squared_error(y_val_orig[:, 0], y_pred[:, 0])
            elastic_r2 = r2_score(y_val_orig[:, 0], y_pred[:, 0])
            yield_mse = mean_squared_error(y_val_orig[:, 1], y_pred[:, 1])
            yield_r2 = r2_score(y_val_orig[:, 1], y_pred[:, 1])

            cv_scores.append({
                'mse': mse, 'r2': r2, 'mae': mae,
                'elastic_mse': elastic_mse, 'elastic_r2': elastic_r2,
                'yield_mse': yield_mse, 'yield_r2': yield_r2
            })
            fold_models.append(model)

            print(f"  Fold {fold + 1} results - Overall MSE: {mse:.6f}, R²: {r2:.6f}")
            print(f"    Elastic modulus - MSE: {elastic_mse:.6f}, R²: {elastic_r2:.6f}")
            print(f"    Yield strength - MSE: {yield_mse:.6f}, R²: {yield_r2:.6f}")

            # Clear memory
            clear_training_cache()

        # Calculate average scores
        avg_mse = np.mean([score['mse'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        avg_elastic_r2 = np.mean([score['elastic_r2'] for score in cv_scores])
        avg_yield_r2 = np.mean([score['yield_r2'] for score in cv_scores])

        print(f"\nDual network cross-validation results:")
        print(f"  Average overall MSE: {avg_mse:.6f}")
        print(f"  Average overall R²: {avg_r2:.6f}")
        print(f"  Average elastic modulus R²: {avg_elastic_r2:.6f}")
        print(f"  Average yield strength R²: {avg_yield_r2:.6f}")

        self.cv_scores = cv_scores
        self.fold_models = fold_models

        return cv_scores, fold_models

    def create_dual_ensemble_model(self, fold_models):
        """Create ensemble model for dual network."""
        class DualEnsembleModel:
            def __init__(self, models, parent):
                self.models = models
                self.parent = parent

            def predict(self, x, verbose=0):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                elif x.ndim > 2:
                    x = x.reshape(x.shape[0], -1)

                # Standardize input
                x_scaled = self.parent.element_scaler.transform(x)

                predictions = []
                for model in self.models:
                    pred_scaled = model.predict(x_scaled, verbose=0)
                    pred = self.parent.property_scaler.inverse_transform(pred_scaled)
                    predictions.append(pred)

                # Ensemble prediction
                predictions = np.array(predictions)
                mean_pred = np.mean(predictions, axis=0)

                return mean_pred

            def summary(self):
                print(f"Dual network ensemble model contains {len(self.models)} sub-models")
                if len(self.models) > 0:
                    print("\nSingle model architecture:")
                    self.models[0].summary()

        return DualEnsembleModel(fold_models, self)
