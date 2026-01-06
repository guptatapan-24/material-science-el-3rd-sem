"""
Centralized ML inference module for Battery RUL Prediction
Handles model loading, feature preparation, and prediction logic
Decoupled from Streamlit for use in FastAPI backend
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Optional, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from pathlib import Path

# Cross-platform path resolution
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Go up one level to get the project root (parent of backend/)
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for utils imports
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Cross-platform paths
MODELS_DIR = str(PROJECT_ROOT / "models")
DATA_DIR = str(PROJECT_ROOT / "data")

# Environment variable to force model retraining
FORCE_RETRAIN = os.environ.get('FORCE_RETRAIN', 'false').lower() in ('true', '1', 'yes')

# Feature columns expected by the models (must match training)
# V1 (NASA baseline) feature order - cycle_normalized at end
FEATURE_COLUMNS_V1 = [
    'cycle', 'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'voltage_range', 'voltage_skew', 'voltage_kurtosis', 'voltage_slope',
    'current_mean', 'current_std', 'current_min', 'current_max', 'current_range',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_rise',
    'discharge_time', 'power_mean', 'power_max', 'energy', 'capacity',
    'initial_capacity', 'capacity_fade', 'capacity_ratio', 'ambient_temperature',
    'Re', 'Rct', 'soh', 'cycle_normalized'
]

# V2 (physics-augmented) feature order - cycle_normalized after cycle
FEATURE_COLUMNS_V2 = [
    'cycle', 'cycle_normalized', 'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'voltage_range', 'voltage_skew', 'voltage_kurtosis', 'voltage_slope',
    'current_mean', 'current_std', 'current_min', 'current_max', 'current_range',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_rise',
    'discharge_time', 'power_mean', 'power_max', 'energy', 'capacity',
    'initial_capacity', 'capacity_fade', 'capacity_ratio', 'ambient_temperature',
    'Re', 'Rct', 'soh'
]

# Default to V1 for backward compatibility
FEATURE_COLUMNS = FEATURE_COLUMNS_V1


class BatteryPredictor:
    """Centralized battery RUL predictor.
    
    Handles model loading at initialization and provides
    prediction functionality independent of Streamlit.
    
    Phase 4: Supports versioned models:
    - v1_nasa: Original NASA-only trained baseline
    - v2_physics_augmented: CALCE physics-informed augmented model (default)
    """
    
    def __init__(self):
        """Initialize predictor and load models."""
        self.models: Dict[str, Any] = {}
        self.v2_models: Dict[str, Any] = {}  # Physics-augmented models
        self.v1_models: Dict[str, Any] = {}  # Baseline NASA-only models
        self.scaler: Optional[MinMaxScaler] = None
        self.scaler_v2: Optional[MinMaxScaler] = None
        self.scaler_v1: Optional[MinMaxScaler] = None
        self.feature_names: list = FEATURE_COLUMNS
        self.feature_names_v1: list = FEATURE_COLUMNS_V1
        self.feature_names_v2: list = FEATURE_COLUMNS_V2
        self._models_loaded = False
        
        # Model versions (Phase 4)
        self.model_versions: Dict[str, str] = {}
        self.current_model_version = "v2_physics_augmented"
        self.baseline_model_version = "v1_nasa"
        self.available_versions = ["v2_physics_augmented", "v1_nasa"]
        
        # Try to load models on initialization
        self._load_models()
        self._load_versioned_models()
        self._load_or_create_scaler()
    
    def _load_models(self) -> None:
        """Load pre-trained models from disk or train new ones.
        
        On startup:
        1. Check FORCE_RETRAIN environment variable
        2. Check if models exist
        3. Validate feature order matches FEATURE_COLUMNS
        4. Check for sklearn version compatibility
        5. If invalid, missing, or force retrain, retrain automatically
        
        Phase 4: Loads both v1 (baseline) and v2 (physics-augmented) models.
        """
        model_files = {
            'Linear Regression': 'linear_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl',
        }
        
        # Phase 4: Versioned model files
        v2_model_files = {
            'Linear Regression': 'linear_regression_v2_physics_augmented.pkl',
            'Random Forest': 'random_forest_v2_physics_augmented.pkl',
            'XGBoost': 'xgboost_v2_physics_augmented.pkl',
        }
        
        v1_model_files = {
            'Linear Regression (Baseline)': 'linear_regression_v1_nasa.pkl',
            'Random Forest (Baseline)': 'random_forest_v1_nasa.pkl',
            'XGBoost (Baseline)': 'xgboost_v1_nasa.pkl',
        }
        
        logger.info(f"Models directory: {MODELS_DIR}")
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Check for force retrain flag
        if FORCE_RETRAIN:
            logger.info("FORCE_RETRAIN=true detected. Will retrain all models...")
            self._clear_old_models()
            self._train_default_models()
            self._models_loaded = len(self.models) > 0
            logger.info(f"Models ready: {list(self.models.keys())}")
            return
        
        # Track if any model had version issues
        version_mismatch = False
        
        # Try loading existing models (default models are v2_physics_augmented)
        for name, filename in model_files.items():
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    # Capture warnings to detect version mismatches
                    import warnings
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        self.models[name] = joblib.load(filepath)
                        self.model_versions[name] = self.current_model_version
                        
                        # Check for version mismatch warnings
                        for warning in w:
                            if 'InconsistentVersionWarning' in str(warning.category.__name__):
                                logger.warning(f"Version mismatch detected for {name}: {warning.message}")
                                version_mismatch = True
                    
                    logger.info(f"Loaded {name} model from {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        
        # Try to load LSTM model
        lstm_path = os.path.join(MODELS_DIR, 'lstm.keras')
        if os.path.exists(lstm_path):
            try:
                from tensorflow import keras
                self.models['LSTM'] = keras.models.load_model(lstm_path)
                logger.info("Loaded LSTM model")
            except Exception as e:
                logger.warning(f"Failed to load LSTM: {e}")
        
        # Validate models have correct feature order
        needs_retrain = False
        if self.models:
            needs_retrain = not self._validate_model_features()
        
        # If version mismatch detected, recommend retraining
        if version_mismatch and not needs_retrain:
            logger.warning("="*60)
            logger.warning("SKLEARN VERSION MISMATCH DETECTED!")
            logger.warning("Models were trained with a different sklearn version.")
            logger.warning("To retrain models with current sklearn version, either:")
            logger.warning("  1. Set environment variable: FORCE_RETRAIN=true")
            logger.warning("  2. Delete the models folder and restart")
            logger.warning("="*60)
        
        # If no models or invalid features, train fresh models
        if not self.models or needs_retrain:
            if needs_retrain:
                logger.warning("Model feature order mismatch detected. Retraining models...")
                self._clear_old_models()
            else:
                logger.info("No pre-trained models found. Training default models...")
            self._train_default_models()
        
        self._models_loaded = len(self.models) > 0
        logger.info(f"Models ready: {list(self.models.keys())}")
    
    def _load_versioned_models(self) -> None:
        """Load versioned models (v1 baseline and v2 physics-augmented)."""
        # V2 Physics-Augmented Models
        v2_model_files = {
            'XGBoost': 'xgboost_v2_physics_augmented.pkl',
            'Random Forest': 'random_forest_v2_physics_augmented.pkl',
            'Linear Regression': 'linear_regression_v2_physics_augmented.pkl',
        }
        
        # V1 NASA-only Baseline Models
        v1_model_files = {
            'XGBoost': 'xgboost_v1_nasa.pkl',
            'Random Forest': 'random_forest_v1_nasa.pkl',
            'Linear Regression': 'linear_regression_v1_nasa.pkl',
        }
        
        # Load v2 models
        for name, filename in v2_model_files.items():
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    self.v2_models[name] = joblib.load(filepath)
                    logger.info(f"Loaded v2 {name} model from {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to load v2 {name}: {e}")
        
        # Load v2 scaler
        scaler_v2_path = os.path.join(MODELS_DIR, 'scaler_v2_physics_augmented.pkl')
        if os.path.exists(scaler_v2_path):
            try:
                self.scaler_v2 = joblib.load(scaler_v2_path)
                logger.info(f"Loaded v2 scaler from {scaler_v2_path}")
            except Exception as e:
                logger.warning(f"Failed to load v2 scaler: {e}")
        
        # Load v1 models
        for name, filename in v1_model_files.items():
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    self.v1_models[name] = joblib.load(filepath)
                    logger.info(f"Loaded v1 {name} model from {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to load v1 {name}: {e}")
        
        # Load v1 scaler
        scaler_v1_path = os.path.join(MODELS_DIR, 'scaler_v1_nasa.pkl')
        if os.path.exists(scaler_v1_path):
            try:
                self.scaler_v1 = joblib.load(scaler_v1_path)
                logger.info(f"Loaded v1 scaler from {scaler_v1_path}")
            except Exception as e:
                logger.warning(f"Failed to load v1 scaler: {e}")
        
        logger.info(f"Versioned models loaded - v2: {list(self.v2_models.keys())}, v1: {list(self.v1_models.keys())}")
    
    def _validate_model_features(self) -> bool:
        """Validate that loaded models have correct feature order.
        
        Returns:
            True if features match FEATURE_COLUMNS, False otherwise
        """
        try:
            # Check XGBoost model (has feature_names_ attribute)
            if 'XGBoost' in self.models:
                xgb_model = self.models['XGBoost']
                if hasattr(xgb_model, 'feature_names_in_'):
                    model_features = list(xgb_model.feature_names_in_)
                    if model_features != self.feature_names:
                        logger.warning(f"XGBoost feature mismatch!")
                        logger.warning(f"Expected: {self.feature_names[:5]}...")
                        logger.warning(f"Got: {model_features[:5]}...")
                        return False
            
            # Check Random Forest model
            if 'Random Forest' in self.models:
                rf_model = self.models['Random Forest']
                if hasattr(rf_model, 'feature_names_in_'):
                    model_features = list(rf_model.feature_names_in_)
                    if model_features != self.feature_names:
                        logger.warning(f"Random Forest feature mismatch!")
                        return False
            
            logger.info("Model feature validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"Feature validation error: {e}")
            return False
    
    def _clear_old_models(self) -> None:
        """Clear old model files before retraining."""
        self.models = {}
        files_to_remove = [
            'linear_regression.pkl',
            'random_forest.pkl', 
            'xgboost.pkl',
            'scaler.pkl'
        ]
        for filename in files_to_remove:
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old model: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to remove {filename}: {e}")
    
    def _load_or_create_scaler(self) -> None:
        """Load or create a feature scaler.
        
        Validates scaler has correct feature names before using.
        """
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            try:
                loaded_scaler = joblib.load(scaler_path)
                # Validate scaler features match
                if hasattr(loaded_scaler, 'feature_names_in_'):
                    scaler_features = list(loaded_scaler.feature_names_in_)
                    if scaler_features == self.feature_names:
                        self.scaler = loaded_scaler
                        logger.info("Loaded and validated scaler from disk")
                        return
                    else:
                        logger.warning("Scaler feature mismatch, creating new scaler...")
                else:
                    # Old scaler without feature names, recreate
                    self.scaler = loaded_scaler
                    logger.info("Loaded scaler from disk (legacy format)")
                    return
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")
        
        # Create a default scaler with reasonable ranges
        self.scaler = MinMaxScaler()
        # Fit with dummy data representing expected ranges
        dummy_data = self._create_dummy_training_data()
        self.scaler.fit(dummy_data)
        
        # Save scaler
        try:
            joblib.dump(self.scaler, scaler_path)
            logger.info("Created and saved new scaler")
        except Exception as e:
            logger.warning(f"Failed to save scaler: {e}")
    
    def _create_dummy_training_data(self) -> pd.DataFrame:
        """Create dummy data for scaler fitting.
        
        Features are created in the exact order specified by FEATURE_COLUMNS
        to ensure consistency with model training.
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Create data in the EXACT order of FEATURE_COLUMNS
        data = {
            'cycle': np.random.randint(1, 200, n_samples),
            'voltage_mean': np.random.uniform(3.0, 4.0, n_samples),
            'voltage_std': np.random.uniform(0.1, 0.5, n_samples),
            'voltage_min': np.random.uniform(2.5, 3.5, n_samples),
            'voltage_max': np.random.uniform(3.8, 4.2, n_samples),
            'voltage_range': np.random.uniform(0.5, 1.7, n_samples),
            'voltage_skew': np.random.uniform(-0.5, 0.5, n_samples),
            'voltage_kurtosis': np.random.uniform(-0.5, 0.5, n_samples),
            'voltage_slope': np.random.uniform(-0.002, 0, n_samples),
            'current_mean': np.random.uniform(-2.0, 0, n_samples),
            'current_std': np.random.uniform(0.01, 0.1, n_samples),
            'current_min': np.random.uniform(-2.0, -0.5, n_samples),
            'current_max': np.random.uniform(-0.1, 0.1, n_samples),
            'current_range': np.random.uniform(0.5, 2.0, n_samples),
            'temp_mean': np.random.uniform(20, 45, n_samples),
            'temp_std': np.random.uniform(1, 5, n_samples),
            'temp_min': np.random.uniform(15, 35, n_samples),
            'temp_max': np.random.uniform(25, 50, n_samples),
            'temp_range': np.random.uniform(5, 20, n_samples),
            'temp_rise': np.random.uniform(2, 10, n_samples),
            'discharge_time': np.random.uniform(2000, 4000, n_samples),
            'power_mean': np.random.uniform(2, 8, n_samples),
            'power_max': np.random.uniform(4, 10, n_samples),
            'energy': np.random.uniform(5000, 15000, n_samples),
            'capacity': np.random.uniform(1.2, 2.0, n_samples),
            'initial_capacity': np.full(n_samples, 2.0),
            'capacity_fade': np.random.uniform(0, 0.8, n_samples),
            'capacity_ratio': np.random.uniform(0.6, 1.0, n_samples),
            'ambient_temperature': np.random.uniform(20, 40, n_samples),
            'Re': np.random.uniform(0.05, 0.15, n_samples),
            'Rct': np.random.uniform(0.15, 0.35, n_samples),
            'soh': np.random.uniform(60, 100, n_samples),
            'cycle_normalized': np.random.uniform(0, 1, n_samples),
        }
        
        # Create DataFrame with explicit column ordering
        df = pd.DataFrame(data)
        return df[self.feature_names]
    
    def _train_default_models(self) -> None:
        """Train default models if none are available."""
        try:
            # Generate synthetic training data
            X_train, y_train = self._generate_training_data()
            
            # Train XGBoost
            logger.info("Training XGBoost model...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            self.models['XGBoost'] = xgb_model
            joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost.pkl'))
            
            # Train Random Forest
            logger.info("Training Random Forest model...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            self.models['Random Forest'] = rf_model
            joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest.pkl'))
            
            # Train Linear Regression
            logger.info("Training Linear Regression model...")
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            self.models['Linear Regression'] = lr_model
            joblib.dump(lr_model, os.path.join(MODELS_DIR, 'linear_regression.pkl'))
            
            logger.info("Default models trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training default models: {e}")
            raise
    
    def _generate_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data for model initialization.
        
        Features are created in the exact order specified by FEATURE_COLUMNS
        to ensure consistency with model training.
        """
        np.random.seed(42)
        n_samples = 2000
        
        data = []
        for _ in range(n_samples):
            cycle = np.random.randint(1, 200)
            initial_capacity = 2.0
            # Simulate degradation
            capacity_fade_val = 0.004 * cycle + np.random.normal(0, 0.02)
            capacity = max(1.2, initial_capacity - capacity_fade_val)
            capacity_ratio = capacity / initial_capacity
            
            # Calculate RUL based on degradation
            threshold = initial_capacity * 0.8
            if capacity > threshold:
                rul = int((capacity - threshold) / 0.004)
            else:
                rul = 0
            
            # Create sample in EXACT order of FEATURE_COLUMNS
            sample = {
                'cycle': cycle,
                'voltage_mean': 3.5 + np.random.normal(0, 0.2),
                'voltage_std': 0.35,
                'voltage_min': 2.7,
                'voltage_max': 4.1,
                'voltage_range': 1.4,
                'voltage_skew': -0.1,
                'voltage_kurtosis': 0.2,
                'voltage_slope': -0.0008,
                'current_mean': -1.0 + np.random.normal(0, 0.1),
                'current_std': 0.05,
                'current_min': -1.1,
                'current_max': 0,
                'current_range': 1.1,
                'temp_mean': 30 + np.random.normal(0, 3),
                'temp_std': 2.0,
                'temp_min': 25,
                'temp_max': 35,
                'temp_range': 10,
                'temp_rise': 5,
                'discharge_time': 3000 * capacity_ratio,
                'power_mean': 3.5,
                'power_max': 4.2,
                'energy': 10000 * capacity_ratio,
                'capacity': capacity,
                'initial_capacity': initial_capacity,
                'capacity_fade': initial_capacity - capacity,
                'capacity_ratio': capacity_ratio,
                'ambient_temperature': 25 + np.random.normal(0, 5),
                'Re': 0.055 + cycle * 0.00015,
                'Rct': 0.18 + cycle * 0.0003,
                'soh': capacity_ratio * 100,
                'cycle_normalized': cycle / 200.0,
                'rul': max(0, rul)
            }
            data.append(sample)
        
        df = pd.DataFrame(data)
        # Ensure correct column ordering
        X = df[self.feature_names]
        y = df['rul']
        
        return X, y
    
    def create_features(self, voltage: float, current: float, temperature: float,
                       cycle: int, capacity: float, version: str = 'v1') -> pd.DataFrame:
        """Create feature vector from user inputs.
        
        Args:
            voltage: Battery voltage (V)
            current: Battery current (A)
            temperature: Temperature (°C)
            cycle: Cycle count
            capacity: Current capacity (Ah)
            version: 'v1' for baseline feature order, 'v2' for physics-augmented order
            
        Returns:
            DataFrame with features matching model input in correct order
        """
        initial_capacity = 2.0
        capacity_fade = initial_capacity - capacity
        capacity_ratio = capacity / initial_capacity
        
        # Impedance estimates
        Re = 0.055 + (cycle * 0.00015) + (capacity_fade * 0.01)
        Rct = 0.18 + (cycle * 0.0003) + (capacity_fade * 0.02)
        
        # Voltage features
        voltage_range = 1.7
        voltage_min = max(2.5, voltage - (voltage_range / 2))
        voltage_max = min(4.2, voltage + (voltage_range / 2))
        voltage_slope = -0.0008 - (cycle * 0.000002)
        voltage_skew = -0.1 - (capacity_fade * 0.2)
        voltage_kurtosis = 0.2 + (capacity_fade * 0.3)
        
        # Current features
        current_abs = abs(current)
        
        # Temperature features
        temp_rise = 3 + (current_abs * 2)
        temp_min = temperature - 2
        temp_max = temperature + temp_rise
        temp_range = temp_max - temp_min
        
        # Power/Energy
        power_mean = voltage * current_abs
        power_max = voltage_max * current_abs * 1.1
        discharge_time = 3000 * capacity_ratio
        energy = power_mean * discharge_time / 3600
        
        # Create all features as dict
        features = {
            'cycle': cycle,
            'cycle_normalized': min(cycle / 168.0, 1.0),
            'voltage_mean': voltage,
            'voltage_std': 0.35 + (capacity_fade * 0.1),
            'voltage_min': voltage_min,
            'voltage_max': voltage_max,
            'voltage_range': voltage_range,
            'voltage_skew': voltage_skew,
            'voltage_kurtosis': voltage_kurtosis,
            'voltage_slope': voltage_slope,
            'current_mean': current,
            'current_std': 0.05 + (abs(current) * 0.02),
            'current_min': current if current < 0 else current - 0.1,
            'current_max': 0 if current < 0 else current,
            'current_range': current_abs + 0.1,
            'temp_mean': temperature + (temp_rise / 2),
            'temp_std': 1.5 + (temp_rise * 0.2),
            'temp_min': temp_min,
            'temp_max': temp_max,
            'temp_range': temp_range,
            'temp_rise': temp_rise,
            'discharge_time': discharge_time,
            'power_mean': power_mean,
            'power_max': power_max,
            'energy': energy,
            'capacity': capacity,
            'initial_capacity': initial_capacity,
            'capacity_fade': capacity_fade,
            'capacity_ratio': capacity_ratio,
            'ambient_temperature': temperature,
            'Re': Re,
            'Rct': Rct,
            'soh': capacity_ratio * 100,
        }
        
        # Select correct feature order based on version
        if version == 'v2':
            feature_names = self.feature_names_v2
        else:
            feature_names = self.feature_names_v1
        
        # Create DataFrame with explicit column ordering
        df = pd.DataFrame([features])
        return df[feature_names]
    
    def predict(self, voltage: float, current: float, temperature: float,
                cycle: int, capacity: float, model_name: str = 'XGBoost',
                model_version: str = 'v2_physics_augmented') -> dict:
        """Make RUL prediction.
        
        Args:
            voltage: Battery voltage (V)
            current: Battery current (A)
            temperature: Temperature (°C)
            cycle: Cycle count
            capacity: Current capacity (Ah)
            model_name: Model to use (XGBoost, Random Forest, Linear Regression)
            model_version: Model version ('v2_physics_augmented' or 'v1_nasa')
            
        Returns:
            Dictionary with prediction results
        """
        # Select model based on version
        if model_version == 'v1_nasa':
            models_dict = self.v1_models if self.v1_models else self.models
            scaler = self.scaler_v1 if self.scaler_v1 else self.scaler
            version_used = 'v1_nasa'
            feature_version = 'v1'
        else:
            models_dict = self.v2_models if self.v2_models else self.models
            scaler = self.scaler_v2 if self.scaler_v2 else self.scaler
            version_used = 'v2_physics_augmented'
            feature_version = 'v2'
        
        # Fallback to default models if versioned models not available
        if not models_dict:
            models_dict = self.models
            scaler = self.scaler
            version_used = self.current_model_version
            feature_version = 'v1'  # Default models use v1 feature order
        
        if model_name not in models_dict:
            available = list(models_dict.keys())
            if not available:
                raise RuntimeError("No models available")
            model_name = available[0]
            logger.warning(f"Requested model not available, using {model_name}")
        
        model = models_dict[model_name]
        
        # Create features with correct order for model version
        features = self.create_features(voltage, current, temperature, cycle, capacity, feature_version)
        
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(features),
            columns=features.columns
        )
        
        # Predict
        if model_name == 'LSTM':
            X_reshaped = features_scaled.values.reshape((1, 1, features_scaled.shape[1]))
            predicted_rul = float(model.predict(X_reshaped, verbose=0).flatten()[0])
        else:
            predicted_rul = float(model.predict(features_scaled)[0])
        
        # Ensure non-negative
        predicted_rul = max(0, predicted_rul)
        
        # Health classification (updated for Phase 4 with broader RUL range)
        health = self._classify_health(predicted_rul)
        
        # Confidence level
        confidence = self._calculate_confidence(predicted_rul, capacity, cycle)
        
        # Generate recommendation based on updated Phase 4 RUL ranges
        recommendation = self._generate_recommendation(predicted_rul)
        
        return {
            'predicted_rul_cycles': int(round(predicted_rul)),
            'battery_health': health,
            'confidence_level': confidence,
            'model_used': model_name,
            'model_version': version_used,
            'recommendation': recommendation
        }
    
    def predict_with_baseline_comparison(self, voltage: float, current: float, 
                                         temperature: float, cycle: int, 
                                         capacity: float, model_name: str = 'XGBoost') -> dict:
        """Make predictions with both v2 and v1 models for comparison.
        
        Useful for academic validation and showing improvement over baseline.
        """
        # Predict with v2 (physics-augmented)
        v2_result = self.predict(voltage, current, temperature, cycle, capacity, 
                                 model_name, 'v2_physics_augmented')
        
        # Predict with v1 (baseline)
        v1_result = self.predict(voltage, current, temperature, cycle, capacity, 
                                 model_name, 'v1_nasa')
        
        # Calculate improvement
        improvement = v2_result['predicted_rul_cycles'] - v1_result['predicted_rul_cycles']
        
        return {
            'v2_prediction': v2_result,
            'v1_baseline': v1_result,
            'rul_difference': improvement,
            'comparison_note': (
                f"V2 physics-augmented model predicts {abs(improvement)} cycles "
                f"{'more' if improvement > 0 else 'fewer'} RUL than V1 baseline"
            )
        }
    
    def _generate_recommendation(self, rul: float) -> str:
        """Generate maintenance recommendation based on RUL.
        
        Phase 4 Updated Ranges (aligned with NASA dataset distribution):
        - RUL > 150: Early healthy state (based on actual max ~296)
        - 50 <= RUL <= 150: Mid-life
        - RUL < 50: End-of-life approaching
        """
        if rul > 150:
            return "Battery in early healthy state – no action required"
        elif 50 <= rul <= 150:
            return "Battery in mid-life – optimize charging behavior"
        else:
            return "Battery approaching end-of-life – plan maintenance or replacement"
    
    def _classify_health(self, rul: float) -> str:
        """Classify battery health based on RUL.
        
        Phase 4 Updated Classification (aligned with NASA dataset):
        - Healthy: RUL > 100 cycles (significant remaining life)
        - Moderate: 30-100 cycles (mid-life degradation)
        - Critical: < 30 cycles (approaching EOL)
        
        Args:
            rul: Predicted remaining useful life in cycles
            
        Returns:
            Health classification string
        """
        if rul > 100:
            return "Healthy"
        elif rul >= 30:
            return "Moderate"
        else:
            return "Critical"
    
    def _calculate_confidence(self, rul: float, capacity: float, cycle: int) -> str:
        """Calculate prediction confidence level.
        
        Args:
            rul: Predicted RUL
            capacity: Current capacity
            cycle: Current cycle
            
        Returns:
            Confidence level string
        """
        # Higher confidence when:
        # - Capacity is within typical range (1.4-1.9 Ah)
        # - Cycle count is within training data range (1-168)
        # - RUL is within reasonable bounds
        
        score = 0
        
        # Capacity score
        if 1.4 <= capacity <= 1.9:
            score += 2
        elif 1.2 <= capacity <= 2.0:
            score += 1
        
        # Cycle score
        if 1 <= cycle <= 168:
            score += 2
        elif cycle <= 200:
            score += 1
        
        # RUL reasonability
        if 0 <= rul <= 300:
            score += 2
        elif rul <= 500:
            score += 1
        
        if score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"
    
    @property
    def is_ready(self) -> bool:
        """Check if predictor is ready for predictions."""
        return self._models_loaded and self.scaler is not None
    
    @property
    def available_models(self) -> list:
        """Get list of available models."""
        return list(self.models.keys())
    
    @property
    def available_model_versions(self) -> list:
        """Get list of available model versions."""
        versions = []
        if self.v2_models:
            versions.append('v2_physics_augmented')
        if self.v1_models:
            versions.append('v1_nasa')
        if not versions:
            versions.append(self.current_model_version)
        return versions
    
    def get_model_info(self) -> dict:
        """Get detailed information about loaded models."""
        return {
            'default_models': list(self.models.keys()),
            'v2_physics_augmented': list(self.v2_models.keys()),
            'v1_nasa_baseline': list(self.v1_models.keys()),
            'current_version': self.current_model_version,
            'available_versions': self.available_model_versions,
            'has_v2_scaler': self.scaler_v2 is not None,
            'has_v1_scaler': self.scaler_v1 is not None,
        }


# Global predictor instance (loaded once at module import)
_predictor: Optional[BatteryPredictor] = None


def get_predictor() -> BatteryPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = BatteryPredictor()
    return _predictor
