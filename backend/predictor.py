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

# Add parent directory to path for utils imports
sys.path.insert(0, '/app')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "/app/models"
DATA_DIR = "/app/data"

# Feature columns expected by the models (must match training)
# Order matches the trained model's feature_names
FEATURE_COLUMNS = [
    'cycle', 'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'voltage_range', 'voltage_skew', 'voltage_kurtosis', 'voltage_slope',
    'current_mean', 'current_std', 'current_min', 'current_max', 'current_range',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_rise',
    'discharge_time', 'power_mean', 'power_max', 'energy', 'capacity',
    'initial_capacity', 'capacity_fade', 'capacity_ratio', 'ambient_temperature',
    'Re', 'Rct', 'soh', 'cycle_normalized'
]


class BatteryPredictor:
    """Centralized battery RUL predictor.
    
    Handles model loading at initialization and provides
    prediction functionality independent of Streamlit.
    """
    
    def __init__(self):
        """Initialize predictor and load models."""
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[MinMaxScaler] = None
        self.feature_names: list = FEATURE_COLUMNS
        self._models_loaded = False
        
        # Try to load models on initialization
        self._load_models()
        self._load_or_create_scaler()
    
    def _load_models(self) -> None:
        """Load pre-trained models from disk."""
        model_files = {
            'Linear Regression': 'linear_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl',
        }
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        for name, filename in model_files.items():
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    self.models[name] = joblib.load(filepath)
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
        
        # If no models loaded, train default models
        if not self.models:
            logger.info("No pre-trained models found. Training default models...")
            self._train_default_models()
        
        self._models_loaded = len(self.models) > 0
        logger.info(f"Models loaded: {list(self.models.keys())}")
    
    def _load_or_create_scaler(self) -> None:
        """Load or create a feature scaler."""
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler from disk")
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
                       cycle: int, capacity: float) -> pd.DataFrame:
        """Create feature vector from user inputs.
        
        Args:
            voltage: Battery voltage (V)
            current: Battery current (A)
            temperature: Temperature (°C)
            cycle: Cycle count
            capacity: Current capacity (Ah)
            
        Returns:
            DataFrame with features matching model input
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
        
        features = {
            'cycle': cycle,
            'cycle_normalized': min(cycle / 168.0, 1.0),
            'capacity': capacity,
            'initial_capacity': initial_capacity,
            'capacity_fade': capacity_fade,
            'capacity_ratio': capacity_ratio,
            'soh': capacity_ratio * 100,
            'ambient_temperature': temperature,
            'Re': Re,
            'Rct': Rct,
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
        }
        
        return pd.DataFrame([features])[self.feature_names]
    
    def predict(self, voltage: float, current: float, temperature: float,
                cycle: int, capacity: float, model_name: str = 'XGBoost') -> dict:
        """Make RUL prediction.
        
        Args:
            voltage: Battery voltage (V)
            current: Battery current (A)
            temperature: Temperature (°C)
            cycle: Cycle count
            capacity: Current capacity (Ah)
            model_name: Model to use
            
        Returns:
            Dictionary with prediction results
        """
        if model_name not in self.models:
            available = list(self.models.keys())
            if not available:
                raise RuntimeError("No models available")
            model_name = available[0]
            logger.warning(f"Requested model not available, using {model_name}")
        
        model = self.models[model_name]
        
        # Create features
        features = self.create_features(voltage, current, temperature, cycle, capacity)
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
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
        
        # Health classification
        health = self._classify_health(predicted_rul)
        
        # Confidence level
        confidence = self._calculate_confidence(predicted_rul, capacity, cycle)
        
        return {
            'predicted_rul_cycles': int(round(predicted_rul)),
            'battery_health': health,
            'confidence_level': confidence,
            'model_used': model_name
        }
    
    def _classify_health(self, rul: float) -> str:
        """Classify battery health based on RUL.
        
        Args:
            rul: Predicted remaining useful life in cycles
            
        Returns:
            Health classification string
        """
        if rul > 500:
            return "Healthy"
        elif rul >= 200:
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


# Global predictor instance (loaded once at module import)
_predictor: Optional[BatteryPredictor] = None


def get_predictor() -> BatteryPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = BatteryPredictor()
    return _predictor
