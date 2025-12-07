import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib
import os
import logging
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "/app/models"

@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models and return trained instances with metrics."""
    models = {}
    metrics = {}
    
    try:
        # 1. Linear Regression
        logger.info("Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        models['Linear Regression'] = lr_model
        metrics['Linear Regression'] = calculate_metrics(y_test, y_pred_lr)
        
        # 2. Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        models['Random Forest'] = rf_model
        metrics['Random Forest'] = calculate_metrics(y_test, y_pred_rf)
        
        # 3. XGBoost (Main Model)
        logger.info("Training XGBoost...")
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
        y_pred_xgb = xgb_model.predict(X_test)
        models['XGBoost'] = xgb_model
        metrics['XGBoost'] = calculate_metrics(y_test, y_pred_xgb)
        
        # 4. LSTM (Pre-train and cache)
        logger.info("Training LSTM...")
        lstm_model, lstm_metrics = train_lstm_model(X_train, y_train, X_test, y_test)
        if lstm_model:
            models['LSTM'] = lstm_model
            metrics['LSTM'] = lstm_metrics
        
        # Save models
        save_models(models)
        
        logger.info("All models trained successfully!")
        return models, metrics
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise

def train_lstm_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train LSTM model with proper reshaping."""
    try:
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = model.fit(
            X_train_lstm, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predict
        y_pred_lstm = model.predict(X_test_lstm, verbose=0).flatten()
        
        # Calculate metrics
        lstm_metrics = calculate_metrics(y_test, y_pred_lstm)
        
        return model, lstm_metrics
        
    except Exception as e:
        logger.error(f"Error training LSTM: {e}")
        return None, {'MAE': 999, 'RMSE': 999, 'R2': 0}

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def save_models(models):
    """Save trained models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for name, model in models.items():
        try:
            filename = name.lower().replace(' ', '_')
            if isinstance(model, keras.Model):
                model.save(f"{MODELS_DIR}/{filename}.keras")
            else:
                joblib.dump(model, f"{MODELS_DIR}/{filename}.pkl")
            logger.info(f"Saved {name} model")
        except Exception as e:
            logger.warning(f"Failed to save {name}: {e}")

@st.cache_resource
def load_models():
    """Load pre-trained models from disk."""
    models = {}
    
    model_files = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'LSTM': 'lstm.keras'
    }
    
    for name, filename in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            try:
                if filename.endswith('.keras'):
                    models[name] = keras.models.load_model(filepath)
                else:
                    models[name] = joblib.load(filepath)
                logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
    
    return models

def predict_rul(model, X, model_name='XGBoost'):
    """Make RUL prediction with proper data handling."""
    try:
        if model_name == 'LSTM':
            # Reshape for LSTM
            X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
            prediction = model.predict(X_reshaped, verbose=0).flatten()[0]
        else:
            prediction = model.predict(X)[0]
        
        # Ensure positive prediction
        prediction = max(0, prediction)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

def simulate_what_if(model, base_features, parameter, delta, scaler, feature_names):
    """Simulate what-if scenario by adjusting a parameter."""
    try:
        # Create modified features
        modified_features = base_features.copy()
        
        # Map parameter to feature columns
        param_mapping = {
            'temperature': ['temp_mean', 'temp_min', 'temp_max'],
            'voltage': ['voltage_mean', 'voltage_min', 'voltage_max'],
            'current': ['current_mean', 'current_min', 'current_max'],
            'cycle_count': ['cycle_count']
        }
        
        if parameter in param_mapping:
            for col in param_mapping[parameter]:
                if col in modified_features.columns:
                    modified_features[col] += delta
        
        # Predict with modified features
        new_rul = predict_rul(model, modified_features)
        
        return new_rul
        
    except Exception as e:
        logger.error(f"What-if simulation error: {e}")
        return None