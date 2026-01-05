#!/usr/bin/env python3
"""
Battery RUL Model Training Script
Trains ML models for battery remaining useful life prediction.

Usage:
    python train_models.py

This script will:
1. Load/generate training data
2. Train XGBoost, Random Forest, Linear Regression models
3. Save models and scaler to /app/models directory
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
MODELS_DIR = "/app/models"
DATA_DIR = "/app/data"

# Feature columns - EXACT ORDER required by the models
FEATURE_COLUMNS = [
    'cycle', 'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'voltage_range', 'voltage_skew', 'voltage_kurtosis', 'voltage_slope',
    'current_mean', 'current_std', 'current_min', 'current_max', 'current_range',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_rise',
    'discharge_time', 'power_mean', 'power_max', 'energy', 'capacity',
    'initial_capacity', 'capacity_fade', 'capacity_ratio', 'ambient_temperature',
    'Re', 'Rct', 'soh', 'cycle_normalized'
]


def generate_synthetic_data(n_samples=2000):
    """Generate synthetic battery data for training.
    
    This creates realistic battery degradation data based on
    typical lithium-ion battery characteristics.
    """
    logger.info(f"Generating {n_samples} synthetic training samples...")
    np.random.seed(42)
    
    data = []
    for _ in range(n_samples):
        cycle = np.random.randint(1, 200)
        initial_capacity = 2.0
        
        # Simulate degradation
        capacity_fade_val = 0.004 * cycle + np.random.normal(0, 0.02)
        capacity = max(1.2, initial_capacity - capacity_fade_val)
        capacity_ratio = capacity / initial_capacity
        
        # Calculate RUL (remaining cycles until 80% capacity)
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
    return df


def train_models(X_train, y_train, X_test, y_test):
    """Train all ML models."""
    models = {}
    metrics = {}
    
    # 1. XGBoost (Primary Model)
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
    logger.info(f"  XGBoost - MAE: {metrics['XGBoost']['MAE']:.2f}, R²: {metrics['XGBoost']['R2']:.4f}")
    
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
    logger.info(f"  Random Forest - MAE: {metrics['Random Forest']['MAE']:.2f}, R²: {metrics['Random Forest']['R2']:.4f}")
    
    # 3. Linear Regression
    logger.info("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    models['Linear Regression'] = lr_model
    metrics['Linear Regression'] = calculate_metrics(y_test, y_pred_lr)
    logger.info(f"  Linear Regression - MAE: {metrics['Linear Regression']['MAE']:.2f}, R²: {metrics['Linear Regression']['R2']:.4f}")
    
    return models, metrics


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def save_models(models, scaler):
    """Save models and scaler to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, filepath)
        logger.info(f"Saved {name} to {filepath}")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Battery RUL Model Training")
    print("=" * 60)
    
    # Step 1: Generate/Load data
    logger.info("Step 1: Generating training data...")
    df = generate_synthetic_data(n_samples=2000)
    
    # Step 2: Prepare features and target
    logger.info("Step 2: Preparing features...")
    X = df[FEATURE_COLUMNS]
    y = df['rul']
    
    # Step 3: Split data
    logger.info("Step 3: Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Step 4: Create and fit scaler
    logger.info("Step 4: Creating scaler...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS
    )
    
    # Step 5: Train models
    logger.info("Step 5: Training models...")
    models, metrics = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Step 6: Save models
    logger.info("Step 6: Saving models...")
    save_models(models, scaler)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel Performance:")
    for name, m in metrics.items():
        print(f"  {name:20s} - MAE: {m['MAE']:6.2f}, RMSE: {m['RMSE']:6.2f}, R²: {m['R2']:.4f}")
    
    print(f"\nModels saved to: {MODELS_DIR}")
    print("Files created:")
    for f in os.listdir(MODELS_DIR):
        print(f"  - {f}")
    
    print("\n✅ Training successful! You can now use the prediction API.")


if __name__ == "__main__":
    main()
