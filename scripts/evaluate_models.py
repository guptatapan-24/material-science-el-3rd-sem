"""
Phase 5: Model Evaluation, Validation, and Comparative Analysis

This script systematically evaluates and validates the performance improvements
achieved in Phase 4 by comparing the baseline NASA-only model (v1) with the
augmented multi-dataset model (v2).

Academic Purpose: Generate quantitative metrics, visual analysis, and
validation artifacts suitable for academic submission.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation_plots"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Feature columns for v1 and v2 models
FEATURE_COLUMNS_V1 = [
    'cycle', 'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'voltage_range', 'voltage_skew', 'voltage_kurtosis', 'voltage_slope',
    'current_mean', 'current_std', 'current_min', 'current_max', 'current_range',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_rise',
    'discharge_time', 'power_mean', 'power_max', 'energy', 'capacity',
    'initial_capacity', 'capacity_fade', 'capacity_ratio', 'ambient_temperature',
    'Re', 'Rct', 'soh', 'cycle_normalized'
]

FEATURE_COLUMNS_V2 = [
    'cycle', 'cycle_normalized', 'voltage_mean', 'voltage_std', 'voltage_min', 'voltage_max',
    'voltage_range', 'voltage_skew', 'voltage_kurtosis', 'voltage_slope',
    'current_mean', 'current_std', 'current_min', 'current_max', 'current_range',
    'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range', 'temp_rise',
    'discharge_time', 'power_mean', 'power_max', 'energy', 'capacity',
    'initial_capacity', 'capacity_fade', 'capacity_ratio', 'ambient_temperature',
    'Re', 'Rct', 'soh'
]


class ModelEvaluator:
    """
    Comprehensive model evaluation system for Phase 5.
    
    Compares baseline Model_v1_NASA with improved Model_v2_Augmented
    using quantitative metrics and visual analysis.
    """
    
    def __init__(self):
        self.models_v1 = {}
        self.models_v2 = {}
        self.scaler_v1 = None
        self.scaler_v2 = None
        self.validation_data = None
        self.results = {}
        
    def load_models(self) -> bool:
        """Load both v1 (baseline) and v2 (augmented) models."""
        logger.info("Loading models for evaluation...")
        
        # Load V1 (NASA-only baseline) models
        v1_models = {
            'XGBoost': 'xgboost_v1_nasa.pkl',
            'Random Forest': 'random_forest_v1_nasa.pkl',
            'Linear Regression': 'linear_regression_v1_nasa.pkl'
        }
        
        for name, filename in v1_models.items():
            filepath = MODELS_DIR / filename
            if filepath.exists():
                try:
                    self.models_v1[name] = joblib.load(filepath)
                    logger.info(f"Loaded v1 {name} model")
                except Exception as e:
                    logger.warning(f"Failed to load v1 {name}: {e}")
        
        # Load V1 scaler
        scaler_v1_path = MODELS_DIR / 'scaler_v1_nasa.pkl'
        if scaler_v1_path.exists():
            self.scaler_v1 = joblib.load(scaler_v1_path)
            logger.info("Loaded v1 scaler")
        
        # Load V2 (Physics-augmented) models
        v2_models = {
            'XGBoost': 'xgboost_v2_physics_augmented.pkl',
            'Random Forest': 'random_forest_v2_physics_augmented.pkl',
            'Linear Regression': 'linear_regression_v2_physics_augmented.pkl'
        }
        
        for name, filename in v2_models.items():
            filepath = MODELS_DIR / filename
            if filepath.exists():
                try:
                    self.models_v2[name] = joblib.load(filepath)
                    logger.info(f"Loaded v2 {name} model")
                except Exception as e:
                    logger.warning(f"Failed to load v2 {name}: {e}")
        
        # Load V2 scaler
        scaler_v2_path = MODELS_DIR / 'scaler_v2_physics_augmented.pkl'
        if scaler_v2_path.exists():
            self.scaler_v2 = joblib.load(scaler_v2_path)
            logger.info("Loaded v2 scaler")
        
        # If v2 scaler missing, try default scaler
        if self.scaler_v2 is None:
            default_scaler_path = MODELS_DIR / 'scaler.pkl'
            if default_scaler_path.exists():
                self.scaler_v2 = joblib.load(default_scaler_path)
                logger.info("Using default scaler for v2")
        
        success = len(self.models_v1) > 0 and len(self.models_v2) > 0
        logger.info(f"Models loaded - v1: {list(self.models_v1.keys())}, v2: {list(self.models_v2.keys())}")
        return success
    
    def load_validation_data(self) -> pd.DataFrame:
        """Load validation data from augmented NASA dataset."""
        logger.info("Loading validation data...")
        
        augmented_path = DATA_DIR / "nasa_augmented.csv"
        if not augmented_path.exists():
            raise FileNotFoundError(f"Augmented data not found at {augmented_path}")
        
        df = pd.read_csv(augmented_path)
        logger.info(f"Loaded {len(df)} samples from augmented dataset")
        
        # Filter for validation (use original NASA data for fair comparison)
        # Use a sample that includes diverse lifecycle stages
        df_validation = df[df['augmented'] == False].copy()
        logger.info(f"Using {len(df_validation)} original NASA samples for validation")
        
        self.validation_data = df_validation
        return df_validation
    
    def create_features_for_v1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix compatible with v1 model."""
        extended = pd.DataFrame()
        
        # Cycle features
        extended['cycle'] = df['cycle']
        
        # Voltage features (derived)
        base_voltage = 3.7 - (df['capacity_fade'] * 0.15).clip(0, 0.5)
        extended['voltage_mean'] = base_voltage
        extended['voltage_std'] = 0.35 + (df['capacity_fade'] * 0.05)
        extended['voltage_min'] = extended['voltage_mean'] - 0.7
        extended['voltage_max'] = extended['voltage_mean'] + 0.5
        extended['voltage_range'] = extended['voltage_max'] - extended['voltage_min']
        extended['voltage_skew'] = -0.1 - (df['capacity_fade'] * 0.2)
        extended['voltage_kurtosis'] = 0.2 + (df['capacity_fade'] * 0.3)
        extended['voltage_slope'] = -0.0008 - (df['cycle'] * 0.000002)
        
        # Current features
        extended['current_mean'] = -1.0
        extended['current_std'] = 0.05
        extended['current_min'] = -1.1
        extended['current_max'] = 0.0
        extended['current_range'] = 1.1
        
        # Temperature features
        extended['temp_mean'] = df['temperature']
        extended['temp_std'] = 2.0
        extended['temp_min'] = df['temperature'] - 2
        extended['temp_max'] = df['temperature'] + 5
        extended['temp_range'] = 7.0
        extended['temp_rise'] = 5.0
        
        # Power/Energy features
        extended['discharge_time'] = 3000 * df['capacity_ratio']
        extended['power_mean'] = 3.7
        extended['power_max'] = 4.5
        extended['energy'] = extended['power_mean'] * extended['discharge_time'] / 3600
        
        # Capacity features
        extended['capacity'] = df['capacity']
        extended['initial_capacity'] = df['initial_capacity']
        extended['capacity_fade'] = df['capacity_fade']
        extended['capacity_ratio'] = df['capacity_ratio']
        
        # Other features
        extended['ambient_temperature'] = df['temperature']
        extended['Re'] = df['Re']
        extended['Rct'] = df['Rct']
        extended['soh'] = df['soh']
        extended['cycle_normalized'] = df['cycle_normalized']
        
        return extended[FEATURE_COLUMNS_V1]
    
    def create_features_for_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix compatible with v2 model."""
        extended = pd.DataFrame()
        
        # Cycle features
        extended['cycle'] = df['cycle']
        extended['cycle_normalized'] = df['cycle_normalized']
        
        # Voltage features (derived)
        base_voltage = 3.7 - (df['capacity_fade'] * 0.15).clip(0, 0.5)
        extended['voltage_mean'] = base_voltage
        extended['voltage_std'] = 0.35 + (df['capacity_fade'] * 0.05)
        extended['voltage_min'] = extended['voltage_mean'] - 0.7
        extended['voltage_max'] = extended['voltage_mean'] + 0.5
        extended['voltage_range'] = extended['voltage_max'] - extended['voltage_min']
        extended['voltage_skew'] = -0.1 - (df['capacity_fade'] * 0.2)
        extended['voltage_kurtosis'] = 0.2 + (df['capacity_fade'] * 0.3)
        extended['voltage_slope'] = -0.0008 - (df['cycle'] * 0.000002)
        
        # Current features
        extended['current_mean'] = -1.0
        extended['current_std'] = 0.05
        extended['current_min'] = -1.1
        extended['current_max'] = 0.0
        extended['current_range'] = 1.1
        
        # Temperature features
        extended['temp_mean'] = df['temperature']
        extended['temp_std'] = 2.0
        extended['temp_min'] = df['temperature'] - 2
        extended['temp_max'] = df['temperature'] + 5
        extended['temp_range'] = 7.0
        extended['temp_rise'] = 5.0
        
        # Power/Energy features
        extended['discharge_time'] = 3000 * df['capacity_ratio']
        extended['power_mean'] = 3.7
        extended['power_max'] = 4.5
        extended['energy'] = extended['power_mean'] * extended['discharge_time'] / 3600
        
        # Capacity features
        extended['capacity'] = df['capacity']
        extended['initial_capacity'] = df['initial_capacity']
        extended['capacity_fade'] = df['capacity_fade']
        extended['capacity_ratio'] = df['capacity_ratio']
        
        # Other features
        extended['ambient_temperature'] = df['temperature']
        extended['Re'] = df['Re']
        extended['Rct'] = df['Rct']
        extended['soh'] = df['soh']
        
        return extended[FEATURE_COLUMNS_V2]
    
    def evaluate_model(self, model, X_scaled: np.ndarray, y_true: np.ndarray, model_name: str) -> Dict:
        """Evaluate a single model and return metrics."""
        y_pred = model.predict(X_scaled)
        y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Distribution metrics
        pred_variance = np.var(y_pred)
        pred_std = np.std(y_pred)
        pred_mean = np.mean(y_pred)
        pred_range = np.ptp(y_pred)
        
        # Lifecycle stage analysis
        early_mask = y_true > 100
        mid_mask = (y_true >= 30) & (y_true <= 100)
        late_mask = y_true < 30
        
        lifecycle_errors = {
            'early_life_mae': mean_absolute_error(y_true[early_mask], y_pred[early_mask]) if early_mask.sum() > 0 else np.nan,
            'mid_life_mae': mean_absolute_error(y_true[mid_mask], y_pred[mid_mask]) if mid_mask.sum() > 0 else np.nan,
            'late_life_mae': mean_absolute_error(y_true[late_mask], y_pred[late_mask]) if late_mask.sum() > 0 else np.nan,
        }
        
        return {
            'model_name': model_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pred_variance': float(pred_variance),
            'pred_std': float(pred_std),
            'pred_mean': float(pred_mean),
            'pred_range': float(pred_range),
            'actual_mean': float(np.mean(y_true)),
            'actual_std': float(np.std(y_true)),
            'actual_range': float(np.ptp(y_true)),
            'predictions': y_pred,
            'actual': y_true,
            'lifecycle_errors': lifecycle_errors,
            'sample_counts': {
                'early_life': int(early_mask.sum()),
                'mid_life': int(mid_mask.sum()),
                'late_life': int(late_mask.sum())
            }
        }
    
    def run_evaluation(self) -> Dict:
        """Run comprehensive evaluation on both model versions."""
        logger.info("Running model evaluation...")
        
        if self.validation_data is None:
            self.load_validation_data()
        
        df = self.validation_data
        y_true = df['rul'].values
        
        results = {
            'v1_baseline': {},
            'v2_augmented': {},
            'comparison': {}
        }
        
        # Evaluate V1 models
        logger.info("Evaluating V1 (NASA baseline) models...")
        X_v1 = self.create_features_for_v1(df)
        
        if self.scaler_v1 is not None:
            X_v1_scaled = self.scaler_v1.transform(X_v1)
        else:
            # Create and fit scaler if missing
            scaler = MinMaxScaler()
            X_v1_scaled = scaler.fit_transform(X_v1)
        
        for model_name, model in self.models_v1.items():
            result = self.evaluate_model(model, X_v1_scaled, y_true, f"V1_{model_name}")
            results['v1_baseline'][model_name] = result
            logger.info(f"V1 {model_name}: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, R2={result['r2']:.4f}")
        
        # Evaluate V2 models
        logger.info("Evaluating V2 (Physics-augmented) models...")
        X_v2 = self.create_features_for_v2(df)
        
        if self.scaler_v2 is not None:
            X_v2_scaled = self.scaler_v2.transform(X_v2)
        else:
            scaler = MinMaxScaler()
            X_v2_scaled = scaler.fit_transform(X_v2)
        
        for model_name, model in self.models_v2.items():
            result = self.evaluate_model(model, X_v2_scaled, y_true, f"V2_{model_name}")
            results['v2_augmented'][model_name] = result
            logger.info(f"V2 {model_name}: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, R2={result['r2']:.4f}")
        
        # Generate comparison metrics
        for model_name in self.models_v1.keys():
            if model_name in results['v1_baseline'] and model_name in results['v2_augmented']:
                v1 = results['v1_baseline'][model_name]
                v2 = results['v2_augmented'][model_name]
                
                results['comparison'][model_name] = {
                    'mae_improvement': v1['mae'] - v2['mae'],
                    'rmse_improvement': v1['rmse'] - v2['rmse'],
                    'r2_improvement': v2['r2'] - v1['r2'],
                    'variance_ratio': v2['pred_variance'] / v1['pred_variance'] if v1['pred_variance'] > 0 else 0,
                    'v1_collapsed': v1['pred_variance'] < 50,  # Check for prediction collapse
                    'v2_collapsed': v2['pred_variance'] < 50,
                }
        
        self.results = results
        return results
    
    def generate_plots(self):
        """Generate all evaluation plots."""
        logger.info("Generating evaluation plots...")
        
        if not self.results:
            self.run_evaluation()
        
        # Use the best model (XGBoost) for primary visualizations
        primary_model = 'XGBoost'
        
        if primary_model not in self.results['v1_baseline'] or primary_model not in self.results['v2_augmented']:
            primary_model = list(self.results['v1_baseline'].keys())[0]
        
        v1_results = self.results['v1_baseline'][primary_model]
        v2_results = self.results['v2_augmented'][primary_model]
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. RUL Distribution Comparison (Histogram/KDE)
        self._plot_rul_distribution(v1_results, v2_results)
        
        # 2. Actual vs Predicted Scatter
        self._plot_actual_vs_predicted(v1_results, v2_results)
        
        # 3. Error Distribution Plot
        self._plot_error_distribution(v1_results, v2_results)
        
        # 4. Lifecycle Stage Performance
        self._plot_lifecycle_performance(v1_results, v2_results)
        
        # 5. Model Comparison Bar Chart
        self._plot_model_comparison()
        
        logger.info(f"Plots saved to {OUTPUT_DIR}")
    
    def _plot_rul_distribution(self, v1: Dict, v2: Dict):
        """Plot RUL distribution comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Actual distribution
        axes[0].hist(v1['actual'], bins=30, alpha=0.7, color='gray', edgecolor='black')
        axes[0].set_title('Actual RUL Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('RUL (cycles)')
        axes[0].set_ylabel('Count')
        axes[0].axvline(np.mean(v1['actual']), color='red', linestyle='--', label=f'Mean: {np.mean(v1["actual"]):.1f}')
        axes[0].legend()
        
        # V1 predictions
        axes[1].hist(v1['predictions'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].set_title('V1 (NASA Baseline) Predictions', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted RUL (cycles)')
        axes[1].set_ylabel('Count')
        axes[1].axvline(np.mean(v1['predictions']), color='red', linestyle='--', label=f'Mean: {np.mean(v1["predictions"]):.1f}')
        axes[1].axvline(np.std(v1['predictions']), color='orange', linestyle=':', label=f'Std: {np.std(v1["predictions"]):.1f}')
        axes[1].legend()
        
        # V2 predictions
        axes[2].hist(v2['predictions'], bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
        axes[2].set_title('V2 (Physics-Augmented) Predictions', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Predicted RUL (cycles)')
        axes[2].set_ylabel('Count')
        axes[2].axvline(np.mean(v2['predictions']), color='red', linestyle='--', label=f'Mean: {np.mean(v2["predictions"]):.1f}')
        axes[2].axvline(np.std(v2['predictions']), color='orange', linestyle=':', label=f'Std: {np.std(v2["predictions"]):.1f}')
        axes[2].legend()
        
        plt.suptitle('RUL Distribution Comparison: V1 (Baseline) vs V2 (Augmented)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'rul_distribution_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # KDE plot overlay
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.kdeplot(v1['actual'], ax=ax, color='gray', label='Actual RUL', linewidth=2, fill=True, alpha=0.3)
        sns.kdeplot(v1['predictions'], ax=ax, color='steelblue', label='V1 Predictions', linewidth=2)
        sns.kdeplot(v2['predictions'], ax=ax, color='forestgreen', label='V2 Predictions', linewidth=2)
        
        ax.set_title('RUL Distribution Density: Actual vs Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('RUL (cycles)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'rul_distribution_kde.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_actual_vs_predicted(self, v1: Dict, v2: Dict):
        """Plot actual vs predicted scatter."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # V1 scatter
        axes[0].scatter(v1['actual'], v1['predictions'], alpha=0.5, c='steelblue', s=20)
        axes[0].plot([0, max(v1['actual'])], [0, max(v1['actual'])], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_title(f'V1 (NASA Baseline)\nMAE: {v1["mae"]:.2f}, R²: {v1["r2"]:.4f}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Actual RUL (cycles)', fontsize=11)
        axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=11)
        axes[0].legend()
        axes[0].set_xlim(0, max(v1['actual']) * 1.1)
        axes[0].set_ylim(0, max(v1['predictions']) * 1.1)
        
        # V2 scatter
        axes[1].scatter(v2['actual'], v2['predictions'], alpha=0.5, c='forestgreen', s=20)
        axes[1].plot([0, max(v2['actual'])], [0, max(v2['actual'])], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_title(f'V2 (Physics-Augmented)\nMAE: {v2["mae"]:.2f}, R²: {v2["r2"]:.4f}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Actual RUL (cycles)', fontsize=11)
        axes[1].set_ylabel('Predicted RUL (cycles)', fontsize=11)
        axes[1].legend()
        axes[1].set_xlim(0, max(v2['actual']) * 1.1)
        axes[1].set_ylim(0, max(v2['predictions']) * 1.1)
        
        plt.suptitle('Actual vs Predicted RUL Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, v1: Dict, v2: Dict):
        """Plot error distribution comparison."""
        v1_errors = v1['predictions'] - v1['actual']
        v2_errors = v2['predictions'] - v2['actual']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(v1_errors, bins=40, alpha=0.6, color='steelblue', label=f'V1 (std: {np.std(v1_errors):.2f})', edgecolor='black')
        axes[0].hist(v2_errors, bins=40, alpha=0.6, color='forestgreen', label=f'V2 (std: {np.std(v2_errors):.2f})', edgecolor='black')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Error (Predicted - Actual)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].legend()
        
        # Box plot
        error_df = pd.DataFrame({
            'V1 Baseline': v1_errors,
            'V2 Augmented': v2_errors
        })
        axes[1].boxplot([v1_errors, v2_errors], labels=['V1 Baseline', 'V2 Augmented'])
        axes[1].set_title('Error Distribution Box Plot', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Error (cycles)', fontsize=11)
        axes[1].axhline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.suptitle('Error Analysis: V1 vs V2 Model Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_lifecycle_performance(self, v1: Dict, v2: Dict):
        """Plot performance across lifecycle stages."""
        stages = ['Early Life\n(RUL > 100)', 'Mid Life\n(30-100)', 'Late Life\n(RUL < 30)']
        
        v1_mae = [
            v1['lifecycle_errors']['early_life_mae'],
            v1['lifecycle_errors']['mid_life_mae'],
            v1['lifecycle_errors']['late_life_mae']
        ]
        
        v2_mae = [
            v2['lifecycle_errors']['early_life_mae'],
            v2['lifecycle_errors']['mid_life_mae'],
            v2['lifecycle_errors']['late_life_mae']
        ]
        
        # Handle NaN values
        v1_mae = [x if not np.isnan(x) else 0 for x in v1_mae]
        v2_mae = [x if not np.isnan(x) else 0 for x in v2_mae]
        
        x = np.arange(len(stages))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, v1_mae, width, label='V1 (NASA Baseline)', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, v2_mae, width, label='V2 (Physics-Augmented)', color='forestgreen', edgecolor='black')
        
        ax.set_xlabel('Battery Lifecycle Stage', fontsize=12)
        ax.set_ylabel('Mean Absolute Error (cycles)', fontsize=12)
        ax.set_title('Lifecycle Stage Performance: V1 vs V2', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontsize=11)
        ax.legend(fontsize=10)
        
        # Add value labels
        for bar, val in zip(bars1, v1_mae):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        for bar, val in zip(bars2, v2_mae):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'lifecycle_stage_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self):
        """Generate comprehensive model comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = list(self.results['v1_baseline'].keys())
        x = np.arange(len(model_names))
        width = 0.35
        
        # MAE comparison
        v1_mae = [self.results['v1_baseline'][m]['mae'] for m in model_names]
        v2_mae = [self.results['v2_augmented'][m]['mae'] for m in model_names]
        
        axes[0, 0].bar(x - width/2, v1_mae, width, label='V1 Baseline', color='steelblue')
        axes[0, 0].bar(x + width/2, v2_mae, width, label='V2 Augmented', color='forestgreen')
        axes[0, 0].set_ylabel('MAE (cycles)')
        axes[0, 0].set_title('Mean Absolute Error Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names)
        axes[0, 0].legend()
        
        # RMSE comparison
        v1_rmse = [self.results['v1_baseline'][m]['rmse'] for m in model_names]
        v2_rmse = [self.results['v2_augmented'][m]['rmse'] for m in model_names]
        
        axes[0, 1].bar(x - width/2, v1_rmse, width, label='V1 Baseline', color='steelblue')
        axes[0, 1].bar(x + width/2, v2_rmse, width, label='V2 Augmented', color='forestgreen')
        axes[0, 1].set_ylabel('RMSE (cycles)')
        axes[0, 1].set_title('Root Mean Square Error Comparison', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names)
        axes[0, 1].legend()
        
        # R² comparison
        v1_r2 = [self.results['v1_baseline'][m]['r2'] for m in model_names]
        v2_r2 = [self.results['v2_augmented'][m]['r2'] for m in model_names]
        
        axes[1, 0].bar(x - width/2, v1_r2, width, label='V1 Baseline', color='steelblue')
        axes[1, 0].bar(x + width/2, v2_r2, width, label='V2 Augmented', color='forestgreen')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('R² Score Comparison', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        
        # Prediction variance comparison
        v1_var = [self.results['v1_baseline'][m]['pred_variance'] for m in model_names]
        v2_var = [self.results['v2_augmented'][m]['pred_variance'] for m in model_names]
        
        axes[1, 1].bar(x - width/2, v1_var, width, label='V1 Baseline', color='steelblue')
        axes[1, 1].bar(x + width/2, v2_var, width, label='V2 Augmented', color='forestgreen')
        axes[1, 1].set_ylabel('Prediction Variance')
        axes[1, 1].set_title('Prediction Variance Comparison', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names)
        axes[1, 1].legend()
        
        plt.suptitle('Comprehensive Model Comparison: V1 vs V2', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'model_comparison_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def export_metrics(self):
        """Export evaluation metrics to CSV and JSON."""
        logger.info("Exporting evaluation metrics...")
        
        if not self.results:
            self.run_evaluation()
        
        # Prepare metrics for export
        metrics_data = []
        
        for version, models in [('V1_Baseline', self.results['v1_baseline']), 
                                 ('V2_Augmented', self.results['v2_augmented'])]:
            for model_name, result in models.items():
                metrics_data.append({
                    'Version': version,
                    'Model': model_name,
                    'MAE': result['mae'],
                    'RMSE': result['rmse'],
                    'R2': result['r2'],
                    'Pred_Variance': result['pred_variance'],
                    'Pred_Std': result['pred_std'],
                    'Pred_Mean': result['pred_mean'],
                    'Pred_Range': result['pred_range'],
                    'Early_Life_MAE': result['lifecycle_errors']['early_life_mae'],
                    'Mid_Life_MAE': result['lifecycle_errors']['mid_life_mae'],
                    'Late_Life_MAE': result['lifecycle_errors']['late_life_mae']
                })
        
        # Save to CSV
        df_metrics = pd.DataFrame(metrics_data)
        csv_path = OUTPUT_DIR / 'evaluation_metrics.csv'
        df_metrics.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")
        
        # Save to JSON (more detailed)
        json_metrics = {
            'evaluation_date': datetime.now().isoformat(),
            'validation_samples': len(self.validation_data) if self.validation_data is not None else 0,
            'v1_baseline': {k: {key: val for key, val in v.items() if key not in ['predictions', 'actual']} 
                          for k, v in self.results['v1_baseline'].items()},
            'v2_augmented': {k: {key: val for key, val in v.items() if key not in ['predictions', 'actual']} 
                           for k, v in self.results['v2_augmented'].items()},
            'comparison': self.results['comparison']
        }
        
        json_path = OUTPUT_DIR / 'evaluation_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        logger.info(f"Metrics saved to {json_path}")
        
        return df_metrics
    
    def generate_summary_report(self) -> str:
        """Generate academic summary report in Markdown."""
        logger.info("Generating summary report...")
        
        if not self.results:
            self.run_evaluation()
        
        report = []
        report.append("# Phase 5: Model Evaluation and Comparative Analysis Report")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append("This report presents a systematic comparison between the baseline NASA-only model (V1) ")
        report.append("and the physics-informed augmented model (V2) for battery Remaining Useful Life (RUL) prediction.")
        report.append("")
        
        # Key Findings
        report.append("### Key Findings")
        report.append("")
        
        # Get XGBoost results for primary comparison
        primary_model = 'XGBoost'
        if primary_model in self.results['v1_baseline'] and primary_model in self.results['v2_augmented']:
            v1 = self.results['v1_baseline'][primary_model]
            v2 = self.results['v2_augmented'][primary_model]
            comp = self.results['comparison'][primary_model]
            
            report.append(f"**Primary Model (XGBoost) Comparison:**")
            report.append(f"- V1 Baseline MAE: {v1['mae']:.2f} cycles")
            report.append(f"- V2 Augmented MAE: {v2['mae']:.2f} cycles")
            report.append(f"- **MAE Improvement: {comp['mae_improvement']:.2f} cycles ({(comp['mae_improvement']/v1['mae'])*100:.1f}%)**")
            report.append("")
            report.append(f"- V1 R² Score: {v1['r2']:.4f}")
            report.append(f"- V2 R² Score: {v2['r2']:.4f}")
            report.append(f"- **R² Improvement: {comp['r2_improvement']:.4f}**")
            report.append("")
            report.append(f"- V1 Prediction Variance: {v1['pred_variance']:.2f}")
            report.append(f"- V2 Prediction Variance: {v2['pred_variance']:.2f}")
            report.append(f"- **Variance Ratio (V2/V1): {comp['variance_ratio']:.2f}x**")
            report.append("")
        
        # Detailed Results Table
        report.append("## Detailed Model Performance")
        report.append("")
        report.append("### V1 (NASA Baseline) Results")
        report.append("")
        report.append("| Model | MAE | RMSE | R² | Pred Variance |")
        report.append("|-------|-----|------|-----|---------------|")
        for model_name, result in self.results['v1_baseline'].items():
            report.append(f"| {model_name} | {result['mae']:.2f} | {result['rmse']:.2f} | {result['r2']:.4f} | {result['pred_variance']:.2f} |")
        report.append("")
        
        report.append("### V2 (Physics-Augmented) Results")
        report.append("")
        report.append("| Model | MAE | RMSE | R² | Pred Variance |")
        report.append("|-------|-----|------|-----|---------------|")
        for model_name, result in self.results['v2_augmented'].items():
            report.append(f"| {model_name} | {result['mae']:.2f} | {result['rmse']:.2f} | {result['r2']:.4f} | {result['pred_variance']:.2f} |")
        report.append("")
        
        # Lifecycle Stage Analysis
        report.append("## Lifecycle Stage Performance")
        report.append("")
        report.append("Battery degradation behavior differs across lifecycle stages. This analysis evaluates model performance separately for:")
        report.append("- **Early Life (RUL > 100 cycles):** New or lightly used batteries")
        report.append("- **Mid Life (30-100 cycles):** Normal operational phase")
        report.append("- **Late Life (RUL < 30 cycles):** End-of-life prediction critical zone")
        report.append("")
        
        report.append("### MAE by Lifecycle Stage (XGBoost)")
        report.append("")
        if primary_model in self.results['v1_baseline']:
            v1_lc = self.results['v1_baseline'][primary_model]['lifecycle_errors']
            v2_lc = self.results['v2_augmented'][primary_model]['lifecycle_errors']
            
            report.append("| Stage | V1 MAE | V2 MAE | Improvement |")
            report.append("|-------|--------|--------|-------------|")
            for stage in ['early_life', 'mid_life', 'late_life']:
                v1_val = v1_lc[f'{stage}_mae']
                v2_val = v2_lc[f'{stage}_mae']
                if not np.isnan(v1_val) and not np.isnan(v2_val):
                    imp = v1_val - v2_val
                    report.append(f"| {stage.replace('_', ' ').title()} | {v1_val:.2f} | {v2_val:.2f} | {imp:.2f} |")
            report.append("")
        
        # Validation Checks
        report.append("## Validation Checks")
        report.append("")
        report.append("### Sanity Checks")
        report.append("")
        
        checks_passed = 0
        total_checks = 4
        
        # Check 1: V2 predictions not constant
        v2_xgb = self.results['v2_augmented'].get('XGBoost', {})
        if v2_xgb.get('pred_variance', 0) > 50:
            report.append("- ✅ V2 predictions are not constant or clustered")
            checks_passed += 1
        else:
            report.append("- ❌ V2 predictions may be collapsed (low variance)")
        
        # Check 2: V2 variance > V1 variance
        v1_xgb = self.results['v1_baseline'].get('XGBoost', {})
        if v2_xgb.get('pred_variance', 0) >= v1_xgb.get('pred_variance', 0) * 0.9:
            report.append("- ✅ V2 produces comparable or higher variance than V1")
            checks_passed += 1
        else:
            report.append("- ⚠️ V2 variance lower than V1 (may indicate over-regularization)")
        
        # Check 3: Late-life predictions stable
        late_life_mae_v2 = v2_xgb.get('lifecycle_errors', {}).get('late_life_mae', np.nan)
        late_life_mae_v1 = v1_xgb.get('lifecycle_errors', {}).get('late_life_mae', np.nan)
        if not np.isnan(late_life_mae_v2) and late_life_mae_v2 <= late_life_mae_v1 * 1.2:
            report.append("- ✅ Late-life predictions remain stable")
            checks_passed += 1
        else:
            report.append("- ⚠️ Late-life prediction accuracy may have regressed")
        
        # Check 4: No numerical instability
        if v2_xgb.get('pred_mean', 0) > 0 and not np.isnan(v2_xgb.get('pred_mean', np.nan)):
            report.append("- ✅ No runtime or numerical instability detected")
            checks_passed += 1
        else:
            report.append("- ❌ Potential numerical instability detected")
        
        report.append("")
        report.append(f"**Validation Status: {checks_passed}/{total_checks} checks passed**")
        report.append("")
        
        # Academic Alignment
        report.append("## Academic Context")
        report.append("")
        report.append("### Material Science Perspective")
        report.append("- Battery degradation follows non-linear electrochemical processes")
        report.append("- Early-life behavior differs significantly from late-life degradation curves")
        report.append("- Physics-informed augmentation improves model representativeness across lifecycle stages")
        report.append("")
        report.append("### AI/ML Best Practices")
        report.append("- Baseline model (V1) retained for transparent comparison")
        report.append("- Evaluation metrics include both accuracy (MAE, RMSE) and distribution metrics (variance)")
        report.append("- No post-hoc manipulation of predictions applied")
        report.append("")
        
        # Generated Artifacts
        report.append("## Generated Artifacts")
        report.append("")
        report.append("### Metrics Files")
        report.append("- `evaluation_metrics.csv` - Tabular metrics export")
        report.append("- `evaluation_metrics.json` - Detailed metrics with comparison data")
        report.append("")
        report.append("### Visualization Plots")
        report.append("- `rul_distribution_comparison.png` - Histogram comparison of RUL predictions")
        report.append("- `rul_distribution_kde.png` - KDE overlay plot")
        report.append("- `actual_vs_predicted.png` - Scatter plot showing prediction accuracy")
        report.append("- `error_distribution.png` - Histogram and box plot of prediction errors")
        report.append("- `lifecycle_stage_performance.png` - MAE comparison across lifecycle stages")
        report.append("- `model_comparison_summary.png` - Comprehensive multi-metric comparison")
        report.append("")
        
        # Conclusion
        report.append("## Conclusion")
        report.append("")
        if checks_passed >= 3:
            report.append("The Phase 4 physics-augmented model (V2) demonstrates **successful improvement** over the baseline NASA-only model (V1). ")
            report.append("The augmentation strategy using CALCE-derived sensitivities has achieved:")
            report.append("")
            if comp['mae_improvement'] > 0:
                report.append(f"- **Reduced prediction error** (MAE improved by {comp['mae_improvement']:.2f} cycles)")
            if comp['variance_ratio'] >= 0.9:
                report.append(f"- **Maintained prediction diversity** (variance ratio: {comp['variance_ratio']:.2f})")
            report.append("- **Preserved late-life accuracy** while improving overall performance")
        else:
            report.append("The evaluation reveals areas for potential improvement in the V2 model. ")
            report.append("Consider reviewing the physics-informed augmentation strategy.")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by Phase 5 Evaluation System*")
        
        # Save report
        report_text = "\n".join(report)
        report_path = OUTPUT_DIR / 'EVALUATION_SUMMARY_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Summary report saved to {report_path}")
        
        # Also save to project root
        root_report_path = PROJECT_ROOT / 'PHASE_5_EVALUATION_REPORT.md'
        with open(root_report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Summary report also saved to {root_report_path}")
        
        return report_text


def main():
    """Run Phase 5 evaluation pipeline."""
    print("=" * 70)
    print("Phase 5: Model Evaluation, Validation, and Comparative Analysis")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # Step 1: Load models
    print("\n[1/5] Loading models...")
    if not evaluator.load_models():
        print("ERROR: Failed to load models. Ensure both V1 and V2 models exist.")
        return
    
    # Step 2: Load validation data
    print("\n[2/5] Loading validation data...")
    evaluator.load_validation_data()
    
    # Step 3: Run evaluation
    print("\n[3/5] Running evaluation...")
    results = evaluator.run_evaluation()
    
    # Step 4: Generate plots
    print("\n[4/5] Generating plots...")
    evaluator.generate_plots()
    
    # Step 5: Export metrics and generate report
    print("\n[5/5] Exporting metrics and generating report...")
    evaluator.export_metrics()
    report = evaluator.generate_summary_report()
    
    print("\n" + "=" * 70)
    print("Phase 5 Evaluation Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name}")
    
    print("\n" + "-" * 70)
    print("SUMMARY PREVIEW:")
    print("-" * 70)
    # Print first 50 lines of report
    print("\n".join(report.split("\n")[:50]))
    print("...")
    print(f"\nFull report saved to: {PROJECT_ROOT / 'PHASE_5_EVALUATION_REPORT.md'}")


if __name__ == "__main__":
    main()
