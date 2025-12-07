import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def create_shap_explainer(_model, X_train, model_name='XGBoost'):
    """Create SHAP explainer for the model."""
    try:
        if model_name == 'LSTM':
            # For LSTM, use sampling
            X_sample = X_train.sample(min(100, len(X_train)))
            X_reshaped = X_sample.values.reshape((X_sample.shape[0], 1, X_sample.shape[1]))
            explainer = shap.KernelExplainer(
                lambda x: _model.predict(x.reshape((x.shape[0], 1, x.shape[1])), verbose=0),
                X_reshaped
            )
        else:
            # For tree-based models
            explainer = shap.TreeExplainer(_model)
        
        return explainer
    except Exception as e:
        logger.error(f"Error creating SHAP explainer: {e}")
        return None

def get_shap_values(explainer, X, model_name='XGBoost'):
    """Calculate SHAP values for given input."""
    try:
        if model_name == 'LSTM':
            X_reshaped = X.values.reshape((X.shape[0], 1, X.shape[1]))
            shap_values = explainer.shap_values(X_reshaped)
        else:
            shap_values = explainer.shap_values(X)
        
        return shap_values
    except Exception as e:
        logger.error(f"Error calculating SHAP values: {e}")
        return None

def plot_shap_summary(explainer, X, feature_names):
    """Create SHAP summary plot."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_type='bar')
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating SHAP summary plot: {e}")
        return None

def plot_shap_force(explainer, X, feature_names, index=0):
    """Create SHAP force plot for a single prediction."""
    try:
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
        
        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            shap_values[index],
            X.iloc[index],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        
        return force_plot
    except Exception as e:
        logger.error(f"Error creating SHAP force plot: {e}")
        return None

def get_feature_importance(explainer, X, feature_names, top_n=10):
    """Get top N most important features."""
    try:
        shap_values = explainer.shap_values(X)
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return None