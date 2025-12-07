import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def plot_capacity_fade_curve(cycles, predicted_rul, initial_capacity=2.0):
    """Plot predicted capacity fade curve."""
    # Generate future cycles
    future_cycles = np.arange(0, cycles + predicted_rul, 10)
    
    # Simulate capacity fade
    fade_rate = (initial_capacity - initial_capacity * 0.8) / predicted_rul
    capacity = initial_capacity - fade_rate * future_cycles
    capacity = np.maximum(capacity, initial_capacity * 0.7)  # Floor at 70%
    
    # Create plot
    fig = go.Figure()
    
    # Capacity curve
    fig.add_trace(go.Scatter(
        x=future_cycles,
        y=capacity,
        mode='lines',
        name='Predicted Capacity',
        line=dict(color='#FF4B4B', width=3)
    ))
    
    # 80% threshold line
    fig.add_hline(
        y=initial_capacity * 0.8,
        line_dash="dash",
        line_color="orange",
        annotation_text="80% Capacity Threshold",
        annotation_position="right"
    )
    
    # Current cycle marker
    fig.add_vline(
        x=cycles,
        line_dash="dot",
        line_color="green",
        annotation_text="Current Cycle",
        annotation_position="top"
    )
    
    # RUL marker
    fig.add_vline(
        x=cycles + predicted_rul,
        line_dash="dot",
        line_color="red",
        annotation_text=f"EOL (RUL: {predicted_rul:.0f} cycles)",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Battery Capacity Fade Prediction",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity (Ah)",
        template="plotly_dark",
        hovermode='x unified',
        height=500
    )
    
    return fig

def plot_model_comparison(metrics_dict):
    """Plot comparison of model metrics."""
    # Prepare data
    models = list(metrics_dict.keys())
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    r2_values = [metrics_dict[m]['R2'] for m in models]
    
    # Create subplots
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='MAE',
        x=models,
        y=mae_values,
        marker_color='#FF4B4B'
    ))
    
    fig.add_trace(go.Bar(
        name='RMSE',
        x=models,
        y=rmse_values,
        marker_color='#FFA500'
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Error (cycles)",
        template="plotly_dark",
        barmode='group',
        height=400
    )
    
    return fig

def plot_r2_comparison(metrics_dict):
    """Plot R² comparison."""
    models = list(metrics_dict.keys())
    r2_values = [metrics_dict[m]['R2'] for m in models]
    
    fig = go.Figure(go.Bar(
        x=models,
        y=r2_values,
        marker_color='#00CC96',
        text=[f"{r2:.3f}" for r2 in r2_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="R² Score Comparison",
        xaxis_title="Model",
        yaxis_title="R² Score",
        template="plotly_dark",
        height=400
    )
    
    return fig

def plot_feature_importance(importance_df):
    """Plot feature importance from SHAP or model."""
    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker_color='#AB63FA'
    ))
    
    fig.update_layout(
        title="Top Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        template="plotly_dark",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def plot_what_if_comparison(base_rul, scenarios):
    """Plot what-if scenario comparisons."""
    scenario_names = list(scenarios.keys())
    rul_values = [scenarios[s] for s in scenario_names]
    deltas = [rul - base_rul for rul in rul_values]
    
    colors = ['green' if d > 0 else 'red' for d in deltas]
    
    fig = go.Figure(go.Bar(
        x=scenario_names,
        y=deltas,
        marker_color=colors,
        text=[f"{d:+.0f}" for d in deltas],
        textposition='auto'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    
    fig.update_layout(
        title="What-If Scenario Analysis (Change in RUL)",
        xaxis_title="Scenario",
        yaxis_title="RUL Change (cycles)",
        template="plotly_dark",
        height=400
    )
    
    return fig

def plot_sustainability_impact(rul_improvement):
    """Visualize sustainability impact of RUL prediction."""
    # Calculate metrics
    ewaste_reduction = rul_improvement * 0.5  # kg per battery
    co2_reduction = rul_improvement * 2.3  # kg CO2
    cost_savings = rul_improvement * 15  # USD
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=ewaste_reduction,
        title={"text": "E-Waste Reduction (kg)"},
        delta={'reference': 0, 'valueformat': '.1f'},
        domain={'x': [0, 0.33], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=co2_reduction,
        title={"text": "CO2 Reduction (kg)"},
        delta={'reference': 0, 'valueformat': '.1f'},
        domain={'x': [0.34, 0.66], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=cost_savings,
        title={"text": "Cost Savings (USD)"},
        delta={'reference': 0, 'valueformat': '.0f'},
        domain={'x': [0.67, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=250
    )
    
    return fig

def create_metrics_table(metrics_dict):
    """Create formatted metrics table."""
    df = pd.DataFrame(metrics_dict).T
    df = df.round(2)
    return df