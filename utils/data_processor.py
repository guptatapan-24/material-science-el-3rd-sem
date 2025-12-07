import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import requests
import os
import logging
from scipy.signal import savgol_filter
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NASA_DATASET_URL = "https://data.nasa.gov/resource/yjcy-ig6g.csv?$limit=50000"
LOCAL_DATA_PATH = "/app/data/nasa_battery_data.csv"
SAMPLE_DATA_PATH = "/app/data/sample_battery_data.csv"

@st.cache_data(show_spinner=False)
def load_nasa_dataset():
    """Load NASA battery dataset with automatic download fallback."""
    try:
        # Try loading from local cache first
        if os.path.exists(LOCAL_DATA_PATH):
            logger.info("Loading dataset from local cache...")
            df = pd.read_csv(LOCAL_DATA_PATH)
            return df
        
        # Try downloading from NASA
        logger.info("Downloading NASA battery dataset...")
        response = requests.get(NASA_DATASET_URL, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(pd.io.common.BytesIO(response.content))
        
        # Save to local cache
        os.makedirs(os.path.dirname(LOCAL_DATA_PATH), exist_ok=True)
        df.to_csv(LOCAL_DATA_PATH, index=False)
        logger.info(f"Dataset downloaded and cached: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.warning(f"Failed to download NASA dataset: {e}")
        # Fallback to sample data
        if os.path.exists(SAMPLE_DATA_PATH):
            logger.info("Using sample dataset...")
            return pd.read_csv(SAMPLE_DATA_PATH)
        else:
            # Generate synthetic sample data
            logger.info("Generating synthetic sample data...")
            return generate_sample_data()

def generate_sample_data(n_batteries=4, n_cycles=200):
    """Generate synthetic battery data for demonstration."""
    data = []
    
    for battery_id in range(1, n_batteries + 1):
        initial_capacity = np.random.uniform(1.8, 2.0)
        
        for cycle in range(1, n_cycles + 1):
            # Simulate capacity fade
            fade_rate = np.random.uniform(0.0005, 0.001)
            capacity = initial_capacity * (1 - fade_rate * cycle) + np.random.normal(0, 0.02)
            
            # Simulate voltage, current, temperature
            voltage = np.random.uniform(3.2, 4.2)
            current = np.random.uniform(-2.0, 2.0)
            temperature = np.random.uniform(20, 45)
            
            data.append({
                'battery_id': f'B{battery_id:04d}',
                'cycle': cycle,
                'voltage_measured': voltage,
                'current_measured': current,
                'temperature_measured': temperature,
                'capacity': capacity
            })
    
    df = pd.DataFrame(data)
    # Save sample data
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df

def preprocess_data(df):
    """Preprocess battery data with cleaning and validation."""
    try:
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove outliers using IQR method
        for col in ['voltage_measured', 'current_measured', 'temperature_measured', 'capacity']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Apply Savitzky-Golay filter for noise reduction (simpler than CEEMDAN)
        if 'capacity' in df.columns and len(df) > 10:
            window_length = min(11, len(df) if len(df) % 2 == 1 else len(df) - 1)
            df['capacity_smooth'] = savgol_filter(df['capacity'], window_length, 3)
        
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return df

def engineer_features(df):
    """Engineer 40+ features from raw battery data."""
    features = pd.DataFrame()
    
    # Group by battery_id if available
    if 'battery_id' in df.columns:
        grouped = df.groupby('battery_id')
    else:
        grouped = [(None, df)]
    
    all_features = []
    
    for battery_id, group in grouped:
        group = group.sort_values('cycle').reset_index(drop=True)
        
        # Basic features
        feat = {
            'battery_id': battery_id,
            'cycle_count': len(group),
            'voltage_mean': group['voltage_measured'].mean() if 'voltage_measured' in group else 0,
            'voltage_std': group['voltage_measured'].std() if 'voltage_measured' in group else 0,
            'voltage_min': group['voltage_measured'].min() if 'voltage_measured' in group else 0,
            'voltage_max': group['voltage_measured'].max() if 'voltage_measured' in group else 0,
            'voltage_range': group['voltage_measured'].max() - group['voltage_measured'].min() if 'voltage_measured' in group else 0,
            
            'current_mean': group['current_measured'].mean() if 'current_measured' in group else 0,
            'current_std': group['current_measured'].std() if 'current_measured' in group else 0,
            'current_min': group['current_measured'].min() if 'current_measured' in group else 0,
            'current_max': group['current_measured'].max() if 'current_measured' in group else 0,
            'current_range': group['current_measured'].max() - group['current_measured'].min() if 'current_measured' in group else 0,
            
            'temp_mean': group['temperature_measured'].mean() if 'temperature_measured' in group else 0,
            'temp_std': group['temperature_measured'].std() if 'temperature_measured' in group else 0,
            'temp_min': group['temperature_measured'].min() if 'temperature_measured' in group else 0,
            'temp_max': group['temperature_measured'].max() if 'temperature_measured' in group else 0,
            'temp_range': group['temperature_measured'].max() - group['temperature_measured'].min() if 'temperature_measured' in group else 0,
        }
        
        # Capacity-based features
        if 'capacity' in group.columns:
            feat['capacity_initial'] = group['capacity'].iloc[0]
            feat['capacity_current'] = group['capacity'].iloc[-1]
            feat['capacity_mean'] = group['capacity'].mean()
            feat['capacity_std'] = group['capacity'].std()
            feat['capacity_min'] = group['capacity'].min()
            feat['capacity_fade'] = group['capacity'].iloc[0] - group['capacity'].iloc[-1]
            
            # Fade rate
            if len(group) > 1:
                feat['fade_rate'] = feat['capacity_fade'] / len(group)
            else:
                feat['fade_rate'] = 0
            
            # RUL calculation (cycles until 80% capacity)
            threshold = feat['capacity_initial'] * 0.8
            feat['rul'] = max(0, (feat['capacity_current'] - threshold) / feat['fade_rate']) if feat['fade_rate'] > 0 else 1000
        
        # Power and energy features
        if 'voltage_measured' in group and 'current_measured' in group:
            group['power'] = group['voltage_measured'] * group['current_measured']
            feat['power_mean'] = group['power'].mean()
            feat['power_max'] = group['power'].max()
            feat['energy_total'] = group['power'].sum()
        
        # Internal resistance estimation
        if 'voltage_measured' in group and 'current_measured' in group:
            if group['current_measured'].std() > 0:
                feat['internal_resistance'] = group['voltage_measured'].std() / group['current_measured'].std()
            else:
                feat['internal_resistance'] = 0
        
        # Statistical moments
        for col in ['voltage_measured', 'current_measured', 'temperature_measured']:
            if col in group:
                feat[f'{col}_skew'] = group[col].skew()
                feat[f'{col}_kurtosis'] = group[col].kurtosis()
        
        # Trend features
        if len(group) > 2:
            feat['voltage_trend'] = np.polyfit(range(len(group)), group['voltage_measured'], 1)[0] if 'voltage_measured' in group else 0
            feat['temp_trend'] = np.polyfit(range(len(group)), group['temperature_measured'], 1)[0] if 'temperature_measured' in group else 0
            if 'capacity' in group:
                feat['capacity_trend'] = np.polyfit(range(len(group)), group['capacity'], 1)[0]
        
        all_features.append(feat)
    
    features_df = pd.DataFrame(all_features)
    
    # Fill any remaining NaN values
    features_df = features_df.fillna(0)
    
    return features_df

def prepare_training_data(df):
    """Prepare data for model training."""
    features_df = engineer_features(df)
    
    # Separate features and target
    target_col = 'rul'
    if target_col not in features_df.columns:
        raise ValueError("RUL target not found in features")
    
    X = features_df.drop(columns=[target_col, 'battery_id'], errors='ignore')
    y = features_df[target_col]
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler, X.columns.tolist()

def create_user_input_features(voltage, current, temperature, cycle_count, capacity=None):
    """Create feature vector from user inputs."""
    # Create a simple feature dict
    feat = {
        'cycle_count': cycle_count,
        'voltage_mean': voltage,
        'voltage_std': 0.1,
        'voltage_min': voltage - 0.2,
        'voltage_max': voltage + 0.2,
        'voltage_range': 0.4,
        'current_mean': current,
        'current_std': 0.1,
        'current_min': current - 0.5,
        'current_max': current + 0.5,
        'current_range': 1.0,
        'temp_mean': temperature,
        'temp_std': 2.0,
        'temp_min': temperature - 5,
        'temp_max': temperature + 5,
        'temp_range': 10,
        'capacity_initial': capacity if capacity else 2.0,
        'capacity_current': capacity if capacity else 1.8,
        'capacity_mean': capacity if capacity else 1.9,
        'capacity_std': 0.1,
        'capacity_min': capacity - 0.2 if capacity else 1.6,
        'capacity_fade': 0.2,
        'fade_rate': 0.001,
        'power_mean': voltage * abs(current),
        'power_max': voltage * abs(current) * 1.2,
        'energy_total': voltage * abs(current) * cycle_count,
        'internal_resistance': 0.05,
        'voltage_measured_skew': 0,
        'voltage_measured_kurtosis': 0,
        'current_measured_skew': 0,
        'current_measured_kurtosis': 0,
        'temperature_measured_skew': 0,
        'temperature_measured_kurtosis': 0,
        'voltage_trend': -0.001,
        'temp_trend': 0.01,
        'capacity_trend': -0.001,
    }
    
    return pd.DataFrame([feat])