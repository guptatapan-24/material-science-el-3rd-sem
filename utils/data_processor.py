import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging
from scipy.signal import savgol_filter
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths for NASA dataset
NASA_DATASET_PATH = "/app/nasa_dataset/cleaned_dataset"
METADATA_PATH = f"{NASA_DATASET_PATH}/metadata.csv"
DATA_FOLDER_PATH = f"{NASA_DATASET_PATH}/data"
PROCESSED_DATA_PATH = "/app/data/processed_battery_data.csv"
SAMPLE_DATA_PATH = "/app/data/sample_battery_data.csv"

# Batteries with good data quality (sufficient cycles and clear degradation)
QUALITY_BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018', 'B0042', 'B0043', 'B0044', 
                     'B0045', 'B0046', 'B0047', 'B0048']


@st.cache_data(show_spinner=False)
def load_nasa_dataset():
    """Load NASA battery dataset from extracted archive."""
    try:
        # Check if processed data exists
        if os.path.exists(PROCESSED_DATA_PATH):
            logger.info("Loading processed NASA dataset...")
            df = pd.read_csv(PROCESSED_DATA_PATH)
            logger.info(f"Loaded {len(df)} rows from processed data")
            return df
        
        # Load from raw NASA data
        if os.path.exists(METADATA_PATH):
            logger.info("Processing NASA battery dataset...")
            df = process_nasa_raw_data()
            if df is not None and len(df) > 0:
                # Save processed data for faster loading
                os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
                df.to_csv(PROCESSED_DATA_PATH, index=False)
                logger.info(f"Processed and saved {len(df)} rows")
                return df
        
        # Fallback to sample data
        logger.warning("NASA dataset not found, using sample data")
        if os.path.exists(SAMPLE_DATA_PATH):
            return pd.read_csv(SAMPLE_DATA_PATH)
        else:
            return generate_sample_data()
            
    except Exception as e:
        logger.error(f"Error loading NASA dataset: {e}")
        return generate_sample_data()


def process_nasa_raw_data():
    """Process raw NASA dataset into a structured format for ML."""
    try:
        # Load metadata
        metadata = pd.read_csv(METADATA_PATH)
        
        # Convert numeric columns
        metadata['Capacity'] = pd.to_numeric(metadata['Capacity'], errors='coerce')
        metadata['Re'] = pd.to_numeric(metadata['Re'], errors='coerce')
        metadata['Rct'] = pd.to_numeric(metadata['Rct'], errors='coerce')
        
        # Filter to discharge cycles only (they have capacity measurements)
        discharge_df = metadata[metadata['type'] == 'discharge'].copy()
        
        # Focus on batteries with good data quality
        discharge_df = discharge_df[discharge_df['battery_id'].isin(QUALITY_BATTERIES)]
        
        # Process each battery to create training samples
        all_data = []
        
        for battery_id in discharge_df['battery_id'].unique():
            battery_data = discharge_df[discharge_df['battery_id'] == battery_id].copy()
            battery_data = battery_data.sort_values('test_id').reset_index(drop=True)
            
            # Filter out invalid capacity measurements
            battery_data = battery_data[battery_data['Capacity'] > 0.5]
            
            if len(battery_data) < 10:  # Need at least 10 cycles
                continue
            
            # Get initial capacity (first valid measurement)
            initial_capacity = battery_data['Capacity'].iloc[:5].mean()
            
            # Calculate 80% threshold (End of Life)
            eol_threshold = initial_capacity * 0.8
            
            # Find total cycles until EOL (or use actual cycles if battery reached EOL)
            capacities = battery_data['Capacity'].values
            eol_cycle = None
            for i, cap in enumerate(capacities):
                if cap <= eol_threshold:
                    eol_cycle = i
                    break
            
            if eol_cycle is None:
                # Battery didn't reach EOL, estimate based on trend
                if len(capacities) > 5:
                    # Linear extrapolation
                    x = np.arange(len(capacities))
                    coeffs = np.polyfit(x, capacities, 1)
                    if coeffs[0] < 0:  # Degrading
                        # Estimate cycle when capacity reaches threshold
                        eol_cycle = int((eol_threshold - coeffs[1]) / coeffs[0])
                        eol_cycle = max(eol_cycle, len(capacities))
                    else:
                        eol_cycle = len(capacities) * 2  # Conservative estimate
                else:
                    eol_cycle = len(capacities) * 2
            
            # Create samples at different points in battery life
            for i in range(len(battery_data)):
                row = battery_data.iloc[i]
                current_cycle = i + 1
                current_capacity = row['Capacity']
                
                # Skip invalid data
                if pd.isna(current_capacity) or current_capacity <= 0:
                    continue
                
                # Calculate RUL
                rul = max(0, eol_cycle - current_cycle)
                
                # Load cycle-level time series data for feature extraction
                filename = row['filename']
                cycle_data_path = os.path.join(DATA_FOLDER_PATH, filename)
                
                if os.path.exists(cycle_data_path):
                    cycle_features = extract_cycle_features(cycle_data_path)
                else:
                    cycle_features = get_default_cycle_features()
                
                # Combine all features
                sample = {
                    'battery_id': battery_id,
                    'cycle': current_cycle,
                    'capacity': current_capacity,
                    'initial_capacity': initial_capacity,
                    'capacity_fade': initial_capacity - current_capacity,
                    'capacity_ratio': current_capacity / initial_capacity,
                    'ambient_temperature': row['ambient_temperature'],
                    'Re': row['Re'] if pd.notna(row['Re']) else 0,
                    'Rct': row['Rct'] if pd.notna(row['Rct']) else 0,
                    'rul': rul,
                    **cycle_features
                }
                
                all_data.append(sample)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Processed {len(df)} samples from {df['battery_id'].nunique()} batteries")
        return df
        
    except Exception as e:
        logger.error(f"Error processing NASA data: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_cycle_features(cycle_data_path):
    """Extract features from individual cycle time-series data."""
    try:
        df = pd.read_csv(cycle_data_path)
        
        features = {}
        
        # Voltage features
        if 'Voltage_measured' in df.columns:
            v = df['Voltage_measured']
            features['voltage_mean'] = v.mean()
            features['voltage_std'] = v.std()
            features['voltage_min'] = v.min()
            features['voltage_max'] = v.max()
            features['voltage_range'] = v.max() - v.min()
            features['voltage_skew'] = v.skew() if len(v) > 3 else 0
            features['voltage_kurtosis'] = v.kurtosis() if len(v) > 4 else 0
            # Voltage drop rate (slope)
            if len(v) > 10:
                x = np.arange(len(v))
                coeffs = np.polyfit(x, v.values, 1)
                features['voltage_slope'] = coeffs[0]
            else:
                features['voltage_slope'] = 0
        else:
            features.update(get_default_voltage_features())
        
        # Current features
        if 'Current_measured' in df.columns:
            c = df['Current_measured']
            features['current_mean'] = c.mean()
            features['current_std'] = c.std()
            features['current_min'] = c.min()
            features['current_max'] = c.max()
            features['current_range'] = c.max() - c.min()
        else:
            features.update(get_default_current_features())
        
        # Temperature features
        if 'Temperature_measured' in df.columns:
            t = df['Temperature_measured']
            features['temp_mean'] = t.mean()
            features['temp_std'] = t.std()
            features['temp_min'] = t.min()
            features['temp_max'] = t.max()
            features['temp_range'] = t.max() - t.min()
            # Temperature rise during discharge
            if len(t) > 10:
                features['temp_rise'] = t.iloc[-10:].mean() - t.iloc[:10].mean()
            else:
                features['temp_rise'] = 0
        else:
            features.update(get_default_temp_features())
        
        # Time-based features
        if 'Time' in df.columns:
            features['discharge_time'] = df['Time'].max() - df['Time'].min()
        else:
            features['discharge_time'] = 0
        
        # Power features
        if 'Voltage_measured' in df.columns and 'Current_measured' in df.columns:
            power = df['Voltage_measured'] * np.abs(df['Current_measured'])
            features['power_mean'] = power.mean()
            features['power_max'] = power.max()
            features['energy'] = power.sum() * (features.get('discharge_time', 1) / len(df)) if len(df) > 0 else 0
        else:
            features['power_mean'] = 0
            features['power_max'] = 0
            features['energy'] = 0
        
        return features
        
    except Exception as e:
        logger.warning(f"Error extracting features from {cycle_data_path}: {e}")
        return get_default_cycle_features()


def get_default_cycle_features():
    """Return default cycle features when data is unavailable."""
    return {
        **get_default_voltage_features(),
        **get_default_current_features(),
        **get_default_temp_features(),
        'discharge_time': 0,
        'power_mean': 0,
        'power_max': 0,
        'energy': 0
    }


def get_default_voltage_features():
    return {
        'voltage_mean': 3.5,
        'voltage_std': 0.3,
        'voltage_min': 2.5,
        'voltage_max': 4.2,
        'voltage_range': 1.7,
        'voltage_skew': 0,
        'voltage_kurtosis': 0,
        'voltage_slope': 0
    }


def get_default_current_features():
    return {
        'current_mean': -1.0,
        'current_std': 0.1,
        'current_min': -1.0,
        'current_max': 0,
        'current_range': 1.0
    }


def get_default_temp_features():
    return {
        'temp_mean': 25,
        'temp_std': 5,
        'temp_min': 20,
        'temp_max': 35,
        'temp_range': 15,
        'temp_rise': 0
    }


def generate_sample_data(n_batteries=4, n_cycles=200):
    """Generate synthetic battery data for demonstration."""
    np.random.seed(42)
    data = []
    
    for battery_id in range(1, n_batteries + 1):
        initial_capacity = np.random.uniform(1.8, 2.0)
        
        # Different degradation rates for different batteries
        base_fade_rate = np.random.uniform(0.002, 0.004)
        
        for cycle in range(1, n_cycles + 1):
            # Non-linear capacity fade
            fade = base_fade_rate * cycle + 0.00001 * cycle**1.5
            capacity = initial_capacity * (1 - fade) + np.random.normal(0, 0.01)
            capacity = max(0.5, capacity)  # Floor at 0.5
            
            # Simulate voltage, current, temperature
            voltage = 3.5 + np.random.normal(0, 0.3)
            current = -1.0 + np.random.normal(0, 0.05)
            temperature = 25 + cycle * 0.02 + np.random.normal(0, 2)
            
            # Calculate RUL based on 80% threshold
            threshold = initial_capacity * 0.8
            if capacity > threshold:
                rul = int((capacity - threshold) / (base_fade_rate * initial_capacity))
            else:
                rul = 0
            
            data.append({
                'battery_id': f'B{battery_id:04d}',
                'cycle': cycle,
                'voltage_mean': voltage,
                'voltage_std': 0.1 + np.random.uniform(0, 0.1),
                'voltage_min': voltage - 0.5,
                'voltage_max': voltage + 0.5,
                'voltage_range': 1.0,
                'voltage_skew': np.random.normal(0, 0.1),
                'voltage_kurtosis': np.random.normal(0, 0.1),
                'voltage_slope': -0.001 * cycle / 100,
                'current_mean': current,
                'current_std': 0.05,
                'current_min': current - 0.1,
                'current_max': 0,
                'current_range': abs(current),
                'temp_mean': temperature,
                'temp_std': 2.0,
                'temp_min': temperature - 5,
                'temp_max': temperature + 5,
                'temp_range': 10,
                'temp_rise': np.random.uniform(2, 8),
                'discharge_time': 3000 + np.random.uniform(-500, 500),
                'power_mean': voltage * abs(current),
                'power_max': voltage * abs(current) * 1.2,
                'energy': voltage * abs(current) * 3000,
                'capacity': capacity,
                'initial_capacity': initial_capacity,
                'capacity_fade': initial_capacity - capacity,
                'capacity_ratio': capacity / initial_capacity,
                'ambient_temperature': 24,
                'Re': 0.05 + cycle * 0.0001,
                'Rct': 0.15 + cycle * 0.0002,
                'rul': max(0, rul)
            })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return df


def preprocess_data(df):
    """Preprocess battery data with cleaning and validation."""
    try:
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Remove extreme outliers using IQR method
        critical_cols = ['capacity', 'voltage_mean', 'temp_mean', 'rul']
        for col in critical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.05)
                Q3 = df[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Ensure RUL is non-negative
        if 'rul' in df.columns:
            df['rul'] = df['rul'].clip(lower=0)
        
        logger.info(f"Preprocessed data: {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return df


def engineer_features(df):
    """Engineer additional features from raw battery data."""
    df = df.copy()
    
    # Calculate derived features if not present
    if 'capacity_ratio' not in df.columns and 'capacity' in df.columns and 'initial_capacity' in df.columns:
        df['capacity_ratio'] = df['capacity'] / df['initial_capacity']
    
    if 'capacity_fade' not in df.columns and 'capacity' in df.columns and 'initial_capacity' in df.columns:
        df['capacity_fade'] = df['initial_capacity'] - df['capacity']
    
    # State of Health (SOH)
    if 'capacity' in df.columns and 'initial_capacity' in df.columns:
        df['soh'] = (df['capacity'] / df['initial_capacity']) * 100
    
    # Cycle-normalized features
    if 'cycle' in df.columns:
        max_cycle = df['cycle'].max()
        df['cycle_normalized'] = df['cycle'] / max_cycle if max_cycle > 0 else 0
    
    return df


def prepare_training_data(df):
    """Prepare data for model training."""
    df = preprocess_data(df)
    df = engineer_features(df)
    
    # Define feature columns (exclude target and identifiers)
    exclude_cols = ['rul', 'battery_id', 'filename']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    # Ensure RUL exists
    if 'rul' not in df.columns:
        raise ValueError("RUL target not found in data")
    
    X = df[feature_cols].copy()
    y = df['rul'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    logger.info(f"Prepared training data: {len(X_scaled)} samples, {len(feature_cols)} features")
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"RUL range: {y.min():.0f} - {y.max():.0f}")
    
    return X_scaled, y, scaler, feature_cols


def create_user_input_features(voltage, current, temperature, cycle_count, capacity=None):
    """Create feature vector from user inputs matching training features.
    
    This function creates features that align with the NASA battery dataset characteristics.
    The NASA dataset shows batteries degrade from ~2.0Ah to ~1.2Ah over ~50-170 cycles.
    """
    # Use provided capacity or estimate based on cycle (NASA data shows degradation pattern)
    initial_capacity = 2.0  # Typical initial capacity for NASA dataset batteries
    
    if capacity is not None:
        current_capacity = capacity
    else:
        # Estimate capacity based on cycle if not provided
        # Approximate degradation: ~0.002 Ah per cycle for NASA batteries
        current_capacity = max(1.2, initial_capacity - (cycle_count * 0.004))
    
    # Calculate capacity metrics
    capacity_fade = initial_capacity - current_capacity
    capacity_ratio = current_capacity / initial_capacity
    
    # Estimate impedance based on cycle and degradation (from NASA data patterns)
    # Re and Rct increase with aging
    re_base = 0.055  # Base internal resistance
    rct_base = 0.18  # Base charge transfer resistance
    Re = re_base + (cycle_count * 0.00015) + (capacity_fade * 0.01)
    Rct = rct_base + (cycle_count * 0.0003) + (capacity_fade * 0.02)
    
    # Calculate derived voltage features (based on NASA discharge curves)
    voltage_range = 1.7  # Typical range from 4.2V to 2.5V
    voltage_min = max(2.5, voltage - (voltage_range / 2))
    voltage_max = min(4.2, voltage + (voltage_range / 2))
    
    # Voltage slope becomes more negative with aging
    voltage_slope = -0.0008 - (cycle_count * 0.000002)
    
    # Voltage skew and kurtosis change with battery health
    voltage_skew = -0.1 - (capacity_fade * 0.2)
    voltage_kurtosis = 0.2 + (capacity_fade * 0.3)
    
    # Current features (discharge current, typically negative)
    current_abs = abs(current)
    
    # Temperature features (temperature rise during discharge)
    temp_rise = 3 + (current_abs * 2)  # Higher current = more heat
    temp_min = temperature - 2
    temp_max = temperature + temp_rise
    temp_range = temp_max - temp_min
    
    # Power and energy calculations
    power_mean = voltage * current_abs
    power_max = voltage_max * current_abs * 1.1
    
    # Discharge time inversely related to capacity fade
    discharge_time = 3000 * capacity_ratio  # Shorter discharge as capacity fades
    energy = power_mean * discharge_time / 3600  # Wh
    
    feat = {
        # Cycle information
        'cycle': cycle_count,
        'cycle_normalized': min(cycle_count / 168.0, 1.0),  # Normalized to max cycles in NASA data
        
        # Capacity features (most important for RUL prediction)
        'capacity': current_capacity,
        'initial_capacity': initial_capacity,
        'capacity_fade': capacity_fade,
        'capacity_ratio': capacity_ratio,
        'soh': capacity_ratio * 100,  # State of Health percentage
        
        # Environmental
        'ambient_temperature': temperature,
        
        # Impedance (important degradation indicators)
        'Re': Re,
        'Rct': Rct,
        
        # Voltage features
        'voltage_mean': voltage,
        'voltage_std': 0.35 + (capacity_fade * 0.1),  # More variation with age
        'voltage_min': voltage_min,
        'voltage_max': voltage_max,
        'voltage_range': voltage_range,
        'voltage_skew': voltage_skew,
        'voltage_kurtosis': voltage_kurtosis,
        'voltage_slope': voltage_slope,
        
        # Current features
        'current_mean': current,  # Keep sign for direction
        'current_std': 0.05 + (abs(current) * 0.02),
        'current_min': current if current < 0 else current - 0.1,
        'current_max': 0 if current < 0 else current,
        'current_range': current_abs + 0.1,
        
        # Temperature features
        'temp_mean': temperature + (temp_rise / 2),
        'temp_std': 1.5 + (temp_rise * 0.2),
        'temp_min': temp_min,
        'temp_max': temp_max,
        'temp_range': temp_range,
        'temp_rise': temp_rise,
        
        # Time and power
        'discharge_time': discharge_time,
        'power_mean': power_mean,
        'power_max': power_max,
        'energy': energy,
    }
    
    return pd.DataFrame([feat])
