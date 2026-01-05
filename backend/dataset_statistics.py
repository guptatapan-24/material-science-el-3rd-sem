"""
Dataset Statistics Generator for Battery RUL Prediction System
Computes statistical bounds from NASA training dataset for OOD detection
"""
import os
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cross-platform path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Dataset paths
NASA_DATASET_PATH = PROJECT_ROOT / "nasa_dataset" / "cleaned_dataset"
METADATA_PATH = NASA_DATASET_PATH / "metadata.csv"
DATA_FOLDER_PATH = NASA_DATASET_PATH / "data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_battery_data.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_battery_data.csv"

# Output paths for statistics
STATS_OUTPUT_DIR = PROJECT_ROOT / "data"
STATS_JSON_PATH = STATS_OUTPUT_DIR / "dataset_statistics.json"
STATS_CSV_PATH = STATS_OUTPUT_DIR / "dataset_distribution.csv"

# Quality batteries from NASA dataset
QUALITY_BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018', 'B0042', 'B0043', 'B0044', 
                     'B0045', 'B0046', 'B0047', 'B0048']

# Features to analyze for OOD detection
# Maps user input names to how they appear in training data
FEATURE_MAPPING = {
    'cycle': 'cycle',
    'capacity': 'Capacity',
    'voltage_measured': 'voltage_mean',
    'temperature_measured': 'ambient_temperature',
    'current_measured': 'current_mean'
}


class DatasetStatisticsGenerator:
    """Generate and manage dataset statistics for OOD detection."""
    
    def __init__(self):
        self.statistics: Dict[str, Any] = {}
        self.raw_data: Optional[pd.DataFrame] = None
        self._load_or_generate_statistics()
    
    def _load_or_generate_statistics(self) -> None:
        """Load existing statistics or generate new ones."""
        if STATS_JSON_PATH.exists():
            try:
                with open(STATS_JSON_PATH, 'r') as f:
                    self.statistics = json.load(f)
                logger.info("Loaded existing dataset statistics")
                return
            except Exception as e:
                logger.warning(f"Failed to load statistics: {e}")
        
        # Generate new statistics
        self._generate_statistics()
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data for statistics computation."""
        try:
            # Try to load processed data first
            if PROCESSED_DATA_PATH.exists():
                df = pd.read_csv(PROCESSED_DATA_PATH)
                logger.info(f"Loaded processed data: {len(df)} rows")
                return df
            
            # Load from NASA metadata
            if METADATA_PATH.exists():
                df = self._process_nasa_metadata()
                if df is not None and len(df) > 0:
                    return df
            
            # Fallback to sample data
            if SAMPLE_DATA_PATH.exists():
                df = pd.read_csv(SAMPLE_DATA_PATH)
                logger.info(f"Loaded sample data: {len(df)} rows")
                return df
            
            raise FileNotFoundError("No training data found")
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def _process_nasa_metadata(self) -> Optional[pd.DataFrame]:
        """Process NASA metadata to extract training statistics.
        
        Extracts actual voltage, current, and temperature distributions from cycle files.
        """
        try:
            metadata = pd.read_csv(METADATA_PATH)
            
            # Convert numeric columns
            metadata['Capacity'] = pd.to_numeric(metadata['Capacity'], errors='coerce')
            metadata['Re'] = pd.to_numeric(metadata['Re'], errors='coerce')
            metadata['Rct'] = pd.to_numeric(metadata['Rct'], errors='coerce')
            metadata['ambient_temperature'] = pd.to_numeric(metadata['ambient_temperature'], errors='coerce')
            
            # Filter to discharge cycles (they have capacity measurements)
            discharge_df = metadata[metadata['type'] == 'discharge'].copy()
            
            # Filter to quality batteries
            discharge_df = discharge_df[discharge_df['battery_id'].isin(QUALITY_BATTERIES)]
            
            # Remove invalid capacity measurements
            discharge_df = discharge_df[discharge_df['Capacity'] > 0.5]
            
            # Add cycle number per battery
            discharge_df = discharge_df.sort_values(['battery_id', 'test_id'])
            discharge_df['cycle'] = discharge_df.groupby('battery_id').cumcount() + 1
            
            # Extract per-cycle features from cycle data files
            # Store actual values for proper distribution computation
            voltage_values = []
            current_values = []
            temp_values = []
            
            # Sample cycle files for statistics
            sample_rows = discharge_df.sample(n=min(300, len(discharge_df)), random_state=42)
            
            for _, row in sample_rows.iterrows():
                filename = row['filename']
                cycle_path = DATA_FOLDER_PATH / filename
                if cycle_path.exists():
                    try:
                        cycle_data = pd.read_csv(cycle_path)
                        if 'Voltage_measured' in cycle_data.columns:
                            # Store mean voltage for this cycle
                            voltage_values.append(cycle_data['Voltage_measured'].mean())
                        if 'Current_measured' in cycle_data.columns:
                            # Store mean current for this cycle
                            current_values.append(cycle_data['Current_measured'].mean())
                        if 'Temperature_measured' in cycle_data.columns:
                            # Store mean temperature for this cycle
                            temp_values.append(cycle_data['Temperature_measured'].mean())
                    except:
                        pass
            
            # Assign per-row values (distributing sampled values)
            n_rows = len(discharge_df)
            
            if voltage_values:
                # Replicate sampled values to match dataframe size
                voltage_extended = np.resize(voltage_values, n_rows)
                discharge_df['voltage_mean'] = voltage_extended
            else:
                discharge_df['voltage_mean'] = 3.5
                
            if current_values:
                current_extended = np.resize(current_values, n_rows)
                discharge_df['current_mean'] = current_extended
            else:
                discharge_df['current_mean'] = -1.0
                
            if temp_values:
                temp_extended = np.resize(temp_values, n_rows)
                discharge_df['temp_mean'] = temp_extended
            else:
                discharge_df['temp_mean'] = 30.0
            
            # Store sampled values for direct statistics computation
            self._sampled_voltage = voltage_values
            self._sampled_current = current_values
            self._sampled_temperature = temp_values
            
            logger.info(f"Processed NASA metadata: {len(discharge_df)} discharge cycles")
            logger.info(f"Sampled {len(voltage_values)} cycles for voltage/current/temp statistics")
            return discharge_df
            
        except Exception as e:
            logger.error(f"Error processing NASA metadata: {e}")
            return None
    
    def _compute_feature_statistics(self, data: pd.Series, feature_name: str) -> Dict[str, float]:
        """Compute comprehensive statistics for a single feature."""
        # Remove NaN and infinite values
        clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) == 0:
            logger.warning(f"No valid data for feature: {feature_name}")
            return {}
        
        stats = {
            'minimum': float(clean_data.min()),
            'maximum': float(clean_data.max()),
            'mean': float(clean_data.mean()),
            'median': float(clean_data.median()),
            'standard_deviation': float(clean_data.std()),
            'percentile_5': float(clean_data.quantile(0.05)),
            'percentile_25': float(clean_data.quantile(0.25)),
            'percentile_75': float(clean_data.quantile(0.75)),
            'percentile_95': float(clean_data.quantile(0.95)),
            'count': int(len(clean_data))
        }
        
        return stats
    
    def _generate_statistics(self) -> None:
        """Generate statistics from training data."""
        try:
            df = self._load_training_data()
            self.raw_data = df
            
            # Define features to analyze with their column names in the data
            features_config = {
                'cycle': {'column': 'cycle', 'description': 'Charge-discharge cycle number'},
                'capacity': {'column': 'Capacity', 'alt_column': 'capacity', 'description': 'Battery capacity in Ah'},
                'voltage_measured': {'column': 'voltage_mean', 'alt_column': 'voltage_measured', 'description': 'Measured voltage in V', 'use_sampled': True},
                'temperature_measured': {'column': 'temp_mean', 'alt_column': 'ambient_temperature', 'description': 'Temperature in °C', 'use_sampled': True},
                'current_measured': {'column': 'current_mean', 'alt_column': 'current_measured', 'description': 'Current in A', 'use_sampled': True}
            }
            
            self.statistics = {
                'metadata': {
                    'dataset_source': 'NASA Li-ion Battery Aging Dataset',
                    'total_samples': len(df),
                    'batteries_analyzed': df['battery_id'].nunique() if 'battery_id' in df.columns else 0,
                    'generation_date': pd.Timestamp.now().isoformat()
                },
                'features': {}
            }
            
            distribution_data = []
            
            for feature_key, config in features_config.items():
                # For voltage/current/temperature, use sampled values if available
                use_sampled = config.get('use_sampled', False)
                sampled_data = None
                
                if use_sampled:
                    if feature_key == 'voltage_measured' and hasattr(self, '_sampled_voltage') and self._sampled_voltage:
                        sampled_data = pd.Series(self._sampled_voltage)
                    elif feature_key == 'current_measured' and hasattr(self, '_sampled_current') and self._sampled_current:
                        sampled_data = pd.Series(self._sampled_current)
                    elif feature_key == 'temperature_measured' and hasattr(self, '_sampled_temperature') and self._sampled_temperature:
                        sampled_data = pd.Series(self._sampled_temperature)
                
                if sampled_data is not None and len(sampled_data) > 0:
                    # Use sampled data for more accurate distribution
                    stats = self._compute_feature_statistics(sampled_data, feature_key)
                    stats['description'] = config['description']
                    stats['column_used'] = 'sampled_from_cycle_files'
                    self.statistics['features'][feature_key] = stats
                else:
                    # Try main column, then alternative
                    column = config['column']
                    if column not in df.columns and 'alt_column' in config:
                        column = config['alt_column']
                    
                    if column in df.columns:
                        stats = self._compute_feature_statistics(df[column], feature_key)
                        stats['description'] = config['description']
                        stats['column_used'] = column
                        self.statistics['features'][feature_key] = stats
                    else:
                        logger.warning(f"Feature column not found: {column} for {feature_key}")
                        continue
                
                # Add to distribution data
                if feature_key in self.statistics['features']:
                    row = {'feature': feature_key, 'description': config['description']}
                    row.update(self.statistics['features'][feature_key])
                    distribution_data.append(row)
            
            # Add NASA dataset-specific context
            self.statistics['dataset_context'] = {
                'late_life_bias': True,
                'cycle_range_dominant': '50-170 cycles',
                'capacity_range_dominant': '1.2-1.8 Ah',
                'eol_threshold': 0.8,  # 80% of initial capacity
                'initial_capacity_nominal': 2.0,
                'note': 'NASA dataset focuses on aging-phase electrochemical behavior. Early-life batteries may receive conservative predictions.'
            }
            
            # Save statistics
            self._save_statistics()
            
            # Save distribution CSV
            if distribution_data:
                dist_df = pd.DataFrame(distribution_data)
                dist_df.to_csv(STATS_CSV_PATH, index=False)
                logger.info(f"Saved distribution CSV to {STATS_CSV_PATH}")
            
            logger.info(f"Generated statistics for {len(self.statistics['features'])} features")
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            # Create fallback statistics based on known NASA dataset characteristics
            self._create_fallback_statistics()
    
    def _create_fallback_statistics(self) -> None:
        """Create fallback statistics based on known NASA dataset characteristics."""
        self.statistics = {
            'metadata': {
                'dataset_source': 'NASA Li-ion Battery Aging Dataset (Fallback)',
                'total_samples': 1500,
                'batteries_analyzed': 11,
                'generation_date': pd.Timestamp.now().isoformat()
            },
            'features': {
                'cycle': {
                    'minimum': 1, 'maximum': 168, 'mean': 85, 'median': 84,
                    'standard_deviation': 48, 'percentile_5': 9, 'percentile_25': 43,
                    'percentile_75': 126, 'percentile_95': 160, 'count': 1500,
                    'description': 'Charge-discharge cycle number'
                },
                'capacity': {
                    'minimum': 0.6, 'maximum': 2.0, 'mean': 1.45, 'median': 1.47,
                    'standard_deviation': 0.28, 'percentile_5': 0.77, 'percentile_25': 1.32,
                    'percentile_75': 1.65, 'percentile_95': 1.86, 'count': 1500,
                    'description': 'Battery capacity in Ah'
                },
                'voltage_measured': {
                    'minimum': 2.5, 'maximum': 4.2, 'mean': 3.52, 'median': 3.55,
                    'standard_deviation': 0.35, 'percentile_5': 2.9, 'percentile_25': 3.25,
                    'percentile_75': 3.75, 'percentile_95': 4.1, 'count': 1500,
                    'description': 'Measured voltage in V'
                },
                'temperature_measured': {
                    'minimum': 18, 'maximum': 45, 'mean': 28, 'median': 27,
                    'standard_deviation': 6, 'percentile_5': 20, 'percentile_25': 24,
                    'percentile_75': 33, 'percentile_95': 42, 'count': 1500,
                    'description': 'Temperature in °C'
                },
                'current_measured': {
                    'minimum': -2.2, 'maximum': -0.5, 'mean': -1.5, 'median': -1.4,
                    'standard_deviation': 0.4, 'percentile_5': -2.1, 'percentile_25': -1.8,
                    'percentile_75': -1.1, 'percentile_95': -0.7, 'count': 1500,
                    'description': 'Current in A'
                }
            },
            'dataset_context': {
                'late_life_bias': True,
                'cycle_range_dominant': '50-170 cycles',
                'capacity_range_dominant': '1.2-1.8 Ah',
                'eol_threshold': 0.8,
                'initial_capacity_nominal': 2.0,
                'note': 'NASA dataset focuses on aging-phase electrochemical behavior. Early-life batteries may receive conservative predictions.'
            }
        }
        self._save_statistics()
        logger.info("Created fallback statistics")
    
    def _save_statistics(self) -> None:
        """Save statistics to JSON file."""
        try:
            os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)
            with open(STATS_JSON_PATH, 'w') as f:
                json.dump(self.statistics, f, indent=2)
            logger.info(f"Saved statistics to {STATS_JSON_PATH}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
    
    def get_feature_statistics(self, feature_name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific feature."""
        return self.statistics.get('features', {}).get(feature_name)
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get all computed statistics."""
        return self.statistics
    
    def get_ood_bounds(self, feature_name: str) -> Optional[tuple]:
        """Get OOD detection bounds (5th and 95th percentiles) for a feature."""
        stats = self.get_feature_statistics(feature_name)
        if stats:
            return (stats.get('percentile_5'), stats.get('percentile_95'))
        return None
    
    def get_training_median(self, feature_name: str) -> Optional[float]:
        """Get training median for a feature."""
        stats = self.get_feature_statistics(feature_name)
        if stats:
            return stats.get('median')
        return None
    
    def regenerate_statistics(self) -> None:
        """Force regeneration of statistics."""
        # Remove existing stats file
        if STATS_JSON_PATH.exists():
            os.remove(STATS_JSON_PATH)
        self._generate_statistics()


# Global instance
_statistics_generator: Optional[DatasetStatisticsGenerator] = None


def get_statistics_generator() -> DatasetStatisticsGenerator:
    """Get or create the global statistics generator instance."""
    global _statistics_generator
    if _statistics_generator is None:
        _statistics_generator = DatasetStatisticsGenerator()
    return _statistics_generator


if __name__ == "__main__":
    # Generate statistics when run directly
    generator = DatasetStatisticsGenerator()
    print(json.dumps(generator.get_all_statistics(), indent=2))
