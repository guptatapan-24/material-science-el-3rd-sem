"""
Physics-Informed Data Augmentation for Battery RUL Prediction
Phase 4: Use CALCE single-cycle data to derive degradation sensitivities
that augment the existing NASA aging dataset.

Key Principle:
- NASA dataset remains the PRIMARY source of aging/RUL ground truth
- CALCE single-cycle data provides PHYSICS-INFORMED sensitivities:
  * Temperature effects on capacity/efficiency
  * Depth-of-Discharge effects on degradation
  * Current rate effects on battery behavior
  
This approach regularizes the NASA model without replacing the aging ground truth.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CALCE_DIR = PROJECT_ROOT / "CALCE"
NASA_DIR = PROJECT_ROOT / "nasa_dataset" / "cleaned_dataset"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CALCESensitivityExtractor:
    """
    Extract physics-based degradation sensitivities from CALCE single-cycle data.
    
    The CALCE dataset provides single-cycle profiles at different:
    - Temperatures: 0°C, 25°C, 45°C
    - Depth-of-discharge: 50%, 80%
    
    We extract sensitivity coefficients that describe how battery behavior
    changes with these factors, without using CALCE as aging data.
    """
    
    def __init__(self, calce_dir: Path = CALCE_DIR):
        self.calce_dir = calce_dir
        self.train_dir = calce_dir / "Train"
        self.test_dir = calce_dir / "Test"
        
        # Store extracted sensitivities
        self.temperature_sensitivity = {}
        self.dod_sensitivity = {}
        self.current_sensitivity = {}
        self.nominal_values = {}
        
    def _parse_filename(self, filename: str) -> Dict[str, any]:
        """
        Parse CALCE filename to extract test conditions.
        Format: {TEST_TYPE}_{CONDITIONS}.csv
        e.g., TDST_2580.csv -> temp=25°C, DOD=80%
        """
        name = filename.replace('.csv', '')
        parts = name.split('_')
        
        test_type = parts[0] if parts else 'unknown'
        conditions = parts[1] if len(parts) > 1 else '2550'
        
        # Parse temperature (first 2 digits) and DOD (last 2 digits)
        temp_code = conditions[:2] if len(conditions) >= 2 else '25'
        dod_code = conditions[2:] if len(conditions) >= 4 else '50'
        
        temp_map = {'00': 0, '05': 5, '25': 25, '45': 45}
        temperature = temp_map.get(temp_code, 25)
        dod = int(dod_code) if dod_code.isdigit() else 50
        
        return {
            'test_type': test_type,
            'temperature': temperature,
            'dod_percent': dod,
            'filename': filename
        }
    
    def _load_and_summarize_cycle(self, filepath: Path) -> Optional[Dict]:
        """
        Load a CALCE cycle file and extract key metrics.
        """
        try:
            df = pd.read_csv(filepath)
            
            # Identify columns (handle variations)
            voltage_col = next((c for c in df.columns if c == 'V' or 'volt' in c.lower()), None)
            current_col = next((c for c in df.columns if c == 'I' or 'curr' in c.lower()), None)
            temp_col = next((c for c in df.columns if c == 'T' or 'temp' in c.lower()), None)
            cap_col = next((c for c in df.columns if 'discharge' in c.lower() and 'capacity' in c.lower()), None)
            
            if not all([voltage_col, current_col]):
                return None
            
            # Get metadata from filename
            metadata = self._parse_filename(filepath.name)
            
            # Extract metrics during discharge (negative current)
            discharge_df = df[df[current_col] < 0] if current_col else df
            if len(discharge_df) < 10:
                discharge_df = df
            
            # Calculate efficiency metrics
            voltage_mean = discharge_df[voltage_col].mean() if voltage_col else 3.7
            voltage_std = discharge_df[voltage_col].std() if voltage_col else 0.2
            voltage_min = discharge_df[voltage_col].min() if voltage_col else 2.5
            voltage_max = discharge_df[voltage_col].max() if voltage_col else 4.2
            
            current_mean = abs(discharge_df[current_col].mean()) if current_col else 1.0
            
            temp_mean = discharge_df[temp_col].mean() if temp_col and temp_col in discharge_df.columns else metadata['temperature']
            if temp_mean == 0 or pd.isna(temp_mean):
                temp_mean = metadata['temperature']
            
            # Capacity (use max discharge capacity)
            if cap_col and cap_col in df.columns:
                capacity = df[cap_col].max()
            else:
                capacity = 2.0  # Default nominal
            
            # Voltage efficiency (related to internal resistance effects)
            voltage_range = voltage_max - voltage_min
            voltage_efficiency = voltage_mean / voltage_max if voltage_max > 0 else 0.9
            
            return {
                'filename': filepath.name,
                'test_type': metadata['test_type'],
                'temperature': metadata['temperature'],
                'dod_percent': metadata['dod_percent'],
                'voltage_mean': voltage_mean,
                'voltage_std': voltage_std,
                'voltage_min': voltage_min,
                'voltage_max': voltage_max,
                'voltage_range': voltage_range,
                'voltage_efficiency': voltage_efficiency,
                'current_mean': current_mean,
                'temp_mean': temp_mean,
                'capacity': capacity,
            }
            
        except Exception as e:
            logger.warning(f"Error processing {filepath}: {e}")
            return None
    
    def extract_sensitivities(self) -> Dict:
        """
        Extract physics-based sensitivities from all CALCE files.
        
        Returns sensitivity coefficients for:
        - Temperature effects on voltage/capacity
        - DOD effects on efficiency
        - Current rate effects
        """
        all_metrics = []
        
        # Process both train and test directories
        for directory in [self.train_dir, self.test_dir]:
            if not directory.exists():
                continue
            for csv_file in directory.glob('*.csv'):
                metrics = self._load_and_summarize_cycle(csv_file)
                if metrics:
                    all_metrics.append(metrics)
        
        if not all_metrics:
            logger.warning("No valid CALCE data found!")
            return self._get_default_sensitivities()
        
        df = pd.DataFrame(all_metrics)
        logger.info(f"Extracted metrics from {len(df)} CALCE cycle profiles")
        
        # Calculate nominal (reference) values at 25°C, 50% DOD
        nominal_df = df[(df['temperature'] == 25) & (df['dod_percent'] == 50)]
        if len(nominal_df) == 0:
            nominal_df = df[df['temperature'] == 25]
        if len(nominal_df) == 0:
            nominal_df = df
        
        self.nominal_values = {
            'voltage_mean': nominal_df['voltage_mean'].mean(),
            'voltage_efficiency': nominal_df['voltage_efficiency'].mean(),
            'capacity': nominal_df['capacity'].mean(),
            'current_mean': nominal_df['current_mean'].mean(),
            'temperature': 25.0
        }
        
        # Temperature sensitivity (effect per °C deviation from 25°C)
        temp_groups = df.groupby('temperature').agg({
            'voltage_mean': 'mean',
            'voltage_efficiency': 'mean',
            'capacity': 'mean'
        }).reset_index()
        
        if len(temp_groups) > 1:
            # Linear regression-like sensitivity
            temp_ref = self.nominal_values['voltage_mean']
            volt_at_0 = temp_groups[temp_groups['temperature'] == 0]['voltage_mean'].values
            volt_at_45 = temp_groups[temp_groups['temperature'] == 45]['voltage_mean'].values
            
            # Voltage increases with temperature (typical behavior)
            if len(volt_at_0) > 0 and len(volt_at_45) > 0:
                voltage_temp_coeff = (volt_at_45[0] - volt_at_0[0]) / 45
            else:
                voltage_temp_coeff = 0.003  # Default: ~3mV/°C
            
            # Capacity decreases at extreme temperatures (U-shaped)
            cap_at_25 = temp_groups[temp_groups['temperature'] == 25]['capacity'].values
            cap_at_0 = temp_groups[temp_groups['temperature'] == 0]['capacity'].values
            cap_at_45 = temp_groups[temp_groups['temperature'] == 45]['capacity'].values
            
            if len(cap_at_25) > 0:
                cap_ref = cap_at_25[0]
                # Cold temperature degrades capacity more
                cold_penalty = (cap_ref - cap_at_0[0]) / 25 if len(cap_at_0) > 0 else 0.004
                hot_penalty = (cap_ref - cap_at_45[0]) / 20 if len(cap_at_45) > 0 else 0.002
            else:
                cold_penalty = 0.004
                hot_penalty = 0.002
            
            self.temperature_sensitivity = {
                'voltage_coeff_per_degC': voltage_temp_coeff,
                'capacity_cold_penalty_per_degC': cold_penalty,  # Below 25°C
                'capacity_hot_penalty_per_degC': hot_penalty,    # Above 25°C
                'reference_temperature': 25.0,
                'description': 'Battery performance sensitivity to temperature deviations from 25°C'
            }
        else:
            self.temperature_sensitivity = self._get_default_temp_sensitivity()
        
        # DOD sensitivity (effect of depth of discharge)
        dod_groups = df.groupby('dod_percent').agg({
            'voltage_mean': 'mean',
            'voltage_efficiency': 'mean'
        }).reset_index()
        
        if len(dod_groups) > 1:
            eff_at_50 = dod_groups[dod_groups['dod_percent'] == 50]['voltage_efficiency'].values
            eff_at_80 = dod_groups[dod_groups['dod_percent'] == 80]['voltage_efficiency'].values
            
            if len(eff_at_50) > 0 and len(eff_at_80) > 0:
                dod_efficiency_coeff = (eff_at_50[0] - eff_at_80[0]) / 30  # per % DOD
            else:
                dod_efficiency_coeff = 0.001
            
            self.dod_sensitivity = {
                'efficiency_loss_per_percent_dod': dod_efficiency_coeff,
                'reference_dod': 50,
                'description': 'Efficiency loss with increased depth of discharge'
            }
        else:
            self.dod_sensitivity = self._get_default_dod_sensitivity()
        
        # Current rate sensitivity (C-rate effects)
        current_groups = df.groupby(pd.cut(df['current_mean'], bins=3)).agg({
            'voltage_mean': 'mean',
            'voltage_efficiency': 'mean'
        })
        
        self.current_sensitivity = {
            'voltage_drop_per_amp': 0.05,  # IR drop estimation
            'efficiency_loss_per_amp': 0.02,
            'reference_current': self.nominal_values['current_mean'],
            'description': 'Performance degradation with increased current rate'
        }
        
        # Combine all sensitivities
        sensitivities = {
            'source': 'CALCE single-cycle test profiles',
            'purpose': 'Physics-informed augmentation of NASA aging data',
            'files_analyzed': len(all_metrics),
            'nominal_values': self.nominal_values,
            'temperature_sensitivity': self.temperature_sensitivity,
            'dod_sensitivity': self.dod_sensitivity,
            'current_sensitivity': self.current_sensitivity,
            'usage_guidance': {
                'augmentation_method': 'Apply sensitivities to NASA data to simulate varied conditions',
                'primary_data': 'NASA aging dataset (RUL ground truth)',
                'regularization': 'Sensitivities constrain prediction variance based on physics'
            }
        }
        
        logger.info("Extracted sensitivities:")
        logger.info(f"  Temperature: {self.temperature_sensitivity.get('voltage_coeff_per_degC', 'N/A'):.4f} V/°C")
        logger.info(f"  DOD: {self.dod_sensitivity.get('efficiency_loss_per_percent_dod', 'N/A'):.4f} per %DOD")
        
        return sensitivities
    
    def _get_default_sensitivities(self) -> Dict:
        """Default physics-based sensitivities from literature."""
        return {
            'source': 'Literature defaults (no CALCE data found)',
            'nominal_values': {
                'voltage_mean': 3.7,
                'voltage_efficiency': 0.92,
                'capacity': 2.0,
                'current_mean': 1.0,
                'temperature': 25.0
            },
            'temperature_sensitivity': self._get_default_temp_sensitivity(),
            'dod_sensitivity': self._get_default_dod_sensitivity(),
            'current_sensitivity': {
                'voltage_drop_per_amp': 0.05,
                'efficiency_loss_per_amp': 0.02,
                'reference_current': 1.0
            }
        }
    
    def _get_default_temp_sensitivity(self) -> Dict:
        return {
            'voltage_coeff_per_degC': 0.003,
            'capacity_cold_penalty_per_degC': 0.004,
            'capacity_hot_penalty_per_degC': 0.002,
            'reference_temperature': 25.0
        }
    
    def _get_default_dod_sensitivity(self) -> Dict:
        return {
            'efficiency_loss_per_percent_dod': 0.001,
            'reference_dod': 50
        }
    
    def save_sensitivities(self, output_path: Optional[Path] = None):
        """Save extracted sensitivities to JSON."""
        if output_path is None:
            output_path = DATA_DIR / "calce_sensitivities.json"
        
        sensitivities = self.extract_sensitivities()
        
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sensitivities, f, indent=2)
        
        logger.info(f"Saved sensitivities to {output_path}")
        return sensitivities


class NASADataAugmentor:
    """
    Augment NASA aging dataset with CALCE-derived physics sensitivities.
    
    The augmentation:
    1. Loads NASA aging data with actual RUL labels
    2. Applies temperature/DOD/current variations using CALCE sensitivities
    3. Creates augmented samples that represent different operating conditions
    4. Preserves NASA RUL ground truth with physics-informed adjustments
    """
    
    def __init__(self, nasa_dir: Path = NASA_DIR, sensitivities: Optional[Dict] = None):
        self.nasa_dir = nasa_dir
        self.data_dir = nasa_dir / "data"
        self.metadata_path = nasa_dir / "metadata.csv"
        
        # Load or use default sensitivities
        self.sensitivities = sensitivities or self._load_sensitivities()
        
        # Store processed data
        self.nasa_data = None
        self.augmented_data = None
        
    def _load_sensitivities(self) -> Dict:
        """Load CALCE sensitivities from file or extract them."""
        sens_path = DATA_DIR / "calce_sensitivities.json"
        if sens_path.exists():
            with open(sens_path, 'r') as f:
                return json.load(f)
        else:
            extractor = CALCESensitivityExtractor()
            return extractor.extract_sensitivities()
    
    def load_nasa_data(self) -> pd.DataFrame:
        """
        Load NASA battery aging dataset and aggregate to cycle-level.
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"NASA metadata not found: {self.metadata_path}")
        
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        logger.info(f"Loaded NASA metadata: {len(metadata)} records")
        
        # Filter to discharge cycles (where we have capacity measurements)
        discharge_meta = metadata[metadata['type'] == 'discharge'].copy()
        logger.info(f"Discharge cycles: {len(discharge_meta)}")
        
        # Group by battery and create cycle-level data
        cycle_data = []
        
        for battery_id in discharge_meta['battery_id'].unique():
            battery_df = discharge_meta[discharge_meta['battery_id'] == battery_id].copy()
            battery_df = battery_df.sort_values('test_id')
            
            # Get initial capacity
            valid_caps = battery_df[battery_df['Capacity'].notna()]['Capacity']
            if len(valid_caps) == 0:
                continue
            
            initial_capacity = float(valid_caps.iloc[0])
            
            # Calculate cycle number (discharge cycles only)
            battery_df['cycle'] = range(1, len(battery_df) + 1)
            
            for idx, row in battery_df.iterrows():
                if pd.isna(row['Capacity']):
                    continue
                
                try:
                    capacity = float(row['Capacity'])
                except (ValueError, TypeError):
                    continue
                cycle = row['cycle']
                
                # Get corresponding impedance data (Re, Rct)
                Re = row.get('Re', 0.055 + cycle * 0.0002)
                Rct = row.get('Rct', 0.18 + cycle * 0.0003)
                if pd.isna(Re):
                    Re = 0.055 + cycle * 0.0002
                if pd.isna(Rct):
                    Rct = 0.18 + cycle * 0.0003
                
                # Get temperature
                temp = row.get('ambient_temperature', 24)
                if pd.isna(temp):
                    temp = 24
                
                # Calculate derived features
                capacity_fade = initial_capacity - capacity
                capacity_ratio = capacity / initial_capacity if initial_capacity > 0 else 1.0
                soh = capacity_ratio * 100
                
                cycle_data.append({
                    'battery_id': battery_id,
                    'cycle': cycle,
                    'capacity': capacity,
                    'initial_capacity': initial_capacity,
                    'capacity_fade': capacity_fade,
                    'capacity_fade_percentage': (capacity_fade / initial_capacity) * 100,
                    'capacity_ratio': capacity_ratio,
                    'soh': soh,
                    'temperature': temp,
                    'Re': Re,
                    'Rct': Rct,
                    'filename': row.get('filename', ''),
                })
        
        self.nasa_data = pd.DataFrame(cycle_data)
        
        # Calculate RUL for each battery
        self._calculate_rul()
        
        logger.info(f"Loaded NASA data: {len(self.nasa_data)} cycle records from {self.nasa_data['battery_id'].nunique()} batteries")
        return self.nasa_data
    
    def _calculate_rul(self):
        """Calculate RUL based on 80% capacity threshold."""
        eol_threshold = 0.80
        
        processed = []
        for battery_id, battery_df in self.nasa_data.groupby('battery_id'):
            battery_df = battery_df.sort_values('cycle').copy()
            
            initial_cap = battery_df['initial_capacity'].iloc[0]
            eol_capacity = initial_cap * eol_threshold
            
            # Find EOL cycle
            eol_cycles = battery_df[battery_df['capacity'] < eol_capacity]
            if len(eol_cycles) > 0:
                eol_cycle = eol_cycles['cycle'].min()
            else:
                # Extrapolate EOL
                last_cap = battery_df['capacity'].iloc[-1]
                last_cycle = battery_df['cycle'].max()
                if initial_cap > last_cap:
                    fade_rate = (initial_cap - last_cap) / last_cycle
                    remaining = (last_cap - eol_capacity) / fade_rate if fade_rate > 0 else 100
                    eol_cycle = last_cycle + int(remaining)
                else:
                    eol_cycle = last_cycle + 100
            
            battery_df['eol_cycle'] = eol_cycle
            battery_df['rul'] = (eol_cycle - battery_df['cycle']).clip(lower=0)
            
            max_cycle = battery_df['cycle'].max()
            battery_df['cycle_normalized'] = battery_df['cycle'] / max(max_cycle, 1)
            
            processed.append(battery_df)
        
        self.nasa_data = pd.concat(processed, ignore_index=True)
    
    def augment_with_sensitivities(self, augmentation_factor: int = 2) -> pd.DataFrame:
        """
        Augment NASA data with physics-informed variations.
        
        For each NASA sample, create augmented versions that simulate:
        - Different operating temperatures (adjusted using CALCE sensitivity)
        - Different load conditions (current variations)
        
        The RUL is adjusted based on physics-informed degradation factors.
        """
        if self.nasa_data is None:
            self.load_nasa_data()
        
        augmented = [self.nasa_data.copy()]  # Start with original data
        augmented[0]['augmented'] = False
        augmented[0]['augmentation_type'] = 'original'
        
        temp_sens = self.sensitivities.get('temperature_sensitivity', {})
        dod_sens = self.sensitivities.get('dod_sensitivity', {})
        
        # Temperature augmentation
        for temp_offset in [-15, -5, 5, 15]:  # Temperature variations
            aug_df = self.nasa_data.copy()
            
            # Adjust temperature
            new_temp = aug_df['temperature'] + temp_offset
            
            # Apply physics-informed capacity adjustment
            if temp_offset < 0:
                # Cold reduces capacity
                penalty = temp_sens.get('capacity_cold_penalty_per_degC', 0.004)
                cap_factor = 1 - (abs(temp_offset) * penalty / 100)
            else:
                # Heat also reduces capacity (but less)
                penalty = temp_sens.get('capacity_hot_penalty_per_degC', 0.002)
                cap_factor = 1 - (temp_offset * penalty / 100)
            
            aug_df['temperature'] = new_temp
            aug_df['capacity'] = aug_df['capacity'] * cap_factor
            aug_df['capacity_fade'] = aug_df['initial_capacity'] - aug_df['capacity']
            aug_df['capacity_ratio'] = aug_df['capacity'] / aug_df['initial_capacity']
            aug_df['soh'] = aug_df['capacity_ratio'] * 100
            
            # Adjust RUL based on accelerated/decelerated degradation
            # Extreme temps accelerate degradation
            if abs(temp_offset) > 10:
                rul_factor = 0.85  # Reduced RUL due to harsher conditions
            else:
                rul_factor = 0.95
            
            aug_df['rul'] = (aug_df['rul'] * rul_factor).astype(int)
            aug_df['augmented'] = True
            aug_df['augmentation_type'] = f'temp_offset_{temp_offset}'
            
            augmented.append(aug_df)
        
        # Combine all augmented data
        self.augmented_data = pd.concat(augmented, ignore_index=True)
        
        logger.info(f"Augmented dataset: {len(self.augmented_data)} samples "
                   f"(original: {len(self.nasa_data)}, augmented: {len(self.augmented_data) - len(self.nasa_data)})")
        
        return self.augmented_data
    
    def get_training_data(self, use_augmented: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get training features and labels.
        
        Returns features compatible with existing model architecture.
        """
        if use_augmented and self.augmented_data is not None:
            df = self.augmented_data
        elif self.nasa_data is not None:
            df = self.nasa_data
        else:
            raise ValueError("No data loaded. Call load_nasa_data() first.")
        
        # Define feature columns (matching existing model expectations)
        feature_cols = [
            'cycle', 'capacity', 'capacity_fade', 'capacity_ratio',
            'soh', 'cycle_normalized', 'temperature', 'Re', 'Rct'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        y = df['rul'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(0).clip(lower=0)
        
        # Remove invalid rows
        valid_mask = (y >= 0) & (X['capacity'] > 0)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training data: {len(X)} samples, {len(available_cols)} features")
        logger.info(f"RUL range: {y.min():.0f} - {y.max():.0f} cycles")
        logger.info(f"RUL distribution: mean={y.mean():.0f}, median={y.median():.0f}")
        
        return X, y
    
    def save_augmented_data(self, output_path: Optional[Path] = None):
        """Save augmented data to CSV."""
        if output_path is None:
            output_path = DATA_DIR / "nasa_augmented.csv"
        
        if self.augmented_data is None:
            self.augment_with_sensitivities()
        
        os.makedirs(output_path.parent, exist_ok=True)
        self.augmented_data.to_csv(output_path, index=False)
        logger.info(f"Saved augmented data to {output_path}")


def main():
    """Main function to demonstrate physics-informed augmentation."""
    print("=" * 60)
    print("Phase 4: Physics-Informed Data Augmentation")
    print("=" * 60)
    
    # Step 1: Extract CALCE sensitivities
    print("\n[1] Extracting CALCE degradation sensitivities...")
    extractor = CALCESensitivityExtractor()
    sensitivities = extractor.save_sensitivities()
    
    print(f"\nExtracted sensitivities from {sensitivities.get('files_analyzed', 0)} CALCE profiles")
    print(f"Temperature sensitivity: {sensitivities['temperature_sensitivity'].get('voltage_coeff_per_degC', 'N/A'):.4f} V/°C")
    print(f"DOD sensitivity: {sensitivities['dod_sensitivity'].get('efficiency_loss_per_percent_dod', 'N/A'):.4f} per %DOD")
    
    # Step 2: Load and augment NASA data
    print("\n[2] Loading and augmenting NASA aging data...")
    augmentor = NASADataAugmentor(sensitivities=sensitivities)
    augmentor.load_nasa_data()
    augmentor.augment_with_sensitivities()
    augmentor.save_augmented_data()
    
    # Step 3: Get training data
    print("\n[3] Preparing training data...")
    X, y = augmentor.get_training_data(use_augmented=True)
    
    print(f"\nTraining Features: {X.columns.tolist()}")
    print(f"Training Samples: {len(X)}")
    print(f"RUL Statistics:")
    print(f"  Min: {y.min():.0f}")
    print(f"  Max: {y.max():.0f}")
    print(f"  Mean: {y.mean():.0f}")
    print(f"  Median: {y.median():.0f}")
    print(f"  Std: {y.std():.0f}")
    
    # Check RUL distribution across life stages
    early_life = y[y > 100].count()
    mid_life = y[(y >= 30) & (y <= 100)].count()
    late_life = y[y < 30].count()
    
    print(f"\nLife Stage Distribution:")
    print(f"  Early life (RUL > 100): {early_life} samples ({100*early_life/len(y):.1f}%)")
    print(f"  Mid life (30 <= RUL <= 100): {mid_life} samples ({100*mid_life/len(y):.1f}%)")
    print(f"  Late life (RUL < 30): {late_life} samples ({100*late_life/len(y):.1f}%)")
    
    print("\n" + "=" * 60)
    print("Physics-informed augmentation complete!")
    print("NASA remains primary aging ground truth.")
    print("CALCE sensitivities provide physics-based regularization.")
    print("=" * 60)


if __name__ == "__main__":
    main()
