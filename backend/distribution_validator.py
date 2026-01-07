"""
Distribution Validator for Battery RUL Prediction System
Implements Out-of-Distribution (OOD) detection and prediction explanation logic

Phase 3.5: Enhanced with multi-dataset awareness (NASA, CALCE, Oxford, MATR1)
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass

from .dataset_statistics import get_statistics_generator
from .multi_dataset_statistics import (
    get_multi_dataset_manager,
    analyze_input_cross_dataset,
    DatasetSource,
    CrossDatasetConfidence
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributionStatus(str, Enum):
    """Distribution status classification."""
    IN_DISTRIBUTION = "in_distribution"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


class LifeStageContext(str, Enum):
    """Battery life stage classification."""
    EARLY_LIFE = "early_life"
    MID_LIFE = "mid_life"
    LATE_LIFE = "late_life"


class ConfidenceLevel(str, Enum):
    """Prediction confidence levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class FeatureOODResult:
    """Result of OOD check for a single feature."""
    feature_name: str
    input_value: float
    lower_bound: float
    upper_bound: float
    is_ood: bool
    ood_direction: Optional[str]  # 'low' or 'high' or None
    training_median: float
    deviation_from_median: float


@dataclass
class ValidationResult:
    """Complete validation result for an input."""
    distribution_status: DistributionStatus
    life_stage_context: LifeStageContext
    confidence_level: ConfidenceLevel
    confidence_explanation: str
    inference_warning: Optional[str]
    feature_results: List[FeatureOODResult]
    ood_features: List[str]
    early_life_indicators: List[str]


class DistributionValidator:
    """
    Validates input data against training data distribution.
    Implements OOD detection and prediction explanation logic.
    """
    
    def __init__(self):
        self.stats_generator = get_statistics_generator()
        self.statistics = self.stats_generator.get_all_statistics()
    
    def check_feature_ood(
        self,
        feature_name: str,
        input_value: float
    ) -> FeatureOODResult:
        """
        Check if a single feature value is out of distribution.
        
        Uses 5th and 95th percentiles as OOD bounds.
        """
        stats = self.stats_generator.get_feature_statistics(feature_name)
        
        if not stats:
            # Feature not found, assume in-distribution
            return FeatureOODResult(
                feature_name=feature_name,
                input_value=input_value,
                lower_bound=float('-inf'),
                upper_bound=float('inf'),
                is_ood=False,
                ood_direction=None,
                training_median=input_value,
                deviation_from_median=0.0
            )
        
        lower_bound = stats.get('percentile_5', stats.get('minimum', 0))
        upper_bound = stats.get('percentile_95', stats.get('maximum', float('inf')))
        training_median = stats.get('median', stats.get('mean', 0))
        
        is_ood = False
        ood_direction = None
        
        if input_value < lower_bound:
            is_ood = True
            ood_direction = 'low'
        elif input_value > upper_bound:
            is_ood = True
            ood_direction = 'high'
        
        # Calculate deviation from median
        std = stats.get('standard_deviation', 1)
        deviation_from_median = (input_value - training_median) / std if std > 0 else 0
        
        return FeatureOODResult(
            feature_name=feature_name,
            input_value=input_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            is_ood=is_ood,
            ood_direction=ood_direction,
            training_median=training_median,
            deviation_from_median=deviation_from_median
        )
    
    def classify_life_stage(
        self,
        cycle: int,
        capacity: float,
        cycle_median: float = 84,
        capacity_median: float = 1.4
    ) -> LifeStageContext:
        """
        Classify battery life stage based on cycle and capacity.
        
        NASA dataset is biased toward late-life data (aging phase).
        """
        # Get training medians
        cycle_stats = self.stats_generator.get_feature_statistics('cycle')
        capacity_stats = self.stats_generator.get_feature_statistics('capacity')
        
        if cycle_stats:
            cycle_median = cycle_stats.get('median', 84)
        if capacity_stats:
            capacity_median = capacity_stats.get('median', 1.4)
        
        initial_capacity = 2.0  # Nominal for NASA dataset
        
        # Life stage classification logic
        capacity_ratio = capacity / initial_capacity if initial_capacity > 0 else 0.7
        
        # Early life: low cycle count AND high capacity
        if cycle < cycle_median * 0.5 and capacity_ratio > 0.9:
            return LifeStageContext.EARLY_LIFE
        
        # Late life: high cycle count OR low capacity ratio
        if cycle > cycle_median * 1.2 or capacity_ratio < 0.75:
            return LifeStageContext.LATE_LIFE
        
        # Mid life: everything in between
        return LifeStageContext.MID_LIFE
    
    def detect_early_life_indicators(
        self,
        cycle: int,
        capacity: float,
        temperature: float
    ) -> List[str]:
        """
        Detect indicators that suggest input is from early battery life.
        
        These inputs are likely to receive conservative (low) RUL predictions
        because the NASA dataset is biased toward late-life data.
        """
        indicators = []
        
        # Get training statistics
        cycle_stats = self.stats_generator.get_feature_statistics('cycle')
        capacity_stats = self.stats_generator.get_feature_statistics('capacity')
        temp_stats = self.stats_generator.get_feature_statistics('temperature_measured')
        
        cycle_median = cycle_stats.get('median', 84) if cycle_stats else 84
        capacity_median = capacity_stats.get('median', 1.4) if capacity_stats else 1.4
        temp_median = temp_stats.get('median', 24) if temp_stats else 24
        temp_range = (temp_stats.get('percentile_25', 20), temp_stats.get('percentile_75', 28)) if temp_stats else (20, 28)
        
        # Check cycle count
        if cycle < cycle_median * 0.5:
            indicators.append(f"Cycle count ({cycle}) is significantly below training median ({cycle_median:.0f})")
        
        # Check capacity
        initial_capacity = 2.0
        if capacity > capacity_median * 1.1:
            indicators.append(f"Capacity ({capacity:.2f} Ah) is above training median ({capacity_median:.2f} Ah)")
        
        # Check capacity ratio (SoH)
        capacity_ratio = capacity / initial_capacity
        if capacity_ratio > 0.9:
            indicators.append(f"State of Health ({capacity_ratio*100:.1f}%) indicates early life (>90%)")
        
        # Check temperature
        if temperature < temp_range[0] or temperature > temp_range[1]:
            indicators.append(f"Temperature ({temperature}°C) is outside dominant training range ({temp_range[0]}-{temp_range[1]}°C)")
        
        return indicators
    
    def generate_confidence_explanation(
        self,
        distribution_status: DistributionStatus,
        life_stage: LifeStageContext,
        ood_features: List[str],
        early_life_indicators: List[str],
        predicted_rul: int
    ) -> Tuple[ConfidenceLevel, str]:
        """
        Generate confidence level and explanation for the prediction.
        """
        confidence_factors = []
        confidence_score = 100  # Start with full confidence
        
        # OOD penalty
        if distribution_status == DistributionStatus.OUT_OF_DISTRIBUTION:
            ood_penalty = min(30, len(ood_features) * 10)
            confidence_score -= ood_penalty
            confidence_factors.append(f"Input outside training distribution ({', '.join(ood_features)})")
        
        # Life stage penalty for early-life inputs
        if life_stage == LifeStageContext.EARLY_LIFE:
            confidence_score -= 20
            confidence_factors.append("Input appears to be from early battery life; NASA dataset is biased toward aging phase")
        elif life_stage == LifeStageContext.MID_LIFE:
            confidence_score -= 5
        
        # Early life indicator penalty
        if early_life_indicators:
            penalty = min(20, len(early_life_indicators) * 7)
            confidence_score -= penalty
        
        # Determine confidence level
        if confidence_score >= 70:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 40:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        # Build explanation
        if not confidence_factors and not early_life_indicators:
            explanation = (
                f"Prediction has {confidence_level.value.lower()} confidence. "
                f"Input parameters are consistent with NASA training dataset characteristics."
            )
        else:
            explanation_parts = []
            
            if life_stage == LifeStageContext.EARLY_LIFE:
                explanation_parts.append(
                    "The input suggests an early-life battery, but the NASA dataset primarily contains "
                    "data from batteries in mid-to-late life stages. This can result in conservative RUL predictions."
                )
            
            if distribution_status == DistributionStatus.OUT_OF_DISTRIBUTION:
                explanation_parts.append(
                    f"Some input values ({', '.join(ood_features)}) fall outside the typical "
                    f"training data range (5th-95th percentile bounds)."
                )
            
            if predicted_rul < 50 and early_life_indicators:
                explanation_parts.append(
                    "Low RUL prediction may be due to model extrapolation from late-life training data "
                    "to early-life input conditions."
                )
            
            explanation = " ".join(explanation_parts) if explanation_parts else f"Confidence: {confidence_level.value}"
        
        return confidence_level, explanation
    
    def generate_inference_warning(
        self,
        distribution_status: DistributionStatus,
        life_stage: LifeStageContext,
        ood_features: List[str],
        early_life_indicators: List[str],
        predicted_rul: int
    ) -> Optional[str]:
        """
        Generate warning message for inference results when needed.
        """
        warnings = []
        
        if distribution_status == DistributionStatus.OUT_OF_DISTRIBUTION:
            warnings.append(
                f"⚠️ Input values for [{', '.join(ood_features)}] are outside the training data distribution. "
                f"Prediction may be less reliable."
            )
        
        if life_stage == LifeStageContext.EARLY_LIFE and predicted_rul < 100:
            warnings.append(
                "⚠️ Conservative prediction: Input suggests early battery life, but training data is "
                "dominated by aging-phase measurements. Actual RUL may be higher."
            )
        
        if early_life_indicators and not warnings:
            # Only add if not already warned about early life
            if len(early_life_indicators) >= 2:
                warnings.append(
                    "ℹ️ Multiple early-life indicators detected. NASA dataset focuses on batteries "
                    "approaching end-of-life (80% capacity threshold)."
                )
        
        return " | ".join(warnings) if warnings else None
    
    def validate_input(
        self,
        voltage: float,
        current: float,
        temperature: float,
        cycle: int,
        capacity: float,
        predicted_rul: int = 0
    ) -> ValidationResult:
        """
        Perform complete validation of input parameters.
        
        Returns comprehensive validation result with OOD status,
        life stage, confidence, and explanations.
        """
        # Map inputs to feature names used in statistics
        feature_inputs = {
            'cycle': cycle,
            'capacity': capacity,
            'voltage_measured': voltage,
            'temperature_measured': temperature,
            'current_measured': current
        }
        
        # Check each feature for OOD
        feature_results = []
        ood_features = []
        
        for feature_name, value in feature_inputs.items():
            result = self.check_feature_ood(feature_name, value)
            feature_results.append(result)
            if result.is_ood:
                ood_features.append(feature_name)
        
        # Determine overall distribution status
        distribution_status = (
            DistributionStatus.OUT_OF_DISTRIBUTION if ood_features 
            else DistributionStatus.IN_DISTRIBUTION
        )
        
        # Classify life stage
        life_stage = self.classify_life_stage(cycle, capacity)
        
        # Detect early life indicators
        early_life_indicators = self.detect_early_life_indicators(cycle, capacity, temperature)
        
        # Generate confidence and explanation
        confidence_level, confidence_explanation = self.generate_confidence_explanation(
            distribution_status,
            life_stage,
            ood_features,
            early_life_indicators,
            predicted_rul
        )
        
        # Generate warning if needed
        inference_warning = self.generate_inference_warning(
            distribution_status,
            life_stage,
            ood_features,
            early_life_indicators,
            predicted_rul
        )
        
        return ValidationResult(
            distribution_status=distribution_status,
            life_stage_context=life_stage,
            confidence_level=confidence_level,
            confidence_explanation=confidence_explanation,
            inference_warning=inference_warning,
            feature_results=feature_results,
            ood_features=ood_features,
            early_life_indicators=early_life_indicators
        )


# Global instance
_validator: Optional[DistributionValidator] = None


def get_validator() -> DistributionValidator:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        _validator = DistributionValidator()
    return _validator


def validate_prediction_input(
    voltage: float,
    current: float,
    temperature: float,
    cycle: int,
    capacity: float,
    predicted_rul: int = 0
) -> Dict[str, Any]:
    """
    Convenience function to validate prediction input.
    
    Returns a dictionary suitable for including in API response.
    """
    validator = get_validator()
    result = validator.validate_input(
        voltage, current, temperature, cycle, capacity, predicted_rul
    )
    
    return {
        'distribution_status': result.distribution_status.value,
        'life_stage_context': result.life_stage_context.value,
        'confidence_level': result.confidence_level.value,
        'confidence_explanation': result.confidence_explanation,
        'inference_warning': result.inference_warning,
        'ood_features': result.ood_features,
        'early_life_indicators': result.early_life_indicators
    }
