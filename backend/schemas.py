"""
Pydantic schemas for Battery RUL Prediction API
Defines request/response models with validation rules
Phase 3: Extended with distribution validation and batch prediction support
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class BatteryHealthStatus(str, Enum):
    """Battery health classification based on RUL."""
    HEALTHY = "Healthy"
    MODERATE = "Moderate"
    CRITICAL = "Critical"


class ConfidenceLevel(str, Enum):
    """Prediction confidence levels."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class DistributionStatus(str, Enum):
    """Distribution status relative to training data."""
    IN_DISTRIBUTION = "in_distribution"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


class LifeStageContext(str, Enum):
    """Battery life stage classification."""
    EARLY_LIFE = "early_life"
    MID_LIFE = "mid_life"
    LATE_LIFE = "late_life"


class DatasetSource(str, Enum):
    """Supported dataset sources for multi-dataset analysis."""
    NASA = "NASA"
    CALCE = "CALCE"
    OXFORD = "OXFORD"
    MATR1 = "MATR1"


class CrossDatasetConfidence(str, Enum):
    """Cross-dataset confidence levels based on dataset agreement."""
    HIGH = "high"      # Closest dataset + at least one additional agrees
    MEDIUM = "medium"  # Strong match with single dataset only
    LOW = "low"        # Weak or no match across datasets


class PredictionRequest(BaseModel):
    """Request schema for battery RUL prediction.
    
    All parameters must be within valid operational ranges
    for lithium-ion batteries.
    """
    voltage: float = Field(
        ...,
        ge=2.0,
        le=4.5,
        description="Battery voltage in volts (2.0V - 4.5V)"
    )
    current: float = Field(
        ...,
        description="Battery current in amperes (typically negative for discharge)"
    )
    temperature: float = Field(
        ...,
        ge=0,
        le=70,
        description="Battery temperature in Celsius (0°C - 70°C)"
    )
    cycle: int = Field(
        ...,
        ge=0,
        description="Current charge-discharge cycle count (non-negative)"
    )
    capacity: float = Field(
        ...,
        gt=0.01,
        description="Current battery capacity in Ah (must be > 0.01)"
    )
    model_name: Optional[str] = Field(
        default="XGBoost",
        description="ML model to use for prediction"
    )

    @field_validator('voltage')
    @classmethod
    def validate_voltage(cls, v):
        if v < 2.0 or v > 4.5:
            raise ValueError('Voltage must be between 2.0V and 4.5V')
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if v < 0 or v > 70:
            raise ValueError('Temperature must be between 0°C and 70°C')
        return v

    @field_validator('capacity')
    @classmethod
    def validate_capacity(cls, v):
        if v <= 0.01:
            raise ValueError('Capacity must be greater than 0.01 Ah')
        return v

    @field_validator('cycle')
    @classmethod
    def validate_cycle(cls, v):
        if v < 0:
            raise ValueError('Cycle count must be non-negative')
        return v

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        valid_models = ['XGBoost', 'Random Forest', 'Linear Regression', 'LSTM']
        if v and v not in valid_models:
            raise ValueError(f'Model must be one of: {valid_models}')
        return v or 'XGBoost'

    class Config:
        json_schema_extra = {
            "example": {
                "voltage": 3.7,
                "current": -1.0,
                "temperature": 25.0,
                "cycle": 100,
                "capacity": 1.8,
                "model_name": "XGBoost"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for battery RUL prediction.
    
    Phase 3.5 Enhanced: Includes multi-dataset analysis and cross-dataset confidence.
    """
    predicted_rul_cycles: int = Field(
        ...,
        description="Predicted Remaining Useful Life in cycles"
    )
    battery_health: BatteryHealthStatus = Field(
        ...,
        description="Battery health classification"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Prediction confidence level"
    )
    model_used: str = Field(
        ...,
        description="ML model used for prediction"
    )
    # Phase 3: Distribution validation fields
    distribution_status: DistributionStatus = Field(
        default=DistributionStatus.IN_DISTRIBUTION,
        description="Whether input is within training data distribution"
    )
    life_stage_context: LifeStageContext = Field(
        default=LifeStageContext.MID_LIFE,
        description="Inferred battery life stage based on input parameters"
    )
    confidence_explanation: str = Field(
        default="",
        description="Detailed explanation of confidence level and prediction context"
    )
    inference_warning: Optional[str] = Field(
        default=None,
        description="Warning message if input is unusual or OOD"
    )
    # Phase 3.5: Multi-dataset analysis fields
    dominant_dataset: Optional[str] = Field(
        default="NASA",
        description="Dataset that best matches the input characteristics (NASA, CALCE, OXFORD, MATR1)"
    )
    cross_dataset_confidence: Optional[str] = Field(
        default="medium",
        description="Confidence based on cross-dataset agreement (high, medium, low)"
    )
    dataset_coverage_note: Optional[str] = Field(
        default=None,
        description="Explanation of which dataset lifecycle patterns the input matches"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_rul_cycles": 350,
                "battery_health": "Moderate",
                "confidence_level": "High",
                "model_used": "XGBoost",
                "distribution_status": "in_distribution",
                "life_stage_context": "mid_life",
                "confidence_explanation": "Prediction has high confidence. Input parameters are consistent with NASA training dataset characteristics.",
                "inference_warning": None,
                "dominant_dataset": "NASA",
                "cross_dataset_confidence": "high",
                "dataset_coverage_note": "Input best matches NASA dataset (score: 85%). Input characteristics suggest late battery life (approaching EOL). Supporting agreement from: CALCE."
            }
        }


class BatchPredictionRow(BaseModel):
    """Single row result from batch prediction."""
    row_index: int = Field(..., description="Original row index in CSV")
    predicted_rul_cycles: int = Field(..., description="Predicted RUL in cycles")
    battery_health: str = Field(..., description="Health classification")
    distribution_status: str = Field(..., description="Distribution status")
    life_stage_context: str = Field(..., description="Life stage")
    confidence_level: str = Field(..., description="Confidence level")
    inference_warning: Optional[str] = Field(None, description="Warning if any")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    # Phase 3.5: Multi-dataset fields
    dominant_dataset: Optional[str] = Field(default="NASA", description="Best matching dataset")
    cross_dataset_confidence: Optional[str] = Field(default="medium", description="Cross-dataset confidence")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    success: bool = Field(..., description="Whether batch prediction was successful")
    total_rows: int = Field(..., description="Total rows in input CSV")
    processed_rows: int = Field(..., description="Number of rows successfully processed")
    failed_rows: int = Field(..., description="Number of rows that failed")
    results: List[BatchPredictionRow] = Field(..., description="Per-row prediction results")
    summary: dict = Field(default_factory=dict, description="Summary statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_rows": 10,
                "processed_rows": 9,
                "failed_rows": 1,
                "results": [
                    {
                        "row_index": 0,
                        "predicted_rul_cycles": 120,
                        "battery_health": "Moderate",
                        "distribution_status": "in_distribution",
                        "life_stage_context": "mid_life",
                        "confidence_level": "High",
                        "inference_warning": None,
                        "error": None
                    }
                ],
                "summary": {
                    "avg_rul": 115,
                    "healthy_count": 3,
                    "moderate_count": 5,
                    "critical_count": 1,
                    "ood_count": 2
                }
            }
        }


class DatasetStatisticsResponse(BaseModel):
    """Response schema for dataset statistics endpoint."""
    metadata: dict = Field(..., description="Dataset metadata")
    features: dict = Field(..., description="Per-feature statistics")
    dataset_context: dict = Field(..., description="Context about dataset bias and characteristics")

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "dataset_source": "NASA Li-ion Battery Aging Dataset",
                    "total_samples": 1500,
                    "batteries_analyzed": 11
                },
                "features": {
                    "cycle": {
                        "minimum": 1,
                        "maximum": 168,
                        "mean": 85,
                        "median": 84,
                        "standard_deviation": 48,
                        "percentile_5": 9,
                        "percentile_95": 160
                    }
                },
                "dataset_context": {
                    "late_life_bias": True,
                    "cycle_range_dominant": "50-170 cycles",
                    "eol_threshold": 0.8
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    status: str = Field(default="error")
    message: str = Field(..., description="Human-readable error description")
    detail: Optional[str] = Field(default=None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Voltage must be between 2.0V and 4.5V",
                "detail": None
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    available_models: list = Field(..., description="List of available models")
