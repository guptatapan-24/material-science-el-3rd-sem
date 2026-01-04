"""
Pydantic schemas for Battery RUL Prediction API
Defines request/response models with validation rules
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
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
    """Response schema for battery RUL prediction."""
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

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_rul_cycles": 350,
                "battery_health": "Moderate",
                "confidence_level": "High",
                "model_used": "XGBoost"
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
