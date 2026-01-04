"""
FastAPI Backend for Battery RUL Prediction System
Provides REST API endpoints for battery life prediction
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import os

# Import schemas and predictor
from schemas import (
    PredictionRequest, 
    PredictionResponse, 
    ErrorResponse, 
    HealthResponse,
    BatteryHealthStatus,
    ConfidenceLevel
)
from predictor import get_predictor, BatteryPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: BatteryPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - load models at startup."""
    global predictor
    logger.info("Starting Battery RUL Prediction API...")
    
    try:
        # Initialize predictor and load models
        predictor = get_predictor()
        logger.info(f"Models loaded: {predictor.available_models}")
        logger.info("API ready to serve predictions")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Battery RUL Prediction API...")


# Create FastAPI application
app = FastAPI(
    title="Battery RUL Prediction API",
    description="AI-powered Remaining Useful Life prediction for Lithium-Ion batteries",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle Pydantic validation errors with 422 status."""
    errors = exc.errors()
    error_messages = []
    
    for error in errors:
        field = error.get('loc', ['unknown'])[-1]
        msg = error.get('msg', 'Validation error')
        error_messages.append(f"{field}: {msg}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "detail": "; ".join(error_messages)
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Battery RUL Prediction API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.
    
    Returns service status and model availability.
    """
    global predictor
    
    models_loaded = predictor is not None and predictor.is_ready
    available_models = predictor.available_models if predictor else []
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        service="battery-rul-prediction",
        models_loaded=models_loaded,
        available_models=available_models
    )


@app.get("/api/status")
async def status_check():
    """Detailed status endpoint."""
    global predictor
    
    return {
        "status": "running",
        "models_loaded": predictor.is_ready if predictor else False,
        "available_models": predictor.available_models if predictor else [],
        "api_version": "1.0.0"
    }


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Model Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def predict_rul(request: PredictionRequest):
    """Predict Remaining Useful Life for a battery.
    
    Accepts battery parameters and returns RUL prediction with
    health classification and confidence level.
    
    **Input Parameters:**
    - **voltage**: Battery voltage (2.0V - 4.5V)
    - **current**: Battery current in Amperes (typically negative for discharge)
    - **temperature**: Battery temperature (0°C - 70°C)
    - **cycle**: Current charge-discharge cycle count (non-negative)
    - **capacity**: Current battery capacity in Ah (> 0.01)
    - **model_name**: ML model to use (optional, default: XGBoost)
    
    **Response:**
    - **predicted_rul_cycles**: Predicted remaining cycles until 80% capacity
    - **battery_health**: Health classification (Healthy/Moderate/Critical)
    - **confidence_level**: Prediction confidence (High/Medium/Low)
    - **model_used**: ML model used for prediction
    """
    global predictor
    
    # Check if predictor is available
    if predictor is None or not predictor.is_ready:
        logger.error("Predictor not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": "Prediction service unavailable. Models not loaded."
            }
        )
    
    try:
        # Make prediction
        result = predictor.predict(
            voltage=request.voltage,
            current=request.current,
            temperature=request.temperature,
            cycle=request.cycle,
            capacity=request.capacity,
            model_name=request.model_name
        )
        
        logger.info(
            f"Prediction made: RUL={result['predicted_rul_cycles']} cycles, "
            f"Health={result['battery_health']}, Model={result['model_used']}"
        )
        
        return PredictionResponse(
            predicted_rul_cycles=result['predicted_rul_cycles'],
            battery_health=BatteryHealthStatus(result['battery_health']),
            confidence_level=ConfidenceLevel(result['confidence_level']),
            model_used=result['model_used']
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "status": "error",
                "message": str(e)
            }
        )
    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Model error: {str(e)}"
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": f"Prediction failed: {str(e)}"
            }
        )


@app.get("/api/models")
async def list_models():
    """List available ML models."""
    global predictor
    
    if predictor is None:
        return {"models": []}
    
    return {
        "models": predictor.available_models,
        "default": "XGBoost"
    }
