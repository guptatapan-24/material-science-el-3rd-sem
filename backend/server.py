"""
FastAPI Backend for Battery RUL Prediction System
Provides REST API endpoints for battery life prediction

Phase 3: Enhanced with distribution validation and batch prediction
"""
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import os
import io
import pandas as pd

# Import schemas and predictor
from schemas import (
    PredictionRequest, 
    PredictionResponse, 
    ErrorResponse, 
    HealthResponse,
    BatteryHealthStatus,
    ConfidenceLevel,
    DistributionStatus,
    LifeStageContext,
    BatchPredictionResponse,
    BatchPredictionRow,
    DatasetStatisticsResponse
)
from predictor import get_predictor, BatteryPredictor
from distribution_validator import validate_prediction_input, get_validator
from dataset_statistics import get_statistics_generator
from multi_dataset_statistics import (
    get_multi_dataset_manager,
    analyze_input_cross_dataset
)

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
        
        # Initialize statistics generator (generates stats if not exist)
        stats_gen = get_statistics_generator()
        logger.info("Dataset statistics initialized")
        
        # Initialize distribution validator
        validator = get_validator()
        logger.info("Distribution validator initialized")
        
        # Phase 3.5: Initialize multi-dataset statistics manager
        multi_dataset_mgr = get_multi_dataset_manager()
        logger.info(f"Multi-dataset statistics initialized for: {list(multi_dataset_mgr.statistics.keys())}")
        
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
    description="AI-powered Remaining Useful Life prediction for Lithium-Ion batteries. Phase 3: Enhanced with OOD detection and batch prediction.",
    version="3.0.0",
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
        "version": "3.0.0",
        "docs": "/api/docs",
        "features": [
            "Single prediction with OOD detection",
            "Batch CSV prediction",
            "Dataset statistics",
            "Confidence explanation"
        ]
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
        "api_version": "3.0.0",
        "phase": "Phase 3 - Dataset Compatibility Verification"
    }


@app.get("/api/statistics", response_model=DatasetStatisticsResponse)
async def get_dataset_statistics():
    """Get training dataset statistics.
    
    Returns statistical bounds and context about the NASA training dataset.
    Useful for understanding prediction confidence and OOD detection bounds.
    """
    try:
        stats_gen = get_statistics_generator()
        stats = stats_gen.get_all_statistics()
        
        return DatasetStatisticsResponse(
            metadata=stats.get('metadata', {}),
            features=stats.get('features', {}),
            dataset_context=stats.get('dataset_context', {})
        )
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Failed to fetch statistics: {str(e)}"}
        )


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
    
    Phase 4 Enhanced: Supports model versioning and baseline comparison.
    
    **Input Parameters:**
    - **voltage**: Battery voltage (2.0V - 4.5V)
    - **current**: Battery current in Amperes (typically negative for discharge)
    - **temperature**: Battery temperature (0°C - 70°C)
    - **cycle**: Current charge-discharge cycle count (non-negative)
    - **capacity**: Current battery capacity in Ah (> 0.01)
    - **model_name**: ML model to use (optional, default: XGBoost)
    - **model_version**: Model version ('v2_physics_augmented' or 'v1_nasa')
    - **compare_baseline**: If true, include baseline comparison in response
    
    **Response:**
    - **predicted_rul_cycles**: Predicted remaining cycles until 80% capacity
    - **battery_health**: Health classification (Healthy/Moderate/Critical)
    - **confidence_level**: Prediction confidence (High/Medium/Low)
    - **model_version**: Model version used for prediction
    - **recommendation**: Maintenance recommendation
    - **distribution_status**: Whether input is in/out of training distribution
    - **life_stage_context**: Inferred battery life stage
    - **confidence_explanation**: Detailed explanation of prediction context
    - **inference_warning**: Warning if input is unusual (optional)
    - **baseline_comparison**: Comparison with v1 baseline (if requested)
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
        # Make prediction with specified model version
        model_version = request.model_version or "v2_physics_augmented"
        
        result = predictor.predict(
            voltage=request.voltage,
            current=request.current,
            temperature=request.temperature,
            cycle=request.cycle,
            capacity=request.capacity,
            model_name=request.model_name,
            model_version=model_version
        )
        
        predicted_rul = result['predicted_rul_cycles']
        model_version_used = result.get('model_version', model_version)
        recommendation = result.get('recommendation', '')
        
        # Perform distribution validation
        validation = validate_prediction_input(
            voltage=request.voltage,
            current=request.current,
            temperature=request.temperature,
            cycle=request.cycle,
            capacity=request.capacity,
            predicted_rul=predicted_rul
        )
        
        # Phase 3.5: Perform cross-dataset analysis
        cross_dataset_result = analyze_input_cross_dataset(
            voltage=request.voltage,
            current=request.current,
            temperature=request.temperature,
            cycle=request.cycle,
            capacity=request.capacity
        )
        
        # Determine final confidence based on cross-dataset agreement
        cross_dataset_conf = cross_dataset_result.get('cross_dataset_confidence', 'medium')
        dominant_dataset = cross_dataset_result.get('dominant_dataset', 'NASA')
        dataset_coverage_note = cross_dataset_result.get('dataset_coverage_note', '')
        
        # Combine OOD validation confidence with cross-dataset confidence
        final_confidence = validation['confidence_level']
        if validation['distribution_status'] == 'out_of_distribution':
            # OOD inputs get reduced confidence
            if result['confidence_level'] == 'High':
                final_confidence = 'Medium'
            elif result['confidence_level'] == 'Medium':
                final_confidence = 'Low'
        
        # Enhance confidence explanation with multi-dataset context
        enhanced_explanation = validation['confidence_explanation']
        if dataset_coverage_note:
            enhanced_explanation = f"{enhanced_explanation} {dataset_coverage_note}"
        
        # Phase 4: Baseline comparison if requested
        baseline_comparison = None
        if request.compare_baseline and model_version != 'v1_nasa':
            baseline_result = predictor.predict(
                voltage=request.voltage,
                current=request.current,
                temperature=request.temperature,
                cycle=request.cycle,
                capacity=request.capacity,
                model_name=request.model_name,
                model_version='v1_nasa'
            )
            improvement = predicted_rul - baseline_result['predicted_rul_cycles']
            baseline_comparison = {
                'v1_predicted_rul': baseline_result['predicted_rul_cycles'],
                'v1_health': baseline_result['battery_health'],
                'rul_difference': improvement,
                'comparison_note': (
                    f"V2 physics-augmented model predicts {abs(improvement)} cycles "
                    f"{'more' if improvement > 0 else 'fewer'} RUL than V1 baseline"
                )
            }
        
        logger.info(
            f"Prediction made: RUL={predicted_rul} cycles, "
            f"Health={result['battery_health']}, Model={result['model_used']}, "
            f"Version={model_version_used}, "
            f"Distribution={validation['distribution_status']}, LifeStage={validation['life_stage_context']}, "
            f"DominantDataset={dominant_dataset}, CrossDatasetConf={cross_dataset_conf}"
        )
        
        return PredictionResponse(
            predicted_rul_cycles=predicted_rul,
            battery_health=BatteryHealthStatus(result['battery_health']),
            confidence_level=ConfidenceLevel(final_confidence),
            model_used=result['model_used'],
            model_version=model_version_used,
            recommendation=recommendation,
            distribution_status=DistributionStatus(validation['distribution_status']),
            life_stage_context=LifeStageContext(validation['life_stage_context']),
            confidence_explanation=enhanced_explanation,
            inference_warning=validation['inference_warning'],
            dominant_dataset=dominant_dataset,
            cross_dataset_confidence=cross_dataset_conf,
            dataset_coverage_note=dataset_coverage_note,
            baseline_comparison=baseline_comparison
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


@app.post(
    "/api/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid CSV Format"},
        500: {"model": ErrorResponse, "description": "Processing Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV file upload.
    
    **Required CSV Columns:**
    - voltage: Battery voltage (V)
    - current: Battery current (A)
    - temperature: Battery temperature (°C)
    - cycle: Cycle count
    - capacity: Current capacity (Ah)
    
    **Optional Columns:**
    - model_name: ML model to use (default: XGBoost)
    
    **Response:**
    Returns per-row predictions with distribution status and confidence.
    """
    global predictor
    
    # Check if predictor is available
    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "error", "message": "Prediction service unavailable"}
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "File must be a CSV"}
        )
    
    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Validate required columns
        required_columns = ['voltage', 'current', 'temperature', 'cycle', 'capacity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "message": f"Missing required columns: {missing_columns}",
                    "required_columns": required_columns
                }
            )
        
        # Process each row
        results = []
        processed = 0
        failed = 0
        
        # Summary counters
        healthy_count = 0
        moderate_count = 0
        critical_count = 0
        ood_count = 0
        total_rul = 0
        
        for idx, row in df.iterrows():
            try:
                # Validate row values
                voltage = float(row['voltage'])
                current = float(row['current'])
                temperature = float(row['temperature'])
                cycle = int(row['cycle'])
                capacity = float(row['capacity'])
                model_name = str(row.get('model_name', 'XGBoost'))
                
                # Make prediction
                pred_result = predictor.predict(
                    voltage=voltage,
                    current=current,
                    temperature=temperature,
                    cycle=cycle,
                    capacity=capacity,
                    model_name=model_name
                )
                
                # Validate distribution
                validation = validate_prediction_input(
                    voltage=voltage,
                    current=current,
                    temperature=temperature,
                    cycle=cycle,
                    capacity=capacity,
                    predicted_rul=pred_result['predicted_rul_cycles']
                )
                
                # Phase 3.5: Cross-dataset analysis for batch
                cross_dataset_result = analyze_input_cross_dataset(
                    voltage=voltage,
                    current=current,
                    temperature=temperature,
                    cycle=cycle,
                    capacity=capacity
                )
                
                dominant_dataset = cross_dataset_result.get('dominant_dataset', 'NASA')
                cross_dataset_conf = cross_dataset_result.get('cross_dataset_confidence', 'medium')
                
                # Determine final confidence
                final_confidence = validation['confidence_level']
                if validation['distribution_status'] == 'out_of_distribution':
                    if pred_result['confidence_level'] == 'High':
                        final_confidence = 'Medium'
                    elif pred_result['confidence_level'] == 'Medium':
                        final_confidence = 'Low'
                    ood_count += 1
                
                # Update counters
                health = pred_result['battery_health']
                if health == 'Healthy':
                    healthy_count += 1
                elif health == 'Moderate':
                    moderate_count += 1
                else:
                    critical_count += 1
                
                total_rul += pred_result['predicted_rul_cycles']
                
                results.append(BatchPredictionRow(
                    row_index=idx,
                    predicted_rul_cycles=pred_result['predicted_rul_cycles'],
                    battery_health=pred_result['battery_health'],
                    distribution_status=validation['distribution_status'],
                    life_stage_context=validation['life_stage_context'],
                    confidence_level=final_confidence,
                    inference_warning=validation['inference_warning'],
                    error=None,
                    dominant_dataset=dominant_dataset,
                    cross_dataset_confidence=cross_dataset_conf
                ))
                processed += 1
                
            except Exception as e:
                results.append(BatchPredictionRow(
                    row_index=idx,
                    predicted_rul_cycles=0,
                    battery_health="Unknown",
                    distribution_status="unknown",
                    life_stage_context="unknown",
                    confidence_level="Low",
                    inference_warning=None,
                    error=str(e),
                    dominant_dataset="NASA",
                    cross_dataset_confidence="low"
                ))
                failed += 1
        
        # Calculate summary
        summary = {
            "avg_rul": int(total_rul / processed) if processed > 0 else 0,
            "healthy_count": healthy_count,
            "moderate_count": moderate_count,
            "critical_count": critical_count,
            "ood_count": ood_count
        }
        
        logger.info(f"Batch prediction complete: {processed}/{len(df)} rows processed, {failed} failed")
        
        return BatchPredictionResponse(
            success=failed == 0,
            total_rows=len(df),
            processed_rows=processed,
            failed_rows=failed,
            results=results,
            summary=summary
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": "CSV file is empty"}
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Batch processing failed: {str(e)}"}
        )


@app.get("/api/predict/batch/template")
async def get_batch_template():
    """Get CSV template for batch prediction.
    
    Returns a sample CSV file with the required column structure.
    """
    # Create sample data
    sample_data = {
        'voltage': [3.7, 3.5, 3.3, 3.8],
        'current': [-1.0, -1.0, -0.8, -1.2],
        'temperature': [25, 30, 35, 28],
        'cycle': [50, 100, 150, 75],
        'capacity': [1.8, 1.6, 1.4, 1.7]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_prediction_template.csv"}
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
