"""
Streamlit Frontend for Battery RUL Prediction System
Decoupled client - delegates all ML inference to FastAPI backend
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import logging
import os
from typing import Optional, Dict, Any

from utils.auth import check_authentication, login_page, logout
from utils.data_processor import load_nasa_dataset, preprocess_data
from utils.visualizer import (
    plot_capacity_fade_curve, plot_model_comparison, plot_r2_comparison,
    plot_feature_importance, plot_what_if_comparison, plot_sustainability_impact,
    create_metrics_table
)
from utils.report_generator import generate_csv_report, generate_pdf_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend API Configuration
# Use environment variable or default to localhost for local development
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8001')
API_PREDICT_ENDPOINT = f"{BACKEND_URL}/api/predict"
API_PREDICT_BATCH_ENDPOINT = f"{BACKEND_URL}/api/predict/batch"
API_HEALTH_ENDPOINT = f"{BACKEND_URL}/api/health"
API_MODELS_ENDPOINT = f"{BACKEND_URL}/api/models"
API_STATISTICS_ENDPOINT = f"{BACKEND_URL}/api/statistics"

# Request timeout in seconds
API_TIMEOUT = 30

# Page config
st.set_page_config(
    page_title="🔋 Battery RUL Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF4B4B, #FFA500, #00CC96);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    .health-healthy {
        color: #00CC96;
        font-weight: bold;
    }
    .health-moderate {
        color: #FFA500;
        font-weight: bold;
    }
    .health-critical {
        color: #FF4B4B;
        font-weight: bold;
    }
    .confidence-high {
        color: #00CC96;
    }
    .confidence-medium {
        color: #FFA500;
    }
    .confidence-low {
        color: #FF4B4B;
    }
    .dist-in {
        color: #00CC96;
        font-weight: bold;
    }
    .dist-out {
        color: #FF4B4B;
        font-weight: bold;
    }
    .stage-early {
        color: #00CC96;
    }
    .stage-mid {
        color: #FFA500;
    }
    .stage-late {
        color: #FF4B4B;
    }
    .explanation-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00CC96;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #5f3a1e 0%, #875a2d 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFA500;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'backend_status' not in st.session_state:
    st.session_state.backend_status = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []


# ============================================================================
# API CLIENT FUNCTIONS - All backend communication goes through these
# ============================================================================

def check_backend_health() -> Dict[str, Any]:
    """
    Check if the backend API is healthy and models are loaded.
    
    Returns:
        dict: Health status including available models
    """
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        logger.error("Backend connection failed - service may be down")
        return {"status": "unavailable", "error": "Cannot connect to prediction service"}
    except requests.exceptions.Timeout:
        logger.error("Backend health check timed out")
        return {"status": "timeout", "error": "Service response timeout"}
    except Exception as e:
        logger.error(f"Backend health check error: {e}")
        return {"status": "error", "error": str(e)}


def get_available_models() -> list:
    """
    Fetch list of available ML models from backend.
    
    Returns:
        list: Available model names
    """
    try:
        response = requests.get(API_MODELS_ENDPOINT, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get('models', [])
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return []


def predict_rul_via_api(
    voltage: float,
    current: float,
    temperature: float,
    cycle: int,
    capacity: float,
    model_name: str = "XGBoost"
) -> Dict[str, Any]:
    """
    Make RUL prediction by calling the backend API.
    
    Args:
        voltage: Battery voltage (V)
        current: Battery current (A)
        temperature: Temperature (°C)
        cycle: Cycle count
        capacity: Current capacity (Ah)
        model_name: ML model to use
        
    Returns:
        dict: Prediction result or error information
    """
    payload = {
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "cycle": cycle,
        "capacity": capacity,
        "model_name": model_name
    }
    
    try:
        response = requests.post(
            API_PREDICT_ENDPOINT,
            json=payload,
            timeout=API_TIMEOUT
        )
        
        # Handle different response codes
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        elif response.status_code == 422:
            # Validation error from backend
            error_detail = response.json().get('detail', {})
            error_msg = error_detail.get('message', 'Invalid input parameters')
            return {
                "success": False,
                "error_type": "validation",
                "message": error_msg
            }
        elif response.status_code == 503:
            # Service unavailable
            return {
                "success": False,
                "error_type": "service_unavailable",
                "message": "Prediction service is temporarily unavailable. Please try again later."
            }
        elif response.status_code == 500:
            # Server error
            error_detail = response.json().get('detail', {})
            error_msg = error_detail.get('message', 'Internal server error')
            return {
                "success": False,
                "error_type": "server_error",
                "message": f"Server error: {error_msg}"
            }
        else:
            return {
                "success": False,
                "error_type": "unknown",
                "message": f"Unexpected response (HTTP {response.status_code})"
            }
            
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to backend API")
        return {
            "success": False,
            "error_type": "connection",
            "message": "Cannot connect to prediction service. Please ensure the backend is running."
        }
    except requests.exceptions.Timeout:
        logger.error("Backend API request timed out")
        return {
            "success": False,
            "error_type": "timeout",
            "message": "Request timed out. Please try again."
        }
    except Exception as e:
        logger.error(f"API request error: {e}")
        return {
            "success": False,
            "error_type": "error",
            "message": f"An error occurred: {str(e)}"
        }


def predict_batch_via_api(file) -> Dict[str, Any]:
    """
    Make batch RUL prediction by uploading CSV to backend API.
    
    Args:
        file: Uploaded CSV file object
        
    Returns:
        dict: Batch prediction result or error information
    """
    try:
        files = {"file": (file.name, file.getvalue(), "text/csv")}
        response = requests.post(
            API_PREDICT_BATCH_ENDPOINT,
            files=files,
            timeout=API_TIMEOUT * 2  # Allow more time for batch processing
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            error_detail = response.json().get('detail', {})
            error_msg = error_detail.get('message', f'HTTP {response.status_code}')
            return {
                "success": False,
                "error_type": "server_error",
                "message": error_msg
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_type": "timeout",
            "message": "Batch prediction timed out. Try with fewer rows."
        }
    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return {
            "success": False,
            "error_type": "error",
            "message": str(e)
        }


def get_dataset_statistics() -> Dict[str, Any]:
    """Fetch dataset statistics from backend API."""
    try:
        response = requests.get(API_STATISTICS_ENDPOINT, timeout=API_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch statistics: {e}")
        return {}


def display_api_error(error_result: Dict[str, Any]):
    """
    Display user-friendly error message based on API error type.
    
    Args:
        error_result: Error information from API call
    """
    error_type = error_result.get('error_type', 'unknown')
    message = error_result.get('message', 'An unknown error occurred')
    
    if error_type == 'validation':
        st.error(f"⚠️ **Validation Error**: {message}")
        st.info("💡 Please check your input values are within valid ranges.")
    elif error_type == 'service_unavailable':
        st.error(f"🔧 **Service Unavailable**: {message}")
        st.warning("⏳ The prediction models may still be loading. Please wait and try again.")
    elif error_type == 'connection':
        st.error(f"🔌 **Connection Error**: {message}")
        st.info("💡 Make sure the backend server is running on the expected port.")
    elif error_type == 'timeout':
        st.error(f"⏱️ **Timeout**: {message}")
    else:
        st.error(f"❌ **Error**: {message}")


def get_health_badge(health_status: str) -> str:
    """
    Get HTML badge for battery health status.
    
    Args:
        health_status: Health classification from backend
        
    Returns:
        str: HTML formatted badge
    """
    health_lower = health_status.lower()
    if health_lower == 'healthy':
        return f'<span class="health-healthy">🟢 {health_status}</span>'
    elif health_lower == 'moderate':
        return f'<span class="health-moderate">🟡 {health_status}</span>'
    else:
        return f'<span class="health-critical">🔴 {health_status}</span>'


def get_confidence_badge(confidence: str) -> str:
    """
    Get HTML badge for confidence level.
    
    Args:
        confidence: Confidence level from backend
        
    Returns:
        str: HTML formatted badge
    """
    conf_lower = confidence.lower()
    if conf_lower == 'high':
        return f'<span class="confidence-high">✓ {confidence}</span>'
    elif conf_lower == 'medium':
        return f'<span class="confidence-medium">~ {confidence}</span>'
    else:
        return f'<span class="confidence-low">? {confidence}</span>'


def get_distribution_badge(status: str) -> str:
    """Get HTML badge for distribution status."""
    if status == 'in_distribution':
        return '<span class="dist-in">✓ In Distribution</span>'
    else:
        return '<span class="dist-out">⚠️ Out of Distribution</span>'


def get_life_stage_badge(stage: str) -> str:
    """Get HTML badge for life stage context."""
    if stage == 'early_life':
        return '<span class="stage-early">🌱 Early Life</span>'
    elif stage == 'mid_life':
        return '<span class="stage-mid">🔋 Mid Life</span>'
    else:
        return '<span class="stage-late">🔻 Late Life</span>'


def display_phase3_info(data: Dict[str, Any]):
    """
    Display Phase 3 dataset compatibility information.
    Shows distribution status, life stage, confidence explanation, and warnings.
    Phase 3.5: Enhanced with multi-dataset context.
    """
    distribution_status = data.get('distribution_status', 'unknown')
    life_stage = data.get('life_stage_context', 'unknown')
    confidence_explanation = data.get('confidence_explanation', '')
    inference_warning = data.get('inference_warning')
    
    # Phase 3.5: Multi-dataset fields
    dominant_dataset = data.get('dominant_dataset', 'NASA')
    cross_dataset_confidence = data.get('cross_dataset_confidence', 'medium')
    dataset_coverage_note = data.get('dataset_coverage_note', '')
    
    # Distribution and Life Stage badges
    st.markdown("### 📊 Dataset Compatibility Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔍 Distribution Status**")
        if distribution_status == 'in_distribution':
            st.success("✓ Within Training Distribution")
            st.caption("Input parameters match NASA dataset characteristics")
        else:
            st.warning("⚠️ Out of Distribution")
            st.caption("Some values fall outside typical training range")
    
    with col2:
        st.markdown("**🔋 Battery Life Stage**")
        if life_stage == 'early_life':
            st.info("🌱 Early Life Stage")
            st.caption("Battery shows early-life characteristics")
        elif life_stage == 'mid_life':
            st.success("🔋 Mid Life Stage")
            st.caption("Battery in typical operational phase")
        else:
            st.warning("🔻 Late Life Stage")
            st.caption("Battery approaching end-of-life")
    
    with col3:
        st.markdown("**📚 Dominant Dataset**")
        dataset_colors = {
            'NASA': ('🛰️', 'info'),
            'CALCE': ('🔬', 'success'),
            'OXFORD': ('🎓', 'warning'),
            'MATR1': ('📈', 'info')
        }
        icon, color = dataset_colors.get(dominant_dataset, ('📊', 'info'))
        if color == 'success':
            st.success(f"{icon} {dominant_dataset}")
        elif color == 'warning':
            st.warning(f"{icon} {dominant_dataset}")
        else:
            st.info(f"{icon} {dominant_dataset}")
        st.caption(f"Best matching dataset for input")
    
    # Cross-dataset confidence display
    st.markdown("### 🎯 Cross-Dataset Confidence")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if cross_dataset_confidence == 'high':
            st.success("🟢 HIGH")
            st.caption("Multiple datasets agree on lifecycle phase")
        elif cross_dataset_confidence == 'medium':
            st.warning("🟡 MEDIUM")
            st.caption("Single dataset match")
        else:
            st.error("🔴 LOW")
            st.caption("Weak match across datasets")
    
    with col2:
        # Dataset coverage explanation
        if dataset_coverage_note:
            st.markdown("**📝 Dataset Coverage Analysis:**")
            st.info(dataset_coverage_note)
    
    # Confidence explanation
    if confidence_explanation:
        st.markdown("### 💡 Prediction Context")
        st.info(confidence_explanation)
    
    # Inference warning
    if inference_warning:
        st.markdown("### ⚠️ Important Notice")
        st.warning(inference_warning)
    
    # Multi-dataset explanation expander
    with st.expander("ℹ️ About Multi-Dataset Analysis (Phase 3.5)", expanded=False):
        st.markdown("""
        **Phase 3.5 Enhancement: Multi-Dataset Expansion**
        
        This system now compares your input against multiple battery datasets to improve confidence estimation:
        
        | Dataset | Focus | Best For |
        |---------|-------|----------|
        | **NASA** | Late-life degradation | Batteries approaching EOL |
        | **CALCE** | Early-to-mid life | Fresh batteries, EV usage patterns |
        | **Oxford** | Mid-life, high-resolution | Precise degradation signals |
        | **MATR1** | Full lifecycle | Long-term cycling behavior |
        
        **Confidence Rules:**
        - 🟢 **High**: Input matches closest dataset AND at least one additional dataset agrees
        - 🟡 **Medium**: Strong match with a single dataset only
        - 🔴 **Low**: Weak or no match across all datasets
        
        **Note:** The ML models are trained exclusively on NASA data. Other datasets provide context 
        for confidence estimation only - they don't change the numerical prediction.
        """)


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def home_page():
    """Display home page."""
    st.markdown("<h1 class='main-header'>🔋 AI-Powered Battery RUL Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predict Remaining Useful Life of Lithium-Ion Batteries in Electric Vehicles</p>", unsafe_allow_html=True)
    
    # Backend status indicator
    health = check_backend_health()
    if health.get('status') == 'healthy':
        st.success(f"✅ Backend Connected | Models Available: {', '.join(health.get('available_models', []))}")
        st.session_state.available_models = health.get('available_models', [])
    else:
        st.warning(f"⚠️ Backend Status: {health.get('status', 'unknown')} - {health.get('error', '')}")
    
    # Hero section with image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1593941707882-a5bba14938c7?w=800", use_column_width=True)
    
    st.markdown("---")
    
    # Features
    st.markdown("### 🚀 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🤖 AI Models
        - XGBoost (Primary)
        - Random Forest
        - Linear Regression
        - LSTM Neural Network
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Analytics
        - Real-time RUL prediction
        - Model explainability (SHAP)
        - What-if simulations
        - Performance comparisons
        """)
    
    with col3:
        st.markdown("""
        #### 🌍 Sustainability
        - E-waste reduction tracking
        - CO₂ emission estimates
        - Cost savings analysis
        - Environmental impact
        """)
    
    st.markdown("---")
    
    # Sustainability stats
    st.markdown("### 🌱 Environmental Impact")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔋 Batteries Analyzed", "10,000+", "↑ 15%")
    with col2:
        st.metric("♻️ E-Waste Reduced", "5,000 kg", "↑ 20%")
    with col3:
        st.metric("🌍 CO₂ Saved", "11,500 kg", "↑ 18%")
    with col4:
        st.metric("💰 Cost Savings", "$150,000", "↑ 22%")
    
    st.markdown("---")
    
    # Call to action
    st.markdown("### 🎯 Get Started")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Predict RUL", use_container_width=True, type="primary"):
            st.session_state.current_page = "Predict RUL"
            st.rerun()
    
    with col2:
        if st.button("🔬 What-If Analysis", use_container_width=True):
            st.session_state.current_page = "What-If"
            st.rerun()
    
    with col3:
        if st.button("ℹ️ About", use_container_width=True):
            st.session_state.current_page = "About"
            st.rerun()


def predict_rul_page():
    """Display RUL prediction page - uses backend API for all predictions."""
    st.title("📈 Battery RUL Prediction")
    st.markdown("Input battery parameters to predict remaining useful life via our AI backend.")
    
    # Check backend status
    health = check_backend_health()
    if health.get('status') != 'healthy':
        st.error(f"⚠️ Backend service unavailable: {health.get('error', 'Unknown error')}")
        st.info("💡 Please ensure the backend server is running and try again.")
        if st.button("🔄 Retry Connection"):
            st.rerun()
        return
    
    # Get available models from backend
    available_models = health.get('available_models', ['XGBoost'])
    
    # Show backend info
    with st.expander("📊 Backend Service Info", expanded=False):
        st.json({
            "status": health.get('status'),
            "service": health.get('service'),
            "models_loaded": health.get('models_loaded'),
            "available_models": available_models
        })
    
    # Input method selection
    input_method = st.radio("📥 Input Method", ["Manual Input", "Upload CSV"], horizontal=True)
    
    if input_method == "Manual Input":
        st.markdown("### ⚙️ Battery Parameters")
        st.info("💡 Based on NASA Li-ion Battery Dataset: Batteries typically reach End-of-Life (80% capacity) within 50-170 cycles.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "🌡️ Temperature (°C)", 
                0, 70, 24, 
                help="Ambient temperature during operation (0-70°C)"
            )
            voltage = st.slider(
                "⚡ Voltage (V)", 
                2.0, 4.5, 3.7, 0.05, 
                help="Average measured voltage (2.0-4.5V)"
            )
            cycle_count = st.number_input(
                "🔄 Cycle Count", 
                0, 500, 50, 1, 
                help="Current charge-discharge cycle number"
            )
        
        with col2:
            current = st.slider(
                "⚡ Current (A)", 
                -2.0, 0.0, -1.0, 0.1, 
                help="Discharge current (typically negative)"
            )
            capacity = st.slider(
                "🔋 Current Capacity (Ah)", 
                0.5, 2.5, 1.8, 0.05, 
                help="Current measured capacity"
            )
            model_choice = st.selectbox(
                "🤖 Model", 
                available_models,
                index=available_models.index("XGBoost") if "XGBoost" in available_models else 0,
                help="Select ML model for prediction"
            )
        
        if st.button("🚀 Predict RUL", use_container_width=True, type="primary"):
            with st.spinner("🔮 Calling prediction API..."):
                # Call backend API for prediction
                result = predict_rul_via_api(
                    voltage=voltage,
                    current=current,
                    temperature=temperature,
                    cycle=cycle_count,
                    capacity=capacity,
                    model_name=model_choice
                )
                
                if result['success']:
                    data = result['data']
                    predicted_rul = data['predicted_rul_cycles']
                    battery_health = data['battery_health']
                    confidence = data['confidence_level']
                    model_used = data['model_used']
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔋 Predicted RUL", f"{predicted_rul} cycles")
                    
                    with col2:
                        years = predicted_rul / 365
                        st.metric("📅 Estimated Time", f"{years:.1f} years")
                    
                    with col3:
                        st.markdown(f"**💚 Battery Health**")
                        st.markdown(get_health_badge(battery_health), unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"**📊 Confidence**")
                        st.markdown(get_confidence_badge(confidence), unsafe_allow_html=True)
                    
                    # Model info footer
                    st.caption(f"🤖 Model used: {model_used}")
                    
                    st.markdown("---")
                    
                    # Phase 3: Display dataset compatibility information
                    display_phase3_info(data)
                    
                    st.markdown("---")
                    
                    # Capacity fade curve
                    st.markdown("### 📉 Capacity Fade Projection")
                    st.plotly_chart(
                        plot_capacity_fade_curve(cycle_count, predicted_rul, capacity),
                        use_container_width=True
                    )
                    
                    # Sustainability impact
                    st.markdown("### 🌍 Environmental Impact")
                    st.plotly_chart(
                        plot_sustainability_impact(predicted_rul),
                        use_container_width=True
                    )
                    
                    # Export options
                    st.markdown("### 📥 Export Results")
                    col1, col2 = st.columns(2)
                    
                    prediction_data = {
                        'rul': f"{predicted_rul}",
                        'time_estimate': f"{years:.1f}",
                        'cycle': cycle_count,
                        'temperature': temperature,
                        'voltage': voltage,
                        'current': current,
                        'capacity': capacity,
                        'model': model_used,
                        'battery_health': battery_health,
                        'confidence': confidence,
                        'distribution_status': data.get('distribution_status', 'unknown'),
                        'life_stage_context': data.get('life_stage_context', 'unknown'),
                        'confidence_explanation': data.get('confidence_explanation', ''),
                        'inference_warning': data.get('inference_warning', '')
                    }
                    
                    with col1:
                        csv_data = generate_csv_report(prediction_data)
                        if csv_data:
                            st.download_button(
                                "📊 Download CSV",
                                csv_data,
                                "battery_rul_report.csv",
                                "text/csv",
                                use_container_width=True
                            )
                    
                    with col2:
                        pdf_data = generate_pdf_report(
                            prediction_data, 
                            st.session_state.get('username', 'User')
                        )
                        if pdf_data:
                            st.download_button(
                                "📄 Download PDF",
                                pdf_data,
                                "battery_rul_report.pdf",
                                "application/pdf",
                                use_container_width=True
                            )
                    
                    st.balloons()
                else:
                    # Display error from API
                    display_api_error(result)
    
    else:  # Upload CSV
        st.markdown("### 📤 Batch Prediction via CSV Upload")
        st.markdown("""
        Upload a CSV file with battery parameters to get batch RUL predictions.
        
        **Required columns:** `voltage`, `current`, `temperature`, `cycle`, `capacity`
        """)
        
        # Download template button
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("📥 Download Template", use_container_width=True):
                template_data = {
                    'voltage': [3.7, 3.5, 3.3],
                    'current': [-1.0, -1.2, -0.8],
                    'temperature': [25, 30, 35],
                    'cycle': [50, 100, 150],
                    'capacity': [1.8, 1.6, 1.4]
                }
                template_df = pd.DataFrame(template_data)
                st.download_button(
                    "Download CSV Template",
                    template_df.to_csv(index=False),
                    "batch_prediction_template.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="File must contain columns: voltage, current, temperature, cycle, capacity"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded: {len(df)} rows")
                
                # Validate columns
                required_cols = ['voltage', 'current', 'temperature', 'cycle', 'capacity']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info(f"Required columns: {required_cols}")
                else:
                    st.markdown("**Preview:**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("🚀 Run Batch Prediction", use_container_width=True, type="primary"):
                        with st.spinner(f"🔮 Processing {len(df)} predictions..."):
                            # Reset file pointer and call batch API
                            uploaded_file.seek(0)
                            batch_result = predict_batch_via_api(uploaded_file)
                            
                            if batch_result['success']:
                                data = batch_result['data']
                                
                                # Summary metrics
                                st.success(f"✅ Batch prediction complete!")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📊 Total Rows", data['total_rows'])
                                with col2:
                                    st.metric("✅ Processed", data['processed_rows'])
                                with col3:
                                    st.metric("❌ Failed", data['failed_rows'])
                                with col4:
                                    st.metric("📈 Avg RUL", f"{data['summary'].get('avg_rul', 0)} cycles")
                                
                                st.markdown("---")
                                
                                # Summary statistics
                                st.markdown("### 📊 Batch Summary")
                                summary = data['summary']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("🟢 Healthy", summary.get('healthy_count', 0))
                                with col2:
                                    st.metric("🟡 Moderate", summary.get('moderate_count', 0))
                                with col3:
                                    st.metric("🔴 Critical", summary.get('critical_count', 0))
                                with col4:
                                    st.metric("⚠️ Out-of-Distribution", summary.get('ood_count', 0))
                                
                                st.markdown("---")
                                
                                # Results table
                                st.markdown("### 📋 Detailed Results")
                                
                                results_df = pd.DataFrame(data['results'])
                                
                                # Color code the dataframe
                                def style_health(val):
                                    if val == 'Healthy':
                                        return 'background-color: #d4edda'
                                    elif val == 'Moderate':
                                        return 'background-color: #fff3cd'
                                    else:
                                        return 'background-color: #f8d7da'
                                
                                # Display results
                                st.dataframe(
                                    results_df[[
                                        'row_index', 'predicted_rul_cycles', 'battery_health',
                                        'distribution_status', 'life_stage_context', 
                                        'confidence_level', 'dominant_dataset', 
                                        'cross_dataset_confidence', 'inference_warning'
                                    ]].rename(columns={
                                        'row_index': 'Row',
                                        'predicted_rul_cycles': 'RUL (cycles)',
                                        'battery_health': 'Health',
                                        'distribution_status': 'Distribution',
                                        'life_stage_context': 'Life Stage',
                                        'confidence_level': 'Confidence',
                                        'dominant_dataset': 'Dataset',
                                        'cross_dataset_confidence': 'Cross-DS Conf',
                                        'inference_warning': 'Warning'
                                    }),
                                    use_container_width=True
                                )
                                
                                # Download results
                                st.markdown("### 📥 Export Results")
                                csv_output = results_df.to_csv(index=False)
                                st.download_button(
                                    "📊 Download Results CSV",
                                    csv_output,
                                    "batch_prediction_results.csv",
                                    "text/csv",
                                    use_container_width=True
                                )
                                
                                # OOD explanation if any
                                if summary.get('ood_count', 0) > 0:
                                    st.markdown("---")
                                    st.markdown("### ℹ️ About Out-of-Distribution Results")
                                    st.info("""
                                    Some inputs were flagged as **out-of-distribution** because their values 
                                    fall outside the typical range seen in the NASA training dataset 
                                    (5th-95th percentile bounds). These predictions may be less reliable.
                                    
                                    The NASA dataset focuses on batteries in mid-to-late life stages, 
                                    so early-life batteries may receive conservative RUL estimates.
                                    """)
                                
                            else:
                                display_api_error(batch_result)
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")


def what_if_page():
    """Display what-if analysis page - uses backend API for predictions."""
    st.title("🔬 What-If Scenario Analysis")
    st.markdown("Explore how changing battery parameters affects the predicted RUL.")
    
    # Check backend status
    health = check_backend_health()
    if health.get('status') != 'healthy':
        st.error(f"⚠️ Backend service unavailable: {health.get('error', 'Unknown error')}")
        return
    
    # Base parameters
    st.markdown("### 🎯 Base Scenario")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_temp = st.number_input("Temperature (°C)", 0, 70, 35)
    with col2:
        base_voltage = st.number_input("Voltage (V)", 2.0, 4.5, 3.7, 0.1)
    with col3:
        base_current = st.number_input("Current (A)", -2.0, 0.0, -1.0, 0.1)
    
    col1, col2 = st.columns(2)
    with col1:
        base_cycle = st.number_input("Cycle Count", 0, 500, 50)
    with col2:
        base_capacity = st.number_input("Capacity (Ah)", 0.5, 2.5, 1.8, 0.1)
    
    # Calculate base RUL
    if st.button("📊 Run Analysis", use_container_width=True, type="primary"):
        with st.spinner("🔮 Running simulations via API..."):
            # Base prediction
            base_result = predict_rul_via_api(
                base_voltage, base_current, base_temp, base_cycle, base_capacity, "XGBoost"
            )
            
            if not base_result['success']:
                display_api_error(base_result)
                return
            
            base_rul = base_result['data']['predicted_rul_cycles']
            st.success(f"✅ Base RUL: **{base_rul} cycles** ({base_result['data']['battery_health']})")
            
            # What-if scenarios
            st.markdown("### 🔄 Scenario Comparisons")
            
            scenarios = {}
            scenario_configs = [
                ("Temp -5°C", base_voltage, base_current, base_temp - 5, base_cycle, base_capacity),
                ("Temp +5°C", base_voltage, base_current, base_temp + 5, base_cycle, base_capacity),
                ("Current -0.3A", base_voltage, base_current - 0.3, base_temp, base_cycle, base_capacity),
                ("Current +0.3A", base_voltage, min(0, base_current + 0.3), base_temp, base_cycle, base_capacity),
                ("Capacity +0.1Ah", base_voltage, base_current, base_temp, base_cycle, min(2.5, base_capacity + 0.1)),
                ("Capacity -0.1Ah", base_voltage, base_current, base_temp, base_cycle, max(0.5, base_capacity - 0.1)),
            ]
            
            progress_bar = st.progress(0)
            for i, (name, v, c, t, cy, cap) in enumerate(scenario_configs):
                result = predict_rul_via_api(v, c, t, cy, cap, "XGBoost")
                if result['success']:
                    scenarios[name] = result['data']['predicted_rul_cycles']
                progress_bar.progress((i + 1) / len(scenario_configs))
            
            progress_bar.empty()
            
            if scenarios:
                # Display comparison
                st.plotly_chart(
                    plot_what_if_comparison(base_rul, scenarios),
                    use_container_width=True
                )
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                best_scenario = max(scenarios.items(), key=lambda x: x[1])
                worst_scenario = min(scenarios.items(), key=lambda x: x[1])
                
                if best_scenario[1] > base_rul:
                    improvement = best_scenario[1] - base_rul
                    st.success(f"🏆 Best scenario: **{best_scenario[0]}** improves RUL by **{improvement} cycles**")
                
                if worst_scenario[1] < base_rul:
                    degradation = base_rul - worst_scenario[1]
                    st.warning(f"⚠️ Worst scenario: **{worst_scenario[0]}** reduces RUL by **{degradation} cycles**")
                
                # Specific recommendations
                recommendations = []
                if scenarios.get('Temp -5°C', 0) > base_rul:
                    recommendations.append("🌡️ Reducing operating temperature can extend battery life")
                if scenarios.get('Current -0.3A', 0) > base_rul:
                    recommendations.append("⚡ Lower discharge current improves longevity")
                
                for rec in recommendations:
                    st.info(rec)
            else:
                st.error("❌ Failed to run scenario simulations")


def about_page():
    """Display about page."""
    st.title("ℹ️ About")
    
    # Backend architecture info
    with st.expander("🏗️ System Architecture", expanded=True):
        st.markdown("""
        ### Client-Server Architecture
        
        This application uses a **decoupled architecture**:
        
        - **Frontend (Streamlit)**: User interface and visualization
        - **Backend (FastAPI)**: ML inference and business logic
        
        ```
        ┌─────────────┐     HTTP/REST      ┌─────────────┐
        │  Streamlit  │ ◄───────────────► │   FastAPI   │
        │  Frontend   │    /api/predict    │   Backend   │
        └─────────────┘                    └─────────────┘
                                                  │
                                                  ▼
                                          ┌─────────────┐
                                          │  ML Models  │
                                          │ (XGB, RF,   │
                                          │  LR, LSTM)  │
                                          └─────────────┘
        ```
        
        **Benefits:**
        - 🔄 Independent scaling of frontend and backend
        - 🔌 Backend can serve multiple clients (web, mobile, API)
        - 🛡️ Centralized ML logic and model versioning
        - 🚀 Easier deployment and maintenance
        """)
    
    st.markdown("""
    ## 🔋 Battery RUL Prediction System
    
    This application uses advanced machine learning techniques to predict the Remaining Useful Life (RUL) 
    of lithium-ion batteries in electric vehicles. By analyzing battery parameters such as voltage, current, 
    temperature, and cycle count, our AI models can accurately forecast when a battery will reach its 
    end-of-life threshold (80% of original capacity).
    
    ### 🤖 Machine Learning Models
    
    - **XGBoost**: Gradient boosting algorithm optimized for performance (Primary model)
    - **Random Forest**: Ensemble learning method for robust predictions
    - **Linear Regression**: Baseline model for comparison
    - **LSTM**: Deep learning model for sequential pattern recognition
    
    ### 📊 Features
    
    - Real-time RUL prediction via REST API
    - What-if scenario analysis
    - Sustainability impact tracking
    - CSV/PDF report generation
    - Interactive visualizations
    
    ### 🌍 Environmental Impact
    
    Accurate battery health prediction helps:
    - Reduce electronic waste by optimizing replacement timing
    - Lower CO₂ emissions through extended battery life
    - Save costs by preventing premature replacements
    - Support circular economy through better recycling planning
    
    ### 📚 Dataset
    
    This system uses the NASA Randomized Battery Usage Dataset, which contains real-world battery 
    degradation data from controlled experiments.
    
    ### 🔗 API Endpoints
    
    | Endpoint | Method | Description |
    |----------|--------|-------------|
    | `/api/health` | GET | Health check and model status |
    | `/api/models` | GET | List available models |
    | `/api/predict` | POST | Make RUL prediction |
    | `/api/predict/batch` | POST | Batch CSV prediction |
    | `/api/statistics` | GET | Dataset statistics |
    
    ---
    
    **Version**: 3.0.0 (Phase 3 - Dataset Compatibility Verification)  
    **Last Updated**: 2025
    """)


def dataset_statistics_page():
    """Display dataset statistics for Phase 3 transparency."""
    st.title("📊 Training Dataset Statistics")
    st.markdown("""
    Understanding the NASA Li-ion Battery Dataset characteristics helps explain 
    prediction behavior and confidence levels.
    """)
    
    # Fetch statistics from backend
    stats = get_dataset_statistics()
    
    if not stats:
        st.error("❌ Unable to fetch dataset statistics from backend")
        return
    
    # Metadata section
    st.markdown("### 📁 Dataset Metadata")
    metadata = stats.get('metadata', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Samples", f"{metadata.get('total_samples', 'N/A'):,}")
    with col2:
        st.metric("🔋 Batteries Analyzed", metadata.get('batteries_analyzed', 'N/A'))
    with col3:
        st.metric("📅 Generated", metadata.get('generation_date', 'N/A')[:10] if metadata.get('generation_date') else 'N/A')
    
    st.info(f"**Source:** {metadata.get('dataset_source', 'NASA Li-ion Battery Dataset')}")
    
    st.markdown("---")
    
    # Feature statistics section
    st.markdown("### 📈 Feature Statistics")
    st.markdown("""
    These statistics define the bounds for **out-of-distribution (OOD) detection**.
    Values outside the 5th-95th percentile range are flagged as OOD.
    """)
    
    features = stats.get('features', {})
    
    if features:
        # Create a DataFrame for display
        feature_data = []
        for feature_name, feature_stats in features.items():
            feature_data.append({
                'Feature': feature_name,
                'Min': f"{feature_stats.get('minimum', 0):.3f}",
                '5th %': f"{feature_stats.get('percentile_5', 0):.3f}",
                '25th %': f"{feature_stats.get('percentile_25', 0):.3f}",
                'Median': f"{feature_stats.get('median', 0):.3f}",
                'Mean': f"{feature_stats.get('mean', 0):.3f}",
                '75th %': f"{feature_stats.get('percentile_75', 0):.3f}",
                '95th %': f"{feature_stats.get('percentile_95', 0):.3f}",
                'Max': f"{feature_stats.get('maximum', 0):.3f}",
                'Std Dev': f"{feature_stats.get('standard_deviation', 0):.3f}",
                'Count': feature_stats.get('count', 0)
            })
        
        stats_df = pd.DataFrame(feature_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Individual feature details
        st.markdown("### 🔍 Feature Details")
        
        for feature_name, feature_stats in features.items():
            with st.expander(f"📊 {feature_name.replace('_', ' ').title()}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Statistical Summary:**")
                    st.write(f"- **Description:** {feature_stats.get('description', 'N/A')}")
                    st.write(f"- **Mean:** {feature_stats.get('mean', 0):.4f}")
                    st.write(f"- **Median:** {feature_stats.get('median', 0):.4f}")
                    st.write(f"- **Std Deviation:** {feature_stats.get('standard_deviation', 0):.4f}")
                
                with col2:
                    st.markdown("**OOD Detection Bounds:**")
                    p5 = feature_stats.get('percentile_5', 0)
                    p95 = feature_stats.get('percentile_95', 0)
                    st.write(f"- **Lower Bound (5th %):** {p5:.4f}")
                    st.write(f"- **Upper Bound (95th %):** {p95:.4f}")
                    st.write(f"- **Valid Range:** [{p5:.3f}, {p95:.3f}]")
    
    st.markdown("---")
    
    # Dataset context section
    st.markdown("### ⚠️ Dataset Bias & Limitations")
    context = stats.get('dataset_context', {})
    
    if context:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Characteristics:**")
            st.write(f"- **Late-Life Bias:** {'Yes' if context.get('late_life_bias') else 'No'}")
            st.write(f"- **Dominant Cycle Range:** {context.get('cycle_range_dominant', 'N/A')}")
            st.write(f"- **Dominant Capacity Range:** {context.get('capacity_range_dominant', 'N/A')}")
        
        with col2:
            st.markdown("**End-of-Life Definition:**")
            eol = context.get('eol_threshold', 0.8)
            initial = context.get('initial_capacity_nominal', 2.0)
            st.write(f"- **EOL Threshold:** {eol*100:.0f}% of initial capacity")
            st.write(f"- **Nominal Initial Capacity:** {initial} Ah")
            st.write(f"- **EOL Capacity:** {initial * eol:.2f} Ah")
    
    # Important note
    st.markdown("---")
    st.warning("""
    **Important Note for Users:**
    
    The NASA dataset primarily contains data from batteries in **mid-to-late life stages** 
    (aging phase). This means:
    
    1. **Early-life batteries** (high capacity, low cycle count) may receive **conservative RUL predictions**
    2. Predictions are most accurate for batteries matching the training data characteristics
    3. Out-of-distribution warnings indicate reduced prediction reliability
    
    This is scientifically appropriate for safety-critical applications where conservative 
    estimates are preferable to overestimates.
    """)


def main():
    """Main application logic."""
    # Check authentication
    if not check_authentication():
        login_page()
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown(f"### 👤 Welcome, {st.session_state.username}!")
        st.markdown("---")
        
        # Backend status indicator
        health = check_backend_health()
        if health.get('status') == 'healthy':
            st.success("🟢 Backend Online")
        else:
            st.error("🔴 Backend Offline")
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        pages = {
            "🏠 Home": "Home",
            "📈 Predict RUL": "Predict RUL",
            "🔬 What-If Analysis": "What-If",
            "📊 Dataset Statistics": "Dataset Statistics",
            "ℹ️ About": "About"
        }
        
        for label, page in pages.items():
            if st.button(label, use_container_width=True, key=page):
                st.session_state.current_page = page
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Route to appropriate page
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Predict RUL":
        predict_rul_page()
    elif st.session_state.current_page == "What-If":
        what_if_page()
    elif st.session_state.current_page == "Dataset Statistics":
        dataset_statistics_page()
    elif st.session_state.current_page == "About":
        about_page()


if __name__ == "__main__":
    main()
