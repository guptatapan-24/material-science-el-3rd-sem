# 🔋 Phase 3.5 Testing Guide: Multi-Dataset Expansion for Battery RUL Prediction

## 📋 Complete End-to-End Testing Guide for Windows 11

This guide provides complete steps, commands, and expected outcomes to test Phase 3.5 implementation from start to end, covering both backend and frontend.

---

## 🎯 Phase 3.5 Overview

**Goal**: Extend the validated and explainable battery RUL prediction system by integrating multiple complementary lithium-ion battery datasets (NASA, CALCE, Oxford, MATR1) to improve lifecycle coverage, generalization, and confidence estimation.

### Key Features to Test:
1. ✅ Multi-dataset statistics files (NASA, CALCE, OXFORD, MATR1)
2. ✅ Dataset-aware inference logic
3. ✅ Cross-dataset confidence scoring
4. ✅ Enhanced API responses with dominant dataset and coverage notes
5. ✅ Extended batch prediction with multi-dataset columns
6. ✅ Frontend display of multi-dataset analysis

---

## 🔧 Prerequisites & Environment Setup (Windows 11)

### Step 1: Install Required Software

```powershell
# Check Python version (Python 3.8+ required)
python --version

# Check if pip is available
pip --version

# Check if Node.js/npm is installed (for frontend if using React)
node --version
npm --version
```

### Step 2: Clone/Navigate to Project

```powershell
# Navigate to your project directory
cd C:\path\to\battery-rul-prediction

# Or if using WSL/Git Bash, navigate accordingly
```

### Step 3: Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# OR for CMD:
.\venv\Scripts\activate.bat

# OR for Git Bash:
source venv/Scripts/activate
```

### Step 4: Install Backend Dependencies

```powershell
# Navigate to backend directory
cd backend

# Install requirements
pip install -r requirements.txt

# Key packages needed:
# - fastapi==0.110.1
# - uvicorn==0.25.0
# - pandas
# - numpy
# - scikit-learn
# - xgboost
# - joblib
# - python-multipart (for file uploads)
```

### Step 5: Install Frontend Dependencies (if applicable)

```powershell
# Navigate to frontend/streamlit directory
cd ..  # back to project root

# Install Streamlit dependencies
pip install streamlit plotly requests
```

---

## 🚀 Starting the Services

### Step 1: Start Backend Server

Open a new terminal/PowerShell window:

```powershell
# Navigate to backend directory
cd C:\path\to\battery-rul-prediction\backend

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start FastAPI backend with uvicorn
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**Expected Output:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Loading models...
INFO:     Multi-dataset statistics initialized for: ['NASA', 'CALCE', 'OXFORD', 'MATR1']
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

### Step 2: Start Frontend (Streamlit)

Open another terminal/PowerShell window:

```powershell
# Navigate to project root
cd C:\path\to\battery-rul-prediction

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set backend URL environment variable (optional, defaults to localhost:8001)
$env:BACKEND_URL = "http://localhost:8001"

# Start Streamlit frontend
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://xxx.xxx.xxx.xxx:8501
```

---

## 🧪 BACKEND TESTING

### Test 1: Health Check Endpoint

**Command:**
```powershell
curl http://localhost:8001/api/health
```

**Or using PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8001/api/health" -Method Get | ConvertTo-Json
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "battery-rul-prediction",
  "models_loaded": true,
  "available_models": ["XGBoost", "Random Forest", "Linear Regression"]
}
```

---

### Test 2: Multi-Dataset Statistics Files Existence

**Command (PowerShell):**
```powershell
# Check all 4 dataset statistics files exist
Get-ChildItem -Path ".\data\*_statistics.json" | Select-Object Name, Length, LastWriteTime
```

**Expected Output:**
```
Name                      Length LastWriteTime
----                      ------ -------------
nasa_statistics.json        2048 xx/xx/xxxx
calce_statistics.json       1856 xx/xx/xxxx
oxford_statistics.json      1834 xx/xx/xxxx
matr1_statistics.json       1867 xx/xx/xxxx
```

---

### Test 3: Dataset Statistics Endpoint

**Command:**
```powershell
curl http://localhost:8001/api/statistics
```

**Or PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8001/api/statistics" -Method Get | ConvertTo-Json -Depth 5
```

**Expected Response (truncated):**
```json
{
  "metadata": {
    "dataset_source": "NASA Li-ion Battery Aging Dataset",
    "total_samples": 1108,
    "batteries_analyzed": 11
  },
  "features": {
    "cycle": { "minimum": 1.0, "maximum": 168.0, "mean": 60.95 },
    "capacity": { "minimum": 0.61, "maximum": 2.04, "mean": 1.46 }
  },
  "dataset_context": {
    "late_life_bias": true,
    "lifecycle_coverage": "late_life"
  }
}
```

---

### Test 4: Single Prediction - Early Life Input (Should Match CALCE)

**Command:**
```powershell
curl -X POST "http://localhost:8001/api/predict" -H "Content-Type: application/json" -d "{\"voltage\": 3.8, \"current\": -1.0, \"temperature\": 25, \"cycle\": 20, \"capacity\": 1.95}"
```

**PowerShell:**
```powershell
$body = @{
    voltage = 3.8
    current = -1.0
    temperature = 25
    cycle = 20
    capacity = 1.95
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/api/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 3
```

**Expected Response:**
```json
{
  "predicted_rul_cycles": 180,
  "battery_health": "Healthy",
  "confidence_level": "High",
  "model_used": "XGBoost",
  "distribution_status": "in_distribution",
  "life_stage_context": "early_life",
  "confidence_explanation": "...",
  "inference_warning": null,
  "dominant_dataset": "CALCE",        // ← Key Phase 3.5 field
  "cross_dataset_confidence": "high", // ← Key Phase 3.5 field
  "dataset_coverage_note": "Input best matches CALCE dataset (score: XX%). Input characteristics suggest early battery life..."
}
```

**Validation Criteria:**
- ✅ `dominant_dataset` should be `"CALCE"` (early-life input)
- ✅ `cross_dataset_confidence` should be `"high"` or `"medium"`
- ✅ `dataset_coverage_note` should mention early battery life
- ✅ `life_stage_context` should be `"early_life"`

---

### Test 5: Single Prediction - Late Life Input (Should Match NASA)

**Command:**
```powershell
curl -X POST "http://localhost:8001/api/predict" -H "Content-Type: application/json" -d "{\"voltage\": 3.3, \"current\": -1.5, \"temperature\": 35, \"cycle\": 150, \"capacity\": 1.3}"
```

**Expected Response:**
```json
{
  "predicted_rul_cycles": 35,
  "battery_health": "Critical",
  "confidence_level": "High",
  "distribution_status": "in_distribution",
  "life_stage_context": "late_life",
  "dominant_dataset": "NASA",           // ← Should be NASA for late-life
  "cross_dataset_confidence": "high",
  "dataset_coverage_note": "Input best matches NASA dataset... late battery life (approaching EOL)..."
}
```

**Validation Criteria:**
- ✅ `dominant_dataset` should be `"NASA"` (late-life input)
- ✅ `life_stage_context` should be `"late_life"`
- ✅ RUL should be low (< 100 cycles)

---

### Test 6: Single Prediction - Mid Life Input (Should Match OXFORD)

**Command:**
```powershell
curl -X POST "http://localhost:8001/api/predict" -H "Content-Type: application/json" -d "{\"voltage\": 3.6, \"current\": -0.9, \"temperature\": 26, \"cycle\": 100, \"capacity\": 0.95}"
```

**Expected Response:**
```json
{
  "predicted_rul_cycles": 120,
  "battery_health": "Moderate",
  "life_stage_context": "mid_life",
  "dominant_dataset": "OXFORD",         // ← Should match Oxford mid-life
  "cross_dataset_confidence": "medium",
  "dataset_coverage_note": "Input best matches OXFORD dataset... mid battery life..."
}
```

---

### Test 7: Single Prediction - High Capacity Long Cycle (Should Match MATR1)

**Command:**
```powershell
curl -X POST "http://localhost:8001/api/predict" -H "Content-Type: application/json" -d "{\"voltage\": 3.6, \"current\": -1.5, \"temperature\": 28, \"cycle\": 300, \"capacity\": 2.5}"
```

**Expected Response:**
```json
{
  "dominant_dataset": "MATR1",          // ← High capacity, long cycle matches MATR1
  "cross_dataset_confidence": "medium",
  "dataset_coverage_note": "... MATR1 covers extended lifecycle..."
}
```

---

### Test 8: Out-of-Distribution (OOD) Detection

**Command (extreme values):**
```powershell
curl -X POST "http://localhost:8001/api/predict" -H "Content-Type: application/json" -d "{\"voltage\": 2.2, \"current\": -0.1, \"temperature\": 65, \"cycle\": 5, \"capacity\": 2.4}"
```

**Expected Response:**
```json
{
  "distribution_status": "out_of_distribution",
  "confidence_level": "Low",
  "inference_warning": "⚠️ Input values for [...] are outside the training data distribution...",
  "cross_dataset_confidence": "low"
}
```

**Validation Criteria:**
- ✅ `distribution_status` should be `"out_of_distribution"`
- ✅ `confidence_level` should be reduced (`"Low"` or `"Medium"`)
- ✅ `inference_warning` should be present
- ✅ `cross_dataset_confidence` should be `"low"`

---

### Test 9: Batch Prediction via CSV Upload

**Step 1: Create Test CSV File**

Create a file `test_batch.csv` with content:

```csv
voltage,current,temperature,cycle,capacity
3.8,-1.0,25,20,1.95
3.5,-1.2,30,80,1.6
3.3,-1.5,35,150,1.3
3.6,-0.9,26,100,0.95
3.7,-1.3,28,300,2.5
```

**Step 2: Upload via curl**

```powershell
curl -X POST "http://localhost:8001/api/predict/batch" -F "file=@test_batch.csv"
```

**Or PowerShell:**
```powershell
$filePath = ".\test_batch.csv"
$response = Invoke-RestMethod -Uri "http://localhost:8001/api/predict/batch" -Method Post -InFile $filePath -ContentType "multipart/form-data"
$response | ConvertTo-Json -Depth 5
```

**Expected Response:**
```json
{
  "success": true,
  "total_rows": 5,
  "processed_rows": 5,
  "failed_rows": 0,
  "results": [
    {
      "row_index": 0,
      "predicted_rul_cycles": 180,
      "battery_health": "Healthy",
      "distribution_status": "in_distribution",
      "life_stage_context": "early_life",
      "dominant_dataset": "CALCE",              // ← Phase 3.5 field
      "cross_dataset_confidence": "high"        // ← Phase 3.5 field
    },
    {
      "row_index": 1,
      "dominant_dataset": "NASA",
      "cross_dataset_confidence": "medium"
    },
    // ... more rows
  ],
  "summary": {
    "avg_rul": 150,
    "healthy_count": 2,
    "moderate_count": 2,
    "critical_count": 1,
    "ood_count": 0
  }
}
```

**Validation Criteria:**
- ✅ Each row should have `dominant_dataset` field
- ✅ Each row should have `cross_dataset_confidence` field
- ✅ Different rows should match different datasets based on their lifecycle stage

---

### Test 10: Available Models Endpoint

**Command:**
```powershell
curl http://localhost:8001/api/models
```

**Expected Response:**
```json
{
  "models": ["Linear Regression", "Random Forest", "XGBoost"],
  "default": "XGBoost"
}
```

---

### Test 11: API Documentation Access

**Test:** Open browser and navigate to:
- Swagger UI: `http://localhost:8001/api/docs`
- ReDoc: `http://localhost:8001/api/redoc`

**Expected:** Interactive API documentation with all endpoints documented

---

## 🎨 FRONTEND TESTING (Streamlit)

### Test 12: Frontend Connection & Backend Status

1. **Navigate to:** `http://localhost:8501`
2. **Login:** Use credentials `admin` / `battery123`

**Expected:**
- ✅ "Backend Online" indicator in sidebar (green)
- ✅ Homepage loads with "Backend Connected" message

---

### Test 13: Single Prediction with Multi-Dataset Display

1. **Navigate to:** "Predict RUL" page
2. **Enter Early-Life Parameters:**
   - Temperature: 25°C
   - Voltage: 3.8V
   - Cycle Count: 20
   - Current: -1.0A
   - Capacity: 1.95 Ah
3. **Click:** "🚀 Predict RUL"

**Expected UI Elements:**
- ✅ **Predicted RUL**: ~180+ cycles
- ✅ **Battery Health**: "Healthy" (green)
- ✅ **Distribution Status**: "✓ Within Training Distribution"
- ✅ **Battery Life Stage**: "🌱 Early Life Stage"
- ✅ **Dominant Dataset**: Shows "CALCE" with icon
- ✅ **Cross-Dataset Confidence**: Shows "HIGH" or "MEDIUM" badge
- ✅ **Dataset Coverage Analysis**: Shows detailed explanation

---

### Test 14: Multi-Dataset Information Expander

1. After prediction, **click** on "ℹ️ About Multi-Dataset Analysis (Phase 3.5)"

**Expected:**
- ✅ Expander opens showing table with 4 datasets
- ✅ Table shows: NASA, CALCE, OXFORD, MATR1 with their focus areas
- ✅ Confidence rules explanation (High/Medium/Low)

---

### Test 15: Late-Life Prediction Display

1. **Enter Late-Life Parameters:**
   - Temperature: 35°C
   - Voltage: 3.3V
   - Cycle Count: 150
   - Current: -1.5A
   - Capacity: 1.3 Ah
2. **Click:** "🚀 Predict RUL"

**Expected:**
- ✅ **Predicted RUL**: Low (<50 cycles)
- ✅ **Battery Health**: "Critical" (red)
- ✅ **Life Stage**: "🔻 Late Life Stage"
- ✅ **Dominant Dataset**: Shows "NASA"
- ✅ **Dataset Coverage Note**: Mentions late-life/approaching EOL

---

### Test 16: Batch CSV Upload & Multi-Dataset Results

1. **Navigate to:** "Predict RUL" page
2. **Select:** "Upload CSV" input method
3. **Upload:** `test_batch.csv` (created earlier)
4. **Click:** "🚀 Run Batch Prediction"

**Expected Results Table Columns:**
- ✅ Row
- ✅ RUL (cycles)
- ✅ Health
- ✅ Distribution
- ✅ Life Stage
- ✅ Confidence
- ✅ **Dataset** (new Phase 3.5 column)
- ✅ **Cross-DS Conf** (new Phase 3.5 column)
- ✅ Warning

**Expected Summary:**
- ✅ Shows count of Healthy/Moderate/Critical
- ✅ Shows OOD count

---

### Test 17: Dataset Statistics Page

1. **Navigate to:** "Dataset Statistics" page (sidebar)

**Expected:**
- ✅ Shows NASA dataset metadata (samples, batteries analyzed)
- ✅ Shows feature statistics table (cycle, capacity, voltage, temperature, current)
- ✅ Shows OOD detection bounds (5th-95th percentiles)
- ✅ Shows dataset bias & limitations section
- ✅ Warning box about late-life bias

---

### Test 18: What-If Analysis with Multi-Dataset

1. **Navigate to:** "What-If Analysis" page
2. **Set Base Scenario:**
   - Temperature: 30°C
   - Voltage: 3.6V
   - Cycle: 80
   - Current: -1.0A
   - Capacity: 1.7 Ah
3. **Click:** "📊 Run Analysis"

**Expected:**
- ✅ Shows base RUL prediction
- ✅ Shows scenario comparisons chart
- ✅ Each scenario prediction uses backend API

---

### Test 19: Export Report with Multi-Dataset Fields

1. **After a prediction**, click "📊 Download CSV" or "📄 Download PDF"

**Expected CSV Fields:**
```
rul,time_estimate,cycle,temperature,voltage,current,capacity,model,battery_health,confidence,distribution_status,life_stage_context,confidence_explanation,inference_warning,dominant_dataset,cross_dataset_confidence,dataset_coverage_note
```

---

## 📊 VALIDATION CHECKLIST

### Backend Validation
| Test | Expected Outcome | Status |
|------|------------------|--------|
| Health endpoint returns healthy | ✅ | ⬜ |
| 4 statistics JSON files exist | ✅ | ⬜ |
| Statistics endpoint returns NASA data | ✅ | ⬜ |
| Early-life input → CALCE dominant | ✅ | ⬜ |
| Late-life input → NASA dominant | ✅ | ⬜ |
| Mid-life input → OXFORD dominant | ✅ | ⬜ |
| High-capacity long-cycle → MATR1 | ✅ | ⬜ |
| OOD detection works | ✅ | ⬜ |
| Batch prediction includes dataset fields | ✅ | ⬜ |
| API docs accessible | ✅ | ⬜ |

### Frontend Validation
| Test | Expected Outcome | Status |
|------|------------------|--------|
| Backend status indicator green | ✅ | ⬜ |
| Single prediction shows dominant dataset | ✅ | ⬜ |
| Cross-dataset confidence badge shown | ✅ | ⬜ |
| Dataset coverage note displayed | ✅ | ⬜ |
| Multi-dataset expander works | ✅ | ⬜ |
| Batch results include dataset columns | ✅ | ⬜ |
| Dataset statistics page works | ✅ | ⬜ |
| Export includes new fields | ✅ | ⬜ |

### Academic/Research Validation
| Criteria | Expected | Status |
|----------|----------|--------|
| Early-life inputs not always flagged as low-confidence | ✅ | ⬜ |
| Late-life predictions dominated by NASA | ✅ | ⬜ |
| Cross-dataset disagreement reduces confidence | ✅ | ⬜ |
| Dataset-aware explanations in responses | ✅ | ⬜ |
| Existing NASA functionality preserved | ✅ | ⬜ |
| No artificial RUL inflation | ✅ | ⬜ |

---

## 🐛 Troubleshooting Common Issues

### Issue 1: Backend Won't Start
```
Error: ModuleNotFoundError: No module named 'xxx'
```
**Solution:**
```powershell
pip install -r requirements.txt
```

### Issue 2: CORS Errors in Browser Console
**Solution:** Backend already has CORS configured for all origins. Check if backend is running.

### Issue 3: Statistics Files Not Found
**Solution:**
```powershell
# The multi_dataset_statistics.py auto-generates missing files on startup
# Force regeneration by deleting existing files:
Remove-Item .\data\*_statistics.json
# Then restart backend
```

### Issue 4: Frontend Can't Connect to Backend
**Solution:**
```powershell
# Ensure backend URL is correct
$env:BACKEND_URL = "http://localhost:8001"
# Restart Streamlit
```

### Issue 5: Models Not Loading
**Solution:**
```powershell
# Check models directory
Get-ChildItem .\models\

# Force retrain by setting environment variable
$env:FORCE_RETRAIN = "true"
# Restart backend
```

---

## 📁 Phase 3.5 Artifacts Checklist

| Artifact | Location | Status |
|----------|----------|--------|
| NASA statistics | `data/nasa_statistics.json` | ⬜ |
| CALCE statistics | `data/calce_statistics.json` | ⬜ |
| OXFORD statistics | `data/oxford_statistics.json` | ⬜ |
| MATR1 statistics | `data/matr1_statistics.json` | ⬜ |
| Multi-dataset manager | `backend/multi_dataset_statistics.py` | ⬜ |
| Enhanced predictor | `backend/predictor.py` | ⬜ |
| Updated schemas | `backend/schemas.py` | ⬜ |
| Enhanced server | `backend/server.py` | ⬜ |
| Updated frontend | `app.py` | ⬜ |

---

## 🎓 Summary

Phase 3.5 successfully integrates:
1. **4 Battery Datasets**: NASA, CALCE, OXFORD, MATR1
2. **Feature Harmonization**: Common feature set across all datasets
3. **Dataset-Aware Inference**: Input matching to closest dataset
4. **Cross-Dataset Confidence**: High/Medium/Low based on agreement
5. **Enhanced API Responses**: `dominant_dataset`, `cross_dataset_confidence`, `dataset_coverage_note`
6. **Batch Prediction Extension**: Per-row dataset analysis
7. **Frontend Updates**: Visual display of multi-dataset context

**Ready for Phase 4**: Testing, Validation, and Academic Deliverables
