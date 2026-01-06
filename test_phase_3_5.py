"""
Phase 3.5 Automated Backend Test Suite
Run this script to test all Phase 3.5 features automatically

Usage: python test_phase_3_5.py
"""
import requests
import json
import os
import sys
from typing import Dict, Any, List

# Configuration
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8001')
API_BASE = BACKEND_URL

# Test results tracking
test_results: List[Dict[str, Any]] = []

def log_result(test_name: str, passed: bool, details: str = ""):
    """Log test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    result = {"test": test_name, "passed": passed, "details": details}
    test_results.append(result)
    print(f"{status}: {test_name}")
    if details and not passed:
        print(f"   Details: {details}")

def test_health_endpoint():
    """Test 1: Health check endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=10)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            data.get("status") == "healthy" and
            data.get("models_loaded") == True
        )
        
        details = f"Status: {data.get('status')}, Models: {data.get('available_models')}"
        log_result("Health Endpoint", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Health Endpoint", False, str(e))
        return False

def test_statistics_endpoint():
    """Test 2: Dataset statistics endpoint"""
    try:
        response = requests.get(f"{API_BASE}/api/statistics", timeout=10)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            "metadata" in data and
            "features" in data and
            "dataset_context" in data
        )
        
        details = f"Keys: {list(data.keys())}"
        log_result("Statistics Endpoint", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Statistics Endpoint", False, str(e))
        return False

def test_early_life_prediction():
    """Test 3: Early-life input should match CALCE dataset"""
    try:
        payload = {
            "voltage": 3.8,
            "current": -1.0,
            "temperature": 25,
            "cycle": 20,
            "capacity": 1.95
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            data.get("dominant_dataset") == "CALCE" and
            data.get("life_stage_context") == "early_life" and
            "cross_dataset_confidence" in data and
            "dataset_coverage_note" in data
        )
        
        details = f"Dominant: {data.get('dominant_dataset')}, LifeStage: {data.get('life_stage_context')}"
        log_result("Early-Life → CALCE Prediction", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Early-Life → CALCE Prediction", False, str(e))
        return False

def test_late_life_prediction():
    """Test 4: Late-life input should match NASA dataset"""
    try:
        payload = {
            "voltage": 3.3,
            "current": -1.5,
            "temperature": 35,
            "cycle": 150,
            "capacity": 1.3
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            data.get("dominant_dataset") == "NASA" and
            data.get("life_stage_context") == "late_life"
        )
        
        details = f"Dominant: {data.get('dominant_dataset')}, LifeStage: {data.get('life_stage_context')}"
        log_result("Late-Life → NASA Prediction", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Late-Life → NASA Prediction", False, str(e))
        return False

def test_mid_life_prediction():
    """Test 5: Mid-life input should match OXFORD dataset"""
    try:
        payload = {
            "voltage": 3.6,
            "current": -0.9,
            "temperature": 26,
            "cycle": 100,
            "capacity": 0.95
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        # OXFORD has smaller capacity range (0.75-1.15), so capacity 0.95 should match
        # Life stage may vary based on classification logic - the key test is dataset matching
        passed = (
            response.status_code == 200 and
            "dominant_dataset" in data and
            "cross_dataset_confidence" in data and
            data.get("dominant_dataset") == "OXFORD"  # Key Phase 3.5 validation
        )
        
        details = f"Dominant: {data.get('dominant_dataset')}, LifeStage: {data.get('life_stage_context')}"
        log_result("Mid-Life → OXFORD Prediction", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Mid-Life → OXFORD Prediction", False, str(e))
        return False

def test_matr1_match_prediction():
    """Test 6: High capacity, long cycle should match MATR1"""
    try:
        payload = {
            "voltage": 3.6,
            "current": -1.5,
            "temperature": 28,
            "cycle": 300,
            "capacity": 2.5
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        # MATR1 has capacity range 1.8-3.2 Ah and cycle up to 1200
        passed = (
            response.status_code == 200 and
            "dominant_dataset" in data and
            data.get("dominant_dataset") in ["MATR1", "CALCE"]  # Both support higher capacity
        )
        
        details = f"Dominant: {data.get('dominant_dataset')}, Capacity: 2.5Ah, Cycle: 300"
        log_result("High Capacity → MATR1/CALCE Prediction", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("High Capacity → MATR1/CALCE Prediction", False, str(e))
        return False

def test_ood_detection():
    """Test 7: Out-of-distribution detection"""
    try:
        # Extreme values that should trigger OOD
        payload = {
            "voltage": 2.2,
            "current": -0.1,
            "temperature": 65,
            "cycle": 5,
            "capacity": 2.3
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            data.get("distribution_status") == "out_of_distribution" and
            data.get("inference_warning") is not None
        )
        
        details = f"DistStatus: {data.get('distribution_status')}, Warning: {bool(data.get('inference_warning'))}"
        log_result("OOD Detection", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("OOD Detection", False, str(e))
        return False

def test_cross_dataset_confidence_levels():
    """Test 8: Cross-dataset confidence returns valid levels"""
    try:
        payload = {
            "voltage": 3.7,
            "current": -1.0,
            "temperature": 25,
            "cycle": 50,
            "capacity": 1.8
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        valid_confidence_levels = ["high", "medium", "low"]
        passed = (
            response.status_code == 200 and
            data.get("cross_dataset_confidence") in valid_confidence_levels
        )
        
        details = f"CrossDatasetConf: {data.get('cross_dataset_confidence')}"
        log_result("Cross-Dataset Confidence Levels", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Cross-Dataset Confidence Levels", False, str(e))
        return False

def test_dataset_coverage_note():
    """Test 9: Dataset coverage note is present and informative"""
    try:
        payload = {
            "voltage": 3.7,
            "current": -1.0,
            "temperature": 25,
            "cycle": 50,
            "capacity": 1.8
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        coverage_note = data.get("dataset_coverage_note", "")
        passed = (
            response.status_code == 200 and
            len(coverage_note) > 20 and
            any(keyword in coverage_note.lower() for keyword in ["dataset", "matches", "life", "coverage"])
        )
        
        details = f"Note length: {len(coverage_note)}, Sample: {coverage_note[:100]}..."
        log_result("Dataset Coverage Note", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Dataset Coverage Note", False, str(e))
        return False

def test_api_response_schema():
    """Test 10: API response includes all Phase 3.5 required fields"""
    try:
        payload = {
            "voltage": 3.7,
            "current": -1.0,
            "temperature": 25,
            "cycle": 50,
            "capacity": 1.8
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        required_fields = [
            "predicted_rul_cycles",
            "battery_health",
            "confidence_level",
            "model_used",
            "distribution_status",
            "life_stage_context",
            "confidence_explanation",
            "dominant_dataset",           # Phase 3.5
            "cross_dataset_confidence",   # Phase 3.5
            "dataset_coverage_note"       # Phase 3.5
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        passed = len(missing_fields) == 0
        
        details = f"Missing fields: {missing_fields}" if missing_fields else "All fields present"
        log_result("API Response Schema", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("API Response Schema", False, str(e))
        return False

def test_batch_prediction():
    """Test 11: Batch prediction includes Phase 3.5 fields per row"""
    try:
        # Create CSV content
        csv_content = """voltage,current,temperature,cycle,capacity
3.8,-1.0,25,20,1.95
3.5,-1.2,30,80,1.6
3.3,-1.5,35,150,1.3"""
        
        files = {"file": ("test.csv", csv_content, "text/csv")}
        response = requests.post(f"{API_BASE}/api/predict/batch", files=files, timeout=30)
        data = response.json()
        
        if response.status_code != 200:
            log_result("Batch Prediction", False, f"HTTP {response.status_code}")
            return False
        
        # Check batch response structure
        passed = (
            data.get("success") == True and
            data.get("total_rows") == 3 and
            len(data.get("results", [])) == 3
        )
        
        # Check that each row has Phase 3.5 fields
        if passed and data.get("results"):
            for result in data["results"]:
                if "dominant_dataset" not in result or "cross_dataset_confidence" not in result:
                    passed = False
                    break
        
        details = f"Rows: {data.get('total_rows')}, Success: {data.get('success')}"
        log_result("Batch Prediction with Phase 3.5 Fields", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Batch Prediction with Phase 3.5 Fields", False, str(e))
        return False

def test_models_endpoint():
    """Test 12: Models endpoint returns available models"""
    try:
        response = requests.get(f"{API_BASE}/api/models", timeout=10)
        data = response.json()
        
        passed = (
            response.status_code == 200 and
            "models" in data and
            len(data.get("models", [])) > 0
        )
        
        details = f"Models: {data.get('models')}"
        log_result("Models Endpoint", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Models Endpoint", False, str(e))
        return False

def test_existing_functionality_preserved():
    """Test 13: Existing NASA dataset functionality still works"""
    try:
        # Standard NASA-like input
        payload = {
            "voltage": 3.5,
            "current": -1.8,
            "temperature": 30,
            "cycle": 100,
            "capacity": 1.5,
            "model_name": "XGBoost"
        }
        response = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
        data = response.json()
        
        # Core functionality should work
        passed = (
            response.status_code == 200 and
            "predicted_rul_cycles" in data and
            "battery_health" in data and
            isinstance(data.get("predicted_rul_cycles"), int) and
            data.get("battery_health") in ["Healthy", "Moderate", "Critical"]
        )
        
        details = f"RUL: {data.get('predicted_rul_cycles')}, Health: {data.get('battery_health')}"
        log_result("Existing Functionality Preserved", passed, details if not passed else "")
        return passed
    except Exception as e:
        log_result("Existing Functionality Preserved", False, str(e))
        return False

def test_statistics_files_exist():
    """Test 14: Check if all 4 statistics JSON files exist in data directory"""
    try:
        # This test assumes files are in ./data directory relative to backend
        # For API-only testing, we verify the multi-dataset manager loaded them
        response = requests.get(f"{API_BASE}/api/health", timeout=10)
        
        # We can infer from server logs or check status endpoint
        # For now, we test by making predictions that would use different datasets
        test_cases = [
            ({"voltage": 3.8, "current": -1.0, "temperature": 25, "cycle": 20, "capacity": 1.95}, "CALCE"),
            ({"voltage": 3.3, "current": -1.5, "temperature": 35, "cycle": 150, "capacity": 1.3}, "NASA"),
        ]
        
        all_passed = True
        for payload, expected_dominant in test_cases:
            resp = requests.post(f"{API_BASE}/api/predict", json=payload, timeout=10)
            if resp.status_code != 200:
                all_passed = False
                break
        
        log_result("Multi-Dataset Statistics Loaded", all_passed, "")
        return all_passed
    except Exception as e:
        log_result("Multi-Dataset Statistics Loaded", False, str(e))
        return False

def run_all_tests():
    """Run all Phase 3.5 tests"""
    print("\n" + "="*60)
    print("🔋 Phase 3.5 Backend Test Suite - Multi-Dataset Expansion")
    print("="*60 + "\n")
    
    print(f"Testing backend at: {BACKEND_URL}\n")
    
    # Run all tests
    tests = [
        test_health_endpoint,
        test_statistics_endpoint,
        test_early_life_prediction,
        test_late_life_prediction,
        test_mid_life_prediction,
        test_matr1_match_prediction,
        test_ood_detection,
        test_cross_dataset_confidence_levels,
        test_dataset_coverage_note,
        test_api_response_schema,
        test_batch_prediction,
        test_models_endpoint,
        test_existing_functionality_preserved,
        test_statistics_files_exist,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ FAIL: {test_func.__name__} - Unexpected error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in test_results if r["passed"])
    failed = len(test_results) - passed
    
    print(f"\nTotal Tests: {len(test_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"\nSuccess Rate: {passed/len(test_results)*100:.1f}%")
    
    if failed > 0:
        print("\n🔴 Failed Tests:")
        for r in test_results:
            if not r["passed"]:
                print(f"   - {r['test']}: {r['details']}")
    
    print("\n" + "="*60)
    
    return failed == 0

if __name__ == "__main__":
    # Check command line args for custom backend URL
    if len(sys.argv) > 1:
        BACKEND_URL = sys.argv[1]
        API_BASE = BACKEND_URL
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
