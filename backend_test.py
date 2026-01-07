#!/usr/bin/env python3
"""
Backend API Testing for Battery RUL Prediction System
Tests all endpoints and validates Phase 4 functionality
"""
import requests
import json
import sys
from datetime import datetime
from typing import Dict, Any

class BatteryRULAPITester:
    def __init__(self, base_url="https://demobackend.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        self.passed_tests = []
        
    def log_test(self, name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            self.passed_tests.append(name)
            print(f"✅ {name} - PASSED")
        else:
            self.failed_tests.append({"test": name, "details": details})
            print(f"❌ {name} - FAILED: {details}")
        
        if details:
            print(f"   Details: {details}")
        print()

    def test_health_endpoint(self):
        """Test /api/health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'service', 'models_loaded', 'available_models']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Health Endpoint", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check if models are loaded
                if not data.get('models_loaded', False):
                    self.log_test("Health Endpoint", False, "Models not loaded")
                    return False
                
                # Check if available models list is not empty
                if not data.get('available_models', []):
                    self.log_test("Health Endpoint", False, "No available models")
                    return False
                
                self.log_test("Health Endpoint", True, f"Status: {data['status']}, Models: {data['available_models']}")
                return True
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_models_endpoint(self):
        """Test /api/models endpoint - should return both v1_nasa and v2_physics_augmented"""
        try:
            response = requests.get(f"{self.base_url}/api/models", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for required fields
                required_fields = ['models', 'versions', 'default_model', 'default_version']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Models Endpoint", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check for both model versions
                versions = data.get('versions', [])
                expected_versions = ['v1_nasa', 'v2_physics_augmented']
                
                missing_versions = [v for v in expected_versions if v not in versions]
                if missing_versions:
                    self.log_test("Models Endpoint", False, f"Missing versions: {missing_versions}, Got: {versions}")
                    return False
                
                # Check default version is v2_physics_augmented
                if data.get('default_version') != 'v2_physics_augmented':
                    self.log_test("Models Endpoint", False, f"Default version should be v2_physics_augmented, got: {data.get('default_version')}")
                    return False
                
                self.log_test("Models Endpoint", True, f"Versions: {versions}, Default: {data['default_version']}")
                return True
            else:
                self.log_test("Models Endpoint", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Models Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_prediction_v2_default(self):
        """Test prediction with v2_physics_augmented model (default)"""
        try:
            # Test case: mid-life battery
            payload = {
                "voltage": 3.6,
                "current": -1.0,
                "temperature": 25.0,
                "cycle": 75,
                "capacity": 1.7,
                "model_name": "XGBoost"
                # model_version defaults to v2_physics_augmented
            }
            
            response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['predicted_rul_cycles', 'battery_health', 'confidence_level', 
                                 'model_used', 'model_version', 'recommendation']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Prediction V2 Default", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check model version is v2_physics_augmented
                if data.get('model_version') != 'v2_physics_augmented':
                    self.log_test("Prediction V2 Default", False, f"Expected v2_physics_augmented, got: {data.get('model_version')}")
                    return False
                
                # Check RUL is reasonable
                rul = data.get('predicted_rul_cycles', 0)
                if rul < 0 or rul > 500:
                    self.log_test("Prediction V2 Default", False, f"Unreasonable RUL: {rul}")
                    return False
                
                self.log_test("Prediction V2 Default", True, 
                            f"RUL: {rul}, Health: {data['battery_health']}, Model: {data['model_version']}")
                return True
            else:
                self.log_test("Prediction V2 Default", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Prediction V2 Default", False, f"Exception: {str(e)}")
            return False

    def test_prediction_v1_baseline(self):
        """Test prediction with v1_nasa baseline model"""
        try:
            payload = {
                "voltage": 3.6,
                "current": -1.0,
                "temperature": 25.0,
                "cycle": 75,
                "capacity": 1.7,
                "model_name": "XGBoost",
                "model_version": "v1_nasa"
            }
            
            response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check model version is v1_nasa
                if data.get('model_version') != 'v1_nasa':
                    self.log_test("Prediction V1 Baseline", False, f"Expected v1_nasa, got: {data.get('model_version')}")
                    return False
                
                # Check RUL is reasonable
                rul = data.get('predicted_rul_cycles', 0)
                if rul < 0 or rul > 500:
                    self.log_test("Prediction V1 Baseline", False, f"Unreasonable RUL: {rul}")
                    return False
                
                self.log_test("Prediction V1 Baseline", True, 
                            f"RUL: {rul}, Health: {data['battery_health']}, Model: {data['model_version']}")
                return True
            else:
                self.log_test("Prediction V1 Baseline", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Prediction V1 Baseline", False, f"Exception: {str(e)}")
            return False

    def test_baseline_comparison(self):
        """Test baseline comparison feature"""
        try:
            payload = {
                "voltage": 3.6,
                "current": -1.0,
                "temperature": 25.0,
                "cycle": 75,
                "capacity": 1.7,
                "model_name": "XGBoost",
                "model_version": "v2_physics_augmented",
                "compare_baseline": True
            }
            
            response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check baseline comparison is present
                if 'baseline_comparison' not in data or data['baseline_comparison'] is None:
                    self.log_test("Baseline Comparison", False, "baseline_comparison field missing or null")
                    return False
                
                baseline = data['baseline_comparison']
                required_baseline_fields = ['v1_predicted_rul', 'v1_health', 'rul_difference', 'comparison_note']
                missing_fields = [field for field in required_baseline_fields if field not in baseline]
                
                if missing_fields:
                    self.log_test("Baseline Comparison", False, f"Missing baseline fields: {missing_fields}")
                    return False
                
                self.log_test("Baseline Comparison", True, 
                            f"V2 RUL: {data['predicted_rul_cycles']}, V1 RUL: {baseline['v1_predicted_rul']}, Diff: {baseline['rul_difference']}")
                return True
            else:
                self.log_test("Baseline Comparison", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Baseline Comparison", False, f"Exception: {str(e)}")
            return False

    def test_health_classification_thresholds(self):
        """Test health classification thresholds: Healthy >100, Moderate 30-100, Critical <30"""
        test_cases = [
            # (expected_health, cycle, capacity, description)
            ("Healthy", 10, 1.95, "Early life - should be Healthy (RUL > 100)"),
            ("Moderate", 100, 1.6, "Mid life - should be Moderate (30-100 RUL)"),
            ("Critical", 150, 1.3, "Late life - should be Critical (< 30 RUL)")
        ]
        
        all_passed = True
        
        for expected_health, cycle, capacity, description in test_cases:
            try:
                payload = {
                    "voltage": 3.5,
                    "current": -1.0,
                    "temperature": 25.0,
                    "cycle": cycle,
                    "capacity": capacity,
                    "model_name": "XGBoost"
                }
                
                response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    actual_health = data.get('battery_health')
                    rul = data.get('predicted_rul_cycles', 0)
                    
                    if actual_health == expected_health:
                        print(f"   ✅ {description} - Got {actual_health} (RUL: {rul})")
                    else:
                        print(f"   ❌ {description} - Expected {expected_health}, got {actual_health} (RUL: {rul})")
                        all_passed = False
                else:
                    print(f"   ❌ {description} - HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {description} - Exception: {str(e)}")
                all_passed = False
        
        self.log_test("Health Classification Thresholds", all_passed, 
                     "Tested Healthy >100, Moderate 30-100, Critical <30")
        return all_passed

    def test_recommendation_thresholds(self):
        """Test recommendation thresholds: >150 early healthy, 50-150 mid-life, <50 EOL"""
        test_cases = [
            # (cycle, capacity, expected_keyword, description)
            (5, 1.98, "early healthy", "Very early life - should get early healthy recommendation"),
            (80, 1.7, "mid-life", "Mid life - should get mid-life recommendation"),
            (140, 1.4, "end-of-life", "Late life - should get EOL recommendation")
        ]
        
        all_passed = True
        
        for cycle, capacity, expected_keyword, description in test_cases:
            try:
                payload = {
                    "voltage": 3.5,
                    "current": -1.0,
                    "temperature": 25.0,
                    "cycle": cycle,
                    "capacity": capacity,
                    "model_name": "XGBoost"
                }
                
                response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    recommendation = data.get('recommendation', '').lower()
                    rul = data.get('predicted_rul_cycles', 0)
                    
                    if expected_keyword in recommendation:
                        print(f"   ✅ {description} - Got '{data.get('recommendation')}' (RUL: {rul})")
                    else:
                        print(f"   ❌ {description} - Expected '{expected_keyword}' in recommendation, got '{data.get('recommendation')}' (RUL: {rul})")
                        all_passed = False
                else:
                    print(f"   ❌ {description} - HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {description} - Exception: {str(e)}")
                all_passed = False
        
        self.log_test("Recommendation Thresholds", all_passed, 
                     "Tested >150 early healthy, 50-150 mid-life, <50 EOL")
        return all_passed

    def test_distribution_status(self):
        """Test distribution status (in_distribution vs out_of_distribution)"""
        test_cases = [
            # (voltage, current, temp, cycle, capacity, description)
            (3.7, -1.0, 25, 50, 1.8, "Normal parameters - should be in_distribution"),
            (5.0, -10.0, 100, 500, 0.5, "Extreme parameters - should be out_of_distribution")
        ]
        
        all_passed = True
        
        for voltage, current, temp, cycle, capacity, description in test_cases:
            try:
                payload = {
                    "voltage": voltage,
                    "current": current,
                    "temperature": temp,
                    "cycle": cycle,
                    "capacity": capacity,
                    "model_name": "XGBoost"
                }
                
                response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    dist_status = data.get('distribution_status')
                    
                    if dist_status in ['in_distribution', 'out_of_distribution']:
                        print(f"   ✅ {description} - Got {dist_status}")
                    else:
                        print(f"   ❌ {description} - Invalid distribution status: {dist_status}")
                        all_passed = False
                else:
                    print(f"   ❌ {description} - HTTP {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {description} - Exception: {str(e)}")
                all_passed = False
        
        self.log_test("Distribution Status", all_passed, "Tested in_distribution vs out_of_distribution")
        return all_passed

    def test_cross_dataset_analysis(self):
        """Test cross-dataset analysis (dominant_dataset and cross_dataset_confidence)"""
        try:
            payload = {
                "voltage": 3.6,
                "current": -1.0,
                "temperature": 25.0,
                "cycle": 75,
                "capacity": 1.7,
                "model_name": "XGBoost"
            }
            
            response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for cross-dataset fields
                required_fields = ['dominant_dataset', 'cross_dataset_confidence']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Cross-Dataset Analysis", False, f"Missing fields: {missing_fields}")
                    return False
                
                dominant_dataset = data.get('dominant_dataset')
                cross_confidence = data.get('cross_dataset_confidence')
                
                # Check valid values
                valid_datasets = ['NASA', 'CALCE', 'OXFORD', 'MATR1']
                valid_confidences = ['high', 'medium', 'low']
                
                if dominant_dataset not in valid_datasets:
                    self.log_test("Cross-Dataset Analysis", False, f"Invalid dominant_dataset: {dominant_dataset}")
                    return False
                
                if cross_confidence not in valid_confidences:
                    self.log_test("Cross-Dataset Analysis", False, f"Invalid cross_dataset_confidence: {cross_confidence}")
                    return False
                
                self.log_test("Cross-Dataset Analysis", True, 
                            f"Dominant: {dominant_dataset}, Confidence: {cross_confidence}")
                return True
            else:
                self.log_test("Cross-Dataset Analysis", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Cross-Dataset Analysis", False, f"Exception: {str(e)}")
            return False

    def test_statistics_endpoint(self):
        """Test /api/statistics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/statistics", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['metadata', 'features', 'dataset_context']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Statistics Endpoint", False, f"Missing fields: {missing_fields}")
                    return False
                
                # Check if features dict is not empty
                if not data.get('features', {}):
                    self.log_test("Statistics Endpoint", False, "Features dict is empty")
                    return False
                
                self.log_test("Statistics Endpoint", True, 
                            f"Metadata keys: {list(data['metadata'].keys())}, Feature count: {len(data['features'])}")
                return True
            else:
                self.log_test("Statistics Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Statistics Endpoint", False, f"Exception: {str(e)}")
            return False

    def test_input_validation(self):
        """Test input validation for edge cases"""
        test_cases = [
            # (payload, expected_status, description)
            ({"voltage": 1.5, "current": -1.0, "temperature": 25, "cycle": 50, "capacity": 1.8}, 422, "Voltage too low"),
            ({"voltage": 5.0, "current": -1.0, "temperature": 25, "cycle": 50, "capacity": 1.8}, 422, "Voltage too high"),
            ({"voltage": 3.7, "current": -1.0, "temperature": -5, "cycle": 50, "capacity": 1.8}, 422, "Temperature too low"),
            ({"voltage": 3.7, "current": -1.0, "temperature": 80, "cycle": 50, "capacity": 1.8}, 422, "Temperature too high"),
            ({"voltage": 3.7, "current": -1.0, "temperature": 25, "cycle": -1, "capacity": 1.8}, 422, "Negative cycle"),
            ({"voltage": 3.7, "current": -1.0, "temperature": 25, "cycle": 50, "capacity": 0.005}, 422, "Capacity too low"),
        ]
        
        all_passed = True
        
        for payload, expected_status, description in test_cases:
            try:
                response = requests.post(f"{self.base_url}/api/predict", json=payload, timeout=10)
                
                if response.status_code == expected_status:
                    print(f"   ✅ {description} - Correctly rejected with {response.status_code}")
                else:
                    print(f"   ❌ {description} - Expected {expected_status}, got {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {description} - Exception: {str(e)}")
                all_passed = False
        
        self.log_test("Input Validation", all_passed, "Tested edge cases and invalid inputs")
        return all_passed

    def run_all_tests(self):
        """Run all tests and return summary"""
        print("="*60)
        print("Battery RUL Prediction API Testing")
        print("="*60)
        print(f"Testing endpoint: {self.base_url}")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all tests
        self.test_health_endpoint()
        self.test_models_endpoint()
        self.test_prediction_v2_default()
        self.test_prediction_v1_baseline()
        self.test_baseline_comparison()
        self.test_health_classification_thresholds()
        self.test_recommendation_thresholds()
        self.test_distribution_status()
        self.test_cross_dataset_analysis()
        self.test_statistics_endpoint()
        self.test_input_validation()
        
        # Print summary
        print("="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.failed_tests:
            print("\nFAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test['test']}: {test['details']}")
        
        print("\nPASSED TESTS:")
        for test in self.passed_tests:
            print(f"  - {test}")
        
        return {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": len(self.failed_tests),
            "success_rate": (self.tests_passed/self.tests_run)*100,
            "failed_test_details": self.failed_tests,
            "passed_test_list": self.passed_tests
        }

def main():
    """Main test execution"""
    tester = BatteryRULAPITester()
    results = tester.run_all_tests()
    
    # Exit with error code if tests failed
    if results["failed_tests"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()