#!/usr/bin/env python3
"""
Health check script for Battery RUL Prediction Application
"""

import sys
import subprocess
import importlib
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version OK")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_dependencies():
    """Check if all required packages are installed."""
    packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'tensorflow',
        'shap',
        'plotly',
        'matplotlib',
        'seaborn',
        'reportlab'
    ]
    
    print("\n📦 Checking Dependencies:")
    all_ok = True
    
    for package in packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {package}: {version}")
        except ImportError:
            print(f"   ❌ {package}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_files():
    """Check if all required files exist."""
    files = [
        '/app/app.py',
        '/app/requirements.txt',
        '/app/utils/auth.py',
        '/app/utils/data_processor.py',
        '/app/utils/ml_models.py',
        '/app/utils/explainer.py',
        '/app/utils/visualizer.py',
        '/app/utils/report_generator.py',
        '/app/.streamlit/config.toml'
    ]
    
    print("\n📁 Checking Files:")
    all_ok = True
    
    for file in files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}: MISSING")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check if required directories exist."""
    dirs = [
        '/app/utils',
        '/app/models',
        '/app/data',
        '/app/reports',
        '/app/.streamlit'
    ]
    
    print("\n📂 Checking Directories:")
    for directory in dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ⚠️  {directory}: Creating...")
            os.makedirs(directory, exist_ok=True)
    
    return True

def check_streamlit_service():
    """Check if Streamlit service is running."""
    print("\n🚀 Checking Streamlit Service:")
    try:
        result = subprocess.run(
            ['supervisorctl', 'status', 'streamlit'],
            capture_output=True,
            text=True
        )
        
        if 'RUNNING' in result.stdout:
            print("   ✅ Streamlit service is RUNNING")
            return True
        else:
            print(f"   ❌ Streamlit service status: {result.stdout.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Error checking service: {e}")
        return False

def check_data():
    """Check if data files exist."""
    print("\n📊 Checking Data:")
    
    if os.path.exists('/app/data/nasa_battery_data.csv'):
        print("   ✅ NASA dataset cached")
    else:
        print("   ℹ️  NASA dataset not cached (will download on first run)")
    
    if os.path.exists('/app/data/sample_battery_data.csv'):
        print("   ✅ Sample dataset available")
    else:
        print("   ℹ️  Sample dataset will be generated on first run")
    
    return True

def check_models():
    """Check if pre-trained models exist."""
    print("\n🤖 Checking Models:")
    
    models = [
        'xgboost.pkl',
        'random_forest.pkl',
        'linear_regression.pkl',
        'lstm.keras'
    ]
    
    found = False
    for model in models:
        path = f'/app/models/{model}'
        if os.path.exists(path):
            print(f"   ✅ {model}")
            found = True
        else:
            print(f"   ℹ️  {model}: Will be trained on first run")
    
    if not found:
        print("   ℹ️  No pre-trained models (training will occur on first use)")
    
    return True

def main():
    """Run all health checks."""
    print("=" * 60)
    print("🔋 Battery RUL Prediction - Health Check")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_files(),
        check_directories(),
        check_streamlit_service(),
        check_data(),
        check_models()
    ]
    
    print("\n" + "=" * 60)
    
    if all(checks):
        print("✅ All health checks passed!")
        print("🎉 Application is ready to use!")
        print("\n📝 Access the app at: http://localhost:8501")
        print("📖 See QUICKSTART.md for usage guide")
        return 0
    else:
        print("⚠️  Some health checks failed")
        print("Please review the errors above and fix them")
        return 1

if __name__ == "__main__":
    sys.exit(main())
