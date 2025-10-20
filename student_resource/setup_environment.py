#!/usr/bin/env python3
"""
Environment Setup and Verification Script for ML Challenge
This script verifies that all required packages are installed and working correctly.
"""

import sys
import importlib
import warnings
warnings.filterwarnings('ignore')

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - FAILED: {e}")
        return False

def main():
    """Main function to check all required packages."""
    print("🔍 Checking ML Challenge Environment Setup...")
    print("=" * 50)
    
    # Core packages
    packages_to_check = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("joblib", "joblib"),
        ("tqdm", "tqdm"),
        
        # ML packages
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        
        # Deep Learning
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("sentence-transformers", "sentence_transformers"),
        
        # Image processing
        ("Pillow", "PIL"),
        ("opencv-python", "cv2"),
        
        # Text processing
        ("nltk", "nltk"),
        
        # Jupyter
        ("jupyter", "jupyter"),
        ("notebook", "notebook"),
        ("jupyterlab", "jupyterlab"),
    ]
    
    failed_packages = []
    
    for package_name, import_name in packages_to_check:
        if not check_package(package_name, import_name):
            failed_packages.append(package_name)
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print(f"❌ {len(failed_packages)} packages failed to import:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\n💡 Try running: pip install -r requirements.txt")
        return False
    else:
        print("🎉 All packages are installed and working correctly!")
        
        # Test basic functionality
        print("\n🧪 Testing basic functionality...")
        try:
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor
            
            # Create test data
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            
            # Test model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X[:5])
            
            print("✅ Basic ML functionality test - PASSED")
            
        except Exception as e:
            print(f"❌ Basic ML functionality test - FAILED: {e}")
            return False
        
        print("\n🚀 Environment is ready for the ML Challenge!")
        print("\n📝 Next steps:")
        print("   1. Open Jupyter: jupyter lab")
        print("   2. Navigate to src/ folder")
        print("   3. Start with ml_challenge_day1.ipynb")
        
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)