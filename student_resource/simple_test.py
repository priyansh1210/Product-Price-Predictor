#!/usr/bin/env python3
"""
Simple Environment Test for ML Challenge
Tests core packages without problematic dependencies.
"""

def test_core_packages():
    """Test core ML packages."""
    print(" Testing Core ML Packages...")
    print("=" * 40)
    
    try:
        import pandas as pd
        print("✅ pandas - OK")
        
        import numpy as np
        print("✅ numpy - OK")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib - OK")
        
        import seaborn as sns
        print("✅ seaborn - OK")
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error
        print("✅ scikit-learn - OK")
        
        import joblib
        print("✅ joblib - OK")
        
        # Test basic functionality
        print("\n🧪 Testing Basic ML Workflow...")
        
        # Create sample data
        np.random.seed(42)
        X = np.random.rand(1000, 10)
        y = np.sum(X[:, :3], axis=1) + np.random.normal(0, 0.1, 1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"   - Model trained successfully")
        print(f"   - Test MAE: {mae:.4f}")
        
        # Test data loading
        print("\n📊 Testing Data Loading...")
        try:
            train_df = pd.read_csv('dataset/train.csv')
            test_df = pd.read_csv('dataset/test.csv')
            print(f"   - Training data: {train_df.shape}")
            print(f"   - Test data: {test_df.shape}")
            print(f"   - Columns: {list(train_df.columns)}")
        except FileNotFoundError:
            print("   - Dataset files not found (this is OK for setup)")
        
        print("\n🎉 Core ML environment is working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_optional_packages():
    """Test optional packages that might have issues."""
    print("\n🔍 Testing Optional Packages...")
    print("=" * 40)
    
    optional_packages = [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
    ]
    
    working_packages = []
    
    for name, import_name in optional_packages:
        try:
            __import__(import_name)
            print(f"✅ {name} - OK")
            working_packages.append(name)
        except Exception as e:
            print(f"⚠️  {name} - Issue: {str(e)[:50]}...")
    
    return working_packages

if __name__ == "__main__":
    print("🚀 ML Challenge Environment Test")
    print("=" * 50)
    
    core_ok = test_core_packages()
    working_optional = test_optional_packages()
    
    print("\n" + "=" * 50)
    if core_ok:
        print("✅ CORE ENVIRONMENT: Ready for ML Challenge!")
        print(f"✅ Working optional packages: {', '.join(working_optional)}")
        print("\n📝 You can now:")
        print("   1. Open Jupyter: jupyter lab")
        print("   2. Navigate to src/ folder") 
        print("   3. Start with ml_challenge_day1.ipynb")
        print("\n💡 Note: Some advanced packages may have compatibility issues")
        print("   but core ML functionality is working!")
    else:
        print("❌ CORE ENVIRONMENT: Issues detected")
        print("   Please check package installations")