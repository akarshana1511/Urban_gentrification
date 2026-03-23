"""
Quick Start Script
Run this script to update existing folder and app.py alone
This helps when you want to run the app without re-training
"""

import subprocess
import sys
from pathlib import Path

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Packages installed")
        return True
    except Exception as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_data_files():
    """Check if data files exist"""
    print("\n📁 Checking data files...")
    
    required_files = [
        "data/raw/Bengaluru_House_Data.csv",
        "data/raw/NammaMetro_Ridership_Dataset.csv",
        "data/raw/Bangalore restaurant chain.csv"
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_trained_models():
    """Check if models are trained"""
    print("\n🤖 Checking trained models...")
    
    model_files = [
        "models/rent_prediction_model.pkl",
        "models/gentrification_model.pkl",
        "models/displacement_risk_model.pkl",
        "models/scaler.pkl",
        "data/processed/final_features.csv"
    ]
    
    all_exist = True
    for file in model_files:
        path = Path(file)
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️  {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    print("\n" + "="*70)
    print("  🏘️ URBAN GENTRIFICATION PREDICTION SYSTEM - QUICK START")
    print("="*70)
    
    # Check Python
    if not check_python():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Continue anyway? (y/n)")
        if input().lower() != 'y':
            sys.exit(1)
    
    # Check data files
    data_ok = check_data_files()
    if not data_ok:
        print("\n❌ Missing data files! Please ensure all CSV files are in data/raw/")
        sys.exit(1)
    
    # Check models
    models_ok = check_trained_models()
    
    if not models_ok:
        print("\n⚠️  Models not found. Training required.")
        print("\nRun this to train models:")
        print("  → python train_pipeline.py")
        print("\nAfter training, run:")
        print("  → streamlit run app.py")
        sys.exit(1)
    
    # All good - run app
    print("\n" + "="*70)
    print("  ✅ ALL CHECKS PASSED - LAUNCHING APP")
    print("="*70)
    print("\n🚀 Starting Streamlit app...")
    print("📱 Open browser: http://localhost:8501")
    print("\nPress Ctrl+C to stop the app\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
