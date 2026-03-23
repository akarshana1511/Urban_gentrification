"""
Configuration file for Urban Gentrification Prediction System
All settings in one place for easy modification
"""

import os
from pathlib import Path

# ============== PROJECT PATHS ==============
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============== DATA FILES ==============
# Raw datasets
HOUSE_DATA_FILE = RAW_DATA_DIR / "Bengaluru_House_Data.csv"
METRO_DATA_FILE = RAW_DATA_DIR / "NammaMetro_Ridership_Dataset.csv"
RESTAURANT_DATA_FILE = RAW_DATA_DIR / "Bangalore restaurant chain.csv"

# Processed datasets
PROCESSED_HOUSE_DATA = PROCESSED_DATA_DIR / "processed_house_data.csv"
PROCESSED_RESTAURANT_DATA = PROCESSED_DATA_DIR / "processed_restaurant_data.csv"
MERGED_DATASET = PROCESSED_DATA_DIR / "merged_dataset.csv"
FINAL_FEATURES = PROCESSED_DATA_DIR / "final_features.csv"

# ============== MODEL FILES ==============
RENT_PREDICTION_MODEL = MODELS_DIR / "rent_prediction_model.pkl"
GENTRIFICATION_MODEL = MODELS_DIR / "gentrification_model.pkl"
DISPLACEMENT_RISK_MODEL = MODELS_DIR / "displacement_risk_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"

# ============== FEATURE ENGINEERING PARAMS ==============
RENT_GROWTH_THRESHOLD = 0.20  # 20% rent increase = gentrification
CRIME_CHANGE_THRESHOLD = -0.10  # 10% crime reduction
DISPLACEMENT_RENT_INCOME_RATIO = 0.40  # Rent > 40% of income = displacement risk

# ============== MODEL HYPERPARAMETERS ==============
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42
}

# ============== FEATURE IMPORTANCE WEIGHTS ==============
# Urban Growth Momentum Index (UGMI) weights
UGMI_WEIGHTS = {
    'price_growth': 0.30,
    'business_density': 0.20,
    'transport_access': 0.20,
    'crime_reduction': 0.15,
    'population_growth': 0.15
}

# ============== SPATIAL PARAMETERS ==============
BENGALURU_CENTER_LAT = 12.9716
BENGALURU_CENTER_LON = 77.5946
MAX_DISTANCE_FROM_CENTER = 50  # km

# ============== THRESHOLDS ==============
GENTRIFICATION_PROBABILITY_THRESHOLD = 0.5
HIGH_DISPLACEMENT_RISK_THRESHOLD = 0.6
VERY_HIGH_DISPLACEMENT_RISK_THRESHOLD = 0.8

# ============== TRAINING PARAMS ==============
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# ============== UI COLORS ==============
COLOR_LOW_RISK = '#2ecc71'      # Green
COLOR_MEDIUM_RISK = '#f39c12'   # Orange
COLOR_HIGH_RISK = '#e74c3c'     # Red
COLOR_VERY_HIGH_RISK = '#c0392b' # Dark Red

print("✓ Configuration loaded successfully")
