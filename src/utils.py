"""
Utility functions for data handling, model saving/loading, and common operations
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import config

def create_directories():
    """Create all required directories if they don't exist"""
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✓ All directories created/verified")

def load_raw_data():
    """
    Load all raw CSV files
    Returns: dict with dataframes
    """
    print("\n📂 Loading raw data files...")
    
    data = {}
    
    try:
        # Load housing data
        if config.HOUSE_DATA_FILE.exists():
            data['house'] = pd.read_csv(config.HOUSE_DATA_FILE)
            print(f"  ✓ Housing data: {len(data['house'])} records")
        
        # Load metro data
        if config.METRO_DATA_FILE.exists():
            data['metro'] = pd.read_csv(config.METRO_DATA_FILE)
            print(f"  ✓ Metro data: {len(data['metro'])} records")
        
        # Load restaurant data
        if config.RESTAURANT_DATA_FILE.exists():
            data['restaurant'] = pd.read_csv(config.RESTAURANT_DATA_FILE)
            print(f"  ✓ Restaurant data: {len(data['restaurant'])} records")
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    
    return data

def save_dataframe(df, filepath, description=""):
    """Save dataframe to CSV"""
    try:
        df.to_csv(filepath, index=False)
        print(f"  ✓ {description} saved: {filepath}")
    except Exception as e:
        print(f"❌ Error saving {description}: {e}")
        raise

def load_dataframe(filepath, description=""):
    """Load dataframe from CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"  ✓ {description} loaded: {len(df)} records")
        return df
    except Exception as e:
        print(f"❌ Error loading {description}: {e}")
        raise

def save_model(model, filepath, description=""):
    """Save trained model using joblib"""
    try:
        joblib.dump(model, filepath)
        print(f"  ✓ {description} saved")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise

def load_model(filepath, description=""):
    """Load trained model"""
    try:
        model = joblib.load(filepath)
        print(f"  ✓ {description} loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def print_dataframe_info(df, name="DataFrame"):
    """Print useful info about a dataframe"""
    print(f"\n📊 {name} Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Data types:\n{df.dtypes}\n")

def standardize_area_names(location_name):
    """
    Standardize area/location names
    Remove extra spaces, convert to lowercase, etc.
    """
    if pd.isna(location_name):
        return None
    
    # Convert to string and lowercase
    location = str(location_name).strip().lower()
    
    # Remove common suffixes
    for suffix in ['phase i', 'phase ii', 'phase iii', 'phase 1', 'phase 2', 'phase 3',
                   'layout', 'main road', 'road', 'street', 'ext', 'extension']:
        location = location.replace(suffix, '').strip()
    
    return location

def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in dataframe
    strategy: 'mean', 'median', 'drop', or 'forward_fill'
    """
    print(f"\n🧹 Handling missing values (strategy: {strategy})...")
    
    missing_before = df.isnull().sum().sum()
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill')
    
    missing_after = df.isnull().sum().sum()
    print(f"  Missing values: {missing_before} → {missing_after}")
    
    return df

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers using Z-score method
    threshold: number of standard deviations (default 3)
    """
    print(f"  Removing outliers in '{column}' (Z-score > {threshold})...")
    
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    rows_before = len(df)
    df = df[(np.abs(stats.zscore(df[column].dropna())) < threshold)]
    rows_after = len(df)
    
    print(f"    Rows removed: {rows_before - rows_after}")
    
    return df

def normalize_column(df, column):
    """Normalize a column to 0-1 range"""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates in kilometers
    Using Haversine formula
    """
    from math import radians, cos, sin, asin, sqrt
    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Kilometer
    km = 6371 * c
    return km

def print_success_message(message):
    """Print success message with emoji"""
    print(f"\n✅ {message}")

def print_error_message(message):
    """Print error message with emoji"""
    print(f"\n❌ {message}")

def print_section_header(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
