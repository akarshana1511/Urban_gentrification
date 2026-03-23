"""
Data Preprocessing Module
- Load raw data
- Clean and standardize
- Handle missing values
- Remove outliers
- Save processed data
"""

import pandas as pd
import numpy as np
from src import utils
import config

class DataPreprocessor:
    """
    Handles all data cleaning and preprocessing tasks
    """
    
    def __init__(self):
        print("\n🔧 Initializing DataPreprocessor...")
        self.raw_data = {}
        self.processed_data = {}
    
    def load_all_data(self):
        """Load all raw datasets"""
        print("\n📂 Loading raw datasets...")
        self.raw_data = utils.load_raw_data()
        return self.raw_data
    
    def preprocess_house_data(self):
        """
        Clean and preprocess house price data
        Key tasks:
        - Remove missing values
        - Remove outliers in price
        - Standardize location names
        - Create area features
        """
        print("\n🏠 Preprocessing house data...")
        
        df = self.raw_data['house'].copy()
        print(f"  Original shape: {df.shape}")
        
        # Remove rows with missing critical columns
        df = df.dropna(subset=['location', 'price', 'total_sqft'])
        print(f"  After removing missing values: {df.shape}")
        
        # Standardize location names
        df['location'] = df['location'].apply(utils.standardize_area_names)
        
        # Handle price outliers (remove prices > 500 crores or < 10 lakhs)
        df = df[(df['price'] >= 10) & (df['price'] <= 500)]
        print(f"  After removing price outliers: {df.shape}")
        
        # Convert price to numeric if needed
        if df['price'].dtype == 'object':
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Handle bath and balcony - fill missing with median
        df['bath'] = df['bath'].fillna(df['bath'].median())
        df['balcony'] = df['balcony'].fillna(df['balcony'].median())
        
        # Create new features
        # Price per square feet
        df['price_per_sqft'] = (df['price'] * 10000000) / df['total_sqft']  # price is in crores
        
        # Extract number of bedrooms from size
        df['bhk'] = df['size'].str.extract('(\d+)', expand=False).fillna(0).astype(int)
        
        # Calculate price per bhk
        df['price_per_bhk'] = df['price'] / (df['bhk'] + 1)  # +1 to avoid division by zero
        
        self.processed_data['house'] = df
        print(f"  ✓ House data processed: {df.shape}")
        return df
    
    def preprocess_restaurant_data(self):
        """
        Process restaurant data
        Key tasks:
        - Clean location data
        - Create business density metric
        - Extract geographic features
        """
        print("\n🍽️  Preprocessing restaurant data...")
        
        df = self.raw_data['restaurant'].copy()
        print(f"  Original shape: {df.shape}")
        
        # Remove missing locations
        df = df.dropna(subset=['Latitude', 'Longitude', 'Address'])
        print(f"  After removing missing locations: {df.shape}")
        
        # Standard location names
        df['Location'] = df['Address'].apply(utils.standardize_area_names)
        
        # Remove duplicate restaurants in same location
        df = df.drop_duplicates(subset=['Latitude', 'Longitude'], keep='first')
        print(f"  After removing duplicates: {df.shape}")
        
        # Rating might have missing values
        df['Rating'] = df['Rating'].fillna(df['Rating'].median())
        
        # Calculate business quality score (Rating * Review_count normalized)
        df['business_quality'] = df['Rating'] * (1 + np.log1p(df['Review_count']))
        
        self.processed_data['restaurant'] = df
        print(f"  ✓ Restaurant data processed: {df.shape}")
        return df
    
    def create_area_aggregates(self):
        """
        Aggregate house data by area
        This creates one row per area with aggregated metrics
        """
        print("\n🗺️  Creating area-level aggregates...")
        
        house_df = self.processed_data['house'].copy()
        restaurant_df = self.processed_data['restaurant'].copy()
        
        # Aggregate house data by location
        house_agg = house_df.groupby('location').agg({
            'price': ['mean', 'std', 'median', 'count'],
            'price_per_sqft': ['mean', 'median'],
            'total_sqft': 'mean',
            'bath': 'mean',
            'balcony': 'mean',
            'bhk': 'mean',
            'price_per_bhk': 'mean'
        }).reset_index()
        
        # Flatten column names
        house_agg.columns = ['_'.join(col).strip('_') for col in house_agg.columns.values]
        house_agg.rename(columns={'location': 'area'}, inplace=True)
        
        # Count restaurants per area (business density)
        restaurant_count = restaurant_df.groupby('Location').agg({
            'Name': 'count',
            'Rating': 'mean',
            'Review_count': 'sum',
            'business_quality': 'mean'
        }).reset_index()
        
        restaurant_count.columns = ['area', 'restaurant_count', 'avg_rating', 
                                    'total_reviews', 'avg_business_quality']
        restaurant_count['restaurant_count'] = restaurant_count['restaurant_count'].fillna(0)
        
        # Merge
        area_data = house_agg.merge(restaurant_count, on='area', how='left')
        area_data['restaurant_count'] = area_data['restaurant_count'].fillna(0)
        area_data['avg_rating'] = area_data['avg_rating'].fillna(area_data['avg_rating'].median())
        
        print(f"  ✓ Created aggregates for {len(area_data)} areas")
        return area_data
    
    def save_processed_data(self):
        """Save all processed dataframes"""
        print("\n💾 Saving processed data...")
        
        # Save individual processed files
        utils.save_dataframe(
            self.processed_data['house'],
            config.PROCESSED_HOUSE_DATA,
            "Processed house data"
        )
        
        utils.save_dataframe(
            self.processed_data['restaurant'],
            config.PROCESSED_RESTAURANT_DATA,
            "Processed restaurant data"
        )
        
        # Create and save area aggregates
        area_data = self.create_area_aggregates()
        utils.save_dataframe(
            area_data,
            config.MERGED_DATASET,
            "Merged area-level dataset"
        )
        
        print("✓ All processed data saved")
        return area_data
    
    def run_full_pipeline(self):
        """
        Run complete preprocessing pipeline
        """
        utils.print_section_header("DATA PREPROCESSING PIPELINE")
        
        self.load_all_data()
        self.preprocess_house_data()
        self.preprocess_restaurant_data()
        
        final_dataset = self.save_processed_data()
        
        print("\n✅ Data preprocessing complete!")
        print(f"   Final dataset shape: {final_dataset.shape}")
        print(f"   Final dataset columns: {len(final_dataset.columns)}")
        
        return final_dataset

# ============== STANDALONE FUNCTIONS FOR QUICK USE ==============

def preprocess_all():
    """Quick function to run full preprocessing"""
    preprocessor = DataPreprocessor()
    return preprocessor.run_full_pipeline()

if __name__ == "__main__":
    # This allows running the module directly
    # python -m src.data_preprocessing
    
    preprocessor = DataPreprocessor()
    dataset = preprocessor.run_full_pipeline()
    print("\n" + "="*60)
    print("Preprocessing complete. Ready for feature engineering.")
    print("="*60)
