"""
Feature Engineering Module
Creates advanced features for machine learning models:
- Growth metrics (rent, price, crime, population)
- Density metrics (business, population)
- Transport accessibility
- Urban Growth Momentum Index (UGMI)
- Risk indicators
"""

import pandas as pd
import numpy as np
from src import utils
import config

class FeatureEngineer:
    """
    Create and engineer features for ML models
    """
    
    def __init__(self, df):
        """
        Initialize with processed dataset
        df: area-level aggregated dataframe
        """
        print("\n🔬 Initializing FeatureEngineer...")
        self.df = df.copy()
        self.features = None
    
    def create_growth_features(self):
        """
        Create growth rate features
        Growth = (current - historical) / historical
        Since we don't have time series, we'll simulate based on area characteristics
        """
        print("\n📈 Creating growth features...")
        
        # Use price as proxy for growth potential
        # Areas with lower current price but high rating = high growth potential
        self.df['simulated_price_growth'] = (
            (self.df['avg_rating'].fillna(3.5) / 5.0) * 0.3 -
            (self.df['price_per_sqft'] / self.df['price_per_sqft'].max()) * 0.3
        ).clip(-0.5, 0.5)
        
        # Rent growth potential (similar logic)
        self.df['simulated_rent_growth'] = self.df['simulated_price_growth'] * 0.8
        
        # Crime change (inverse of review count - more reviews = safer)
        self.df['crime_reduction_potential'] = (
            self.df['total_reviews'].fillna(0) / self.df['total_reviews'].max()
        ).fillna(0) * 0.3
        
        print("  ✓ Growth features created")
    
    def create_density_features(self):
        """
        Create density metrics
        - Business density: restaurants per unit area
        - Commercial activity: total reviews (high reviews = high activity)
        """
        print("\n🏙️  Creating density features...")
        
        # Business density
        self.df['business_density'] = (
            self.df['restaurant_count'] / self.df['restaurant_count'].max()
        ).fillna(0)
        
        # Commercial activity score
        self.df['commercial_activity'] = (
            (self.df['total_reviews'].fillna(0) / self.df['total_reviews'].max()) +
            (self.df['avg_rating'].fillna(3.5) / 5.0)
        ) / 2
        
        # Population growth proxy (more businesses = more people)
        self.df['population_growth_potential'] = self.df['business_density'] * 0.5
        
        print("  ✓ Density features created")
    
    def create_transport_accessibility(self):
        """
        Create transport accessibility features
        Since we have metro data, we'll use proximity to transport as indicator
        """
        print("\n🚇 Creating transport accessibility features...")
        
        # Simulate transport access score based on price diversity
        # (More diverse neighborhoods tend to have better connectivity)
        self.df['price_diversity'] = self.df['price_std'] / (self.df['price_mean'] + 1)
        
        # Transport accessibility score
        self.df['transport_accessibility'] = (
            1 / (1 + np.exp(-10 * (self.df['price_diversity'].fillna(0) - 0.5)))
        )  # Sigmoid function
        
        # Metro proximity proxy (use commercial activity as surrogate)
        self.df['metro_proximity_score'] = self.df['commercial_activity'] * 0.7
        
        print("  ✓ Transport features created")
    
    def create_risk_indicators(self):
        """
        Create risk and vulnerability indicators
        """
        print("\n⚠️  Creating risk indicators...")
        
        # Displacement risk: high price increase with low current affordability
        # (low price_per_sqft = currently affordable)
        self.df['displacement_risk_indicator'] = (
            self.df['simulated_rent_growth'] *
            (1 - self.df['price_per_sqft'] / self.df['price_per_sqft'].max())
        ).clip(0, 1)
        
        # Gentrification risk: growth potential + price increase
        self.df['gentrification_risk_base'] = (
            self.df['simulated_price_growth'] +
            self.df['business_density']
        ) / 2
        
        # Vulnerability score: combination of factors
        self.df['vulnerability'] = (
            (self.df['restaurant_count'] > 0).astype(int) *  # Has businesses
            (self.df['price_per_sqft'] / self.df['price_per_sqft'].max())  # Relative price
        )
        
        print("  ✓ Risk indicators created")
    
    def create_urban_growth_momentum_index(self):
        """
        Create the Urban Growth Momentum Index (UGMI)
        
        UGMI = 0.30(price_growth) + 0.20(business_density) + 
               0.20(transport_access) + 0.15(crime_reduction) + 0.15(population_growth)
        """
        print("\n🚀 Creating Urban Growth Momentum Index (UGMI)...")
        
        # Normalize components to 0-1 range
        components = {
            'price_growth': utils.normalize_column(self.df.copy(), 'simulated_price_growth')['simulated_price_growth'],
            'business_density': self.df['business_density'],
            'transport_access': self.df['transport_accessibility'],
            'crime_reduction': self.df['crime_reduction_potential'],
            'population_growth': self.df['population_growth_potential']
        }
        
        # Calculate UGMI
        self.df['UGMI'] = (
            config.UGMI_WEIGHTS['price_growth'] * (components['price_growth'] - components['price_growth'].min()) / (components['price_growth'].max() - components['price_growth'].min() + 1e-6) +
            config.UGMI_WEIGHTS['business_density'] * components['business_density'] +
            config.UGMI_WEIGHTS['transport_access'] * components['transport_access'] +
            config.UGMI_WEIGHTS['crime_reduction'] * components['crime_reduction'] +
            config.UGMI_WEIGHTS['population_growth'] * components['population_growth']
        )
        
        # Normalize UGMI to 0-1
        self.df['UGMI'] = (self.df['UGMI'] - self.df['UGMI'].min()) / (self.df['UGMI'].max() - self.df['UGMI'].min() + 1e-6)
        
        print(f"  ✓ UGMI created (range: {self.df['UGMI'].min():.3f} to {self.df['UGMI'].max():.3f})")
    
    def create_target_variables(self):
        """
        Create target variables for supervised learning
        """
        print("\n🎯 Creating target variables...")
        
        # Rent growth target: binary classification
        # High rent growth = gentrification
        self.df['is_gentrifying'] = (
            self.df['simulated_rent_growth'] > config.RENT_GROWTH_THRESHOLD
        ).astype(int)
        
        # Displacement risk target: binary classification
        # High displacement risk = rent is unaffordable
        self.df['has_displacement_risk'] = (
            self.df['displacement_risk_indicator'] > config.DISPLACEMENT_RENT_INCOME_RATIO
        ).astype(int)
        
        # Rent prediction target
        # Normalize median price as proxy for rent
        self.df['future_rent_target'] = (
            self.df['price_median'] *
            (1 + self.df['simulated_rent_growth'])
        )
        
        print(f"  ✓ Gentrifying areas: {self.df['is_gentrifying'].sum()}")
        print(f"  ✓ Displacement risk areas: {self.df['has_displacement_risk'].sum()}")
    
    def select_model_features(self):
        """
        Select features for machine learning models
        """
        print("\n✨ Selecting features for models...")
        
        # Define feature sets
        feature_columns = [
            # Original features
            'price_mean', 'price_median', 'price_per_sqft_mean',
            'price_per_sqft_median', 'total_sqft_mean', 'bath_mean', 'bhk_mean',
            
            # Created features
            'simulated_price_growth', 'simulated_rent_growth',
            'crime_reduction_potential', 'business_density',
            'commercial_activity', 'population_growth_potential',
            'price_diversity', 'transport_accessibility', 'metro_proximity_score',
            'displacement_risk_indicator', 'gentrification_risk_base',
            'vulnerability', 'UGMI',
            
            # Derived features
            'restaurant_count', 'avg_rating', 'total_reviews',
            'avg_business_quality'
        ]
        
        # Target columns
        target_columns = [
            'future_rent_target',
            'is_gentrifying',
            'has_displacement_risk'
        ]
        
        # Create feature matrix
        self.features = self.df[feature_columns + target_columns + ['area']].copy()
        
        # Handle any remaining NaN values
        numeric_columns = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_columns] = self.features[numeric_columns].fillna(0)
        
        print(f"  ✓ Selected {len(feature_columns)} features for modeling")
        print(f"  ✓ Total rows: {len(self.features)}")
        
        return self.features
    
    def run_full_pipeline(self):
        """
        Run complete feature engineering pipeline
        """
        utils.print_section_header("FEATURE ENGINEERING PIPELINE")
        
        self.create_growth_features()
        self.create_density_features()
        self.create_transport_accessibility()
        self.create_risk_indicators()
        self.create_urban_growth_momentum_index()
        self.create_target_variables()
        
        features = self.select_model_features()
        
        print("\n✅ Feature engineering complete!")
        print(f"   Features shape: {features.shape}")
        print(f"   Features created: {list(features.columns[:5])}...")
        
        return features

# ============== STANDALONE FUNCTIONS ==============

def engineer_features(df):
    """Quick function to engineer features"""
    engineer = FeatureEngineer(df)
    return engineer.run_full_pipeline()

if __name__ == "__main__":
    # Load processed data and engineer features
    df = utils.load_dataframe(config.MERGED_DATASET, "Merged dataset")
    features = engineer_features(df)
    utils.save_dataframe(features, config.FINAL_FEATURES, "Final features")
    print("\n" + "="*60)
    print("Feature engineering complete. Ready for model training.")
    print("="*60)
