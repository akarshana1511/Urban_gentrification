"""
Predictions Module
Make predictions using trained models
- Rent prediction
- Gentrification probability
- Displacement risk score
- Urban Growth Momentum Index mapping to risk levels
"""

import pandas as pd
import numpy as np
from src import utils
import config

class PredictionEngine:
    """
    Uses trained models to make predictions on new data
    """
    
    def __init__(self):
        """Load trained models"""
        print("\n🔮 Initializing PredictionEngine...")
        
        try:
            self.rent_model = utils.load_model(
                config.RENT_PREDICTION_MODEL,
                "Rent prediction model"
            )
            self.gentrification_model = utils.load_model(
                config.GENTRIFICATION_MODEL,
                "Gentrification model"
            )
            self.displacement_model = utils.load_model(
                config.DISPLACEMENT_RISK_MODEL,
                "Displacement risk model"
            )
            self.scaler = utils.load_model(
                config.SCALER_FILE,
                "Feature scaler"
            )
        except Exception as e:
            print(f"❌ Error loading models. Make sure training has been completed.")
            raise
    
    def predict_rent(self, X):
        """Predict future rent for given features"""
        return self.rent_model.predict(X)
    
    def predict_gentrification(self, X):
        """Predict gentrification probability"""
        return self.gentrification_model.predict_proba(X)[:, 1]
    
    def predict_displacement_risk(self, X):
        """Predict displacement risk probability"""
        return self.displacement_model.predict_proba(X)[:, 1]
    
    def get_risk_level(self, probability):
        """
        Convert probability to risk level
        """
        if probability < 0.25:
            return "Low Risk", config.COLOR_LOW_RISK
        elif probability < 0.50:
            return "Medium Risk", config.COLOR_MEDIUM_RISK
        elif probability < 0.80:
            return "High Risk", config.COLOR_HIGH_RISK
        else:
            return "Very High Risk", config.COLOR_VERY_HIGH_RISK
    
    def predict_for_areas(self, features_df):
        """
        Make all predictions for given areas
        features_df: dataframe with area predictions or new areas
        """
        print("\n🎯 Making predictions...")
        
        # Get feature columns (same as training)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['area', 'future_rent_target', 
                                       'is_gentrifying', 'has_displacement_risk']]
        
        X = features_df[feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        rent_pred = self.predict_rent(X_scaled)
        gent_prob = self.predict_gentrification(X_scaled)
        disp_prob = self.predict_displacement_risk(X_scaled)
        
        # Create results dataframe
        results = features_df[['area']].copy()
        results['predicted_rent'] = rent_pred
        results['gentrification_probability'] = gent_prob
        results['gentrification_risk_level'] = gent_prob.apply(
            lambda x: self.get_risk_level(x)[0]
        )
        results['displacement_risk_probability'] = disp_prob
        results['displacement_risk_level'] = disp_prob.apply(
            lambda x: self.get_risk_level(x)[0]
        )
        
        # Calculate combined risk score
        results['combined_risk_score'] = (
            (gent_prob + disp_prob) / 2
        )
        
        # Map UGMI to risk
        if 'UGMI' in features_df.columns:
            results['urban_growth_momentum_index'] = features_df['UGMI']
        
        print(f"  ✓ Predictions made for {len(results)} areas")
        
        return results
    
    def get_top_gentrifying_areas(self, results_df, top_n=10):
        """Get top N areas with highest gentrification risk"""
        top_areas = results_df.nlargest(top_n, 'gentrification_probability')
        return top_areas[['area', 'gentrification_probability', 'predicted_rent']]
    
    def get_top_displacement_areas(self, results_df, top_n=10):
        """Get top N areas with highest displacement risk"""
        top_areas = results_df.nlargest(top_n, 'displacement_risk_probability')
        return top_areas[['area', 'displacement_risk_probability', 'gentrification_probability']]
    
    def get_areas_by_risk_level(self, results_df, risk_level="High Risk"):
        """Filter areas by risk level"""
        filtered = results_df[results_df['gentrification_risk_level'] == risk_level]
        return filtered[['area', 'gentrification_probability', 'predicted_rent']]
    
    def generate_area_report(self, results_df, area_name):
        """Generate detailed report for a specific area"""
        area_data = results_df[results_df['area'] == area_name]
        
        if len(area_data) == 0:
            return f"Area '{area_name}' not found in predictions."
        
        area_data = area_data.iloc[0]
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         AREA GENTRIFICATION RISK REPORT                     ║
╚══════════════════════════════════════════════════════════════╝

AREA: {area_data['area'].upper()}

📊 PREDICTED METRICS:
  • Predicted Rent: ₹{area_data['predicted_rent']:,.0f}
  • Gentrification Probability: {area_data['gentrification_probability']:.1%}
  • Displacement Risk Probability: {area_data['displacement_risk_probability']:.1%}
  • Combined Risk Score: {area_data['combined_risk_score']:.2f}/1.00

⚠️  RISK LEVELS:
  • Gentrification: {area_data['gentrification_risk_level']}
  • Displacement: {area_data['displacement_risk_level']}

📈 GROWTH METRICS:
  • Urban Growth Momentum Index: {area_data.get('urban_growth_momentum_index', 0):.3f}

💡 INTERPRETATION:
"""
        
        gent_prob = area_data['gentrification_probability']
        disp_prob = area_data['displacement_risk_probability']
        
        if gent_prob > 0.7:
            report += "  → This area has HIGH gentrification risk. Rapid price increases expected.\n"
        elif gent_prob > 0.4:
            report += "  → This area has MODERATE gentrification risk. Price increases likely.\n"
        else:
            report += "  → This area has LOW gentrification risk. Stable prices expected.\n"
        
        if disp_prob > 0.7:
            report += "  → This area has HIGH displacement risk. Vulnerable populations at serious risk.\n"
        elif disp_prob > 0.4:
            report += "  → This area has MODERATE displacement risk. Some vulnerable populations at risk.\n"
        else:
            report += "  → This area has LOW displacement risk. Vulnerable populations relatively safe.\n"
        
        report += "\n" + "="*62 + "\n"
        
        return report
    
    def get_policy_simulation_impact(self, features_df, policy_changes):
        """
        Simulate impact of policy interventions
        
        policy_changes: dict with keys like:
        {
            'business_increase': 0.2,  # 20% more businesses
            'crime_reduction': 0.15,   # 15% crime reduction
            'metro_improvement': 0.1   # 10% improved metro access
        }
        """
        print("\n🏛️  Simulating policy impact...")
        
        features_modified = features_df.copy()
        
        # Apply policy changes
        if 'business_increase' in policy_changes:
            features_modified['business_density'] *= (1 + policy_changes['business_increase'])
            features_modified['commercial_activity'] *= (1 + policy_changes['business_increase'])
        
        if 'crime_reduction' in policy_changes:
            features_modified['crime_reduction_potential'] *= (1 + policy_changes['crime_reduction'])
        
        if 'metro_improvement' in policy_changes:
            features_modified['transport_accessibility'] *= (1 + policy_changes['metro_improvement'])
            features_modified['metro_proximity_score'] *= (1 + policy_changes['metro_improvement'])
        
        if 'population_increase' in policy_changes:
            features_modified['population_growth_potential'] *= (1 + policy_changes['population_increase'])
        
        # Recalculate UGMI with modified features
        if 'UGMI' in features_modified.columns:
            features_modified['UGMI'] = (
                features_modified['UGMI'] * (1 + sum(policy_changes.values()) * 0.1)
            ).clip(0, 1)
        
        # Get predictions with modified features
        results_before = self.predict_for_areas(features_df)
        results_after = self.predict_for_areas(features_modified)
        
        # Calculate changes
        comparison = pd.DataFrame({
            'area': results_before['area'],
            'gent_prob_before': results_before['gentrification_probability'],
            'gent_prob_after': results_after['gentrification_probability'],
            'gent_change': (results_after['gentrification_probability'] - 
                           results_before['gentrification_probability']) * 100,
            'disp_prob_before': results_before['displacement_risk_probability'],
            'disp_prob_after': results_after['displacement_risk_probability'],
            'disp_change': (results_after['displacement_risk_probability'] - 
                           results_before['displacement_risk_probability']) * 100
        })
        
        return comparison

# ============== STANDALONE FUNCTIONS ==============

def make_predictions(features_df):
    """Quick function to make predictions"""
    engine = PredictionEngine()
    return engine.predict_for_areas(features_df)

def generate_report_for_area(results_df, area_name):
    """Quick function to generate area report"""
    engine = PredictionEngine()
    return engine.generate_area_report(results_df, area_name)

if __name__ == "__main__":
    # Load features and make predictions
    features = utils.load_dataframe(config.FINAL_FEATURES, "Final features")
    predictions = make_predictions(features)
    utils.save_dataframe(predictions, config.RESULTS_DIR / "predictions.csv", "Predictions")
    print("\n" + "="*60)
    print("Predictions complete. Ready for visualization.")
    print("="*60)
