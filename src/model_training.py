"""
Model Training Module
Train three separate models:
1. Rent Prediction (Regression) - XGBoost
2. Gentrification Classification - Random Forest
3. Displacement Risk Classification - Logistic Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import utils
import config

class ModelTrainer:
    """
    Trains and evaluates machine learning models
    """
    
    def __init__(self, df):
        """
        Initialize with feature dataframe
        df: dataframe with features and targets
        """
        print("\n🤖 Initializing ModelTrainer...")
        self.df = df.copy()
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
        # Separate features and targets
        self.feature_columns = [col for col in self.df.columns 
                               if col not in ['future_rent_target', 'is_gentrifying', 
                                            'has_displacement_risk', 'area']]
        
        self.X = self.df[self.feature_columns]
        self.y_rent = self.df['future_rent_target']
        self.y_gentrification = self.df['is_gentrifying']
        self.y_displacement = self.df['has_displacement_risk']
        
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Samples: {len(self.X)}")
    
    def prepare_data(self):
        """
        Prepare and split data for training
        """
        print("\n📊 Preparing data for training...")
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.feature_columns)
        
        # Train-test split
        self.X_train, self.X_test, self.y_rent_train, self.y_rent_test = train_test_split(
            self.X_scaled, self.y_rent,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        _, _, self.y_gen_train, self.y_gen_test = train_test_split(
            self.X_scaled, self.y_gentrification,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        _, _, self.y_disp_train, self.y_disp_test = train_test_split(
            self.X_scaled, self.y_displacement,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        print(f"  Train size: {len(self.X_train)}")
        print(f"  Test size: {len(self.X_test)}")
    
    def train_rent_prediction_model(self):
        """
        Train XGBoost model for rent prediction (Regression)
        Target: future_rent
        """
        print("\n💰 Training Rent Prediction Model (XGBoost)...")
        
        model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        
        # Train
        model.fit(
            self.X_train, self.y_rent_train,
            eval_set=[(self.X_test, self.y_rent_test)],
            verbose=False
        )
        
        # Predict
        y_pred = model.predict(self.X_test)
        
        # Evaluate
        mse = mean_squared_error(self.y_rent_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_rent_test, y_pred)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        self.models['rent_prediction'] = model
        self.results['rent_prediction'] = {
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        return model
    
    def train_gentrification_model(self):
        """
        Train Random Forest for gentrification classification
        Target: is_gentrifying (binary)
        """
        print("\n🏘️  Training Gentrification Classification Model (Random Forest)...")
        
        model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        
        # Train
        model.fit(self.X_train, self.y_gen_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(self.y_gen_test, y_pred)
        precision = precision_score(self.y_gen_test, y_pred, zero_division=0)
        recall = recall_score(self.y_gen_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_gen_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(self.y_gen_test, y_pred_proba)
        except:
            auc = 0.0
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        self.models['gentrification'] = model
        self.results['gentrification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return model
    
    def train_displacement_risk_model(self):
        """
        Train Logistic Regression for displacement risk classification
        Target: has_displacement_risk (binary)
        """
        print("\n⚠️  Training Displacement Risk Model (Logistic Regression)...")
        
        model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
        
        # Train
        model.fit(self.X_train, self.y_disp_train)
        
        # Predict
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(self.y_disp_test, y_pred)
        precision = precision_score(self.y_disp_test, y_pred, zero_division=0)
        recall = recall_score(self.y_disp_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_disp_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(self.y_disp_test, y_pred_proba)
        except:
            auc = 0.0
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        self.models['displacement_risk'] = model
        self.results['displacement_risk'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return model
    
    def get_feature_importance(self):
        """
        Get feature importance from all models
        """
        print("\n📊 Extracting feature importance...")
        
        importance_dict = {}
        
        # XGBoost feature importance
        if 'rent_prediction' in self.models:
            importance_dict['rent_prediction'] = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models['rent_prediction'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Random Forest feature importance
        if 'gentrification' in self.models:
            importance_dict['gentrification'] = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models['gentrification'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Logistic Regression coefficients
        if 'displacement_risk' in self.models:
            importance_dict['displacement_risk'] = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(self.models['displacement_risk'].coef_[0])
            }).sort_values('importance', ascending=False)
        
        # Print top 10 features for each model
        for model_name, importance_df in importance_dict.items():
            print(f"\n  Top 10 features ({model_name}):")
            for idx, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']:30s}: {row['importance']:.4f}")
        
        return importance_dict
    
    def save_models(self):
        """
        Save trained models to disk
        """
        print("\n💾 Saving models...")
        
        utils.save_model(self.models['rent_prediction'], 
                        config.RENT_PREDICTION_MODEL, 
                        "Rent prediction model")
        
        utils.save_model(self.models['gentrification'],
                        config.GENTRIFICATION_MODEL,
                        "Gentrification model")
        
        utils.save_model(self.models['displacement_risk'],
                        config.DISPLACEMENT_RISK_MODEL,
                        "Displacement risk model")
        
        utils.save_model(self.scaler, config.SCALER_FILE, "Feature scaler")
    
    def run_full_pipeline(self):
        """
        Run complete model training pipeline
        """
        utils.print_section_header("MODEL TRAINING PIPELINE")
        
        self.prepare_data()
        self.train_rent_prediction_model()
        self.train_gentrification_model()
        self.train_displacement_risk_model()
        
        importance = self.get_feature_importance()
        self.save_models()
        
        print("\n✅ Model training complete!")
        print(f"   Models trained: {len(self.models)}")
        print(f"   Models saved: Yes")
        
        return self.models, self.results

# ============== STANDALONE FUNCTIONS ==============

def train_models(df):
    """Quick function to train all models"""
    trainer = ModelTrainer(df)
    return trainer.run_full_pipeline()

if __name__ == "__main__":
    # Load features and train models
    features = utils.load_dataframe(config.FINAL_FEATURES, "Final features")
    models, results = train_models(features)
    
    print("\n" + "="*60)
    print("Model training complete. Ready for predictions.")
    print("="*60)
