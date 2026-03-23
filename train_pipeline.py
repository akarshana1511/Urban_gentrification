"""
Master Training Script
Run the entire pipeline from data loading to model training
This is the main script to execute once when setting up the project
"""

import sys
import traceback
from src import utils, data_preprocessing, feature_engineering, model_training, predictions
import config

def run_complete_pipeline():
    """
    Execute the complete ML pipeline:
    1. Data preprocessing
    2. Feature engineering
    3. Model training
    4. Make predictions
    """
    
    print("\n" + "="*70)
    print("  URBAN GENTRIFICATION PREDICTION SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Create directories
        print("\n📁 Step 1: Creating project directories...")
        utils.create_directories()
        
        # Step 2: Data Preprocessing
        print("\n📂 Step 2: Data Preprocessing...")
        preprocessor = data_preprocessing.DataPreprocessor()
        merged_dataset = preprocessor.run_full_pipeline()
        
        # Step 3: Feature Engineering
        print("\n🔬 Step 3: Feature Engineering...")
        engineer = feature_engineering.FeatureEngineer(merged_dataset)
        features = engineer.run_full_pipeline()
        utils.save_dataframe(features, config.FINAL_FEATURES, "Final features")
        
        # Step 4: Model Training
        print("\n🤖 Step 4: Model Training...")
        trainer = model_training.ModelTrainer(features)
        models, results = trainer.run_full_pipeline()
        
        # Step 5: Make Predictions
        print("\n🔮 Step 5: Making Predictions...")
        engine = predictions.PredictionEngine()
        predictions_df = engine.predict_for_areas(features)
        utils.save_dataframe(
            predictions_df,
            config.RESULTS_DIR / "area_predictions.csv",
            "Area predictions"
        )
        
        # Print summary
        print("\n" + "="*70)
        print("  ✅ PIPELINE EXECUTION COMPLETE")
        print("="*70)
        
        print("\n📊 Summary:")
        print(f"  • Data processed: {len(merged_dataset)} areas")
        print(f"  • Features created: {len(features.columns)} features")
        print(f"  • Models trained: 3 (Rent, Gentrification, Displacement)")
        print(f"  • Predictions generated: {len(predictions_df)} areas")
        
        print("\n🎯 Top Gentrifying Areas:")
        top_gent = engine.get_top_gentrifying_areas(predictions_df, 5)
        for idx, row in top_gent.iterrows():
            print(f"  {row['area']:30s} | Gen. Prob: {row['gentrification_probability']:.1%}")
        
        print("\n⚠️  Top Displacement Risk Areas:")
        top_disp = engine.get_top_displacement_areas(predictions_df, 5)
        for idx, row in top_disp.iterrows():
            print(f"  {row['area']:30s} | Disp. Risk: {row['displacement_risk_probability']:.1%}")
        
        print("\n🚀 Next Steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open: http://localhost:8501")
        print("  3. Explore predictions and visualizations!")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("  ❌ PIPELINE EXECUTION FAILED")
        print("="*70)
        print(f"\nError: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_complete_pipeline()
    sys.exit(0 if success else 1)
