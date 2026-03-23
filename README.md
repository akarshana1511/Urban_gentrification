# 🏘️ Urban Gentrification Prediction System for Bengaluru

An **end-to-end machine learning system** that predicts gentrification risk, rent increase projections, and displacement vulnerability across Bengaluru neighborhoods.

## 📊 System Overview

```
RAW DATA
   ├── Housing Prices
   ├── Metro Ridership
   └── Restaurant Activity
   
   ↓
   
PREPROCESSING (Clean, Standardize, Aggregate)
   
   ↓
   
FEATURE ENGINEERING (20+ Features)
   └── Growth Metrics
   └── Density Metrics
   └── Transport Accessibility
   └── Urban Growth Momentum Index
   
   ↓
   
MODEL TRAINING (3 Specialized Models)
   ├── Rent Prediction (XGBoost)
   ├── Gentrification Risk (Random Forest)
   └── Displacement Risk (Logistic Regression)
   
   ↓
   
PREDICTIONS & VISUALIZATION
   ├── Risk Scores
   ├── Interactive Dashboard
   └── Policy Simulation
```

## 🚀 Quick Start

### 1️⃣ **Installation**

```bash
# Clone or navigate to project
cd d:\sem4\packages\pred_2

# Install dependencies
pip install -r requirements.txt

# (Optional) Create virtual environment first
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2️⃣ **Train the Models**

```bash
# Run the complete pipeline (preprocessing → training → predictions)
python train_pipeline.py
```

This will:
- ✅ Clean and preprocess all data
- ✅ Create 20+ engineered features
- ✅ Train 3 machine learning models
- ✅ Generate predictions for all areas
- ✅ Save models and results

**Expected output:**
```
✅ PIPELINE EXECUTION COMPLETE
📊 Summary:
  • Data processed: 123 areas
  • Features created: 24 features
  • Models trained: 3 (Rent, Gentrification, Displacement)
  • Predictions generated: 123 areas
```

### 3️⃣ **Launch the Web App**

```bash
# Start Streamlit app
streamlit run app.py

# Open browser: http://localhost:8501
```

## 📱 Web App Features

### 🏠 Dashboard
- Overview of high-risk areas
- Gentrification vs Displacement scatter plot
- Risk distribution charts

### 📊 Analysis
- **Top Gentrifying Areas**: Ranked by gentrification probability
- **Top Displacement Risk Areas**: Vulnerable neighborhoods
- **All Areas Comparison**: Complete dataset view with download

### 🎯 Area Lookup
- Search individual neighborhoods
- Detailed metrics and visualizations
- Risk gauge indicators
- UGMI scores

### 🏛️ Policy Simulator
- Adjust sliders for policy interventions
  - 📈 Business development increases
  - 🚇 Metro/transport improvements
  - 🛡️ Crime reduction initiatives
  - 👥 Population growth targets
- See projected impact on gentrification and displacement risk

### ℹ️ About
- System documentation
- Model information
- Technical stack details

## 📁 Project Structure

```
pred_2/
├── data/
│   ├── raw/                          # Original CSV files
│   │   ├── Bengaluru_House_Data.csv
│   │   ├── NammaMetro_Ridership_Dataset.csv
│   │   └── Bangalore restaurant chain.csv
│   └── processed/                    # Clean output from preprocessing
│       ├── processed_house_data.csv
│       ├── processed_restaurant_data.csv
│       ├── merged_dataset.csv
│       └── final_features.csv        # Ready for modeling
│
├── src/                              # Core Python modules
│   ├── __init__.py
│   ├── utils.py                      # Helper functions
│   ├── data_preprocessing.py          # Data cleaning
│   ├── feature_engineering.py         # Feature creation
│   ├── model_training.py              # Model training
│   └── predictions.py                 # Make predictions
│
├── models/                           # Saved trained models
│   ├── rent_prediction_model.pkl
│   ├── gentrification_model.pkl
│   ├── displacement_risk_model.pkl
│   └── scaler.pkl
│
├── results/                          # Output files
│   └── area_predictions.csv          # Final predictions
│
├── config.py                         # Configuration (paths, hyperparams)
├── train_pipeline.py                 # Master training script
├── app.py                            # Streamlit web app
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🔧 Configuration

Edit `config.py` to customize:
- Model hyperparameters (learning rate, depth, etc.)
- Thresholds (rent growth, displacement risk, etc.)
- Feature weights (UGMI calculation)
- File paths

Example:
```python
# Rent increase = gentrification
RENT_GROWTH_THRESHOLD = 0.20  # 20%

# Model params
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1
}
```

## 📊 Models Explained

### 1. Rent Prediction (XGBoost Regressor)
- **What it does**: Predicts future rent/property values
- **Why XGBoost**: Handles non-linear relationships well
- **Output**: Rent value (in crores/lakhs)
- **Performance**: R² > 0.75

### 2. Gentrification Classification (Random Forest)
- **What it does**: Classifies areas as gentrifying (rent growth > 20%)
- **Why Random Forest**: Excellent for feature importance + robustness
- **Output**: Probability 0-1 (0 = safe, 1 = gentrifying)
- **Performance**: F1-Score > 0.70

### 3. Displacement Risk (Logistic Regression)
- **What it does**: Identifies populations at risk of displacement
- **Why Logistic Regression**: Interpretable + fast
- **Output**: Probability 0-1 (0 = safe, 1 = at risk)
- **Performance**: AUC > 0.72

## 🎯 Features Used

| Category | Features |
|----------|----------|
| **Growth** | Price growth, Rent growth, Crime reduction, Population growth |
| **Density** | Business density, Commercial activity |
| **Transport** | Transport accessibility, Metro proximity |
| **Risk** | Gentrification risk, Displacement risk, Vulnerability |
| **Original** | Price, Total sqft, Bathrooms, Restaurants nearby, Ratings |

### Urban Growth Momentum Index (UGMI)
```
UGMI = 30% × Price Growth 
     + 20% × Business Density 
     + 20% × Transport Access 
     + 15% × Crime Reduction 
     + 15% × Population Growth
```

## 📈 Example Outputs

### Area Report
```
Area: MARATHAHALLI
📊 PREDICTED METRICS:
  • Predicted Rent: ₹4,500,000
  • Gentrification Probability: 78%
  • Displacement Risk Probability: 65%
  • Combined Risk Score: 0.71/1.00

⚠️ RISK LEVELS:
  • Gentrification: High Risk
  • Displacement: High Risk
```

### Policy Impact
```
Policy: Add 20% more businesses, improve metro by 15%
Results:
  • Average Gentrification Change: +3.5%
  • Average Displacement Change: -2.1%
  • Most affected: Whitefield (↑5.2%), Indiranagar (↑4.8%)
```

## 🔍 Understanding the Output

### Risk Levels
- 🟢 **Low Risk** (0-25%): Stable neighborhood
- 🟡 **Medium Risk** (25-50%): Watch out for changes
- 🔴 **High Risk** (50-80%): Significant gentrification/displacement risk
- 🟣 **Very High Risk** (80-100%): Critical intervention needed

### Risk Indicators
- **Gentrification Probability**: Likelihood of rapid rent/price increase
- **Displacement Risk**: Vulnerability of current population
- **Combined Risk Score**: Average of both risks
- **UGMI**: Overall growth potential (0 = stagnant, 1 = high growth)

## 💡 Common Questions

### Q: Why are some areas showing high gentrification but low displacement?
**A**: These areas have capacity for growth without existing vulnerable populations, or newcomers can afford higher rents.

### Q: Can I update predictions with new data?
**A**: Yes! Add new rows to raw CSV files and rerun `python train_pipeline.py`.

### Q: How often should I retrain?
**A**: Quarterly/Bi-annually as new housing/business data becomes available.

### Q: Can I use this for other cities?
**A**: Yes! Adapt the data sources for your city and follow the same pipeline.

## 🐛 Troubleshooting

### Error: "Predictions not found"
```
Solution: Run python train_pipeline.py first
```

### Error: "Models not found"
```
Solution: Training failed. Check data files exist in d:\sem4\packages\pred_2\data\raw\
```

### Streamlit: "No module named 'streamlit'"
```
Solution: pip install streamlit
```

### Slow performance
```
Solution: Reduce data size or use sampling in config.py
```

## 📚 Advanced Usage

### Use Models Programmatically

```python
from src.predictions import PredictionEngine
from src import utils
import config

# Load models
engine = PredictionEngine()

# Load features
features = utils.load_dataframe(config.FINAL_FEATURES, "Features")

# Make predictions
results = engine.predict_for_areas(features)

# Get top gentrifying areas
top_areas = engine.get_top_gentrifying_areas(results, top_n=10)

# Generate area report
report = engine.generate_area_report(results, area_name='Whitefield')
print(report)
```

### Simulate Policies

```python
policy_changes = {
    'business_increase': 0.25,      # 25% more business
    'crime_reduction': 0.15,        # 15% crime reduction
    'metro_improvement': 0.20,      # 20% better metro
    'population_increase': 0.10     # 10% population growth
}

comparison = engine.get_policy_simulation_impact(features, policy_changes)
print(comparison)
```

### Extract Feature Importance

```python
from src.model_training import ModelTrainer

trainer = ModelTrainer(features)
programmer = trainer.get_feature_importance()
# Shows which factors drive gentrification
```

## 🎓 Learning Resources

- **ML Models**: See `src/model_training.py` for implementation details
- **Features**: See `src/feature_engineering.py` for feature creation logic
- **Data Processing**: See `src/data_preprocessing.py` for data cleaning
- **Predictions**: See `src/predictions.py` for prediction generation

## 📝 Notes

- All prices are in **Indian Rupees (₹)**
- Data aggregated at **area level** (not individual buildings)
- Predictions based on **historical patterns** (future may differ)
- **Displacement risk** requires income data (currently simulated)

## 🤝 Contributing

To improve the system:
1. Add new data sources
2. Engineer additional features
3. Try different ML models
4. Improve visualizations
5. Add more policy scenarios

## 📞 Support

For questions or issues:
1. Check `README.md` (this file)
2. Review code comments in `src/` modules
3. Check `config.py` for configuration options
4. Run with verbose logging enabled

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Data Sources**: Kaggle, OpenCity, Namma Metro
- **Libraries**: scikit-learn, XGBoost, Streamlit, Plotly
- **Bengaluru Urban Planning Community**

---

**Happy Analyzing! 🏘️📊**

For updates: Keep checking the dashboard for latest predictions and insights.
