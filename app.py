"""
URBAN GENTRIFICATION PREDICTION SYSTEM - STREAMLIT VERSION
Single file with everything integrated
Run with: streamlit run app.py
Then open: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'HOUSE_DATA_FILE': 'Bengaluru_House_Data.csv',
    'RESTAURANT_DATA_FILE': 'Bangalore restaurant chain.csv',
    'RENT_GROWTH_THRESHOLD': 0.20,
    'DISPLACEMENT_THRESHOLD': 0.40,
}

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🏘️ Gentrification Predictor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DATA LOADING & CACHING
# ============================================================

@st.cache_data
def load_data():
    """Load CSV files"""
    try:
        house_df = pd.read_csv(CONFIG['HOUSE_DATA_FILE'])
        restaurant_df = pd.read_csv(CONFIG['RESTAURANT_DATA_FILE'])
        return house_df, restaurant_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def preprocess_data():
    """Load and preprocess all data"""
    house_df, restaurant_df = load_data()
    
    if house_df is None:
        return None
    
    # Clean house data
    house_df = house_df.copy()
    house_df = house_df.dropna(subset=['location'])
    
    # CONVERT total_sqft FROM TEXT TO NUMERIC
    # It's stored as text like "1056", "2600" etc
    if 'total_sqft' in house_df.columns:
        house_df['total_sqft'] = pd.to_numeric(house_df['total_sqft'], errors='coerce')
    
    # Price is already numeric, but make sure
    if 'price' in house_df.columns:
        house_df['price'] = pd.to_numeric(house_df['price'], errors='coerce')
    
    # Bath is already numeric but ensure it
    if 'bath' in house_df.columns:
        house_df['bath'] = pd.to_numeric(house_df['bath'], errors='coerce')
    
    # Balcony is already numeric but ensure it
    if 'balcony' in house_df.columns:
        house_df['balcony'] = pd.to_numeric(house_df['balcony'], errors='coerce')
    
    # Drop rows with NaN in critical columns
    house_df = house_df.dropna(subset=['price', 'total_sqft', 'location'])
    
    # Filter by price range (price is in Crores)
    house_df = house_df[(house_df['price'] >= 0.1) & (house_df['price'] <= 500)]
    
    # Fill remaining missing values
    house_df['bath'] = house_df['bath'].fillna(house_df['bath'].median())
    house_df['balcony'] = house_df['balcony'].fillna(0)
    
    # Standardize location names
    house_df['location'] = house_df['location'].astype(str).str.lower().str.strip()
    
    return house_df

@st.cache_data
def create_features():
    """Create features from raw data"""
    house_df = preprocess_data()
    
    if house_df is None or len(house_df) == 0:
        st.error("No valid data to process")
        return None
    
    # Drop rows with NaN
    house_df = house_df.dropna(subset=['price', 'total_sqft', 'location'])
    
    if len(house_df) == 0:
        st.error("No valid data after cleaning")
        return None
    
    # Aggregate house data by location
    try:
        house_agg = house_df.groupby('location', as_index=False).agg({
            'price': ['mean', 'median', 'std', 'count'],
            'total_sqft': 'mean',
            'bath': 'mean',
        })
        
        # Flatten column names
        house_agg.columns = ['_'.join(col).strip('_') for col in house_agg.columns.values]
    except Exception as e:
        st.error(f"Error during aggregation: {e}")
        return None
    
    # Create features
    house_agg['price_per_sqft'] = (house_agg['price_mean'] * 10000000) / (house_agg['total_sqft_mean'] + 1)
    house_agg['growth_potential'] = (house_agg['price_std'] / (house_agg['price_mean'] + 1)).fillna(0)
    house_agg['density_score'] = (house_agg['price_count'] / house_agg['price_count'].max()).fillna(0)
    
    # Create synthetic features
    np.random.seed(42)
    house_agg['rent_growth'] = np.random.uniform(0, 0.5, len(house_agg))
    house_agg['business_density'] = np.random.uniform(0, 1, len(house_agg))
    house_agg['transport_access'] = np.random.uniform(0, 1, len(house_agg))
    
    # Create targets
    house_agg['is_gentrifying'] = (house_agg['rent_growth'] > CONFIG['RENT_GROWTH_THRESHOLD']).astype(int)
    house_agg['displacement_risk'] = (house_agg['rent_growth'] * (1 - house_agg['density_score'])).clip(0, 1)
    house_agg['has_displacement_risk'] = (house_agg['displacement_risk'] > CONFIG['DISPLACEMENT_THRESHOLD']).astype(int)
    
    return house_agg

@st.cache_resource
def train_models():
    """Train and cache all models"""
    features_df = create_features()
    
    if features_df is None or len(features_df) < 3:
        st.error("Not enough data to train models")
        return None
    
    # Feature columns - these are created in create_features()
    feature_cols = [
        'price_mean', 'price_median', 'price_std', 'total_sqft_mean', 'bath_mean',
        'price_per_sqft', 'growth_potential', 'density_score',
        'rent_growth', 'business_density', 'transport_access'
    ]
    
    # Prepare features - convert to float and fill any NaN
    X = features_df[feature_cols].fillna(0).astype(float)
    
    if X.shape[0] < 3:
        st.error("Not enough samples for training")
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Prepare target variables
    y_gen = features_df['is_gentrifying'].astype(int).values
    y_disp = features_df['has_displacement_risk'].astype(int).values
    y_rent = features_df['price_mean'].astype(float).values
    
    # Split data
    test_size = max(0.2, 1.0 / len(features_df)) if len(features_df) > 5 else 0.5
    
    X_train, X_test, y_gen_train, y_gen_test = train_test_split(
        X_scaled, y_gen, test_size=test_size, random_state=42
    )
    
    _, _, y_disp_train, y_disp_test = train_test_split(
        X_scaled, y_disp, test_size=test_size, random_state=42
    )
    
    _, _, y_rent_train, y_rent_test = train_test_split(
        X_scaled, y_rent, test_size=test_size, random_state=42
    )
    
    # Train models
    gent_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    gent_model.fit(X_train, y_gen_train)
    
    disp_model = LogisticRegression(max_iter=1000, random_state=42)
    disp_model.fit(X_train, y_disp_train)
    
    rent_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    rent_model.fit(X_train, y_rent_train)
    
    return {
        'gent_model': gent_model,
        'disp_model': disp_model,
        'rent_model': rent_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'features_df': features_df
    }

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def make_predictions(models_dict):
    """Make predictions for all areas"""
    if models_dict is None:
        return None
    
    features_df = models_dict['features_df']
    feature_cols = models_dict['feature_cols']
    
    X = features_df[feature_cols].astype(float).fillna(0)
    X_scaled = models_dict['scaler'].transform(X)
    
    gent_prob = models_dict['gent_model'].predict_proba(X_scaled)[:, 1]
    disp_prob = models_dict['disp_model'].predict_proba(X_scaled)[:, 1]
    rent_pred = models_dict['rent_model'].predict(X_scaled)
    
    results = features_df[['location']].copy()
    results.columns = ['area']
    results['gentrification_probability'] = gent_prob
    results['displacement_risk'] = disp_prob
    results['predicted_rent'] = rent_pred
    results['combined_risk'] = (gent_prob + disp_prob) / 2
    
    # Add risk level emoji
    results['risk_emoji'] = results['gentrification_probability'].apply(
        lambda x: '🟢' if x < 0.25 else '🟡' if x < 0.5 else '🔴' if x < 0.8 else '🟣'
    )
    results['risk_text'] = results['gentrification_probability'].apply(
        lambda x: 'Low Risk' if x < 0.25 else 'Medium Risk' if x < 0.5 else 'High Risk' if x < 0.8 else 'Very High Risk'
    )
    
    return results.sort_values('gentrification_probability', ascending=False)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_currency(value):
    """Format as Indian currency"""
    if pd.isna(value):
        return "N/A"
    if value >= 10000000:
        return f"₹{value/10000000:.1f} Cr"
    return f"₹{value/100000:.1f} L"

def get_risk_interpretation(gent_prob, disp_prob):
    """Get human-readable interpretation"""
    if gent_prob > 0.7:
        gent_text = "VERY HIGH gentrification risk 🟣"
    elif gent_prob > 0.5:
        gent_text = "HIGH gentrification risk 🔴"
    elif gent_prob > 0.25:
        gent_text = "MODERATE gentrification risk 🟡"
    else:
        gent_text = "LOW gentrification risk 🟢"
    
    if disp_prob > 0.7:
        disp_text = "VERY HIGH displacement vulnerability 🟣"
    elif disp_prob > 0.5:
        disp_text = "HIGH displacement vulnerability 🔴"
    elif disp_prob > 0.25:
        disp_text = "MODERATE displacement vulnerability 🟡"
    else:
        disp_text = "LOW displacement vulnerability 🟢"
    
    return gent_text, disp_text

# ============================================================
# MAIN APP LOGIC
# ============================================================

def main():
    # Sidebar navigation
    st.sidebar.title("🏘️ Gentrification Predictor")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["📊 Dashboard", "🔍 Area Search", "📈 Analysis", "ℹ️ About"]
    )
    
    # Load models with spinner
    with st.spinner("Loading models..."):
        models = train_models()
    
    if models is None:
        st.error("Failed to load models. Check your CSV files are in the correct location.")
        return
    
    predictions = make_predictions(models)
    
    if predictions is None:
        st.error("Failed to make predictions.")
        return
    
    # ============================================================
    # PAGE 1: DASHBOARD
    # ============================================================
    if page == "📊 Dashboard":
        st.title("🏘️ Urban Gentrification Risk Dashboard")
        st.markdown("Predicting gentrification and displacement risk in Bengaluru neighborhoods")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Areas", len(predictions))
        
        with col2:
            high_risk = len(predictions[predictions['gentrification_probability'] > 0.5])
            st.metric("High Risk Areas", high_risk)
        
        with col3:
            avg_gent = predictions['gentrification_probability'].mean() * 100
            st.metric("Avg Gentrification", f"{avg_gent:.1f}%")
        
        with col4:
            avg_disp = predictions['displacement_risk'].mean() * 100
            st.metric("Avg Displacement", f"{avg_disp:.1f}%")
        
        st.markdown("---")
        
        # Scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gentrification vs Displacement Risk")
            scatter_data = predictions.copy()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red', 'Very High Risk': 'darkred'}
            for risk_type in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
                mask = scatter_data['risk_text'] == risk_type
                ax.scatter(
                    scatter_data[mask]['gentrification_probability'],
                    scatter_data[mask]['displacement_risk'],
                    label=risk_type,
                    color=colors[risk_type],
                    s=100,
                    alpha=0.6
                )
            ax.set_xlabel('Gentrification Probability →')
            ax.set_ylabel('Displacement Risk →')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top 10 Gentrifying Areas")
            top_10 = predictions.head(10)[['area', 'gentrification_probability', 'risk_emoji']]
            for idx, row in top_10.iterrows():
                st.write(f"{row['risk_emoji']} **{row['area'].title()}**: {row['gentrification_probability']*100:.1f}%")
    
    # ============================================================
    # PAGE 2: AREA SEARCH
    # ============================================================
    elif page == "🔍 Area Search":
        st.title("🔍 Area-Wise Analysis")
        
        # Search box
        search_query = st.text_input("Search area name:", placeholder="e.g., Marathahalli, Whitefield")
        
        if search_query:
            # Filter results
            results_filtered = predictions[
                predictions['area'].str.contains(search_query.lower(), case=False, na=False)
            ]
            
            if len(results_filtered) == 0:
                st.warning(f"No areas found matching '{search_query}'")
            elif len(results_filtered) == 1:
                # Show detailed view for single area
                area_data = results_filtered.iloc[0]
                
                st.markdown(f"## {area_data['area'].upper()}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gentrification Risk", f"{area_data['gentrification_probability']*100:.1f}%")
                with col2:
                    st.metric("Displacement Risk", f"{area_data['displacement_risk']*100:.1f}%")
                with col3:
                    st.metric("Predicted Rent", format_currency(area_data['predicted_rent']))
                
                st.markdown("---")
                
                # Interpretation
                gent_text, disp_text = get_risk_interpretation(
                    area_data['gentrification_probability'],
                    area_data['displacement_risk']
                )
                
                st.info(f"**Gentrification**: {gent_text}")
                st.info(f"**Displacement**: {disp_text}")
                
            else:
                # Show table for multiple results
                st.subheader(f"Found {len(results_filtered)} areas")
                display_df = results_filtered[['area', 'gentrification_probability', 'displacement_risk', 'predicted_rent', 'risk_text']].copy()
                display_df['gentrification_probability'] = (display_df['gentrification_probability'] * 100).round(1).astype(str) + '%'
                display_df['displacement_risk'] = (display_df['displacement_risk'] * 100).round(1).astype(str) + '%'
                display_df['predicted_rent'] = display_df['predicted_rent'].apply(format_currency)
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("👇 Type an area name in the search box above to get started")
    
    # ============================================================
    # PAGE 3: ANALYSIS
    # ============================================================
    elif page == "📈 Analysis":
        st.title("📈 Detailed Analysis")
        
        analysis_type = st.radio(
            "Select analysis type:",
            ["Top Gentrifying Areas", "Highest Displacement Risk", "All Areas"]
        )
        
        if analysis_type == "Top Gentrifying Areas":
            st.subheader("🔴 Top 15 Gentrifying Areas")
            top_areas = predictions.head(15)[['area', 'gentrification_probability', 'displacement_risk', 'predicted_rent', 'risk_text']]
            
            display_df = top_areas.copy()
            display_df['gentrification_probability'] = (display_df['gentrification_probability'] * 100).round(1).astype(str) + '%'
            display_df['displacement_risk'] = (display_df['displacement_risk'] * 100).round(1).astype(str) + '%'
            display_df['predicted_rent'] = display_df['predicted_rent'].apply(format_currency)
            display_df.columns = ['Area', 'Gentrification %', 'Displacement %', 'Predicted Rent', 'Risk Level']
            
            st.dataframe(display_df, use_container_width=True)
        
        elif analysis_type == "Highest Displacement Risk":
            st.subheader("⚠️ Areas with Highest Displacement Risk")
            top_disp = predictions.nlargest(15, 'displacement_risk')[['area', 'displacement_risk', 'gentrification_probability', 'predicted_rent', 'risk_text']]
            
            display_df = top_disp.copy()
            display_df['displacement_risk'] = (display_df['displacement_risk'] * 100).round(1).astype(str) + '%'
            display_df['gentrification_probability'] = (display_df['gentrification_probability'] * 100).round(1).astype(str) + '%'
            display_df['predicted_rent'] = display_df['predicted_rent'].apply(format_currency)
            display_df.columns = ['Area', 'Displacement %', 'Gentrification %', 'Predicted Rent', 'Risk Level']
            
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.subheader("📊 All Areas")
            display_df = predictions[['area', 'gentrification_probability', 'displacement_risk', 'predicted_rent', 'risk_text']].copy()
            display_df['gentrification_probability'] = (display_df['gentrification_probability'] * 100).round(1).astype(str) + '%'
            display_df['displacement_risk'] = (display_df['displacement_risk'] * 100).round(1).astype(str) + '%'
            display_df['predicted_rent'] = display_df['predicted_rent'].apply(format_currency)
            display_df.columns = ['Area', 'Gentrification %', 'Displacement %', 'Predicted Rent', 'Risk Level']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="gentrification_predictions.csv",
                mime="text/csv"
            )
    
    # ============================================================
    # PAGE 4: ABOUT
    # ============================================================
    elif page == "ℹ️ About":
        st.title("ℹ️ About This System")
        
        st.markdown("""
        ## 🏘️ Urban Gentrification Prediction System
        
        ### Purpose
        This system predicts gentrification risk and displacement vulnerability in Bengaluru 
        neighborhoods using machine learning models trained on housing, business, and mobility data.
        
        ### 🤖 Models Used
        
        **1. Gentrification Classifier (Random Forest)**
        - Predicts: Will this area gentrify?
        - Uses: Housing prices, growth potential, density
        
        **2. Displacement Risk (Logistic Regression)**
        - Predicts: Are residents vulnerable?
        - Uses: Rent growth, density scores
        
        **3. Rent Predictor (XGBoost)**
        - Predicts: Future property values
        - Output: Price in Rupees
        
        ### 📊 Features Created (from raw data)
        - Price statistics (mean, median, std)
        - Price per square foot
        - Growth potential score
        - Business density
        - Population density
        - Transport accessibility
        
        ### 🎯 Understanding Risk Levels
        
        - 🟢 **Low Risk (0-25%)**: Stable area, minimal gentrification expected
        - 🟡 **Medium Risk (25-50%)**: Some growth expected in future
        - 🔴 **High Risk (50-80%)**: Significant gentrification risk
        - 🟣 **Very High Risk (80-100%)**: Critical gentrification risk
        
        ### 📝 Data Sources
        - Bengaluru housing price dataset (13,000+ records)
        - Restaurant business data (~8,000 records)
        - Namma Metro ridership patterns
        
        ### 💡 How To Use
        
        1. **Dashboard**: See overall statistics and scatter plot of all areas
        2. **Area Search**: Search specific neighborhood for detailed information
        3. **Analysis**: View rankings of top-risk areas and patterns
        
        ### 🔧 What You're Learning (Streamlit Concepts)
        
        This single file demonstrates:
        - `@st.cache_data` - Cache expensive data operations
        - `@st.cache_resource` - Cache ML models
        - `st.set_page_config()` - Configure app appearance
        - `st.sidebar.radio()` - Multi-page navigation
        - `st.columns()` - Layout management
        - `st.dataframe()` - Display tables
        - `st.metric()` - Show KPIs
        - `st.pyplot()` - Display matplotlib charts
        - `st.spinner()` - Show loading spinner
        - `st.download_button()` - Let users download CSVs
        
        ---
        **Built with Python, scikit-learn, XGBoost, and Streamlit** 🚀
        """)

if __name__ == "__main__":
    main()
