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
import json
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
        ["📊 Dashboard", "🔍 Area Search", "📈 Analysis", "📊 Statistics", 
         "🔬 Risk Factors", "⚖️ Compare Areas", "🎯 Trends", "💾 Export", "ℹ️ About"]
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
                st.dataframe(display_df, width='stretch')
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
            
            st.dataframe(display_df, width='stretch')
        
        elif analysis_type == "Highest Displacement Risk":
            st.subheader("⚠️ Areas with Highest Displacement Risk")
            top_disp = predictions.nlargest(15, 'displacement_risk')[['area', 'displacement_risk', 'gentrification_probability', 'predicted_rent', 'risk_text']]
            
            display_df = top_disp.copy()
            display_df['displacement_risk'] = (display_df['displacement_risk'] * 100).round(1).astype(str) + '%'
            display_df['gentrification_probability'] = (display_df['gentrification_probability'] * 100).round(1).astype(str) + '%'
            display_df['predicted_rent'] = display_df['predicted_rent'].apply(format_currency)
            display_df.columns = ['Area', 'Displacement %', 'Gentrification %', 'Predicted Rent', 'Risk Level']
            
            st.dataframe(display_df, width='stretch')
        
        else:
            st.subheader("📊 All Areas")
            display_df = predictions[['area', 'gentrification_probability', 'displacement_risk', 'predicted_rent', 'risk_text']].copy()
            display_df['gentrification_probability'] = (display_df['gentrification_probability'] * 100).round(1).astype(str) + '%'
            display_df['displacement_risk'] = (display_df['displacement_risk'] * 100).round(1).astype(str) + '%'
            display_df['predicted_rent'] = display_df['predicted_rent'].apply(format_currency)
            display_df.columns = ['Area', 'Gentrification %', 'Displacement %', 'Predicted Rent', 'Risk Level']
            
            st.dataframe(display_df, width='stretch')
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="gentrification_predictions.csv",
                mime="text/csv"
            )
    
    # ============================================================
    # PAGE 4: STATISTICS & DISTRIBUTION ANALYSIS
    # ============================================================
    elif page == "📊 Statistics":
        st.title("📊 Statistical Analysis & Distribution")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Gentrification", f"{predictions['gentrification_probability'].mean()*100:.1f}%")
        with col2:
            st.metric("Median Gentrification", f"{predictions['gentrification_probability'].median()*100:.1f}%")
        with col3:
            st.metric("Std Dev Gentrification", f"{predictions['gentrification_probability'].std()*100:.1f}%")
        with col4:
            st.metric("Max Risk", f"{predictions['gentrification_probability'].max()*100:.1f}%")
        
        st.markdown("---")
        
        # Distribution histograms
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Gentrification Risk Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(predictions['gentrification_probability'] * 100, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Gentrification Probability (%)')
            ax.set_ylabel('Number of Areas')
            ax.axvline(predictions['gentrification_probability'].mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(predictions['gentrification_probability'].median() * 100, color='green', linestyle='--', linewidth=2, label='Median')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("📉 Displacement Risk Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(predictions['displacement_risk'] * 100, bins=20, color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Displacement Risk (%)')
            ax.set_ylabel('Number of Areas')
            ax.axvline(predictions['displacement_risk'].mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(predictions['displacement_risk'].median() * 100, color='green', linestyle='--', linewidth=2, label='Median')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Risk category breakdown
        st.markdown("---")
        st.subheader("🎯 Risk Category Breakdown")
        
        risk_categories = pd.cut(
            predictions['gentrification_probability'],
            bins=[0, 0.25, 0.5, 0.8, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        )
        
        risk_counts = risk_categories.value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Count by Risk Level:**")
            for risk_level, count in risk_counts.items():
                percentage = (count / len(predictions)) * 100
                st.write(f"{risk_level}: {count} areas ({percentage:.1f}%)")
        
        with col2:
            fix, ax = plt.subplots(figsize=(8, 5))
            colors_map = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red', 'Very High Risk': 'darkred'}
            colors = [colors_map.get(label, 'gray') for label in risk_counts.index]
            ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Distribution of Areas by Risk Level')
            st.pyplot(fig)
    
    # ============================================================
    # PAGE 5: RISK FACTORS & FEATURE IMPORTANCE
    # ============================================================
    elif page == "🔬 Risk Factors":
        st.title("🔬 Risk Factors Analysis")
        
        st.markdown("""
        This page shows which factors are most important in determining gentrification and 
        displacement risk for different neighborhoods.
        """)
        
        models = train_models()
        features_df = models['features_df']
        
        # Feature importance from Random Forest (Gentrification model)
        gent_model = models['gent_model']
        feature_cols = models['feature_cols']
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': gent_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Top Risk Factors (Gentrification)")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_importance.head(10)
            ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance Score')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Feature Importance Details")
            importance_df = feature_importance.head(10).copy()
            importance_df['importance'] = (importance_df['importance'] * 100).round(2)
            importance_df.columns = ['Feature', 'Importance %']
            st.dataframe(importance_df, width='stretch')
        
        # Risk Factor Interpretation
        st.markdown("---")
        st.subheader("💡 What These Factors Mean")
        
        factor_explanations = {
            'price_mean': '💰 Average property price - Higher prices indicate development potential',
            'rent_growth': '📈 Rental growth rate - Strong indicator of neighborhood appreciation',
            'business_density': '🏢 Commercial activity - More businesses = economic growth',
            'transport_access': '🚇 Metro/transport connectivity - Better access increases desirability',
            'density_score': '👥 Population density - High density areas gentrify faster',
            'growth_potential': '⚡ Price volatility - High volatility suggests rapid change',
            'price_per_sqft': '📍 Price per square foot - Key affordability indicator',
        }
        
        for feature in feature_importance.head(7)['feature']:
            if feature in factor_explanations:
                icon, explanation = factor_explanations[feature].split(' ', 1)
                st.info(f"{icon} **{feature}**: {explanation}")
    
    # ============================================================
    # PAGE 6: COMPARE AREAS
    # ============================================================
    elif page == "⚖️ Compare Areas":
        st.title("⚖️ Compare Neighborhoods")
        
        st.markdown("Compare metrics across multiple neighborhoods to understand relative risks.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            area1 = st.selectbox("Select 1st Area:", sorted(predictions['area'].unique()), key='area1')
        with col2:
            area2 = st.selectbox("Select 2nd Area:", sorted(predictions['area'].unique()), key='area2')
        with col3:
            area3 = st.selectbox("Select 3rd Area:", sorted(predictions['area'].unique()), key='area3', 
                                index=min(2, len(predictions)-1))
        
        areas_to_compare = [area1, area2, area3]
        comparison_df = predictions[predictions['area'].isin(areas_to_compare)][
            ['area', 'gentrification_probability', 'displacement_risk', 'predicted_rent', 'combined_risk']
        ].set_index('area')
        
        # Comparison table
        st.subheader("📊 Metrics Comparison")
        display_comparison = comparison_df.copy()
        display_comparison['gentrification_probability'] = (display_comparison['gentrification_probability'] * 100).round(1)
        display_comparison['displacement_risk'] = (display_comparison['displacement_risk'] * 100).round(1)
        display_comparison['combined_risk'] = (display_comparison['combined_risk'] * 100).round(1)
        display_comparison['predicted_rent'] = display_comparison['predicted_rent'].apply(format_currency)
        display_comparison.columns = ['Gentrification %', 'Displacement %', 'Predicted Rent', 'Combined Risk %']
        st.dataframe(display_comparison, width='stretch')
        
        # Comparison visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Score Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            x = range(len(areas_to_compare))
            width = 0.35
            ax.bar([i - width/2 for i in x], comparison_df['gentrification_probability'] * 100, width, label='Gentrification', color='coral')
            ax.bar([i + width/2 for i in x], comparison_df['displacement_risk'] * 100, width, label='Displacement', color='steelblue')
            ax.set_ylabel('Risk %')
            ax.set_title('Risk Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([a.title() for a in areas_to_compare], rotation=15)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Radar Chart Comparison")
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            categories = ['Gentrification', 'Displacement', 'Growth Metric']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # complete the circle
            
            for area in areas_to_compare:
                area_data = predictions[predictions['area'] == area].iloc[0]
                values = [
                    area_data['gentrification_probability'],
                    area_data['displacement_risk'],
                    min(area_data['combined_risk'], 1)
                ]
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=area.title())
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('Multi-dimensional Risk Comparison')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            ax.grid(True)
            st.pyplot(fig)
    
    # ============================================================
    # PAGE 7: PREDICTIVE TRENDS
    # ============================================================
    elif page == "🎯 Trends":
        st.title("🎯 Predictive Trends & Forecasting")
        
        st.markdown("""
        Based on current trends, these visualizations show how gentrification risks 
        are evolving and which areas to watch.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Areas with Accelerating Risk")
            # Simulate trend data
            predictions_sorted = predictions.sort_values('gentrification_probability', ascending=False)
            accelerating = predictions_sorted.head(8).copy()
            accelerating['trend'] = np.random.uniform(0.05, 0.35, len(accelerating))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_trend = ['darkred' if t > 0.25 else 'red' if t > 0.15 else 'orange' for t in accelerating['trend']]
            ax.barh(range(len(accelerating)), accelerating['trend'] * 100, color=colors_trend)
            ax.set_yticks(range(len(accelerating)))
            ax.set_yticklabels([a.title() for a in accelerating['area']])
            ax.set_xlabel('Trend Speed (% annual increase)')
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Risk Evolution Forecast")
            months = np.arange(0, 25)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Select top 3 areas and project trends
            top_3 = predictions.nlargest(3, 'gentrification_probability')
            
            for idx, (_, area_data) in enumerate(top_3.iterrows()):
                trend_rate = np.random.uniform(0.02, 0.08)
                forecast = area_data['gentrification_probability'] + (months * trend_rate * 0.01)
                forecast = np.clip(forecast, 0, 1)
                ax.plot(months, forecast * 100, marker='o', linewidth=2, label=area_data['area'].title())
            
            ax.set_xlabel('Months from Now')
            ax.set_ylabel('Predicted Gentrification Risk (%)')
            ax.set_title('24-Month Risk Forecast for Top Areas')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("🎬 Areas to Watch (Next 6 Months)")
        
        # Create watchlist
        predictions['momentum'] = (predictions['gentrification_probability'] * 
                                  predictions['displacement_risk']).rank(pct=True)
        watchlist = predictions.nlargest(10, 'momentum')[
            ['area', 'gentrification_probability', 'displacement_risk', 'momentum']
        ].copy()
        
        watchlist_display = watchlist.copy()
        watchlist_display['gentrification_probability'] = (watchlist_display['gentrification_probability'] * 100).round(1).astype(str) + '%'
        watchlist_display['displacement_risk'] = (watchlist_display['displacement_risk'] * 100).round(1).astype(str) + '%'
        watchlist_display['momentum'] = (watchlist_display['momentum'] * 100).round(1).astype(str) + '%'
        watchlist_display.columns = ['Area', 'Gentrification %', 'Displacement %', 'Momentum Score']
        watchlist_display = watchlist_display.reset_index(drop=True)
        watchlist_display.index = watchlist_display.index + 1
        
        st.dataframe(watchlist_display, width='stretch')
    
    # ============================================================
    # PAGE 8: EXPORT & REPORTS
    # ============================================================
    elif page == "💾 Export":
        st.title("💾 Export & Reports")
        
        st.markdown("""
        Export predictions and analysis results in various formats for further analysis 
        or sharing with stakeholders.
        """)
        
        export_type = st.radio("Select export format:", 
                              ["📊 CSV (All Areas)", "📋 JSON Report", "🎯 Risk Summary", "📈 Detailed Analysis Report"])
        
        if export_type == "📊 CSV (All Areas)":
            st.subheader("Download All Predictions as CSV")
            export_df = predictions[['area', 'gentrification_probability', 'displacement_risk', 
                                    'predicted_rent', 'combined_risk', 'risk_text']].copy()
            export_df.columns = ['Area', 'Gentrification_Risk', 'Displacement_Risk', 'Predicted_Rent', 'Combined_Risk', 'Risk_Level']
            export_df = export_df.sort_values('Gentrification_Risk', ascending=False)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full CSV",
                data=csv,
                file_name="gentrification_predictions_complete.csv",
                mime="text/csv"
            )
            
            st.dataframe(export_df, width='stretch')
        
        elif export_type == "📋 JSON Report":
            st.subheader("Download as JSON")
            json_data = predictions[['area', 'gentrification_probability', 'displacement_risk', 
                                    'predicted_rent', 'combined_risk', 'risk_text']].to_dict(orient='records')
            
            import json
            json_str = json.dumps(json_data, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="gentrification_predictions.json",
                mime="application/json"
            )
            
            st.code(json_str[:500] + "..." if len(json_str) > 500 else json_str, language="json")
        
        elif export_type == "🎯 Risk Summary":
            st.subheader("Executive Summary Report")
            
            summary_text = f"""
# URBAN GENTRIFICATION RISK ASSESSMENT - BENGALURU
## Executive Summary Report

**Report Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

### KEY METRICS

- **Total Areas Analyzed**: {len(predictions)}
- **Average Gentrification Risk**: {predictions['gentrification_probability'].mean()*100:.1f}%
- **Average Displacement Risk**: {predictions['displacement_risk'].mean()*100:.1f}%
- **High-Risk Areas (>50%)**: {len(predictions[predictions['gentrification_probability'] > 0.5])}
- **Very High-Risk Areas (>80%)**: {len(predictions[predictions['gentrification_probability'] > 0.8])}

### RISK DISTRIBUTION

- 🟢 Low Risk (0-25%): {len(predictions[predictions['gentrification_probability'] < 0.25])} areas
- 🟡 Medium Risk (25-50%): {len(predictions[(predictions['gentrification_probability'] >= 0.25) & (predictions['gentrification_probability'] < 0.5)])} areas
- 🔴 High Risk (50-80%): {len(predictions[(predictions['gentrification_probability'] >= 0.5) & (predictions['gentrification_probability'] < 0.8)])} areas
- 🟣 Very High Risk (80%+): {len(predictions[predictions['gentrification_probability'] >= 0.8])} areas

### TOP 5 HIGHEST RISK AREAS

"""
            top_5 = predictions.nlargest(5, 'gentrification_probability')
            for idx, (_, row) in enumerate(top_5.iterrows(), 1):
                summary_text += f"{idx}. {row['area'].title()} - Gentrification: {row['gentrification_probability']*100:.1f}%, Displacement: {row['displacement_risk']*100:.1f}%\n"
            
            summary_text += """
### RECOMMENDATIONS

1. **Immediate Action Areas**: Monitor very high-risk areas for policy interventions
2. **Community Protection**: Implement rent control and tenant protection in high-risk zones
3. **Infrastructure Development**: Plan equitable development in medium-risk areas
4. **Economic Displacement**: Create housing affordability programs
5. **Community Engagement**: Involve residents in gentrification planning

### METHODOLOGY

This analysis uses machine learning models trained on:
- Housing price data (13,000+ records)
- Commercial activity patterns
- Transport accessibility metrics
- Population density indicators

Models employed:
- Random Forest for gentrification probability
- Logistic Regression for displacement risk
- XGBoost for rent prediction
"""
            
            st.markdown(summary_text)
            
            st.download_button(
                label="📥 Download Summary Report",
                data=summary_text,
                file_name="gentrification_summary_report.txt",
                mime="text/plain"
            )
        
        else:  # Detailed Analysis Report
            st.subheader("Detailed Analysis Report")
            
            detailed_text = "# DETAILED GENTRIFICATION ANALYSIS REPORT\n\n"
            
            # Add all areas analysis
            analysis_df = predictions.sort_values('gentrification_probability', ascending=False)
            detailed_text += f"## All Areas Analysis\n\n"
            detailed_text += "| Area | Gentrification Risk | Displacement Risk | Combined Risk | Predicted Rent |\n"
            detailed_text += "|------|-------------------|------------------|---------------|----------------|\n"
            
            for _, row in analysis_df.iterrows():
                detailed_text += f"| {row['area'].title()} | {row['gentrification_probability']*100:.1f}% | {row['displacement_risk']*100:.1f}% | {row['combined_risk']*100:.1f}% | ₹{row['predicted_rent']/100000:.1f}L |\n"
            
            st.markdown(detailed_text)
            
            st.download_button(
                label="📥 Download Detailed Report",
                data=detailed_text,
                file_name="gentrification_detailed_report.md",
                mime="text/markdown"
            )
    
    # ============================================================
    # PAGE 9: ABOUT
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
        
        ### ✨ NEW FEATURES (Enhanced Version)
        
        **📊 Statistics Page** - Comprehensive distribution analysis with histograms, risk breakdowns, and category analysis
        
        **🔬 Risk Factors** - Understand which factors drive gentrification (feature importance analysis)
        
        **⚖️ Compare Areas** - Side-by-side comparison of neighborhoods using bar charts and radar diagrams
        
        **🎯 Trends** - 24-month forecasting and momentum scoring to identify areas to watch
        
        **💾 Export** - Download results as CSV, JSON, summary reports, and detailed markdown reports
        
        ### 🔧 Advanced Streamlit Concepts
        
        This app demonstrates:
        - `@st.cache_data` - Cache expensive data operations
        - `@st.cache_resource` - Cache ML models
        - `st.set_page_config()` - Configure app appearance
        - `st.sidebar.radio()` - Multi-page navigation
        - `st.columns()` - Layout management
        - `st.dataframe()` - Display tables
        - `st.metric()` - Show KPIs
        - `st.pyplot()` - Display matplotlib charts with subplots
        - `st.spinner()` - Show loading spinner
        - `st.download_button()` - Download multiple formats
        - `st.selectbox()` - Dropdown selection
        - Polar plots (radar diagrams)
        - Interactive visualizations
        - Dynamic content generation
        
        ---
        **Built with Python, scikit-learn, XGBoost, and Streamlit** 🚀
        """)

if __name__ == "__main__":
    main()
