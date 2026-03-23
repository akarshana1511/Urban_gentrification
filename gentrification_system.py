"""
URBAN GENTRIFICATION PREDICTION SYSTEM - COMPLETE IN ONE FILE
Simplified single-file implementation with Flask web interface
Run: python gentrification_system.py
Then open: http://localhost:5000
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

CONFIG = {
    'HOUSE_DATA_FILE': 'Bengaluru_House_Data.csv',
    'RESTAURANT_DATA_FILE': 'Bangalore restaurant chain.csv',
    'METRO_DATA_FILE': 'NammaMetro_Ridership_Dataset.csv',
    'RENT_GROWTH_THRESHOLD': 0.20,  # 20% increase = gentrification
    'DISPLACEMENT_THRESHOLD': 0.40,  # 40% rent-to-income = displacement risk
}

# ============================================================
# SECTION 2: DATA LOADING
# ============================================================

def load_data():
    """Load all raw CSV files"""
    print("\n📂 Loading data files...")
    
    try:
        house_df = pd.read_csv(CONFIG['HOUSE_DATA_FILE'])
        restaurant_df = pd.read_csv(CONFIG['RESTAURANT_DATA_FILE'])
        
        print(f"  ✓ Houses loaded: {len(house_df)} records")
        print(f"  ✓ Restaurants loaded: {len(restaurant_df)} records")
        
        return house_df, restaurant_df
    except FileNotFoundError as e:
        print(f"  ❌ Error: {e}")
        print(f"  Make sure CSV files are in the same folder as this script")
        return None, None

# ============================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================

def preprocess_data(house_df, restaurant_df):
    """Clean and prepare data"""
    print("\n🧹 Preprocessing data...")
    
    # Clean house data
    house_df = house_df.dropna(subset=['location', 'price', 'total_sqft'])
    house_df['price'] = pd.to_numeric(house_df['price'], errors='coerce')
    house_df = house_df[(house_df['price'] >= 10) & (house_df['price'] <= 500)]
    
    # Fill missing values
    house_df['bath'] = house_df['bath'].fillna(house_df['bath'].median())
    house_df['balcony'] = house_df['balcony'].fillna(0)
    
    # Standardize location names
    house_df['location'] = house_df['location'].str.lower().str.strip()
    
    # Clean restaurant data
    restaurant_df = restaurant_df.dropna(subset=['Latitude', 'Longitude', 'Address'])
    restaurant_df['Rating'] = restaurant_df['Rating'].fillna(restaurant_df['Rating'].median())
    
    print(f"  ✓ House data cleaned: {len(house_df)} records")
    print(f"  ✓ Restaurant data cleaned: {len(restaurant_df)} records")
    
    return house_df, restaurant_df

# ============================================================
# SECTION 4: FEATURE ENGINEERING
# ============================================================

def create_features(house_df, restaurant_df):
    """Create ML features by aggregating at area level"""
    print("\n🔬 Creating features...")
    
    # Aggregate house data by location
    house_agg = house_df.groupby('location').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'total_sqft': 'mean',
        'bath': 'mean',
        'balcony': 'mean'
    }).reset_index()
    
    # Flatten column names
    house_agg.columns = ['_'.join(col).strip('_') for col in house_agg.columns.values]
    house_agg.rename(columns={'location': 'area'}, inplace=True)
    
    # Create price-per-sqft feature
    house_agg['price_per_sqft'] = (house_agg['price_mean'] * 10000000) / house_agg['total_sqft_mean']
    
    # Count restaurants per area
    restaurant_count = restaurant_df.groupby(restaurant_df['Address'].str.lower().str.strip()).size().reset_index(name='restaurant_count')
    restaurant_rating = restaurant_df.groupby(restaurant_df['Address'].str.lower().str.strip())['Rating'].mean().reset_index()
    restaurant_rating.columns = ['Address', 'avg_rating']
    
    # Create features for growth potential
    house_agg['growth_potential'] = (house_agg['price_std'] / (house_agg['price_mean'] + 1)).fillna(0)
    house_agg['density_score'] = (house_agg['price_count'] / house_agg['price_count'].max()).fillna(0)
    
    # Create synthetic growth metrics
    house_agg['rent_growth'] = np.random.uniform(0, 0.5, len(house_agg))
    house_agg['business_density'] = np.random.uniform(0, 1, len(house_agg))
    house_agg['transport_access'] = np.random.uniform(0, 1, len(house_agg))
    
    # Create target variables
    house_agg['is_gentrifying'] = (house_agg['rent_growth'] > CONFIG['RENT_GROWTH_THRESHOLD']).astype(int)
    house_agg['displacement_risk'] = (house_agg['rent_growth'] * (1 - house_agg['density_score'])).clip(0, 1)
    house_agg['has_displacement_risk'] = (house_agg['displacement_risk'] > CONFIG['DISPLACEMENT_THRESHOLD']).astype(int)
    
    print(f"  ✓ Features created for {len(house_agg)} areas")
    print(f"  ✓ Gentrifying areas: {house_agg['is_gentrifying'].sum()}")
    print(f"  ✓ Displacement risk areas: {house_agg['has_displacement_risk'].sum()}")
    
    return house_agg

# ============================================================
# SECTION 5: MODEL TRAINING
# ============================================================

class GentrificationPredictor:
    """Main ML model class"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.rent_model = None
        self.gent_model = None
        self.disp_model = None
        self.feature_columns = None
    
    def train(self, features_df):
        """Train all three models"""
        print("\n🤖 Training models...")
        
        # Prepare features
        self.feature_columns = [
            'price_mean', 'price_median', 'price_std', 'total_sqft_mean',
            'bath_mean', 'price_per_sqft', 'growth_potential', 'density_score',
            'rent_growth', 'business_density', 'transport_access'
        ]
        
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_gen_train, y_gen_test = train_test_split(
            X_scaled, features_df['is_gentrifying'],
            test_size=0.2, random_state=42
        )
        
        _, _, y_disp_train, y_disp_test = train_test_split(
            X_scaled, features_df['has_displacement_risk'],
            test_size=0.2, random_state=42
        )
        
        _, _, y_rent_train, y_rent_test = train_test_split(
            X_scaled, features_df['price_mean'],
            test_size=0.2, random_state=42
        )
        
        # Train Gentrification Model (Random Forest)
        self.gent_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.gent_model.fit(X_train, y_gen_train)
        gent_acc = accuracy_score(y_gen_test, self.gent_model.predict(X_test))
        print(f"  ✓ Gentrification Model - Accuracy: {gent_acc:.3f}")
        
        # Train Displacement Model (Logistic Regression)
        self.disp_model = LogisticRegression(max_iter=1000, random_state=42)
        self.disp_model.fit(X_train, y_disp_train)
        disp_acc = accuracy_score(y_disp_test, self.disp_model.predict(X_test))
        print(f"  ✓ Displacement Model - Accuracy: {disp_acc:.3f}")
        
        # Train Rent Prediction Model (XGBoost)
        self.rent_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.rent_model.fit(X_train, y_rent_train)
        rent_r2 = self.rent_model.score(X_test, y_rent_test)
        print(f"  ✓ Rent Model - R² Score: {rent_r2:.3f}")
    
    def predict(self, features_df):
        """Make predictions for all areas"""
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        gent_prob = self.gent_model.predict_proba(X_scaled)[:, 1]
        disp_prob = self.disp_model.predict_proba(X_scaled)[:, 1]
        rent_pred = self.rent_model.predict(X_scaled)
        
        # Create results dataframe
        results = features_df[['area']].copy()
        results['gentrification_probability'] = gent_prob
        results['displacement_risk'] = disp_prob
        results['predicted_rent'] = rent_pred
        results['combined_risk'] = (gent_prob + disp_prob) / 2
        
        # Add risk levels
        results['risk_level'] = results['gentrification_probability'].apply(
            lambda x: '🟢 Low' if x < 0.25 else '🟡 Medium' if x < 0.5 else '🔴 High' if x < 0.8 else '🟣 Very High'
        )
        
        return results
    
    def save(self):
        """Save models to disk"""
        with open('gentrification_model.pkl', 'wb') as f:
            pickle.dump((self.rent_model, self.gent_model, self.disp_model, self.scaler, self.feature_columns), f)
        print("  ✓ Models saved")
    
    def load(self):
        """Load models from disk"""
        try:
            with open('gentrification_model.pkl', 'rb') as f:
                self.rent_model, self.gent_model, self.disp_model, self.scaler, self.feature_columns = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

# ============================================================
# SECTION 6: FLASK WEB APP
# ============================================================

app = Flask(__name__)
predictor = None
results_df = None

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🏘️ Gentrification Predictor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.5em;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label { font-size: 0.9em; opacity: 0.9; }
        
        .search-box {
            margin: 30px 0;
            display: flex;
            gap: 10px;
        }
        .search-box input {
            flex: 1;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
        }
        .search-box button {
            padding: 12px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: 0.3s;
        }
        .search-box button:hover {
            background: #764ba2;
            transform: translateY(-2px);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .area-name {
            font-weight: bold;
            color: #333;
        }
        .probability {
            font-weight: bold;
        }
        .prob-high { color: #e74c3c; }
        .prob-medium { color: #f39c12; }
        .prob-low { color: #2ecc71; }
        
        .detail-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            font-size: 1.1em;
        }
        .detail-label {
            font-weight: bold;
            color: #666;
        }
        .detail-value {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏘️ Urban Gentrification Predictor</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Predicting gentrification risk, displacement vulnerability, and rent trends in Bengaluru neighborhoods
        </p>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Areas Analyzed</div>
                <div class="stat-value" id="total-areas">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">High Risk Areas</div>
                <div class="stat-value" id="high-risk">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Gentrification</div>
                <div class="stat-value" id="avg-gent">0%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Displacement</div>
                <div class="stat-value" id="avg-disp">0%</div>
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="area-search" placeholder="Search area name...">
            <button onclick="searchArea()">🔍 Search</button>
            <button onclick="loadAllAreas()">📊 Show All</button>
        </div>
        
        <div id="detail-view" style="display: none;">
            <div id="detail-content"></div>
        </div>
        
        <div id="table-view">
            <h2 style="color: #333; margin-top: 30px; margin-bottom: 20px;">Top Gentrifying Areas</h2>
            <table id="results-table">
                <thead>
                    <tr>
                        <th>Area</th>
                        <th>Gentrification Risk</th>
                        <th>Displacement Risk</th>
                        <th>Predicted Rent (₹)</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>
    
    <script>
        // Load data on page load
        window.onload = function() {
            loadData();
        };
        
        function loadData() {
            fetch('/data')
                .then(r => r.json())
                .then(data => {
                    displayStats(data);
                    displayTable(data.slice(0, 15));
                });
        }
        
        function displayStats(data) {
            document.getElementById('total-areas').textContent = data.length;
            const highRisk = data.filter(d => d.gentrification_probability > 0.5).length;
            document.getElementById('high-risk').textContent = highRisk;
            const avgGent = (data.reduce((s, d) => s + d.gentrification_probability, 0) / data.length * 100).toFixed(1);
            document.getElementById('avg-gent').textContent = avgGent + '%';
            const avgDisp = (data.reduce((s, d) => s + d.displacement_risk, 0) / data.length * 100).toFixed(1);
            document.getElementById('avg-disp').textContent = avgDisp + '%';
        }
        
        function displayTable(data) {
            const tbody = document.querySelector('#results-table tbody');
            tbody.innerHTML = '';
            data.forEach(row => {
                const tr = document.createElement('tr');
                const gentProb = (row.gentrification_probability * 100).toFixed(1);
                const dispProb = (row.displacement_risk * 100).toFixed(1);
                const risk = row.risk_level;
                const rent = Math.round(row.predicted_rent).toLocaleString();
                
                tr.innerHTML = `
                    <td class="area-name">${row.area}</td>
                    <td class="probability prob-${gentProb > 50 ? 'high' : gentProb > 25 ? 'medium' : 'low'}">${gentProb}%</td>
                    <td class="probability prob-${dispProb > 50 ? 'high' : dispProb > 25 ? 'medium' : 'low'}">${dispProb}%</td>
                    <td>₹${rent}</td>
                    <td>${risk}</td>
                `;
                tr.style.cursor = 'pointer';
                tr.onclick = () => showDetail(row);
                tbody.appendChild(tr);
            });
        }
        
        function showDetail(row) {
            document.getElementById('table-view').style.display = 'none';
            document.getElementById('detail-view').style.display = 'block';
            
            const html = `
                <button onclick="loadAllAreas()" style="padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px;">← Back to List</button>
                <h2 style="color: #333; margin: 20px 0;">${row.area.toUpperCase()}</h2>
                <div class="detail-card">
                    <div class="detail-row">
                        <span class="detail-label">Gentrification Risk:</span>
                        <span class="detail-value probability prob-${row.gentrification_probability * 100 > 50 ? 'high' : row.gentrification_probability * 100 > 25 ? 'medium' : 'low'}">
                            ${(row.gentrification_probability * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Displacement Risk:</span>
                        <span class="detail-value probability prob-${row.displacement_risk * 100 > 50 ? 'high' : row.displacement_risk * 100 > 25 ? 'medium' : 'low'}">
                            ${(row.displacement_risk * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Predicted Rent:</span>
                        <span class="detail-value">₹${Math.round(row.predicted_rent).toLocaleString()}</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Combined Risk Score:</span>
                        <span class="detail-value">${(row.combined_risk * 100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">Risk Level:</span>
                        <span class="detail-value">${row.risk_level}</span>
                    </div>
                    <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="color: #666; font-size: 0.9em; margin-top: 15px;">
                        <strong>💡 Interpretation:</strong> ${getInterpretation(row)}
                    </p>
                </div>
            `;
            
            document.getElementById('detail-content').innerHTML = html;
        }
        
        function getInterpretation(row) {
            const gent = row.gentrification_probability;
            const disp = row.displacement_risk;
            
            if (gent > 0.7) {
                return "This area has VERY HIGH gentrification risk. Expect rapid rent/price increases.";
            } else if (gent > 0.5) {
                return "This area has HIGH gentrification risk. Noticeable rent increases expected.";
            } else if (gent > 0.25) {
                return "This area has MODERATE gentrification risk. Some rent increases likely.";
            } else {
                return "This area has LOW gentrification risk. Prices likely to remain stable.";
            }
        }
        
        function searchArea() {
            const query = document.getElementById('area-search').value.toLowerCase();
            if (query.trim() === '') {
                loadAllAreas();
                return;
            }
            
            fetch('/data')
                .then(r => r.json())
                .then(data => {
                    const filtered = data.filter(d => d.area.toLowerCase().includes(query));
                    if (filtered.length === 0) {
                        alert('No areas found matching: ' + query);
                        return;
                    }
                    if (filtered.length === 1) {
                        showDetail(filtered[0]);
                    } else {
                        displayTable(filtered);
                    }
                });
        }
        
        function loadAllAreas() {
            document.getElementById('table-view').style.display = 'block';
            document.getElementById('detail-view').style.display = 'none';
            loadData();
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/data')
def get_data():
    """Return predictions as JSON"""
    if results_df is None:
        return jsonify([])
    
    # Sort by gentrification probability (highest first)
    sorted_df = results_df.sort_values('gentrification_probability', ascending=False)
    return jsonify(sorted_df.to_dict('records'))

# ============================================================
# SECTION 7: MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    global predictor, results_df
    
    print(("\n" + "="*70))
    print("  🏘️  URBAN GENTRIFICATION PREDICTION SYSTEM")
    print("="*70)
    
    # Load data
    house_df, restaurant_df = load_data()
    if house_df is None:
        return
    
    # Preprocess
    house_df, restaurant_df = preprocess_data(house_df, restaurant_df)
    
    # Engineer features
    features_df = create_features(house_df, restaurant_df)
    
    # Train models
    predictor = GentrificationPredictor()
    predictor.train(features_df)
    
    # Make predictions
    print("\n🔮 Making predictions...")
    results_df = predictor.predict(features_df)
    predictor.save()
    
    # Print top areas
    print("\n🎯 TOP GENTRIFYING AREAS:")
    top_gent = results_df.nlargest(5, 'gentrification_probability')[['area', 'gentrification_probability']]
    for idx, row in top_gent.iterrows():
        print(f"  {row['area']:30s} | {row['gentrification_probability']*100:5.1f}%")
    
    print("\n⚠️  TOP DISPLACEMENT RISK AREAS:")
    top_disp = results_df.nlargest(5, 'displacement_risk')[['area', 'displacement_risk']]
    for idx, row in top_disp.iterrows():
        print(f"  {row['area']:30s} | {row['displacement_risk']*100:5.1f}%")
    
    # Launch web app
    print("\n" + "="*70)
    print("\n🚀 LAUNCHING WEB APP...")
    print("\n📱 Open your browser and go to:\n   → http://localhost:5000\n")
    print("Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=False, host='127.0.0.1', port=5000)

if __name__ == '__main__':
    main()
