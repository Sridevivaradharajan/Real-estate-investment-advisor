"""
Real Estate Investment Predictor - Streamlit Application
Cloud-Ready Deployment with Custom Color Palette
Author: Sridevi V
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Real Estate Investment Predictor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS WITH COOLORS PALETTE
# Color Palette: https://coolors.co/palette/780000-c1121f-fdf0d5-003049-669bbc
# ============================================================================
st.markdown("""
<style>
    /* Color Variables */
    :root {
        --burgundy: #780000;
        --red: #C1121F;
        --cream: #FDF0D5;
        --navy: #003049;
        --blue: #669BBC;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #FDF0D5 0%, #FFFFFF 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003049 0%, #669BBC 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FDF0D5;
    }
    
    /* Headers */
    h1 {
        color: #780000;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #003049;
        font-weight: 700;
    }
    
    h3 {
        color: #669BBC;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #780000;
        font-size: 2rem;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #003049;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #C1121F 0%, #780000 100%);
        color: #FDF0D5;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #780000 0%, #C1121F 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Input Fields */
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: #FFFFFF;
        border: 2px solid #669BBC;
        border-radius: 8px;
    }
    
    /* Cards/Containers */
    .prediction-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FDF0D5 100%);
        border: 3px solid #669BBC;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,48,73,0.2);
    }
    
    .success-card {
        background: linear-gradient(135deg, #669BBC 0%, #003049 100%);
        color: #FDF0D5;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #C1121F 0%, #780000 100%);
        color: #FDF0D5;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #003049;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FDF0D5;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #C1121F;
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #669BBC;
        color: #FDF0D5;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Dataframe */
    .dataframe {
        border: 2px solid #669BBC;
        border-radius: 10px;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #C1121F;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND ARTIFACTS
# ============================================================================
@st.cache_resource
def load_artifacts():
    """Load all model artifacts"""
    try:
        artifacts = {}
        
        # Load models
        with open('best_classification_model.pkl', 'rb') as f:
            artifacts['clf_model'] = pickle.load(f)
        
        with open('best_regression_model.pkl', 'rb') as f:
            artifacts['reg_model'] = pickle.load(f)
        
        # Load features
        with open('classification_features.pkl', 'rb') as f:
            artifacts['clf_features'] = pickle.load(f)
        
        with open('regression_features.pkl', 'rb') as f:
            artifacts['reg_features'] = pickle.load(f)
        
        # Load metadata
        with open('model_metadata.pkl', 'rb') as f:
            artifacts['metadata'] = pickle.load(f)
        
        # Load scalers if they exist
        try:
            with open('classification_scaler.pkl', 'rb') as f:
                artifacts['clf_scaler'] = pickle.load(f)
        except FileNotFoundError:
            artifacts['clf_scaler'] = None
        
        try:
            with open('regression_scaler.pkl', 'rb') as f:
                artifacts['reg_scaler'] = pickle.load(f)
        except FileNotFoundError:
            artifacts['reg_scaler'] = None
        
        return artifacts
    
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please ensure all model files are in the same directory as this app.")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_feature_dict(artifacts):
    """Create a mapping of feature names for user input"""
    feature_mapping = {
        'BHK': {'name': 'Number of Bedrooms', 'type': 'number', 'min': 1, 'max': 5, 'default': 2},
        'Size_in_SqFt': {'name': 'Property Size (Sq Ft)', 'type': 'number', 'min': 500, 'max': 5000, 'default': 1500},
        'Age_of_Property': {'name': 'Age of Property (Years)', 'type': 'number', 'min': 0, 'max': 35, 'default': 10},
        'Floor_No': {'name': 'Floor Number', 'type': 'number', 'min': 0, 'max': 30, 'default': 5},
        'Total_Floors': {'name': 'Total Floors in Building', 'type': 'number', 'min': 1, 'max': 30, 'default': 10},
        'Nearby_Schools': {'name': 'Nearby Schools', 'type': 'number', 'min': 0, 'max': 10, 'default': 3},
        'Nearby_Hospitals': {'name': 'Nearby Hospitals', 'type': 'number', 'min': 0, 'max': 10, 'default': 2},
        'Price_in_Lakhs': {'name': 'Current Price (₹ Lakhs)', 'type': 'number', 'min': 10, 'max': 500, 'default': 100},
    }
    return feature_mapping

def calculate_derived_features(input_data):
    """Calculate derived features from user input"""
    derived = {}
    
    # Infrastructure Score
    transport_map = {'High': 2, 'Medium': 1, 'Low': 0}
    transport_score = transport_map.get(input_data.get('Public_Transport_Accessibility', 'Medium'), 1)
    
    derived['Transport_Score'] = transport_score
    derived['Infrastructure_Score'] = (
        input_data['Nearby_Schools'] * 0.4 + 
        input_data['Nearby_Hospitals'] * 0.3 + 
        transport_score * 0.3
    )
    derived['Total_Infrastructure'] = input_data['Nearby_Schools'] + input_data['Nearby_Hospitals']
    
    # Property metrics
    derived['Size_per_BHK'] = input_data['Size_in_SqFt'] / (input_data['BHK'] + 1)
    derived['Floor_Position_Ratio'] = input_data['Floor_No'] / (input_data['Total_Floors'] + 1)
    derived['School_Density'] = input_data['Nearby_Schools'] / (input_data['Age_of_Property'] + 1)
    derived['Hospital_Density'] = input_data['Nearby_Hospitals'] / (input_data['Age_of_Property'] + 1)
    
    # Amenities (simplified)
    amenities_count = input_data.get('Amenities_Count', 3)
    derived['Amenities_Count'] = amenities_count
    derived['Has_Pool'] = 1 if amenities_count >= 4 else 0
    derived['Has_Gym'] = 1 if amenities_count >= 3 else 0
    derived['Has_Clubhouse'] = 1 if amenities_count >= 3 else 0
    derived['Premium_Amenities'] = derived['Has_Pool'] + derived['Has_Gym'] + derived['Has_Clubhouse']
    
    # Boolean flags
    derived['Has_Parking'] = 1 if input_data.get('Parking_Space') == 'Yes' else 0
    derived['Has_Security'] = 1 if input_data.get('Security') == 'Yes' else 0
    derived['Is_New_Property'] = 1 if input_data['Age_of_Property'] <= 5 else 0
    derived['Is_Mid_Age'] = 1 if 5 < input_data['Age_of_Property'] <= 15 else 0
    derived['Is_Top_Floor'] = 1 if input_data['Floor_No'] == input_data['Total_Floors'] else 0
    derived['Is_Ground_Floor'] = 1 if input_data['Floor_No'] == 0 else 0
    derived['Is_Ready_to_Move'] = 1 if input_data.get('Availability_Status') == 'Ready_to_Move' else 0
    derived['Is_Large_Property'] = 1 if input_data['Size_in_SqFt'] > 2500 else 0
    derived['Is_High_BHK'] = 1 if input_data['BHK'] >= 3 else 0
    
    # Interaction features
    derived['BHK_x_Size'] = input_data['BHK'] * input_data['Size_in_SqFt']
    derived['Age_x_Infrastructure'] = input_data['Age_of_Property'] * derived['Infrastructure_Score']
    derived['BHK_x_Amenities'] = input_data['BHK'] * amenities_count
    
    # Advanced features (use defaults for city/locality averages)
    derived['City_BHK_Level'] = input_data['BHK']  # Simplified
    derived['City_Size_Level'] = input_data['Size_in_SqFt']  # Simplified
    derived['Location_Infrastructure_Quality'] = derived['Infrastructure_Score']
    derived['Property_vs_City_BHK'] = 1.0
    derived['Property_vs_City_Size'] = 1.0
    derived['Locality_Quality'] = derived['Infrastructure_Score']
    derived['Property_vs_Locality_Quality'] = 1.0
    derived['Space_per_Person'] = input_data['Size_in_SqFt'] / ((input_data['BHK'] * 2) + 1)
    derived['Modern_Property_Score'] = (
        derived['Is_New_Property'] * 3 + 
        derived['Premium_Amenities'] * 2 + 
        derived['Has_Security'] * 1 +
        derived['Has_Parking'] * 1
    )
    derived['Is_Premium_Property'] = 1 if (
        input_data['BHK'] >= 3 and 
        derived['Premium_Amenities'] >= 2 and 
        derived['Has_Security'] == 1
    ) else 0
    derived['Is_Budget_Property'] = 1 if (
        input_data['BHK'] <= 2 and 
        derived['Premium_Amenities'] == 0 and 
        input_data['Age_of_Property'] > 15
    ) else 0
    
    # Add log transforms for common features
    for feature in ['Size_in_SqFt', 'Age_of_Property', 'BHK', 'Infrastructure_Score']:
        if feature in input_data:
            derived[f'{feature}_Log'] = np.log1p(input_data[feature])
        elif feature in derived:
            derived[f'{feature}_Log'] = np.log1p(derived[feature])
    
    # Price-related (for regression)
    if 'Price_in_Lakhs' in input_data:
        derived['Price_per_SqFt'] = input_data['Price_in_Lakhs'] / (input_data['Size_in_SqFt'] / 100)
        derived['Price_in_Lakhs_Log'] = np.log1p(input_data['Price_in_Lakhs'])
        derived['Price_per_SqFt_Log'] = np.log1p(derived['Price_per_SqFt'])
    
    # Categorical encoding (use mode/default values)
    derived['City_Encoded'] = 20  # Median city code
    derived['Locality_Encoded'] = 250  # Median locality code
    derived['Amenities_Encoded'] = 160  # Median amenities code
    
    # One-hot encoded features (default values)
    property_types = ['Property_Type_Independent House', 'Property_Type_Villa']
    furnished_status = ['Furnished_Status_Semi-furnished', 'Furnished_Status_Unfurnished']
    facing = ['Facing_North', 'Facing_South', 'Facing_West']
    owner = ['Owner_Type_Builder', 'Owner_Type_Owner']
    
    for feat in property_types + furnished_status + facing + owner:
        derived[feat] = 0
    
    # State encoding (default to 0 for all states)
    states = ['State_Bihar', 'State_Chhattisgarh', 'State_Gujarat', 'State_Haryana', 
              'State_Jharkhand', 'State_Karnataka', 'State_Kerala', 'State_Madhya Pradesh',
              'State_Maharashtra', 'State_Punjab', 'State_Rajasthan', 'State_Tamil Nadu',
              'State_Telangana', 'State_Uttar Pradesh', 'State_West Bengal']
    for state in states:
        derived[state] = 0
    
    return derived

def prepare_features(input_data, required_features):
    """Prepare feature vector for prediction"""
    # Calculate all derived features
    all_features = {**input_data, **calculate_derived_features(input_data)}
    
    # Create feature vector in correct order
    feature_vector = []
    for feature in required_features:
        if feature in all_features:
            feature_vector.append(all_features[feature])
        else:
            # Default value for missing features
            feature_vector.append(0)
    
    return np.array(feature_vector).reshape(1, -1)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Load artifacts
    artifacts = load_artifacts()
    
    if artifacts is None:
        st.stop()
    
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🏘️ Real Estate Investment Predictor</h1>
            <p style='font-size: 1.2rem; color: #003049;'>
                AI-Powered Property Investment Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: #FDF0D5;'>📊 Model Info</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
        <div style='color: #FDF0D5; padding: 1rem;'>
            <h3>Classification Model</h3>
            <p><b>Type:</b> {artifacts['metadata']['classification']['model_name']}</p>
            <p><b>Accuracy:</b> {artifacts['metadata']['classification']['test_accuracy']:.2%}</p>
            
            <h3 style='margin-top: 2rem;'>Regression Model</h3>
            <p><b>Type:</b> {artifacts['metadata']['regression']['model_name']}</p>
            <p><b>R² Score:</b> {artifacts['metadata']['regression']['test_r2']:.4f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style='color: #FDF0D5; padding: 1rem; text-align: center;'>
            <p><b>Developed by:</b> Sridevi V</p>
            <p style='font-size: 0.9rem;'>© 2024 All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Property Analysis", "📈 Batch Prediction", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: SINGLE PROPERTY ANALYSIS
    # ========================================================================
    with tab1:
        st.markdown("<h2>Property Details</h2>", unsafe_allow_html=True)
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏗️ Basic Info")
            bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=5, value=3)
            size = st.number_input("Property Size (Sq Ft)", min_value=500, max_value=5000, value=1500)
            age = st.number_input("Age of Property (Years)", min_value=0, max_value=35, value=10)
            price = st.number_input("Current Price (₹ Lakhs)", min_value=10.0, max_value=500.0, value=150.0)
        
        with col2:
            st.markdown("### 🏢 Building Info")
            floor_no = st.number_input("Floor Number", min_value=0, max_value=30, value=5)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=30, value=10)
            property_type = st.selectbox("Property Type", 
                                        ["Apartment", "Independent House", "Villa"])
            furnished = st.selectbox("Furnished Status", 
                                    ["Furnished", "Semi-furnished", "Unfurnished"])
        
        with col3:
            st.markdown("### 🌆 Location & Amenities")
            schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=3)
            hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=2)
            transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])
            amenities = st.slider("Amenities Count", min_value=0, max_value=5, value=3)
        
        col4, col5 = st.columns(2)
        
        with col4:
            parking = st.selectbox("Parking Space", ["Yes", "No"])
            security = st.selectbox("Security", ["Yes", "No"])
        
        with col5:
            facing = st.selectbox("Facing Direction", ["North", "South", "East", "West"])
            availability = st.selectbox("Availability", 
                                       ["Ready_to_Move", "Under_Construction"])
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Analyze Property", use_container_width=True):
            with st.spinner("Analyzing property..."):
                # Prepare input data
                input_data = {
                    'BHK': bhk,
                    'Size_in_SqFt': size,
                    'Age_of_Property': age,
                    'Price_in_Lakhs': price,
                    'Floor_No': floor_no,
                    'Total_Floors': total_floors,
                    'Nearby_Schools': schools,
                    'Nearby_Hospitals': hospitals,
                    'Public_Transport_Accessibility': transport,
                    'Amenities_Count': amenities,
                    'Parking_Space': parking,
                    'Security': security,
                    'Availability_Status': availability
                }
                
                # Classification prediction
                clf_features_vector = prepare_features(input_data, artifacts['clf_features'])
                
                if artifacts['clf_scaler'] is not None:
                    clf_features_vector = artifacts['clf_scaler'].transform(clf_features_vector)
                
                clf_prediction = artifacts['clf_model'].predict(clf_features_vector)[0]
                clf_proba = artifacts['clf_model'].predict_proba(clf_features_vector)[0]
                
                # Regression prediction
                reg_features_vector = prepare_features(input_data, artifacts['reg_features'])
                
                if artifacts['reg_scaler'] is not None:
                    reg_features_vector = artifacts['reg_scaler'].transform(reg_features_vector)
                
                future_price = artifacts['reg_model'].predict(reg_features_vector)[0]
                appreciation = ((future_price / price) - 1) * 100
                
                # Display results
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{price:.1f}L")
                
                with col2:
                    st.metric("Future Price (5Y)", f"₹{future_price:.1f}L", 
                             delta=f"+₹{future_price-price:.1f}L")
                
                with col3:
                    st.metric("Expected Growth", f"{appreciation:.1f}%")
                
                with col4:
                    quality_label = "GOOD ✅" if clf_prediction == 1 else "RISKY ⚠️"
                    st.metric("Investment Quality", quality_label)
                
                # Detailed cards
                col1, col2 = st.columns(2)
                
                with col1:
                    if clf_prediction == 1:
                        st.markdown(f"""
                        <div class='success-card'>
                            <h3 style='color: #FDF0D5;'>✅ Good Investment Property</h3>
                            <p style='font-size: 1.1rem;'>
                                Confidence: <b>{clf_proba[1]*100:.1f}%</b>
                            </p>
                            <p>This property shows strong investment potential based on:</p>
                            <ul>
                                <li>Location quality and infrastructure</li>
                                <li>Property age and condition</li>
                                <li>Amenities and facilities</li>
                                <li>Market positioning</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='warning-card'>
                            <h3 style='color: #FDF0D5;'>⚠️ Investment Caution Advised</h3>
                            <p style='font-size: 1.1rem;'>
                                Risk Level: <b>{clf_proba[0]*100:.1f}%</b>
                            </p>
                            <p>This property may have limited investment potential due to:</p>
                            <ul>
                                <li>Suboptimal location factors</li>
                                <li>Limited amenities or infrastructure</li>
                                <li>Property age or condition concerns</li>
                                <li>Market positioning challenges</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <h3 style='color: #003049;'>💰 Price Forecast</h3>
                        <p style='font-size: 1.1rem;'>
                            <b>5-Year Projection:</b> ₹{future_price:.2f} Lakhs
                        </p>
                        <p style='font-size: 1.1rem;'>
                            <b>Expected Appreciation:</b> {appreciation:.1f}%
                        </p>
                        <p style='font-size: 1.1rem;'>
                            <b>Absolute Gain:</b> ₹{future_price-price:.2f} Lakhs
                        </p>
                        <p style='font-size: 1.1rem;'>
                            <b>Annual Return:</b> {appreciation/5:.1f}% per year
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h3>📈 Price Projection Chart</h3>", unsafe_allow_html=True)
                
                # Create projection data
                years = np.arange(0, 6)
                annual_rate = (future_price / price) ** (1/5) - 1
                projected_prices = [price * (1 + annual_rate) ** year for year in years]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=projected_prices,
                    mode='lines+markers',
                    name='Projected Price',
                    line=dict(color='#C1121F', width=4),
                    marker=dict(size=12, color='#780000')
                ))
                
                fig.update_layout(
                    title=f"Property Value Projection (₹{price:.1f}L → ₹{future_price:.1f}L)",
                    xaxis_title="Years from Now",
                    yaxis_title="Property Value (₹ Lakhs)",
                    height=400,
                    template="plotly_white",
                    font=dict(size=14, color='#003049')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Investment recommendation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h3>💡 Investment Recommendation</h3>", unsafe_allow_html=True)
                
                if clf_prediction == 1 and appreciation > 40:
                    recommendation = "🌟 **STRONG BUY** - Excellent investment opportunity with high growth potential"
                    rec_color = "success-card"
                elif clf_prediction == 1:
                    recommendation = "✅ **BUY** - Good investment with moderate growth expected"
                    rec_color = "prediction-card"
                elif appreciation > 40:
                    recommendation = "⚠️ **HOLD/CONSIDER** - High appreciation potential but quality concerns"
                    rec_color = "prediction-card"
                else:
                    recommendation = "❌ **AVOID** - Limited investment potential"
                    rec_color = "warning-card"
                
                st.markdown(f"""
                <div class='{rec_color}' style='text-align: center;'>
                    <h2>{recommendation}</h2>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: BATCH PREDICTION
    # ========================================================================
    with tab2:
        st.markdown("<h2>Batch Property Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
            Upload a CSV file with multiple properties for bulk analysis.
            
            **Required columns:** BHK, Size_in_SqFt, Age_of_Property, Price_in_Lakhs, 
            Floor_No, Total_Floors, Nearby_Schools, Nearby_Hospitals
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} properties")
                
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🚀 Analyze All Properties", use_container_width=True):
                    with st.spinner("Processing properties..."):
                        # Add default values for missing columns
                        defaults = {
                            'Public_Transport_Accessibility': 'Medium',
                            'Amenities_Count': 3,
                            'Parking_Space': 'Yes',
                            'Security': 'Yes',
                            'Availability_Status': 'Ready_to_Move'
                        }
                        
                        for col, val in defaults.items():
                            if col not in df.columns:
                                df[col] = val
                        
                        # Predictions
                        clf_predictions = []
                        clf_probas = []
                        future_prices = []
                        
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            # Classification
                            clf_vec = prepare_features(row.to_dict(), artifacts['clf_features'])
                            if artifacts['clf_scaler']:
                                clf_vec = artifacts['clf_scaler'].transform(clf_vec)
                            
                            clf_pred = artifacts['clf_model'].predict(clf_vec)[0]
                            clf_prob = artifacts['clf_model'].predict_proba(clf_vec)[0][1]
                            
                            # Regression
                            reg_vec = prepare_features(row.to_dict(), artifacts['reg_features'])
                            if artifacts['reg_scaler']:
                                reg_vec = artifacts['reg_scaler'].transform(reg_vec)
                            
                            future_price = artifacts['reg_model'].predict(reg_vec)[0]
                            
                            clf_predictions.append(clf_pred)
                            clf_probas.append(clf_prob)
                            future_prices.append(future_price)
                            
                            progress_bar.progress((idx + 1) / len(df))
                        
                        # Add results to dataframe
                        df['Investment_Quality'] = ['Good' if p == 1 else 'Not Good' for p in clf_predictions]
                        df['Confidence_%'] = [p * 100 for p in clf_probas]
                        df['Future_Price_5Y'] = future_prices
                        df['Appreciation_%'] = ((df['Future_Price_5Y'] / df['Price_in_Lakhs']) - 1) * 100
                        
                        st.success("✅ Analysis complete!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            good_investments = sum(clf_predictions)
                            st.metric("Good Investments", f"{good_investments}/{len(df)}")
                        
                        with col2:
                            avg_appreciation = df['Appreciation_%'].mean()
                            st.metric("Avg Appreciation", f"{avg_appreciation:.1f}%")
                        
                        with col3:
                            total_current = df['Price_in_Lakhs'].sum()
                            st.metric("Total Current Value", f"₹{total_current:.1f}L")
                        
                        with col4:
                            total_future = df['Future_Price_5Y'].sum()
                            st.metric("Total Future Value", f"₹{total_future:.1f}L")
                        
                        # Results table
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="property_analysis_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.markdown("### 📊 Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Investment distribution
                            fig = px.pie(
                                values=[good_investments, len(df) - good_investments],
                                names=['Good Investment', 'Not Good'],
                                title='Investment Quality Distribution',
                                color_discrete_sequence=['#669BBC', '#C1121F']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Appreciation distribution
                            fig = px.histogram(
                                df,
                                x='Appreciation_%',
                                title='Expected Appreciation Distribution',
                                color_discrete_sequence=['#780000']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Top opportunities
                        st.markdown("### 🌟 Top 5 Investment Opportunities")
                        top_5 = df.nlargest(5, 'Appreciation_%')
                        st.dataframe(
                            top_5[['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 
                                   'Future_Price_5Y', 'Appreciation_%', 'Investment_Quality']],
                            use_container_width=True
                        )
            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.markdown("<h2>About This Application</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='prediction-card'>
            <h3>🎯 Purpose</h3>
            <p>
                This AI-powered application helps investors make informed decisions about 
                real estate investments by predicting:
            </p>
            <ul>
                <li><b>Investment Quality:</b> Whether a property is a good investment</li>
                <li><b>Future Price:</b> Expected property value after 5 years</li>
                <li><b>Appreciation Rate:</b> Expected growth percentage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='prediction-card'>
            <h3>🤖 Machine Learning Models</h3>
            <p>
                The application uses two complementary models:
            </p>
            <ul>
                <li><b>Classification Model:</b> Predicts investment quality (Good/Not Good)</li>
                <li><b>Regression Model:</b> Forecasts future property price</li>
            </ul>
            <p>
                Both models are trained on 250,000+ real estate transactions with 
                extensive feature engineering including location quality, infrastructure 
                scores, and property characteristics.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='prediction-card'>
            <h3>📊 Key Features</h3>
            <ul>
                <li>Property size, age, and configuration (BHK)</li>
                <li>Location infrastructure (schools, hospitals, transport)</li>
                <li>Building amenities and facilities</li>
                <li>Market positioning and property type</li>
                <li>Historical price trends</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='success-card'>
            <h3>👨‍💻 Developer Information</h3>
            <p><b>Author:</b> Sridevi V</p>
            <p><b>Project Type:</b> Machine Learning Capstone - Classification & Regression</p>
            <p><b>Technology Stack:</b> Python, Scikit-learn, XGBoost, Streamlit, MLflow</p>
            <p><b>Color Palette:</b> <a href='https://coolors.co/palette/780000-c1121f-fdf0d5-003049-669bbc' 
               target='_blank' style='color: #FDF0D5;'>View Palette</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='prediction-card'>
            <h3>⚠️ Disclaimer</h3>
            <p>
                This application provides predictions based on historical data and machine learning models.
                Actual property values may vary due to market conditions, economic factors, and other 
                variables not captured in the model. Always consult with real estate professionals 
                before making investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()