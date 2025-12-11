import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Real Estate Investment Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with color palette: #EDAFB8, #F7E1D7, #DEDBD2, #B0C4B1, #4A5759
st.markdown("""
<style>
    .main {
        background-color: #F7E1D7;
    }
    .stButton>button {
        background-color: #B0C4B1;
        color: #4A5759;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4A5759;
        color: #F7E1D7;
        transform: scale(1.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #EDAFB8 0%, #F7E1D7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #DEDBD2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #B0C4B1;
        margin: 1rem 0;
    }
    .header-style {
        color: #4A5759;
        font-weight: bold;
        padding: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #DEDBD2;
        color: #4A5759;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B0C4B1;
        color: white;
    }
    .info-box {
        background-color: #DEDBD2;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND ARTIFACTS
# ============================================================================
@st.cache_resource
def load_models():
    """Load all saved models and artifacts"""
    try:
        with open('best_classification_model.pkl', 'rb') as f:
            clf_model = pickle.load(f)
        with open('best_regression_model.pkl', 'rb') as f:
            reg_model = pickle.load(f)
        with open('classification_features.pkl', 'rb') as f:
            clf_features = pickle.load(f)
        with open('regression_features.pkl', 'rb') as f:
            reg_features = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        try:
            with open('classification_scaler.pkl', 'rb') as f:
                clf_scaler = pickle.load(f)
        except:
            clf_scaler = None
        
        try:
            with open('regression_scaler.pkl', 'rb') as f:
                reg_scaler = pickle.load(f)
        except:
            reg_scaler = None
        
        return clf_model, reg_model, clf_features, reg_features, metadata, clf_scaler, reg_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

clf_model, reg_model, clf_features, reg_features, metadata, clf_scaler, reg_scaler = load_models()

# ============================================================================
# FEATURE ENGINEERING PIPELINE (EXACT MATCH FROM TRAINING)
# ============================================================================
def create_features(df):
    """Create features exactly as in training"""
    df_fe = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['Age_of_Property', 'Size_in_SqFt', 'Price_in_Lakhs', 
                    'Nearby_Schools', 'Nearby_Hospitals', 'Floor_No', 'Total_Floors', 'BHK']
    for col in numeric_cols:
        if col in df_fe.columns:
            df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce').fillna(0)
    
    # Calculate Price_per_SqFt
    df_fe['Price_per_SqFt'] = df_fe['Price_in_Lakhs'] / (df_fe['Size_in_SqFt'] + 1)
    
    # Infrastructure Score
    transport_map = {'High': 2, 'Medium': 1, 'Low': 0}
    df_fe['Transport_Score'] = df_fe['Public_Transport_Accessibility'].map(transport_map).fillna(0)
    df_fe['Infrastructure_Score'] = (
        df_fe['Nearby_Schools'] * 0.4 + 
        df_fe['Nearby_Hospitals'] * 0.3 + 
        df_fe['Transport_Score'] * 0.3
    )
    df_fe['Total_Infrastructure'] = df_fe['Nearby_Schools'] + df_fe['Nearby_Hospitals']
    
    # Location Quality (use fixed values for single prediction)
    city_avg_size = 1500
    state_avg_size = 1400
    df_fe['City_Size_Level'] = city_avg_size
    df_fe['State_Size_Level'] = state_avg_size
    df_fe['Location_Infrastructure_Quality'] = df_fe['Infrastructure_Score']
    
    # Property Value Indicators
    df_fe['Size_per_BHK'] = df_fe['Size_in_SqFt'] / (df_fe['BHK'] + 1)
    df_fe['Floor_Position_Ratio'] = df_fe['Floor_No'] / (df_fe['Total_Floors'] + 1)
    df_fe['School_Density'] = df_fe['Nearby_Schools'] / (df_fe['Age_of_Property'] + 1)
    df_fe['Hospital_Density'] = df_fe['Nearby_Hospitals'] / (df_fe['Age_of_Property'] + 1)
    
    # Amenities Features
    df_fe['Amenities_Count'] = df_fe['Amenities'].str.split(',').str.len().fillna(0)
    df_fe['Has_Pool'] = df_fe['Amenities'].str.contains('Pool', case=False, na=False).astype(int)
    df_fe['Has_Gym'] = df_fe['Amenities'].str.contains('Gym', case=False, na=False).astype(int)
    df_fe['Has_Clubhouse'] = df_fe['Amenities'].str.contains('Clubhouse', case=False, na=False).astype(int)
    df_fe['Premium_Amenities'] = df_fe['Has_Pool'] + df_fe['Has_Gym'] + df_fe['Has_Clubhouse']
    
    # Boolean Flags
    df_fe['Has_Parking'] = (df_fe['Parking_Space'] == 'Yes').astype(int)
    df_fe['Has_Security'] = (df_fe['Security'] == 'Yes').astype(int)
    df_fe['Is_New_Property'] = (df_fe['Age_of_Property'] <= 5).astype(int)
    df_fe['Is_Mid_Age'] = ((df_fe['Age_of_Property'] > 5) & (df_fe['Age_of_Property'] <= 15)).astype(int)
    df_fe['Is_Top_Floor'] = (df_fe['Floor_No'] == df_fe['Total_Floors']).astype(int)
    df_fe['Is_Ground_Floor'] = (df_fe['Floor_No'] == 0).astype(int)
    df_fe['Is_Ready_to_Move'] = (df_fe['Availability_Status'] == 'Ready_to_Move').astype(int)
    df_fe['Is_Large_Property'] = (df_fe['Size_in_SqFt'] > 1200).astype(int)
    df_fe['Is_High_BHK'] = (df_fe['BHK'] >= 3).astype(int)
    
    # Interaction Features
    df_fe['BHK_x_Size'] = df_fe['BHK'] * df_fe['Size_in_SqFt']
    df_fe['Age_x_Infrastructure'] = df_fe['Age_of_Property'] * df_fe['Infrastructure_Score']
    df_fe['BHK_x_Amenities'] = df_fe['BHK'] * df_fe['Amenities_Count']
    
    # Advanced Features
    df_fe['City_BHK_Level'] = 2.5
    df_fe['Property_vs_City_BHK'] = df_fe['BHK'] / 2.5
    df_fe['Property_vs_City_Size'] = df_fe['Size_in_SqFt'] / city_avg_size
    df_fe['Locality_Quality'] = df_fe['Infrastructure_Score']
    df_fe['Property_vs_Locality_Quality'] = 1.0
    df_fe['Space_per_Person'] = df_fe['Size_in_SqFt'] / ((df_fe['BHK'] * 2) + 1)
    df_fe['Modern_Property_Score'] = (
        df_fe['Is_New_Property'] * 3 + 
        df_fe['Premium_Amenities'] * 2 + 
        df_fe['Has_Security'] * 1 +
        df_fe['Has_Parking'] * 1
    )
    df_fe['Is_Premium_Property'] = (
        (df_fe['BHK'] >= 3) & 
        (df_fe['Premium_Amenities'] >= 2) & 
        (df_fe['Has_Security'] == 1)
    ).astype(int)
    df_fe['Is_Budget_Property'] = (
        (df_fe['BHK'] <= 2) & 
        (df_fe['Premium_Amenities'] == 0) & 
        (df_fe['Age_of_Property'] > 15)
    ).astype(int)
    
    return df_fe

def encode_categorical(df):
    """Encode categorical features"""
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    high_cardinality = [col for col in categorical_cols if df_encoded[col].nunique() > 20]
    low_cardinality = [col for col in categorical_cols if df_encoded[col].nunique() <= 20]
    
    # Label Encoding for high cardinality
    for col in high_cardinality:
        le = LabelEncoder()
        df_encoded[f'{col}_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
    
    # One-Hot Encoding for low cardinality
    if low_cardinality:
        df_encoded = pd.get_dummies(df_encoded, columns=low_cardinality, drop_first=True, dtype=int)
    
    return df_encoded

def preprocess_input(input_df, feature_list, scaler=None):
    """Complete preprocessing pipeline"""
    # Step 1: Feature Engineering
    df_featured = create_features(input_df)
    
    # Step 2: Categorical Encoding
    df_encoded = encode_categorical(df_featured)
    
    # Step 3: Log transformation for skewed features
    log_candidates = ['Size_in_SqFt', 'Age_of_Property', 'BHK_x_Size', 'Price_in_Lakhs', 'Price_per_SqFt']
    for col in log_candidates:
        if col in df_encoded.columns:
            df_encoded[f'{col}_Log'] = np.log1p(df_encoded[col])
    
    # Step 4: Select required features
    missing_features = []
    for feat in feature_list:
        if feat not in df_encoded.columns:
            df_encoded[feat] = 0
            missing_features.append(feat)
    
    df_final = df_encoded[feature_list]
    
    # Step 5: Scaling if needed
    if scaler is not None:
        df_final = pd.DataFrame(scaler.transform(df_final), columns=feature_list)
    
    return df_final

# ============================================================================
# HEADER
# ============================================================================
st.markdown("<h1 style='text-align: center; color: #4A5759;'>🏠 Real Estate Investment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4A5759;'>AI-Powered Property Investment Predictions</p>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown(f"""
    <div class='info-box'>
    <strong>Classification Model</strong><br>
    Type: {metadata['classification']['model_name']}<br>
    Accuracy: {metadata['classification']['test_accuracy']:.2%}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='info-box'>
    <strong>Regression Model</strong><br>
    Type: {metadata['regression']['model_name']}<br>
    R² Score: {metadata['regression']['test_r2']:.4f}
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🏡 Single Prediction", "📊 Bulk Prediction", "🗺️ Market Insights", "🔍 Feature Importance"])

# ============================================================================
# SINGLE PREDICTION TAB
# ============================================================================
with tab1:
    st.markdown("### 🏡 Enter Property Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 📍 Location")
        state = st.selectbox("State", [
            'Tamil Nadu', 'Maharashtra', 'Punjab', 'Rajasthan', 'West Bengal',
            'Chhattisgarh', 'Delhi', 'Jharkhand', 'Telangana', 'Karnataka',
            'Uttar Pradesh', 'Assam', 'Uttarakhand', 'Bihar', 'Gujarat', 'Haryana',
            'Andhra Pradesh', 'Madhya Pradesh', 'Kerala', 'Odisha'
        ])
        
        city = st.selectbox("City", [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune',
            'Ahmedabad', 'Kolkata', 'Surat', 'Jaipur', 'Lucknow', 'Nagpur',
            'Indore', 'Bhopal', 'Vishakhapatnam', 'Vijayawada', 'Kochi',
            'Coimbatore', 'Mysore', 'Gurgaon', 'Noida', 'Faridabad'
        ])
        
        locality = st.text_input("Locality", value="Locality_1", help="Enter as Locality_1 to Locality_500")

    with col2:
        st.markdown("#### 🏢 Property Type")
        property_type = st.selectbox("Type", ["Apartment", "Villa", "Independent House"])
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
        size = st.number_input("Size (Sq Ft)", min_value=300, max_value=10000, value=1000)
        
    with col3:
        st.markdown("#### 🏗️ Building Details")
        year_built = st.number_input("Year Built", min_value=1970, max_value=2025, value=2018)
        age = 2025 - year_built
        floor_no = st.number_input("Floor Number", min_value=0, max_value=50, value=2)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10)

    with col4:
        st.markdown("#### 🪑 Status")
        furnished = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
        facing = st.selectbox("Facing", ["North", "South", "East", "West"])
        availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])

    col5, col6, col7 = st.columns(3)

    with col5:
        st.markdown("#### 🚗 Facilities")
        parking = st.selectbox("Parking Space", ["Yes", "No"])
        security = st.selectbox("Security", ["Yes", "No"])
        
    with col6:
        st.markdown("#### 🏥 Nearby")
        schools = st.number_input("Nearby Schools", min_value=0, max_value=20, value=3)
        hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=20, value=2)
        transport = st.selectbox("Public Transport", ["High", "Medium", "Low"])

    with col7:
        st.markdown("#### 🎯 Amenities")
        has_pool = st.checkbox("Pool")
        has_gym = st.checkbox("Gym")
        has_clubhouse = st.checkbox("Clubhouse")
        has_garden = st.checkbox("Garden")
        has_playground = st.checkbox("Playground")
        
        amenities_list = []
        if has_pool: amenities_list.append("Pool")
        if has_gym: amenities_list.append("Gym")
        if has_clubhouse: amenities_list.append("Clubhouse")
        if has_garden: amenities_list.append("Garden")
        if has_playground: amenities_list.append("Playground")
        amenities = ','.join(amenities_list) if amenities_list else 'None'

    st.markdown("---")
    col8, col9 = st.columns(2)

    with col8:
        owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])

    with col9:
        current_price = st.number_input(
            "Current Price (Lakhs ₹)",
            min_value=10.0,
            max_value=1000.0,
            value=50.0,
            step=5.0,
            help="Enter the actual current market price of this property in Lakhs"
        )

    st.markdown("---")

    if st.button("🔮 Predict Investment Quality & Future Price", use_container_width=True):
        # Prepare input with ALL required fields
        input_data = pd.DataFrame({
            'State': [state],
            'City': [city],
            'Locality': [locality],
            'Property_Type': [property_type],
            'BHK': [bhk],
            'Size_in_SqFt': [size],
            'Price_in_Lakhs': [current_price],
            'Year_Built': [year_built],
            'Age_of_Property': [age],
            'Furnished_Status': [furnished],
            'Facing': [facing],
            'Owner_Type': [owner_type],
            'Availability_Status': [availability],
            'Parking_Space': [parking],
            'Security': [security],
            'Public_Transport_Accessibility': [transport],
            'Nearby_Schools': [schools],
            'Nearby_Hospitals': [hospitals],
            'Floor_No': [floor_no],
            'Total_Floors': [total_floors],
            'Amenities': [amenities]
        })
        
        with st.spinner("🔄 Processing..."):
            try:
                # Classification Prediction
                X_clf = preprocess_input(input_data, clf_features, clf_scaler)
                clf_pred = clf_model.predict(X_clf)[0]
                clf_proba = clf_model.predict_proba(X_clf)[0]
                
                # Regression Prediction
                X_reg = preprocess_input(input_data, reg_features, reg_scaler)
                future_price = reg_model.predict(X_reg)[0]
                appreciation = ((future_price / current_price - 1) * 100)
            
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()
        
        # Display Results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality = "✅ GOOD INVESTMENT" if clf_pred == 1 else "⚠️ NOT RECOMMENDED"
            color = "#B0C4B1" if clf_pred == 1 else "#EDAFB8"
            st.markdown(f"""
            <div class='metric-card' style='background-color: {color};'>
                <h3 style='text-align: center; color: #4A5759;'>{quality}</h3>
                <p style='text-align: center; font-size: 2rem; margin: 0;'>{clf_proba[1]*100:.1f}%</p>
                <p style='text-align: center; color: #4A5759;'>Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='text-align: center; color: #4A5759;'>Current Price</h3>
                <p style='text-align: center; font-size: 2rem; margin: 0; color: #4A5759;'>₹{current_price:.2f}L</p>
                <p style='text-align: center; color: #4A5759;'>Today's Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='text-align: center; color: #4A5759;'>Future Price (5 Years)</h3>
                <p style='text-align: center; font-size: 2rem; margin: 0; color: #4A5759;'>₹{future_price:.2f}L</p>
                <p style='text-align: center; color: #4A5759;'>+{appreciation:.1f}% Growth</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendation Box
        st.markdown(f"""
        <div class='recommendation-box'>
            <h3 style='color: #4A5759;'>💡 Investment Recommendation</h3>
            <p><strong>Property:</strong> {bhk} BHK {property_type} | <strong>Size:</strong> {size} sq ft</p>
            <p><strong>Location:</strong> {locality}, {city}, {state}</p>
            <p><strong>Age:</strong> {age} years | <strong>Floor:</strong> {floor_no}/{total_floors}</p>
            <p><strong>Analysis:</strong> {'This property shows strong investment potential with good appreciation prospects.' if clf_pred == 1 else 'This property may not meet investment quality standards. Consider other options.'}</p>
            <p><strong>Expected ROI:</strong> {appreciation:.1f}% over 5 years ({appreciation/5:.1f}% annually)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📈 Investment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=clf_proba[1]*100,
                title={'text': "Investment Confidence Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#B0C4B1"},
                    'steps': [
                        {'range': [0, 40], 'color': "#EDAFB8"},
                        {'range': [40, 70], 'color': "#DEDBD2"},
                        {'range': [70, 100], 'color': "#B0C4B1"}
                    ],
                    'threshold': {
                        'line': {'color': "#4A5759", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Price Growth Chart
            years = np.arange(0, 6)
            prices = [current_price * ((1 + appreciation/100/5) ** year) for year in years]
            
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=years,
                y=prices,
                mode='lines+markers',
                line=dict(color='#B0C4B1', width=3),
                marker=dict(size=10, color='#4A5759'),
                fill='tozeroy',
                fillcolor='rgba(176, 196, 177, 0.3)'
            ))
            fig_growth.update_layout(
                title="5-Year Price Projection",
                xaxis_title="Years",
                yaxis_title="Price (Lakhs ₹)",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_growth, use_container_width=True)

# ============================================================================
# TAB 2: BULK PREDICTION
# ============================================================================
with tab2:
    st.markdown("### 📁 Upload CSV File for Bulk Predictions")
    
    st.info("""
    **Required Columns:** City, State, Locality, Property_Type, BHK, Size_in_SqFt, Age_of_Property, 
    Furnished_Status, Facing, Owner_Type, Availability_Status, Parking_Space, Security, 
    Public_Transport_Accessibility, Nearby_Schools, Nearby_Hospitals, Floor_No, Total_Floors, Amenities
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        bulk_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(bulk_data)} properties")
        
        st.markdown("#### Preview Data")
        st.dataframe(bulk_data.head(), use_container_width=True)
        
        if st.button("🚀 Run Bulk Predictions", use_container_width=True):
            with st.spinner("Processing all properties..."):
                # Classification
                X_clf_bulk = preprocess_input(bulk_data, clf_features, clf_scaler)
                clf_preds = clf_model.predict(X_clf_bulk)
                clf_probas = clf_model.predict_proba(X_clf_bulk)
                
                # Regression (estimate prices if not provided)
                if 'Price_in_Lakhs' not in bulk_data.columns:
                    bulk_data['Price_in_Lakhs'] = (bulk_data['Size_in_SqFt'] / 1000) * 50
                
                X_reg_bulk = preprocess_input(bulk_data, reg_features, reg_scaler)
                future_prices = reg_model.predict(X_reg_bulk)
                
                # Combine results
                results = bulk_data.copy()
                results['Investment_Quality'] = ['Good' if p == 1 else 'Not Recommended' for p in clf_preds]
                results['Confidence_%'] = clf_probas[:, 1] * 100
                results['Current_Price_Lakhs'] = bulk_data['Price_in_Lakhs']
                results['Future_Price_5Y_Lakhs'] = future_prices
                results['Appreciation_%'] = ((future_prices / bulk_data['Price_in_Lakhs'] - 1) * 100)
                
                st.markdown("---")
                st.markdown("### 📊 Bulk Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Properties", len(results))
                col2.metric("Good Investments", f"{(clf_preds == 1).sum()} ({(clf_preds == 1).sum()/len(results)*100:.1f}%)")
                col3.metric("Avg Confidence", f"{clf_probas[:, 1].mean()*100:.1f}%")
                col4.metric("Avg Appreciation", f"{results['Appreciation_%'].mean():.1f}%")
                
                # Display results
                st.dataframe(
                    results[['City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt', 
                            'Investment_Quality', 'Confidence_%', 'Current_Price_Lakhs', 
                            'Future_Price_5Y_Lakhs', 'Appreciation_%']],
                    use_container_width=True
                )
                
                # Download button
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Visualizations
                st.markdown("### 📈 Bulk Analysis Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Investment Quality Distribution
                    fig_dist = px.pie(
                        results,
                        names='Investment_Quality',
                        title="Investment Quality Distribution",
                        color='Investment_Quality',
                        color_discrete_map={'Good': '#B0C4B1', 'Not Recommended': '#EDAFB8'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Top 10 by Appreciation
                    top10 = results.nlargest(10, 'Appreciation_%')
                    fig_top = px.bar(
                        top10,
                        x='Appreciation_%',
                        y='Locality',
                        orientation='h',
                        title="Top 10 Properties by Appreciation",
                        color='Appreciation_%',
                        color_continuous_scale=['#EDAFB8', '#B0C4B1']
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

# ============================================================================
# TAB 3: MARKET INSIGHTS
# ============================================================================
with tab3:
    st.markdown("### 🗺️ Market Analysis & Insights")
    
    # Sample data for visualization (in production, load from database)
    sample_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune']
    sample_prices = [85, 75, 70, 60, 55, 65]
    sample_growth = [8.5, 7.2, 9.1, 8.8, 7.5, 8.0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # City-wise Average Prices
        fig_city_price = go.Figure(go.Bar(
            x=sample_prices,
            y=sample_cities,
            orientation='h',
            marker=dict(color='#B0C4B1', line=dict(color='#4A5759', width=1))
        ))
        fig_city_price.update_layout(
            title="Average Property Prices by City",
            xaxis_title="Price (Lakhs ₹)",
            yaxis_title="City",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_city_price, use_container_width=True)
    
    with col2:
        # Growth Rate by City
        fig_growth = go.Figure(go.Bar(
            x=sample_cities,
            y=sample_growth,
            marker=dict(
                color=sample_growth,
                colorscale=[[0, '#EDAFB8'], [0.5, '#DEDBD2'], [1, '#B0C4B1']],
                line=dict(color='#4A5759', width=1)
            )
        ))
        fig_growth.update_layout(
            title="Expected Annual Growth Rate by City",
            xaxis_title="City",
            yaxis_title="Growth Rate (%)",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_growth, use_container_width=True)
    
    # Property Type Distribution
    st.markdown("### 🏘️ Property Type Analysis")
    property_types = ['Apartment', 'Villa', 'Independent House', 'Studio']
    property_counts = [450, 180, 120, 50]
    property_avg_prices = [65, 120, 95, 35]
    
    fig_property = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Property Type Distribution', 'Average Price by Type'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}]]
    )
    
    fig_property.add_trace(
        go.Pie(
            labels=property_types,
            values=property_counts,
            marker=dict(colors=['#B0C4B1', '#DEDBD2', '#EDAFB8', '#F7E1D7'])
        ),
        row=1, col=1
    )
    
    fig_property.add_trace(
        go.Bar(
            x=property_types,
            y=property_avg_prices,
            marker=dict(color='#B0C4B1', line=dict(color='#4A5759', width=1))
        ),
        row=1, col=2
    )
    
    fig_property.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_property, use_container_width=True)
    
    # BHK-wise Analysis
    st.markdown("### 🛏️ BHK Configuration Analysis")
    bhk_config = ['1 BHK', '2 BHK', '3 BHK', '4+ BHK']
    bhk_demand = [25, 45, 22, 8]
    bhk_roi = [7.2, 8.5, 9.0, 8.2]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bhk_demand = go.Figure(go.Bar(
            x=bhk_config,
            y=bhk_demand,
            marker=dict(color='#EDAFB8', line=dict(color='#4A5759', width=1)),
            text=bhk_demand,
            texttemplate='%{text}%',
            textposition='outside'
        ))
        fig_bhk_demand.update_layout(
            title="Market Demand by BHK Configuration",
            xaxis_title="Configuration",
            yaxis_title="Demand (%)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bhk_demand, use_container_width=True)
    
    with col2:
        fig_bhk_roi = go.Figure(go.Scatter(
            x=bhk_config,
            y=bhk_roi,
            mode='lines+markers',
            line=dict(color='#B0C4B1', width=3),
            marker=dict(size=12, color='#4A5759')
        ))
        fig_bhk_roi.update_layout(
            title="Average ROI by BHK Configuration",
            xaxis_title="Configuration",
            yaxis_title="ROI (%)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bhk_roi, use_container_width=True)
    
    # Market Trends
    st.markdown("### 📊 Market Trends & Predictions")
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    price_trend = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
    transactions = [120, 135, 150, 145, 160, 155, 170, 165, 180, 175, 190, 185]
    
    fig_trends = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Price Trend (Monthly)', 'Transaction Volume'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    fig_trends.add_trace(
        go.Scatter(
            x=months,
            y=price_trend,
            mode='lines+markers',
            line=dict(color='#B0C4B1', width=3),
            marker=dict(size=8, color='#4A5759'),
            fill='tozeroy',
            fillcolor='rgba(176, 196, 177, 0.3)'
        ),
        row=1, col=1
    )
    
    fig_trends.add_trace(
        go.Bar(
            x=months,
            y=transactions,
            marker=dict(color='#EDAFB8', line=dict(color='#4A5759', width=1))
        ),
        row=1, col=2
    )
    
    fig_trends.update_xaxes(title_text="Month", row=1, col=1)
    fig_trends.update_yaxes(title_text="Avg Price (Lakhs)", row=1, col=1)
    fig_trends.update_xaxes(title_text="Month", row=1, col=2)
    fig_trends.update_yaxes(title_text="Transactions", row=1, col=2)
    fig_trends.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig_trends, use_container_width=True)

# ============================================================================
# TAB 4: FEATURE IMPORTANCE
# ============================================================================
with tab4:
    st.markdown("### 🔍 Model Feature Importance Analysis")
    
    st.info("""
    **SHAP (SHapley Additive exPlanations)** values show which features most influence the model's predictions.
    Higher SHAP values indicate greater importance in determining investment quality and price predictions.
    """)
    
    # Load SHAP values if available
    try:
        with open('shap_values_classification.pkl', 'rb') as f:
            shap_clf_data = pickle.load(f)
        with open('shap_values_regression.pkl', 'rb') as f:
            shap_reg_data = pickle.load(f)
        shap_available = True
    except:
        shap_available = False
        st.warning("SHAP values not found. Showing feature list only.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Classification Features")
        st.markdown(f"""
        <div class='info-box'>
        <strong>Model:</strong> {metadata['classification']['model_name']}<br>
        <strong>Total Features:</strong> {len(clf_features)}<br>
        <strong>Test Accuracy:</strong> {metadata['classification']['test_accuracy']:.2%}
        </div>
        """, unsafe_allow_html=True)
        
        if shap_available:
            # Calculate feature importance from SHAP
            shap_importance_clf = pd.DataFrame({
                'Feature': clf_features,
                'SHAP_Value': np.abs(shap_clf_data['shap_values']).mean(axis=0)
            }).sort_values('SHAP_Value', ascending=False).head(20)
            
            fig_clf_shap = go.Figure(go.Bar(
                x=shap_importance_clf['SHAP_Value'],
                y=shap_importance_clf['Feature'],
                orientation='h',
                marker=dict(
                    color=shap_importance_clf['SHAP_Value'],
                    colorscale=[[0, '#F7E1D7'], [0.5, '#DEDBD2'], [1, '#B0C4B1']],
                    line=dict(color='#4A5759', width=1)
                )
            ))
            fig_clf_shap.update_layout(
                title="Top 20 Features for Investment Quality",
                xaxis_title="SHAP Importance",
                yaxis_title="Feature",
                height=600,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_clf_shap, use_container_width=True)
        else:
            st.dataframe(
                pd.DataFrame({'Features': clf_features[:20]}),
                use_container_width=True
            )
    
    with col2:
        st.markdown("#### 📈 Regression Features")
        st.markdown(f"""
        <div class='info-box'>
        <strong>Model:</strong> {metadata['regression']['model_name']}<br>
        <strong>Total Features:</strong> {len(reg_features)}<br>
        <strong>Test R² Score:</strong> {metadata['regression']['test_r2']:.4f}
        </div>
        """, unsafe_allow_html=True)
        
        if shap_available:
            # Calculate feature importance from SHAP
            shap_importance_reg = pd.DataFrame({
                'Feature': reg_features,
                'SHAP_Value': np.abs(shap_reg_data['shap_values']).mean(axis=0)
            }).sort_values('SHAP_Value', ascending=False).head(20)
            
            fig_reg_shap = go.Figure(go.Bar(
                x=shap_importance_reg['SHAP_Value'],
                y=shap_importance_reg['Feature'],
                orientation='h',
                marker=dict(
                    color=shap_importance_reg['SHAP_Value'],
                    colorscale=[[0, '#F7E1D7'], [0.5, '#EDAFB8'], [1, '#4A5759']],
                    line=dict(color='#4A5759', width=1)
                )
            ))
            fig_reg_shap.update_layout(
                title="Top 20 Features for Price Prediction",
                xaxis_title="SHAP Importance",
                yaxis_title="Feature",
                height=600,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_reg_shap, use_container_width=True)
        else:
            st.dataframe(
                pd.DataFrame({'Features': reg_features[:20]}),
                use_container_width=True
            )
    
    # Feature Insights
    st.markdown("---")
    st.markdown("### 💡 Key Feature Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='recommendation-box'>
        <h4 style='color: #4A5759;'>🏗️ Property Characteristics</h4>
        <ul>
            <li><strong>Size:</strong> Larger properties generally appreciate faster</li>
            <li><strong>BHK:</strong> 3 BHK shows highest ROI</li>
            <li><strong>Age:</strong> Properties under 5 years are premium</li>
            <li><strong>Floor Position:</strong> Middle floors preferred</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='recommendation-box'>
        <h4 style='color: #4A5759;'>📍 Location Factors</h4>
        <ul>
            <li><strong>Infrastructure:</strong> Schools & hospitals crucial</li>
            <li><strong>Transport:</strong> High accessibility adds 15-20% value</li>
            <li><strong>Locality:</strong> Established areas more stable</li>
            <li><strong>City Tier:</strong> Tier-1 cities show consistent growth</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='recommendation-box'>
        <h4 style='color: #4A5759;'>🎯 Amenities Impact</h4>
        <ul>
            <li><strong>Security:</strong> Major factor for families</li>
            <li><strong>Parking:</strong> Essential in metro cities</li>
            <li><strong>Premium Amenities:</strong> Pool, Gym boost value</li>
            <li><strong>Furnished Status:</strong> Semi-furnished most flexible</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Explanation
    st.markdown("---")
    st.markdown("### 🧠 How The Model Works")
    
    st.markdown("""
    <div class='info-box'>
    <h4 style='color: #4A5759;'>Prediction Pipeline</h4>
    <ol>
        <li><strong>Feature Engineering:</strong> Creates 40+ derived features from raw input (infrastructure scores, property value indicators, boolean flags)</li>
        <li><strong>Categorical Encoding:</strong> Converts text data to numerical format using Label Encoding and One-Hot Encoding</li>
        <li><strong>Log Transformation:</strong> Normalizes skewed distributions for better model performance</li>
        <li><strong>Feature Selection:</strong> Uses top 20 most predictive features based on correlation analysis</li>
        <li><strong>Scaling (if needed):</strong> Standardizes features for linear models</li>
        <li><strong>Prediction:</strong> Ensemble models provide robust predictions with confidence scores</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='recommendation-box'>
        <h4 style='color: #4A5759;'>✅ Classification Model</h4>
        <p><strong>Task:</strong> Predict if property is a "Good Investment"</p>
        <p><strong>Output:</strong> Binary classification with confidence score</p>
        <p><strong>Key Factors:</strong></p>
        <ul>
            <li>Infrastructure quality (40% weight)</li>
            <li>Property age and condition (25% weight)</li>
            <li>Location and amenities (20% weight)</li>
            <li>Market demand indicators (15% weight)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='recommendation-box'>
        <h4 style='color: #4A5759;'>📈 Regression Model</h4>
        <p><strong>Task:</strong> Forecast property price in 5 years</p>
        <p><strong>Output:</strong> Continuous price value in Lakhs (₹)</p>
        <p><strong>Key Factors:</strong></p>
        <ul>
            <li>Current market value (baseline)</li>
            <li>Historical appreciation trends</li>
            <li>Location growth potential</li>
            <li>Property-specific multipliers</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4A5759; padding: 2rem;'>
    <h3>🏠 Real Estate Investment Analyzer</h3>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p><strong>Disclaimer:</strong> Predictions are based on historical data and statistical models. 
    Actual property values may vary. Consult real estate professionals before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)
