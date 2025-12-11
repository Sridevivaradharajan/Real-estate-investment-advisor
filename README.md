#  **Real Estate Investment Advisor**

## **Project Title**

**Real Estate Investment Advisor: Predicting Property Profitability & Future Value**

## **Objective**

Build a machine-learning system that helps real estate investors evaluate:

1. **Whether a property is a Good Investment (Classification)**
2. **The estimated property price after 5 years (Regression)**

The solution includes a deployed **Streamlit application** with prediction, visualization, and explainability modules.

---

# **Skills & Technologies**

**Python · ML · EDA · Feature Engineering · Classification · Regression · Streamlit · MLflow · SHAP · Model Evaluation**

---

# **Problem Statement**

Develop a reliable ML-driven decision-support system that can:

* Predict future price appreciation for any property
* Classify a property as a profitable or non-profitable investment
* Integrate both predictions inside one unified Streamlit interface
* Track and manage all ML experiments using MLflow
* Provide transparent explainability through feature importance & SHAP

---

# **Business Use Cases**

* Assist investors in selecting high-ROI properties
* Enable real estate companies to automate investment evaluation
* Provide data-driven recommendations rather than speculative decisions
* Build customer trust with accurate long-term price forecasting

---

# **Approach**

### **1️⃣ Data Preparation**

* Handled missing values, outliers, and duplicates
* Encoded categorical features
* Scaled numerical variables
* Engineered features such as Price_per_SqFt, Age_of_Property, Growth Indicators

### **2️⃣ Modeling**

**Classification Target:** Good_Investment
Models Tested → Logistic Regression, Random Forest, XGBoost

**Regression Target:** Future_Price_5Y
Models Tested → Random Forest, Ridge, XGBoost

### **3️⃣ Model Evaluation Metrics**

**Classification:** Accuracy, F1, ROC-AUC, Overfitting Gap
**Regression:** RMSE, MAE, R², Overfitting Gap

### **4️⃣ MLflow Integration**

* Logged experiments, metrics, parameters, and model artifacts
* Stored and compared best-performing models

### **5️⃣ Streamlit Application**

Tabs included:

1. **Single Prediction**
2. **Bulk Prediction**
3. **Market Insights (Visualizations)**
4. **Feature Importance & SHAP Analysis**

---

# **Final Model Selection**

### **Classification – Selected Model: XGBoost**

* Best Accuracy: **95.82%**
* Best F1-score and ROC-AUC
* Stable with minimal overfitting

### **Regression – Selected Model: Ridge Regression**

* R² Score: **0.9954**
* Lowest RMSE
* Most stable generalization performance

---

# **Model Explainability (SHAP & Feature Importance)**

Top contributing features include:

* City & Locality
* Price_per_SqFt
* BHK
* Size_in_SqFt
* Age_of_Property
* Amenities & Transport Accessibility

Explainability helps justify predictions and build investor confidence.




✅ A **1-page executive summary**
✅ A **final PPT for presentation**
