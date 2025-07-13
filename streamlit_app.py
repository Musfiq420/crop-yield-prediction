import streamlit as st
import pandas as pd
import joblib

# Load trained model and features
model = joblib.load("xgb_crop_yield_model_bd.joblib")
feature_names = joblib.load("model_features.joblib")

# Extract crops and country columns from feature names
crop_columns = [col for col in feature_names if col.startswith("Item_")]
area_column = "Area"

# Title
st.title("🌾 Global Crop Yield Prediction App")

st.markdown("""
Predict crop yield (hg/ha) based on:
- Country (Area)
- Crop Type
- Year
- Rainfall (mm)
- Avg Temperature (°C)
- Pesticide Usage (tonnes)
""")

# Input fields
area = st.text_input("Country (Area)", "Bangladesh")
year = st.number_input("Year", min_value=1960, max_value=2035, value=2025)
rainfall = st.number_input("Average Rainfall (mm)", value=2000.0)
temperature = st.number_input("Average Temperature (°C)", value=25.0)
pesticides = st.number_input("Pesticide Usage (tonnes)", value=450.0)
crop = st.selectbox("Crop", [col.replace("Item_", "") for col in crop_columns])

# Prepare input row
input_data = {
    "Year": year,
    "average_rain_fall_mm_per_year": rainfall,
    "pesticides_tonnes": pesticides,
    "avg_temp": temperature,
    "Area": area
}

# Add one-hot crop fields (ensure all 0s, then 1 for selected)
for col in crop_columns:
    input_data[col] = 1 if col == f"Item_{crop}" else 0

# Convert to DataFrame and align columns
input_df = pd.DataFrame([input_data])

# Handle missing columns (e.g. Area one-hot not used, just passed as text)
missing_cols = set(feature_names) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Ensure correct column order
input_df = input_df[feature_names]

# Predict
if st.button("🔮 Predict Crop Yield"):
    yield_pred = model.predict(input_df)[0]
    st.success(f"🌾 Predicted Yield for {crop} in {area}, {year}: **{yield_pred:.2f} hg/ha**")
