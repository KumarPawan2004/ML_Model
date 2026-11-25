import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Car Price & Mileage Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price & Mileage Prediction System")
st.markdown("Enter the car's specifications below to predict its **Price** and **Mileage**.")

# Load trained model and features
@st.cache_resource
def load_model():
    try:
        data = joblib.load("car_model.pkl")
        return data["model"], data["columns"]
    except Exception as e:
        st.error("Model file `car_model.pkl` not found. Please train and save the model first from your notebook.")
        return None, None

model, feature_columns = load_model()

if model is not None:
    st.subheader("📋 Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_year = st.number_input("Model Year", min_value=1990, max_value=2026, value=2021, step=1)
        engine_size = st.number_input("Engine Size (Liters)", min_value=0.8, max_value=6.5, value=2.0, step=0.1)
        horsepower = st.number_input("Horsepower (HP)", min_value=50, max_value=800, value=180, step=10)
        doors = st.selectbox("Number of Doors", options=[2, 3, 4, 5], index=2)
        
    with col2:
        owner_count = st.selectbox("Previous Owners", options=[1, 2, 3, 4, 5], index=0)
        brand = st.selectbox("Brand", options=["Ford", "Honda", "Hyundai", "Tesla", "Toyota", "Other"])
        fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Hybrid", "Electric"])
        transmission = st.selectbox("Transmission", options=["Automatic", "Manual"])

    st.markdown("---")
    
    if st.button("🔮 Predict Price & Mileage", type="primary", use_container_width=True):
        # Build single-row DataFrame matching the exact training columns
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Populate numeric features
        input_df["Model_Year"] = model_year
        input_df["Engine_Size"] = engine_size
        input_df["Horsepower"] = horsepower
        input_df["Doors"] = doors
        input_df["Owner_Count"] = owner_count
        
        # Populate one-hot encoded categorical features
        brand_col = f"Brand_{brand}"
        if brand_col in input_df.columns:
            input_df[brand_col] = 1
            
        fuel_col = f"Fuel_Type_{fuel_type}"
        if fuel_col in input_df.columns:
            input_df[fuel_col] = 1
            
        if transmission == "Manual" and "Transmission_Manual" in input_df.columns:
            input_df["Transmission_Manual"] = 1

        # Predict
        prediction = model.predict(input_df)
        pred_price = max(0, prediction[0, 0])
        pred_mileage = max(0, prediction[0, 1])

        # Display results
        st.success("### Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Estimated Price", value=f"${pred_price:,.2f}")
        with res_col2:
            st.metric(label=" Estimated Mileage", value=f"{pred_mileage:,.0f} miles")
