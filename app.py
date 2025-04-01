import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta

# Load saved model and preprocessing objects
rf = joblib.load('fertilizer_classifier.pkl')
preprocessor = joblib.load('preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Weather API Configuration
OPENWEATHER_API_KEY = "afeacd8627fcc0f84097417310f54c33" 

def get_weather_data(city):
    """Fetch temperature, humidity, and rainfall using OpenWeatherMap API."""
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url)
        
        if geo_response.status_code == 200 and geo_response.json():
            lat = geo_response.json()[0]['lat']
            lon = geo_response.json()[0]['lon']
            
            # Get forecast data
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
            forecast_response = requests.get(forecast_url)
            
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                
                # Calculate expected rainfall for next 24 hours
                rainfall = 0
                now = datetime.now()
                for entry in forecast_data['list']:
                    forecast_time = datetime.fromtimestamp(entry['dt'])
                    if forecast_time <= now + timedelta(hours=24):
                        rainfall += entry.get('rain', {}).get('3h', 0)
                
                # Get current weather
                current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
                current_response = requests.get(current_url)
                current_data = current_response.json()
                
                return {
                    'temperature': current_data['main']['temp'],
                    'humidity': current_data['main']['humidity'],
                    'rainfall': round(rainfall, 1)
                }
        return None
    except Exception as e:
        st.error(f"Weather API Error: {str(e)}")
        return None

def calculate_irrigation(crop_type, soil_type, moisture, temp, humidity, rainfall):
    crop_requirements = {
        'Cotton': 6.5, 'Sugarcane': 7.0, 'Wheat': 4.5, 'Maize': 5.0,
        'Paddy': 5.5, 'Tobacco': 5.2, 'Barley': 4.0, 'Millets': 3.8,
        'Oil seeds': 4.8, 'Pulses': 3.5, 'Ground Nuts': 4.2
    }
    
    soil_factors = {'Sandy': 1.3, 'Loamy': 1.0, 'Clayey': 0.7, 'Black': 0.9, 'Red': 1.1}
    
    et0 = 0.0023 * (temp + 17.8) * (np.sqrt(temp) - 0.32 * humidity)
    irrigation_need = ((crop_requirements[crop_type] - rainfall / 2) / 0.8) * soil_factors[soil_type]
    return max(0, round(irrigation_need - moisture / 10, 2))

# Streamlit UI
st.title("🌾Agricultural Assistant")
st.header("Smart Fertilizer & Irrigation Management")

with st.expander("🌱 Crop Configuration"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        city = st.text_input("City")
        soil_type = st.selectbox("Soil Type", 
        ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])
        crop_type = st.selectbox("Crop Type", [
        'Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
        'Barley', 'Millets', 'Oil seeds', 'Pulses', 'Wheat', 
        'Ground Nuts'
    ])
    moisture = st.slider("Soil Moisture (%)", 0, 100, 40)
    with col2:
        if st.button("Fetch Weather Data"):
            weather_data = get_weather_data(city)
            if weather_data:
                st.session_state['weather'] = weather_data
                st.success("Weather data fetched successfully!")
            else:
                st.error("Failed to fetch weather data")
    with col3: nitrogen = st.slider("Nitrogen (ppm)", 0, 100, 40)
    with col4: phosphorous = st.slider("Phosphorous (ppm)", 0, 100, 35)
    with col5: potassium = st.slider("Potassium (ppm)", 0, 100, 30)

# Weather display
weather_data = st.session_state.get('weather', None)
manual_mode = False

if weather_data:
    temp = weather_data['temperature']
    humidity = weather_data['humidity']
    rainfall = weather_data['rainfall']
    st.success(f"Current Weather: {temp}°C | Humidity: {humidity}% | Expected Rainfall: {rainfall}mm")
else:
    st.warning("Weather API unavailable. Using manual input:")
    manual_mode = True
    col1, col2, col3 = st.columns(3)
    with col1: temp = st.number_input("Temperature (°C)", 20.0, 45.0, 30.0)
    with col2: humidity = st.number_input("Humidity (%)", 30.0, 90.0, 60.0)
    with col3: rainfall = st.number_input("Expected Rainfall (mm)", 0.0, 50.0, 10.0)



# Predictions
col1, col2 = st.columns(2)
with col1:
    # Create input data dictionary
    input_data = {
        'Temparature': [temp],    # Note spelling to match training data
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Soil Type': [soil_type],
        'Crop Type': [crop_type],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous]  # Note spelling variant
    }
    
    # Create DataFrame from input data
    input_df = pd.DataFrame(input_data)
    
    # Compute aggregated features for prediction
    input_df['soil_param'] = input_df[['Temparature', 'Humidity', 'Moisture']].mean(axis=1)
    input_df['nutrient_param'] = input_df[['Nitrogen', 'Phosphorous', 'Potassium']].mean(axis=1)
    
    # Drop original columns used for aggregation to match training data format
    input_df.drop(['Temparature', 'Humidity', 'Moisture',
                   'Nitrogen', 'Phosphorous', 'Potassium'], axis=1, inplace=True)
    
    try:
        # Preprocess input data
        processed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = rf.predict(processed_data)
        fertilizer = label_encoder.inverse_transform(prediction)[0]
        
        # Display recommendation with custom message
        fertilizer_messages = {
            "10-26-26": "Use High Phosphorus Fertilizer (10-26-26)",
            "14-35-14": "Use Balanced NPK fertilizer with high phosphorus fertilizer (14-35-14)",
            "17-17-17": "Use Balanced NPK fertilizer (17-17-17)",
            "20-20": "Use equal nitrogen and phosphorus fertilizer (20-20)",
            "DAP": "Use DAP fertilizer",
            "28-28": "Use high concentration equal nitrogen and phosphorus fertilizer (28-28)",
            "Urea": "Use High nitrogen fertilizer (Urea)"
        }
        
        st.success(fertilizer_messages.get(fertilizer, f"Fertilizer: {fertilizer}"))
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

with col2:
    if st.button("💧 Calculate Irrigation Needs"):
        irrigation = calculate_irrigation(crop_type, soil_type, moisture, temp, humidity, rainfall)
        st.info(f"**Daily Irrigation Requirement:** {irrigation} mm/day")

st.markdown("---")
st.caption("Developed by Cipher | Powered by ML Models")
