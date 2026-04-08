import streamlit as st
import pandas as pd
import pickle
import joblib

# Set page configuration
st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️", layout="centered")

st.title("🌧️ ML Rainfall Forecast")
st.write("Enter the weather details below to predict whether it will rain.")

@st.cache_resource
def load_model():
    try:
        with open('rainfall_rf_model.pkl', 'rb') as file:
            return joblib.load(file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Sidebar for user input
st.sidebar.header("Weather Features")

def get_user_input():
    pressure = st.sidebar.slider("Pressure (hPa)", 980.0, 1050.0, 1015.0)
    maxtemp = st.sidebar.slider("Max Temperature (°C)", -5.0, 50.0, 25.0)
    temparature = st.sidebar.slider("Temperature (°C)", -5.0, 45.0, 20.0)
    mintemp = st.sidebar.slider("Min Temperature (°C)", -10.0, 35.0, 15.0)
    dewpoint = st.sidebar.slider("Dewpoint (°C)", -10.0, 35.0, 15.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 70.0)
    cloud = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 50.0)
    sunshine = st.sidebar.slider("Sunshine (hours)", 0.0, 24.0, 5.0)
    winddirection = st.sidebar.slider("Wind Direction (°)", 0, 360, 180)
    windspeed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 150.0, 15.0)
    
    # Dictionary matching exact spelling/spaces from the CSV features
    user_data = {
        'pressure': pressure,
        'maxtemp': maxtemp,
        'temparature': temparature,  # Notice the spelling
        'mintemp': mintemp,
        'dewpoint': dewpoint,
        'humidity': humidity,
        'cloud': cloud,
        'sunshine': sunshine,
        'winddirection': winddirection,
        'windspeed': windspeed
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

df = get_user_input()

st.subheader("User Input Parameters:")
st.write(df)

if st.button("Predict Rainfall 🌧️", use_container_width=True):
    if model is not None:
        try:
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)
            
            st.subheader("Prediction Result:")
            
            # Assuming output is mapped closely to original 'yes' or 'no' or 1/0
            pred_value = prediction[0]
            if pred_value == 'yes' or pred_value == 1:
                st.success("🌧️ **It is highly likely to RAIN.**")
            else:
                st.success("☀️ **It is NOT likely to rain.**")
                
            st.info(f"Prediction Probability: {prediction_proba[0].max()*100:.2f}% confidence.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("This may happen if the input tabular shape or column names don't exactly match the training data. Check column spacing.")
    else:
        st.warning("Model missing! Please ensure 'rainfall_rf_model.pkl' is present.")
