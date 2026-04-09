import streamlit as st
import pandas as pd
import pickle
import joblib

# Set page configuration
st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .pred-card-rain {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .pred-card-sun {
        background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    h1 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("ML Rainfall Forecast")
st.markdown("<p style='font-size: 1.2rem; opacity: 0.8;'>Enter the weather details or fetch by location to predict whether it will rain today.</p>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_model():
    try:
        with open('rainfall_rf_model.pkl', 'rb') as file:
            return joblib.load(file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

import requests

def display_weather_metrics(df):
    st.markdown("### 🌤️ Weather Conditions Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pressure", f"{df['pressure'].iloc[0]:.1f} hPa")
    col2.metric("Max Temp", f"{df['maxtemp'].iloc[0]:.1f} °C")
    col3.metric("Min Temp", f"{df['mintemp'].iloc[0]:.1f} °C")
    col4.metric("Temperature", f"{df['temparature'].iloc[0]:.1f} °C", help="Current Temperature")
    col5.metric("Dewpoint", f"{df['dewpoint'].iloc[0]:.1f} °C")
    
    st.write("") # Spacer
    col6, col7, col8, col9, col10 = st.columns(5)
    col6.metric("Humidity", f"{df['humidity'].iloc[0]:.0f}%")
    col7.metric("Cloud Cover", f"{df['cloud'].iloc[0]:.0f}%")
    col8.metric("Sunshine", f"{df['sunshine'].iloc[0]:.1f} hrs")
    col9.metric("Wind Dir", f"{df['winddirection'].iloc[0]:.0f}°")
    col10.metric("Wind Speed", f"{df['windspeed'].iloc[0]:.1f} km/h")
    st.markdown("<br>", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose how to provide weather data:", ("Manual Input", "Fetch via Location"))

def get_location_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["latitude"], data["results"][0]["longitude"], data["results"][0].get("name", city_name), data["results"][0].get("country", "")
    return None, None, None, None

def get_weather_features(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,dew_point_2m,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m&daily=temperature_2m_max,temperature_2m_min,sunshine_duration&timezone=auto"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        # Open-Meteo returns sunshine_duration in seconds, converting to hours
        sunshine_hours = daily.get("sunshine_duration", [0])[0] / 3600.0 if daily.get("sunshine_duration", [None])[0] is not None else 0.0

        user_data = {
            'pressure': current.get("surface_pressure", 1015.0),
            'maxtemp': daily.get("temperature_2m_max", [25.0])[0],
            'temparature': current.get("temperature_2m", 20.0),
            'mintemp': daily.get("temperature_2m_min", [15.0])[0],
            'dewpoint': current.get("dew_point_2m", 15.0),
            'humidity': current.get("relative_humidity_2m", 70.0),
            'cloud': current.get("cloud_cover", 50.0),
            'sunshine': sunshine_hours,
            'winddirection': current.get("wind_direction_10m", 180),
            'windspeed': current.get("wind_speed_10m", 15.0),
        }
        return pd.DataFrame(user_data, index=[0])
    return None

df = None

if input_method == "Fetch via Location":
    city = st.sidebar.text_input("Enter City Name:", "London")
    if st.sidebar.button("Fetch Weather"):
        with st.spinner("Fetching coordinates..."):
            lat, lon, name, country = get_location_coordinates(city)
            if lat is not None and lon is not None:
                st.sidebar.success(f"Found: {name}, {country}")
                with st.spinner("Fetching weather data..."):
                    df = get_weather_features(lat, lon)
                    if df is not None:
                        st.session_state['fetched_data'] = df
                    else:
                        st.sidebar.error("Failed to fetch weather data.")
            else:
                st.sidebar.error("City not found.")
                
    st.sidebar.markdown("---")
    if 'fetched_data' in st.session_state:
        df = st.session_state['fetched_data']
        display_weather_metrics(df)
    else:
        st.info("Please fetch weather data to proceed.")

else:
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
        # Assuming windspeed from API is in km/h. Open-Meteo default is km/h.
        windspeed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 150.0, 15.0)
        
        user_data = {
            'pressure': pressure,
            'maxtemp': maxtemp,
            'temparature': temparature,
            'mintemp': mintemp,
            'dewpoint': dewpoint,
            'humidity': humidity,
            'cloud': cloud,
            'sunshine': sunshine,
            'winddirection': winddirection,
            'windspeed': windspeed
        }
        
        return pd.DataFrame(user_data, index=[0])

    df = get_user_input()
    display_weather_metrics(df)

if st.button("Predict Rainfall 🌧️", use_container_width=True):
    if df is None:
        st.error("⚠️ Please provide weather data first (either manually or by fetching weather).")
    elif model is not None:
        try:
            prediction = model.predict(df)
            prediction_proba = model.predict_proba(df)
            
            # Assuming output is mapped closely to original 'yes' or 'no' or 1/0
            pred_value = prediction[0]
            confidence = prediction_proba[0].max()
            
            if pred_value == 'yes' or pred_value == 1:
                st.markdown(f"""
                <div class="pred-card-rain">
                    <h2 style="color: white; margin-bottom: 0;">🌧️ High Probability of Rain</h2>
                    <p style="font-size: 1.2rem; opacity: 0.9;">Expect precipitation today.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-card-sun">
                    <h2 style="color: white; margin-bottom: 0;">☀️ Clear Skies Expected</h2>
                    <p style="font-size: 1.2rem; opacity: 0.9;">It is not likely to rain.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.markdown(f"**Prediction Confidence: {confidence*100:.1f}%**")
            st.progress(float(confidence))
            

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("This may happen if the input tabular shape or column names don't exactly match the training data. Check column spacing.")
    else:
        st.warning("Model missing! Please ensure 'rainfall_rf_model.pkl' is present.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #888; font-size: 0.85em; padding-top: 20px;'>Developed by <b>Aniruddha garai</b></div>", unsafe_allow_html=True)
