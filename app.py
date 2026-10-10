import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(page_title="AgriSmart Forecast", page_icon="🌾", layout="wide")

DATA_FILE = "data/Jalgaon_all_crops_monthly.csv"
WEATHER_FILE = "data/jalgaon_weather_history.csv"
MODEL_FILE = "models/crop_price_weather_model.pkl"

# Load your Groq API Key securely from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# HELPERS
# ============================================================

@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    """Fetch Open-Meteo weather for Jalgaon."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    # Removed bounding box to fix the longitude error, 
    # relying directly on the Jalgaon coordinates for accurate forecasting.
    params = {
        "latitude": 21.0077,
        "longitude": 75.5626,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation_probability", "precipitation", "rain", "wind_speed_10m", "evapotranspiration", "soil_temperature_0cm", "soil_moisture_0_to_1cm"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_probability_max", "wind_speed_10m_max"],
        "timezone": "Asia/Kolkata",
        "past_days": 0,
        "forecast_days": 14
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        daily = response.Daily()
        
        # Hourly calculations for model
        temp = hourly.Variables(0).ValuesAsNumpy()
        humidity = hourly.Variables(1).ValuesAsNumpy()
        precip_prob = hourly.Variables(2).ValuesAsNumpy()
        rain = hourly.Variables(4).ValuesAsNumpy()
        wind_speed = hourly.Variables(5).ValuesAsNumpy()
        et0 = hourly.Variables(6).ValuesAsNumpy()
        soil_moisture = hourly.Variables(8).ValuesAsNumpy()
        
        avg_temp = np.nanmean(temp)
        avg_humidity = np.nanmean(humidity)
        total_rain = np.nansum(rain)
        avg_wind = np.nanmean(wind_speed)
        total_et0 = np.nansum(et0)
        avg_soil_moisture = np.nanmean(soil_moisture)
        max_precip_prob = np.nanmax(precip_prob)
        
        est_monthly_rainfall = total_rain * (30.0 / 14.0)
        
        # Daily calculations for UI
        daily_time = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        daily_df = pd.DataFrame({
            "Date": daily_time.strftime("%Y-%m-%d"),
            "Max Temp (°C)": daily.Variables(0).ValuesAsNumpy(),
            "Min Temp (°C)": daily.Variables(1).ValuesAsNumpy(),
            "Rain (mm)": daily.Variables(2).ValuesAsNumpy(),
            "Rain Prob (%)": daily.Variables(3).ValuesAsNumpy(),
            "Wind Max (km/h)": daily.Variables(4).ValuesAsNumpy()
        })
        
        metrics = {
            "avg_temp": avg_temp,
            "avg_humidity": avg_humidity,
            "total_rain": total_rain,
            "avg_wind": avg_wind,
            "total_et0": total_et0,
            "avg_soil_moisture": avg_soil_moisture,
            "max_precip_prob": max_precip_prob,
            "est_monthly_rainfall": est_monthly_rainfall,
            "daily_df": daily_df
        }
        return metrics
    except Exception as e:
        st.error(f"Failed to fetch weather API: {e}")
        return None

def get_historical_weather_averages(weather_df, month_num):
    m_data = weather_df[weather_df["Month_num"] == month_num]
    if m_data.empty:
        return 30.0, 50.0, 10.0
    return m_data["Temperature"].mean(), m_data["Humidity"].mean(), m_data["Rainfall"].mean()

def create_future_features(history_df, crop, prediction_date, temp, humidity, rain):
    crop_data = history_df[history_df["crop"] == crop].sort_values("month")
    prices = crop_data["avg_modal_price"].dropna().values
    quantities = crop_data["total_quantity"].dropna().values
    month_num = prediction_date.month

    current_price = prices[-1]
    current_qty = quantities[-1]

    row = {
        "Month_num": month_num,
        "month_sin": np.sin(2 * np.pi * month_num / 12),
        "month_cos": np.cos(2 * np.pi * month_num / 12),
        
        "price_lag_1_ratio": prices[-1] / current_price if current_price else 1.0,
        "price_lag_2_ratio": (prices[-2] if len(prices) >= 2 else prices[0]) / current_price if current_price else 1.0,
        "price_lag_3_ratio": (prices[-3] if len(prices) >= 3 else prices[0]) / current_price if current_price else 1.0,
        "price_lag_6_ratio": (prices[-6] if len(prices) >= 6 else prices[0]) / current_price if current_price else 1.0,
        "price_lag_12_ratio": (prices[-12] if len(prices) >= 12 else prices[0]) / current_price if current_price else 1.0,
        
        "quantity_lag_1_ratio": quantities[-1] / current_qty if current_qty else 1.0,
        "quantity_lag_2_ratio": (quantities[-2] if len(quantities) >= 2 else quantities[0]) / current_qty if current_qty else 1.0,
        "quantity_lag_3_ratio": (quantities[-3] if len(quantities) >= 3 else quantities[0]) / current_qty if current_qty else 1.0,
        "quantity_lag_6_ratio": (quantities[-6] if len(quantities) >= 6 else quantities[0]) / current_qty if current_qty else 1.0,
        "quantity_lag_12_ratio": (quantities[-12] if len(quantities) >= 12 else quantities[0]) / current_qty if current_qty else 1.0,
        
        "price_rolling_mean_3_ratio": (np.mean(prices[-3:]) if len(prices) >= 3 else np.mean(prices)) / current_price if current_price else 1.0,
        "price_rolling_mean_6_ratio": (np.mean(prices[-6:]) if len(prices) >= 6 else np.mean(prices)) / current_price if current_price else 1.0,
        "price_rolling_mean_12_ratio": (np.mean(prices[-12:]) if len(prices) >= 12 else np.mean(prices)) / current_price if current_price else 1.0,
        
        "quantity_rolling_mean_3_ratio": (np.mean(quantities[-3:]) if len(quantities) >= 3 else np.mean(quantities)) / current_qty if current_qty else 1.0,
        "quantity_rolling_mean_6_ratio": (np.mean(quantities[-6:]) if len(quantities) >= 6 else np.mean(quantities)) / current_qty if current_qty else 1.0,
        
        "Temperature": temp,
        "Humidity": humidity,
        "Rainfall": rain
    }
    return pd.DataFrame([row])

# ============================================================
# APP UI
# ============================================================

st.title("🌾 AgriSmart: Crop Price Forecast & Agronomic Advisory")
st.markdown("Predict future crop prices using historical trends and live weather data. Powered by Open-Meteo & Groq LLM.")

if not os.path.exists(DATA_FILE):
    st.error("Data file missing! Please ensure you have placed it in the 'data' folder inside AgriSmart_App.")
    st.stop()

# Load Data
df = pd.read_csv(DATA_FILE)
df["month"] = pd.to_datetime(df["month"])
crops = sorted(df["crop"].unique())

weather_df = pd.read_csv(WEATHER_FILE)
weather_df["Month_num"] = weather_df["Month"].astype(int)

# Removed single pipeline loading from here

# Sidebar
st.sidebar.header("Configuration")
selected_crop = st.sidebar.selectbox("Select Crop", crops)

# 2026 Estimated baseline prices to bridge the dataset gap automatically
CURRENT_MARKET_PRICES_2026 = {
    "Bajra": 2500.0,
    "Banana": 1520.0,
    "Bengal_Gram": 6110.0,
    "Cotton": 9015.0,
    "Groundnut": 8500.0,
    "Jowar": 2740.0,
    "Lime": 4000.0,
    "Mousambi": 2700.0,
    "Soyabean": 6245.0,
    "Wheat": 2750.0
}

# --- NEW: Manual Data Entry for Missing Months ---
history = df[["crop", "month", "avg_modal_price", "total_quantity"]].copy()
history = history[history["crop"] == selected_crop].sort_values("month")
latest_csv_date = history["month"].max()

st.sidebar.markdown("---")
st.sidebar.subheader("補 Current Market Data")
st.sidebar.write(f"Dataset currently ends at: **{latest_csv_date.strftime('%B %Y')}**")

use_custom_data = st.sidebar.checkbox("Add Recent Market Data to Improve Accuracy")
if use_custom_data:
    # Let the user set the "current" month to bypass the gap in the dataset
    custom_year = st.sidebar.number_input("Current Year", min_value=2024, max_value=2028, value=pd.Timestamp.today().year)
    custom_month = st.sidebar.slider("Current Month", 1, 12, pd.Timestamp.today().month)
    
    default_price = CURRENT_MARKET_PRICES_2026.get(selected_crop, float(history["avg_modal_price"].iloc[-1]))
    
    price_1_month_ago = st.sidebar.number_input(f"1 Month Ago Price (₹)", value=float(default_price * 0.95))
    price_7_days_ago = st.sidebar.number_input(f"7 Days Ago Price (₹)", value=float(default_price))
    custom_price = st.sidebar.number_input(f"Current Price (₹)", value=float(default_price))
    
    current_date_input = pd.Timestamp(year=custom_year, month=custom_month, day=1)
    prev_month_input = current_date_input - pd.DateOffset(months=1)
    
    last_qty = float(history["total_quantity"].iloc[-1])
    
    # Combine 7 days ago and current to represent the current month's average trajectory
    current_month_avg = (custom_price + price_7_days_ago) / 2.0
    
    # Append both the previous month and current month to feed accurate lag features
    new_rows = pd.DataFrame({
        "crop": [selected_crop, selected_crop],
        "month": [prev_month_input, current_date_input],
        "avg_modal_price": [price_1_month_ago, current_month_avg],
        "total_quantity": [last_qty, last_qty]
    })
    history = pd.concat([history, new_rows], ignore_index=True)
    latest_date = current_date_input
else:
    latest_date = latest_csv_date

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Settings")
target_year = st.sidebar.number_input("Target Forecast Year", min_value=2024, max_value=2028, value=2026)
target_month = st.sidebar.slider("Target Forecast Month", 1, 12, 12)
target_date = pd.Timestamp(year=target_year, month=target_month, day=1)

# Fetch Weather
weather_metrics = fetch_weather_forecast()

if weather_metrics:
    st.subheader("🌤️ Live Agronomic Weather Advisory (Next 14 Days)")
    cols = st.columns(4)
    cols[0].metric("Avg Temperature", f"{weather_metrics['avg_temp']:.1f} °C", "Heat stress / growth")
    cols[1].metric("Avg Humidity", f"{weather_metrics['avg_humidity']:.1f} %", "Disease risk")
    cols[2].metric("Total Rain", f"{weather_metrics['total_rain']:.1f} mm", f"{weather_metrics['max_precip_prob']:.0f}% Peak Prob")
    cols[3].metric("Avg Wind Speed", f"{weather_metrics['avg_wind']:.1f} km/h", "Spraying conditions")
    
    cols2 = st.columns(4)
    cols2[0].metric("Total ET₀", f"{weather_metrics['total_et0']:.1f} mm", "Water demand")
    cols2[1].metric("Soil Moisture", f"{weather_metrics['avg_soil_moisture']:.3f}", "0-1cm depth")
    
    with st.expander("📅 View Date-wise Daily Weather Details"):
        st.dataframe(weather_metrics["daily_df"], use_container_width=True)

st.markdown("---")
st.subheader(f"📈 Price Forecast for {selected_crop}")

MODEL_FILE = f"models/crop_price_weather_model_{selected_crop}.pkl"
if not os.path.exists(MODEL_FILE):
    st.error(f"Model file for {selected_crop} not found! Please run train_weather_model.py again.")
    st.stop()
pipeline = joblib.load(MODEL_FILE)

if target_date <= latest_date:
    st.warning(f"Target date {target_date.strftime('%B %Y')} is not in the future compared to {latest_date.strftime('%B %Y')}.")
else:
    with st.spinner("Calculating future prices recursively..."):
        current_date = latest_date + pd.DateOffset(months=1)
        forecast_results = []
        is_first_prediction = True

        while current_date <= target_date:
            if is_first_prediction and weather_metrics is not None:
                temp = weather_metrics["avg_temp"]
                hum = weather_metrics["avg_humidity"]
                rain = weather_metrics["est_monthly_rainfall"]
                is_first_prediction = False
            else:
                temp, hum, rain = get_historical_weather_averages(weather_df, current_date.month)
                
            future_features = create_future_features(history, selected_crop, current_date, temp, hum, rain)
            
            # The model now predicts the *ratio* of (next_month_price / this_month_price)
            current_price = history["avg_modal_price"].iloc[-1]
            predicted_ratio = pipeline.predict(future_features)[0]
            predicted_price = current_price * predicted_ratio
            
            forecast_results.append({
                "Month": current_date.strftime("%Y-%m"),
                "Predicted Price (₹)": round(predicted_price, 2),
                "Used Temp (°C)": round(temp, 1),
                "Used Rain (mm)": round(rain, 1)
            })
            
            latest_qty = history["total_quantity"].iloc[-1]
            new_row = pd.DataFrame({
                "crop": [selected_crop],
                "month": [current_date],
                "avg_modal_price": [predicted_price],
                "total_quantity": [latest_qty]
            })
            history = pd.concat([history, new_row], ignore_index=True)
            current_date = current_date + pd.DateOffset(months=1)

        # Create a display dataframe focusing on current date (-3 months) to target date
        current_system_date = pd.Timestamp(year=pd.Timestamp.today().year, month=pd.Timestamp.today().month, day=1)
        start_display_date = current_system_date - pd.DateOffset(months=3)
        
        display_history = history[(history["month"] >= start_display_date) & (history["month"] <= target_date)].copy()
        if display_history.empty:
            display_history = history.tail(6).copy() # Fallback if dates are weird
            
        display_history["Month"] = display_history["month"].dt.strftime("%Y-%m")
        display_history["Price (₹)"] = display_history["avg_modal_price"].round(2)
        
        def get_type(d):
            if d > latest_date: return "Predicted"
            elif d == latest_date: return "Current"
            else: return "Historical"
            
        display_history["Type"] = display_history["month"].apply(get_type)
        display_df = display_history[["Month", "Price (₹)", "Type"]]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Plot full history + predictions
        full_plot_df = history.copy()
        full_plot_df["Month"] = full_plot_df["month"].dt.strftime("%Y-%m")
        full_plot_df["Price (₹)"] = full_plot_df["avg_modal_price"].round(2)
        st.line_chart(full_plot_df.set_index("Month")["Price (₹)"])

    # LLM Integration
    st.markdown("---")
    st.subheader("💬 Farm AI Assistant")
    st.markdown("Ask the AI about your crop, prices, or weather!")
    
    if GROQ_API_KEY:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if prompt := st.chat_input("Ask about your crop prices or weather..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = Groq(api_key=GROQ_API_KEY)
                        
                        system_prompt = f"""
                        You are an expert agricultural advisor. The user is asking about {selected_crop}.
                        Forecasted prices:
                        {display_df.to_string(index=False)}
                        Live 14-day weather: Temp: {weather_metrics['avg_temp']:.1f}°C, Rain: {weather_metrics['total_rain']:.1f} mm.
                        Provide a concise and helpful response directly to the farmer.
                        """
                        
                        api_messages = [{"role": "system", "content": system_prompt}]
                        for m in st.session_state.messages:
                            api_messages.append(m)
                            
                        chat_completion = client.chat.completions.create(
                            messages=api_messages,
                            model="qwen/qwen3.8-27b",
                            temperature=0.5,
                        )
                        
                        response = chat_completion.choices[0].message.content
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error communicating with Groq API: {e}")
    else:
        st.warning("⚠️ Groq API Key is not set in the .env file. Please add GROQ_API_KEY=your_key to the .env file.")
