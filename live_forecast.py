import pandas as pd
import numpy as np
import joblib
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "raw_data/monthly_joined/Jalgaon_all_crops_monthly.csv"
WEATHER_FILE = "jalgaon_weather_history.csv"
MODEL_FILE = "crop_price_weather_model.pkl"

# ============================================================
# 2. WEATHER API INTEGRATION & AGRONOMIC ADVISORY
# ============================================================

def fetch_current_weather_forecast():
    """
    Fetches the comprehensive agronomic weather forecast using Open-Meteo API.
    Uses the exact parameters and variables provided by the user.
    """
    print("\n" + "=" * 70)
    print("🌤️ FETCHING LIVE 14-DAY AGRONOMIC WEATHER FORECAST")
    print("=" * 70)
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    
    # We use Jalgaon coordinates but could also use bounding_box logic
    # Jalgaon is approx 21.00°N, 75.56°E, which fits in the bounding box: 
    # 20.269694,76.398154,21.414925,74.761903
    params = {
        "latitude": 21.0077,
        "longitude": 75.5626,
        "hourly": [
            "temperature_2m", 
            "relative_humidity_2m", 
            "precipitation_probability", 
            "precipitation", 
            "rain", 
            "wind_speed_10m", 
            "evapotranspiration", 
            "soil_temperature_0cm", 
            "soil_moisture_0_to_1cm"
        ],
        "past_days": 0,
        "forecast_days": 14
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        
        # Extract variables
        temp = hourly.Variables(0).ValuesAsNumpy()
        humidity = hourly.Variables(1).ValuesAsNumpy()
        precip_prob = hourly.Variables(2).ValuesAsNumpy()
        precipitation = hourly.Variables(3).ValuesAsNumpy()
        rain = hourly.Variables(4).ValuesAsNumpy()
        wind_speed = hourly.Variables(5).ValuesAsNumpy()
        et0 = hourly.Variables(6).ValuesAsNumpy()
        soil_temp = hourly.Variables(7).ValuesAsNumpy()
        soil_moisture = hourly.Variables(8).ValuesAsNumpy()
        
        # Calculate summaries for advisory
        avg_temp = np.nanmean(temp)
        avg_humidity = np.nanmean(humidity)
        total_rain = np.nansum(rain)
        avg_wind = np.nanmean(wind_speed)
        total_et0 = np.nansum(et0)
        avg_soil_moisture = np.nanmean(soil_moisture)
        max_precip_prob = np.nanmax(precip_prob)
        
        print("💡 AGRONOMIC ADVISORY FOR THE NEXT 14 DAYS:")
        print(f"  • Temperature (Avg): {avg_temp:.1f}°C -> Important for heat stress and crop growth.")
        print(f"  • Relative Humidity: {avg_humidity:.1f}% -> Watch for disease-risk indicators.")
        print(f"  • Peak Rain Probability: {max_precip_prob:.1f}% -> Chance of rain.")
        print(f"  • Total Expected Rain: {total_rain:.1f} mm -> Rainfall amount expected.")
        print(f"  • Wind Speed (Avg): {avg_wind:.1f} km/h -> Plan wind-sensitive spraying.")
        print(f"  • Evapotranspiration (ET₀): {total_et0:.1f} mm -> Estimated crop water demand.")
        print(f"  • Soil Moisture (0-1cm): {avg_soil_moisture:.3f} m³/m³ -> Surface moisture conditions.\n")
        
        # Scale 14-day rainfall to estimate the full monthly rainfall for the ML model
        est_monthly_rainfall = total_rain * (30.0 / 14.0)
        
        return avg_temp, avg_humidity, est_monthly_rainfall
        
    except Exception as e:
        print(f"❌ Failed to fetch weather API: {e}")
        return None, None, None

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

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

    row = {
        "crop": crop,
        "Month_num": month_num,
        "month_sin": np.sin(2 * np.pi * month_num / 12),
        "month_cos": np.cos(2 * np.pi * month_num / 12),
        
        "price_lag_1": prices[-1],
        "price_lag_2": prices[-2],
        "price_lag_3": prices[-3],
        "price_lag_6": prices[-6] if len(prices) >= 6 else prices[0],
        "price_lag_12": prices[-12] if len(prices) >= 12 else prices[0],
        
        "quantity_lag_1": quantities[-1],
        "quantity_lag_2": quantities[-2],
        "quantity_lag_3": quantities[-3],
        "quantity_lag_6": quantities[-6] if len(quantities) >= 6 else quantities[0],
        "quantity_lag_12": quantities[-12] if len(quantities) >= 12 else quantities[0],
        
        "price_rolling_mean_3": np.mean(prices[-3:]),
        "price_rolling_mean_6": np.mean(prices[-6:]) if len(prices) >= 6 else np.mean(prices),
        "price_rolling_mean_12": np.mean(prices[-12:]) if len(prices) >= 12 else np.mean(prices),
        "price_rolling_std_3": np.std(prices[-3:], ddof=1) if len(prices) >= 3 else 0,
        
        "quantity_rolling_mean_3": np.mean(quantities[-3:]),
        "quantity_rolling_mean_6": np.mean(quantities[-6:]) if len(quantities) >= 6 else np.mean(quantities),
        
        "Temperature": temp,
        "Humidity": humidity,
        "Rainfall": rain
    }
    
    return pd.DataFrame([row])


# ============================================================
# 4. MAIN PREDICTION LOOP
# ============================================================

def live_predict():
    if not os.path.exists(MODEL_FILE):
        print(f"Model file '{MODEL_FILE}' not found! Please run train_weather_model.py first.")
        return

    pipeline = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)
    df["month"] = pd.to_datetime(df["month"])
    
    weather_df = pd.read_csv(WEATHER_FILE)
    weather_df["Month_num"] = weather_df["Month"].astype(int)
    
    print("\n" + "=" * 70)
    print("LIVE CROP PRICE FORECAST")
    print("=" * 70)
    
    crops = sorted(df["crop"].unique())
    for i, c in enumerate(crops, 1):
        print(f"{i}. {c}")
        
    try:
        crop_idx = int(input("\nSelect crop number: ")) - 1
        selected_crop = crops[crop_idx]
        target_month = int(input("Enter 2026 month to forecast up to (1-12): "))
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    target_date = pd.Timestamp(year=2026, month=target_month, day=1)
    
    history = df[["crop", "month", "avg_modal_price", "total_quantity"]].copy()
    history = history[history["crop"] == selected_crop].sort_values("month")
    
    latest_date = history["month"].max()
    
    if target_date <= latest_date:
        print("Target date is in the past or already has data.")
        return
        
    # 1. Fetch live weather & Agronomic Advisory
    api_temp, api_hum, api_rain = fetch_current_weather_forecast()
    
    print(f"\nForecasting {selected_crop} from {latest_date.strftime('%B %Y')} to {target_date.strftime('%B %Y')}...")
    
    current_date = latest_date + pd.DateOffset(months=1)
    forecast_results = []
    
    is_first_prediction = True

    while current_date <= target_date:
        
        if is_first_prediction and api_temp is not None:
            temp, hum, rain = api_temp, api_hum, api_rain
            is_first_prediction = False
        else:
            temp, hum, rain = get_historical_weather_averages(weather_df, current_date.month)
            
        future_features = create_future_features(
            history, selected_crop, current_date, temp, hum, rain
        )
        
        predicted_price = pipeline.predict(future_features)[0]
        
        print(f"[{current_date.strftime('%Y-%m')}] Predicted: ₹{predicted_price:,.2f} (Temp: {temp:.1f}°C, Rain: {rain:.1f}mm)")
        
        forecast_results.append({
            "Crop": selected_crop,
            "Month": current_date.strftime("%Y-%m"),
            "Predicted Price (₹)": round(predicted_price, 2),
            "Used_Temp": round(temp, 1),
            "Used_Rain": round(rain, 1)
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

    print("\n" + "=" * 70)
    print("FORECAST SUMMARY")
    print("=" * 70)
    res_df = pd.DataFrame(forecast_results)
    print(res_df.to_string(index=False))
    
    out_file = f"live_forecast_{selected_crop}_2026.csv"
    res_df.to_csv(out_file, index=False)
    print(f"\nSaved forecast to {out_file}")

if __name__ == "__main__":
    live_predict()
