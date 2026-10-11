from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

app = FastAPI(title="AgriSmart Backend", version="1.0.0")

# Enable CORS for Flutter Web, iOS, Android, and Desktop
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

def resolve_path(rel_path: str) -> str:
    candidates = [
        os.path.normpath(os.path.join(REPO_ROOT, rel_path)),
        os.path.normpath(os.path.join(BASE_DIR, rel_path)),
        os.path.normpath(os.path.join(BASE_DIR, "..", rel_path)),
        os.path.normpath(os.path.join(os.getcwd(), rel_path)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.normpath(os.path.join(REPO_ROOT, rel_path))

DATA_FILE = resolve_path("data/Jalgaon_all_crops_monthly.csv")
WEATHER_FILE = resolve_path("data/jalgaon_weather_history.csv")

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

# Pre-load data
try:
    df = pd.read_csv(DATA_FILE)
    df["month"] = pd.to_datetime(df["month"])
    weather_df = pd.read_csv(WEATHER_FILE)
    weather_df["Month_num"] = weather_df["Month"].astype(int)
except Exception as e:
    print("Warning: Could not load data files.", e)
    df = pd.DataFrame()
    weather_df = pd.DataFrame()

def fetch_weather_forecast():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
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
        
        temp = hourly.Variables(0).ValuesAsNumpy()
        humidity = hourly.Variables(1).ValuesAsNumpy()
        precip_prob = hourly.Variables(2).ValuesAsNumpy()
        rain = hourly.Variables(4).ValuesAsNumpy()
        wind_speed = hourly.Variables(5).ValuesAsNumpy()
        et0 = hourly.Variables(6).ValuesAsNumpy()
        soil_moisture = hourly.Variables(8).ValuesAsNumpy()
        
        avg_temp = float(np.nanmean(temp))
        avg_humidity = float(np.nanmean(humidity))
        total_rain = float(np.nansum(rain))
        avg_wind = float(np.nanmean(wind_speed))
        total_et0 = float(np.nansum(et0))
        avg_soil_moisture = float(np.nanmean(soil_moisture))
        max_precip_prob = float(np.nanmax(precip_prob))
        
        est_monthly_rainfall = total_rain * (30.0 / 14.0)
        
        daily_time = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        daily_list = []
        for i in range(len(daily_time)):
            daily_list.append({
                "Date": daily_time[i].strftime("%Y-%m-%d"),
                "Max Temp": float(daily.Variables(0).ValuesAsNumpy()[i]),
                "Min Temp": float(daily.Variables(1).ValuesAsNumpy()[i]),
                "Rain": float(daily.Variables(2).ValuesAsNumpy()[i]),
                "Rain Prob": float(daily.Variables(3).ValuesAsNumpy()[i]),
                "Wind Max": float(daily.Variables(4).ValuesAsNumpy()[i])
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
            "daily": daily_list
        }
        return metrics
    except Exception as e:
        print("Weather fetch error:", e)
        return None

def get_historical_weather_averages(w_df, month_num):
    m_data = w_df[w_df["Month_num"] == month_num]
    if m_data.empty:
        return 30.0, 50.0, 10.0
    return float(m_data["Temperature"].mean()), float(m_data["Humidity"].mean()), float(m_data["Rainfall"].mean())

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

@app.get("/api/health")
def get_health():
    return {
        "status": "ok",
        "app": "AgriSmart Backend",
        "version": "1.0.0",
        "crops_loaded": len(df["crop"].unique()) if not df.empty else 0
    }

@app.get("/api/weather")
def get_weather():
    data = fetch_weather_forecast()
    if not data:
        raise HTTPException(status_code=500, detail="Could not fetch weather data")
    return data

@app.get("/api/crops")
def get_crops():
    if df.empty:
        return []
    return sorted(df["crop"].unique().tolist())

@app.get("/api/crop-info/{crop}")
def get_crop_info(crop: str):
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded")
    history = df[df["crop"] == crop].sort_values("month")
    if history.empty:
        raise HTTPException(status_code=404, detail="Crop not found")
    latest_csv_date = history["month"].max()
    latest_price = float(history["avg_modal_price"].iloc[-1])
    default_price = CURRENT_MARKET_PRICES_2026.get(crop, latest_price)

    return {
        "crop": crop,
        "latest_date_formatted": latest_csv_date.strftime("%B %Y"),
        "latest_date_str": latest_csv_date.strftime("%Y-%m"),
        "latest_price": round(latest_price, 2),
        "default_market_price_2026": round(default_price, 2),
        "default_price_1_month_ago": round(default_price * 0.95, 2),
        "default_price_7_days_ago": round(default_price, 2),
        "current_year": datetime.today().year,
        "current_month": datetime.today().month
    }

class ForecastParams(BaseModel):
    crop: str
    target_year: int = 2026
    target_month: int = 12
    use_custom_data: bool = False
    custom_year: Optional[int] = None
    custom_month: Optional[int] = None
    price_1_month_ago: Optional[float] = None
    price_7_days_ago: Optional[float] = None
    custom_price: Optional[float] = None

def _compute_forecast(
    crop: str,
    target_year: int,
    target_month: int,
    use_custom_data: bool,
    custom_year: Optional[int],
    custom_month: Optional[int],
    price_1_month_ago: Optional[float],
    price_7_days_ago: Optional[float],
    custom_price: Optional[float]
):
    model_file = resolve_path(f"models/crop_price_weather_model_{crop}.pkl")
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail=f"Model for {crop} not found! Please run train_weather_model.py again.")
        
    pipeline = joblib.load(model_file)
    weather_metrics = fetch_weather_forecast()
    
    history = df[df["crop"] == crop].sort_values("month").copy()
    if history.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for {crop}")
        
    latest_csv_date = history["month"].max()

    if use_custom_data and custom_year and custom_month:
        default_price = CURRENT_MARKET_PRICES_2026.get(crop, float(history["avg_modal_price"].iloc[-1]))
        p_1m = price_1_month_ago if price_1_month_ago is not None else (default_price * 0.95)
        p_7d = price_7_days_ago if price_7_days_ago is not None else default_price
        p_curr = custom_price if custom_price is not None else default_price

        current_date_input = pd.Timestamp(year=custom_year, month=custom_month, day=1)
        prev_month_input = current_date_input - pd.DateOffset(months=1)
        last_qty = float(history["total_quantity"].iloc[-1])
        current_month_avg = (p_curr + p_7d) / 2.0

        new_rows = pd.DataFrame({
            "crop": [crop, crop],
            "month": [prev_month_input, current_date_input],
            "avg_modal_price": [p_1m, current_month_avg],
            "total_quantity": [last_qty, last_qty]
        })
        history = pd.concat([history, new_rows], ignore_index=True)
        latest_date = current_date_input
    else:
        # Default: Use only the authentic data available in the CSV dataset.
        latest_date = latest_csv_date
        
    target_date = pd.Timestamp(year=max(target_year, latest_date.year), month=target_month, day=1)
    
    if target_date <= latest_date:
        target_date = pd.Timestamp(year=latest_date.year, month=12, day=1)
        if target_date <= latest_date:
            target_date = latest_date + pd.DateOffset(months=2)
        
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
            
        future_features = create_future_features(history, crop, current_date, temp, hum, rain)
        current_price = history["avg_modal_price"].iloc[-1]
        predicted_ratio = pipeline.predict(future_features)[0]
        predicted_price = current_price * predicted_ratio
        
        forecast_results.append({
            "month": current_date.strftime("%Y-%m"),
            "price": float(round(predicted_price, 2)),
            "temp": float(round(temp, 1)),
            "rain": float(round(rain, 1)),
            "type": "Predicted"
        })
        
        latest_qty = history["total_quantity"].iloc[-1]
        new_row = pd.DataFrame({
            "crop": [crop],
            "month": [current_date],
            "avg_modal_price": [predicted_price],
            "total_quantity": [latest_qty]
        })
        history = pd.concat([history, new_row], ignore_index=True)
        current_date = current_date + pd.DateOffset(months=1)
        
    historical_results = []
    start_display_date = latest_date - pd.DateOffset(months=5)
    display_history = history[(history["month"] >= start_display_date) & (history["month"] <= latest_date)].copy()
    if display_history.empty:
        display_history = history[history["month"] <= latest_date].tail(6).copy()

    for _, row in display_history.iterrows():
        is_current = (row["month"] == latest_date and use_custom_data and row["month"].year >= 2026)
        historical_results.append({
            "month": row["month"].strftime("%Y-%m"),
            "price": float(round(row["avg_modal_price"], 2)),
            "temp": None,
            "rain": None,
            "type": "Current" if is_current else "Historical"
        })
        
    return {
        "crop": crop,
        "latest_date": latest_date.strftime("%Y-%m"),
        "target_date": target_date.strftime("%Y-%m"),
        "historical": historical_results,
        "forecast": forecast_results,
        "warning": None
    }

@app.post("/api/forecast")
def post_forecast(params: ForecastParams):
    return _compute_forecast(
        crop=params.crop,
        target_year=params.target_year,
        target_month=params.target_month,
        use_custom_data=params.use_custom_data,
        custom_year=params.custom_year,
        custom_month=params.custom_month,
        price_1_month_ago=params.price_1_month_ago,
        price_7_days_ago=params.price_7_days_ago,
        custom_price=params.custom_price
    )

@app.get("/api/forecast/{crop}")
def get_forecast(
    crop: str,
    target_year: int = 2026,
    target_month: int = 12,
    use_custom_data: bool = False,
    custom_year: Optional[int] = None,
    custom_month: Optional[int] = None,
    price_1_month_ago: Optional[float] = None,
    price_7_days_ago: Optional[float] = None,
    custom_price: Optional[float] = None
):
    return _compute_forecast(
        crop=crop,
        target_year=target_year,
        target_month=target_month,
        use_custom_data=use_custom_data,
        custom_year=custom_year,
        custom_month=custom_month,
        price_1_month_ago=price_1_month_ago,
        price_7_days_ago=price_7_days_ago,
        custom_price=custom_price
    )

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    crop: str
    messages: List[ChatMessage]
    forecast_text: str
    weather_text: str
    language: str = "English"
    api_key: Optional[str] = None

@app.post("/api/chat")
def chat(request: ChatRequest, api_key: Optional[str] = None):
    key = request.api_key or api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(
            status_code=401,
            detail="Groq API key not provided. Please enter your Groq API key in Settings or add GROQ_API_KEY to your backend .env file."
        )
        
    try:
        client = Groq(api_key=key)
        
        if request.language == "मराठी":
            system_prompt = f"""तुम्ही एक तज्ज्ञ कृषी सल्लागार आहात. वापरकर्ता या पिकाबद्दल विचारत आहे: {request.crop}.
अपेक्षित भाव (Forecasted prices): {request.forecast_text}
पुढील १४ दिवसांचे हवामान: {request.weather_text}
शेतकऱ्याला थेट आणि संक्षिप्त स्वरूपात मदत करणारा प्रतिसाद द्या. तुमचा संपूर्ण प्रतिसाद मराठी भाषेत (in Marathi) असणे आवश्यक आहे. वापरकर्त्याला नेहमी थोडक्यात आठवण करून द्या की हे केवळ अंदाजित भाव आहेत आणि वास्तविक जीवनातील भावांची हमी देत नाहीत."""
        else:
            system_prompt = f"""You are an expert agricultural advisor. The user is asking about {request.crop}.
Forecasted prices: {request.forecast_text}
Live 14-day weather: {request.weather_text}
Provide a concise and helpful response directly to the farmer in English. Always briefly remind the user that these are predicted prices and not guaranteed real-life prices."""
        
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in request.messages:
            api_messages.append({"role": m.role, "content": m.content})
            
        models_to_try = ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        last_error = None
        response_text = None

        for model_name in models_to_try:
            try:
                chat_completion = client.chat.completions.create(
                    messages=api_messages,
                    model=model_name,
                    temperature=0.5,
                )
                response_text = chat_completion.choices[0].message.content
                break
            except Exception as e:
                last_error = e
                continue

        if response_text is not None:
            return {"response": response_text}
        else:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {last_error}")

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
