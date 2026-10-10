# AgriSmart Forecast 🌾

AgriSmart is an AI-powered agricultural forecasting tool designed to help farmers make data-driven decisions. By analyzing historical crop prices, tracking real-time meteorological data (Open-Meteo), and employing machine learning (Scikit-Learn Random Forest), AgriSmart provides robust multi-month crop price predictions. 

Additionally, AgriSmart integrates with Groq's high-speed AI to offer conversational agronomic advice based directly on the predicted prices and 14-day weather forecasts!

## Features
- **Machine Learning Price Forecasting:** Predicts the price of 10 different major crops using historical pricing paired with lag momentum and seasonal cyclical features.
- **Live Weather Integration:** Fetches real-time 14-day weather forecasts from Open-Meteo (Temperature, Rainfall, Humidity, ET₀, Soil Moisture, Wind).
- **Interactive Data Visualizations:** Clean Pandas Dataframes and Streamlit Charts that plot price trends dynamically.
- **Farm AI Assistant:** A built-in LLM chat assistant powered by Groq and Qwen that provides personalized, context-aware advice to farmers based on their specific crop, predicted prices, and incoming weather.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd AgriSmart_App
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

4. **(Optional) Train the Machine Learning models:**
   If the `models/` directory is missing or you want to retrain the models with new data, run:
   ```bash
   python train_weather_model.py
   ```

## Running the App

Start the Streamlit application by running:
```bash
streamlit run app.py
```
This will open the app locally in your default web browser.

## File Structure
- `app.py`: The main Streamlit web application.
- `train_weather_model.py`: Script to train the Random Forest models for all crops.
- `live_forecast.py`: CLI tool for testing real-time predictions in the terminal.
- `data/`: Contains historical CSV datasets for model training and baseline referencing.
- `models/`: Stores the trained `.pkl` Scikit-Learn models.
