import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def prepare_data():
    crop_file = "raw_data/monthly_joined/Jalgaon_all_crops_monthly.csv"
    weather_file = "jalgaon_weather_history.csv"
    
    # Load crop prices
    df = pd.read_csv(crop_file)
    df["month_dt"] = pd.to_datetime(df["month"])
    df["Year"] = df["month_dt"].dt.year
    df["Month_num"] = df["month_dt"].dt.month
    
    # Load weather data
    if os.path.exists(weather_file):
        df_weather = pd.read_csv(weather_file)
        df_weather["Month_num"] = df_weather["Month"].astype(int)
        df_weather["Year"] = df_weather["Year"].astype(int)
        
        # Merge weather data
        df = pd.merge(df, df_weather, on=["Year", "Month_num"], how="left")
    else:
        df["Temperature"] = np.nan
        df["Humidity"] = np.nan
        df["Rainfall"] = np.nan
        
    df = df.sort_values(["crop", "month_dt"]).reset_index(drop=True)
    
    # Cyclical month features
    df["month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12)
    
    # We must use RELATIVE features so the model can extrapolate to ANY price level (e.g. 9000 in 2026)
    # Ratios = (Lagged Value) / (Current Value)
    
    for lag in [1, 2, 3, 6, 12]:
        df[f"price_lag_{lag}_ratio"] = df.groupby("crop")["avg_modal_price"].shift(lag) / df["avg_modal_price"]
        # Replace infs or zeros
        df[f"price_lag_{lag}_ratio"] = df[f"price_lag_{lag}_ratio"].replace([np.inf, -np.inf], np.nan)
        
        df[f"quantity_lag_{lag}_ratio"] = df.groupby("crop")["total_quantity"].shift(lag) / df["total_quantity"]
        df[f"quantity_lag_{lag}_ratio"] = df[f"quantity_lag_{lag}_ratio"].replace([np.inf, -np.inf], np.nan)
        
    df["price_rolling_mean_3_ratio"] = df.groupby("crop")["avg_modal_price"].transform(lambda x: x.shift(1).rolling(3).mean()) / df["avg_modal_price"]
    df["price_rolling_mean_6_ratio"] = df.groupby("crop")["avg_modal_price"].transform(lambda x: x.shift(1).rolling(6).mean()) / df["avg_modal_price"]
    df["price_rolling_mean_12_ratio"] = df.groupby("crop")["avg_modal_price"].transform(lambda x: x.shift(1).rolling(12).mean()) / df["avg_modal_price"]
    
    df["quantity_rolling_mean_3_ratio"] = df.groupby("crop")["total_quantity"].transform(lambda x: x.shift(1).rolling(3).mean()) / df["total_quantity"]
    df["quantity_rolling_mean_6_ratio"] = df.groupby("crop")["total_quantity"].transform(lambda x: x.shift(1).rolling(6).mean()) / df["total_quantity"]
    
    # Target is the ratio of NEXT month's price to THIS month's price
    df["target_price"] = df.groupby("crop")["avg_modal_price"].shift(-1)
    df["target_ratio"] = df["target_price"] / df["avg_modal_price"]
    df["target_ratio"] = df["target_ratio"].replace([np.inf, -np.inf], np.nan)
    
    return df

def train_individual_models():
    print("Preparing dataset...")
    df = prepare_data()
    
    features = [
        "Month_num", "month_sin", "month_cos",
        "price_lag_1_ratio", "price_lag_2_ratio", "price_lag_3_ratio", "price_lag_6_ratio", "price_lag_12_ratio",
        "quantity_lag_1_ratio", "quantity_lag_2_ratio", "quantity_lag_3_ratio", "quantity_lag_6_ratio", "quantity_lag_12_ratio",
        "price_rolling_mean_3_ratio", "price_rolling_mean_6_ratio", "price_rolling_mean_12_ratio",
        "quantity_rolling_mean_3_ratio", "quantity_rolling_mean_6_ratio",
        "Temperature", "Humidity", "Rainfall"
    ]
    
    os.makedirs("AgriSmart_App/models", exist_ok=True)
    
    crops = df["crop"].unique()
    
    for crop in crops:
        print(f"\n{'='*50}\nTraining RELATIVE model for {crop}...\n{'='*50}")
        
        df_crop = df[df["crop"] == crop].copy()
        df_model = df_crop.dropna(subset=features + ["target_ratio"]).copy()
        
        if len(df_model) < 20:
            print(f"Skipping {crop} due to insufficient data ({len(df_model)} rows).")
            continue
            
        train_mask = df_model["month_dt"] < "2022-01-01"
        test_mask = df_model["month_dt"] >= "2022-01-01"
        
        X_train = df_model.loc[train_mask, features]
        y_train = df_model.loc[train_mask, "target_ratio"]
        X_test = df_model.loc[test_mask, features]
        y_test = df_model.loc[test_mask, "target_ratio"]
        test_actual_prices = df_model.loc[test_mask, "target_price"]
        current_prices = df_model.loc[test_mask, "avg_modal_price"]
        
        # Use Random Forest on RATIOS. It can predict ratios like 1.05 or 0.95 easily.
        model = RandomForestRegressor(
            n_estimators=300, 
            max_depth=10, 
            min_samples_leaf=3, 
            random_state=42, 
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        if len(X_test) > 0:
            pred_ratios = model.predict(X_test)
            pred_prices = current_prices * pred_ratios
            
            mae = mean_absolute_error(test_actual_prices, pred_prices)
            rmse = np.sqrt(mean_squared_error(test_actual_prices, pred_prices))
            print(f"Test MAE : ₹{mae:,.2f}")
            print(f"Test RMSE: ₹{rmse:,.2f}")
            
        model_filename = f"AgriSmart_App/models/crop_price_weather_model_{crop}.pkl"
        joblib.dump(model, model_filename)
        print(f"✅ Saved model to '{model_filename}'")

if __name__ == "__main__":
    train_individual_models()
