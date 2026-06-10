import pandas as pd
import numpy as np
import joblib

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv("jiri_weather_2010_2026_final.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.dropna().reset_index(drop=True)

# ====================================
# LOAD MODEL
# ====================================

model = joblib.load("xgb_model.pkl")

# ====================================
# FEATURES
# ====================================

features = [
    "temp",
    "temp_max",
    "temp_min",
    "humidity",
    "rainfall",
    "wind_speed",

    "temp_lag1",
    "temp_lag3",
    "temp_lag7",
    "temp_lag14",
    "temp_lag30",

    "rain_lag1",
    "rain_lag3",
    "rain_lag7",
    "rain_lag14",
    "rain_lag30",

    "temp_roll7",
    "rain_roll7",

    "month",
    "dayofyear"
]

# ====================================
# LAST ROW
# ====================================

current = df.iloc[-1].copy()

future_predictions = []

# ====================================
# FORECAST LOOP
# ====================================

for _ in range(30):

    X_future = pd.DataFrame(
        [current[features]]
    )

    pred = model.predict(X_future)[0]

    future_predictions.append(pred)

    # Update lags
    current["temp_lag30"] = current["temp_lag14"]
    current["temp_lag14"] = current["temp_lag7"]
    current["temp_lag7"] = current["temp_lag3"]
    current["temp_lag3"] = current["temp_lag1"]
    current["temp_lag1"] = current["temp"]

    current["temp"] = pred

    # Update date
    current["date"] += pd.Timedelta(days=1)

    current["month"] = current["date"].month
    current["dayofyear"] = current["date"].dayofyear

# ====================================
# CREATE DATES
# ====================================

future_dates = pd.date_range(
    start=df["date"].max() + pd.Timedelta(days=1),
    periods=30
)

# ====================================
# SAVE
# ====================================

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_temp": future_predictions
})

forecast_df.to_csv(
    "xgb_30day_forecast.csv",
    index=False
)

print(forecast_df)
