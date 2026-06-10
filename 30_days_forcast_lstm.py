import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv("jiri_weather_2010_2026_final.csv")

features = [
    "temp",
    "humidity",
    "rainfall",
    "wind_speed"
]

# ====================================
# LOAD MODEL + SCALER
# ====================================

model = load_model("jiri_lstm_model.keras")

scaler = joblib.load("scaler_lstm.pkl")

# ====================================
# PREPARE LAST 30 DAYS
# ====================================

scaled = scaler.transform(df[features])

current_seq = scaled[-30:]

# ====================================
# FORECAST 30 DAYS
# ====================================

future_preds = []

for _ in range(30):

    pred = model.predict(
        current_seq.reshape(1, 30, 4),
        verbose=0
    )[0][0]

    future_preds.append(pred)

    new_row = current_seq[-1].copy()

    # Replace temperature
    new_row[0] = pred

    current_seq = np.vstack([
        current_seq[1:],
        new_row
    ])

# ====================================
# CONVERT BACK TO CELSIUS
# ====================================

dummy = np.zeros((30, 4))

dummy[:, 0] = future_preds

forecast_temp = scaler.inverse_transform(dummy)[:, 0]

# ====================================
# DATES
# ====================================

last_date = pd.to_datetime(
    df["date"].iloc[-1]
)

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=30
)

# ====================================
# SAVE
# ====================================

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_temp": forecast_temp
})

forecast_df.to_csv(
    "lstm_30day_forecast.csv",
    index=False
)

print(forecast_df)
