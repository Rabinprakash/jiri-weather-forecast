import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# LOAD FORECASTS
# ==========================

lstm = pd.read_csv("lstm_30day_forecast.csv")
xgb = pd.read_csv("xgb_30day_forecast.csv")

# rename columns
lstm.rename(
    columns={"forecast_temp": "LSTM"},
    inplace=True
)

xgb.rename(
    columns={"forecast_temp": "XGBoost"},
    inplace=True
)

# merge by date
forecast = pd.merge(
    lstm,
    xgb,
    on="date"
)

print(forecast.head())

# save combined results
forecast.to_csv(
    "forecast_comparison.csv",
    index=False
)

# ==========================
# PLOT
# ==========================

plt.figure(figsize=(14,6))

plt.plot(
    forecast["date"],
    forecast["LSTM"],
    label="LSTM Forecast",
    linewidth=2
)

plt.plot(
    forecast["date"],
    forecast["XGBoost"],
    label="XGBoost Forecast",
    linewidth=2
)

plt.title(
    "30-Day Temperature Forecast Comparison\nJiri, Dolakha, Nepal"
)

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.legend()

plt.tight_layout()

plt.show()
