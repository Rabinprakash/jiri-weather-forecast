import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Jiri Weather Forecast",
    layout="wide"
)

st.title("🌦️ Jiri Weather Forecast Dashboard (30 Days)")
st.write("Comparison of XGBoost vs LSTM predictions")

# ==========================
# LOAD FORECAST FILES
# ==========================

@st.cache_data
def load_data():
    xgb = pd.read_csv("xgb_30day_forecast.csv")
    lstm = pd.read_csv("lstm_30day_forecast.csv")

    xgb["date"] = pd.to_datetime(xgb["date"])
    lstm["date"] = pd.to_datetime(lstm["date"])

    xgb.rename(columns={"forecast_temp": "XGBoost"}, inplace=True)
    lstm.rename(columns={"forecast_temp": "LSTM"}, inplace=True)

    df = pd.merge(xgb, lstm, on="date")

    return df

df = load_data()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Options")

show_data = st.sidebar.checkbox("Show Forecast Table", True)

model_choice = st.sidebar.radio(
    "Select Model View",
    ["Both", "XGBoost", "LSTM"]
)

# ==========================
# DATA TABLE
# ==========================

if show_data:
    st.subheader("📊 Forecast Data")
    st.dataframe(df)

# ==========================
# PLOT
# ==========================

st.subheader("📈 30-Day Forecast Graph")

fig, ax = plt.subplots(figsize=(12,6))

if model_choice in ["Both", "XGBoost"]:
    ax.plot(df["date"], df["XGBoost"], label="XGBoost", linewidth=2)

if model_choice in ["Both", "LSTM"]:
    ax.plot(df["date"], df["LSTM"], label="LSTM", linewidth=2)

ax.set_title("Temperature Forecast Comparison - Jiri, Nepal")
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(True)

plt.xticks(rotation=45)

st.pyplot(fig)

# ==========================
# SUMMARY STATS
# ==========================

st.subheader("📌 Quick Statistics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Avg XGBoost Temp", f"{df['XGBoost'].mean():.2f} °C")

with col2:
    st.metric("Avg LSTM Temp", f"{df['LSTM'].mean():.2f} °C")

# ==========================
# DOWNLOAD
# ==========================

st.subheader("⬇️ Download Forecast")

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download CSV",
    csv,
    "weather_forecast_30_days.csv",
    "text/csv"
)
