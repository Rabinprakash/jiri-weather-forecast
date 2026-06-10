import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("jiri_weather_2010_2026_final.csv")

print("Original Shape:", df.shape)

# Remove rows with NaN values
df = df.dropna().reset_index(drop=True)

print("After DropNA:", df.shape)

# ==========================================
# FEATURES
# ==========================================

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

target = "temp_target"

# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df[features].isna().sum())

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================
# MODEL
# ==========================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# ==========================================
# TRAIN
# ==========================================

print("\nTraining Model...")

model.fit(X_train, y_train)

# ==========================================
# PREDICT
# ==========================================

pred = model.predict(X_test)

# ==========================================
# METRICS
# ==========================================

mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

r2 = r2_score(y_test, pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "xgb_model.pkl")
joblib.dump(features, "xgb_features.pkl")

print("\nSaved:")
print("  xgb_model.pkl")
print("  xgb_features.pkl")

# ==========================================
# SAVE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": pred
})

results.to_csv(
    "xgb_predictions.csv",
    index=False
)

print("Saved: xgb_predictions.csv")

# ==========================================
# PLOT ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values[:200],
    label="Actual"
)

plt.plot(
    pred[:200],
    label="Predicted"
)

plt.title(
    "XGBoost Temperature Forecast - Jiri, Dolakha, Nepal"
)

plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

plt.figure(figsize=(10, 8))

plot_importance(
    model,
    max_num_features=15
)

plt.title("Top Feature Importance")

plt.tight_layout()
plt.show()

# ==========================================
# SAVE FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("Saved: feature_importance.csv")

print("\nTop 10 Features:")
print(importance.head(10))
