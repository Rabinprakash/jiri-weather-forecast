import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("jiri_weather_2010_2026_final.csv")

features = [
    "temp",
    "humidity",
    "rainfall",
    "wind_speed"
]

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

# =====================================
# SCALE FEATURES
# =====================================

scaler = MinMaxScaler()

scaled = scaler.fit_transform(df[features])

# =====================================
# CREATE SEQUENCES
# 30 previous days -> next day temperature
# =====================================

SEQ_LEN = 30

X = []
y = []

for i in range(SEQ_LEN, len(scaled)):
    X.append(scaled[i-SEQ_LEN:i])

    # predict next-day temperature
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =====================================
# TRAIN / TEST SPLIT
# =====================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =====================================
# MODEL
# =====================================

model = Sequential([
    LSTM(
        64,
        input_shape=(SEQ_LEN, len(features))
    ),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# =====================================
# TRAIN
# =====================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# =====================================
# PREDICT
# =====================================

pred = model.predict(X_test)

# =====================================
# CONVERT BACK TO CELSIUS
# =====================================

dummy_pred = np.zeros((len(pred), 4))
dummy_pred[:, 0] = pred.flatten()

pred_temp = scaler.inverse_transform(dummy_pred)[:, 0]

dummy_actual = np.zeros((len(y_test), 4))
dummy_actual[:, 0] = y_test

actual_temp = scaler.inverse_transform(dummy_actual)[:, 0]

# =====================================
# METRICS
# =====================================

mae = mean_absolute_error(actual_temp, pred_temp)

rmse = np.sqrt(
    mean_squared_error(actual_temp, pred_temp)
)

print("\nResults")
print("MAE :", mae)
print("RMSE:", rmse)
model.save("jiri_lstm_model.keras")

print("Model saved: jiri_lstm_model.keras")

joblib.dump(scaler, "scaler_lstm.pkl")

print("Scaler saved: scaler_lstm.pkl")
