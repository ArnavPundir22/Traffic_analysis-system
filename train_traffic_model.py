import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import save_model

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("traffic.csv")
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.sort_values("DateTime")

# We only predict Vehicles using time sequence
vehicle_data = data["Vehicles"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(vehicle_data)

# -------------------------------
# Create Sequences
# -------------------------------
SEQUENCE_LENGTH = 5  # use last 5 hours to predict next

X = []
y = []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# -------------------------------
# Build LSTM Model
# -------------------------------
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=20, batch_size=16)

# -------------------------------
# Save Model + Scaler
# -------------------------------
model.save("traffic_lstm_model.h5")
joblib.dump(scaler, "traffic_scaler.pkl")

print("✅ LSTM model trained and saved!")
