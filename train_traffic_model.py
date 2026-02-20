import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data = pd.read_csv("traffic.csv")

# Convert DateTime column properly
data["DateTime"] = pd.to_datetime(data["DateTime"])
data = data.sort_values("DateTime")

# We are only using vehicle count for prediction
vehicle_data = data["Vehicles"].values.reshape(-1, 1)

# Scale values between 0 and 1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(vehicle_data)

sequence_length = 5  # number of previous time steps used

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Split into train and test (simple split)
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(sequence_length, 1)))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Evaluate quickly
loss = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)

# Save model and scaler
model.save("traffic_lstm_model.h5")
joblib.dump(scaler, "traffic_scaler.pkl")

print("Model training complete and saved.")
