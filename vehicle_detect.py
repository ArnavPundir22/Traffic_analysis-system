import cv2
import datetime
import numpy as np
import joblib
import math
import time
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from traffic_predict import traffic_status, save_traffic_log

# -------------------------------
# Load Models
# -------------------------------
model = YOLO("yolov8n.pt")

lstm_model = load_model("traffic_lstm_model.h5", compile=False)
scaler = joblib.load("traffic_scaler.pkl")

cap = cv2.VideoCapture("road2.mp4")

vehicle_classes = ["car", "truck", "bus", "motorbike"]

# -------------------------------
# Speed Settings
# -------------------------------
pixels_per_meter = 20   # 🔥 Calibrate properly
speed_limit = 60        # km/h

previous_positions = {}
vehicle_speeds = {}

os.makedirs("violations", exist_ok=True)

# -------------------------------
# LSTM Settings
# -------------------------------
SEQUENCE_LENGTH = 5
recent_counts = []

last_log_time = datetime.datetime.now()

# -------------------------------
# Main Loop
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 600))

    # YOLO built-in tracking
    results = model.track(frame, persist=True, verbose=False)

    vehicle_count = 0
    speed_values = []

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):

            label = model.names[int(cls)]

            if label in vehicle_classes:

                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box)
                obj_id = int(obj_id)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                current_time = time.time()

                # -------------------------------
                # Speed Calculation
                # -------------------------------
                if obj_id in previous_positions:

                    prev_x, prev_y, prev_time = previous_positions[obj_id]
                    time_diff = current_time - prev_time

                    if time_diff > 0:
                        pixel_distance = math.sqrt(
                            (cx - prev_x)**2 + (cy - prev_y)**2
                        )

                        meters = pixel_distance / pixels_per_meter
                        speed_mps = meters / time_diff
                        speed_kmph = speed_mps * 3.6

                        # Smooth speed
                        if obj_id in vehicle_speeds:
                            speed_kmph = (
                                vehicle_speeds[obj_id] + speed_kmph
                            ) / 2

                        vehicle_speeds[obj_id] = speed_kmph
                        speed_values.append(speed_kmph)

                        cv2.putText(frame,
                                    f"{int(speed_kmph)} km/h",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    2)

                        # Overspeed detection
                        if speed_kmph > speed_limit:
                            cv2.putText(frame,
                                        "OVERSPEED!",
                                        (x1, y1 - 35),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (0, 0, 255),
                                        3)

                            timestamp = datetime.datetime.now().strftime("%H%M%S")
                            cv2.imwrite(
                                f"violations/vehicle_{obj_id}_{timestamp}.jpg",
                                frame
                            )

                previous_positions[obj_id] = (cx, cy, current_time)

    # -------------------------------
    # Speed Meter Display
    # -------------------------------
    avg_speed = int(np.mean(speed_values)) if speed_values else 0
    max_speed = int(np.max(speed_values)) if speed_values else 0

    cv2.putText(frame, f"Vehicles: {vehicle_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)

    cv2.putText(frame, f"Avg Speed: {avg_speed} km/h",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.putText(frame, f"Max Speed: {max_speed} km/h",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    # -------------------------------
    # Rule-based Traffic
    # -------------------------------
    status = traffic_status(vehicle_count)

    # -------------------------------
    # LSTM Prediction
    # -------------------------------
    recent_counts.append(vehicle_count)

    if len(recent_counts) > SEQUENCE_LENGTH:
        recent_counts.pop(0)

    predicted_value = 0

    if len(recent_counts) == SEQUENCE_LENGTH:

        input_seq = np.array(recent_counts).reshape(-1, 1)
        input_scaled = scaler.transform(input_seq)
        input_scaled = np.reshape(input_scaled, (1, SEQUENCE_LENGTH, 1))

        prediction_scaled = lstm_model.predict(input_scaled, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)

        predicted_value = int(prediction[0][0])

    cv2.putText(frame, f"Traffic: {status}",
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2)

    cv2.putText(frame, f"Predicted Next: {predicted_value}",
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2)

    # -------------------------------
    # Save Log
    # -------------------------------
    now = datetime.datetime.now()

    if (now - last_log_time).seconds >= 5:
        save_traffic_log(vehicle_count)
        last_log_time = now

    cv2.imshow("AI Traffic Monitoring - YOLO Tracker", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
