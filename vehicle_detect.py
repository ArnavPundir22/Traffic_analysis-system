import cv2
import datetime
import numpy as np
import joblib
import math
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from traffic_predict import traffic_status, save_traffic_log


# Load YOLO model
model = YOLO("yolov8s.pt")

# Load trained LSTM + scaler
lstm_model = load_model("traffic_lstm_model.h5", compile=False)
scaler = joblib.load("traffic_scaler.pkl")

cap = cv2.VideoCapture("road.mp4")

# FPS for speed calculation
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
frame_time = 1 / fps

vehicle_classes = ["car", "truck", "bus", "motorcycle"]

# You may need to adjust this based on your camera
pixels_per_meter = 40
speed_limit = 60

previous_positions = {}
vehicle_speeds = {}

os.makedirs("violations", exist_ok=True)

SEQUENCE_LENGTH = 5
recent_counts = []
last_log_time = datetime.datetime.now()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (900, 600))
    results = model.track(frame, persist=True, verbose=False)

    vehicle_count = 0
    speed_values = []

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, obj_id, cls in zip(boxes, ids, classes):

            label = model.names[int(cls)]

            if label not in vehicle_classes:
                continue

            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box)
            obj_id = int(obj_id)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # Speed calculation
            if obj_id in previous_positions:
                prev_x, prev_y = previous_positions[obj_id]

                pixel_distance = abs(cy - prev_y)
                meters = pixel_distance / pixels_per_meter
                speed_mps = meters / frame_time
                speed_kmph = speed_mps * 3.6

                # simple smoothing
                if obj_id in vehicle_speeds:
                    speed_kmph = (vehicle_speeds[obj_id] * 0.7 +
                                  speed_kmph * 0.3)

                vehicle_speeds[obj_id] = speed_kmph
                speed_values.append(speed_kmph)

                cv2.putText(frame,
                            f"{int(speed_kmph)} km/h",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0), 2)

                if speed_kmph > speed_limit:
                    cv2.putText(frame, "OVERSPEED",
                                (x1, y1 - 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 3)

                    timestamp = datetime.datetime.now().strftime("%H%M%S")
                    cv2.imwrite(f"violations/{obj_id}_{timestamp}.jpg", frame)

            previous_positions[obj_id] = (cx, cy)

    # Basic stats display
    avg_speed = int(np.mean(speed_values)) if speed_values else 0
    max_speed = int(np.max(speed_values)) if speed_values else 0

    cv2.putText(frame, f"Vehicles: {vehicle_count}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    cv2.putText(frame, f"Avg Speed: {avg_speed} km/h",
                (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Max Speed: {max_speed} km/h",
                (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    # Rule-based traffic status
    status = traffic_status(vehicle_count)

    # LSTM prediction
    recent_counts.append(vehicle_count)
    if len(recent_counts) > SEQUENCE_LENGTH:
        recent_counts.pop(0)

    predicted_value = 0
    if len(recent_counts) == SEQUENCE_LENGTH:
        seq = np.array(recent_counts).reshape(-1, 1)
        seq_scaled = scaler.transform(seq)
        seq_scaled = np.reshape(seq_scaled, (1, SEQUENCE_LENGTH, 1))

        pred_scaled = lstm_model.predict(seq_scaled, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        predicted_value = int(pred[0][0])

    cv2.putText(frame, f"Traffic: {status}",
                (30, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 0), 2)

    cv2.putText(frame, f"Predicted Next: {predicted_value}",
                (30, 220), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)

    # Save log every 5 seconds
    now = datetime.datetime.now()
    if (now - last_log_time).seconds >= 5:
        save_traffic_log(vehicle_count)
        last_log_time = now

    cv2.imshow("Traffic Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
