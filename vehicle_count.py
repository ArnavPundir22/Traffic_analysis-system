import cv2
import math
import datetime
import os
import time
from ultralytics import YOLO
from sort import Sort

# ----------------------------
# Load Models
# ----------------------------
model = YOLO("yolov8n.pt")
tracker = Sort()

cap = cv2.VideoCapture("road.mp4")

# ----------------------------
# Settings
# ----------------------------
vehicle_classes = ["car", "truck", "bus", "motorbike"]

line_y = 400
count = 0
counted_ids = set()

# 🔥 IMPORTANT: CALIBRATE THIS VALUE
pixels_per_meter = 20   # Adjust based on your video
speed_limit = 60        # km/h

previous_positions = {}
vehicle_speeds = {}

os.makedirs("violations", exist_ok=True)

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

    tracked_objects = tracker.update(detections)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        x1, y1, x2, y2, obj_id = int(x1), int(y1), int(x2), int(y2), int(obj_id)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # ----------------------------
        # Vehicle Counting
        # ----------------------------
        if line_y - 10 < cy < line_y + 10:
            if obj_id not in counted_ids:
                count += 1
                counted_ids.add(obj_id)

        # ----------------------------
        # SPEED CALCULATION (Time-Based)
        # ----------------------------
        current_time = time.time()

        if obj_id in previous_positions:
            prev_x, prev_y, prev_time = previous_positions[obj_id]
            time_diff = current_time - prev_time

            if time_diff > 0:
                pixel_distance = math.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                meters = pixel_distance / pixels_per_meter
                speed_mps = meters / time_diff
                speed_kmph = speed_mps * 3.6

                # Smooth speed
                if obj_id in vehicle_speeds:
                    speed_kmph = (vehicle_speeds[obj_id] + speed_kmph) / 2

                vehicle_speeds[obj_id] = speed_kmph

                cv2.putText(frame,
                            f"{int(speed_kmph)} km/h",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2)

                # ----------------------------
                # OVERSPEED DETECTION
                # ----------------------------
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

    cv2.putText(frame, f"Count: {count}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3)

    cv2.imshow("Smart Traffic Monitoring", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
