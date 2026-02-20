# 🚦 AI Traffic Monitoring & Prediction System

A computer vision-based traffic analysis system that detects, tracks,
and analyzes vehicles from video footage using YOLOv8 and LSTM.

------------------------------------------------------------------------

## 📌 Project Overview

This project performs:

-   🚗 Vehicle detection using YOLOv8\
-   🆔 Object tracking with built-in YOLO tracker\
-   📏 Real-time speed estimation\
-   ⚠️ Overspeed detection with image capture\
-   📊 Traffic density classification\
-   🤖 LSTM-based traffic prediction\
-   📝 Automatic CSV logging

The system processes recorded road footage and displays real-time
analytics on screen.

------------------------------------------------------------------------

## 🛠 Technologies Used

-   Python\
-   OpenCV\
-   Ultralytics YOLOv8\
-   TensorFlow / Keras\
-   NumPy\
-   Pandas\
-   Scikit-learn

------------------------------------------------------------------------

## 📂 Project Structure

    Traffic_analysis-system/
    │
    ├── vehicle_detect.py        # Main traffic monitoring system
    ├── traffic_predict.py       # Rule-based traffic classification + logging
    ├── train_traffic_model.py   # LSTM training script
    ├── traffic_lstm_model.h5    # Saved trained LSTM model
    ├── traffic_scaler.pkl       # Saved MinMax scaler
    ├── traffic.csv              # Dataset used for training
    ├── traffic_log.csv          # Generated traffic logs
    ├── yolov8n.pt               # YOLOv8 pretrained model
    └── violations/              # Overspeed vehicle captures

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

    git clone https://github.com/your-username/Traffic_analysis-system.git
    cd Traffic_analysis-system

Install required packages:

    pip install opencv-python ultralytics tensorflow scikit-learn pandas joblib

------------------------------------------------------------------------

## ▶️ How to Run

### 1️⃣ Train LSTM Model (Optional)

If you want to retrain the prediction model:

    python train_traffic_model.py

This generates:

-   `traffic_lstm_model.h5`
-   `traffic_scaler.pkl`

------------------------------------------------------------------------

### 2️⃣ Run Traffic Monitoring System

    python vehicle_detect.py

Press **Q** to exit.

------------------------------------------------------------------------

## 📊 System Features

### 🚗 Vehicle Detection

Uses YOLOv8 Nano model for real-time object detection.

### 📏 Speed Estimation

Speed is calculated using pixel displacement between frames and
calibrated conversion to meters.

### ⚠️ Overspeed Detection

Vehicles exceeding the speed limit are: - Marked on screen - Saved in
`violations/` folder

### 📈 Traffic Classification

Rule-based classification: - FREE ROAD - MODERATE TRAFFIC - HEAVY
TRAFFIC

### 🤖 LSTM Prediction

Predicts next traffic count based on previous time sequence data.

------------------------------------------------------------------------

## 🧠 How Speed is Calculated

1.  Vehicle centroid is tracked across frames\
2.  Vertical pixel displacement is measured\
3.  Pixels converted to meters\
4.  Speed calculated using FPS timing

Calibration factor:

    pixels_per_meter = 40  # Adjust based on camera setup

------------------------------------------------------------------------

## 📌 Future Improvements

-   Perspective transformation for more accurate speed
-   Lane-wise analytics
-   Real-time dashboard (Streamlit / Flask)
-   Live webcam integration
-   Advanced congestion forecasting model

------------------------------------------------------------------------

## 👨‍💻 Author

Developed as a Computer Vision and AI project focused on traffic
analytics and predictive modeling.

------------------------------------------------------------------------

## ⭐ If You Found This Useful

Consider giving the repository a star and exploring improvements.
