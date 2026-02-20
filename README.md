<p align="center">
  <img src="assets/banner.png" alt="Handwritten Digit Recognition Banner" width="100%">
</p>

# 📊 Traffic Analysis System

![Traffic Banner](A_banner_image_for_a_"Traffic_Analysis_System"_is_.png)

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

A simple **Python‑based traffic analysis project** that detects and counts vehicles in a video feed and predicts traffic conditions using YOLOv8 and basic traffic logic.

---

## 🧠 Features

✔ **Detect vehicles** in video frames using YOLOv8  
✔ **Count vehicles** crossing a line  
✔ **Predict traffic condition** (Free, Moderate, Heavy)  
✔ **(Optional) Log traffic data** into CSV  
✔ Includes YOLO model (`yolov8n.pt`)  

---

## 📦 Contents

| File | Purpose |
|------|---------|
| `vehicle_detect.py` | Runs vehicle detection + traffic prediction display |
| `vehicle_count.py` | Tracks and counts vehicles |
| `traffic_predict.py` | Traffic status logic + optional logging |
| `traffic_log.csv` | Log of past traffic data |
| `yolov8n.pt` | YOLOv8 pretrained weights |
| `.gitignore` / `.gitattributes` | Git configuration |

---

## 🚀 Setup & Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/ArnavPundir22/Traffic_analysis-system.git
   cd Traffic_analysis-system
   ```

2. **Create a Python virtual environment (optional but recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install opencv-python ultralytics sort-python
   ```

---

## ▶️ How to Use

### 1. 🚘 Vehicle Detection + Traffic Status
```bash
python vehicle_detect.py
```

---

### 2. 🧮 Count Vehicles
```bash
python vehicle_count.py
```

---

### 3. 🚦 Traffic Prediction Logic
```bash
python traffic_predict.py
```

---

## 🛠 Requirements

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.8 |
| OpenCV      | Installed via pip |
| YOLOv8      | Ultralytics YOLO |
| SORT Tracker| For tracking |

---

## 📄 License

MIT License — You are free to use/modify this project.

---

## 📫 Feedback & Contributions

Feel free to open an issue or make a pull request!
