# Intelligent Traffic Monitoring System

An end-to-end AI-powered system that detects, tracks, and counts vehicles from images and videos, analyzes traffic flow, and estimates traffic conditions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)

---

## Features

- Vehicle Detection & Classification (Car, Bus, Truck, Motorcycle, Bicycle)
- Multi-object Tracking with unique IDs
- Accurate Vehicle Counting
- Traffic Flow Analysis (vehicles per minute, average density)
- Traffic Condition Estimation (Free Flow / Moderate / Heavy / Congested)
- Interactive Web Demo using Streamlit

---

## Project Structure
# Intelligent Traffic Monitoring System

An end-to-end AI-powered system that detects, tracks, and counts vehicles from images and videos, analyzes traffic flow, and estimates traffic conditions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)

---

## Features

- Vehicle Detection & Classification (Car, Bus, Truck, Motorcycle, Bicycle)
- Multi-object Tracking with unique IDs
- Accurate Vehicle Counting
- Traffic Flow Analysis (vehicles per minute, average density)
- Traffic Condition Estimation (Free Flow / Moderate / Heavy / Congested)
- Interactive Web Demo using Streamlit

---

## Project Structure
```
Intelligent-Traffic-Monitoring-System/

├── 1_vehicle_detection/

│   └── detect.py

├── 2_tracking_counting/

├── 3_flow_analysis/

│   └── flow_analyzer.py

├── 4_condition_prediction/

├── app/

│   └── streamlit_app.py

├── data/

│   ├── raw_videos/

│   ├── sample_images/

│   └── processed/

├── results/

│   └── detections/

├── notebooks/

├── requirements.txt

└── README.md
```

---

## Tech Stack

- **Computer Vision:** YOLOv8 (Ultralytics)
- **Tracking:** ByteTrack (built-in with YOLOv8)
- **Web Interface:** Streamlit
- **Core Libraries:** OpenCV, NumPy, Pandas
- **Language:** Python

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Intelligent-Traffic-Monitoring-System.git
cd Intelligent-Traffic-Monitoring-System

# Install dependencies
pip install -r requirements.txt

---

## How to Run

### 1. Run Detection / Tracking
```bash
python 1_vehicle_detection/detect.py
2. Run Traffic Flow Analysis'
python 3_flow_analysis/flow_analyzer.py
3. Launch Web Demo
streamlit run app/streamlit_app.py

AuthorS
yed AliS
Bachelor's in Data Science & AI