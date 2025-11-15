# RoboMaster Obstacle Avoidance Algorithms

This repository contains two obstacle avoidance algorithms for DJI RoboMaster EP robots, implementing autonomous line-following with intelligent obstacle detection and avoidance.

## 📁 Project Structure

```
.
├── README.md
├── final(1).py                          # Sensor-based obstacle avoidance algorithm
└── 基于yolo的自动避障算法.py              # YOLO-based obstacle avoidance algorithm
```

## 🔬 Experimental Environment

### Hardware Requirements
- **Robot**: DJI RoboMaster EP
- **Sensors**: 
  - Distance sensors (4-way TOF sensors)
  - RGB Camera (1280×720)
  - Gimbal with pitch/yaw control
- **Network**: Wi-Fi connection (AP mode)

### Software Requirements

#### Python Environment
- **Python Version**: 3.7.2 or higher (Recommended: 3.7.2)
- **Operating System**: macOS / Windows / Linux

#### Core Dependencies
```
opencv-python>=4.5.0
numpy>=1.19.0
robomaster>=0.1.1
```

#### Additional Dependencies (YOLO-based algorithm only)
```
ultralytics>=8.0.0
torch>=1.9.0
```

### Conda Environment Setup (Recommended)
```bash
# Create conda environment
conda create -n robomaster-env-372 python=3.7.2

# Activate environment
conda activate robomaster-env-372

# Install dependencies
pip install opencv-python numpy robomaster

# For YOLO-based algorithm, additionally install:
pip install ultralytics torch
```

### YOLO Model Requirements
For the YOLO-based algorithm, you need to download the YOLOv5 model:
- **Model**: YOLOv5nu.pt
- **Path**: Update the model path in `基于yolo的自动避障算法.py` (line 64)
```python
model = YOLO(r"path/to/your/yolov5nu.pt")
```

## 🚀 Running Guide

### Algorithm 1: Sensor-based Obstacle Avoidance (`final(1).py`)

This algorithm uses distance sensors for obstacle detection and avoidance.

#### Features
- Blue line following with PID control
- Distance sensor-based obstacle detection
- Traffic jam detection and waiting
- Automatic line search when lost
- Gimbal scanning for line recovery

#### Configuration
Before running, configure the following parameters:

```python
# Line following parameters
BLUE_HSV = (95, 80, 50), (130, 255, 255)  # Blue line HSV range
PID_PARAM = (0.4, 0.001, 0.05)            # PID parameters (Kp, Ki, Kd)
SPD_BASE = 0.3                             # Base forward speed

# Obstacle detection
DIST_THR = 400                             # Distance threshold (mm)

# Save path for snapshots
SAVE_PATH = r"D:\PythonProject"            # Update this path
```

#### Running Steps
1. **Connect to Robot**
   ```bash
   # Ensure robot is powered on and in AP mode
   # Connect your computer to robot's Wi-Fi network (RMEP-XXXXXXX)
   ```

2. **Update Save Path**
   ```python
   # Edit line 55 in final(1).py
   SAVE_PATH = r"your/path/here"
   ```

3. **Run the Algorithm**
   ```bash
   conda activate robomaster-env-372
   python final(1).py
   ```

4. **Operation**
   - Robot will start following the blue line automatically
   - Press `q` to quit the program

#### Expected Behavior
- Robot follows blue line with smooth PID control
- Stops when obstacle detected (distance < 400mm)
- Saves snapshot of obstacle
- Resumes when obstacle clears
- Scans for line if lost

---

### Algorithm 2: YOLO-based Obstacle Avoidance (`基于yolo的自动避障算法.py`)

This algorithm uses YOLO object detection for intelligent obstacle recognition.

#### Features
- Blue line following with PID control
- YOLO-based vehicle detection (car, bus, truck)
- Real-time object recognition
- Bounding box visualization
- Distance sensor fusion

#### Configuration
Before running, configure the following parameters:

```python
# Line following parameters
BLUE_HSV = (95, 80, 50), (130, 255, 255)
PID_PARAM = (0.4, 0.001, 0.05)
SPD_BASE = 0.4

# YOLO model path
model = YOLO(r"E:\anaconda\envs\py3810\yolov5nu.pt")  # Update this

# Save path
SAVE_PATH = r"C:\Users\Suletta Mercury\Desktop\robomaster photo"  # Update this

# Obstacle detection
DIST_THR = 300                             # Distance threshold (mm)
CAR_LABELS = {0, 1, 2}                     # YOLO class IDs for vehicles
```

#### Running Steps
1. **Download YOLO Model**
   - Download YOLOv5nu.pt model
   - Place it in an accessible directory

2. **Update Configuration**
   ```python
   # Edit lines 64 and 54 in 基于yolo的自动避障算法.py
   model = YOLO(r"your/path/to/yolov5nu.pt")
   SAVE_PATH = r"your/save/path"
   ```

3. **Connect to Robot**
   ```bash
   # Connect to robot's Wi-Fi network
   ```

4. **Run the Algorithm**
   ```bash
   conda activate robomaster-env-372
   python 基于yolo的自动避障算法.py
   ```

5. **Operation**
   - Robot will start following the blue line
   - YOLO detects vehicles in real-time
   - Press `q` to quit

#### Expected Behavior
- Robot follows blue line automatically
- YOLO detects vehicles (car/bus/truck) in front
- Stops when vehicle detected and distance < 300mm
- Draws bounding box around detected vehicles
- Saves annotated snapshot
- Resumes when path clears

---

## 📊 Comparison: Sensor vs YOLO

| Feature | Sensor-based | YOLO-based |
|---------|-------------|------------|
| Detection Method | Distance sensors only | YOLO + Distance sensors |
| Object Recognition | No (generic obstacles) | Yes (car, bus, truck) |
| Processing Speed | Fast | Moderate (GPU recommended) |
| Accuracy | Distance-based | Visual + Distance |
| Hardware Requirements | Standard | GPU recommended |
| Model Size | N/A | ~25MB |

## 🛠️ Troubleshooting

### Common Issues

1. **"Could not connect to robot"**
   - Check Wi-Fi connection to robot
   - Ensure robot is in AP mode
   - Verify robot is powered on

2. **"YOLO model not found"**
   - Check model path is correct
   - Ensure model file exists
   - Use absolute path

3. **"Camera not responding"**
   - Restart robot
   - Check camera initialization
   - Ensure no other program is using camera

4. **"Line not detected"**
   - Adjust `BLUE_HSV` range for your lighting
   - Check camera angle (gimbal pitch)
   - Ensure blue line is visible

5. **Import errors**
   - Ensure conda environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

## 📝 Notes

- Both algorithms require a **blue line** on the ground for navigation
- Recommended line width: 3-5 cm
- Optimal lighting: bright, indirect light (avoid shadows)
- Test in a clear area before deployment
- Snapshots are saved automatically when obstacles detected

## 🔒 Safety

- Always supervise the robot during operation
- Keep clear of moving parts
- Have an emergency stop plan (press `q` or cut power)
- Test in a safe, enclosed environment first

## 📄 License

This project is for educational and research purposes.

## 👥 Team

Team 17

---

**For questions or issues, please open an issue on GitHub.**
