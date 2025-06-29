# BabyDriver: Real Time Vehicle Detection \& Speed Estimation

A real-time vehicle detection and speed estimation system using **YOLOv3** model and **Farneback** algorithm. This project tracks multiple vehicles in video footage and estimates their speeds with visual indicators for speed violations.

## Example Demo

<p align="center">
  <img src="/demo/output1.gif" width="250"/>
  <img src="/demo/output2.gif" width="250"/>
  <img src="/demo/output3.gif" width="250"/>
</p>

## Architecture 

https://github.com/user-attachments/assets/67ee68e6-9de8-4d2d-b4d2-adc35fd6ef0a


https://github.com/user-attachments/assets/3c293acc-7d77-46d1-8d23-f27892331525


## Features

- **Multi-Vehicle Detection**: Detects cars, motorcycles, buses, and trucks using YOLOv3
- **Object Tracking**: Persistent tracking across frames with unique IDs
- **Speed Estimation**: Real-time speed calculation in km/h with perspective correction
- **Speed Violation Detection**: Visual alerts for vehicles exceeding 80 km/h


## Installation

### Prerequisites
- Python 3.7+
- OpenCV 4.5+
- NumPy
- Matplotlib

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nihilisticneuralnet/BabyDriver.git
   cd BabyDriver
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv3 model files**
   ```bash
   # Download YOLOv3 weights (248 MB)
   wget https://pjreddie.com/media/files/yolov3.weights -O yolov3-320.weights
   
   # Download YOLOv3 config
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3-320.cfg
   
   # Download COCO class names
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names
   ```

4. **Prepare your video**
   ```bash
   # Place your input video as 'video.mkv' in the project directory
   # Or modify the video path in main.py
   ```

## Usage

### Basic Usage
```bash
python main.py
```

### Custom Configuration
Modify these parameters in `main.py` for your specific use case:

```python
# Detection parameters
confThreshold = 0.2        # Confidence threshold (0.1-0.9)
nmsThreshold = 0.2         # Non-max suppression threshold

# Speed calculation parameters
PIXELS_PER_METER = 4       # Calibrate based on your video
SPEED_MULTIPLIER = 6.0     # Fine-tune speed accuracy
FPS = 30                   # Video frame rate

# Input/Output
input_video = "video.mkv"  # Input video path
output_video = "output.mp4" # Output video path
```

### Calibration Guide

1. **PIXELS_PER_METER**: Measure a known distance in your video (e.g., lane width ≈ 3.5m)
   ```python
   PIXELS_PER_METER = measured_pixels / known_meters
   ```

2. **SPEED_MULTIPLIER**: Adjust based on ground truth or expected speeds
   - Start with 1.0 and increase if speeds seem too low
   - Typical range: 3.0-10.0 depending on camera setup

## Output

The system generates:
- **Processed Video**: `output.mp4` with bounding boxes, IDs, and speed overlays
- **Console Logs**: Processing progress and statistics
- **Speed Indicators**: 
  - 🟢 Green text: Normal speed (≤80 km/h)
  - 🔴 Red text: Speed violation (>80 km/h)



## Technical Architecture

### Core Components

1. **Detection Engine** (`detect` class)
   - YOLOv3 inference pipeline
   - Multi-object tracking
   - Speed calculation with smoothing

2. **Tracking System** (`tracker` class)
   - Centroid-based object association
   - Kalman-like prediction for missing detections
   - Automatic ID management

3. **Speed Estimation**
   - Frame-to-frame displacement calculation
   - Perspective correction for camera angle
   - Temporal smoothing with weighted averages

### Performance Optimizations

- **Multithreading**: OpenCV optimized for 4 threads
- **Resolution Scaling**: 50% downscaling for faster processing
- **Selective Detection**: Only processes relevant vehicle classes

## Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `confThreshold` | Detection confidence | 0.2 | 0.1-0.9 |
| `nmsThreshold` | NMS overlap threshold | 0.2 | 0.1-0.5 |
| `PIXELS_PER_METER` | Spatial calibration | 4 | 1-20 |
| `SPEED_MULTIPLIER` | Speed calibration | 6.0 | 1.0-15.0 |
| `input_size` | YOLO input resolution | 320 | 320/416/608 |

## Troubleshooting

### Common Issues

1. **Unrealistic Speeds**
   - Adjust `PIXELS_PER_METER` and `SPEED_MULTIPLIER`
   - Ensure video has consistent frame rate

2. **Poor Detection**
   - Lower `confThreshold` for more detections
   - Check lighting and video quality

3. **Memory Issues**
   - Reduce video resolution
   - Process shorter video segments

4. **Missing Model Files**
   ```bash
   # Re-download YOLOv3 files
   wget https://pjreddie.com/media/files/yolov3.weights
   ```

## Future Enhancements

- [ ] **YOLOv5/v8 Integration**: Upgrade to newer, faster models
- [ ] **GPU Acceleration**: CUDA/OpenCL support for real-time processing
- [ ] **Other Algos**: DeepSORT or ByteTrack integration or Lucas-Kannade, etc for uniform speed estimation
- [ ] **Analytics Dashboard**: Speed statistics and violation reports
- [ ] **Live Stream Processing**: Real-time camera feed analysis


## License

This project is licensed under the MIT License. See LICENSE file for details.


## Citation

If you use this code in academic work, please cite:
```
@software{babydriver_detect,
  title={BabyDriver: Real Time Vehicle Detection & Speed Estimation},
  author={[Parth]},
  year={2025},
  url={https://github.com/nihilisticneuralnet/BabyDriver}
}
```
