# Eye Tracker

A real-time eye tracking application using MediaPipe Face Mesh and Pygame.

## Features

- Face mesh detection and visualization
- Eye and iris tracking
- Blink detection
- Comprehensive gaze direction estimation:
  - Horizontal (left, center, right)
  - Vertical (up, center, down)
  - Visual arrow indicator showing gaze direction
- Real-time metrics display:
  - Eye Aspect Ratio (EAR)
  - Blink count
  - Gaze direction
  - Current blinking state
- Data logging functionality to CSV for analysis and research:
  - Timestamps
  - EAR values for both eyes
  - Blink detection
  - Gaze direction
  - Pupil positions

## Requirements

- Python 3.6+
- OpenCV
- Pygame
- MediaPipe
- NumPy

## Installation

1. Install the required packages:
```bash
pip install pygame mediapipe opencv-python numpy
```

2. Run the application:
```bash
python basic_eye_tracker.py
```

## Controls

- Press `ESC` to exit the application
- Press `R` to reset the blink counter
- Press `L` to toggle data logging (saves to eye_tracking_logs directory)

## How It Works

This eye tracker uses the MediaPipe Face Mesh to detect 468 facial landmarks in real-time. It identifies the eye regions and pupils, then applies the following techniques:

1. **Eye Aspect Ratio (EAR)** calculation for blink detection
2. **Pupil positioning** relative to eye corners for horizontal and vertical gaze direction estimation:
   - Horizontal position determines left/right gaze
   - Vertical position determines up/down gaze
3. **Temporal smoothing** to reduce jitter in gaze estimation
4. **Visual feedback** with an arrow indicating the estimated gaze direction
5. **Data logging** for further analysis and research applications

## Data Collection

When data logging is enabled (press `L`), the application records:
- Timestamp for each frame
- Left and right eye aspect ratios
- Average eye aspect ratio
- Blinking status (boolean)
- Blink counter
- Horizontal and vertical gaze directions
- Left and right pupil coordinates

The data is saved to the `eye_tracking_logs` directory in CSV format with a timestamp in the filename.

## Future Improvements

- Calibration procedure for more accurate gaze tracking
- Head pose estimation and correction
- Screen mapping to translate gaze to screen coordinates
- More advanced gesture detection (winks, squints, etc.)
- User-configurable thresholds and parameters
- Data visualization and analysis tools

## Acknowledgements

This project utilizes Google's MediaPipe library for facial landmark detection. 