# Simple Eye Tracker

A basic eye tracking application that uses your webcam to predict where you're looking on the screen.

## Features

- Real-time eye tracking using your webcam
- Visual feedback showing where you're looking
- Calibration system to improve accuracy
- Fixation detection (detects when you're focusing on a specific area)
- Gaze smoothing for better user experience

## Requirements

The eye tracker requires the following dependencies:
- Python 3.6+
- OpenCV
- NumPy
- PyGame
- MediaPipe (part of the EyeGestures library)
- scikit-learn (part of the EyeGestures library)

## Installation

1. Make sure you have Python installed on your system
2. Install the required packages:
   ```
   pip install numpy opencv-python pygame mediapipe scikit-learn
   ```
3. The EyeGestures library is already included in this repository

## Usage

1. Run the simple eye tracker:
   ```
   python simple_eye_tracker.py
   ```

2. The application will start with a detection check phase:
   - This phase shows face and eye detection visualization for debugging
   - You can see the detected face landmarks and eye contours in real-time
   - A progress bar shows how many successful detections have been made
   - When you're ready to proceed, press the SPACEBAR to start calibration
   - You can also press 'S' to skip directly to calibration if needed

3. After the detection phase, the calibration will begin:
   - Look at each blue circle as it appears on the screen
   - The calibration uses only 5 points and smaller circles for better focus
   - Stay still during calibration for best results
   - Face and eye detection visualization remains visible for debugging

4. After calibration is complete, you'll see:
   - A red circle showing where you're currently looking
   - A green circle when you're fixating (focusing) on a specific area
   - The coordinates and fixation status displayed at the bottom of the screen

5. Controls:
   - Press 'SPACE' to advance from detection to calibration
   - Press 'D' to toggle the debug visualization of face and eye landmarks
   - Press 'R' to restart the entire process (detection and calibration)
   - Press 'Ctrl+Q' to quit the application

## How It Works

This eye tracker uses the EyeGestures library to:

1. Detect your face and eyes using computer vision
2. Track pupil movements and eye landmarks
3. Map these movements to screen coordinates using the calibration data
4. Apply smoothing to reduce jitter and improve user experience
5. Detect fixations (when you focus on a specific point)

The application uses a two-phase approach:
- A model-based approach for initial gaze estimation
- A machine learning approach to refine the predictions

## Limitations

- Works best when your camera is at arm's length
- Requires good lighting conditions
- May be less accurate without proper calibration
- Performance varies depending on your webcam quality

## Troubleshooting

### Face Detection Issues

If you see the "No face detected" warning:

1. **Ensure good lighting**: The face detection requires adequate lighting to work properly.
2. **Position yourself correctly**: Make sure your face is clearly visible and centered in the camera view.
3. **Distance from camera**: Position yourself at arm's length from the camera (about 50-70cm).
4. **Remove obstructions**: Ensure there are no objects blocking the camera's view of your face.
5. **Check camera permissions**: Make sure your application has permission to access the camera.
6. **Use the debug visualization**: The face and eye landmark visualization can help you see what's being detected.

If problems persist:
- Try restarting the application
- Try a different webcam if available
- Make sure no other applications are using your webcam

### Calibration Tips

For best results during calibration:
- Keep your head relatively still
- Follow the blue circles with your eyes, not your head
- Complete the entire calibration process without interruption (only 5 points)
- If calibration seems inaccurate, press 'R' to restart it

## Extending the Eye Tracker

This is a simple implementation that can be extended with additional features:
- Add gesture recognition
- Implement blink detection
- Create gaze-based controls for applications
- Add heatmap visualization of gaze patterns

## Credits

This application uses the EyeGestures library. 