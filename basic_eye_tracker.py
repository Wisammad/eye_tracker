import cv2
import numpy as np
import pygame
import mediapipe as mp
import sys
import time
import csv
import os
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize constants
# MediaPipe indices for eyes
LEFT_EYE_INDICES = list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_INDICES = list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LEFT_IRIS_INDICES = list(mp_face_mesh.FACEMESH_LEFT_IRIS)
RIGHT_IRIS_INDICES = list(mp_face_mesh.FACEMESH_RIGHT_IRIS)

# Pupil landmarks
LEFT_PUPIL = 468  # Center of the left iris
RIGHT_PUPIL = 473  # Center of the right iris

# For eye aspect ratio calculation
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# More specific eye landmarks for better accuracy
# Top and bottom landmarks for the iris height calculation
LEFT_EYE_TOP_INNER = 158
LEFT_EYE_BOTTOM_INNER = 144
RIGHT_EYE_TOP_INNER = 385
RIGHT_EYE_BOTTOM_INNER = 373

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)

# Data logging
ENABLE_LOGGING = False  # Set to True to enable data logging
log_directory = "eye_tracking_logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_file = os.path.join(log_directory, f"eye_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_header = ['timestamp', 'left_ear', 'right_ear', 'avg_ear', 'is_blinking', 'blink_count', 
             'h_gaze', 'v_gaze', 'left_pupil_x', 'left_pupil_y', 'right_pupil_x', 'right_pupil_y',
             'iris_relative_x', 'iris_relative_y']

# Debug mode
DEBUG_MODE = True  # Set to True to see raw values and debugging info

# Use alternative calculation method for vertical gaze
USE_ALTERNATIVE_METHOD = True  # Use a completely different method for vertical gaze

# Initialize Pygame
pygame.init()
pygame.display.set_caption("Advanced Eye Tracker")

# Try to open webcam with different indices
cap = None
for camera_idx in range(3):  # Try 3 different camera indices
    print(f"Trying to open camera at index {camera_idx}")
    cap = cv2.VideoCapture(camera_idx)
    success, img = cap.read()
    if success and img is not None and img.size > 0:
        print(f"Successfully opened camera at index {camera_idx}")
        break
    cap.release()
    time.sleep(0.5)  # Wait a bit before trying next camera

if cap is None or not cap.isOpened():
    print("Failed to open any webcam. Exiting...")
    pygame.quit()
    sys.exit(1)

# Make sure we have a valid image
success, img = cap.read()
if not success or img is None or img.size == 0:
    print("Failed to read valid frame from webcam. Exiting...")
    cap.release()
    pygame.quit()
    sys.exit(1)

# Set up screen with dimensions based on webcam
img_height, img_width, _ = img.shape
screen_width, screen_height = 1024, 768  # Default size
display_scale = min(screen_width / img_width, screen_height / img_height)
display_width = int(img_width * display_scale)
display_height = int(img_height * display_scale)
screen = pygame.display.set_mode((display_width, display_height))

# Initialize face mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Font for displaying information
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)
debug_font = pygame.font.Font(None, 20)

# Variables for tracking eye state
blink_count = 0
last_blink_time = time.time()
blink_start_time = 0
is_blinking = False
blink_duration = 0
gaze_horizontal = "Center"
gaze_vertical = "Center"

# Current raw iris relative positions
current_iris_relative_x = 0.0
current_iris_relative_y = 0.0

# Eye aspect ratio threshold for blink detection
EAR_THRESHOLD = 0.2

# Variables for gaze smoothing
gaze_h_history = []
gaze_v_history = []
iris_x_history = []
iris_y_history = []
max_history = 10

# Gaze detection thresholds - adjusted for better sensitivity
# These values might need further tuning based on testing
GAZE_H_THRESHOLD = 0.005  # Reduced for more sensitivity

# Asymmetric vertical thresholds - stricter for "Up", more sensitive for "Down"
GAZE_V_THRESHOLD_UP = 0.010    # Increased for almost no upward detection
GAZE_V_THRESHOLD_DOWN = 0.002  # More sensitive threshold for downward gaze

# Vertical bias to prioritize downward gaze (positive shifts toward "Down" detection)
VERTICAL_BIAS = 0.015  # Strong bias to avoid "Up" false positives

# Custom drawing specs
IRIS_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
IRIS_CONNECTIONS_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

# Create CSV file with headers if logging is enabled
if ENABLE_LOGGING:
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
    print(f"Data logging enabled. Saving to {log_file}")

def calculate_ear(landmarks, eye_points):
    """Calculate Eye Aspect Ratio"""
    # Extract the points
    top = landmarks.landmark[eye_points[0]]
    bottom = landmarks.landmark[eye_points[1]]
    left = landmarks.landmark[eye_points[2]]
    right = landmarks.landmark[eye_points[3]]
    
    # Calculate the height
    height = abs(top.y - bottom.y)
    # Calculate the width
    width = abs(left.x - right.x)
    
    # Calculate EAR
    ear = height / width
    return ear

def estimate_gaze(face_landmarks, frame_width, frame_height):
    """Estimate gaze direction based on iris position relative to eye corners"""
    global current_iris_relative_x, current_iris_relative_y
    
    # Get eye landmarks
    left_iris = face_landmarks.landmark[LEFT_PUPIL]
    right_iris = face_landmarks.landmark[RIGHT_PUPIL]
    
    # Get eye corners
    left_eye_left = face_landmarks.landmark[LEFT_EYE_LEFT]
    left_eye_right = face_landmarks.landmark[LEFT_EYE_RIGHT]
    left_eye_top = face_landmarks.landmark[LEFT_EYE_TOP]
    left_eye_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM]
    
    right_eye_left = face_landmarks.landmark[RIGHT_EYE_LEFT]
    right_eye_right = face_landmarks.landmark[RIGHT_EYE_RIGHT]
    right_eye_top = face_landmarks.landmark[RIGHT_EYE_TOP]
    right_eye_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM]
    
    # Get the inner points for more precise vertical positioning
    left_eye_top_inner = face_landmarks.landmark[LEFT_EYE_TOP_INNER]
    left_eye_bottom_inner = face_landmarks.landmark[LEFT_EYE_BOTTOM_INNER]
    right_eye_top_inner = face_landmarks.landmark[RIGHT_EYE_TOP_INNER]
    right_eye_bottom_inner = face_landmarks.landmark[RIGHT_EYE_BOTTOM_INNER]
    
    # HORIZONTAL GAZE CALCULATION (unchanged)
    # Calculate the center of each eye
    left_eye_center_x = (left_eye_left.x + left_eye_right.x) / 2
    right_eye_center_x = (right_eye_left.x + right_eye_right.x) / 2
    
    # Calculate eye width for normalization
    left_eye_width = abs(left_eye_right.x - left_eye_left.x)
    right_eye_width = abs(right_eye_right.x - right_eye_left.x)
    
    # Calculate relative position of iris to eye center (horizontal)
    # Normalize by eye width to account for different face sizes and distances
    left_iris_relative_x = (left_iris.x - left_eye_center_x) / left_eye_width
    right_iris_relative_x = (right_iris.x - right_eye_center_x) / right_eye_width
    
    # Average the values for horizontal position
    iris_relative_x = (left_iris_relative_x + right_iris_relative_x) / 2
    
    # VERTICAL GAZE CALCULATION
    iris_relative_y = 0
    
    if USE_ALTERNATIVE_METHOD:
        # Alternative method: Calculate relative position using the fraction of space 
        # between the top and bottom of the eye
        
        # Calculate the total height of the eye (from top to bottom)
        left_eye_height = left_eye_bottom.y - left_eye_top.y
        right_eye_height = right_eye_bottom.y - right_eye_top.y
        
        # Calculate how far the iris is from the top of the eye, as a fraction of total eye height
        # 0 = iris at top, 0.5 = iris in middle, 1 = iris at bottom
        left_iris_relative_pos = (left_iris.y - left_eye_top.y) / left_eye_height
        right_iris_relative_pos = (right_iris.y - right_eye_top.y) / right_eye_height
        
        # Average the positions
        avg_rel_pos = (left_iris_relative_pos + right_iris_relative_pos) / 2
        
        # Convert to our coordinate system where negative = up, positive = down
        # centered around 0.5 (the middle of the eye)
        iris_relative_y = (avg_rel_pos - 0.5) * 2
    else:
        # Original method with inner eye points for better precision
        # Calculate the center of each eye (vertical)
        left_eye_center_y = (left_eye_top_inner.y + left_eye_bottom_inner.y) / 2
        right_eye_center_y = (right_eye_top_inner.y + right_eye_bottom_inner.y) / 2
        
        # Calculate eye height for normalization
        left_eye_height = abs(left_eye_bottom_inner.y - left_eye_top_inner.y)
        right_eye_height = abs(right_eye_bottom_inner.y - right_eye_top_inner.y)
        
        # Calculate relative position of iris to eye center (vertical)
        # Normalize by eye height
        left_iris_relative_y = (left_iris.y - left_eye_center_y) / left_eye_height
        right_iris_relative_y = (right_iris.y - right_eye_center_y) / right_eye_height
        
        # Average the values for vertical position
        iris_relative_y = (left_iris_relative_y + right_iris_relative_y) / 2
    
    # Apply vertical bias to make "Up" detection much less likely
    iris_relative_y += VERTICAL_BIAS
    
    # Store current values for debugging
    current_iris_relative_x = iris_relative_x
    current_iris_relative_y = iris_relative_y
    
    # Add to history for smoothing
    iris_x_history.append(iris_relative_x)
    iris_y_history.append(iris_relative_y)
    
    if len(iris_x_history) > max_history:
        iris_x_history.pop(0)
    if len(iris_y_history) > max_history:
        iris_y_history.pop(0)
    
    # Use the average of recent values for more stability
    if iris_x_history and iris_y_history:
        iris_relative_x = sum(iris_x_history) / len(iris_x_history)
        iris_relative_y = sum(iris_y_history) / len(iris_y_history)
    
    # Determine horizontal gaze direction
    h_direction = "Center"
    if iris_relative_x < -GAZE_H_THRESHOLD:
        h_direction = "Left"
    elif iris_relative_x > GAZE_H_THRESHOLD:
        h_direction = "Right"
    
    # Determine vertical gaze direction using asymmetric thresholds
    v_direction = "Center"
    if iris_relative_y < -GAZE_V_THRESHOLD_UP:  # Stricter threshold for upward gaze
        v_direction = "Up"
    elif iris_relative_y > GAZE_V_THRESHOLD_DOWN:  # More sensitive threshold for downward gaze
        v_direction = "Down"
    
    return h_direction, v_direction, iris_relative_x, iris_relative_y

def log_data(timestamp, left_ear, right_ear, avg_ear, is_blinking, blink_count,
             h_gaze, v_gaze, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
             iris_relative_x, iris_relative_y):
    """Log eye tracking data to CSV file"""
    if not ENABLE_LOGGING:
        return
    
    with open(log_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, left_ear, right_ear, avg_ear, is_blinking, blink_count,
                        h_gaze, v_gaze, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
                        iris_relative_x, iris_relative_y])

def process_frame(frame):
    global blink_count, last_blink_time, is_blinking, blink_start_time, blink_duration
    global gaze_horizontal, gaze_vertical, gaze_h_history, gaze_v_history
    global current_iris_relative_x, current_iris_relative_y
    
    # Convert the frame colors from BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find face landmarks
    results = face_mesh.process(frame_rgb)
    
    h, w, _ = frame.shape
    
    # Draw face mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh with reduced opacity
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw eye contours with more visibility
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw the iris with custom drawing specs
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_IRIS,
                landmark_drawing_spec=IRIS_SPEC,
                connection_drawing_spec=IRIS_CONNECTIONS_SPEC
            )
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_IRIS,
                landmark_drawing_spec=IRIS_SPEC,
                connection_drawing_spec=IRIS_CONNECTIONS_SPEC
            )
            
            # Draw pupils with larger size and different color for better visibility
            left_pupil_point = face_landmarks.landmark[LEFT_PUPIL]
            right_pupil_point = face_landmarks.landmark[RIGHT_PUPIL]
            
            left_pupil_x, left_pupil_y = int(left_pupil_point.x * w), int(left_pupil_point.y * h)
            right_pupil_x, right_pupil_y = int(right_pupil_point.x * w), int(right_pupil_point.y * h)
            
            cv2.circle(frame, (left_pupil_x, left_pupil_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_pupil_x, right_pupil_y), 5, (0, 255, 0), -1)
            
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            left_ear = calculate_ear(face_landmarks, [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT])
            right_ear = calculate_ear(face_landmarks, [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT])
            
            # Average EAR for both eyes
            avg_ear = (left_ear + right_ear) / 2
            
            # Detect blinks
            if not is_blinking and avg_ear < EAR_THRESHOLD:
                is_blinking = True
                blink_start_time = time.time()
            elif is_blinking and avg_ear >= EAR_THRESHOLD:
                is_blinking = False
                blink_duration = time.time() - blink_start_time
                if blink_duration < 0.5:  # Typical blink is around 0.1-0.4 seconds
                    blink_count += 1
                    last_blink_time = time.time()
            
            # Estimate gaze direction
            h_gaze, v_gaze, iris_rel_x, iris_rel_y = estimate_gaze(face_landmarks, w, h)
            
            # Add to history for smoothing
            gaze_h_history.append(h_gaze)
            gaze_v_history.append(v_gaze)
            
            if len(gaze_h_history) > max_history:
                gaze_h_history.pop(0)
            if len(gaze_v_history) > max_history:
                gaze_v_history.pop(0)
            
            # Determine most common gaze direction from history
            if gaze_h_history and gaze_v_history:
                from collections import Counter
                h_counter = Counter(gaze_h_history)
                v_counter = Counter(gaze_v_history)
                
                gaze_horizontal = h_counter.most_common(1)[0][0]
                gaze_vertical = v_counter.most_common(1)[0][0]
            
            # Visualize gaze direction with an arrow
            gaze_text = f"Looking: {gaze_horizontal}-{gaze_vertical}"
            
            # Add info to frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, gaze_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add debug info if debug mode is enabled
            if DEBUG_MODE:
                cv2.putText(frame, f"Iris-X: {current_iris_relative_x:.4f}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Iris-Y: {current_iris_relative_y:.4f}", (10, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"H-Thresh: {GAZE_H_THRESHOLD}", (10, 190), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"V-Up: {GAZE_V_THRESHOLD_UP} | V-Down: {GAZE_V_THRESHOLD_DOWN}", (10, 210), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"V-Bias: {VERTICAL_BIAS}", (10, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Method: {'Alternative' if USE_ALTERNATIVE_METHOD else 'Standard'}", (10, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Visualize blink status
            blink_status = "Blinking" if is_blinking else "Eyes Open"
            cv2.putText(frame, blink_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Calculate average pupil position for gaze visualization
            avg_pupil_x = (left_pupil_x + right_pupil_x) // 2
            avg_pupil_y = (left_pupil_y + right_pupil_y) // 2
            
            # Draw gaze direction indicator
            arrow_length = 50
            arrow_start = (avg_pupil_x, avg_pupil_y)
            arrow_end = (avg_pupil_x, avg_pupil_y)
            
            if gaze_horizontal == "Left":
                arrow_end = (arrow_start[0] - arrow_length, arrow_start[1])
            elif gaze_horizontal == "Right":
                arrow_end = (arrow_start[0] + arrow_length, arrow_start[1])
                
            if gaze_vertical == "Up":
                arrow_end = (arrow_end[0], arrow_start[1] - arrow_length)
            elif gaze_vertical == "Down":
                arrow_end = (arrow_end[0], arrow_start[1] + arrow_length)
                
            # Draw the gaze direction arrow
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 2)
            
            # Log data
            log_data(time.time(), left_ear, right_ear, avg_ear, is_blinking, blink_count,
                     gaze_horizontal, gaze_vertical, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
                     current_iris_relative_x, current_iris_relative_y)
            
            # If data logging is enabled, display indicator
            if ENABLE_LOGGING:
                cv2.putText(frame, "Recording Data", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

running = True
clock = pygame.time.Clock()

# Variables for threshold adjustment
threshold_adjust_mode = False

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_r:  # Reset blink counter
                blink_count = 0
            if event.key == pygame.K_l:  # Toggle logging
                ENABLE_LOGGING = not ENABLE_LOGGING
                if ENABLE_LOGGING:
                    log_file = os.path.join(log_directory, f"eye_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    with open(log_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(csv_header)
                    print(f"Data logging enabled. Saving to {log_file}")
                else:
                    print("Data logging disabled")
            if event.key == pygame.K_d:  # Toggle debug mode
                DEBUG_MODE = not DEBUG_MODE
                print(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
            if event.key == pygame.K_t:  # Toggle threshold adjustment mode
                threshold_adjust_mode = not threshold_adjust_mode
                print(f"Threshold adjustment mode: {'ON' if threshold_adjust_mode else 'OFF'}")
            if event.key == pygame.K_m:  # Toggle calculation method
                USE_ALTERNATIVE_METHOD = not USE_ALTERNATIVE_METHOD
                # Clear history when switching methods
                iris_y_history.clear()
                print(f"Using {'alternative' if USE_ALTERNATIVE_METHOD else 'standard'} method")
            if event.key == pygame.K_c:  # Calibrate vertical bias
                # Reset any existing bias and set the current iris position as the new center
                if iris_y_history:
                    avg_y = sum(iris_y_history) / len(iris_y_history)
                    VERTICAL_BIAS = -avg_y  # Reverse the current position to make it the new center
                    print(f"Vertical bias calibrated. New bias: {VERTICAL_BIAS:.4f}")
                    iris_y_history.clear()  # Clear history after calibration
            
            # Threshold adjustment keys (when in threshold adjustment mode)
            if threshold_adjust_mode:
                # Vertical threshold adjustment for "Up" gaze
                if event.key == pygame.K_UP:
                    GAZE_V_THRESHOLD_UP += 0.001  # Increase threshold (less sensitive for Up)
                    print(f"Up threshold increased to {GAZE_V_THRESHOLD_UP:.4f}")
                if event.key == pygame.K_DOWN:
                    GAZE_V_THRESHOLD_UP = max(0.001, GAZE_V_THRESHOLD_UP - 0.001)  # Decrease threshold but keep positive
                    print(f"Up threshold decreased to {GAZE_V_THRESHOLD_UP:.4f}")
                
                # Vertical threshold adjustment for "Down" gaze
                if event.key == pygame.K_PAGEUP:
                    GAZE_V_THRESHOLD_DOWN = max(0.001, GAZE_V_THRESHOLD_DOWN - 0.001)  # Make more sensitive for Down
                    print(f"Down threshold decreased to {GAZE_V_THRESHOLD_DOWN:.4f}")
                if event.key == pygame.K_PAGEDOWN:
                    GAZE_V_THRESHOLD_DOWN += 0.001  # Make less sensitive for Down
                    print(f"Down threshold increased to {GAZE_V_THRESHOLD_DOWN:.4f}")
                
                # Horizontal threshold adjustment
                if event.key == pygame.K_LEFT:
                    GAZE_H_THRESHOLD -= 0.001
                    print(f"Horizontal threshold decreased to {GAZE_H_THRESHOLD:.4f}")
                if event.key == pygame.K_RIGHT:
                    GAZE_H_THRESHOLD += 0.001
                    print(f"Horizontal threshold increased to {GAZE_H_THRESHOLD:.4f}")
                
                # Vertical bias adjustment
                if event.key == pygame.K_w:
                    VERTICAL_BIAS += 0.001  # Adjust upward (more "Down" detection)
                    print(f"Vertical bias increased to {VERTICAL_BIAS:.4f}")
                if event.key == pygame.K_s:
                    VERTICAL_BIAS -= 0.001  # Adjust downward (more "Up" detection)
                    print(f"Vertical bias decreased to {VERTICAL_BIAS:.4f}")
                
                # Quick bias adjustments
                if event.key == pygame.K_b:  # Bias strongly toward DOWN
                    VERTICAL_BIAS = 0.03
                    print(f"Strong downward bias applied: {VERTICAL_BIAS:.4f}")
                if event.key == pygame.K_n:  # Bias strongly toward CENTER
                    VERTICAL_BIAS = 0.015
                    print(f"Medium center bias applied: {VERTICAL_BIAS:.4f}")
                if event.key == pygame.K_v:  # Allow more UP detection
                    VERTICAL_BIAS = 0.005
                    print(f"Light bias applied: {VERTICAL_BIAS:.4f}")
    
    # Capture frame
    success, img = cap.read()
    if not success or img is None or img.size == 0:
        print("Failed to read frame from webcam")
        time.sleep(0.1)
        continue
    
    # Flip the image horizontally for a selfie-view display
    img = cv2.flip(img, 1)
    
    # Process the frame
    processed_frame = process_frame(img)
    
    # Convert the frame for pygame
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    processed_frame = cv2.resize(processed_frame, (display_width, display_height))
    
    # Update pygame display
    pygame_surface = pygame.surfarray.make_surface(processed_frame.swapaxes(0, 1))
    screen.blit(pygame_surface, (0, 0))
    
    # Add help text at the bottom
    help_text = small_font.render("ESC: exit | R: reset | L: log | D: debug | T: threshold | M: method | C: calibrate", True, (255, 255, 255))
    screen.blit(help_text, (10, display_height - 30))
    
    # Display threshold adjustment instructions if in that mode
    if threshold_adjust_mode:
        adjust_text = small_font.render("↑/↓: Up thresh | PgUp/PgDn: Down thresh | ←/→: H-Thresh | W/S: V-Bias | B/N/V: Quick bias", True, (255, 255, 0))
        screen.blit(adjust_text, (10, display_height - 50))
    
    # Display logging status
    if ENABLE_LOGGING:
        logging_text = small_font.render("Data Logging: ENABLED", True, (0, 255, 0))
    else:
        logging_text = small_font.render("Data Logging: DISABLED", True, (255, 0, 0))
    screen.blit(logging_text, (display_width - 200, display_height - 30))
    
    pygame.display.flip()
    clock.tick(30)  # Limit to 30fps

# Clean up
cap.release()
pygame.quit()
print("Eye tracker closed successfully.") 