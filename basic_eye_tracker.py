import cv2
import numpy as np
import pygame
import mediapipe as mp
import sys
import time
import csv
import os
import json
import random
import math
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

# Calibration settings
CALIBRATION_MODE = False            # Toggle for calibration mode
CALIBRATION_POSITIONS = ["Center", "Left", "Right", "Up", "Down"]  # Positions to calibrate
CURRENT_CALIBRATION_POS = 0         # Current position in calibration sequence
CALIBRATION_SAMPLES = {}            # Collected samples during calibration
CALIBRATION_SAMPLES_REQUIRED = 30   # Number of samples to collect per position
CALIBRATION_TIME_PER_POSITION = 5   # Seconds to spend on each calibration position
CALIBRATION_POSITION_START_TIME = 0 # When current position started
VALIDATION_REQUIRED = 3             # Number of consecutive correct validations required
VALIDATION_SEQUENCE = []            # Random sequence for validation
VALIDATION_CURRENT = 0              # Current position in validation sequence
VALIDATION_CORRECT = 0              # Number of correct validations
CALIBRATION_STATE = "WAITING"       # Current state: WAITING, SAMPLING, VALIDATING, COMPLETE
CALIBRATION_DATA = {}               # Final calibration data
CALIBRATION_WARMUP_FRAMES = 30      # Frames to skip at the beginning of calibration to stabilize
CALIBRATION_FILTER_OUTLIERS = True  # Filter outlier measurements during calibration
CALIBRATION_TOLERANCE = 0.2         # Tolerance for validation (0.0-1.0)
CALIBRATION_USE_WEIGHTED_SAMPLES = True  # Give more weight to recent samples
CALIBRATION_AUTO_ADJUST = True      # Automatically adjust settings based on calibration results

# Calibration stability settings
CALIBRATION_DAMPING_FACTOR = 0.4    # How much weight to give to new calibration (0.0-1.0)
                                    # Lower = more stable but slower to adapt

# Config file path
CONFIG_FILE = "eye_tracker_config.json"

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

# Gaze thresholds that will be calibrated
GAZE_H_THRESHOLD = 0.005  # Horizontal threshold
GAZE_V_THRESHOLD_UP = 0.010  # Upward threshold  
GAZE_V_THRESHOLD_DOWN = 0.002  # Downward threshold
VERTICAL_BIAS = 0.015  # Vertical bias

# Functions for calibration

def start_calibration():
    """Start the calibration process"""
    global CALIBRATION_MODE, CALIBRATION_STATE, CURRENT_CALIBRATION_POS, CALIBRATION_SAMPLES, calibration_attempt
    global warmup_counter, CALIBRATION_POSITION_START_TIME
    
    CALIBRATION_MODE = True
    CALIBRATION_STATE = "WARMUP"  # Start with warmup period
    CURRENT_CALIBRATION_POS = 0
    warmup_counter = 0  # Initialize warmup counter
    CALIBRATION_SAMPLES = {pos: {'x': [], 'y': [], 'timestamps': []} for pos in CALIBRATION_POSITIONS}
    calibration_attempt += 1
    CALIBRATION_POSITION_START_TIME = time.time()
    print(f"Calibration attempt {calibration_attempt}/{MAX_CALIBRATION_ATTEMPTS} started.")
    print(f"Look at the center of the screen for initial stabilization...")

def collect_calibration_sample(iris_x, iris_y):
    """Collect a sample during calibration"""
    global CALIBRATION_SAMPLES, CURRENT_CALIBRATION_POS, CALIBRATION_STATE, CALIBRATION_POSITION_START_TIME
    
    # Skip if we're still in warmup phase or not in sampling state
    if CALIBRATION_STATE != "SAMPLING":
        return
    
    current_pos = CALIBRATION_POSITIONS[CURRENT_CALIBRATION_POS]
    current_time = time.time()
    
    # Store the sample with timestamp
    CALIBRATION_SAMPLES[current_pos]['x'].append(iris_x)
    CALIBRATION_SAMPLES[current_pos]['y'].append(iris_y)
    CALIBRATION_SAMPLES[current_pos]['timestamps'].append(current_time)

def move_to_next_calibration_position():
    """Move to the next position in the calibration sequence"""
    global CURRENT_CALIBRATION_POS, CALIBRATION_POSITION_START_TIME
    
    CURRENT_CALIBRATION_POS += 1
    CALIBRATION_POSITION_START_TIME = time.time()  # Reset timer for the new position
    
    if CURRENT_CALIBRATION_POS >= len(CALIBRATION_POSITIONS):
        calculate_calibration_values()
        start_validation()
    else:
        print(f"Look at the {CALIBRATION_POSITIONS[CURRENT_CALIBRATION_POS]} of the screen.")

def calculate_calibration_values():
    """Calculate calibration values from collected samples"""
    global CALIBRATION_DATA, GAZE_H_THRESHOLD, GAZE_V_THRESHOLD_UP, GAZE_V_THRESHOLD_DOWN, VERTICAL_BIAS
    
    # Save previous calibration values to allow for weighted averaging
    prev_h_threshold = GAZE_H_THRESHOLD
    prev_v_threshold_up = GAZE_V_THRESHOLD_UP
    prev_v_threshold_down = GAZE_V_THRESHOLD_DOWN
    prev_vertical_bias = VERTICAL_BIAS
    
    # Calculate average values for each position, potentially with weighting
    averages = {}
    expected_positions = {}
    
    for pos in CALIBRATION_POSITIONS:
        if len(CALIBRATION_SAMPLES[pos]['x']) > 0 and len(CALIBRATION_SAMPLES[pos]['y']) > 0:
            # Option to weight more recent samples more heavily
            if CALIBRATION_USE_WEIGHTED_SAMPLES and len(CALIBRATION_SAMPLES[pos]['x']) > 10:
                # Get timestamps and normalize them to weights (more recent = higher weight)
                timestamps = CALIBRATION_SAMPLES[pos]['timestamps']
                min_time = min(timestamps)
                max_time = max(timestamps)
                time_range = max_time - min_time
                
                # Normalize time to weights between 1.0 and 3.0 (newer samples count 3x more)
                if time_range > 0:
                    weights = [1.0 + 2.0 * (t - min_time) / time_range for t in timestamps]
                    total_weight = sum(weights)
                    
                    # Calculate weighted averages
                    avg_x = sum(x * w for x, w in zip(CALIBRATION_SAMPLES[pos]['x'], weights)) / total_weight
                    avg_y = sum(y * w for y, w in zip(CALIBRATION_SAMPLES[pos]['y'], weights)) / total_weight
                else:
                    # Fallback if all timestamps are the same
                    avg_x = sum(CALIBRATION_SAMPLES[pos]['x']) / len(CALIBRATION_SAMPLES[pos]['x'])
                    avg_y = sum(CALIBRATION_SAMPLES[pos]['y']) / len(CALIBRATION_SAMPLES[pos]['y'])
            else:
                # Simple average
                avg_x = sum(CALIBRATION_SAMPLES[pos]['x']) / len(CALIBRATION_SAMPLES[pos]['x'])
                avg_y = sum(CALIBRATION_SAMPLES[pos]['y']) / len(CALIBRATION_SAMPLES[pos]['y'])
            
            averages[pos] = {'x': avg_x, 'y': avg_y}
            
            # Store expected ideal positions for each target
            # These are the theoretical perfect values we'd expect to see
            if pos == "Center":
                expected_positions[pos] = {'x': 0.0, 'y': 0.0}  # Center should have zero offset
            elif pos == "Left":
                expected_positions[pos] = {'x': -0.4, 'y': 0.0}  # Left should have negative x
            elif pos == "Right":
                expected_positions[pos] = {'x': 0.4, 'y': 0.0}   # Right should have positive x
            elif pos == "Up":
                expected_positions[pos] = {'x': 0.0, 'y': -0.4}  # Up should have negative y
            elif pos == "Down":
                expected_positions[pos] = {'x': 0.0, 'y': 0.4}   # Down should have positive y
    
    # If we're automatically adjusting settings based on calibration
    if CALIBRATION_AUTO_ADJUST and len(averages) >= 3:
        # Calculate error between expected and actual values
        errors = {}
        for pos in averages:
            if pos in expected_positions:
                exp_x = expected_positions[pos]['x']
                exp_y = expected_positions[pos]['y']
                act_x = averages[pos]['x']
                act_y = averages[pos]['y']
                
                # Calculate error as a ratio (how much our thresholds need to be scaled)
                errors[pos] = {
                    'x_error': abs(exp_x - act_x) / max(0.001, abs(exp_x)) if exp_x != 0 else 0,
                    'y_error': abs(exp_y - act_y) / max(0.001, abs(exp_y)) if exp_y != 0 else 0
                }
        
        # Adjust thresholds based on errors but with damping
        # For horizontal thresholds, use error from Left and Right positions
        h_scale = 1.0
        if 'Left' in errors:
            h_scale *= (1.0 / max(0.1, errors['Left']['x_error'])) if errors['Left']['x_error'] > 0 else 1.0
        if 'Right' in errors:
            h_scale *= (1.0 / max(0.1, errors['Right']['x_error'])) if errors['Right']['x_error'] > 0 else 1.0
        
        # Apply stronger limiting to scale factors to prevent extreme adjustments
        h_scale = max(0.7, min(1.5, h_scale))  # Limit to 70-150% change
        
        # For vertical thresholds, use error from Up and Down positions
        v_up_scale = 1.0
        v_down_scale = 1.0
        if 'Up' in errors:
            v_up_scale = (1.0 / max(0.1, errors['Up']['y_error'])) if errors['Up']['y_error'] > 0 else 1.0
        if 'Down' in errors:
            v_down_scale = (1.0 / max(0.1, errors['Down']['y_error'])) if errors['Down']['y_error'] > 0 else 1.0
        
        # Apply stronger limiting to vertical scale factors
        v_up_scale = max(0.7, min(1.5, v_up_scale))    # Limit to 70-150% change
        v_down_scale = max(0.7, min(1.5, v_down_scale))  # Limit to 70-150% change
        
        print(f"Auto-adjusting thresholds - H:{h_scale:.2f}x, V-Up:{v_up_scale:.2f}x, V-Down:{v_down_scale:.2f}x")
    
    # Calculate thresholds based on the difference between positions
    new_h_threshold = GAZE_H_THRESHOLD  # Default to current value
    new_v_threshold_up = GAZE_V_THRESHOLD_UP
    new_v_threshold_down = GAZE_V_THRESHOLD_DOWN
    new_vertical_bias = VERTICAL_BIAS
    
    # For horizontal gaze, we compute the midpoint between center and each extreme
    if 'Center' in averages and 'Left' in averages:
        # Use 70% of the distance as the threshold (empirically determined)
        left_threshold = abs(averages['Center']['x'] - averages['Left']['x']) * 0.7
        new_h_threshold = max(0.003, left_threshold)  # Ensure minimum sensitivity
        
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST and 'Left' in errors:
            new_h_threshold *= h_scale
        
        # Store midpoint for classification
        left_midpoint = (averages['Center']['x'] + averages['Left']['x']) / 2
    
    if 'Center' in averages and 'Right' in averages:
        right_threshold = abs(averages['Center']['x'] - averages['Right']['x']) * 0.7
        # Take the average of left and right thresholds for better symmetry
        if 'Left' in averages:
            new_h_threshold = max(0.003, (left_threshold + right_threshold) / 2)
        else:
            new_h_threshold = max(0.003, right_threshold)
        
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST and 'Right' in errors:
            new_h_threshold *= h_scale
        
        # Store midpoint for classification
        right_midpoint = (averages['Center']['x'] + averages['Right']['x']) / 2
    
    # For vertical gaze with enhanced sensitivity
    if 'Center' in averages and 'Up' in averages:
        up_threshold = abs(averages['Center']['y'] - averages['Up']['y']) * 0.65  # Slightly more sensitive
        new_v_threshold_up = max(0.003, up_threshold)
        
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST and 'Up' in errors:
            new_v_threshold_up *= v_up_scale
        
        # Store midpoint for classification
        up_midpoint = (averages['Center']['y'] + averages['Up']['y']) / 2
    
    if 'Center' in averages and 'Down' in averages:
        down_threshold = abs(averages['Center']['y'] - averages['Down']['y']) * 0.65  # Slightly more sensitive
        new_v_threshold_down = max(0.002, down_threshold)
        
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST and 'Down' in errors:
            new_v_threshold_down *= v_down_scale
        
        # Store midpoint for classification
        down_midpoint = (averages['Center']['y'] + averages['Down']['y']) / 2
    
    # Calculate vertical bias more intelligently
    if 'Center' in averages:
        # Bias is calculated for the center position
        new_vertical_bias = -averages['Center']['y'] * 0.8  # Apply a 0.8 factor to avoid over-correction
        
        # Adjust bias based on asymmetry between up and down if both are available
        if 'Up' in averages and 'Down' in averages:
            up_distance = abs(averages['Center']['y'] - averages['Up']['y'])
            down_distance = abs(averages['Center']['y'] - averages['Down']['y'])
            
            # If there's significant asymmetry, adjust the bias to compensate
            if abs(up_distance - down_distance) > 0.01:
                if up_distance > down_distance:
                    # Up is further from center, increase bias to compensate
                    new_vertical_bias += 0.005  # Reduced from 0.01 for less aggressive correction
                else:
                    # Down is further from center, decrease bias
                    new_vertical_bias -= 0.005  # Reduced from 0.01 for less aggressive correction
    
    # Apply weighted average between old and new values for stability
    # This prevents drastic changes between calibration attempts
    GAZE_H_THRESHOLD = (CALIBRATION_DAMPING_FACTOR * new_h_threshold) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_h_threshold)
    GAZE_V_THRESHOLD_UP = (CALIBRATION_DAMPING_FACTOR * new_v_threshold_up) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_v_threshold_up)
    GAZE_V_THRESHOLD_DOWN = (CALIBRATION_DAMPING_FACTOR * new_v_threshold_down) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_v_threshold_down)
    VERTICAL_BIAS = (CALIBRATION_DAMPING_FACTOR * new_vertical_bias) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_vertical_bias)
    
    # Store calculated values for midpoints
    midpoints = {
        'left': left_midpoint if 'Left' in averages and 'Center' in averages else None,
        'right': right_midpoint if 'Right' in averages and 'Center' in averages else None,
        'up': up_midpoint if 'Up' in averages and 'Center' in averages else None,
        'down': down_midpoint if 'Down' in averages and 'Center' in averages else None
    }
    
    # Store all calculated values
    CALIBRATION_DATA = {
        'h_threshold': GAZE_H_THRESHOLD,
        'v_threshold_up': GAZE_V_THRESHOLD_UP,
        'v_threshold_down': GAZE_V_THRESHOLD_DOWN,
        'vertical_bias': VERTICAL_BIAS,
        'use_alternative_method': USE_ALTERNATIVE_METHOD,
        'averages': averages,
        'expected_positions': expected_positions,
        'midpoints': midpoints
    }
    
    print("Calibration values calculated:")
    print(f"Horizontal threshold: {GAZE_H_THRESHOLD:.4f}")
    print(f"Upward threshold: {GAZE_V_THRESHOLD_UP:.4f}")
    print(f"Downward threshold: {GAZE_V_THRESHOLD_DOWN:.4f}")
    print(f"Vertical bias: {VERTICAL_BIAS:.4f}")

def start_validation():
    """Start the validation phase after calibration"""
    global CALIBRATION_STATE, VALIDATION_SEQUENCE, VALIDATION_CURRENT, VALIDATION_CORRECT
    
    # Create a random sequence of positions for validation with more emphasis on problematic directions
    VALIDATION_SEQUENCE = []
    # Make sure we test each position at least once
    for pos in CALIBRATION_POSITIONS:
        VALIDATION_SEQUENCE.append(pos)
    
    # Add some more random positions to make 10 total
    while len(VALIDATION_SEQUENCE) < 10:
        VALIDATION_SEQUENCE.append(random.choice(CALIBRATION_POSITIONS))
    
    # Shuffle the sequence
    random.shuffle(VALIDATION_SEQUENCE)
    
    CALIBRATION_STATE = "VALIDATING"
    VALIDATION_CURRENT = 0
    VALIDATION_CORRECT = 0
    
    print(f"Validation started. Look {VALIDATION_SEQUENCE[VALIDATION_CURRENT]}")

def check_validation(gaze_h, gaze_v):
    """Check if the user is looking at the correct position during validation"""
    global VALIDATION_CURRENT, VALIDATION_CORRECT, CALIBRATION_STATE, calibration_attempt
    
    current_target = VALIDATION_SEQUENCE[VALIDATION_CURRENT]
    correct = False
    confidence_score = 0.0
    
    # Get current measured iris positions
    iris_x = current_iris_relative_x
    iris_y = current_iris_relative_y
    
    # Calculate distance to expected position for more flexible validation
    if current_target in CALIBRATION_DATA['averages']:
        target_x = CALIBRATION_DATA['averages'][current_target]['x']
        target_y = CALIBRATION_DATA['averages'][current_target]['y']
        
        # Calculate normalized distance (0.0 = perfect match, 1.0 = far away)
        distance_x = abs(iris_x - target_x) / max(GAZE_H_THRESHOLD, 0.001)
        distance_y = abs(iris_y - target_y) / max(GAZE_V_THRESHOLD_UP, GAZE_V_THRESHOLD_DOWN, 0.001)
        
        # Higher tolerance for vertical gaze which is naturally less precise
        if current_target in ["Up", "Down"]:
            distance = (distance_x * 0.3) + (distance_y * 0.7)  # Weighted toward vertical for Up/Down
        elif current_target in ["Left", "Right"]:
            distance = (distance_x * 0.7) + (distance_y * 0.3)  # Weighted toward horizontal for Left/Right
        else:  # Center
            distance = (distance_x + distance_y) / 2.0  # Equal weight
        
        # Convert distance to confidence score (1.0 = perfect match, 0.0 = far away)
        confidence_score = max(0.0, 1.0 - distance)
        
        # Consider correct if confidence is above threshold
        if confidence_score > CALIBRATION_TOLERANCE:
            correct = True
    else:
        # Fallback to original logic if target position wasn't saved
        # Map detected gaze to directions
        detected_h = "Center"
        if gaze_h == "Left":
            detected_h = "Left"
        elif gaze_h == "Right":
            detected_h = "Right"
        
        detected_v = "Center"
        if gaze_v == "Up":
            detected_v = "Up"
        elif gaze_v == "Down":
            detected_v = "Down"
        
        # Check if the detected direction matches the target
        if current_target == "Center" and detected_h == "Center" and detected_v == "Center":
            correct = True
        elif current_target == "Left" and detected_h == "Left":
            correct = True
        elif current_target == "Right" and detected_h == "Right":
            correct = True
        elif current_target == "Up" and detected_v == "Up":
            correct = True
        elif current_target == "Down" and detected_v == "Down":
            correct = True
    
    # Map detected gaze to directions for display purpose
    detected_h = gaze_h
    detected_v = gaze_v
    
    if correct:
        VALIDATION_CORRECT += 1
        print(f"Correct! {VALIDATION_CORRECT} in a row. (Confidence: {confidence_score:.2f})")
    else:
        VALIDATION_CORRECT = 0
        print(f"Incorrect. Expected: {current_target}, Detected: {detected_h}-{detected_v} (Confidence: {confidence_score:.2f})")
    
    # Move to next validation target
    VALIDATION_CURRENT += 1
    if VALIDATION_CURRENT >= len(VALIDATION_SEQUENCE):
        # We've gone through all validation targets
        if VALIDATION_CORRECT >= VALIDATION_REQUIRED:
            # Success - save settings and complete
            complete_calibration()
        else:
            # If we haven't met the validation criteria and have attempts left, restart calibration
            if calibration_attempt < MAX_CALIBRATION_ATTEMPTS:
                print(f"Validation unsuccessful. Starting calibration attempt {calibration_attempt+1}/{MAX_CALIBRATION_ATTEMPTS}")
                start_calibration()
            else:
                # We've used all our attempts, save the best we have
                complete_calibration()
    else:
        # If we have enough consecutive correct validations, we're done
        if VALIDATION_CORRECT >= VALIDATION_REQUIRED:
            complete_calibration()
        else:
            print(f"Look {VALIDATION_SEQUENCE[VALIDATION_CURRENT]}")

def complete_calibration():
    """Complete the calibration process and save settings"""
    global CALIBRATION_MODE, CALIBRATION_STATE, calibration_attempt
    
    save_config()
    
    CALIBRATION_MODE = False
    CALIBRATION_STATE = "COMPLETE"
    calibration_attempt = 0
    
    print("Calibration complete and settings saved!")

def save_config():
    """Save the calibration data to a config file"""
    config = {
        'h_threshold': GAZE_H_THRESHOLD,
        'v_threshold_up': GAZE_V_THRESHOLD_UP,
        'v_threshold_down': GAZE_V_THRESHOLD_DOWN,
        'vertical_bias': VERTICAL_BIAS,
        'use_alternative_method': USE_ALTERNATIVE_METHOD,
        'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving config: {e}")

def load_config():
    """Load calibration data from config file"""
    global GAZE_H_THRESHOLD, GAZE_V_THRESHOLD_UP, GAZE_V_THRESHOLD_DOWN, VERTICAL_BIAS, USE_ALTERNATIVE_METHOD
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            GAZE_H_THRESHOLD = config.get('h_threshold', GAZE_H_THRESHOLD)
            GAZE_V_THRESHOLD_UP = config.get('v_threshold_up', GAZE_V_THRESHOLD_UP)
            GAZE_V_THRESHOLD_DOWN = config.get('v_threshold_down', GAZE_V_THRESHOLD_DOWN)
            VERTICAL_BIAS = config.get('vertical_bias', VERTICAL_BIAS)
            USE_ALTERNATIVE_METHOD = config.get('use_alternative_method', USE_ALTERNATIVE_METHOD)
            
            print(f"Loaded configuration from {CONFIG_FILE}")
            print(f"Horizontal threshold: {GAZE_H_THRESHOLD:.4f}")
            print(f"Upward threshold: {GAZE_V_THRESHOLD_UP:.4f}")
            print(f"Downward threshold: {GAZE_V_THRESHOLD_DOWN:.4f}")
            print(f"Vertical bias: {VERTICAL_BIAS:.4f}")
            print(f"Using {'alternative' if USE_ALTERNATIVE_METHOD else 'standard'} method")
            
            return True
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return False

def get_target_position(position, frame_size):
    """Get the x,y coordinates for a target position on the screen
    
    Args:
        position: String position name (Center, Left, Right, Up, Down)
        frame_size: Tuple of (width, height) of the screen
        
    Returns:
        Tuple of (x, y) coordinates for the target
    """
    width, height = frame_size
    center_x = width // 2
    center_y = height // 2
    
    # Position targets with some padding from edges
    padding_x = width // 6
    padding_y = height // 6
    
    if position == "Center":
        return center_x, center_y
    elif position == "Left":
        return padding_x, center_y
    elif position == "Right":
        return width - padding_x, center_y
    elif position == "Up":
        return center_x, padding_y
    elif position == "Down":
        return center_x, height - padding_y
    else:
        # Default to center if unknown position
        return center_x, center_y

def draw_calibration_target(frame, position, time_elapsed, time_per_position, sample_count=0):
    """Draw an animated calibration target at the specified position
    
    Args:
        frame: The frame to draw on
        position: The position name (Center, Left, Right, Up, Down)
        time_elapsed: Elapsed time for this position in seconds
        time_per_position: Total time allowed for this position
        sample_count: Number of samples collected so far
    """
    h, w, _ = frame.shape
    target_x, target_y = get_target_position(position, (w, h))
    
    # Calculate progress
    progress_percent = min(100, (time_elapsed / time_per_position) * 100)
    time_remaining = max(0, time_per_position - time_elapsed)
    
    # Draw animated target with time-based animation
    outer_radius = 30
    inner_radius = 15
    
    # Create pulsing effect based on time
    pulse_rate = 3  # pulses per second
    pulse_phase = (time.time() * pulse_rate) % 1.0  # 0.0 to 1.0
    pulse_size = 1.0 + 0.3 * math.sin(pulse_phase * 2 * math.pi)  # 0.7 to 1.3
    
    # Animated circle that gets smaller as time counts down
    countdown_factor = 1.0 - (time_elapsed / time_per_position) * 0.5
    
    # Draw multiple circles for better visibility
    # Outer animated circle
    cv2.circle(frame, (target_x, target_y), 
               int(outer_radius * pulse_size * countdown_factor), 
               (0, 0, 255), 2)
    
    # Middle circle (fixed)
    cv2.circle(frame, (target_x, target_y), 20, (0, 128, 255), 1)
    
    # Inner circle (target)
    cv2.circle(frame, (target_x, target_y), 
               int(inner_radius * countdown_factor), 
               (255, 255, 255), -1)
    
    # Draw position label
    cv2.putText(frame, position, (target_x - 30, target_y - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show progress text
    progress_text = f"{time_remaining:.1f}s remaining - {sample_count} samples"
    cv2.putText(frame, progress_text, (target_x - 120, target_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw progress bar
    bar_width = 200
    bar_height = 10
    bar_x = target_x - bar_width // 2
    bar_y = target_y + 70
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (100, 100, 100), -1)
    # Progress fill
    fill_width = int(bar_width * progress_percent / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                  (0, 255, 0), -1)

# Try to load config at startup
load_config()

# Initialize Pygame
pygame.init()
pygame.display.set_caption("Advanced Eye Tracker")

# Default to fullscreen mode for better calibration
FULLSCREEN = True
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

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
if FULLSCREEN:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    display_width, display_height = screen.get_size()
else:
    display_width, display_height = 1024, 768  # Default window size
    display_scale = min(display_width / img_width, display_height / img_height)
    display_width = int(img_width * display_scale)
    display_height = int(img_height * display_scale)
    screen = pygame.display.set_mode((display_width, display_height))

print(f"Display size: {display_width}x{display_height}")

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

# For gaze prediction visualization
predicted_gaze_x = None
predicted_gaze_y = None
show_gaze_point = True  # Show the predicted gaze point by default

# Calibration retry settings
MAX_CALIBRATION_ATTEMPTS = 3
calibration_attempt = 0

# Eye aspect ratio threshold for blink detection
EAR_THRESHOLD = 0.2

# Variables for gaze smoothing
gaze_h_history = []
gaze_v_history = []
iris_x_history = []
iris_y_history = []
max_history = 10

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
    """Estimate gaze direction based on iris position relative to eye corners
    with improved accuracy and sensitivity
    """
    global current_iris_relative_x, current_iris_relative_y, predicted_gaze_x, predicted_gaze_y
    
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
    
    # HORIZONTAL GAZE CALCULATION
    # Calculate the center of each eye for horizontal gaze
    left_eye_center_x = (left_eye_left.x + left_eye_right.x) / 2
    right_eye_center_x = (right_eye_left.x + right_eye_right.x) / 2
    
    # Calculate eye width for normalization
    left_eye_width = abs(left_eye_right.x - left_eye_left.x)
    right_eye_width = abs(right_eye_right.x - right_eye_left.x)
    
    # Check for zero width to avoid division by zero
    if left_eye_width < 0.001 or right_eye_width < 0.001:
        # Eyes are likely closed or not tracked well
        return "Center", "Center", 0.0, 0.0
    
    # Calculate relative position of iris to eye center (horizontal)
    # Normalize by eye width to account for different face sizes and distances
    left_iris_relative_x = (left_iris.x - left_eye_center_x) / left_eye_width
    right_iris_relative_x = (right_iris.x - right_eye_center_x) / right_eye_width
    
    # Weight eyes by width to give more importance to the larger/more visible eye
    left_weight = left_eye_width / (left_eye_width + right_eye_width)
    right_weight = right_eye_width / (left_eye_width + right_eye_width)
    
    # Calculate weighted average for more reliable horizontal position
    iris_relative_x = (left_iris_relative_x * left_weight) + (right_iris_relative_x * right_weight)
    
    # VERTICAL GAZE CALCULATION
    iris_relative_y = 0
    
    if USE_ALTERNATIVE_METHOD:
        # Alternative method: Improved relative position calculation using the fraction
        # of space between the top and bottom of the eye
        
        # Calculate eye heights
        left_eye_height = left_eye_bottom.y - left_eye_top.y
        right_eye_height = right_eye_bottom.y - right_eye_top.y
        
        # Check for zero height to avoid division by zero
        if left_eye_height < 0.001 or right_eye_height < 0.001:
            # Eyes are likely closed or not tracked well
            return "Center", "Center", iris_relative_x, 0.0
        
        # Calculate how far the iris is from the top of the eye, as a fraction of total eye height
        # 0 = iris at top, 0.5 = iris in middle, 1 = iris at bottom
        left_iris_relative_pos = (left_iris.y - left_eye_top.y) / left_eye_height
        right_iris_relative_pos = (right_iris.y - right_eye_top.y) / right_eye_height
        
        # Apply weight by eye height to give more importance to the more open eye
        left_height_weight = left_eye_height / (left_eye_height + right_eye_height)
        right_height_weight = right_eye_height / (left_eye_height + right_eye_height)
        
        # Weighted average of relative positions
        avg_rel_pos = (left_iris_relative_pos * left_height_weight) + (right_iris_relative_pos * right_height_weight)
        
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
        
        # Check for zero height
        if left_eye_height < 0.001 or right_eye_height < 0.001:
            return "Center", "Center", iris_relative_x, 0.0
        
        # Calculate relative position of iris to eye center (vertical)
        # Normalize by eye height
        left_iris_relative_y = (left_iris.y - left_eye_center_y) / left_eye_height
        right_iris_relative_y = (right_iris.y - right_eye_center_y) / right_eye_height
        
        # Weight by eye height
        left_height_weight = left_eye_height / (left_eye_height + right_eye_height)
        right_height_weight = right_eye_height / (left_eye_height + right_eye_height)
        
        # Weighted average for vertical position
        iris_relative_y = (left_iris_relative_y * left_height_weight) + (right_iris_relative_y * right_height_weight)
    
    # Apply vertical bias to compensate for eye physiology
    iris_relative_y += VERTICAL_BIAS
    
    # Store current values for debugging
    current_iris_relative_x = iris_relative_x
    current_iris_relative_y = iris_relative_y
    
    # Calculate predicted gaze point on screen
    # Use a more accurate scaling method based on calibration data
    if 'averages' in CALIBRATION_DATA and len(CALIBRATION_DATA['averages']) >= 3:
        # If we have calibration data, use it for more accurate prediction
        # Calculate range of calibrated x and y values
        cal_x_values = [pos_data['x'] for pos_data in CALIBRATION_DATA['averages'].values()]
        cal_y_values = [pos_data['y'] for pos_data in CALIBRATION_DATA['averages'].values()]
        
        x_min, x_max = min(cal_x_values), max(cal_x_values)
        y_min, y_max = min(cal_y_values), max(cal_y_values)
        
        # Calculate safe ranges with some margin
        x_range = max(0.01, x_max - x_min) * 1.2  # Add 20% margin
        y_range = max(0.01, y_max - y_min) * 1.2
        
        # Map iris position to screen using calibration range
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        # Scale factor based on calibration (higher = more sensitive)
        x_scale = frame_width * 0.8 / x_range
        y_scale = frame_height * 0.8 / y_range
        
        # Calculate screen coordinates using calibrated center position
        if 'Center' in CALIBRATION_DATA['averages']:
            center_offset_x = CALIBRATION_DATA['averages']['Center']['x']
            center_offset_y = CALIBRATION_DATA['averages']['Center']['y']
            
            predicted_gaze_x = int(center_x + (iris_relative_x - center_offset_x) * x_scale)
            predicted_gaze_y = int(center_y + (iris_relative_y - center_offset_y) * y_scale)
        else:
            # Fallback if center calibration is missing
            predicted_gaze_x = int(center_x + iris_relative_x * x_scale)
            predicted_gaze_y = int(center_y + iris_relative_y * y_scale)
    else:
        # Default mapping if no calibration data
        x_range = frame_width * 0.8
        y_range = frame_height * 0.8
        
        predicted_gaze_x = int(frame_width/2 + (iris_relative_x * 3 * x_range))
        predicted_gaze_y = int(frame_height/2 + (iris_relative_y * 3 * y_range))
    
    # Ensure the prediction stays within screen bounds
    predicted_gaze_x = max(0, min(frame_width, predicted_gaze_x))
    predicted_gaze_y = max(0, min(frame_height, predicted_gaze_y))
    
    # Add to history for smoothing
    iris_x_history.append(iris_relative_x)
    iris_y_history.append(iris_relative_y)
    
    if len(iris_x_history) > max_history:
        iris_x_history.pop(0)
    if len(iris_y_history) > max_history:
        iris_y_history.pop(0)
    
    # Use the average of recent values for more stability
    if iris_x_history and iris_y_history:
        # Weight more recent samples higher
        weights = [0.7 + 0.3 * (i / len(iris_x_history)) for i in range(len(iris_x_history))]
        total_weight = sum(weights)
        
        # Weighted moving average
        smoothed_x = sum(x * w for x, w in zip(iris_x_history, weights)) / total_weight
        smoothed_y = sum(y * w for y, w in zip(iris_y_history, weights)) / total_weight
        
        iris_relative_x = smoothed_x
        iris_relative_y = smoothed_y
    
    # Determine horizontal gaze direction using calibration midpoints if available
    h_direction = "Center"
    if 'midpoints' in CALIBRATION_DATA:
        if CALIBRATION_DATA['midpoints']['left'] is not None and iris_relative_x < CALIBRATION_DATA['midpoints']['left']:
            h_direction = "Left"
        elif CALIBRATION_DATA['midpoints']['right'] is not None and iris_relative_x > CALIBRATION_DATA['midpoints']['right']:
            h_direction = "Right"
    else:
        # Fallback to threshold-based detection
        if iris_relative_x < -GAZE_H_THRESHOLD:
            h_direction = "Left"
        elif iris_relative_x > GAZE_H_THRESHOLD:
            h_direction = "Right"
    
    # Determine vertical gaze direction using calibration midpoints if available
    v_direction = "Center"
    if 'midpoints' in CALIBRATION_DATA:
        if CALIBRATION_DATA['midpoints']['up'] is not None and iris_relative_y < CALIBRATION_DATA['midpoints']['up']:
            v_direction = "Up"
        elif CALIBRATION_DATA['midpoints']['down'] is not None and iris_relative_y > CALIBRATION_DATA['midpoints']['down']:
            v_direction = "Down"
    else:
        # Fallback to asymmetric thresholds for upward and downward gaze
        if iris_relative_y < -GAZE_V_THRESHOLD_UP:
            v_direction = "Up"
        elif iris_relative_y > GAZE_V_THRESHOLD_DOWN:
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
    global current_iris_relative_x, current_iris_relative_y, predicted_gaze_x, predicted_gaze_y
    global CALIBRATION_POSITION_START_TIME, CURRENT_CALIBRATION_POS, CALIBRATION_STATE, warmup_counter
    
    # Convert the frame colors from BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find face landmarks
    results = face_mesh.process(frame_rgb)
    
    h, w, _ = frame.shape
    
    # Check warmup completion
    if CALIBRATION_MODE and CALIBRATION_STATE == "WARMUP":
        warmup_counter += 1
        if warmup_counter >= CALIBRATION_WARMUP_FRAMES:
            CALIBRATION_STATE = "SAMPLING"
            CALIBRATION_POSITION_START_TIME = time.time()  # Reset timer when moving to sampling
            print(f"Warmup complete. Now look at the {CALIBRATION_POSITIONS[CURRENT_CALIBRATION_POS]} of the screen.")
    
    # Check if we need to force position change due to timeout
    if CALIBRATION_MODE and CALIBRATION_STATE == "SAMPLING":
        current_time = time.time()
        elapsed_time = current_time - CALIBRATION_POSITION_START_TIME
        if elapsed_time >= CALIBRATION_TIME_PER_POSITION and CURRENT_CALIBRATION_POS < len(CALIBRATION_POSITIONS):
            print(f"Time elapsed for {CALIBRATION_POSITIONS[CURRENT_CALIBRATION_POS]}, moving to next position")
            move_to_next_calibration_position()
            
    # Collect sample if in sampling state and face is found
    if CALIBRATION_MODE and CALIBRATION_STATE == "SAMPLING" and results.multi_face_landmarks and not is_blinking:
        face_landmarks = results.multi_face_landmarks[0]
        # Estimate gaze direction using current face landmarks
        _, _, iris_rel_x, iris_rel_y = estimate_gaze(face_landmarks, w, h)
        # Collect calibration sample
        collect_calibration_sample(iris_rel_x, iris_rel_y)
    
    # Draw calibration overlay if in calibration mode
    if CALIBRATION_MODE:
        # Draw a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw calibration instructions
        instruction_text = ""
        
        if CALIBRATION_STATE == "WARMUP":
            instruction_text = f"Look at the center - Stabilizing eye tracking ({warmup_counter}/{CALIBRATION_WARMUP_FRAMES})"
            
            # Draw center target
            center_x, center_y = w//2, h//2
            
            # Draw animated warming up indicator
            progress = min(1.0, warmup_counter / CALIBRATION_WARMUP_FRAMES)
            radius = 30
            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
            # Draw progress arc
            angle = int(360 * progress)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       -90, 0, angle, (0, 255, 0), 3)
            
            # Draw inner target
            cv2.circle(frame, (center_x, center_y), 10, (255, 255, 255), -1)
            
            # Show warmup progress
            progress_text = f"Warmup progress: {int(progress * 100)}%"
            cv2.putText(frame, progress_text, (center_x - 100, center_y + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elif CALIBRATION_STATE == "SAMPLING":
            current_pos = CALIBRATION_POSITIONS[CURRENT_CALIBRATION_POS]
            
            # Calculate elapsed time
            time_elapsed = time.time() - CALIBRATION_POSITION_START_TIME
            
            # Get the number of samples collected for this position
            sample_count = 0
            if current_pos in CALIBRATION_SAMPLES:
                sample_count = len(CALIBRATION_SAMPLES[current_pos]['x'])
            
            instruction_text = f"Calibration - Look at the {current_pos} target ({CURRENT_CALIBRATION_POS+1}/{len(CALIBRATION_POSITIONS)})"
            
            # Draw the target with all visual indicators
            draw_calibration_target(frame, current_pos, time_elapsed, CALIBRATION_TIME_PER_POSITION, sample_count)
            
            # If we have samples, visualize them
            if sample_count > 0:
                # Draw all collected sample points to visualize coverage
                for i, (x, y) in enumerate(zip(CALIBRATION_SAMPLES[current_pos]['x'], CALIBRATION_SAMPLES[current_pos]['y'])):
                    # Convert normalized coordinates to pixel coordinates
                    pixel_x = int(w/2 + x * w)
                    pixel_y = int(h/2 + y * h)
                    
                    # Make more recent points brighter and larger
                    # 0-10 most recent points are white, older points fade to gray
                    point_age = sample_count - i
                    color_intensity = max(50, 255 - point_age * 15)
                    size = max(1, 3 - point_age // 5)
                    
                    # Skip every other point to avoid cluttering
                    if i % 2 == 0:
                        cv2.circle(frame, (pixel_x, pixel_y), size, (color_intensity, color_intensity, color_intensity), -1)
        
        elif CALIBRATION_STATE == "VALIDATING":
            current_target = VALIDATION_SEQUENCE[VALIDATION_CURRENT]
            instruction_text = f"Validation - Look {current_target} ({VALIDATION_CURRENT+1}/{len(VALIDATION_SEQUENCE)})"
            
            # Draw target indicator
            target_x, target_y = get_target_position(current_target, (w, h))
            
            # Draw animated target with different color for validation
            # Increase visibility with concentric circles
            cv2.circle(frame, (target_x, target_y), 40, (0, 255, 255), 1)
            cv2.circle(frame, (target_x, target_y), 30, (0, 255, 255), 2)
            cv2.circle(frame, (target_x, target_y), 20, (0, 255, 255), 1)
            cv2.circle(frame, (target_x, target_y), 10, (255, 255, 255), -1)
            
            # Draw position label
            cv2.putText(frame, current_target, (target_x - 30, target_y - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show validation progress
            progress_text = f"Correct: {VALIDATION_CORRECT}/{VALIDATION_REQUIRED}"
            cv2.putText(frame, progress_text, (target_x - 80, target_y + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show overall validation progress
            overall_progress = f"Validation progress: {VALIDATION_CURRENT+1}/{len(VALIDATION_SEQUENCE)}"
            cv2.putText(frame, overall_progress, (w//2 - 150, h - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        elif CALIBRATION_STATE == "COMPLETE":
            instruction_text = "Calibration Complete!"
            
            # Display a checkmark or success message
            check_text = "✓ Settings Saved"
            cv2.putText(frame, check_text, (w//2 - 100, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Draw instruction text at the top of the screen
        cv2.putText(frame, instruction_text, (w//2 - 250, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        # Draw a "Cancel Calibration" message
        cv2.putText(frame, "Press 'C' again to cancel calibration", (w//2 - 200, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
                
                if predicted_gaze_x is not None and predicted_gaze_y is not None:
                    cv2.putText(frame, f"Pred X,Y: {predicted_gaze_x},{predicted_gaze_y}", (10, 270),
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
            
            # Draw the gaze prediction dot if we have valid coordinates
            if show_gaze_point and predicted_gaze_x is not None and predicted_gaze_y is not None:
                # Draw a crosshair at the predicted gaze point
                crosshair_size = 15
                crosshair_color = (0, 255, 0)  # Green
                cv2.line(frame, (predicted_gaze_x - crosshair_size, predicted_gaze_y), 
                         (predicted_gaze_x + crosshair_size, predicted_gaze_y), crosshair_color, 2)
                cv2.line(frame, (predicted_gaze_x, predicted_gaze_y - crosshair_size), 
                         (predicted_gaze_x, predicted_gaze_y + crosshair_size), crosshair_color, 2)
                cv2.circle(frame, (predicted_gaze_x, predicted_gaze_y), 5, crosshair_color, -1)
            
            # Collect calibration samples if in calibration mode
            if CALIBRATION_MODE and not is_blinking:
                if CALIBRATION_STATE == "SAMPLING":
                    collect_calibration_sample(iris_rel_x, iris_rel_y)
                elif CALIBRATION_STATE == "VALIDATING":
                    check_validation(gaze_horizontal, gaze_vertical)
            
            # Log data
            log_data(time.time(), left_ear, right_ear, avg_ear, is_blinking, blink_count,
                     gaze_horizontal, gaze_vertical, left_pupil_x, left_pupil_y, right_pupil_x, right_pupil_y,
                     current_iris_relative_x, current_iris_relative_y)
            
            # If data logging is enabled, display indicator
            if ENABLE_LOGGING:
                cv2.putText(frame, "Recording Data", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw key controls at the bottom of the frame
    cv2.putText(frame, "C: Calibrate | D: Debug | T: Thresholds | P: Toggle Gaze Dot | F: Toggle Fullscreen | ESC: Exit", 
                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
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
            if event.key == pygame.K_p:  # Toggle gaze prediction dot
                show_gaze_point = not show_gaze_point
                print(f"Gaze prediction dot: {'ON' if show_gaze_point else 'OFF'}")
            if event.key == pygame.K_f:  # Toggle fullscreen
                FULLSCREEN = not FULLSCREEN
                if FULLSCREEN:
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    display_width, display_height = screen.get_size()
                else:
                    display_width, display_height = 1024, 768
                    screen = pygame.display.set_mode((display_width, display_height))
                print(f"Fullscreen: {'ON' if FULLSCREEN else 'OFF'}")
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
            if event.key == pygame.K_c:  # Start/stop calibration
                if not CALIBRATION_MODE:
                    start_calibration()
                    print("Starting calibration mode. Follow the on-screen targets.")
                else:
                    CALIBRATION_MODE = False
                    CALIBRATION_STATE = "WAITING"
                    print("Calibration cancelled.")
            if event.key == pygame.K_s:  # Save current configuration
                save_config()
                print("Configuration saved.")
            if event.key == pygame.K_o:  # Load saved configuration
                load_config()
            
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