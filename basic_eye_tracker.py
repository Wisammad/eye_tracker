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
import glob
from collections import defaultdict

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
CALIBRATION_POSITIONS = [
    "Center",           # 1. Center
    "TopLeft",          # 2. Top left
    "TopCenter",        # 3. Top center
    "TopRight",         # 4. Top right
    "MiddleLeft",       # 5. Middle left
    "MiddleRight",      # 6. Middle right
    "BottomLeft",       # 7. Bottom left
    "BottomCenter",     # 8. Bottom center
    "BottomRight"       # 9. Bottom right
]  # Positions to calibrate in 3x3 grid
CURRENT_CALIBRATION_POS = 0         # Current position in calibration sequence
CALIBRATION_SAMPLES = {}            # Collected samples during calibration
CALIBRATION_SAMPLES_REQUIRED = 30   # Number of samples to collect per position
CALIBRATION_TIME_PER_POSITION = 3   # Seconds to spend on each calibration position
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
CONSOLE_DEBUG = False  # Set to True to see detailed console output

# Use alternative calculation method for vertical gaze
USE_ALTERNATIVE_METHOD = True  # Use a completely different method for vertical gaze

# Gaze thresholds that will be calibrated
GAZE_H_THRESHOLD = 0.0316  # Horizontal threshold (updated from 0.005)
GAZE_V_THRESHOLD_UP = 0.0383  # Upward threshold (updated from 0.010)
GAZE_V_THRESHOLD_DOWN = 0.0220  # Downward threshold (reduced to improve downward detection)
VERTICAL_BIAS = 0.0450  # Vertical bias (reduced from 0.0600 to fix upward pulling)

# Add these with other calibration settings
# Sector-based calibration
USE_SECTOR_CALIBRATION = True  # Whether to use sector-specific calibration
SCREEN_SECTORS = [
    "TopLeft", "TopCenter", "TopRight",
    "MiddleLeft", "Center", "MiddleRight",
    "BottomLeft", "BottomCenter", "BottomRight"
]  # 3x3 grid sectors matching our calibration positions
SECTOR_CALIBRATION_DATA = {}  # Will store sector-specific calibration data

# Add this with other global variables
SECTOR_DISTANCE_WEIGHT = 1.0  # Weight factor for sector distance sensitivity

# Add this with other global variables
current_eye_openness = 0.0  # Will track eye vertical openness

# Add these global variables
DOWN_GAZE_SENSITIVITY = 1.0  # Adjustable sensitivity for downward gaze detection
SHOW_TUNING_DIALOG = False  # Whether to show the tuning dialog

# For cursor visualization and stability
CURSOR_STABLE_FRAMES = 3  # Number of frames to average for additional cursor stability
cursor_history = []  # Store recent cursor positions

# Attention testing settings 
ATTENTION_TEST_MODE = False         # Toggle for attention test mode
CURRENT_TEST_IMAGE = None           # Currently displayed test image
CURRENT_TEST_IMAGE_PATH = None      # Path to current test image
CURRENT_TEST_IMAGE_START_TIME = 0   # When current test image started displaying
TEST_IMAGE_DURATION = 5             # How long to show each test image in seconds
ATTENTION_HEATMAP_DATA = {}         # Store gaze points for heatmap generation
ATTENTION_TEST_RESULTS = {}         # Store test results
ATTENTION_CATEGORIES = []           # List of available categories for testing
CURRENT_CATEGORY_INDEX = 0          # Current category being tested
ATTENTION_TEST_STATE = "WAITING"    # WAITING, CATEGORY_SELECTION, TESTING, SHOWING_HEATMAP, RESULTS
ATTENTION_TEST_IMAGES = []          # List of images for current category
CURRENT_IMAGE_INDEX = 0             # Current image index in test
HEATMAP_OPACITY = 0.7               # Heatmap overlay opacity
GAUSSIAN_SIGMA = 25                 # Gaussian blur sigma for heatmap generation
MAX_GAZE_POINTS = 100               # Maximum number of gaze points to store per image
CURRENT_HEATMAP = None              # Currently displayed heatmap
CURRENT_HEATMAP_START_TIME = 0      # When the current heatmap started displaying

# Attention test folder paths (these will be populated at runtime)
CATEGORIES_DIR = "./Categories"
ATTENTION_TEST_RESULTS_DIR = "./attention_test_results"

def initialize_sector_calibration():
    """Initialize sector-specific calibration data structure"""
    global SECTOR_CALIBRATION_DATA
    
    # Create empty data structure for each sector
    for sector in SCREEN_SECTORS:
        SECTOR_CALIBRATION_DATA[sector] = {
            'reference_iris_x': 0.0,  # Reference iris position when looking at this sector
            'reference_iris_y': 0.0,
            'h_threshold': GAZE_H_THRESHOLD,  # Start with global defaults
            'v_threshold_up': GAZE_V_THRESHOLD_UP,
            'v_threshold_down': GAZE_V_THRESHOLD_DOWN,
            'vertical_bias': VERTICAL_BIAS,
            'samples_count': 0  # Number of samples collected for this sector
        }
    
    print("Initialized sector-based calibration data structure")

# Add this function to apply a stronger vertical bias adjustment for bottom positions
def get_adjusted_vertical_bias(position):
    """Get adjusted vertical bias based on screen position to improve top/bottom detection"""
    global VERTICAL_BIAS
    
    # Default to the standard bias
    bias = VERTICAL_BIAS
    
    # Apply stronger adjustments for bottom positions
    if "Bottom" in position:
        # For bottom positions, reduce bias to make downward detection more likely
        bias = VERTICAL_BIAS * 0.5  # Reduce bias by 50% (increased from 40%) 
    elif "Top" in position:
        # For top positions, increase bias slightly
        bias = VERTICAL_BIAS * 1.1  # Increase bias by 10%
    
    return bias

# Add these functions after the initialize_sector_calibration function

def determine_current_sector(iris_x, iris_y):
    """Determine which screen sector the current gaze is in based on iris position"""
    global CALIBRATION_DATA, SECTOR_CALIBRATION_DATA, SECTOR_DISTANCE_WEIGHT
    
    # If we have calibration data with reference positions, use them to determine the sector
    if 'averages' in CALIBRATION_DATA:
        # Find the closest sector based on calibrated reference positions
        min_distance = float('inf')
        closest_sector = "Center"  # Default to center if no calibration
        
        # Use standard screen layout dimensions for weighting
        h_weight = 1.0 * SECTOR_DISTANCE_WEIGHT
        v_weight = 1.0 * SECTOR_DISTANCE_WEIGHT
        
        # Add bias toward common gaze positions (center is typically more likely)
        position_weights = {
            "Center": 0.9,         # Slight preference for center
            "MiddleLeft": 0.8,     # Increased preference for MiddleLeft (reduced from 0.95)
            "MiddleRight": 0.95,
            "TopCenter": 0.95,
            "BottomCenter": 0.95,
            "TopLeft": 1.0,        # Corners are less common
            "TopRight": 1.0,
            "BottomLeft": 0.7,     # Even stronger preference for BottomLeft (reduced from 0.85)
            "BottomRight": 1.0
        }
        
        for sector, data in CALIBRATION_DATA['averages'].items():
            if sector in SCREEN_SECTORS:  # Only consider valid screen sectors
                # Apply horizontal/vertical weights
                dx = (iris_x - data['x']) * h_weight
                dy = (iris_y - data['y']) * v_weight
                
                # Calculate basic Euclidean distance
                basic_distance = (dx*dx + dy*dy)**0.5
                
                # Apply position-based weighting
                adjusted_distance = basic_distance * position_weights.get(sector, 1.0)
                
                # Add context-based adjustment - use sample quality as factor
                if sector in SECTOR_CALIBRATION_DATA and SECTOR_CALIBRATION_DATA[sector]['samples_count'] > 20:
                    # Slightly prefer sectors with more samples (better calibrated)
                    quality_factor = 0.95  # 5% bonus for well-calibrated sectors
                else:
                    quality_factor = 1.0
                
                final_distance = adjusted_distance * quality_factor
                
                # Special handling for BottomLeft sector which is problematic
                if sector == "BottomLeft":
                    # Apply an additional reduction to make BottomLeft more likely to be selected
                    final_distance = final_distance * 0.4
                # Special handling for MiddleLeft sector which is also problematic
                elif sector == "MiddleLeft":
                    # Apply an additional reduction to make MiddleLeft more likely to be selected
                    final_distance = final_distance * 0.6
                
                # Update if this is closer
                if final_distance < min_distance:
                    min_distance = final_distance
                    closest_sector = sector
        
        # Only log in detailed debug mode to reduce console spam
        if CONSOLE_DEBUG:
            print(f"Selected sector: {closest_sector}, distance: {min_distance:.4f}")
        
        return closest_sector
                
    # Fallback to fixed thresholds if no calibration data
    # These boundaries are approximate and will need tuning
    if iris_x < -0.01:  # Left side (reduced from -0.02)
        if iris_y < -0.01:  # Top (reduced from -0.02)
            return "TopLeft"
        elif iris_y > 0.01:  # Bottom (reduced from 0.02)
            return "BottomLeft"
        else:  # Middle
            return "MiddleLeft"
    elif iris_x > 0.01:  # Right side (reduced from 0.02)
        if iris_y < -0.01:  # Top
            return "TopRight"
        elif iris_y > 0.01:  # Bottom
            return "BottomRight"
        else:  # Middle
            return "MiddleRight"
    else:  # Center column
        if iris_y < -0.01:  # Top
            return "TopCenter"
        elif iris_y > 0.01:  # Bottom
            return "BottomCenter"
        else:  # Middle
            return "Center"

def get_gaze_direction(left_iris_normalized, right_iris_normalized):
    """Calculate the gaze direction based on iris positions"""
    global GAZE_HISTORY, KALMAN_FILTER_X, KALMAN_FILTER_Y, USE_KALMAN_FILTER, SECTOR_CALIBRATION_DATA

    # Average the normalized positions from both eyes
    avg_iris_x = (left_iris_normalized[0] + right_iris_normalized[0]) / 2
    avg_iris_y = (left_iris_normalized[1] + right_iris_normalized[1]) / 2
    
    # Apply Kalman filtering if enabled
    if USE_KALMAN_FILTER and KALMAN_FILTER_X is not None and KALMAN_FILTER_Y is not None:
        filtered_x, _ = KALMAN_FILTER_X.update(avg_iris_x)
        filtered_y, _ = KALMAN_FILTER_Y.update(avg_iris_y)
        avg_iris_x = filtered_x
        avg_iris_y = filtered_y

    # Determine which sector the gaze is currently in
    current_sector = determine_current_sector(avg_iris_x, avg_iris_y)
    
    # Debug values set before determining direction
    direction = "Center"  # Default
    h_direction = "Center"
    v_direction = "Center"
    
    # Use sector-specific calibration if enabled and available for current sector
    if USE_SECTOR_CALIBRATION and current_sector in SECTOR_CALIBRATION_DATA and SECTOR_CALIBRATION_DATA[current_sector]['samples_count'] > 5:
        # Get sector-specific reference positions and thresholds
        ref_x = SECTOR_CALIBRATION_DATA[current_sector]['reference_iris_x']
        ref_y = SECTOR_CALIBRATION_DATA[current_sector]['reference_iris_y']
        h_threshold = SECTOR_CALIBRATION_DATA[current_sector]['h_threshold']
        v_threshold_up = SECTOR_CALIBRATION_DATA[current_sector]['v_threshold_up']
        v_threshold_down = SECTOR_CALIBRATION_DATA[current_sector]['v_threshold_down']
        vertical_bias = SECTOR_CALIBRATION_DATA[current_sector]['vertical_bias']
        
        # Calculate horizontal gaze based on vector from reference position
        h_offset = avg_iris_x - ref_x
        
        if DEBUG_MODE and CONSOLE_DEBUG:
            print(f"Sector: {current_sector}, h_offset: {h_offset:.4f}, thresholds: ±{h_threshold:.4f}")
            print(f"v_offset: {v_offset:.4f}, v_up: {v_threshold_up:.4f}, v_down: {v_threshold_down:.4f}")
            print(f"Using sector {current_sector} calibration - h_thresh: {h_threshold:.4f}, v_up: {v_threshold_up:.4f}, v_down: {v_threshold_down:.4f}, bias: {vertical_bias:.4f}")
        
        if h_offset < -h_threshold:
            h_direction = "Left"
        elif h_offset > h_threshold:
            h_direction = "Right"
        
        # Calculate vertical gaze with sector-specific bias
        # Apply vertical bias to observed position, not reference
        adjusted_bias = get_adjusted_vertical_bias(current_sector)  # Get position-specific bias
        v_adjusted_y = avg_iris_y + adjusted_bias
        v_offset = v_adjusted_y - ref_y
        
        if DEBUG_MODE and CONSOLE_DEBUG:
            print(f"v_offset: {v_offset:.4f}, v_up: {v_threshold_up:.4f}, v_down: {v_threshold_down:.4f}, bias: {adjusted_bias:.4f}")
            
        # Invert the v_offset to fix top/bottom confusion
        v_offset = -v_offset
        
        # Calculate vertical gaze direction - UP is negative, DOWN is positive
        if v_offset < -v_threshold_up:
            v_direction = "Up"
        elif v_offset > v_threshold_down * (0.5 / DOWN_GAZE_SENSITIVITY):  # Make downward detection even more sensitive (0.6->0.5)
            # When looking down, we need to check additional factors
            # Eye openness affects downward gaze detection - the less open, the more sensitive we should be
            eye_openness_factor = min(1.0, current_eye_openness * 35)  # Normalize to 0-1 (increased from 30 to 35)
            # Apply DOWN_GAZE_SENSITIVITY directly to make the effect stronger
            effective_threshold = v_threshold_down * (0.5 / DOWN_GAZE_SENSITIVITY) * eye_openness_factor
            
            if v_offset > effective_threshold:
                v_direction = "Down"
            else:
                v_direction = "Center"
            
            if DEBUG_MODE and CONSOLE_DEBUG:
                print(f"Down detection: v_offset={v_offset:.4f}, threshold={effective_threshold:.4f}, eye_openness={current_eye_openness:.4f}, sensitivity={DOWN_GAZE_SENSITIVITY:.2f}")
        else:
            v_direction = "Center"
        
        if DEBUG_MODE and USE_SECTOR_CALIBRATION:
            if current_sector not in SECTOR_CALIBRATION_DATA:
                print(f"Sector {current_sector} not found in calibration data")
            elif SECTOR_CALIBRATION_DATA[current_sector]['samples_count'] <= 5:
                print(f"Sector {current_sector} has insufficient samples: {SECTOR_CALIBRATION_DATA[current_sector]['samples_count']}")
            
    else:
        # Use global calibration values
        if USE_ALTERNATIVE_METHOD:
            adjusted_bias = get_adjusted_vertical_bias(current_sector)  # Get position-specific bias
            vertical_position = avg_iris_y + adjusted_bias
        else:
            adjusted_bias = get_adjusted_vertical_bias(current_sector)  # Get position-specific bias
            vertical_position = avg_iris_y + adjusted_bias
            
        if avg_iris_x < -GAZE_H_THRESHOLD:
            h_direction = "Left"
        elif avg_iris_x > GAZE_H_THRESHOLD:
            h_direction = "Right"
        
        # Invert vertical position to fix top/bottom confusion
        vertical_position = -vertical_position
        
        # Apply sensitivity adjustment to downward gaze here as well
        if vertical_position < -GAZE_V_THRESHOLD_UP:
            v_direction = "Up"
        elif vertical_position > GAZE_V_THRESHOLD_DOWN * (0.5 / DOWN_GAZE_SENSITIVITY):  # Make downward detection more sensitive (0.6->0.5)
            # Apply the same sensitivity adjustment as in the sector-specific code
            eye_openness_factor = min(1.0, current_eye_openness * 35)  # Normalize to 0-1 (increased from 30 to 35)
            effective_threshold = GAZE_V_THRESHOLD_DOWN * (0.5 / DOWN_GAZE_SENSITIVITY) * eye_openness_factor
            
            if vertical_position > effective_threshold:
                v_direction = "Down"
            else:
                v_direction = "Center"
                
            if DEBUG_MODE and CONSOLE_DEBUG:
                print(f"Global down detection: pos={vertical_position:.4f}, threshold={effective_threshold:.4f}, sensitivity={DOWN_GAZE_SENSITIVITY:.2f}, bias={adjusted_bias:.4f}")
        else:
            v_direction = "Center"
            
        if DEBUG_MODE and USE_SECTOR_CALIBRATION:
            if current_sector not in SECTOR_CALIBRATION_DATA:
                print(f"Sector {current_sector} not found in calibration data")
            elif SECTOR_CALIBRATION_DATA[current_sector]['samples_count'] <= 5:
                print(f"Sector {current_sector} has insufficient samples: {SECTOR_CALIBRATION_DATA[current_sector]['samples_count']}")
    
    # Combine horizontal and vertical directions
    if h_direction == "Center" and v_direction == "Center":
        direction = "Center"
    else:
        if h_direction == "Center":
            direction = v_direction
        elif v_direction == "Center":
            direction = h_direction
        else:
            direction = f"{h_direction}-{v_direction}"
    
    # Store the current gaze direction in history for smoothing
    GAZE_HISTORY.append({
        'direction': direction,
        'position': (avg_iris_x, avg_iris_y),
        'sector': current_sector,
        'timestamp': time.time()
    })
    
    # Limit history length
    if len(GAZE_HISTORY) > GAZE_HISTORY_LENGTH:
        GAZE_HISTORY.pop(0)
    
    # Return smoothed gaze direction to reduce flickering
    return smooth_gaze_direction()

# Functions for calibration

def start_calibration():
    """Start the calibration process"""
    global CALIBRATION_MODE, CALIBRATION_STATE, CURRENT_CALIBRATION_POS, CALIBRATION_SAMPLES, calibration_attempt
    global warmup_counter, CALIBRATION_POSITION_START_TIME
    
    # Reset sector calibration data if this is the first calibration attempt
    if calibration_attempt == 0:
        initialize_sector_calibration()
    
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
    global SECTOR_CALIBRATION_DATA, current_eye_openness
    
    # Skip if we're still in warmup phase or not in sampling state
    if CALIBRATION_STATE != "SAMPLING":
        return
    
    current_pos = CALIBRATION_POSITIONS[CURRENT_CALIBRATION_POS]
    current_time = time.time()
    
    # Apply special handling for bottom positions to compensate for eyelid coverage
    adjusted_iris_y = iris_y
    if "Bottom" in current_pos and current_eye_openness < 0.05:
        # If calibrating bottom positions and eyes are less open, apply a correction
        # This compensates for the natural tendency of eyelids to cover part of iris when looking down
        adjusted_iris_y = iris_y + (0.05 - current_eye_openness) * 0.5
        if DEBUG_MODE and CONSOLE_DEBUG:
            print(f"Bottom position eye openness adjustment: {iris_y:.4f} -> {adjusted_iris_y:.4f}")
    
    # Store the sample with timestamp
    CALIBRATION_SAMPLES[current_pos]['x'].append(iris_x)
    CALIBRATION_SAMPLES[current_pos]['y'].append(adjusted_iris_y)  # Use adjusted y value
    CALIBRATION_SAMPLES[current_pos]['timestamps'].append(current_time)
    
    # Update sector-specific calibration data - current_pos is also a sector name in our 3x3 grid
    if USE_SECTOR_CALIBRATION and current_pos in SECTOR_CALIBRATION_DATA:
        # Update reference iris position as a running average
        sector_data = SECTOR_CALIBRATION_DATA[current_pos]
        count = sector_data['samples_count']
        
        if count == 0:
            # First sample for this sector
            sector_data['reference_iris_x'] = iris_x
            sector_data['reference_iris_y'] = adjusted_iris_y  # Use adjusted y value
        else:
            # Update running average
            weight = min(0.1, 1.0 / (count + 1))  # Lower weight for later samples
            sector_data['reference_iris_x'] = ((1 - weight) * sector_data['reference_iris_x'] + 
                                              weight * iris_x)
            sector_data['reference_iris_y'] = ((1 - weight) * sector_data['reference_iris_y'] + 
                                              weight * adjusted_iris_y)  # Use adjusted y value
        
        # Increment sample count
        sector_data['samples_count'] += 1

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
    global SECTOR_CALIBRATION_DATA
    
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
            
            # Define expected positions based on grid layout
            if pos == "Center":
                expected_positions[pos] = {'x': 0.0, 'y': 0.0}
            elif pos == "TopLeft":
                expected_positions[pos] = {'x': -0.4, 'y': -0.4}
            elif pos == "TopCenter":
                expected_positions[pos] = {'x': 0.0, 'y': -0.4}
            elif pos == "TopRight":
                expected_positions[pos] = {'x': 0.4, 'y': -0.4}
            elif pos == "MiddleLeft":
                expected_positions[pos] = {'x': -0.4, 'y': 0.0}
            elif pos == "MiddleRight":
                expected_positions[pos] = {'x': 0.4, 'y': 0.0}
            elif pos == "BottomLeft":
                expected_positions[pos] = {'x': -0.4, 'y': 0.4}
            elif pos == "BottomCenter":
                expected_positions[pos] = {'x': 0.0, 'y': 0.4}
            elif pos == "BottomRight":
                expected_positions[pos] = {'x': 0.4, 'y': 0.4}
    
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
        # For horizontal thresholds, calculate average scaling factor
        h_scale = 1.0
        h_scales = []
        
        # Collect scaling factors from all horizontal points
        for pos in ["MiddleLeft", "MiddleRight", "Left", "Right"]:
            if pos in errors:
                if errors[pos]['x_error'] > 0:
                    h_scales.append(1.0 / max(0.1, errors[pos]['x_error']))
        
        # Average all horizontal scaling factors
        if h_scales:
            h_scale = sum(h_scales) / len(h_scales)
            
        # Apply stronger limiting to scale factors to prevent extreme adjustments
        h_scale = max(0.7, min(1.5, h_scale))  # Limit to 70-150% change
        
        # For vertical thresholds, calculate separate up/down scaling factors
        v_up_scale = 1.0
        v_down_scale = 1.0
        
        # Collect up scaling factors
        v_up_scales = []
        for pos in ["TopCenter", "TopLeft", "TopRight", "Up"]:
            if pos in errors:
                if errors[pos]['y_error'] > 0:
                    v_up_scales.append(1.0 / max(0.1, errors[pos]['y_error']))
        
        # Collect down scaling factors
        v_down_scales = []
        for pos in ["BottomCenter", "BottomLeft", "BottomRight", "Down"]:
            if pos in errors:
                if errors[pos]['y_error'] > 0:
                    v_down_scales.append(1.0 / max(0.1, errors[pos]['y_error']))
        
        # Average scaling factors
        if v_up_scales:
            v_up_scale = sum(v_up_scales) / len(v_up_scales)
        if v_down_scales:
            v_down_scale = sum(v_down_scales) / len(v_down_scales)
        
        # Apply stronger limiting to vertical scale factors
        v_up_scale = max(0.7, min(1.5, v_up_scale))    # Limit to 70-150% change
        v_down_scale = max(0.7, min(1.5, v_down_scale))  # Limit to 70-150% change
        
        print(f"Auto-adjusting thresholds - H:{h_scale:.2f}x, V-Up:{v_up_scale:.2f}x, V-Down:{v_down_scale:.2f}x")
    
    # Calculate thresholds based on all calibration points
    new_h_threshold = GAZE_H_THRESHOLD  # Default to current value
    new_v_threshold_up = GAZE_V_THRESHOLD_UP
    new_v_threshold_down = GAZE_V_THRESHOLD_DOWN
    new_vertical_bias = VERTICAL_BIAS
    
    # Calculate horizontal thresholds using multiple points
    h_thresholds = []
    if "Center" in averages:
        center_x = averages["Center"]["x"]
        
        # Calculate horizontal thresholds using all available left/right points
        for pos in ["MiddleLeft", "Left", "TopLeft", "BottomLeft"]:
            if pos in averages:
                left_threshold = abs(center_x - averages[pos]["x"]) * 0.7
                h_thresholds.append(left_threshold)
        
        for pos in ["MiddleRight", "Right", "TopRight", "BottomRight"]:
            if pos in averages:
                right_threshold = abs(center_x - averages[pos]["x"]) * 0.7
                h_thresholds.append(right_threshold)
    
    # Calculate new horizontal threshold as average of all calculated thresholds
    if h_thresholds:
        new_h_threshold = max(0.003, sum(h_thresholds) / len(h_thresholds))
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST:
            new_h_threshold *= h_scale
    
    # Calculate vertical thresholds using multiple points
    up_thresholds = []
    down_thresholds = []
    if "Center" in averages:
        center_y = averages["Center"]["y"]
        
        # Calculate upward thresholds
        for pos in ["TopCenter", "TopLeft", "TopRight", "Up"]:
            if pos in averages:
                up_threshold = abs(center_y - averages[pos]["y"]) * 0.65
                up_thresholds.append(up_threshold)
        
        # Calculate downward thresholds
        for pos in ["BottomCenter", "BottomLeft", "BottomRight", "Down"]:
            if pos in averages:
                down_threshold = abs(center_y - averages[pos]["y"]) * 0.65
                down_thresholds.append(down_threshold)
    
    # Calculate new vertical thresholds
    if up_thresholds:
        new_v_threshold_up = max(0.003, sum(up_thresholds) / len(up_thresholds))
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST:
            new_v_threshold_up *= v_up_scale
    
    if down_thresholds:
        new_v_threshold_down = max(0.002, sum(down_thresholds) / len(down_thresholds))
        # Apply auto-adjustment if enabled
        if CALIBRATION_AUTO_ADJUST:
            new_v_threshold_down *= v_down_scale
    
    # Calculate vertical bias more intelligently using all center-row points
    if "Center" in averages:
        # Start with center position
        center_bias = -averages["Center"]["y"] * 0.8  # Apply a 0.8 factor to avoid over-correction
        biases = [center_bias]
        
        # Add more bias calculations from middle row points
        if "MiddleLeft" in averages:
            biases.append(-averages["MiddleLeft"]["y"] * 0.8)
        if "MiddleRight" in averages:
            biases.append(-averages["MiddleRight"]["y"] * 0.8)
        
        # Average all calculated biases
        new_vertical_bias = sum(biases) / len(biases)
        
        # Additional adjustment to compensate for any asymmetry
        if len(up_thresholds) > 0 and len(down_thresholds) > 0:
            avg_up_threshold = sum(up_thresholds) / len(up_thresholds)
            avg_down_threshold = sum(down_thresholds) / len(down_thresholds)
            
            # If there's significant asymmetry, adjust the bias slightly
            if abs(avg_up_threshold - avg_down_threshold) > 0.01:
                if avg_up_threshold > avg_down_threshold:
                    # Up is further from center, increase bias
                    new_vertical_bias += 0.005
                else:
                    # Down is further from center, decrease bias
                    new_vertical_bias -= 0.005
    
    # Apply weighted average between old and new values for stability
    # This prevents drastic changes between calibration attempts
    GAZE_H_THRESHOLD = (CALIBRATION_DAMPING_FACTOR * new_h_threshold) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_h_threshold)
    GAZE_V_THRESHOLD_UP = (CALIBRATION_DAMPING_FACTOR * new_v_threshold_up) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_v_threshold_up)
    GAZE_V_THRESHOLD_DOWN = (CALIBRATION_DAMPING_FACTOR * new_v_threshold_down) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_v_threshold_down)
    VERTICAL_BIAS = (CALIBRATION_DAMPING_FACTOR * new_vertical_bias) + ((1 - CALIBRATION_DAMPING_FACTOR) * prev_vertical_bias)
    
    # Calculate midpoints for classification
    midpoints = {}
    
    # For left/right classification
    if "Center" in averages:
        # For left classification, average all left points
        left_points = []
        for pos in ["MiddleLeft", "Left", "TopLeft", "BottomLeft"]:
            if pos in averages:
                left_points.append((averages["Center"]["x"] + averages[pos]["x"]) / 2)
        if left_points:
            midpoints["left"] = sum(left_points) / len(left_points)
        
        # For right classification, average all right points
        right_points = []
        for pos in ["MiddleRight", "Right", "TopRight", "BottomRight"]:
            if pos in averages:
                right_points.append((averages["Center"]["x"] + averages[pos]["x"]) / 2)
        if right_points:
            midpoints["right"] = sum(right_points) / len(right_points)
        
        # For up classification, average all top points
        up_points = []
        for pos in ["TopCenter", "TopLeft", "TopRight", "Up"]:
            if pos in averages:
                up_points.append((averages["Center"]["y"] + averages[pos]["y"]) / 2)
        if up_points:
            midpoints["up"] = sum(up_points) / len(up_points)
        
        # For down classification, average all bottom points
        down_points = []
        for pos in ["BottomCenter", "BottomLeft", "BottomRight", "Down"]:
            if pos in averages:
                down_points.append((averages["Center"]["y"] + averages[pos]["y"]) / 2)
        if down_points:
            midpoints["down"] = sum(down_points) / len(down_points)
    
    # Calculate sector-specific thresholds if enabled
    if USE_SECTOR_CALIBRATION:
        # For each sector, calculate local thresholds based on differences with neighboring sectors
        for i, sector in enumerate(SCREEN_SECTORS):
            # Skip if we don't have samples for this sector
            if sector not in averages:
                continue
                
            # Get sector position in grid (0-2 for rows, 0-2 for cols)
            row = i // 3
            col = i % 3
            
            # Find neighboring sectors for threshold calculation
            neighbors = []
            
            # Left neighbor
            if col > 0:
                left_sector = SCREEN_SECTORS[i - 1]
                if left_sector in averages:
                    neighbors.append(left_sector)
            
            # Right neighbor
            if col < 2:
                right_sector = SCREEN_SECTORS[i + 1]
                if right_sector in averages:
                    neighbors.append(right_sector)
            
            # Top neighbor
            if row > 0:
                top_sector = SCREEN_SECTORS[i - 3]
                if top_sector in averages:
                    neighbors.append(top_sector)
            
            # Bottom neighbor
            if row < 2:
                bottom_sector = SCREEN_SECTORS[i + 3]
                if bottom_sector in averages:
                    neighbors.append(bottom_sector)
            
            # Calculate local thresholds based on differences with neighbors
            sector_h_thresholds = []
            sector_v_up_thresholds = []
            sector_v_down_thresholds = []
            
            for neighbor in neighbors:
                # Skip if we don't have samples for neighbor
                if neighbor not in averages:
                    continue
                
                # Calculate horizontal difference
                h_diff = abs(averages[sector]['x'] - averages[neighbor]['x'])
                if h_diff > 0.001:  # Avoid tiny values
                    sector_h_thresholds.append(h_diff * 0.6)  # 60% of difference as threshold
                
                # Calculate vertical difference
                v_diff = averages[sector]['y'] - averages[neighbor]['y']
                if v_diff < -0.001:  # Neighbor is below this sector
                    sector_v_down_thresholds.append(abs(v_diff) * 0.6)
                elif v_diff > 0.001:  # Neighbor is above this sector
                    sector_v_up_thresholds.append(abs(v_diff) * 0.6)
            
            # Calculate sector-specific thresholds as average of neighbor differences
            if sector_h_thresholds:
                SECTOR_CALIBRATION_DATA[sector]['h_threshold'] = max(0.003, sum(sector_h_thresholds) / len(sector_h_thresholds))
            else:
                # Fallback to global threshold
                SECTOR_CALIBRATION_DATA[sector]['h_threshold'] = GAZE_H_THRESHOLD
            
            if sector_v_up_thresholds:
                SECTOR_CALIBRATION_DATA[sector]['v_threshold_up'] = max(0.003, sum(sector_v_up_thresholds) / len(sector_v_up_thresholds))
            else:
                # Fallback to global threshold
                SECTOR_CALIBRATION_DATA[sector]['v_threshold_up'] = GAZE_V_THRESHOLD_UP
            
            if sector_v_down_thresholds:
                SECTOR_CALIBRATION_DATA[sector]['v_threshold_down'] = max(0.002, sum(sector_v_down_thresholds) / len(sector_v_down_thresholds))
            else:
                # Fallback to global threshold
                SECTOR_CALIBRATION_DATA[sector]['v_threshold_down'] = GAZE_V_THRESHOLD_DOWN
            
            # Calculate sector-specific vertical bias
            # For each sector, bias should be the offset needed to make the reference position "centered"
            if "Bottom" in sector:
                # Lower bias for bottom positions to improve downward gaze detection
                SECTOR_CALIBRATION_DATA[sector]['vertical_bias'] = -averages[sector]['y'] * 0.25
            else:
                # Standard bias for other positions
                SECTOR_CALIBRATION_DATA[sector]['vertical_bias'] = -averages[sector]['y'] * 0.35
        
        # Debug output for sector-specific calibration
        print("\nSector-specific calibration values:")
        for sector in SCREEN_SECTORS:
            if sector in averages:
                print(f"  {sector}: H={SECTOR_CALIBRATION_DATA[sector]['h_threshold']:.4f}, " +
                      f"V-Up={SECTOR_CALIBRATION_DATA[sector]['v_threshold_up']:.4f}, " +
                      f"V-Down={SECTOR_CALIBRATION_DATA[sector]['v_threshold_down']:.4f}, " +
                      f"Bias={SECTOR_CALIBRATION_DATA[sector]['vertical_bias']:.4f}")
    
    # Store all calculated values
    CALIBRATION_DATA = {
        'h_threshold': GAZE_H_THRESHOLD,
        'v_threshold_up': GAZE_V_THRESHOLD_UP,
        'v_threshold_down': GAZE_V_THRESHOLD_DOWN,
        'vertical_bias': VERTICAL_BIAS,
        'use_alternative_method': USE_ALTERNATIVE_METHOD,
        'averages': averages,
        'expected_positions': expected_positions,
        'midpoints': midpoints,
        'use_sector_calibration': USE_SECTOR_CALIBRATION,
        'sector_data': SECTOR_CALIBRATION_DATA if USE_SECTOR_CALIBRATION else {}
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
    global current_iris_relative_x, current_iris_relative_y
    
    current_target = VALIDATION_SEQUENCE[VALIDATION_CURRENT]
    correct = False
    confidence_score = 0.0
    
    # Get current sector based on the same logic used for gaze detection
    current_sector = determine_current_sector(current_iris_relative_x, current_iris_relative_y)
    
    # For validation, use direct sector matching instead of gaze direction
    # This should be more consistent with how sectors are detected
    if current_sector == current_target:
        # Direct sector match - high confidence
        correct = True
        confidence_score = 0.9
    else:
        # Check if there's still a reasonable match based on neighboring sectors
        adjacent_sectors = get_adjacent_sectors(current_target)
        if current_sector in adjacent_sectors:
            # Close enough - lower confidence but still correct
            correct = True
            confidence_score = 0.5
        else:
            # Calculate distance to expected position for more flexible validation
            if current_target in CALIBRATION_DATA['averages']:
                target_x = CALIBRATION_DATA['averages'][current_target]['x']
                target_y = CALIBRATION_DATA['averages'][current_target]['y']
                
                # Calculate normalized distance (0.0 = perfect match, 1.0 = far away)
                distance_x = abs(current_iris_relative_x - target_x) / max(GAZE_H_THRESHOLD, 0.001)
                distance_y = abs(current_iris_relative_y - target_y) / max(GAZE_V_THRESHOLD_UP, GAZE_V_THRESHOLD_DOWN, 0.001)
                
                # Calculate weighted distance
                if current_target in ["TopCenter", "BottomCenter"]:
                    distance = (distance_x * 0.3) + (distance_y * 0.7)  # Weighted toward vertical
                elif current_target in ["MiddleLeft", "MiddleRight"]:
                    distance = (distance_x * 0.7) + (distance_y * 0.3)  # Weighted toward horizontal
                else:  # Corners and Center
                    distance = (distance_x + distance_y) / 2.0  # Equal weight
                
                # Convert distance to confidence score (1.0 = perfect match, 0.0 = far away)
                confidence_score = max(0.0, 1.0 - distance)
                
                # Consider correct if confidence is above threshold
                if confidence_score > CALIBRATION_TOLERANCE:
                    correct = True
    
    if correct:
        VALIDATION_CORRECT += 1
        print(f"Correct! {VALIDATION_CORRECT} in a row. (Confidence: {confidence_score:.2f})")
    else:
        VALIDATION_CORRECT = 0
        detected_direction = f"{gaze_h}-{gaze_v}" if gaze_h != "Center" and gaze_v != "Center" else \
                             gaze_h if gaze_v == "Center" else gaze_v
        print(f"Incorrect. Expected: {current_target}, Detected: {current_sector} (Direction: {detected_direction}, Confidence: {confidence_score:.2f})")
    
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

# Helper function to identify adjacent sectors
def get_adjacent_sectors(sector):
    """Return a list of sectors adjacent to the given sector"""
    sector_grid = [
        ["TopLeft", "TopCenter", "TopRight"],
        ["MiddleLeft", "Center", "MiddleRight"],
        ["BottomLeft", "BottomCenter", "BottomRight"]
    ]
    
    # Find position in grid
    row, col = -1, -1
    for r, row_sectors in enumerate(sector_grid):
        if sector in row_sectors:
            row = r
            col = row_sectors.index(sector)
            break
    
    if row == -1 or col == -1:
        return []  # Sector not found
    
    # Get adjacent sectors
    adjacent = []
    for r in range(max(0, row-1), min(3, row+2)):
        for c in range(max(0, col-1), min(3, col+2)):
            if r == row and c == col:
                continue  # Skip the sector itself
            adjacent.append(sector_grid[r][c])
    
    return adjacent

def complete_calibration():
    """Complete the calibration process and save settings"""
    global CALIBRATION_MODE, CALIBRATION_STATE, calibration_attempt
    
    save_config()
    
    CALIBRATION_MODE = False
    CALIBRATION_STATE = "COMPLETE"
    calibration_attempt = 0
    
    print("Calibration complete and settings saved!")

def save_config():
    """Save the current configuration to a JSON file"""
    global CALIBRATION_DATA, GAZE_H_THRESHOLD, GAZE_V_THRESHOLD_UP, GAZE_V_THRESHOLD_DOWN, VERTICAL_BIAS
    global USE_ALTERNATIVE_METHOD, USE_SECTOR_CALIBRATION, SECTOR_CALIBRATION_DATA
    
    # Prepare configuration data
    config = {
        'h_threshold': GAZE_H_THRESHOLD,
        'v_threshold_up': GAZE_V_THRESHOLD_UP,
        'v_threshold_down': GAZE_V_THRESHOLD_DOWN,
        'vertical_bias': VERTICAL_BIAS,
        'use_alternative_method': USE_ALTERNATIVE_METHOD,
        'timestamp': time.time(),
        'use_sector_calibration': USE_SECTOR_CALIBRATION,
    }
    
    # Add sector calibration data if available
    if USE_SECTOR_CALIBRATION:
        # Convert sector data to a format that can be JSON serialized
        sector_data = {}
        for sector, data in SECTOR_CALIBRATION_DATA.items():
            sector_data[sector] = {
                'reference_iris_x': float(data['reference_iris_x']),
                'reference_iris_y': float(data['reference_iris_y']),
                'h_threshold': float(data['h_threshold']),
                'v_threshold_up': float(data['v_threshold_up']),
                'v_threshold_down': float(data['v_threshold_down']),
                'vertical_bias': float(data['vertical_bias']),
                'samples_count': int(data['samples_count'])
            }
        config['sector_data'] = sector_data
    
    # Save to file
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {CONFIG_FILE}")

def load_config():
    """Load configuration from file"""
    global GAZE_H_THRESHOLD, GAZE_V_THRESHOLD_UP, GAZE_V_THRESHOLD_DOWN, VERTICAL_BIAS
    global USE_ALTERNATIVE_METHOD, USE_SECTOR_CALIBRATION, SECTOR_CALIBRATION_DATA
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                
            # Load primary gaze thresholds
            GAZE_H_THRESHOLD = config.get('h_threshold', GAZE_H_THRESHOLD)
            GAZE_V_THRESHOLD_UP = config.get('v_threshold_up', GAZE_V_THRESHOLD_UP)
            GAZE_V_THRESHOLD_DOWN = config.get('v_threshold_down', GAZE_V_THRESHOLD_DOWN)
            VERTICAL_BIAS = config.get('vertical_bias', VERTICAL_BIAS)
            
            # Load feature toggles
            USE_ALTERNATIVE_METHOD = config.get('use_alternative_method', USE_ALTERNATIVE_METHOD)
            USE_SECTOR_CALIBRATION = config.get('use_sector_calibration', USE_SECTOR_CALIBRATION)
            
            # Initialize sector calibration data if enabled
            if USE_SECTOR_CALIBRATION:
                # Initialize the data structure if it doesn't exist
                if not SECTOR_CALIBRATION_DATA:
                    initialize_sector_calibration()
                
                # Load sector-specific calibration data if available
                if 'sector_data' in config:
                    for sector, data in config['sector_data'].items():
                        if sector in SECTOR_CALIBRATION_DATA:
                            SECTOR_CALIBRATION_DATA[sector].update(data)
                            
                            # Ensure we have all required fields
                            if 'samples_count' not in SECTOR_CALIBRATION_DATA[sector]:
                                SECTOR_CALIBRATION_DATA[sector]['samples_count'] = 0
            
            # Print loaded calibration data for debugging
            print("\nLoaded sector calibration data:")
            for sector, data in SECTOR_CALIBRATION_DATA.items():
                h_threshold = data.get('h_threshold', GAZE_H_THRESHOLD)
                v_threshold_up = data.get('v_threshold_up', GAZE_V_THRESHOLD_UP)
                v_threshold_down = data.get('v_threshold_down', GAZE_V_THRESHOLD_DOWN)
                vert_bias = data.get('vertical_bias', VERTICAL_BIAS)
                samples = data.get('samples_count', 0)
                print(f"  {sector}: H={h_threshold:.4f}, V-Up={v_threshold_up:.4f}, V-Down={v_threshold_down:.4f}, Bias={vert_bias:.4f}, Samples={samples}")
            
            # Initialize calibration data for the prediction system
            initialize_calibration_for_prediction()
            
            print("Configuration loaded from " + CONFIG_FILE)
            return True
        
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    return False

def initialize_calibration_for_prediction():
    """Initialize calibration data for the prediction system at startup"""
    global CALIBRATION_DATA, SECTOR_CALIBRATION_DATA
    
    # Create empty calibration data structure if it doesn't exist
    if not hasattr(globals(), 'CALIBRATION_DATA') or not CALIBRATION_DATA:
        CALIBRATION_DATA = {'averages': {}}
    
    # If we have sector calibration data, use it to set up prediction
    if USE_SECTOR_CALIBRATION and SECTOR_CALIBRATION_DATA:
        # For each sector, create reference points in CALIBRATION_DATA
        for sector, data in SECTOR_CALIBRATION_DATA.items():
            if 'reference_iris_x' in data and 'reference_iris_y' in data:
                # Use the reference iris positions from the sector calibration
                CALIBRATION_DATA['averages'][sector] = {
                    'x': data['reference_iris_x'],
                    'y': data['reference_iris_y'],
                    'samples': data.get('samples_count', 0)
                }
        
        print("Initialized calibration prediction system with saved calibration data")
        
        # If we don't have all sectors, set some defaults for missing ones
        default_positions = {
            'TopLeft': (-0.04, 0.45),
            'TopCenter': (0.0, 0.45),
            'TopRight': (0.04, 0.45),
            'MiddleLeft': (-0.04, 0.50),
            'Center': (0.0, 0.50),
            'MiddleRight': (0.04, 0.50),
            'BottomLeft': (-0.04, 0.55),
            'BottomCenter': (0.0, 0.55),
            'BottomRight': (0.04, 0.55)
        }
        
        # Fill in any missing sectors with defaults
        for sector, (x, y) in default_positions.items():
            if sector not in CALIBRATION_DATA['averages']:
                CALIBRATION_DATA['averages'][sector] = {
                    'x': x,
                    'y': y,
                    'samples': 0
                }

def get_target_position(position, frame_size):
    """Get the x,y coordinates for a target position on the screen
    
    Args:
        position: String position name (Center, TopLeft, TopRight, etc.)
        frame_size: Tuple of (width, height) of the screen
        
    Returns:
        Tuple of (x, y) coordinates for the target
    """
    width, height = frame_size
    
    # Divide screen into 3x3 grid with padding
    padding_x = width // 6
    padding_y = height // 6
    
    # Calculate positions for 3x3 grid
    left_x = padding_x
    center_x = width // 2
    right_x = width - padding_x
    
    top_y = padding_y
    middle_y = height // 2
    bottom_y = height - padding_y
    
    # Map position names to coordinates
    positions = {
        "Center": (center_x, middle_y),
        "TopLeft": (left_x, top_y),
        "TopCenter": (center_x, top_y),
        "TopRight": (right_x, top_y),
        "MiddleLeft": (left_x, middle_y),
        "MiddleRight": (right_x, middle_y),
        "BottomLeft": (left_x, bottom_y),
        "BottomCenter": (center_x, bottom_y),
        "BottomRight": (right_x, bottom_y),
        # For backward compatibility
        "Left": (left_x, middle_y),
        "Right": (right_x, middle_y),
        "Up": (center_x, top_y),
        "Down": (center_x, bottom_y)
    }
    
    return positions.get(position, (center_x, middle_y))

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
max_history = 30  # Increased from 20 to 30 for more smoothing

# Custom drawing specs
IRIS_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
IRIS_CONNECTIONS_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

# Create CSV file with headers if logging is enabled
if ENABLE_LOGGING:
    with open(log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)
    print(f"Data logging enabled. Saving to {log_file}")

# For low-pass filtering of predicted point coordinates
last_pred_x, last_pred_y = None, None
smoothing_factor = 0.05  # Lower = more smoothing (reduced from 0.15 to 0.05 for smoother movement)

# Add these with other global variables
# For Kalman filter-based smoothing
use_kalman = True  # Set to False to disable Kalman filtering
kalman_initialized = False
kalman_state = np.zeros(4)  # [x, y, dx, dy]
kalman_cov = np.eye(4) * 1000  # High initial uncertainty
process_noise = 0.003  # Reduced from 0.005 for even smoother predictions
measurement_noise = 0.15  # Increased from 0.1 to trust measurements less

def kalman_predict():
    """Predict step of Kalman filter"""
    global kalman_state, kalman_cov
    
    # State transition matrix (position + velocity model)
    F = np.array([[1, 0, 1, 0],  # x' = x + dx
                  [0, 1, 0, 1],  # y' = y + dy
                  [0, 0, 1, 0],  # dx' = dx
                  [0, 0, 0, 1]])  # dy' = dy
    
    # Process noise
    Q = np.eye(4) * process_noise
    
    # Predict next state
    kalman_state = F @ kalman_state
    kalman_cov = F @ kalman_cov @ F.T + Q

def kalman_update(measurement):
    """Update step of Kalman filter"""
    global kalman_state, kalman_cov, kalman_initialized
    
    if not kalman_initialized:
        # Initialize with first measurement
        kalman_state[0] = measurement[0]
        kalman_state[1] = measurement[1]
        kalman_initialized = True
        return (kalman_state[0], kalman_state[1])
    
    # Measurement matrix (we only observe position)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    # Measurement noise
    R = np.eye(2) * measurement_noise
    
    # Kalman gain
    K = kalman_cov @ H.T @ np.linalg.inv(H @ kalman_cov @ H.T + R)
    
    # Update state
    y = measurement - H @ kalman_state
    kalman_state = kalman_state + K @ y
    
    # Update covariance
    kalman_cov = (np.eye(4) - K @ H) @ kalman_cov
    
    # Return filtered position
    return (kalman_state[0], kalman_state[1])

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
    global last_pred_x, last_pred_y, kalman_initialized
    
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
        
        # Calculate eye openness for downward gaze adjustment
        global current_eye_openness
        current_eye_openness = (left_eye_height + right_eye_height) / 2
        
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
        
        # Apply correction for downward gaze when eyes are less open
        # This compensates for the natural eyelid coverage when looking down
        eye_openness_ratio = current_eye_openness / 0.05  # Normalize to typical openness
        if avg_rel_pos > 0.5 and eye_openness_ratio < 1.0:
            # Enhance downward gaze detection when eyes are less open
            # The less open the eyes, the more we boost the downward position
            downward_boost = max(0, (1.0 - eye_openness_ratio) * 0.2)
            avg_rel_pos = avg_rel_pos + downward_boost
        
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
    
    # Invert the vertical axis to fix top/bottom confusion
    iris_relative_y = -iris_relative_y
    
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
    
    # Apply advanced filtering:
    orig_x, orig_y = predicted_gaze_x, predicted_gaze_y
    
    if use_kalman:
        # Apply Kalman filter for complex smooth motion
        kalman_predict()
        x, y = kalman_update(np.array([predicted_gaze_x, predicted_gaze_y]))
        predicted_gaze_x, predicted_gaze_y = int(x), int(y)
    else:
        # Apply basic low-pass filter (exponential smoothing) as fallback
        if last_pred_x is not None:
            # Exponential moving average: new_value = alpha*current + (1-alpha)*previous
            filtered_x = smoothing_factor * predicted_gaze_x + (1 - smoothing_factor) * last_pred_x
            filtered_y = smoothing_factor * predicted_gaze_y + (1 - smoothing_factor) * last_pred_y
            
            predicted_gaze_x = int(filtered_x)
            predicted_gaze_y = int(filtered_y)
    
    # Store current values for next frame's smoothing
    if last_pred_x is not None and last_pred_y is not None:
        # Add a small deadzone - don't update if movement is too small (reduces jitter)
        movement_distance = ((predicted_gaze_x - last_pred_x)**2 + (predicted_gaze_y - last_pred_y)**2)**0.5
        deadzone = 2  # Pixels
        
        if movement_distance < deadzone:
            # Keep previous position to prevent tiny movements
            predicted_gaze_x, predicted_gaze_y = last_pred_x, last_pred_y
    
    # Add additional stabilizing by averaging recent cursor positions
    cursor_history.append((predicted_gaze_x, predicted_gaze_y))
    if len(cursor_history) > CURSOR_STABLE_FRAMES:
        cursor_history.pop(0)
    
    # If we have enough history, use the average position
    if len(cursor_history) == CURSOR_STABLE_FRAMES:
        avg_x = sum(pos[0] for pos in cursor_history) / CURSOR_STABLE_FRAMES
        avg_y = sum(pos[1] for pos in cursor_history) / CURSOR_STABLE_FRAMES
        predicted_gaze_x, predicted_gaze_y = int(avg_x), int(avg_y)
    
    last_pred_x, last_pred_y = predicted_gaze_x, predicted_gaze_y
    
    # Add to history for smoothing iris positions
    iris_x_history.append(iris_relative_x)
    iris_y_history.append(iris_relative_y)
    
    if len(iris_x_history) > max_history:
        iris_x_history.pop(0)
    if len(iris_y_history) > max_history:
        iris_y_history.pop(0)
    
    # Use the weighted average of recent values for more stability
    if iris_x_history and iris_y_history:
        # Modify the weights to more heavily favor previous values
        # Older values (start of list) get weight 0.5, newest values get weight 1.0
        weights = [0.5 + 0.5 * (i / len(iris_x_history)) for i in range(len(iris_x_history))]
        total_weight = sum(weights)
        
        # Weighted moving average
        smoothed_x = sum(x * w for x, w in zip(iris_x_history, weights)) / total_weight
        smoothed_y = sum(y * w for y, w in zip(iris_y_history, weights)) / total_weight
        
        iris_relative_x = smoothed_x
        iris_relative_y = smoothed_y
    
    # Determine horizontal gaze direction using calibration midpoints if available
    h_direction = "Center"
    if 'midpoints' in CALIBRATION_DATA:
        # Fix the left/right direction swapping by correcting the comparison
        if CALIBRATION_DATA['midpoints'].get('left') is not None and iris_relative_x < CALIBRATION_DATA['midpoints']['left']:
            h_direction = "Left"  # Correctly set to Left when looking left (negative X)
        elif CALIBRATION_DATA['midpoints'].get('right') is not None and iris_relative_x > CALIBRATION_DATA['midpoints']['right']:
            h_direction = "Right"  # Correctly set to Right when looking right (positive X)
    else:
        # Fallback to threshold-based detection
        # Ensure proper left/right direction by checking the sign correctly
        if iris_relative_x < -GAZE_H_THRESHOLD:
            h_direction = "Left"  # Looking left (negative X)
        elif iris_relative_x > GAZE_H_THRESHOLD:
            h_direction = "Right"  # Looking right (positive X)
    
    # Determine vertical gaze direction using calibration midpoints if available
    v_direction = "Center"
    if 'midpoints' in CALIBRATION_DATA:
        # Fix vertical direction confusion by ensuring correct comparison
        # Note: negative iris_relative_y means looking up, positive means looking down
        if CALIBRATION_DATA['midpoints'].get('up') is not None and iris_relative_y < CALIBRATION_DATA['midpoints']['up']:
            v_direction = "Up"  # Looking up (negative Y)
        elif CALIBRATION_DATA['midpoints'].get('down') is not None and iris_relative_y > CALIBRATION_DATA['midpoints']['down']:
            v_direction = "Down"  # Looking down (positive Y)
    else:
        # Fallback to asymmetric thresholds for upward and downward gaze
        # Ensure correct vertical orientation by checking the signs correctly
        # In iris coordinate system: negative Y = looking up, positive Y = looking down
        if iris_relative_y < -GAZE_V_THRESHOLD_UP:
            v_direction = "Up"  # Looking up (negative Y)
        elif iris_relative_y > GAZE_V_THRESHOLD_DOWN:
            v_direction = "Down"  # Looking down (positive Y)
    
    # If we have multiple calibration points, we can be more precise with corner gazes
    if 'averages' in CALIBRATION_DATA:
        # Check for diagonal gazes (corners) by comparing with corner position averages
        if h_direction != "Center" and v_direction != "Center":
            # We already detected both horizontal and vertical movement, refine for corners
            # Fix the corner mapping to ensure directions match correctly
            corner_positions = {
                ("Left", "Up"): "TopLeft",
                ("Right", "Up"): "TopRight",
                ("Left", "Down"): "BottomLeft", 
                ("Right", "Down"): "BottomRight"
            }
            corner_name = corner_positions.get((h_direction, v_direction))
            
            # If we have this corner calibrated, check if we're really looking at it
            if corner_name and corner_name in CALIBRATION_DATA['averages']:
                corner_x = CALIBRATION_DATA['averages'][corner_name]['x']
                corner_y = CALIBRATION_DATA['averages'][corner_name]['y']
                
                # Calculate distance to this corner position
                distance_to_corner = ((iris_relative_x - corner_x)**2 + (iris_relative_y - corner_y)**2)**0.5
                
                # If we're not close enough to the corner, revert to just reporting cardinal directions
                if distance_to_corner > 0.1:  # Arbitrary threshold, may need tuning
                    pass  # Keep existing h_direction and v_direction
                else:
                    # We're close to the corner, can explicitly report it if needed
                    pass  # For now, just keep cardinal directions
    
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
    global ATTENTION_TEST_MODE, CURRENT_TEST_IMAGE, CURRENT_TEST_IMAGE_START_TIME
    global ATTENTION_TEST_STATE, CURRENT_HEATMAP, CURRENT_HEATMAP_START_TIME
    
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
                
                # Add sector information to debug display
                current_sector = determine_current_sector(iris_rel_x, iris_rel_y)
                cv2.putText(frame, f"Sector: {current_sector}", (10, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show sector-specific calibration status
                if USE_SECTOR_CALIBRATION:
                    if current_sector in SECTOR_CALIBRATION_DATA and SECTOR_CALIBRATION_DATA[current_sector]['samples_count'] > 5:
                        sector_data = SECTOR_CALIBRATION_DATA[current_sector]
                        sector_status = f"Sector calib: H={sector_data['h_threshold']:.3f} V↑={sector_data['v_threshold_up']:.3f} V↓={sector_data['v_threshold_down']:.3f}"
                        cv2.putText(frame, sector_status, (10, 310), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        sector_status = "Sector calib: Not available for this position"
                        cv2.putText(frame, sector_status, (10, 310), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
            
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
                crosshair_color = (0, 255, 0)  # Green for filtered point
                cv2.line(frame, (predicted_gaze_x - crosshair_size, predicted_gaze_y), 
                         (predicted_gaze_x + crosshair_size, predicted_gaze_y), crosshair_color, 2)
                cv2.line(frame, (predicted_gaze_x, predicted_gaze_y - crosshair_size), 
                         (predicted_gaze_x, predicted_gaze_y + crosshair_size), crosshair_color, 2)
                cv2.circle(frame, (predicted_gaze_x, predicted_gaze_y), 5, crosshair_color, -1)
                
                # If in debug mode, also show the raw (unfiltered) prediction point
                if DEBUG_MODE and 'orig_x' in locals() and 'orig_y' in locals():
                    orig_x = int(orig_x)
                    orig_y = int(orig_y)
                    cv2.circle(frame, (orig_x, orig_y), 3, (0, 0, 255), -1)  # Red dot for raw
                    cv2.putText(frame, "Raw", (orig_x + 5, orig_y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(frame, "Filtered", (predicted_gaze_x + 5, predicted_gaze_y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Draw line to show the smoothing effect
                    cv2.line(frame, (orig_x, orig_y), (predicted_gaze_x, predicted_gaze_y), 
                             (255, 0, 255), 1, cv2.LINE_AA)
                    
                    # Display filtering method used
                    filter_method = "Kalman" if use_kalman else "Low-pass"
                    cv2.putText(frame, f"Filter: {filter_method}", (10, 290),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
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
    
    # Add new code for attention testing
    if ATTENTION_TEST_MODE:
        # Draw a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Handle different test states
        if ATTENTION_TEST_STATE == "CATEGORY_SELECTION":
            # Display category selection menu
            instruction_text = "Attention Testing - Select a category (1-" + str(len(ATTENTION_CATEGORIES)) + ")"
            cv2.putText(frame, instruction_text, (w//2 - 300, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # List available categories
            for i, category in enumerate(ATTENTION_CATEGORIES):
                category_text = f"{i+1}. {category}"
                cv2.putText(frame, category_text, (w//2 - 200, 100 + i*40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif ATTENTION_TEST_STATE == "TESTING":
            # Display the current test image if available
            if CURRENT_TEST_IMAGE is not None:
                # Calculate elapsed time
                time_elapsed = time.time() - CURRENT_TEST_IMAGE_START_TIME
                
                # Display test image
                test_img = CURRENT_TEST_IMAGE.copy()
                
                # Resize test image to fit the frame while maintaining aspect ratio
                test_h, test_w = test_img.shape[:2]
                scale = min(w / test_w, h / test_h)
                new_w = int(test_w * scale)
                new_h = int(test_h * scale)
                
                # Resize the test image
                test_img_resized = cv2.resize(test_img, (new_w, new_h))
                
                # Calculate position to center the image
                x_offset = (w - new_w) // 2
                y_offset = (h - new_h) // 2
                
                # Create a blank canvas
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Place the resized image on the canvas
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = test_img_resized
                
                # Record gaze data if face is detected and draw real-time gaze visualization
                if results and results.multi_face_landmarks and not is_blinking:
                    face_landmarks = results.multi_face_landmarks[0]
                    # Estimate gaze direction
                    h_direction, v_direction, iris_rel_x, iris_rel_y = estimate_gaze(face_landmarks, w, h)
                    
                    # Print debug info to help diagnose the issue
                    if CONSOLE_DEBUG:
                        print(f"Raw gaze: ({iris_rel_x:.3f}, {iris_rel_y:.3f}) h_dir: {h_direction}, v_dir: {v_direction}")
                    
                    # Convert from -1,1 range to 0,1 range for screen coordinates
                    scaled_x = (iris_rel_x + 1) / 2
                    scaled_y = (iris_rel_y + 1) / 2
                    
                    # CRITICAL FIX: Reverse the vertical bias effect that pushes everything down
                    # Get the current sector for sector-specific adjustments
                    current_sector = determine_current_sector(iris_rel_x, iris_rel_y)
                    
                    # Apply vertical correction based on the sector's calibration
                    if USE_SECTOR_CALIBRATION and current_sector in SECTOR_CALIBRATION_DATA:
                        sector_data = SECTOR_CALIBRATION_DATA[current_sector]
                        # The bias is usually negative and pulls downward - we need to counteract it
                        vertical_bias = sector_data.get('vertical_bias', VERTICAL_BIAS)
                        
                        # Apply a correction factor to compensate for the downward bias
                        # This is the key fix to prevent the cursor from staying stuck at the bottom
                        correction_factor = 0.5  # Adjust this if needed (0.5 means counteract half the bias)
                        scaled_y = max(0.1, min(0.9, scaled_y + vertical_bias * correction_factor))
                        
                        if CONSOLE_DEBUG:
                            print(f"Applied bias correction: {vertical_bias:.3f} with factor {correction_factor}")
                            print(f"Adjusted scaled_y: {scaled_y:.3f}")
                    
                    # Calculate pixel coordinates on the displayed image
                    gaze_x = int(x_offset + scaled_x * new_w)
                    gaze_y = int(y_offset + scaled_y * new_h)
                    
                    # Process gaze for heatmap
                    success = process_gaze_for_heatmap(gaze_x / w, gaze_y / h, w, h)
                    if CONSOLE_DEBUG and success:
                        print(f"Gaze point added at ({gaze_x}, {gaze_y})")
                    
                    # Draw debug info for the current state
                    if DEBUG_MODE:
                        cv2.putText(canvas, f"Sector: {current_sector}", (10, h - 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(canvas, f"Raw: ({iris_rel_x:.2f}, {iris_rel_y:.2f})", (10, h - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(canvas, f"Norm: ({scaled_x:.2f}, {scaled_y:.2f})", (10, h - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Show screen coordinates
                        cv2.putText(canvas, f"Screen: ({gaze_x}, {gaze_y})", (10, h - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Always draw the crosshair cursor
                    # Create more visible crosshair cursor with different color based on validity
                    cursor_color = (0, 255, 255)  # Cyan color for all cursor positions
                    
                    # Draw crosshair
                    cv2.line(canvas, (gaze_x - 15, gaze_y), (gaze_x + 15, gaze_y), cursor_color, 2)  # Horizontal
                    cv2.line(canvas, (gaze_x, gaze_y - 15), (gaze_x, gaze_y + 15), cursor_color, 2)  # Vertical
                    
                    # Pulsing circle
                    pulse_size = 5 + int(5 * math.sin(time.time() * 5))  # Pulsing size between 5-10 pixels
                    cv2.circle(canvas, (gaze_x, gaze_y), pulse_size, (0, 0, 255), -1)  # Filled red circle
                    cv2.circle(canvas, (gaze_x, gaze_y), pulse_size + 5, cursor_color, 2)  # Cyan outline
                    
                    # Show a live mini-heatmap in the corner for immediate feedback
                    image_name = os.path.basename(CURRENT_TEST_IMAGE_PATH)
                    if image_name in ATTENTION_HEATMAP_DATA and len(ATTENTION_HEATMAP_DATA[image_name]) > 0:
                        # Create a mini heatmap visualization for the entire screen
                        mini_heatmap_size = 200
                        mini_heatmap = np.zeros((h, w), dtype=np.float32)
                        
                        # Add gaussians for each gaze point
                        for point_x, point_y in ATTENTION_HEATMAP_DATA[image_name]:
                            # Ensure coordinates are within screen bounds
                            if 0 <= point_x < w and 0 <= point_y < h:
                                # Create small gaussian around point
                                x_coords = np.arange(max(0, point_x-15), min(w, point_x+15))
                                y_coords = np.arange(max(0, point_y-15), min(h, point_y+15))
                                
                                for y in y_coords:
                                    for x in x_coords:
                                        dist_sq = (x-point_x)**2 + (y-point_y)**2
                                        # Add bounds checking before accessing mini_heatmap
                                        if 0 <= y < h and 0 <= x < w:
                                            mini_heatmap[y, x] += np.exp(-dist_sq / (2 * 10**2))
                        
                        # Normalize and colorize the heatmap
                        if mini_heatmap.max() > 0:
                            mini_heatmap = mini_heatmap / mini_heatmap.max()
                        
                        mini_heatmap_colored = cv2.applyColorMap((mini_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        
                        # Resize mini heatmap
                        mini_heatmap_h, mini_heatmap_w = mini_heatmap_colored.shape[:2]
                        mini_scale = min(mini_heatmap_size / mini_heatmap_w, mini_heatmap_size / mini_heatmap_h)
                        mini_new_w = int(mini_heatmap_w * mini_scale)
                        mini_new_h = int(mini_heatmap_h * mini_scale)
                        mini_heatmap_resized = cv2.resize(mini_heatmap_colored, (mini_new_w, mini_new_h))
                        
                        # Draw mini heatmap in corner with label
                        cv2.putText(canvas, f"Live Attention Map ({len(ATTENTION_HEATMAP_DATA[image_name])} points)", 
                                   (w - mini_new_w - 10, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Draw border around mini heatmap
                        cv2.rectangle(canvas, (w - mini_new_w - 10, 30), (w - 10, 30 + mini_new_h), (255, 255, 255), 1)
                        
                        # Place mini heatmap on canvas
                        canvas[30:30+mini_new_h, w-mini_new_w-10:w-10] = mini_heatmap_resized
                
                # Blend the canvas with the frame
                frame = canvas
                
                # Draw progress bar at the top
                progress = min(1.0, time_elapsed / TEST_IMAGE_DURATION)
                bar_width = int(w * progress)
                cv2.rectangle(frame, (0, 0), (bar_width, 10), (0, 255, 0), -1)
                
                # Show category and image info
                category = ATTENTION_CATEGORIES[CURRENT_CATEGORY_INDEX]
                image_name = os.path.basename(CURRENT_TEST_IMAGE_PATH)
                info_text = f"Category: {category} - Image: {CURRENT_IMAGE_INDEX+1}/{len(ATTENTION_TEST_IMAGES)}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Check if it's time to move to the next image
                if time_elapsed >= TEST_IMAGE_DURATION:
                    # Generate and show heatmap for the current image before moving on
                    if CURRENT_TEST_IMAGE_PATH:
                        image_name = os.path.basename(CURRENT_TEST_IMAGE_PATH)
                        if image_name in ATTENTION_HEATMAP_DATA and len(ATTENTION_HEATMAP_DATA[image_name]) > 0:
                            heatmap = generate_heatmap(CURRENT_TEST_IMAGE_PATH, ATTENTION_HEATMAP_DATA[image_name])
                            if heatmap is not None:
                                ATTENTION_TEST_STATE = "SHOWING_HEATMAP"
                                global CURRENT_HEATMAP, CURRENT_HEATMAP_START_TIME
                                CURRENT_HEATMAP = heatmap
                                CURRENT_HEATMAP_START_TIME = time.time()
                                return frame
                    
                    move_to_next_test_image()
        
        elif ATTENTION_TEST_STATE == "SHOWING_HEATMAP":
            # Display the heatmap for a few seconds
            time_elapsed = time.time() - CURRENT_HEATMAP_START_TIME
            
            if time_elapsed < 3.0:  # Show heatmap for 3 seconds
                # Resize heatmap to fit the frame while maintaining aspect ratio
                heatmap_h, heatmap_w = CURRENT_HEATMAP.shape[:2]
                scale = min(w / heatmap_w, h / heatmap_h)
                new_w = int(heatmap_w * scale)
                new_h = int(heatmap_h * scale)
                heatmap_resized = cv2.resize(CURRENT_HEATMAP, (new_w, new_h))
                
                # Calculate position to center the image
                x_offset = (w - new_w) // 2
                y_offset = (h - new_h) // 2
                
                # Create a blank canvas
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Place the resized heatmap on the canvas
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = heatmap_resized
                
                # Show title
                cv2.putText(canvas, "Attention Heatmap", (w//2 - 150, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                frame = canvas
            else:
                # After showing the heatmap, continue with next image
                ATTENTION_TEST_STATE = "TESTING"
                move_to_next_test_image()
        
        elif ATTENTION_TEST_STATE == "RESULTS":
            # Show test results with option to restart or exit
            results_text = "Attention Test Completed!"
            cv2.putText(frame, results_text, (w//2 - 200, h//2 - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            # Show summary of results
            if "summary" in ATTENTION_TEST_RESULTS:
                summary = ATTENTION_TEST_RESULTS["summary"]
                cv2.putText(frame, f"Total Images: {summary['total_images_tested']}", (w//2 - 150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Total Gaze Points: {summary['total_gaze_points']}", (w//2 - 150, h//2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Results saved to: {ATTENTION_TEST_RESULTS['results_dir']}", (w//2 - 300, h//2 + 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions for next actions
            cv2.putText(frame, "Press 'R' to restart test or 'A' to exit test mode", (w//2 - 300, h//2 + 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add instruction to exit attention test mode
        cv2.putText(frame, "Press 'A' to exit attention test mode", (w//2 - 200, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

running = True
clock = pygame.time.Clock()

# Variables for threshold adjustment
threshold_adjust_mode = False

# Initialize sector-based calibration if enabled
if USE_SECTOR_CALIBRATION:
    initialize_sector_calibration()
    
# Try to load configuration from file
load_config()

print("Eye Tracker starting up...")
print(f"Calibration mode: {'Enabled' if CALIBRATION_MODE else 'Disabled'}")
print(f"Sector-based calibration: {'Enabled' if USE_SECTOR_CALIBRATION else 'Disabled'}")
print(f"Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")

def smooth_gaze_direction():
    """Apply temporal smoothing to gaze direction to reduce jitter.
    This counts occurrences of different directions in the history and
    selects the most frequent one, giving higher weight to recent samples.
    Also includes sector information for better context-aware smoothing.
    """
    global GAZE_HISTORY
    
    if not GAZE_HISTORY:
        return "Center"
    
    # Determine current sector from most recent entry
    current_sector = GAZE_HISTORY[-1]['sector']
    
    # Count directions with weighted recency
    direction_counts = {}
    total_weight = 0
    
    # Count direction occurrences with recency weighting
    for i, entry in enumerate(GAZE_HISTORY):
        # Weight by recency (more recent entries get higher weight)
        recency_weight = (i + 1) / len(GAZE_HISTORY)
        
        # Weight by sector similarity (same sector gets higher weight)
        sector_weight = 1.5 if entry['sector'] == current_sector else 1.0
        
        # Apply combined weight
        weight = recency_weight * sector_weight
        
        # Add to direction count
        direction = entry['direction']
        if direction not in direction_counts:
            direction_counts[direction] = 0
        direction_counts[direction] += weight
        total_weight += weight
    
    # Find the most common direction
    most_common_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
    
    # Don't override the current detection if it's the same as the most recent one
    current_direction = GAZE_HISTORY[-1]['direction']
    most_recent_confidence = direction_counts[current_direction] / total_weight
    most_common_confidence = direction_counts[most_common_direction] / total_weight
    
    # Use most recent detection if it's at least 80% as confident as the most common
    if current_direction != most_common_direction and most_recent_confidence >= 0.8 * most_common_confidence:
        return current_direction
    
    return most_common_direction

# Add function to draw tuning dialog
def draw_tuning_dialog(screen):
    """Draw a dialog for fine-tuning eye tracker parameters"""
    global DOWN_GAZE_SENSITIVITY
    
    dialog_width = 400
    dialog_height = 250
    x = (screen.get_width() - dialog_width) // 2
    y = (screen.get_height() - dialog_height) // 2
    
    # Draw semi-transparent background
    dialog_surface = pygame.Surface((dialog_width, dialog_height), pygame.SRCALPHA)
    dialog_surface.fill((0, 0, 0, 200))
    
    # Add title
    title_font = pygame.font.Font(None, 32)
    title = title_font.render("Eye Tracker Tuning", True, (255, 255, 255))
    dialog_surface.blit(title, (dialog_width//2 - title.get_width()//2, 15))
    
    # Add subtitle explaining what this does
    subtitle_font = pygame.font.Font(None, 24)
    subtitle = subtitle_font.render("Adjust to improve detection when looking down", True, (220, 220, 220))
    dialog_surface.blit(subtitle, (dialog_width//2 - subtitle.get_width()//2, 45))
    
    # Add controls
    font = pygame.font.Font(None, 26)
    
    # Down gaze sensitivity with clear indication of current value
    text = font.render(f"Down Gaze Sensitivity: {DOWN_GAZE_SENSITIVITY:.1f}", True, (255, 255, 255))
    dialog_surface.blit(text, (20, 80))
    
    # Draw slider background
    pygame.draw.rect(dialog_surface, (50, 50, 50), (20, 110, 360, 15), 0, 3)
    
    # Draw slider position with better visibility
    slider_pos = 20 + int(360 * (DOWN_GAZE_SENSITIVITY - 0.5) / 1.5)  # Map 0.5-2.0 to 0-360
    pygame.draw.circle(dialog_surface, (0, 200, 0), (slider_pos, 117), 10)
    
    # Draw tick marks with labels for slider positions
    ticks = [(0.5, "Less sensitive"), (1.0, "Default"), (1.5, "More"), (2.0, "Most")]
    tick_font = pygame.font.Font(None, 20)
    
    for value, label in ticks:
        x_pos = 20 + int(360 * (value - 0.5) / 1.5)
        # Draw tick mark
        pygame.draw.line(dialog_surface, (150, 150, 150), (x_pos, 125), (x_pos, 130), 2)
        # Draw label
        tick_label = tick_font.render(label, True, (200, 200, 200))
        dialog_surface.blit(tick_label, (x_pos - tick_label.get_width()//2, 132))
    
    # Instructions - made clearer
    instructions = [
        "Press 1-5 to adjust sensitivity:",
        "Higher values (4-5): Better downward detection",
        "Lower values (1-2): Less downward detection",
        "Press E to toggle this dialog"
    ]
    
    y_pos = 160
    for instruction in instructions:
        text = font.render(instruction, True, (220, 220, 220))
        dialog_surface.blit(text, (20, y_pos))
        y_pos += 25
    
    # Blit dialog to screen with slight animation effect
    dialog_rect = dialog_surface.get_rect(center=(screen.get_width()//2, screen.get_height()//2))
    screen.blit(dialog_surface, dialog_rect)

# Add more key handlers for sensitivity adjustment
if SHOW_TUNING_DIALOG:
    if event.key == pygame.K_1:
        DOWN_GAZE_SENSITIVITY = 0.5
        print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY}")
    elif event.key == pygame.K_2:
        DOWN_GAZE_SENSITIVITY = 0.75
        print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY}")
    elif event.key == pygame.K_3:
        DOWN_GAZE_SENSITIVITY = 1.0
        print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY}")
    elif event.key == pygame.K_4:
        DOWN_GAZE_SENSITIVITY = 1.5
        print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY}")
    elif event.key == pygame.K_5:
        DOWN_GAZE_SENSITIVITY = 2.0
        print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY}")

# Add code to draw the dialog in the main loop, after the screen.blit(frame_surface, (0, 0)) line
if SHOW_TUNING_DIALOG:
    draw_tuning_dialog(screen)

# Update the key help text to include E for tuning
help_text = small_font.render("ESC: exit | R: reset | L: log | D: debug | T: threshold | M: method | C: calibrate | K: filter | X: sectors | Z: sensitivity | V: verbose | E: tuning", True, (255, 255, 255))

def initialize_attention_test():
    """Initialize attention testing by finding available categories and creating results directory"""
    global ATTENTION_CATEGORIES, ATTENTION_TEST_RESULTS_DIR
    
    # Create results directory if it doesn't exist
    if not os.path.exists(ATTENTION_TEST_RESULTS_DIR):
        os.makedirs(ATTENTION_TEST_RESULTS_DIR)
    
    # Find all category directories
    category_dirs = [d for d in os.listdir(CATEGORIES_DIR) if os.path.isdir(os.path.join(CATEGORIES_DIR, d))]
    
    # Filter to include only directories that have a train subfolder
    valid_categories = []
    for category in category_dirs:
        train_dir = os.path.join(CATEGORIES_DIR, category, "train")
        if os.path.isdir(train_dir):
            valid_categories.append(category)
    
    ATTENTION_CATEGORIES = valid_categories
    
    print(f"Found {len(ATTENTION_CATEGORIES)} categories for attention testing: {ATTENTION_CATEGORIES}")
    
    return len(ATTENTION_CATEGORIES) > 0

def load_category_images(category_index):
    """Load images for a specific category"""
    global ATTENTION_TEST_IMAGES, CURRENT_IMAGE_INDEX, ATTENTION_TEST_STATE
    
    if 0 <= category_index < len(ATTENTION_CATEGORIES):
        category = ATTENTION_CATEGORIES[category_index]
        train_dir = os.path.join(CATEGORIES_DIR, category, "train")
        
        # Get all image files in the train directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(train_dir, ext)))
        
        # Get JSON annotations file
        json_file = os.path.join(train_dir, "_annotations.coco.json")
        
        # Store the image paths and reset index
        ATTENTION_TEST_IMAGES = image_files
        CURRENT_IMAGE_INDEX = 0
        
        # Initialize empty heatmap data for each image
        for image_path in ATTENTION_TEST_IMAGES:
            image_name = os.path.basename(image_path)
            ATTENTION_HEATMAP_DATA[image_name] = []
        
        print(f"Loaded {len(ATTENTION_TEST_IMAGES)} images for category: {category}")
        
        if len(ATTENTION_TEST_IMAGES) > 0:
            ATTENTION_TEST_STATE = "TESTING"
            return True
        else:
            print(f"No images found for category: {category}")
            ATTENTION_TEST_STATE = "WAITING"
            return False
    else:
        print("Invalid category index")
        ATTENTION_TEST_STATE = "WAITING"
        return False

def process_gaze_for_heatmap(gaze_x, gaze_y, frame_width, frame_height):
    """Process gaze coordinates for the current test image"""
    global ATTENTION_HEATMAP_DATA, CURRENT_TEST_IMAGE_PATH, CURRENT_TEST_IMAGE
    
    if CURRENT_TEST_IMAGE_PATH and ATTENTION_TEST_STATE == "TESTING":
        image_name = os.path.basename(CURRENT_TEST_IMAGE_PATH)
        
        # Convert gaze coordinates (0-1 normalized) to actual screen pixels
        screen_x = int(gaze_x * frame_width)
        screen_y = int(gaze_y * frame_height)
        
        # Store points for the entire screen, not just within image boundaries
        if image_name in ATTENTION_HEATMAP_DATA:
            # Only store up to MAX_GAZE_POINTS points per image
            if len(ATTENTION_HEATMAP_DATA[image_name]) < MAX_GAZE_POINTS:
                # Store the screen coordinates directly
                ATTENTION_HEATMAP_DATA[image_name].append((screen_x, screen_y))
                
                if CONSOLE_DEBUG:
                    print(f"Gaze point added at pixel ({screen_x}, {screen_y}) for image {image_name}")
                    
                return True  # Successfully added gaze point
            
    return False  # Failed to add gaze point

def load_next_test_image():
    """Load the next image for attention testing"""
    global CURRENT_TEST_IMAGE, CURRENT_TEST_IMAGE_PATH, CURRENT_IMAGE_INDEX
    global CURRENT_TEST_IMAGE_START_TIME, ATTENTION_TEST_STATE
    
    if CURRENT_IMAGE_INDEX < len(ATTENTION_TEST_IMAGES):
        # Load the image
        image_path = ATTENTION_TEST_IMAGES[CURRENT_IMAGE_INDEX]
        img = cv2.imread(image_path)
        
        if img is not None:
            CURRENT_TEST_IMAGE = img
            CURRENT_TEST_IMAGE_PATH = image_path
            CURRENT_TEST_IMAGE_START_TIME = time.time()
            print(f"Loaded test image: {os.path.basename(image_path)}")
            return True
        else:
            print(f"Failed to load image: {image_path}")
            CURRENT_IMAGE_INDEX += 1
            return load_next_test_image()  # Try the next image
    else:
        # All images in this category have been tested
        ATTENTION_TEST_STATE = "RESULTS"
        print("Completed testing all images in this category")
        generate_and_save_heatmaps()
        return False

def start_attention_test():
    """Start the attention testing mode"""
    global ATTENTION_TEST_MODE, ATTENTION_TEST_STATE, CURRENT_CATEGORY_INDEX
    global ATTENTION_HEATMAP_DATA
    
    # Initialize if not already done
    if not ATTENTION_CATEGORIES:
        if not initialize_attention_test():
            print("No valid categories found for attention testing")
            return False
    
    # Reset test data
    ATTENTION_HEATMAP_DATA = {}
    CURRENT_CATEGORY_INDEX = 0
    ATTENTION_TEST_MODE = True
    ATTENTION_TEST_STATE = "CATEGORY_SELECTION"  # Start with category selection instead of waiting
    
    print("Starting attention test - Select a category")
    return True

def move_to_next_test_image():
    """Move to the next image in the attention test"""
    global CURRENT_IMAGE_INDEX, CURRENT_CATEGORY_INDEX
    
    # Move to next image
    CURRENT_IMAGE_INDEX += 1
    
    # Check if we've gone through all images in current category
    if CURRENT_IMAGE_INDEX >= len(ATTENTION_TEST_IMAGES):
        # Move to next category
        CURRENT_CATEGORY_INDEX += 1
        
        # Check if we've gone through all categories
        if CURRENT_CATEGORY_INDEX >= len(ATTENTION_CATEGORIES):
            # Test completed
            ATTENTION_TEST_STATE = "RESULTS"
            generate_and_save_heatmaps()
            return False
        else:
            # Load next category
            if load_category_images(CURRENT_CATEGORY_INDEX):
                return load_next_test_image()
            else:
                return False
    else:
        # Load next image in current category
        return load_next_test_image()

def generate_heatmap(image_path, gaze_points):
    """Generate a heatmap visualization for an image based on gaze points"""
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image for heatmap: {image_path}")
        return None
    
    # Get the image display dimensions used during testing
    frame_h, frame_w = 0, 0
    if hasattr(pygame, 'display') and pygame.display.get_surface():
        frame_w, frame_h = pygame.display.get_surface().get_size()
    else:
        # Use fallback values if pygame display info isn't available
        frame_w, frame_h = 1024, 768
    
    # Resize the original image to fit the display
    scale = min(frame_w / image.shape[1], frame_h / image.shape[0])
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Create a blank canvas the size of the screen
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    
    # Place the resized image in the center of the canvas
    y_offset = (frame_h - new_h) // 2
    x_offset = (frame_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    
    # Create an empty heatmap for the entire screen
    heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
    
    print(f"Generating heatmap for {image_path} with {len(gaze_points)} gaze points")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}, Display size: {frame_w}x{frame_h}")
    print(f"Scaled size: {new_w}x{new_h}, Offset: ({x_offset}, {y_offset})")
    
    # Add gaussian at each gaze point
    for point in gaze_points:
        screen_x, screen_y = point
        
        # Ensure coordinates are within screen bounds
        if 0 <= screen_x < frame_w and 0 <= screen_y < frame_h:
            # Create small matrix with single 1 value at gaze point
            small_heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
            small_heatmap[screen_y, screen_x] = 1
            
            # Apply gaussian blur to create a "hotspot"
            small_heatmap = cv2.GaussianBlur(small_heatmap, (0, 0), GAUSSIAN_SIGMA)
            
            # Normalize the small heatmap
            if small_heatmap.max() > 0:
                small_heatmap = small_heatmap / small_heatmap.max()
            
            # Add to the main heatmap
            heatmap = np.maximum(heatmap, small_heatmap)
    
    # Normalize heatmap to 0-1 range
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    else:
        print("Warning: Heatmap is empty - no valid gaze points found")
    
    # Apply colormap to the heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend heatmap with canvas
    blended = cv2.addWeighted(canvas, 1.0 - HEATMAP_OPACITY, heatmap_colored, HEATMAP_OPACITY, 0)
    
    return blended

def generate_and_save_heatmaps():
    """Generate heatmaps for all tested images and save them"""
    global ATTENTION_HEATMAP_DATA, ATTENTION_TEST_RESULTS
    
    results_dir = os.path.join(ATTENTION_TEST_RESULTS_DIR, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Generating and saving heatmaps to {results_dir}")
    
    # Generate heatmap for each image
    for image_name, gaze_points in ATTENTION_HEATMAP_DATA.items():
        if not gaze_points:
            print(f"No gaze data for {image_name}, skipping")
            continue
        
        # Find original image path
        original_path = None
        for category in ATTENTION_CATEGORIES:
            train_dir = os.path.join(CATEGORIES_DIR, category, "train")
            potential_path = os.path.join(train_dir, image_name)
            if os.path.exists(potential_path):
                original_path = potential_path
                break
        
        if original_path:
            # Generate heatmap
            heatmap = generate_heatmap(original_path, gaze_points)
            
            if heatmap is not None:
                # Save heatmap
                output_path = os.path.join(results_dir, f"heatmap_{image_name}")
                cv2.imwrite(output_path, heatmap)
                print(f"Saved heatmap: {output_path}")
                
                # Save original image for reference
                cv2.imwrite(os.path.join(results_dir, f"original_{image_name}"), cv2.imread(original_path))
                
                # Save gaze point data as JSON
                gaze_data_path = os.path.join(results_dir, f"gaze_data_{os.path.splitext(image_name)[0]}.json")
                with open(gaze_data_path, 'w') as f:
                    json.dump(gaze_points, f)
        else:
            print(f"Could not find original image for {image_name}")
    
    # Create summary file
    summary_path = os.path.join(results_dir, "summary.json")
    summary = {
        "test_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "categories_tested": ATTENTION_CATEGORIES,
        "images_per_category": {cat: len([img for img in ATTENTION_TEST_IMAGES if cat in img]) for cat in ATTENTION_CATEGORIES},
        "total_images_tested": len(ATTENTION_HEATMAP_DATA),
        "total_gaze_points": sum(len(points) for points in ATTENTION_HEATMAP_DATA.values())
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Store the path to results for display
    ATTENTION_TEST_RESULTS["results_dir"] = results_dir
    ATTENTION_TEST_RESULTS["summary"] = summary
    
    print(f"Attention test results saved to {results_dir}")
    return results_dir

def select_category(index):
    """Select a category by index"""
    global CURRENT_CATEGORY_INDEX, ATTENTION_TEST_STATE
    
    if 0 <= index < len(ATTENTION_CATEGORIES):
        CURRENT_CATEGORY_INDEX = index
        if load_category_images(CURRENT_CATEGORY_INDEX):
            ATTENTION_TEST_STATE = "TESTING"
            load_next_test_image()
            return True
    return False

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
            
            # Toggle between Kalman and Low-pass filtering
            if event.key == pygame.K_k:
                use_kalman = not use_kalman
                if use_kalman:
                    # Reset Kalman state when switching to it
                    kalman_initialized = False
                    # Use less aggressive parameters
                    measurement_noise = 0.3  # Increased from 0.1 - trust measurements more
                    process_noise = 0.025     # Increased from 0.01 - allow more natural movement
                print(f"Using {'Kalman' if use_kalman else 'Low-pass'} filter")
            
            # Increase smoothing (more filtering)
            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                if use_kalman:
                    # For Kalman, decrease measurement noise (trust measurements less)
                    measurement_noise = max(0.01, measurement_noise * 0.8)
                    print(f"Kalman measurement noise decreased to {measurement_noise:.4f}")
                else:
                    # For low-pass, decrease smoothing factor (more smoothing)
                    smoothing_factor = max(0.05, smoothing_factor * 0.8)
                    print(f"Smoothing factor decreased to {smoothing_factor:.4f} (more smoothing)")
            
            # Decrease smoothing (less filtering)
            if event.key == pygame.K_MINUS:
                if use_kalman:
                    # For Kalman, increase measurement noise (trust measurements more)
                    measurement_noise = min(1.0, measurement_noise * 1.2)
                    print(f"Kalman measurement noise increased to {measurement_noise:.4f}")
                else:
                    # For low-pass, increase smoothing factor (less smoothing)
                    smoothing_factor = min(0.5, smoothing_factor * 1.2)
                    print(f"Smoothing factor increased to {smoothing_factor:.4f} (less smoothing)")
                    
            # Reset filtering parameters to defaults
            if event.key == pygame.K_0:
                if use_kalman:
                    kalman_initialized = False
                    measurement_noise = 0.3    # Increased from 0.1
                    process_noise = 0.025      # Increased from 0.01
                    print("Kalman filter parameters reset to defaults")
                else:
                    smoothing_factor = 0.25    # Increased from 0.15 - less smoothing
                    print("Low-pass filter parameters reset to defaults")
            
            # Find the event.key handling section starting with "if event.type == pygame.KEYDOWN:" and add this new case
            if event.key == pygame.K_x:  # Toggle sector-based calibration
                USE_SECTOR_CALIBRATION = not USE_SECTOR_CALIBRATION
                if USE_SECTOR_CALIBRATION and not SECTOR_CALIBRATION_DATA:
                    initialize_sector_calibration()
                print(f"Sector-based calibration: {'ON' if USE_SECTOR_CALIBRATION else 'OFF'}")
            if event.key == pygame.K_z:  # Adjust sector distance weight
                # Toggle between 3 different values
                if SECTOR_DISTANCE_WEIGHT == 1.0:
                    SECTOR_DISTANCE_WEIGHT = 0.5  # More sensitive (smaller distances)
                    print(f"Sector distance weight decreased to {SECTOR_DISTANCE_WEIGHT:.1f} (more sensitive)")
                elif SECTOR_DISTANCE_WEIGHT == 0.5:
                    SECTOR_DISTANCE_WEIGHT = 2.0  # Less sensitive (larger distances)
                    print(f"Sector distance weight increased to {SECTOR_DISTANCE_WEIGHT:.1f} (less sensitive)")
                else:
                    SECTOR_DISTANCE_WEIGHT = 1.0  # Back to default
                    print(f"Sector distance weight reset to {SECTOR_DISTANCE_WEIGHT:.1f} (default)")
            
            # Add this new key handler
            if event.key == pygame.K_v:  # V for verbose
                CONSOLE_DEBUG = not CONSOLE_DEBUG
                print(f"Console debug output: {'ON' if CONSOLE_DEBUG else 'OFF'}")
                
            # Update help text to include V key
            help_text = small_font.render("ESC: exit | R: reset | L: log | D: debug | T: threshold | M: method | C: calibrate | K: filter | X: sectors | Z: sensitivity | V: verbose", True, (255, 255, 255))
            
            # Add new key handler in the main loop
            if event.key == pygame.K_e:  # E for eye tuning
                SHOW_TUNING_DIALOG = not SHOW_TUNING_DIALOG
                print(f"Tuning dialog: {'visible' if SHOW_TUNING_DIALOG else 'hidden'}")
                
            # Down gaze sensitivity adjustment
            if event.key == pygame.K_1:
                DOWN_GAZE_SENSITIVITY = 0.5
                print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY} (less sensitive)")
            elif event.key == pygame.K_2:
                DOWN_GAZE_SENSITIVITY = 0.75
                print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY} (somewhat less sensitive)")
            elif event.key == pygame.K_3:
                DOWN_GAZE_SENSITIVITY = 1.0
                print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY} (default)")
            elif event.key == pygame.K_4:
                DOWN_GAZE_SENSITIVITY = 1.5
                print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY} (more sensitive)")
            elif event.key == pygame.K_5:
                DOWN_GAZE_SENSITIVITY = 2.0
                print(f"Down gaze sensitivity set to {DOWN_GAZE_SENSITIVITY} (much more sensitive)")
                
            # Attention testing key handler
            if event.key == pygame.K_a:  # Toggle attention test mode
                if not ATTENTION_TEST_MODE:
                    if start_attention_test():
                        print("Starting attention testing mode")
                    else:
                        print("Failed to start attention testing mode")
                else:
                    ATTENTION_TEST_MODE = False
                    ATTENTION_TEST_STATE = "WAITING"
                    print("Exited attention testing mode")
            
            # Handle category selection (keys 1-9)
            if ATTENTION_TEST_MODE and ATTENTION_TEST_STATE == "CATEGORY_SELECTION":
                # Check for number keys 1-9
                if pygame.K_1 <= event.key <= pygame.K_9:
                    category_index = event.key - pygame.K_1  # Convert to 0-based index
                    if select_category(category_index):
                        print(f"Selected category: {ATTENTION_CATEGORIES[category_index]}")
                    else:
                        print(f"Invalid category selection")
            
            # Handle space bar for continuing in attention test
            if event.key == pygame.K_SPACE and ATTENTION_TEST_MODE:
                if ATTENTION_TEST_STATE == "CATEGORY_SELECTION":
                    # Default to first category
                    if select_category(0):
                        print(f"Selected default category: {ATTENTION_CATEGORIES[0]}")
                elif ATTENTION_TEST_STATE == "RESULTS":
                    start_attention_test()  # Restart test
                    
            # Update help text to include all options
            help_text = small_font.render("ESC: exit | R: reset | L: log | D: debug | T: threshold | M: method | C: calibrate | K: filter | X: sectors | Z: sensitivity | V: verbose | E: tuning | A: attention test", True, (255, 255, 255))
    
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
    
    # Draw tuning dialog if enabled
    if SHOW_TUNING_DIALOG:
        draw_tuning_dialog(screen)
    
    # Add help text at the bottom
    help_text = small_font.render("ESC: exit | R: reset | L: log | D: debug | T: threshold | M: method | C: calibrate | K: filter | X: sectors | Z: sensitivity | V: verbose | E: tuning", True, (255, 255, 255))
    screen.blit(help_text, (10, display_height - 30))
    
    # Add a second line of help text for filtering options
    filter_help = small_font.render("+/-: adjust smoothing | 0: reset filter", True, (255, 255, 255))
    screen.blit(filter_help, (10, display_height - 15))
    
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