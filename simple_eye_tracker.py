import cv2
import numpy as np
import pygame
import sys
import os
import time
import mediapipe as mp
import math

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the EyeGestures-main directory to the Python path
sys.path.append(os.path.join(current_dir, 'EyeGestures-main'))

# Import EyeGestures components
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v2

# Initialize pygame
pygame.init()
pygame.font.init()

# Get display information
screen_info = pygame.display.Info()
screen_width = screen_info.current_w
screen_height = screen_info.current_h

# Set up display window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Simple Eye Tracker")

# Initialize fonts
font = pygame.font.Font(None, 36)
large_font = pygame.font.Font(None, 48)
large_font.set_bold(True)
small_font = pygame.font.Font(None, 24)

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIME = (50, 205, 50)

# Initialize eye tracker
gestures = EyeGestures_v2()
cap = VideoCapture(0)  # Use default camera (0)

# Add more specific calibration variables
left_side_correction_factor = 2.0  # Increased from 1.5 to 2.0 for stronger left side correction
current_point_needs_reset = True  # Flag to ensure proper point reset handling
calibration_point_timeout = 10.0   # Increased from 8.0s to give more time for calibration

# Add non-linear correction parameters
use_non_linear_correction = True  # Enable non-linear correction
horizontal_warp_factor = 0.5      # Increased from 0.3 to 0.5 for stronger horizontal correction
left_weight_threshold = 0.5       # Screen position threshold for left side (normalized 0-1)

# Add improved parameters for calibration and correction
calibration_min_points_required = 15  # Increased from 10 to ensure better quality calibration
dynamic_correction_enabled = True  # Enable enhanced dynamic correction based on position
left_side_boost_factor = 2.2  # Increased from 1.8 to 2.2 for stronger far left correction
max_correction_distance = 200  # Increased from 150 to 200 to allow greater influence range

# Constants for face detection stability
face_detection_grace_period = 1.5  # Increased from 1.0 to 1.5 for more tolerance
max_detection_failures = 5  # Increased from 3 to 5 for more robustness

# Quality threshold for calibration data
calibration_min_fixation_threshold = 0.25  # Reduced from 0.3 to 0.25 for more lenient fixation detection
calibration_min_stable_time = 1.5  # Increased from 1.0 to 1.5 for better stability

# Create a custom calibration grid with more points on the left side
def create_asymmetric_calibration_grid():
    """Create a calibration grid with more points on the left side of the screen"""
    points = []
    
    # Right side points (fewer)
    right_x = [0.6, 0.8]
    right_y = [0.2, 0.5, 0.8]
    
    # Left side points (more)
    left_x = [0.1, 0.2, 0.3, 0.4]
    left_y = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Add right side points
    for x in right_x:
        for y in right_y:
            points.append([x, y])
            
    # Add left side points (more dense)
    for x in left_x:
        for y in left_y:
            points.append([x, y])
            
    # Add center column
    center_x = 0.5
    center_y = [0.15, 0.5, 0.85]
    for y in center_y:
        points.append([center_x, y])
    
    # Convert to numpy array and randomize
    points = np.array(points)
    np.random.shuffle(points)
    
    return points

# Replace the calibration grid with the asymmetric one
# Create a simple calibration grid (3x3 instead of 5x5)
if True:  # Easy toggle for asymmetric grid
    # Asymmetric grid with more points on the left side
    calibration_map = create_asymmetric_calibration_grid()
    # Limit to 9 points for a quick calibration
    calibration_map = calibration_map[:9]
    calibration_points = min(9, len(calibration_map))
else:
    # Original grid
    x = np.arange(0.2, 0.9, 0.3)  # 3 points with better spread
    y = np.arange(0.2, 0.9, 0.3)  # 3 points with better spread
    xx, yy = np.meshgrid(x, y)
    calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
    np.random.shuffle(calibration_map)  # Randomize calibration points
    calibration_points = 5  # Reduced from 25 to 5 points

# Store calibration points in screen coordinates
calibration_points_list = []
for point in calibration_map:
    # Convert normalized coordinates to screen coordinates
    screen_x = int(point[0] * screen_width)
    screen_y = int(point[1] * screen_height)
    calibration_points_list.append((screen_x, screen_y))

# Upload calibration map and configure the tracker
gestures.uploadCalibrationMap(calibration_map, context="main")
gestures.setClassicalImpact(2)  # Balance between V1 and V2 algorithms
gestures.setFixation(1.0)  # Sensitivity for fixation detection

# Initialize variables
clock = pygame.time.Clock()
running = True
calibration_counter = 0
prev_x, prev_y = 0, 0
gaze_history = []  # Store recent gaze points for smoothing
face_detection_failure = False
last_detection_time = time.time()
consecutive_detection_failures = 0  # Track consecutive detection failures

# Track application state
# States: "detection_check", "calibration", "tracking"
app_state = "detection_check"
detection_success_count = 0
detection_required_successes = 30  # Need 30 successful detections to proceed
detection_success_threshold = 0.5  # 50% success rate
calibration_point_duration = 5.0  # Each point is shown for 5 seconds
current_calibration_time = 0  # Time spent on current calibration point
calibration_countdown = 5.0  # Countdown for current point
is_adapting_calibration = False  # Flag to indicate if we're adapting calibration
calibration_point_start_time = None  # Absolute timestamp when current point started

# Add countdown for initial system warmup
system_warmup_time = 3.0  # 3 seconds for system to stabilize
warmup_start_time = time.time()
is_warming_up = True

# Add variables for adaptive calibration
adaptive_calibration_enabled = True  # Enable adaptive calibration
gaze_points_for_current_target = []  # Store gaze points for current calibration target
calibration_adaptations = {}  # Store adaptations for each point
calibration_point_idx = 0  # Current calibration point index
current_calibration_point = None  # Current point being calibrated

# Initialize mediapipe for direct face detection (backup visualization)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Face landmarks for visualization
face_landmarks = None
left_eye_landmarks = None
right_eye_landmarks = None
left_upper_eyelid = None
left_lower_eyelid = None
right_upper_eyelid = None
right_lower_eyelid = None
debug_landmarks_enabled = True  # Flag to toggle landmarks visualization
use_direct_mediapipe = True  # Use direct MediaPipe detection for visualization

# Add a variable to track first frame of a calibration point
is_first_calibration_frame = True
calibration_point_stability_threshold = 20  # Increased from 15 pixels to 20 pixels to handle more jitter

# Add function to calculate eye openness
def calculate_eye_openness(upper_eyelid, lower_eyelid):
    """Calculate eye openness as a percentage based on eyelid landmarks"""
    if upper_eyelid is None or lower_eyelid is None or len(upper_eyelid) < 3 or len(lower_eyelid) < 3:
        return None
    
    # Calculate the average vertical distance between upper and lower eyelids
    # Using the middle points for more accurate measurement
    mid_upper = upper_eyelid[len(upper_eyelid)//2]
    mid_lower = lower_eyelid[len(lower_eyelid)//2]
    vertical_distance = abs(mid_lower[1] - mid_upper[1])
    
    # Get the eye width (horizontal distance) for normalization
    leftmost_x = min(min(pt[0] for pt in upper_eyelid), min(pt[0] for pt in lower_eyelid))
    rightmost_x = max(max(pt[0] for pt in upper_eyelid), max(pt[0] for pt in lower_eyelid))
    eye_width = rightmost_x - leftmost_x
    
    # Normalize openness as a percentage of eye width
    # Typical fully open eye is about 30-40% of eye width
    normalized_openness = (vertical_distance / eye_width) * 100
    
    # Cap at 100% for visualization purposes
    return min(100, normalized_openness * 2.5)  # Scaling factor to make it more intuitive

# Add eye openness state variables
left_eye_openness = None
right_eye_openness = None

def draw_gaze_point(screen, point, is_fixation, raw_point=None):
    """Draw the gaze point with enhanced visualization
    
    Args:
        screen: Pygame screen to draw on
        point: The corrected gaze point (x, y)
        is_fixation: Whether the point is considered a fixation
        raw_point: Optional raw gaze point before correction for debugging
    """
    x, y = point
    
    # Draw primary gaze point (corrected)
    if is_fixation:
        # Draw larger filled circle for fixations
        pygame.draw.circle(screen, RED, (int(x), int(y)), 10)
        # Add a contrasting outline for better visibility
        pygame.draw.circle(screen, WHITE, (int(x), int(y)), 11, 1)
        # Add a targeting reticle for precision
        pygame.draw.line(screen, WHITE, (int(x) - 15, int(y)), (int(x) + 15, int(y)), 1)
        pygame.draw.line(screen, WHITE, (int(x), int(y) - 15), (int(x), int(y) + 15), 1)
    else:
        # Smaller circle for non-fixation points
        pygame.draw.circle(screen, ORANGE, (int(x), int(y)), 6)
        pygame.draw.circle(screen, WHITE, (int(x), int(y)), 7, 1)
    
    # Draw raw gaze point if provided (for debugging)
    if raw_point:
        raw_x, raw_y = raw_point
        # Draw smaller circle for raw point
        pygame.draw.circle(screen, BLUE, (int(raw_x), int(raw_y)), 4)
        
        # Draw line connecting raw and corrected points to visualize the correction
        pygame.draw.line(screen, CYAN, (int(raw_x), int(raw_y)), (int(x), int(y)), 1)
        
        # Add distance marker to show correction magnitude
        mid_x = (raw_x + x) / 2
        mid_y = (raw_y + y) / 2
        correction_dist = int(math.sqrt((raw_x - x)**2 + (raw_y - y)**2))
        
        # Only show distance text if significant correction applied
        if correction_dist > 10:
            dist_text = small_font.render(f"{correction_dist}px", True, CYAN)
            screen.blit(dist_text, (int(mid_x), int(mid_y)))
            
    # Draw additional visual element to enhance visibility for current gaze position
    alpha_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
    # Pulsating intensity based on time
    intensity = int(128 + 127 * math.sin(time.time() * 4))
    pygame.draw.circle(alpha_surf, (255, 0, 0, intensity), (10, 10), 8)
    screen.blit(alpha_surf, (int(x) - 10, int(y) - 10))

def draw_face_detection_warning(screen):
    """Draw a warning when face detection fails"""
    warning_text = font.render("No face detected! Position yourself in front of the camera.", True, YELLOW)
    warning_rect = warning_text.get_rect(center=(screen_width//2, screen_height//2))
    screen.blit(warning_text, warning_rect)
    
    suggestion_text = font.render("Ensure good lighting and that your face is clearly visible", True, WHITE)
    suggestion_rect = suggestion_text.get_rect(center=(screen_width//2, screen_height//2 + 50))
    screen.blit(suggestion_text, suggestion_rect)

def draw_detection_phase(screen, success_count, required_successes, success_rate):
    """Draw the face/eye detection phase status"""
    # Draw progress bar background
    bar_width = 500
    bar_height = 30
    bar_x = (screen_width - bar_width) // 2
    bar_y = screen_height // 2 + 100
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    
    # Draw progress
    progress = min(1.0, success_count / required_successes)
    progress_width = int(bar_width * progress)
    progress_color = GREEN if success_rate >= detection_success_threshold else YELLOW
    pygame.draw.rect(screen, progress_color, (bar_x, bar_y, progress_width, bar_height))
    
    # Draw text
    title_text = large_font.render("Face and Eye Detection Check", True, WHITE)
    title_rect = title_text.get_rect(center=(screen_width//2, screen_height//2 - 100))
    screen.blit(title_text, title_rect)
    
    instruction_text = font.render("Please look at the camera and keep your face visible", True, WHITE)
    instruction_rect = instruction_text.get_rect(center=(screen_width//2, screen_height//2 - 40))
    screen.blit(instruction_text, instruction_rect)
    
    status_text = font.render(
        f"Detection rate: {success_rate*100:.1f}% ({success_count}/{required_successes})", 
        True, 
        WHITE
    )
    status_rect = status_text.get_rect(center=(screen_width//2, screen_height//2 + 150))
    screen.blit(status_text, status_rect)
    
    # Add spacebar instruction
    if success_count >= 10:  # Only show after some successful detections
        spacebar_text = font.render("Press SPACEBAR to continue to calibration", True, CYAN)
        spacebar_rect = spacebar_text.get_rect(center=(screen_width//2, screen_height//2 + 200))
        screen.blit(spacebar_text, spacebar_rect)
    
    if success_rate < detection_success_threshold:
        warning_text = font.render("Detection rate is low. Please check lighting and positioning.", True, YELLOW)
        warning_rect = warning_text.get_rect(center=(screen_width//2, screen_height//2 + 250))
        screen.blit(warning_text, warning_rect)

def draw_face_and_eye_landmarks(screen, frame_surf_rect):
    """Draw face and eye landmarks for debugging"""
    if not debug_landmarks_enabled:
        return
        
    # Scale factor from camera frame to display
    # Use fixed values since we can't access from cap.get()
    camera_width = 640  # Default camera width
    camera_height = 480  # Default camera height
    scale_x = frame_surf_rect.width / camera_width
    scale_y = frame_surf_rect.height / camera_height
    
    # Draw face landmarks if available
    if face_landmarks is not None:
        # Offset for the frame position
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw face outline
        for landmark in face_landmarks:
            # Scale and offset the landmark coordinates
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            
            # Draw landmark point
            pygame.draw.circle(screen, YELLOW, (x, y), 2)
    
    # Draw eye landmarks if available
    if left_eye_landmarks is not None:
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw left eye outline
        for landmark in left_eye_landmarks:
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            pygame.draw.circle(screen, GREEN, (x, y), 3)
            
        # Connect the landmarks to form an eye outline
        if len(left_eye_landmarks) >= 2:
            points = [(int(pt[0] * scale_x) + offset_x, int(pt[1] * scale_y) + offset_y) for pt in left_eye_landmarks]
            pygame.draw.lines(screen, GREEN, True, points, 1)
    
    # Draw right eye landmarks
    if right_eye_landmarks is not None:
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw right eye outline
        for landmark in right_eye_landmarks:
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            pygame.draw.circle(screen, BLUE, (x, y), 3)
            
        # Connect the landmarks to form an eye outline
        if len(right_eye_landmarks) >= 2:
            points = [(int(pt[0] * scale_x) + offset_x, int(pt[1] * scale_y) + offset_y) for pt in right_eye_landmarks]
            pygame.draw.lines(screen, BLUE, True, points, 1)
    
    # Draw eyelid landmarks
    # Left upper eyelid (Magenta)
    if left_upper_eyelid is not None and len(left_upper_eyelid) >= 2:
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw points
        for landmark in left_upper_eyelid:
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            pygame.draw.circle(screen, MAGENTA, (x, y), 3)
        
        # Connect points with a thicker line
        points = [(int(pt[0] * scale_x) + offset_x, int(pt[1] * scale_y) + offset_y) for pt in left_upper_eyelid]
        pygame.draw.lines(screen, MAGENTA, False, points, 2)
    
    # Left lower eyelid (Cyan)
    if left_lower_eyelid is not None and len(left_lower_eyelid) >= 2:
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw points
        for landmark in left_lower_eyelid:
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            pygame.draw.circle(screen, CYAN, (x, y), 3)
        
        # Connect points with a thicker line
        points = [(int(pt[0] * scale_x) + offset_x, int(pt[1] * scale_y) + offset_y) for pt in left_lower_eyelid]
        pygame.draw.lines(screen, CYAN, False, points, 2)
    
    # Right upper eyelid (Purple)
    if right_upper_eyelid is not None and len(right_upper_eyelid) >= 2:
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw points
        for landmark in right_upper_eyelid:
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            pygame.draw.circle(screen, PURPLE, (x, y), 3)
        
        # Connect points with a thicker line
        points = [(int(pt[0] * scale_x) + offset_x, int(pt[1] * scale_y) + offset_y) for pt in right_upper_eyelid]
        pygame.draw.lines(screen, PURPLE, False, points, 2)
    
    # Right lower eyelid (Lime)
    if right_lower_eyelid is not None and len(right_lower_eyelid) >= 2:
        offset_x = frame_surf_rect.left
        offset_y = frame_surf_rect.top
        
        # Draw points
        for landmark in right_lower_eyelid:
            x = int(landmark[0] * scale_x) + offset_x
            y = int(landmark[1] * scale_y) + offset_y
            pygame.draw.circle(screen, LIME, (x, y), 3)
        
        # Connect points with a thicker line
        points = [(int(pt[0] * scale_x) + offset_x, int(pt[1] * scale_y) + offset_y) for pt in right_lower_eyelid]
        pygame.draw.lines(screen, LIME, False, points, 2)
    
    # Draw legend for eyelid visualization
    if debug_landmarks_enabled and (left_upper_eyelid is not None or right_upper_eyelid is not None):
        legend_x = 10
        legend_y = 50
        legend_spacing = 25
        
        # Left eye legend
        pygame.draw.circle(screen, GREEN, (legend_x, legend_y), 5)
        legend_text = small_font.render("Left Eye", True, WHITE)
        screen.blit(legend_text, (legend_x + 15, legend_y - 10))
        
        # Left upper eyelid
        pygame.draw.circle(screen, MAGENTA, (legend_x, legend_y + legend_spacing), 5)
        legend_text = small_font.render("Left Upper Eyelid", True, WHITE)
        screen.blit(legend_text, (legend_x + 15, legend_y + legend_spacing - 10))
        
        # Left lower eyelid
        pygame.draw.circle(screen, CYAN, (legend_x, legend_y + 2*legend_spacing), 5)
        legend_text = small_font.render("Left Lower Eyelid", True, WHITE)
        screen.blit(legend_text, (legend_x + 15, legend_y + 2*legend_spacing - 10))
        
        # Right eye legend
        pygame.draw.circle(screen, BLUE, (legend_x, legend_y + 3*legend_spacing), 5)
        legend_text = small_font.render("Right Eye", True, WHITE)
        screen.blit(legend_text, (legend_x + 15, legend_y + 3*legend_spacing - 10))
        
        # Right upper eyelid
        pygame.draw.circle(screen, PURPLE, (legend_x, legend_y + 4*legend_spacing), 5)
        legend_text = small_font.render("Right Upper Eyelid", True, WHITE)
        screen.blit(legend_text, (legend_x + 15, legend_y + 4*legend_spacing - 10))
        
        # Right lower eyelid
        pygame.draw.circle(screen, LIME, (legend_x, legend_y + 5*legend_spacing), 5)
        legend_text = small_font.render("Right Lower Eyelid", True, WHITE)
        screen.blit(legend_text, (legend_x + 15, legend_y + 5*legend_spacing - 10))

    # Draw eye openness percentages
    if left_eye_openness is not None or right_eye_openness is not None:
        openness_y = frame_surf_rect.bottom + 20
        
        # Left eye openness
        if left_eye_openness is not None:
            # Bar background
            bar_width = 150
            bar_height = 15
            bar_x = frame_surf_rect.left + 100
            pygame.draw.rect(screen, (50, 50, 50), (bar_x, openness_y, bar_width, bar_height))
            
            # Fill bar based on openness percentage
            fill_width = int((left_eye_openness / 100) * bar_width)
            fill_color = GREEN if left_eye_openness > 70 else (YELLOW if left_eye_openness > 40 else RED)
            pygame.draw.rect(screen, fill_color, (bar_x, openness_y, fill_width, bar_height))
            
            # Text label
            text = small_font.render(f"Left Eye: {left_eye_openness:.1f}% open", True, WHITE)
            screen.blit(text, (bar_x - 90, openness_y - 2))
        
        # Right eye openness
        if right_eye_openness is not None:
            # Bar background
            bar_width = 150
            bar_height = 15
            bar_x = frame_surf_rect.left + 400
            pygame.draw.rect(screen, (50, 50, 50), (bar_x, openness_y, bar_width, bar_height))
            
            # Fill bar based on openness percentage
            fill_width = int((right_eye_openness / 100) * bar_width)
            fill_color = GREEN if right_eye_openness > 70 else (YELLOW if right_eye_openness > 40 else RED)
            pygame.draw.rect(screen, fill_color, (bar_x, openness_y, fill_width, bar_height))
            
            # Text label
            text = small_font.render(f"Right Eye: {right_eye_openness:.1f}% open", True, WHITE)
            screen.blit(text, (bar_x - 90, openness_y - 2))
            
            # Add a warning if eyes are too closed
            if (left_eye_openness is not None and left_eye_openness < 50) or \
               (right_eye_openness is not None and right_eye_openness < 50):
                warning_text = font.render("Eye openness is low! This may affect calibration accuracy.", True, YELLOW)
                screen.blit(warning_text, (frame_surf_rect.centerx - 250, openness_y + 30))

def process_face_with_mediapipe(frame):
    """Use MediaPipe to directly detect face and eyes for visualization"""
    global face_landmarks, left_eye_landmarks, right_eye_landmarks
    global left_upper_eyelid, left_lower_eyelid, right_upper_eyelid, right_lower_eyelid
    global left_eye_openness, right_eye_openness
    
    if not use_direct_mediapipe:
        return
    
    try:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmark = results.multi_face_landmarks[0]
            
            # Extract face landmarks
            h, w, _ = frame.shape
            face_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmark.landmark]
            
            # Define eye landmark indices for MediaPipe Face Mesh
            # Left eye indices (approximately)
            left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
            # Right eye indices (approximately)
            right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
            
            # Define eyelid indices
            # Left upper eyelid (top to bottom)
            left_upper_eyelid_indices = [157, 158, 159, 160, 161, 246]
            # Left lower eyelid (top to bottom)
            left_lower_eyelid_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133]
            # Right upper eyelid (top to bottom)
            right_upper_eyelid_indices = [384, 385, 386, 387, 388, 466]
            # Right lower eyelid (top to bottom)
            right_lower_eyelid_indices = [362, 398, 382, 381, 380, 374, 373, 390, 249]
            
            # Extract eye landmarks
            left_eye_landmarks = [face_landmarks[idx] for idx in left_eye_indices]
            right_eye_landmarks = [face_landmarks[idx] for idx in right_eye_indices]
            
            # Extract eyelid landmarks
            left_upper_eyelid = [face_landmarks[idx] for idx in left_upper_eyelid_indices]
            left_lower_eyelid = [face_landmarks[idx] for idx in left_lower_eyelid_indices]
            right_upper_eyelid = [face_landmarks[idx] for idx in right_upper_eyelid_indices]
            right_lower_eyelid = [face_landmarks[idx] for idx in right_lower_eyelid_indices]
            
            # Calculate eye openness percentages
            left_eye_openness = calculate_eye_openness(left_upper_eyelid, left_lower_eyelid)
            right_eye_openness = calculate_eye_openness(right_upper_eyelid, right_lower_eyelid)
            
        else:
            # No face detected
            face_landmarks = None
            left_eye_landmarks = None
            right_eye_landmarks = None
            left_upper_eyelid = None
            left_lower_eyelid = None
            right_upper_eyelid = None
            right_lower_eyelid = None
            left_eye_openness = None
            right_eye_openness = None
    except Exception as e:
        print(f"MediaPipe detection error: {e}")
        # Don't crash the app if MediaPipe fails
        pass

# Detection statistics
detection_attempts = 0
detection_successes = 0

# Debug visualization - try to get face and eye landmarks from EyeGestures
def get_landmarks_from_gesture_event(event):
    """Attempt to extract face and eye landmarks from gesture event for debugging"""
    global face_landmarks, left_eye_landmarks, right_eye_landmarks
    
    try:
        # First try to get landmarks directly from the gesture engine internal data
        if hasattr(gestures, '_face') and gestures._face is not None:
            if hasattr(gestures._face, 'landmarks'):
                face_landmarks = gestures._face.landmarks
            
            if hasattr(gestures._face, 'left_eye') and hasattr(gestures._face.left_eye, 'landmarks'):
                left_eye_landmarks = gestures._face.left_eye.landmarks
                
            if hasattr(gestures._face, 'right_eye') and hasattr(gestures._face.right_eye, 'landmarks'):
                right_eye_landmarks = gestures._face.right_eye.landmarks
        
        # Try to access via internal event objects if available
        if event is not None:
            # These attributes may or may not exist, depending on the internal implementation
            if hasattr(event, 'face_landmarks') and event.face_landmarks is not None:
                face_landmarks = event.face_landmarks
            
            if hasattr(event, 'left_eye_landmarks') and event.left_eye_landmarks is not None:
                left_eye_landmarks = event.left_eye_landmarks
                
            if hasattr(event, 'right_eye_landmarks') and event.right_eye_landmarks is not None:
                right_eye_landmarks = event.right_eye_landmarks
                
            # If the landmarks aren't directly accessible, try to access via internal objects
            if hasattr(event, '_face') and event._face is not None:
                if hasattr(event._face, 'landmarks'):
                    face_landmarks = event._face.landmarks
                
                if hasattr(event._face, 'left_eye') and hasattr(event._face.left_eye, 'landmarks'):
                    left_eye_landmarks = event._face.left_eye.landmarks
                    
                if hasattr(event._face, 'right_eye') and hasattr(event._face.right_eye, 'landmarks'):
                    right_eye_landmarks = event._face.right_eye.landmarks
    except Exception as e:
        print(f"Error accessing landmarks: {e}")
        # Don't let visualization errors crash the application
        pass

# Add function to check if eyes are open enough for calibration
def are_eyes_open_enough_for_calibration():
    """Check if eyes are open enough to proceed with calibration"""
    # If we don't have eye openness data, allow calibration (can't check)
    if left_eye_openness is None and right_eye_openness is None:
        return True
    
    # Reduce the eye openness threshold to be more lenient
    left_ok = left_eye_openness is None or left_eye_openness > 40  # Reduced from 60 to 40
    right_ok = right_eye_openness is None or right_eye_openness > 40  # Reduced from 60 to 40
    
    return left_ok or right_ok

# Add variables to track successful calibration
successful_calibration_points = set()  # Set of indices that were successfully calibrated

def draw_calibration_point(screen, point, radius, progress, is_active=True, is_successful=False):
    """Draw a calibration point with visual feedback and countdown"""
    # Draw outer ring (progress indicator)
    if is_active:
        # Calculate progress arc angle
        start_angle = -90  # Start from top
        end_angle = start_angle + (progress * 360)
        
        # Draw progress arc
        pygame.draw.arc(
            screen, 
            WHITE, 
            (point[0] - radius - 10, point[1] - radius - 10, 
             (radius + 10) * 2, (radius + 10) * 2),
            math.radians(start_angle), 
            math.radians(end_angle), 
            5  # Line width
        )
        
        # Draw calibration point (pulsating)
        pulse = 0.75 + 0.25 * math.sin(pygame.time.get_ticks() / 200)
        inner_radius = int(radius * pulse)
        pygame.draw.circle(screen, BLUE, point, inner_radius)
        
        # Draw small center dot
        pygame.draw.circle(screen, WHITE, point, 3)
        
        # Draw countdown timer (make sure it's not negative)
        time_left = max(0, math.ceil(calibration_countdown))
        timer_text = large_font.render(f"{time_left}", True, WHITE)
        text_rect = timer_text.get_rect(center=(point[0], point[1] - radius - 30))
        screen.blit(timer_text, text_rect)
        
        # Add explicit progress indicator
        progress_pct = int((progress) * 100)
        progress_text = small_font.render(f"{progress_pct}%", True, GREEN)
        progress_rect = progress_text.get_rect(center=(point[0], point[1] + radius + 20))
        screen.blit(progress_text, progress_rect)
    elif is_successful:
        # Draw successfully calibrated point (green)
        pygame.draw.circle(screen, GREEN, point, radius, 2)
        pygame.draw.circle(screen, GREEN, point, 5)
        # Add checkmark
        checkmark_text = small_font.render("✓", True, GREEN)
        checkmark_rect = checkmark_text.get_rect(center=(point[0], point[1] - 15))
        screen.blit(checkmark_text, checkmark_rect)
    else:
        # Draw inactive point (outline only) or failed point (red)
        color = RED if point in calibration_points_list[:calibration_counter] else (100, 100, 100)
        pygame.draw.circle(screen, color, point, radius, 2)

def adapt_calibration_point(current_point, gaze_points):
    """Adapt next calibration point based on previous calibration data"""
    if not gaze_points or len(gaze_points) < calibration_min_points_required:
        print(f"Warning: Only {len(gaze_points)} points collected. Need at least {calibration_min_points_required} for adaptation.")
        return None  # Not enough data to adapt
    
    # Calculate median gaze position - more robust than average against outliers
    x_values = [p[0] for p in gaze_points]
    y_values = [p[1] for p in gaze_points]
    x_values.sort()
    y_values.sort()
    
    # Use median for more robustness against outliers
    median_x = x_values[len(x_values) // 2]
    median_y = y_values[len(y_values) // 2]
    
    # Calculate error vector between target and actual gaze
    error_x = current_point[0] - median_x
    error_y = current_point[1] - median_y
    
    # Calculate standard deviation to assess gaze stability
    std_x = np.std(x_values)
    std_y = np.std(y_values)
    gaze_jitter = (std_x, std_y)
    print(f"Gaze jitter: x={std_x:.1f}px, y={std_y:.1f}px")
    
    # Use jitter to weigh the correction factor - lower jitter means more confident correction
    jitter_factor = max(1.0, min(2.5, 10.0 / (std_x + std_y + 1.0)))
    
    # Apply stronger correction for more stable gaze (low jitter)
    error_x *= jitter_factor
    error_y *= jitter_factor
    
    # Calculate error magnitude
    error_magnitude = math.sqrt(error_x**2 + error_y**2)
    print(f"Raw error magnitude: {error_magnitude:.1f}px")
    
    # Special correction for left side of screen which often needs more adjustment
    is_left_side = current_point[0] < (screen_width * 0.4)
    if is_left_side:
        # Apply stronger correction on left side based on how far left
        left_factor = left_side_boost_factor * (1 + (0.4 - current_point[0]/screen_width) * 1.5)
        error_x *= left_factor
        print(f"Applied left side boost factor: {left_factor:.2f}")
    
    # Enhanced maximum correction limits based on position on screen
    max_x_correction = 300  # Increased from 250
    max_y_correction = 300  # Increased from 250
    
    # Only adapt if error is significant but not extreme (to prevent wild corrections)
    if error_magnitude < 15:  # Reduced threshold from 20px to 15px
        print(f"Error magnitude too small ({error_magnitude:.1f}px), minimal adaptation applied.")
        # Apply minimal correction even for small errors
        scale_factor = 15 / error_magnitude if error_magnitude > 0 else 0
        error_x *= scale_factor * 0.5  # Apply half of minimal correction
        error_y *= scale_factor * 0.5
    elif (abs(error_x) > max_x_correction) or (abs(error_y) > max_y_correction):
        print(f"Warning: Extreme correction detected ({error_x:.1f}, {error_y:.1f}), limiting correction magnitude")
        # Scale down the error to prevent overcorrection while preserving direction
        if abs(error_x) > max_x_correction:
            error_x = max_x_correction if error_x > 0 else -max_x_correction
        if abs(error_y) > max_y_correction:
            error_y = max_y_correction if error_y > 0 else -max_y_correction
    
    # Store the adaptation for this point
    print(f"Applied adaptation for point {calibration_counter}: ({error_x:.1f}, {error_y:.1f})")
    return (error_x, error_y)

# Add a variable to track frame times properly
last_frame_time = time.time()

# Enhance the non-linear correction function with more effective parameters
def apply_non_linear_correction(x, y, screen_width, screen_height):
    """Apply enhanced non-linear correction to the gaze coordinates"""
    # Normalize coordinates to 0-1 range
    norm_x = x / screen_width
    norm_y = y / screen_height
    
    # Calculate how much to adjust based on distance from screen center
    center_dist_x = norm_x - 0.5
    center_dist_y = norm_y - 0.5
    
    # Determine screen region for specialized correction
    left_side = center_dist_x < -0.05
    right_side = center_dist_x > 0.05
    top_side = center_dist_y < -0.05
    bottom_side = center_dist_y > 0.05
    
    # Calculate distance from center for directional correction
    center_dist = math.sqrt(center_dist_x**2 + center_dist_y**2)
    
    # Base correction factors that apply to all regions
    # These handle the natural curvature of the eye movement
    base_x_correction = 0
    base_y_correction = 0
    
    # Apply region-specific corrections
    if left_side:
        # Stronger non-linear correction on left side
        # Progressive adjustment that increases more dramatically for far-left points
        adjustment_factor = 1.5 + (abs(center_dist_x) * 3.0)
        
        # Add extra boost for far left side
        if center_dist_x < -0.3:
            adjustment_factor *= left_side_boost_factor * 1.2
            
        # Calculate x-correction with enhanced factors
        x_correction = (horizontal_warp_factor * 1.5) * adjustment_factor * (center_dist_x ** 2) * screen_width
        
        # Special vertical correction for left side - compensates for diagonal drift
        y_correction = center_dist_y * 0.25 * screen_height * (1.0 + abs(center_dist_x) * 2.0)
        
        # Apply corrections
        corrected_x = x + x_correction
        corrected_y = y + y_correction
        
    elif right_side:
        # Right side correction with different parameters
        adjustment_factor = 1.0 + (abs(center_dist_x) * 1.2)
        
        # Special handling for far right region
        if center_dist_x > 0.3:
            adjustment_factor *= 1.3
            
        # Calculate x-correction
        x_correction = horizontal_warp_factor * adjustment_factor * (center_dist_x ** 2) * screen_width
        
        # Subtle vertical correction for right side
        y_correction = center_dist_y * 0.1 * screen_height * (1.0 + abs(center_dist_x))
        
        # Apply corrections - note sign change for right side
        corrected_x = x - x_correction  # Note the negative for right side
        corrected_y = y + y_correction
    else:
        # Center region - minimal correction
        corrected_x = x
        corrected_y = y
    
    # Apply additional vertical correction based on vertical position
    if top_side:
        # Top of screen tends to need upward correction
        y_adjustment = 0.2 * abs(center_dist_y) * screen_height
        corrected_y -= y_adjustment * (1.0 + abs(center_dist_x))
    elif bottom_side:
        # Bottom of screen tends to need downward correction
        y_adjustment = 0.15 * abs(center_dist_y) * screen_height
        corrected_y += y_adjustment * (1.0 + abs(center_dist_x) * 0.5)
    
    # Apply base corrections that affect all regions
    corrected_x += base_x_correction
    corrected_y += base_y_correction
    
    # Add additional edge handling to prevent predictions from going off-screen
    edge_margin = 20  # pixels from edge of screen
    corrected_x = max(edge_margin, min(corrected_x, screen_width - edge_margin))
    corrected_y = max(edge_margin, min(corrected_y, screen_height - edge_margin))
    
    return corrected_x, corrected_y

# Improve the regional correction function with dynamic weighting
def apply_regional_correction(x, y, screen_width, screen_height, calibration_adaptations, calibration_points_list):
    """Apply enhanced regional correction with dynamic weighting based on position"""
    # Return early if corrections are disabled for testing
    global correction_enabled
    if not correction_enabled:
        return x, y
        
    # Find the closest calibration points and use weighted adaptations
    closest_points = []
    
    # Find distances to all calibration points
    for i in range(min(len(calibration_points_list), calibration_points)):
        point = calibration_points_list[i]
        dist = math.sqrt((point[0] - x)**2 + (point[1] - y)**2)
        closest_points.append((i, dist))
    
    # Sort by distance
    closest_points.sort(key=lambda x: x[1])
    
    # Take more points for interpolation when available for better coverage
    # Using more points helps smooth the interpolation but can over-blend corrections
    n_points = min(3, len(closest_points))  # Reduced from 4 to 3 for better specificity
    weighted_correction_x = 0
    weighted_correction_y = 0
    total_weight = 0
    
    # Apply weighted correction from nearest points with enhanced weighting
    for i in range(n_points):
        idx, dist = closest_points[i]
        
        # Skip if no adaptation for this point
        if idx not in calibration_adaptations:
            continue
        
        # Skip extremely distant points to prevent overcorrection
        if dist > max_correction_distance:
            continue
            
        # Use inverse cube weighting with sharper falloff for more localized correction
        # Higher exponent (3.0) makes the nearest point much more influential
        weight = 1.0 / (max(dist, 5) ** 3.0)  # Increased from 2.5 to 3.0
        
        # Apply higher weight to the closest point
        if i == 0:
            weight *= 2.0  # Give twice as much weight to the closest point
            
        error_x, error_y = calibration_adaptations[idx]
        
        # Get calibration point position
        cal_point = calibration_points_list[idx]
        cal_x = cal_point[0] / screen_width  # Normalize
        cal_y = cal_point[1] / screen_height  # Normalize
        
        # Determine if the correction target is in a similar region to the calibration point
        # This helps ensure corrections are more appropriate for the region
        target_x_norm = x / screen_width
        target_y_norm = y / screen_height
        
        # Calculate normalized distance to determine regional similarity
        region_similarity = 1.0 - min(1.0, math.sqrt((cal_x - target_x_norm)**2 + (cal_y - target_y_norm)**2) * 2.0)
        region_similarity = max(0.2, region_similarity)  # Keep at least 20% influence
        
        # Apply region-specific adjustments
        # Enhanced dynamic correction based on screen position
        if dynamic_correction_enabled:
            # Apply stronger correction factor for left-side calibration points
            if cal_x < left_weight_threshold:
                # Progressive factor based on how far left the calibration point is
                left_factor = left_side_correction_factor * (1 + (left_weight_threshold - cal_x) * 2.0)
                error_x *= left_factor
            
            # Apply position-based correction adjustment based on where the user is looking
            if target_x_norm < left_weight_threshold:
                # Calculate dynamic factor based on position (stronger for far left)
                position_factor = 1.0 + ((left_weight_threshold - target_x_norm) * 2.0)
                error_x *= position_factor
                
            # Vertical regions often need different correction amounts
            if target_y_norm < 0.2:  # Top region
                error_y *= 1.2  # Boost vertical correction for top of screen
            elif target_y_norm > 0.8:  # Bottom region
                error_y *= 1.2  # Boost vertical correction for bottom of screen
                
        # Apply region similarity to ensure corrections are appropriate
        error_x *= region_similarity
        error_y *= region_similarity
        
        weighted_correction_x += error_x * weight
        weighted_correction_y += error_y * weight
        total_weight += weight
    
    # Apply weighted average correction if we have data
    if total_weight > 0:
        correction_x = weighted_correction_x / total_weight
        correction_y = weighted_correction_y / total_weight
        
        # Limit maximum correction to prevent wild jumps
        max_correction = max_correction_distance
        correction_x = max(min(correction_x, max_correction), -max_correction)
        correction_y = max(min(correction_y, max_correction), -max_correction)
        
        # Adaptive correction factor based on confidence (influenced by number of points)
        # With fewer points, apply more conservative correction
        points_confidence = min(1.0, len(calibration_adaptations) / 9.0)
        adaptive_correction_factor = 0.5 + (0.4 * points_confidence)  # Ranges from 0.5 to 0.9
        
        # Apply progressive correction amount
        x += correction_x * adaptive_correction_factor
        y += correction_y * adaptive_correction_factor
    
    # Apply non-linear correction after adaptation for additional systematic correction
    if use_non_linear_correction:
        x, y = apply_non_linear_correction(x, y, screen_width, screen_height)
    
    return x, y

# Add a function to visualize the correction map effect
def draw_correction_map(screen, calibration_adaptations, calibration_points_list):
    """Draw a visualization of how the correction map affects different screen regions"""
    # Create a grid of test points
    grid_spacing = 50
    grid_points_x = range(grid_spacing, screen_width, grid_spacing)
    grid_points_y = range(grid_spacing, screen_height, grid_spacing)
    
    # Draw semi-transparent overlay
    overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))  # Semi-transparent black
    
    # Draw grid lines
    for x in grid_points_x:
        pygame.draw.line(overlay, (50, 50, 50, 128), (x, 0), (x, screen_height), 1)
    for y in grid_points_y:
        pygame.draw.line(overlay, (50, 50, 50, 128), (0, y), (screen_width, y), 1)
    
    # Draw calibration points
    for i in range(min(len(calibration_points_list), calibration_points)):
        point = calibration_points_list[i]
        # Draw circle around calibration point
        pygame.draw.circle(overlay, (0, 255, 0, 200), point, 15, 2)
        # Number the point
        point_label = small_font.render(str(i+1), True, (255, 255, 255, 200))
        overlay.blit(point_label, (point[0] - 5, point[1] - 8))
    
    # Draw correction vectors at grid points
    scale_factor = 3.0  # Scale up the vectors to make them visible
    for x in grid_points_x:
        for y in grid_points_y:
            # Skip points that would be off screen after correction
            if x < 10 or y < 10 or x > screen_width - 10 or y > screen_height - 10:
                continue
                
            # Apply the regional correction
            corrected_x, corrected_y = apply_regional_correction(
                x, y, screen_width, screen_height, 
                calibration_adaptations, calibration_points_list
            )
            
            # Draw arrow showing correction
            dx = corrected_x - x
            dy = corrected_y - y
            
            # Only draw if correction is significant
            if abs(dx) > 1 or abs(dy) > 1:
                # Scale the arrow for visibility
                arrow_end_x = x + dx * scale_factor
                arrow_end_y = y + dy * scale_factor
                
                # Color based on magnitude (red = large correction, green = small)
                magnitude = math.sqrt(dx*dx + dy*dy)
                max_mag = 50.0  # Expected maximum correction magnitude
                color_intensity = min(1.0, magnitude / max_mag)
                arrow_color = (
                    int(255 * color_intensity),  # Red
                    int(255 * (1 - color_intensity)),  # Green
                    0,  # Blue
                    200  # Alpha
                )
                
                # Draw small circle at grid point
                pygame.draw.circle(overlay, (100, 100, 100, 128), (x, y), 2)
                
                # Draw arrow line
                pygame.draw.line(overlay, arrow_color, (x, y), (arrow_end_x, arrow_end_y), 1)
                
                # Draw arrowhead
                arrow_size = 4
                angle = math.atan2(arrow_end_y - y, arrow_end_x - x)
                pygame.draw.polygon(overlay, arrow_color, [
                    (arrow_end_x, arrow_end_y),
                    (arrow_end_x - arrow_size * math.cos(angle - math.pi/6), 
                     arrow_end_y - arrow_size * math.sin(angle - math.pi/6)),
                    (arrow_end_x - arrow_size * math.cos(angle + math.pi/6), 
                     arrow_end_y - arrow_size * math.sin(angle + math.pi/6)),
                ])
    
    # Draw a legend
    legend_x = 10
    legend_y = screen_height - 180
    pygame.draw.rect(overlay, (0, 0, 0, 200), (legend_x, legend_y, 200, 100))
    
    # Legend title
    title_text = small_font.render("Correction Map", True, (255, 255, 255))
    overlay.blit(title_text, (legend_x + 10, legend_y + 10))
    
    # Draw example arrows for small and large corrections
    pygame.draw.line(overlay, (0, 255, 0, 200), (legend_x + 20, legend_y + 40), (legend_x + 40, legend_y + 40), 2)
    small_text = small_font.render("Small correction", True, (255, 255, 255))
    overlay.blit(small_text, (legend_x + 50, legend_y + 35))
    
    pygame.draw.line(overlay, (255, 0, 0, 200), (legend_x + 20, legend_y + 70), (legend_x + 40, legend_y + 70), 2)
    large_text = small_font.render("Large correction", True, (255, 255, 255))
    overlay.blit(large_text, (legend_x + 50, legend_y + 65))
    
    # Blit the overlay to the screen
    screen.blit(overlay, (0, 0))

# Add a variable to toggle correction map visualization
show_correction_map = False

# Add variable to toggle correction for testing
correction_enabled = True  # Can be toggled with 'C' key

# Add variable to toggle between showing raw or corrected points
show_raw_points = True  # Toggle with 'R' key to show/hide raw points

# Variables to control calibration point behavior
force_stable_calibration_points = True  # Make points stay in place during calibration
force_sequential_calibration = True  # Force calibration to proceed in sequence

# Main loop
while running:
    # Calculate delta time correctly
    current_time = time.time()
    dt = current_time - last_frame_time
    last_frame_time = current_time
    
    # Limit dt to avoid huge jumps if the program was paused/suspended
    dt = min(dt, 0.1)  # Cap at 100ms to prevent large jumps
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                running = False
            elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                # Reset to detection phase when Ctrl+R is pressed
                app_state = "detection_check"
                detection_success_count = 0
                detection_attempts = 0
                detection_successes = 0
                calibration_counter = 0
                gaze_history = []
                print("Reset to detection phase")
            elif event.key == pygame.K_r and app_state == "tracking":
                # Toggle raw points visualization with just 'R' in tracking mode
                show_raw_points = not show_raw_points
                print(f"Raw points visualization {'enabled' if show_raw_points else 'disabled'}")
            elif event.key == pygame.K_s and app_state == "detection_check":
                # Skip detection check if user presses 'S'
                app_state = "calibration"
            elif event.key == pygame.K_SPACE and app_state == "detection_check":
                # Proceed to calibration if spacebar is pressed
                app_state = "calibration"
            elif event.key == pygame.K_d:
                # Toggle debug visualization
                debug_landmarks_enabled = not debug_landmarks_enabled
            elif event.key == pygame.K_m:
                # Toggle direct MediaPipe detection
                use_direct_mediapipe = not use_direct_mediapipe
                if use_direct_mediapipe:
                    print("Direct MediaPipe detection enabled")
                else:
                    print("Direct MediaPipe detection disabled")
            elif event.key == pygame.K_c:
                # Toggle correction to help debug
                correction_enabled = not correction_enabled
                print(f"Correction {'enabled' if correction_enabled else 'disabled'}")
            elif event.key == pygame.K_v:
                # Toggle correction map visualization
                show_correction_map = not show_correction_map
                print(f"Correction map visualization {'enabled' if show_correction_map else 'disabled'}")
    
    # Capture frame from camera
    ret, frame = cap.read()
    if frame is None or not ret:
        continue  # Skip this iteration if frame is not available
    
    # Check for system warmup period
    if is_warming_up:
        elapsed_warmup_time = time.time() - warmup_start_time
        if elapsed_warmup_time < system_warmup_time:
            # Clear screen and show warmup message
            screen.fill(BLACK)
            warmup_text = large_font.render(f"System initializing... {int(system_warmup_time - elapsed_warmup_time) + 1}s", True, WHITE)
            warmup_rect = warmup_text.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(warmup_text, warmup_rect)
            
            # Additional instruction
            instruction_text = font.render("Please look at the camera and keep your face centered", True, YELLOW)
            instruction_rect = instruction_text.get_rect(center=(screen_width // 2, screen_height // 2 + 60))
            screen.blit(instruction_text, instruction_rect)
            
            # Update display and continue
            pygame.display.flip()
            clock.tick(60)
            continue
        else:
            # Warmup complete, start detection phase
            is_warming_up = False
            print("System warmup complete, starting detection phase")
            
    # Direct face detection with MediaPipe (for visualization)
    if debug_landmarks_enabled and use_direct_mediapipe:
        process_face_with_mediapipe(frame)
    
    # Convert frame to RGB for pygame display
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Clear screen
    screen.fill(BLACK)
    
    # Display the camera frame
    frame_surf = pygame.surfarray.make_surface(np.rot90(display_frame))
    frame_surf = pygame.transform.scale(frame_surf, (640, 480))
    frame_rect = frame_surf.get_rect(center=(screen_width//2, screen_height//2 - 100))
    screen.blit(frame_surf, frame_rect)
    
    # Different processing based on current app state
    if app_state == "detection_check":
        # Attempt face detection
        try:
            detection_attempts += 1
            # Just check if face detection works, don't start calibration yet
            event, calibration = gestures.step(
                frame, False, screen_width, screen_height, context="main"
            )
            detection_successes += 1
            detection_success_count += 1
            face_detection_failure = False
            last_detection_time = time.time()
            
            # Extract landmarks for debugging visualization
            get_landmarks_from_gesture_event(event)
            
        except (TypeError, IndexError) as e:
            # Face detection likely failed
            print(f"Detection check - Face detection error: {e}")
            face_detection_failure = True
        
        # Calculate success rate
        success_rate = detection_successes / detection_attempts if detection_attempts > 0 else 0
        
        if face_detection_failure:
            draw_face_detection_warning(screen)
        else:
            # Draw detected face and eye landmarks on the camera feed
            draw_face_and_eye_landmarks(screen, frame_rect)
        
        # Draw detection phase UI
        draw_detection_phase(screen, detection_success_count, detection_required_successes, success_rate)
        
        # Show controls
        controls_text = small_font.render("Press 'SPACE' to continue | 'S' to skip | 'D' to toggle debug | 'M' to toggle MediaPipe | 'Ctrl+Q' to quit", True, WHITE)
        screen.blit(controls_text, (10, screen_height - 30))
        
        # If we have enough successful detections, highlight the spacebar option
        if detection_success_count >= detection_required_successes:
            proceed_text = font.render("Face detection successful! Press SPACE to continue", True, GREEN)
            proceed_rect = proceed_text.get_rect(center=(screen_width//2, screen_height - 70))
            screen.blit(proceed_text, proceed_rect)
            
    elif app_state == "calibration" or app_state == "tracking":
        # Check if we're still calibrating
        is_calibrating = app_state == "calibration" and (calibration_counter < calibration_points)
        
        if not is_calibrating and app_state == "calibration":
            app_state = "tracking"  # Move to tracking state after calibration
        
        # Process frame with eye tracking (with error handling)
        try:
            # Check if eyes are open enough for calibration
            eyes_open_enough = are_eyes_open_enough_for_calibration()
            
            # Only proceed with actual calibration if eyes are open enough
            event, calibration = gestures.step(
                frame, 
                is_calibrating and eyes_open_enough and not is_adapting_calibration,  # Only calibrate if eyes are open enough and not adapting
                screen_width, 
                screen_height, 
                context="main"
            )
            face_detection_failure = False
            last_detection_time = time.time()
            consecutive_detection_failures = 0  # Reset counter on successful detection
            
            # Extract landmarks for debugging visualization
            get_landmarks_from_gesture_event(event)
            
        except (TypeError, IndexError) as e:
            # Face detection likely failed
            print(f"Face detection error: {e}")
            event = None
            calibration = None
            face_detection_failure = True
            consecutive_detection_failures += 1  # Increment counter on failure
            
            # Only consider it a true failure if we've had multiple consecutive failures
            # or we're past the grace period
            if consecutive_detection_failures <= max_detection_failures and time.time() - last_detection_time < face_detection_grace_period:
                # Temporarily use the last known event/calibration data if available
                # This prevents calibration from resetting during brief detection blips
                face_detection_failure = False
                print(f"Brief detection failure ({consecutive_detection_failures}), using last known data")
        
        # Handle face detection failures
        if face_detection_failure:
            draw_face_detection_warning(screen)
            
            # Show how long detection has been failing
            failure_duration = time.time() - last_detection_time
            failure_text = small_font.render(f"Face detection failed for {failure_duration:.1f}s", True, YELLOW)
            screen.blit(failure_text, (screen_width // 2 - 100, screen_height // 2 + 50))
            
            # If face detection has been failing for a while during calibration,
            # don't advance the calibration counter but don't reset it either
            if failure_duration > 5.0 and app_state == "calibration":
                # Just pause the calibration, don't reset
                print(f"Pausing calibration due to detection failure: {failure_duration:.1f}s")
        # Handle calibration or tracking when face is detected
        elif event is not None or calibration is not None:
            # Draw face and eye landmarks for debugging
            draw_face_and_eye_landmarks(screen, frame_rect)
            
            if is_calibrating:
                if not is_adapting_calibration:
                    if calibration:
                        # If we have a new calibration point or need to reset
                        is_new_point = False
                        
                        # Check if this is the first frame or we need a reset
                        if current_point_needs_reset or current_calibration_point is None:
                            is_new_point = True
                            print("New point detected due to reset flag or initialization")
                        # Check if point has moved significantly (not just minor jitter)
                        elif (abs(calibration.point[0] - prev_x) > calibration_point_stability_threshold or 
                              abs(calibration.point[1] - prev_y) > calibration_point_stability_threshold):
                            # Only allow point changes when force_stable_calibration_points is off
                            if force_stable_calibration_points:
                                # Ignore the movement and keep using the previous point
                                is_new_point = False
                                calibration.point = (prev_x, prev_y)
                                print(f"Forcing stable point, ignoring movement attempt")
                            else:
                                is_new_point = True
                                print(f"Point moved significantly: ({prev_x}, {prev_y}) -> ({calibration.point[0]}, {calibration.point[1]})")
                        
                        if is_new_point:
                            # If we're forcing sequential calibration, always use the next point in sequence
                            if force_sequential_calibration:
                                # Ensure we're using the correct calibration point from our list
                                if calibration_counter < len(calibration_points_list):
                                    next_point = calibration_points_list[calibration_counter]
                                    calibration.point = next_point
                                    print(f"Setting calibration point {calibration_counter+1} at ({next_point[0]}, {next_point[1]})")
                            current_calibration_point = calibration.point
                            prev_x = calibration.point[0]
                            prev_y = calibration.point[1]
                            current_calibration_time = 0.0
                            calibration_countdown = calibration_point_duration
                            gaze_points_for_current_target = []
                            current_point_needs_reset = False
                            is_first_calibration_frame = True
                            # Set absolute start time for the current point
                            calibration_point_start_time = time.time()
                            print(f"New calibration point {calibration_counter+1} started, timer reset")
                        # Ensures we don't continuously reset due to small jitter in the calibration point position
                        else:
                            # Only update position without resetting timer (to handle slight drift)
                            # Average the position to prevent small jitters from affecting calibration
                            prev_x = prev_x * 0.8 + calibration.point[0] * 0.2
                            prev_y = prev_y * 0.8 + calibration.point[1] * 0.2
                        
                        # Calculate elapsed time since point started using absolute timestamps for reliability
                        if calibration_point_start_time is not None:
                            current_calibration_time = time.time() - calibration_point_start_time
                            # Ensure countdown doesn't go below zero
                            calibration_countdown = max(0.0, calibration_point_duration - current_calibration_time)
                        else:
                            # If for some reason we don't have a start time, set one now
                            calibration_point_start_time = time.time()
                            current_calibration_time = 0.0
                            calibration_countdown = calibration_point_duration
                        
                        # Collect gaze data for adaptive calibration only when eyes are open enough
                        if event and adaptive_calibration_enabled and eyes_open_enough:
                            gaze_points_for_current_target.append(event.point)
                        
                        # Improved logging on significant time changes (every second)
                        if int(current_calibration_time) != int(current_calibration_time - dt) or is_first_calibration_frame:
                            is_first_calibration_frame = False
                            print(f"Point {calibration_counter+1}: Time={current_calibration_time:.2f}s, "
                                  f"Countdown={calibration_countdown:.2f}s, "
                                  f"Collected points: {len(gaze_points_for_current_target)}")
                        
                        # Check if calibration point timeout occurred (ensure we don't get stuck)
                        if current_calibration_time >= calibration_point_timeout:
                            print(f"Point {calibration_counter+1} timed out after {calibration_point_timeout}s, forcing advance")
                            current_calibration_time = calibration_point_duration  # Force completion
                        
                        # Draw calibration points (current and remaining)
                        point_radius = min(30, getattr(calibration, 'acceptance_radius', 30))
                        
                        # Show the current gaze prediction (where user is looking) compared to target
                        if event and len(gaze_points_for_current_target) > 0:
                            # Calculate smoothed gaze prediction (last few points)
                            recent_points = gaze_points_for_current_target[-5:] if len(gaze_points_for_current_target) > 5 else gaze_points_for_current_target
                            avg_x = sum(p[0] for p in recent_points) / len(recent_points)
                            avg_y = sum(p[1] for p in recent_points) / len(recent_points)
                            current_gaze_prediction = (int(avg_x), int(avg_y))
                            
                            # Draw the predicted gaze point (red)
                            pygame.draw.circle(screen, RED, current_gaze_prediction, 10)
                            pygame.draw.circle(screen, WHITE, current_gaze_prediction, 3)
                            
                            # Draw a line showing the error between prediction and target
                            pygame.draw.line(screen, YELLOW, calibration.point, current_gaze_prediction, 2)
                            
                            # Calculate and display error distance
                            error_x = calibration.point[0] - current_gaze_prediction[0]
                            error_y = calibration.point[1] - current_gaze_prediction[1]
                            error_distance = math.sqrt(error_x**2 + error_y**2)
                            error_text = small_font.render(f"Error: {int(error_distance)}px | Points: {len(gaze_points_for_current_target)}", True, WHITE)
                            error_text_rect = error_text.get_rect(midtop=(
                                (calibration.point[0] + current_gaze_prediction[0]) // 2,
                                (calibration.point[1] + current_gaze_prediction[1]) // 2 + 10
                            ))
                            screen.blit(error_text, error_text_rect)
                        
                        # Draw all calibration points with progress indicator for current one
                        drawn_points = 0
                        for i in range(min(len(calibration_points_list), calibration_points)):
                            if i < calibration_counter:
                                # Already calibrated points - show as small markers
                                point = calibration_points_list[i]
                                # Show in green if successful, red if failed
                                is_successful = i in successful_calibration_points
                                draw_calibration_point(screen, point, point_radius, 0, False, is_successful)
                                drawn_points += 1
                            elif i == calibration_counter:
                                # Current calibration point - draw with progress
                                progress = min(1.0, current_calibration_time / calibration_point_duration)
                                draw_calibration_point(screen, calibration.point, point_radius, progress, True)
                                drawn_points += 1
                            else:
                                # Upcoming points - draw as inactive
                                point = calibration_points_list[i]
                                draw_calibration_point(screen, point, point_radius, 0, False)
                                drawn_points += 1
                        
                        # Show calibration progress
                        progress_text = large_font.render(f"Point {calibration_counter + 1}/{calibration_points}", True, WHITE)
                        screen.blit(progress_text, (20, 20))
                        
                        # Show time remaining more prominently
                        time_text = large_font.render(f"Time: {int(calibration_countdown)}s", True, WHITE)
                        screen.blit(time_text, (20, 70))
                        
                        # Display debug information about gaze points collected
                        debug_text = small_font.render(
                            f"Gaze points: {len(gaze_points_for_current_target)}/{calibration_min_points_required} | "
                            f"Eyes open: {'Yes' if eyes_open_enough else 'No'}", 
                            True, WHITE)
                        screen.blit(debug_text, (20, 120))
                        
                        # Display instructions
                        instructions = font.render("Look at the blue circle and hold steady", True, WHITE)
                        screen.blit(instructions, (10, screen_height - 40))
                        
                        # Show warning if eyes are not open enough
                        if not eyes_open_enough:
                            warning_text = font.render("Open your eyes wider for calibration to continue", True, YELLOW)
                            warning_rect = warning_text.get_rect(center=(screen_width//2, screen_height - 70))
                            screen.blit(warning_text, warning_rect)
                        
                        # Check if time to advance to next calibration point
                        if current_calibration_time >= calibration_point_duration:
                            print(f"Point {calibration_counter+1} completed after {current_calibration_time:.2f}s, moving to next point")
                            print(f"Final points collected: {len(gaze_points_for_current_target)}")
                            
                            # Calculate and store adaptation
                            if len(gaze_points_for_current_target) >= calibration_min_points_required:
                                # Filter gaze points by fixation quality first
                                quality_points = []
                                stable_period_start = None
                                stable_period_duration = 0.0
                                
                                # Process points to find stable periods with good fixation
                                for i, p in enumerate(gaze_points_for_current_target):
                                    # Only use points where the user was fixating well
                                    if hasattr(event, 'fixation') and event.fixation > calibration_min_fixation_threshold:
                                        quality_points.append(p)
                                        
                                        # Check if this is part of a stable sequence
                                        if stable_period_start is None:
                                            stable_period_start = i
                                        stable_period_duration += dt  # Approximate duration based on frame time
                                    else:
                                        # Reset stable period tracking if fixation quality drops
                                        stable_period_start = None
                                        stable_period_duration = 0.0
                                
                                print(f"Quality points: {len(quality_points)}/{len(gaze_points_for_current_target)} with stable period of {stable_period_duration:.2f}s")
                                
                                # Use quality points when available, otherwise fall back to all points
                                working_points = quality_points if len(quality_points) >= calibration_min_points_required else gaze_points_for_current_target
                                
                                # Use last 2 seconds of data for most stable results
                                last_samples = working_points[-min(30, len(working_points)):]
                                
                                # More sophisticated filtering - remove outliers
                                if len(last_samples) > 15:
                                    # Calculate average position
                                    avg_x = sum(p[0] for p in last_samples) / len(last_samples)
                                    avg_y = sum(p[1] for p in last_samples) / len(last_samples)
                                    
                                    # Calculate standard deviation
                                    std_x = math.sqrt(sum((p[0] - avg_x)**2 for p in last_samples) / len(last_samples))
                                    std_y = math.sqrt(sum((p[1] - avg_y)**2 for p in last_samples) / len(last_samples))
                                    
                                    # Filter out points that are more than 2 standard deviations away
                                    filtered_samples = [p for p in last_samples if 
                                                      abs(p[0] - avg_x) < 2 * std_x and 
                                                      abs(p[1] - avg_y) < 2 * std_y]
                                    
                                    # Recalculate average if we have enough points after filtering
                                    if len(filtered_samples) > 10:
                                        avg_x = sum(p[0] for p in filtered_samples) / len(filtered_samples)
                                        avg_y = sum(p[1] for p in filtered_samples) / len(filtered_samples)
                                        print(f"Filtered {len(last_samples) - len(filtered_samples)} outlier points")
                                    
                                    # Log the standard deviation (jitter)
                                    print(f"Gaze jitter: x={std_x:.1f}px, y={std_y:.1f}px")
                                else:
                                    # Just use simple average for small sample
                                    avg_x = sum(p[0] for p in last_samples) / len(last_samples)
                                    avg_y = sum(p[1] for p in last_samples) / len(last_samples)
                                
                                # Calculate error
                                error_x = current_calibration_point[0] - avg_x
                                error_y = current_calibration_point[1] - avg_y
                                
                                # Calculate error magnitude for logging
                                error_magnitude = math.sqrt(error_x**2 + error_y**2)
                                print(f"Raw error magnitude: {error_magnitude:.1f}px")
                                
                                # Dynamically adjust correction for left side
                                point_x_normalized = current_calibration_point[0] / screen_width
                                if point_x_normalized < left_weight_threshold:
                                    # Progressive factor based on how far left
                                    left_factor = left_side_correction_factor * (1 + (left_weight_threshold - point_x_normalized))
                                    error_x *= left_factor
                                    print(f"Applied left side boost factor: {left_factor:.2f}")
                                
                                # Store adaptation with validation - don't apply extreme corrections
                                if abs(error_x) > 300 or abs(error_y) > 300:
                                    print(f"Warning: Extreme correction detected ({error_x:.1f}, {error_y:.1f}), limiting correction magnitude")
                                    # Limit extreme corrections that are likely errors
                                    error_x = max(min(error_x, 300), -300)
                                    error_y = max(min(error_y, 300), -300)
                                
                                calibration_adaptations[calibration_counter] = (error_x, error_y)
                                print(f"Applied adaptation for point {calibration_counter+1}: ({error_x:.1f}, {error_y:.1f})")
                                
                                # Mark this point as successfully calibrated
                                successful_calibration_points.add(calibration_counter)
                            else:
                                print(f"WARNING: Not enough gaze data for point {calibration_counter+1} ({len(gaze_points_for_current_target)}/{calibration_min_points_required}), no adaptation applied")
                            
                            # Move to next point
                            calibration_counter += 1
                            current_calibration_time = 0.0
                            calibration_point_start_time = None  # Reset the start time for next point
                            current_point_needs_reset = True
                            gaze_points_for_current_target = []
                            
                            # Force small delay between points to ensure reset
                            time.sleep(0.5)
                            
                            # Check if calibration is complete
                            if calibration_counter >= calibration_points:
                                print("Calibration complete! Moving to tracking phase.")
                                print(f"Successfully calibrated {len(successful_calibration_points)}/{calibration_points} points")
                                # Always proceed to tracking mode
                                app_state = "tracking"
                else:
                    # We're in adaptation mode - finish adaptation and move to next point
                    if event:
                        # We have gaze data, apply adaptation and move to next point
                        is_adapting_calibration = False
                        calibration_counter += 1
                        # The adaptation is already stored, we just needed a frame to process
            else:
                if event:
                    # Store gaze point in history (for smoothing)
                    gaze_history.append(event.point)
                    if len(gaze_history) > 5:  # Keep only recent points
                        gaze_history.pop(0)
                    
                    # Calculate smoothed gaze point (average of recent points)
                    if gaze_history:
                        avg_x = sum(p[0] for p in gaze_history) / len(gaze_history)
                        avg_y = sum(p[1] for p in gaze_history) / len(gaze_history)
                        
                        # Apply regional and non-linear correction
                        avg_x, avg_y = apply_regional_correction(
                            avg_x, avg_y, 
                            screen_width, screen_height,
                            calibration_adaptations, 
                            calibration_points_list
                        )
                        
                        # Convert to int for drawing
                        smoothed_point = (int(avg_x), int(avg_y))
                        
                        # Calculate raw gaze point (uncorrected) for visualization
                        raw_x = sum(p[0] for p in gaze_history) / len(gaze_history)
                        raw_y = sum(p[1] for p in gaze_history) / len(gaze_history)
                        raw_point = (int(raw_x), int(raw_y))
                        
                        # Draw the gaze point with both raw and corrected positions
                        # Only show raw point if the toggle is enabled
                        raw_point_for_display = raw_point if show_raw_points else None
                        draw_gaze_point(screen, smoothed_point, event.fixation, raw_point_for_display)
                        
                        # Display correction map if enabled
                        if show_correction_map and len(calibration_adaptations) > 0:
                            draw_correction_map(screen, calibration_adaptations, calibration_points_list)
                        
                        # Display enhanced status with correction magnitude
                        correction_magnitude = int(math.sqrt((raw_x - avg_x)**2 + (raw_y - avg_y)**2))
                        status_text = font.render(
                            f"Gaze: ({int(avg_x)}, {int(avg_y)})  Fixation: {event.fixation} | "
                            f"Correction: {'On' if correction_enabled else 'Off'} ({correction_magnitude}px)", 
                            True, WHITE
                        )
                        screen.blit(status_text, (10, screen_height - 40))
                        
                        # Show which calibration point influenced the correction most
                        # Find closest calibration points
                        closest_points = []
                        for i in range(min(len(calibration_points_list), calibration_points)):
                            point = calibration_points_list[i]
                            dist = math.sqrt((point[0] - avg_x)**2 + (point[1] - avg_y)**2)
                            closest_points.append((i, dist))
                        
                        # Sort by distance
                        if closest_points:
                            closest_points.sort(key=lambda x: x[1])
                        
                        closest_idx = closest_points[0][0] if len(closest_points) > 0 else -1
                        if closest_idx in calibration_adaptations:
                            adapt_text = small_font.render(f"Using regions with strongest influence from point {closest_idx+1}", True, CYAN)
                            screen.blit(adapt_text, (10, screen_height - 70))
        
        # Instructions
        if not is_calibrating and app_state == "tracking":
            controls_text = font.render("Ctrl+R: Restart | R: Toggle raw points | D: Toggle debug | C: Toggle correction | V: Correction map | Ctrl+Q: Quit", True, WHITE)
            screen.blit(controls_text, (10, 10))
        elif is_calibrating:
            debug_toggle_text = small_font.render("D: Toggle debug | M: Toggle MediaPipe | C: Toggle correction | Ctrl+Q: Quit", True, WHITE)
            screen.blit(debug_toggle_text, (10, 10))
    
    # Update display
    pygame.display.flip()
    
    # Cap frame rate
    clock.tick(60)

# Cleanup
try:
    cap.release()  # Try to release, but might not be needed
except AttributeError:
    pass  # Ignore if the method doesn't exist

pygame.quit()
sys.exit() 