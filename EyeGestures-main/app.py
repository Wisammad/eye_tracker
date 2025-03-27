import os
import sys
import numpy as np
import pandas as pd
import json
import time
import pygame
from pygame.locals import *
from PIL import Image
import random

# Add EyeGestures to path
script_dir = os.path.dirname(os.path.abspath(__file__))
eye_gestures_dir = os.path.join(script_dir, 'EyeGestures-main')
sys.path.append(eye_gestures_dir)

# Try to import eyeGestures, with fallback messaging
try:
    # Now import from the local EyeGestures directory
    from eyeGestures import EyeGestures_v3
    import cv2
    eye_tracking_available = True
except ImportError as e:
    print(f"Error importing eye tracking libraries: {e}")
    print("The application will run in demo mode without eye tracking.")
    eye_tracking_available = False

class AdEyeTracker:
    """
    Application to track eye movements while viewing advertisements and
    analyze where users are looking on each advertisement.
    """
    
    def __init__(self):
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Setup display
        self.info = pygame.display.Info()
        self.screen_width = self.info.current_w
        self.screen_height = self.info.current_h
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ad Eye Tracker")
        
        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Setup fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize eye tracker if available
        self.eye_tracking_available = eye_tracking_available
        if self.eye_tracking_available:
            try:
                self.eye_tracker = EyeGestures_v3()
                self.cap = cv2.VideoCapture(0)
                
                # Setup eye tracker context
                self.context = "ad_tracker"
                self.eye_tracker.addContext(self.context)
                self.eye_tracking_initialized = True
            except Exception as e:
                print(f"Error initializing eye tracker: {e}")
                self.eye_tracking_available = False
                self.eye_tracking_initialized = False
        else:
            self.eye_tracking_initialized = False
            
        # Calibration setup
        self.calibration_complete = False
        self.calibration_points = self.create_calibration_points()
        
        # Ad tracking state
        self.categories_dir = "categories"
        self.available_categories = self.get_available_categories()
        self.current_category = None
        self.ad_images = []
        self.ad_classes = []
        self.current_ad_index = 0
        self.gaze_data = []
        self.current_gaze_sequence = []
        self.display_active = False
        self.ad_display_time = 5000  # 5 seconds per ad
        self.view_start_time = 0
        
        # Application state
        self.running = True
        self.current_screen = "main_menu"  # main_menu, calibration, category_select, viewing_ads
        self.clock = pygame.time.Clock()
        
        # Demo mode - simulated gaze point for when eye tracking is unavailable
        self.demo_gaze_x = self.screen_width // 2
        self.demo_gaze_y = self.screen_height // 2
        self.move_direction_x = 1
        self.move_direction_y = 1
    
    def create_calibration_points(self):
        """Create a grid of calibration points for the eye tracker."""
        x = np.arange(0.1, 1.0, 0.2)
        y = np.arange(0.1, 1.0, 0.2)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        np.random.shuffle(points)
        return points
        
    def get_available_categories(self):
        """Get list of available ad categories from directory."""
        if not os.path.exists(self.categories_dir):
            os.makedirs(self.categories_dir)
            return []
        
        return [d for d in os.listdir(self.categories_dir) 
                if os.path.isdir(os.path.join(self.categories_dir, d))]
    
    def run(self):
        """Main application loop."""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    self.handle_key_press(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(pygame.mouse.get_pos())
            
            # Get camera frame for eye tracking
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Clear screen
            self.screen.fill(self.BLACK)
            
            # Render current screen
            if self.current_screen == "main_menu":
                self.render_main_menu()
            elif self.current_screen == "calibration":
                self.run_calibration(frame)
            elif self.current_screen == "category_select":
                self.render_category_select()
            elif self.current_screen == "viewing_ads":
                self.run_ad_viewing(frame)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        self.cap.release()
        pygame.quit()
    
    def render_main_menu(self):
        """Render the main menu screen."""
        # Title
        title = self.font_large.render("Ad Eye Tracker", True, self.WHITE)
        title_rect = title.get_rect(center=(self.screen_width/2, 100))
        self.screen.blit(title, title_rect)
        
        # Buttons
        calibrate_btn = self.create_button("Start Calibration", 
                                         self.screen_width/2, 250, 
                                         self.GREEN if not self.calibration_complete else self.WHITE)
        
        category_btn = self.create_button("Select Ad Category", 
                                         self.screen_width/2, 350, 
                                         self.WHITE if self.calibration_complete else self.RED)
        
        quit_btn = self.create_button("Quit", 
                                     self.screen_width/2, 450, 
                                     self.WHITE)
        
        # Status
        if not self.calibration_complete:
            status = self.font_small.render("Please calibrate eye tracker first", True, self.RED)
            status_rect = status.get_rect(center=(self.screen_width/2, 500))
            self.screen.blit(status, status_rect)
    
    def create_button(self, text, x, y, color, width=300, height=50):
        """Create a clickable button and return its rect."""
        text_surf = self.font_medium.render(text, True, color)
        text_rect = text_surf.get_rect(center=(x, y))
        
        # Draw button background
        button_rect = pygame.Rect(x - width/2, y - height/2, width, height)
        pygame.draw.rect(self.screen, self.BLACK, button_rect)
        pygame.draw.rect(self.screen, color, button_rect, 2)
        
        # Draw text
        self.screen.blit(text_surf, text_rect)
        
        return button_rect
    
    def handle_mouse_click(self, pos):
        """Handle mouse clicks based on current screen."""
        if self.current_screen == "main_menu":
            # Check if calibration button clicked
            calibrate_btn = pygame.Rect(self.screen_width/2 - 150, 250 - 25, 300, 50)
            if calibrate_btn.collidepoint(pos):
                self.start_calibration()
            
            # Check if category button clicked
            category_btn = pygame.Rect(self.screen_width/2 - 150, 350 - 25, 300, 50)
            if category_btn.collidepoint(pos) and self.calibration_complete:
                self.current_screen = "category_select"
            
            # Check if quit button clicked
            quit_btn = pygame.Rect(self.screen_width/2 - 150, 450 - 25, 300, 50)
            if quit_btn.collidepoint(pos):
                self.running = False
        
        elif self.current_screen == "category_select":
            # Check if back button clicked
            back_btn = pygame.Rect(100, self.screen_height - 50, 100, 40)
            if back_btn.collidepoint(pos):
                self.current_screen = "main_menu"
            
            # Check if a category was clicked
            y_pos = 200
            for category in self.available_categories:
                category_btn = pygame.Rect(self.screen_width/2 - 150, y_pos - 25, 300, 50)
                if category_btn.collidepoint(pos):
                    self.select_category(category)
                y_pos += 80
    
    def handle_key_press(self, key):
        """Handle keyboard input."""
        if key == pygame.K_SPACE and self.current_screen == "calibration":
            # Skip to next calibration point
            self.eye_tracker.clb[self.context].movePoint()
    
    def start_calibration(self):
        """Start eye tracker calibration."""
        self.current_screen = "calibration"
        self.calibration_complete = False
        
        # Setup calibration
        self.eye_tracker.clb[self.context] = EyeGestures_v3.Calibrator_v2()
        self.eye_tracker.uploadCalibrationMap(self.calibration_points, context=self.context)
    
    def run_calibration(self, frame):
        """Run eye tracker calibration with current frame."""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        if self.eye_tracking_available and self.eye_tracking_initialized:
            # Process frame with eye tracker
            try:
                point, key_points, blink, fixation, cevent = self.eye_tracker.process(
                    frame, context=self.context
                )
                
                # Get current calibration point
                cal_point = self.eye_tracker.clb[self.context].getCurrentPoint(
                    self.screen_width, self.screen_height
                )
                
                # Draw calibration point
                pygame.draw.circle(self.screen, self.RED, 
                                  (int(cal_point[0]), int(cal_point[1])), 15)
                
                # Draw progress
                progress = self.font_small.render(
                    f"Calibration progress: {self.eye_tracker.clb[self.context].iterator + 1}/{len(self.calibration_points)}", 
                    True, self.WHITE)
                self.screen.blit(progress, (50, 100))
                
                # Check if calibration is complete
                if self.eye_tracker.clb[self.context].iterator >= len(self.calibration_points) - 1:
                    self.finish_calibration()
            except Exception as e:
                error_msg = self.font_medium.render(
                    f"Error during calibration: {str(e)}", True, self.RED)
                self.screen.blit(error_msg, (50, 150))
                self.screen.blit(self.font_medium.render(
                    "Press SPACE to continue without eye tracking", True, self.WHITE), (50, 200))
        else:
            # Draw message about demo mode
            demo_msg = self.font_medium.render(
                "Eye tracking not available. Running in demo mode.", True, self.RED)
            self.screen.blit(demo_msg, (50, 150))
            self.screen.blit(self.font_medium.render(
                "Press SPACE to continue", True, self.WHITE), (50, 200))
        
        # Draw instructions
        instructions = self.font_medium.render(
            "Look at the red dot. Press SPACE to skip to next point.", 
            True, self.WHITE)
        self.screen.blit(instructions, (50, 50))
    
    def finish_calibration(self):
        """Complete the calibration process."""
        self.calibration_complete = True
        self.current_screen = "main_menu"
    
    def render_category_select(self):
        """Render the category selection screen."""
        # Title
        title = self.font_large.render("Select Ad Category", True, self.WHITE)
        title_rect = title.get_rect(center=(self.screen_width/2, 100))
        self.screen.blit(title, title_rect)
        
        # Display available categories
        if not self.available_categories:
            no_cats = self.font_medium.render("No categories available", True, self.RED)
            no_cats_rect = no_cats.get_rect(center=(self.screen_width/2, 300))
            self.screen.blit(no_cats, no_cats_rect)
        else:
            y_pos = 200
            for category in self.available_categories:
                self.create_button(category, self.screen_width/2, y_pos, self.WHITE)
                y_pos += 80
        
        # Back button
        back_btn = self.create_button("Back", 100, self.screen_height - 50, self.WHITE, 100, 40)
    
    def select_category(self, category):
        """Select a category and start viewing ads."""
        self.current_category = category
        self.load_ad_images()
        
        if self.ad_images:
            self.current_ad_index = 0
            self.gaze_data = []
            self.current_screen = "viewing_ads"
            self.view_start_time = pygame.time.get_ticks()
        else:
            # Show error and return to category select
            print(f"No images found in category: {category}")
    
    def load_ad_images(self):
        """Load ad images and class data for the selected category."""
        category_path = os.path.join(self.categories_dir, self.current_category)
        train_path = os.path.join(category_path, "train")
        
        if not os.path.exists(train_path):
            self.ad_images = []
            self.ad_classes = []
            return
        
        # Get list of images
        self.ad_images = [f for f in os.listdir(train_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Load class data for each image
        self.ad_classes = []
        for img_file in self.ad_images:
            json_file = os.path.splitext(img_file)[0] + ".json"
            json_path = os.path.join(train_path, json_file)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.ad_classes.append(json.load(f))
            else:
                self.ad_classes.append(None)
        
    def run_ad_viewing(self, frame):
        """Run ad viewing session with eye tracking."""
        if self.current_ad_index >= len(self.ad_images):
            self.finish_ad_viewing()
            return
            
        # Process eye tracking or use demo mode
        if self.eye_tracking_available and self.eye_tracking_initialized:
            try:
                point, key_points, blink, fixation, cevent = self.eye_tracker.process(
                    frame, context=self.context
                )
            except Exception as e:
                print(f"Eye tracking error: {e}")
                # Fall back to demo mode if eye tracking fails
                point = self.get_demo_gaze_point()
        else:
            # Demo mode - simulated gaze
            point = self.get_demo_gaze_point()
            
        # Load and display current ad
        img_path = os.path.join(
            self.categories_dir, 
            self.current_category, 
            "train", 
            self.ad_images[self.current_ad_index]
        )
        
        # Load and display the image
        try:
            img = pygame.image.load(img_path)
            img = pygame.transform.scale(img, (self.screen_width, self.screen_height))
            self.screen.blit(img, (0, 0))
        except pygame.error as e:
            print(f"Could not load image {img_path}: {e}")
            # Skip to next ad
            self.move_to_next_ad()
            return
            
        # Track gaze if we have valid data
        if point is not None:
            # Scale to screen dimensions
            x = int(point[0] * self.screen_width)
            y = int(point[1] * self.screen_height)
            
            # Find which class this point falls into
            class_name = self.get_class_for_point(x, y)
            
            # Record gaze data with timestamp
            timestamp = pygame.time.get_ticks()
            
            if self.current_gaze_sequence:
                last_entry = self.current_gaze_sequence[-1]
                if last_entry["class"] == class_name:
                    # Update duration for the same class
                    last_entry["duration"] = (timestamp - last_entry["start_time"]) / 1000.0  # in seconds
                else:
                    # New class, add new entry
                    self.current_gaze_sequence.append({
                        "ad_index": self.current_ad_index,
                        "ad_name": self.ad_images[self.current_ad_index],
                        "class": class_name,
                        "sequence": len(self.current_gaze_sequence) + 1,
                        "x": x,
                        "y": y,
                        "start_time": timestamp,
                        "duration": 0.0
                    })
            else:
                # First entry for this ad
                self.current_gaze_sequence.append({
                    "ad_index": self.current_ad_index,
                    "ad_name": self.ad_images[self.current_ad_index],
                    "class": class_name,
                    "sequence": 1,
                    "x": x,
                    "y": y,
                    "start_time": timestamp,
                    "duration": 0.0
                })
            
            # Draw gaze point
            pygame.draw.circle(self.screen, self.RED, (x, y), 10)
        
        # Show time remaining
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.view_start_time
        remaining = max(0, self.ad_display_time - elapsed)
        
        time_text = self.font_small.render(
            f"Time: {remaining / 1000:.1f}s", True, self.WHITE, self.BLACK)
        self.screen.blit(time_text, (50, 50))
        
        # Check if time to move to next ad
        if elapsed >= self.ad_display_time:
            self.move_to_next_ad()
    
    def get_class_for_point(self, x, y):
        """Determine which ad class the gaze point falls into."""
        
        if self.current_ad_index >= len(self.ad_classes) or not self.ad_classes[self.current_ad_index]:
            return "unknown"
            
        # Check each class region
        for class_name, regions in self.ad_classes[self.current_ad_index].items():
            for region in regions:
                # Check if point is inside this region
                if isinstance(region, list) and len(region) >= 4:
                    if (region[0] <= x <= region[2] and 
                        region[1] <= y <= region[3]):
                        return class_name
        
        return "background"  # Default if no specific class found
    
    def move_to_next_ad(self):
        """Move to the next ad in the sequence."""
        # Save current gaze data
        self.gaze_data.extend(self.current_gaze_sequence)
        self.current_gaze_sequence = []
        
        # Move to next ad
        self.current_ad_index += 1
        self.view_start_time = pygame.time.get_ticks()
    
    def finish_ad_viewing(self):
        """Complete the ad viewing sequence and save results."""
        # Save results to CSV
        self.save_results_to_csv()
        
        # Return to main menu
        self.current_screen = "main_menu"
    
    def save_results_to_csv(self):
        """Save gaze tracking results to CSV file."""
        if not self.gaze_data:
            return
            
        # Create DataFrame from gaze data
        df = pd.DataFrame(self.gaze_data)
        
        # Ensure output directory exists
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create or append to CSV file
        csv_path = os.path.join(output_dir, f"{self.current_category}_results.csv")
        
        if os.path.exists(csv_path):
            # Append to existing file
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
    
        print(f"Results saved to {csv_path}")

    def get_demo_gaze_point(self):
        """Generate a simulated gaze point for demo mode."""
        # Move the gaze point around naturally
        self.demo_gaze_x += self.move_direction_x * 5
        self.demo_gaze_y += self.move_direction_y * 3
        
        # Bounce at screen edges
        if self.demo_gaze_x <= 0 or self.demo_gaze_x >= self.screen_width:
            self.move_direction_x *= -1
        if self.demo_gaze_y <= 0 or self.demo_gaze_y >= self.screen_height:
            self.move_direction_y *= -1
            
        # Occasionally change direction randomly
        if random.random() < 0.02:
            self.move_direction_x = random.choice([-1, 1])
        if random.random() < 0.02:
            self.move_direction_y = random.choice([-1, 1])
            
        # Return normalized coordinates
        return [self.demo_gaze_x / self.screen_width, self.demo_gaze_y / self.screen_height]

if __name__ == "__main__":
    tracker = AdEyeTracker()
    tracker.run()
